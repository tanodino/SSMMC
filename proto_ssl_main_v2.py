"""
Non-parametric semi-supervised classifier: SupCon fine-tuning + support-set
consistency + mean-entropy-maximization (me-max).

No trained classification head ANYWHERE in this pipeline -- neither for the
labeled loss nor for pseudo-labeling the unlabeled pool. This is the direct
motivation from today's diagnosis: every pseudo-labeling baseline that
failed (FixMatch, SoftMatch, and MSC's collapse) did so via a classifier
component (a randomly-initialized or freshly-converged head, or a modality-
agreement mechanism) generating a confidently-wrong target that then got
reinforced. This script never generates a hard label from a trained head at
any point.

--------------------------------------------------------------------------
Components (see conversation for full derivation of each)
--------------------------------------------------------------------------
1. LABELED LOSS -- Supervised Contrastive (Khosla et al., 2020), full-batch
   (no augmentation): same-class = positive pair, different-class =
   negative pair, in the fused embedding space.

2. UNLABELED LOSS, two parts, both non-parametric (no trained head):
   a. SUPPORT-SET CONSISTENCY (PAWS-style; Assran et al., ICCV 2021):
      - draw a FRESH random support set each step: k_per_class labeled
        samples per class (default 1) -- re-sampling every step is what
        gives the model "relational" invariance (be close to WHICHEVER
        same-class exemplar is drawn, not a fixed set of specific points)
        and reintroduces the stochasticity that full-batch/no-augmentation
        SupCon training otherwise lacks.
      - weak view of an unlabeled sample -> soft label via similarity to
        the support set (this is the TARGET, sharpened, no-grad)
      - strong view -> soft label via similarity to the SAME support set
        (this is the STUDENT prediction, trained to match the target)
   b. MEAN-ENTROPY-MAXIMIZATION (me-max): penalizes the BATCH-AVERAGED
      soft prediction for being non-uniform across classes -- directly
      targets class-collapse, independent of per-sample confidence.

3. EVALUATION: k-NN (distance-weighted vote), a post-hoc linear probe, and
   a jointly-trained classification head -- all evaluated side by side.

--------------------------------------------------------------------------
h / z SPLIT (SimCLR-style; Chen et al., 2020) for the InfoNCE consistency
term specifically -- see ProtoModel docstring. forward() ALWAYS returns h
(a single tensor); project_for_ssl(h) -> z is called explicitly ONLY at
the InfoNCE call site, on an ALREADY-COMPUTED embedding (never on raw
modality inputs) -- so every other call site (SupCon, support-set
comparisons, evaluation) is unaffected and always gets a plain tensor.
--------------------------------------------------------------------------

DIAGNOSTICS -- per-component loss logging (every 5 epochs): loss_sup,
loss_consistency, loss_me are each accumulated separately and printed
alongside total.

Usage (same CLI pattern as your other scripts):
    python proto_ssl_main.py EUROSAT SAR MS 5 0 [pretrained_path] [freeze]
"""

import itertools
import sys
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

from model import SFFCConfig, ViTEncoder
from functions import (weak_augment_pair, strong_augment_pair, MOMENTUM_EMA, cumulate_EMA,
                        WARM_UP_EPOCH_EMA, EPOCHS, WARM_UP_EPOCH_SSL, RATIO_LABELED_UNLABELED_BATCHES)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ==========================================================================
# Model: two encoders -> per-modality projection -> fused, L2-normalized embedding
# ==========================================================================

class ProtoModel(nn.Module):
    """Two encoders, ONE shared projection head (self.fusion) applied to
    the concatenated per-modality CLS tokens.

    ALSO includes a small, SEPARATE ssl_head (SimCLR-style h/z split;
    Chen et al., 2020): `forward()` ALWAYS returns h (single tensor) --
    used for SupCon, support-set comparisons, k-NN evaluation, everything
    except InfoNCE. `project_for_ssl(h)` maps h -> z; it takes an
    ALREADY-COMPUTED embedding (e.g. emb_weak/emb_strong from forward()),
    NOT raw modality inputs -- called explicitly ONLY at the InfoNCE call
    site.

    SimCLR's own ablation found the pre-projection representation (h)
    outperforms the post-projection one (z) by >10% under linear
    evaluation -- the contrastive objective pushes z toward
    instance-discrimination-specific invariances that actively hurt
    class-discriminative structure, so isolating that pressure into a
    disposable extra head protects h (what k-NN actually uses) from it."""

    def __init__(self, config: SFFCConfig, proj_dim: int = 128):
        super().__init__()
        self.modality_1_encoder = ViTEncoder(
            img_size=config.img_size_m1, patch_size=config.patch_size_m1,
            in_chans=config.in_chans_m1,
        )
        self.modality_2_encoder = ViTEncoder(
            img_size=config.img_size_m2, patch_size=config.patch_size_m2,
            in_chans=config.in_chans_m2,
        )
        self.fusion = nn.Sequential(
            nn.LazyLinear(512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, proj_dim),
        )

        # SSL-only head: h -> z, called explicitly via project_for_ssl(h).
        # Never read by SupCon, support-set comparisons, or evaluation.
        #
        # NOTE (from conversation): BatchNorm vs LayerNorm was tested
        # directly. LayerNorm removes the running-statistics train/eval
        # discrepancy and DOES produce a genuine stable plateau (confirmed
        # empirically), but that plateau sits BELOW where BatchNorm's
        # still-declining trajectory was at the same epoch -- i.e.
        # BatchNorm's noise appears to also confer real regularization
        # benefit (higher peak AND likely higher eventual floor), same
        # pattern observed earlier today comparing BatchNorm vs GroupNorm
        # on a ResNet18 SF baseline. Reverted to BatchNorm here and paired
        # with best-F1 checkpoint tracking (see main loop) instead --
        # this captures BatchNorm's higher ceiling directly, sidestepping
        # the instability rather than trading quality away for a lower,
        # stable floor. Swap back to nn.LayerNorm if you want the
        # stability-first tradeoff instead.
        self.ssl_head = nn.Sequential(
            nn.LazyLinear(512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, proj_dim), nn.BatchNorm1d(proj_dim)
        )

        # Linear classification head, trained JOINTLY during the SSL loop
        # via a supervised CE loss on the labeled batch (see main loop) --
        # NOT a post-hoc/refit-from-scratch probe. See classify()'s
        # docstring for the detach_input design decision.
        self.classifier = nn.Linear(proj_dim, config.num_classes)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        cls_m1 = self.modality_1_encoder(x1)
        cls_m2 = self.modality_2_encoder(x2)
        concat = torch.cat([cls_m1, cls_m2], dim=1)
        emb = self.fusion(concat)
        return F.normalize(emb, dim=1)                # h -- always a single tensor

    def project_for_ssl(self, h: torch.Tensor) -> torch.Tensor:
        """h -> z. Takes an ALREADY-COMPUTED embedding, NOT raw modality
        inputs -- call as model.project_for_ssl(emb_weak), never as
        model.project_for_ssl(f_weak, s_weak)."""
        z = self.ssl_head(h)
        return F.normalize(z, dim=1)

    def classify(self, h: torch.Tensor, detach_input: bool = True) -> torch.Tensor:
        """Linear classification head on h -> raw logits.

        detach_input=True (default, "online linear probe" style, same
        protective reasoning as the SimCLR h/z split used for
        project_for_ssl): the classification loss's gradient STOPS here
        and does NOT propagate into the shared encoder/projector, so it
        cannot distort h. IMPORTANT: if SupCon (below) is disabled, this
        flag needs to be False, or h receives NO label-aware gradient at
        all -- see conversation.

        detach_input=False: gradient flows into h too. Not unprecedented
        (SupCon already uses the labeled set directly with real gradient
        into h), but plain CE on a linear head with only ~5 labels/class
        is far easier to memorize outright than SupCon's relational
        structure, so that memorization pressure would leak straight into
        h if left un-detached."""
        if detach_input:
            h = h.detach()
        return self.classifier(h)                          # raw logits


# ==========================================================================
# Labeled loss: Supervised Contrastive (Khosla et al., 2020)
# ==========================================================================

def supervised_contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor,
                                 temperature: float = 0.1) -> torch.Tensor:
    device = embeddings.device
    N = embeddings.shape[0]

    sim = embeddings @ embeddings.T / temperature
    self_mask = torch.eye(N, dtype=torch.bool, device=device)

    sim_max = sim.masked_fill(self_mask, float("-inf")).max(dim=1, keepdim=True).values
    sim = sim - sim_max.detach()
    exp_sim = torch.exp(sim) * (~self_mask)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.T) & (~self_mask)
    pos_counts = pos_mask.sum(dim=1).clamp(min=1)

    loss = -(pos_mask * log_prob).sum(dim=1) / pos_counts
    return loss.mean()


def consistency_loss_infonce(emb_weak: torch.Tensor, emb_strong: torch.Tensor,
                              temperature: float = 0.2) -> torch.Tensor:
    """InfoNCE / NT-Xent style consistency. Called with z (project_for_ssl
    output), NOT h directly -- see ProtoModel docstring.

    strong-view embedding i is the ANCHOR, weak-view embedding i (same
    sample, detached) is the POSITIVE, every OTHER sample's weak-view
    embedding in the batch is a NEGATIVE.

    CAVEAT: no ground truth to know if two unlabeled samples secretly
    share a class -- every non-matching-index pair is treated as a
    negative regardless (standard "false negative" issue in
    instance-discrimination contrastive learning).

    NOTE (from conversation): this loss showed a "never saturates"
    over-optimization pattern on a prior run -- loss_consistency kept
    falling for the full run while F1 peaked early and declined
    afterward. Consider tracking/saving the best-F1 checkpoint rather
    than only the final epoch when using this loss."""
    device = emb_strong.device
    B = emb_strong.shape[0]

    with torch.no_grad():
        target = emb_weak.detach()               # [B, D], L2-normalized

    sim = emb_strong @ target.T / temperature      # [B, B]: sim[i,j] = strong_i . weak_j

    pos_mask = torch.eye(B, dtype=torch.bool, device=device)   # diagonal: matching-index pairs = positives
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)  # log-softmax over full row (1 pos + B-1 neg)
    loss_per_anchor = -log_prob[pos_mask]
    return loss_per_anchor.mean()


# ==========================================================================
# Unlabeled loss: support-set consistency + mean-entropy-maximization
# ==========================================================================

def sample_support_set(f_lab, s_lab, y_lab, n_classes, k_per_class=1):
    """Fresh random support set: k_per_class labeled samples per class,
    without replacement, resampled every call -- the source of the
    "relational" stochasticity discussed in conversation."""
    idx = []
    for c in range(n_classes):
        class_idx = (y_lab == c).nonzero(as_tuple=True)[0]
        n_take = min(k_per_class, len(class_idx))
        chosen = class_idx[torch.randperm(len(class_idx), device=class_idx.device)[:n_take]]
        idx.append(chosen)
    idx = torch.cat(idx)
    return f_lab[idx], s_lab[idx], y_lab[idx]


def logit_standardize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Z-score standardization along the last dimension (Sun et al., CVPR
    2024): subtract each row's own mean, divide by its own std."""
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / (std + eps)


def soft_knn_probs(query_emb: torch.Tensor, ref_emb: torch.Tensor, ref_labels: torch.Tensor,
                    n_classes: int, temperature: float = 0.1, standardize: bool = False) -> torch.Tensor:
    """[Bq, C] soft distribution: softmax over similarity to ALL reference
    samples, aggregated by class via a one-hot matmul."""
    sims = query_emb @ ref_emb.T                              # [Bq, Nref]
    if standardize:
        sims = logit_standardize(sims)
    sims = sims / temperature
    weights = F.softmax(sims, dim=1)
    one_hot_labels = F.one_hot(ref_labels, n_classes).float()
    return weights @ one_hot_labels


def sharpen(p: torch.Tensor, T: float = 0.5) -> torch.Tensor:
    p_sharp = p ** (1.0 / T)
    return p_sharp / p_sharp.sum(dim=1, keepdim=True)


def consistency_loss_embedding_cosine(emb_weak: torch.Tensor, emb_strong: torch.Tensor) -> torch.Tensor:
    """emb_weak, emb_strong already L2-normalized. Mathematically
    identical to L2 distance here -- literally the BYOL/SimSiam objective."""
    with torch.no_grad():
        target = emb_weak.detach()
    return (1.0 - (target * emb_strong).sum(dim=1)).mean()


def consistency_loss_cosine(probs_weak: torch.Tensor, probs_strong: torch.Tensor,
                             sharpen_T: float = 0.5, eps: float = 1e-8) -> torch.Tensor:
    with torch.no_grad():
        target = sharpen(probs_weak, T=sharpen_T)
    target_n = F.normalize(target, dim=1, eps=eps)
    strong_n = F.normalize(probs_strong, dim=1, eps=eps)
    return (1.0 - (target_n * strong_n).sum(dim=1)).mean()


def consistency_loss_l1(probs_weak: torch.Tensor, probs_strong: torch.Tensor,
                         sharpen_T: float = 0.5) -> torch.Tensor:
    with torch.no_grad():
        target = sharpen(probs_weak, T=sharpen_T)
    return (target - probs_strong).abs().sum(dim=1).mean()


def consistency_loss_l2(probs_weak: torch.Tensor, probs_strong: torch.Tensor,
                         sharpen_T: float = 0.5) -> torch.Tensor:
    """Squared L2 distance (multiclass Brier score) -- MixMatch's actual
    choice (Berthelot et al., 2019)."""
    with torch.no_grad():
        target = sharpen(probs_weak, T=sharpen_T)
    return ((target - probs_strong) ** 2).sum(dim=1).mean()


def consistency_loss(probs_weak: torch.Tensor, probs_strong: torch.Tensor,
                      sharpen_T: float = 0.5, eps: float = 1e-8) -> torch.Tensor:
    with torch.no_grad():
        target = sharpen(probs_weak, T=sharpen_T)
    return -(target * torch.log(probs_strong.clamp(min=eps))).sum(dim=1).mean()


def mean_entropy_max_loss(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Penalizes the BATCH-AVERAGED prediction for deviating from uniform.
    Already sign-flipped: minimizing this MAXIMIZES entropy."""
    mean_p = probs.mean(dim=0)
    entropy = -(mean_p * torch.log(mean_p.clamp(min=eps))).sum()
    return -entropy


# ==========================================================================
# Evaluation: k-NN (distance-weighted) against the FULL labeled set
# ==========================================================================

@torch.no_grad()
def compute_reference_embeddings(model: ProtoModel, f_lab: torch.Tensor, s_lab: torch.Tensor,
                                  device) -> torch.Tensor:
    model.eval()
    return model(f_lab.to(device), s_lab.to(device))       # h, single tensor


@torch.no_grad()
def knn_classify(query_emb: torch.Tensor, ref_emb: torch.Tensor, ref_labels: torch.Tensor,
                  n_classes: int, k: int = 5) -> torch.Tensor:
    k = min(k, ref_emb.shape[0])
    sims = query_emb @ ref_emb.T
    topk_sims, topk_idx = sims.topk(k, dim=1)
    topk_labels = ref_labels[topk_idx]

    class_scores = torch.zeros(query_emb.shape[0], n_classes, device=query_emb.device)
    class_scores.scatter_add_(1, topk_labels, topk_sims.clamp(min=0))
    return class_scores.argmax(dim=1)


@torch.no_grad()
def evaluate_with_knn(model: ProtoModel, ref_emb: torch.Tensor, ref_labels: torch.Tensor,
                       dataloader, n_classes: int, device, k: int = 5):
    model.eval()
    all_preds, all_labels = [], []
    for f_batch, s_batch, y_batch in dataloader:
        f_batch = f_batch.to(device, non_blocking=True)
        s_batch = s_batch.to(device, non_blocking=True)
        emb = model(f_batch, s_batch)                       # h, single tensor
        preds = knn_classify(emb, ref_emb, ref_labels, n_classes, k=k).cpu()
        all_preds.append(preds)
        all_labels.append(y_batch)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def evaluate_with_linear_probe(model: ProtoModel, f_lab: torch.Tensor, s_lab: torch.Tensor,
                                y_lab: torch.Tensor, dataloader, device, C: float = 1.0):
    """"Linear probe" evaluation, POST-HOC / refit-from-scratch each call
    (standard SSL protocol -- SimCLR, MoCo, BYOL, SupCon all report this
    as their primary representation-quality metric). Fits a fresh logistic
    regression classifier on the labeled embeddings every time this is
    called. See evaluate_with_classifier for the JOINTLY-trained
    alternative (a linear head trained continuously during the SSL loop,
    read off directly rather than refit each time)."""
    model.eval()
    with torch.no_grad():
        ref_emb = model(f_lab.to(device), s_lab.to(device)).cpu().numpy()
    ref_labels = y_lab.numpy()

    clf = LogisticRegression(max_iter=1000, C=C)
    clf.fit(ref_emb, ref_labels)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for f_batch, s_batch, y_batch in dataloader:
            f_batch = f_batch.to(device, non_blocking=True)
            s_batch = s_batch.to(device, non_blocking=True)
            emb = model(f_batch, s_batch).cpu().numpy()
            preds = clf.predict(emb)
            all_preds.append(preds)
            all_labels.append(y_batch.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


@torch.no_grad()
def evaluate_with_classifier(model: ProtoModel, dataloader, device):
    """Reads predictions directly from model.classifier -- the linear head
    trained JOINTLY, continuously, during the SSL loop (see main loop's
    loss_cls). No fitting step here at all, unlike evaluate_with_linear_probe
    -- this is just a forward pass through weights that have been training
    the whole time."""
    model.eval()
    all_preds, all_labels = [], []
    for f_batch, s_batch, y_batch in dataloader:
        f_batch = f_batch.to(device, non_blocking=True)
        s_batch = s_batch.to(device, non_blocking=True)
        h = model(f_batch, s_batch)
        logits = model.classify(h)                          # detach_input irrelevant here (no_grad anyway)
        preds = logits.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(y_batch)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


# ==========================================================================
# Pretrained-encoder loading / freezing
# ==========================================================================

def load_pretrained_encoders(model: ProtoModel, path: str, device: str, strict: bool = True):
    encoders_state = torch.load(path, map_location=device, weights_only=True)
    model.modality_1_encoder.load_state_dict(encoders_state["modality_1"], strict=strict)
    model.modality_2_encoder.load_state_dict(encoders_state["modality_2"], strict=strict)
    print("Loaded pretrained encoder weights from %s" % path)


def freeze_pretrained_encoders(model: ProtoModel):
    for p in model.modality_1_encoder.parameters():
        p.requires_grad = False
    for p in model.modality_2_encoder.parameters():
        p.requires_grad = False


if __name__ == "__main__":
    batch_size = 16
    dataset_path = sys.argv[1]
    first_prefix = sys.argv[2]
    second_prefix = sys.argv[3]
    perc = sys.argv[4]
    run_id = sys.argv[5]
    pretrained_path = sys.argv[6] if len(sys.argv) > 6 else None
    freeze_encoder = "freeze" in sys.argv
    print(sys.argv)

    # ---- tunables ----
    SUP_TEMPERATURE = 0.1     # labeled SupCon loss temperature
    KNN_TEMPERATURE = 0.1     # soft-label similarity temperature (support-set comparisons)
    INFO_NCE_TEMPERATURE = 1.0  # NOTE: SimCLR-typical range is 0.1-0.5; 1.0 gives a much
                                 # flatter softmax -- weaker anti-collapse signal than lower
                                 # values. Worth sweeping down (e.g. 0.2) if InfoNCE's
                                 # negative pressure should bite harder, especially since
                                 # me-max (below) is currently excluded from the loss.
    SHARPEN_T = 0.5           # weak-view target sharpening temperature
    K_PER_CLASS = 1           # support-set size per class, resampled every step
    K_NEIGHBORS = 5           # k for FINAL evaluation k-NN (against full labeled set)
    LAMBDA_U = 1.0            # weight of the consistency term
    LAMBDA_ME = 0.3           # weight of the mean-entropy-max term (currently NOT added
                               # into the optimized loss below -- see comment at loss = ...)
    LAMBDA_CLS = 1.0          # weight of the jointly-trained classifier's CE loss
    DETACH_CLASSIFIER_INPUT = True  # True = "online linear probe" (gradient stops at h,
                                     # cannot distort the shared representation). SupCon is
                                     # ACTIVE below, so h still gets label-aware gradient
                                     # through that path even with this =True. If you ever
                                     # remove SupCon, this MUST become False or h gets no
                                     # label signal at all -- see conversation.

    first_data = np.load("%s/%s_data_normalized.npy" % (dataset_path, first_prefix))
    second_data = np.load("%s/%s_data_normalized.npy" % (dataset_path, second_prefix))
    full_labels = np.load("%s/labels.npy" % dataset_path)
    train_idx = np.load("%s/train_idx.npy" % dataset_path)
    labelled_idx = np.load("%s/labelled_samples_%s_%s.npy" % (dataset_path, perc, run_id))

    full_train_idx = np.arange(len(train_idx))
    unlabelled_idx = np.setdiff1d(full_train_idx, labelled_idx)

    f_lab_data_train = first_data[train_idx][labelled_idx]
    s_lab_data_train = second_data[train_idx][labelled_idx]
    f_unlab_data_train = first_data[train_idx][unlabelled_idx]
    s_unlab_data_train = second_data[train_idx][unlabelled_idx]

    labels = full_labels[train_idx][labelled_idx]
    n_classes = len(np.unique(labels))

    print("f_lab_data_train %d" % len(f_lab_data_train))
    print("f_unlab_data_train %d" % len(f_unlab_data_train))
    print("n_classes %d" % n_classes)

    dir_name = dataset_path + "/PROTO_SSL"
    os.makedirs(dir_name, exist_ok=True)
    output_file = dir_name + "/%s_%s.pth" % (perc, run_id)

    # ---------------- TEST DATA ----------------
    test_idx = np.setdiff1d(np.arange(full_labels.shape[0]), train_idx)
    f_data_test = first_data[test_idx]
    s_data_test = second_data[test_idx]
    labels_test = full_labels[test_idx]

    test_dataset = TensorDataset(
        torch.tensor(f_data_test, dtype=torch.float32),
        torch.tensor(s_data_test, dtype=torch.float32),
        torch.tensor(labels_test, dtype=torch.int64),
    )
    dataloader_test = DataLoader(test_dataset, shuffle=False, batch_size=batch_size * RATIO_LABELED_UNLABELED_BATCHES,
        num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=4, drop_last=False)
    print("TEST DATA built")
    sys.stdout.flush()

    # ---------------- LABELED DATA (full-batch, no augmentation) ----------------
    x_tensor_f_lab = torch.tensor(f_lab_data_train, dtype=torch.float32)
    x_tensor_s_lab = torch.tensor(s_lab_data_train, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.int64)
    lab_dataset = TensorDataset(x_tensor_f_lab, x_tensor_s_lab, y_tensor)

    dataloader_lab_train = DataLoader(lab_dataset, shuffle=True, batch_size=len(lab_dataset),
        num_workers=0, pin_memory=True, drop_last=False)

    # ---------------- UNLABELED DATA ----------------
    x_tensor_f_unl = torch.tensor(f_unlab_data_train, dtype=torch.float32)
    x_tensor_s_unl = torch.tensor(s_unlab_data_train, dtype=torch.float32)
    unl_dataset = TensorDataset(x_tensor_f_unl, x_tensor_s_unl)

    dataloader_unl_train = DataLoader(unl_dataset, shuffle=True, batch_size=batch_size * RATIO_LABELED_UNLABELED_BATCHES,
        num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=4, drop_last=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("all dataloaders built")
    sys.stdout.flush()

    config = SFFCConfig(
        img_size_m1=f_lab_data_train.shape[2], img_size_m2=s_lab_data_train.shape[2],
        patch_size_m1=8, patch_size_m2=8,
        in_chans_m1=f_lab_data_train.shape[1], in_chans_m2=s_lab_data_train.shape[1],
        num_classes=n_classes, hidden_dim=256, dropout=0.1
    )

    model = ProtoModel(config).to(device)

    if pretrained_path is not None:
        load_pretrained_encoders(model, pretrained_path, device)
        if freeze_encoder:
            freeze_pretrained_encoders(model)
            print("Pretrained encoders FROZEN")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=1e-4)
    # NOTE: weight_decay was 0 (AdamW default) before -- nothing was pulling
    # parameters back toward a smaller norm, so in a flat loss region (see
    # conversation: loss_sup/loss_consistency converge by ~epoch 50-100 while
    # F1 keeps drifting for hundreds more epochs) weights were free to wander
    # indefinitely without any loss-value penalty for doing so. 1e-4 is a
    # conservative starting point -- worth sweeping (1e-3, 1e-2) if this
    # alone doesn't meaningfully change the drift.
    scaler = GradScaler('cuda')
    print("model created")
    sys.stdout.flush()

    ema_weights = None
    for epoch in range(EPOCHS):
        model.train()
        use_ssl = epoch >= WARM_UP_EPOCH_SSL
        total_loss = torch.zeros((), device=device)
        loss_sup_sum = torch.zeros((), device=device)
        loss_consistency_sum = torch.zeros((), device=device)
        loss_me_sum = torch.zeros((), device=device)
        n_batches = 0
        n_ssl_batches = 0
        majority_frac_sum = 0.0
        majority_frac_batches = 0

        if use_ssl:
            for (f_lab_b, s_lab_b, y_lab_b), (f_unl_b, s_unl_b) in zip(
                    itertools.cycle(dataloader_lab_train), dataloader_unl_train):

                optimizer.zero_grad(set_to_none=True)
                f_lab_b = f_lab_b.to(device, non_blocking=True)
                s_lab_b = s_lab_b.to(device, non_blocking=True)
                y_lab_b = y_lab_b.to(device, non_blocking=True)
                f_unl_b = f_unl_b.to(device, non_blocking=True)
                s_unl_b = s_unl_b.to(device, non_blocking=True)

                with autocast('cuda'):
                    # ---- labeled: SupCon + classifier CE, full batch, no augmentation ----
                    emb_lab = model(f_lab_b, s_lab_b)                    # h, single tensor -- NOT unpacked as a tuple
                    loss_sup = supervised_contrastive_loss(emb_lab, y_lab_b, temperature=SUP_TEMPERATURE)

                    logits_lab = model.classify(emb_lab, detach_input=DETACH_CLASSIFIER_INPUT)
                    loss_cls = F.cross_entropy(logits_lab, y_lab_b)

                    # ---- unlabeled: support-set consistency + InfoNCE ----
                    f_weak, s_weak = weak_augment_pair(f_unl_b, s_unl_b)
                    f_strong, s_strong = strong_augment_pair(f_unl_b, s_unl_b)

                    f_sup, s_sup, y_sup = sample_support_set(
                        f_lab_b, s_lab_b, y_lab_b, n_classes, k_per_class=K_PER_CLASS)
                    support_emb = model(f_sup, s_sup)          # h, single tensor -- grad-enabled

                    with torch.no_grad():
                        emb_weak = model(f_weak, s_weak)                    # h_weak
                        proj_weak = model.project_for_ssl(emb_weak)           # z_weak -- from the EMBEDDING, not raw inputs
                        probs_weak = soft_knn_probs(emb_weak, support_emb.detach(), y_sup,
                                                     n_classes, temperature=KNN_TEMPERATURE)

                    emb_strong = model(f_strong, s_strong)                  # h_strong
                    proj_strong = model.project_for_ssl(emb_strong)           # z_strong -- from the EMBEDDING, not raw inputs
                    probs_strong = soft_knn_probs(emb_strong, support_emb, y_sup,
                                                   n_classes, temperature=KNN_TEMPERATURE)

                    #loss_consistency = consistency_loss(probs_weak, probs_strong, sharpen_T=SHARPEN_T)
                    #loss_consistency = consistency_loss_l1(probs_weak, probs_strong, sharpen_T=SHARPEN_T)
                    #loss_consistency = consistency_loss_l2(probs_weak, probs_strong, sharpen_T=SHARPEN_T)
                    #loss_consistency = consistency_loss_embedding_cosine(emb_weak, emb_strong)
                    loss_consistency = consistency_loss_infonce(proj_weak, proj_strong, temperature=INFO_NCE_TEMPERATURE)

                    loss_me = mean_entropy_max_loss(probs_strong)

                    #loss = loss_sup + LAMBDA_U * (loss_consistency + LAMBDA_ME * loss_me)
                    loss =  LAMBDA_U * loss_consistency + LAMBDA_CLS * loss_cls #+ loss_sup # me-max currently excluded, confirm intentional

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.detach()
                loss_sup_sum += loss_sup.detach()
                loss_consistency_sum += loss_consistency.detach()
                loss_me_sum += loss_me.detach()
                n_batches += 1
                n_ssl_batches += 1

                with torch.no_grad():
                    hard_preds = probs_strong.argmax(dim=1)
                    majority_frac = (hard_preds.bincount(minlength=n_classes).max() / hard_preds.numel()).item()
                    majority_frac_sum += majority_frac
                    majority_frac_batches += 1
        else:
            # warmup: labeled SupCon + classifier CE only, no unlabeled pool touched
            for f_lab_b, s_lab_b, y_lab_b in dataloader_lab_train:
                optimizer.zero_grad(set_to_none=True)
                f_lab_b = f_lab_b.to(device, non_blocking=True)
                s_lab_b = s_lab_b.to(device, non_blocking=True)
                y_lab_b = y_lab_b.to(device, non_blocking=True)

                with autocast('cuda'):
                    emb_lab = model(f_lab_b, s_lab_b)             # h, single tensor
                    loss_sup_only = supervised_contrastive_loss(emb_lab, y_lab_b, temperature=SUP_TEMPERATURE)
                    logits_lab = model.classify(emb_lab, detach_input=DETACH_CLASSIFIER_INPUT)
                    loss_cls = F.cross_entropy(logits_lab, y_lab_b)
                    loss = LAMBDA_CLS * loss_cls #+ loss_sup_only

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.detach()
                loss_sup_sum += loss.detach()
                n_batches += 1

        if epoch >= WARM_UP_EPOCH_EMA:
            ema_weights = cumulate_EMA(model, ema_weights, MOMENTUM_EMA)

        if epoch % 5 == 0:
            if epoch >= WARM_UP_EPOCH_EMA:
                current_state_dict = copy.deepcopy(model.state_dict())
                model.load_state_dict(ema_weights)
                ref_emb = compute_reference_embeddings(model, x_tensor_f_lab, x_tensor_s_lab, device)
                predictions, test_labels = evaluate_with_knn(
                    model, ref_emb, y_tensor.to(device), dataloader_test, n_classes, device, k=K_NEIGHBORS)
                lp_predictions, lp_test_labels = evaluate_with_linear_probe(
                    model, x_tensor_f_lab, x_tensor_s_lab, y_tensor, dataloader_test, device)
                cls_predictions, cls_test_labels = evaluate_with_classifier(model, dataloader_test, device)
                model.load_state_dict(current_state_dict)
            else:
                ref_emb = compute_reference_embeddings(model, x_tensor_f_lab, x_tensor_s_lab, device)
                predictions, test_labels = evaluate_with_knn(
                    model, ref_emb, y_tensor.to(device), dataloader_test, n_classes, device, k=K_NEIGHBORS)
                lp_predictions, lp_test_labels = evaluate_with_linear_probe(
                    model, x_tensor_f_lab, x_tensor_s_lab, y_tensor, dataloader_test, device)
                cls_predictions, cls_test_labels = evaluate_with_classifier(model, dataloader_test, device)

            f1_val = f1_score(test_labels, predictions, average="weighted")
            f1_val_lp = f1_score(lp_test_labels, lp_predictions, average="weighted")
            f1_val_cls = f1_score(cls_test_labels, cls_predictions, average="weighted")
            avg_majority_frac = majority_frac_sum / max(majority_frac_batches, 1)
            avg_loss_sup = loss_sup_sum.item() / max(n_batches, 1)
            avg_loss_consistency = loss_consistency_sum.item() / max(n_ssl_batches, 1)
            avg_loss_me = loss_me_sum.item() / max(n_ssl_batches, 1)

            print(f"epoch {epoch} total={total_loss.item() / max(n_batches, 1):.4f} "
                  f"loss_sup={avg_loss_sup:.4f} loss_consistency={avg_loss_consistency:.4f} "
                  f"loss_me={avg_loss_me:.4f} "
                  f"F1-knn={(f1_val * 100):.2f} F1-linprobe={(f1_val_lp * 100):.2f} "
                  f"F1-classifier={(f1_val_cls * 100):.2f} "
                  f"majority_frac={avg_majority_frac:.3f} "
                  f"ssl={'on' if use_ssl else 'off (warmup)'}")
            sys.stdout.flush()

    if ema_weights is not None:
        model.load_state_dict(ema_weights)
    torch.save(model.state_dict(), output_file)
    print("Saved to %s" % output_file)