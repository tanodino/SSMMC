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
      targets the class-collapse failure mode measured all day as
      "majority_frac" climbing toward 0.6+ in FixMatch/SoftMatch. This is
      the piece FixMatch/SoftMatch structurally lack and FreeMatch/KDMvC
      structurally have (in different forms), which today's ablations
      identified as the actual determinant of SSL stability here.

3. EVALUATION: k-NN (distance-weighted vote) against the FULL labeled set
   (not a support subset) -- uses all available labeled signal at eval
   time, unlike training where support sets are deliberately small and
   resampled for the stochasticity benefit.

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
    """Two encoders, ONE shared projection head applied to the concatenated
    per-modality CLS tokens -- fusion happens at concatenation, before the
    (only) trainable projector, rather than each modality being projected
    independently and only combined afterward. Simpler (one head instead
    of two) and lets the projector learn cross-modal interactions."""

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
        self.projector = nn.Sequential(
            nn.LazyLinear(512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, proj_dim),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        cls_m1 = self.modality_1_encoder(x1)
        cls_m2 = self.modality_2_encoder(x2)
        concat = torch.cat([cls_m1, cls_m2], dim=1)
        emb = self.projector(concat)
        return F.normalize(emb, dim=1)


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
    2024): subtract each row's own mean, divide by its own std. Strips out
    absolute scale, keeping only the RELATIVE pattern across that row's
    entries -- e.g. two views of the same sample can have very different
    absolute similarity magnitudes to the reference set (strong
    augmentation plausibly shrinks/inflates it uniformly) while still
    agreeing on which class is most similar; standardizing removes the
    magnitude difference and compares them on equal footing."""
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / (std + eps)


def soft_knn_probs(query_emb: torch.Tensor, ref_emb: torch.Tensor, ref_labels: torch.Tensor,
                    n_classes: int, temperature: float = 0.1, standardize: bool = False) -> torch.Tensor:
    """[Bq, C] soft distribution: softmax over similarity to ALL reference
    samples, aggregated by class via a one-hot matmul.

    standardize=True applies Z-score standardization to each query's
    similarity vector BEFORE the temperature-scaled softmax (see
    logit_standardize docstring) -- makes the resulting soft label depend
    only on the RELATIVE pattern of similarities to the reference set,
    not their absolute scale. Default False for a clean A/B against the
    original behavior."""
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
    """emb_weak, emb_strong already L2-normalized (ProtoModel guarantees
    this). Mathematically identical to L2 distance here -- this is
    literally the BYOL/SimSiam objective."""
    with torch.no_grad():
        target = emb_weak.detach()
    return (1.0 - (target * emb_strong).sum(dim=1)).mean()


def consistency_loss_cosine(probs_weak: torch.Tensor, probs_strong: torch.Tensor,
                             sharpen_T: float = 0.5, eps: float = 1e-8) -> torch.Tensor:
    """1 - cosine_similarity between sharpened target and prediction.
    NOTE: not equivalent to L2 here, since probs vectors aren't unit-norm --
    this ignores confidence/peakiness differences, keeping only relative
    class-preference pattern."""
    with torch.no_grad():
        target = sharpen(probs_weak, T=sharpen_T)
    target_n = F.normalize(target, dim=1, eps=eps)
    strong_n = F.normalize(probs_strong, dim=1, eps=eps)
    return (1.0 - (target_n * strong_n).sum(dim=1)).mean()

def consistency_loss_l1(probs_weak: torch.Tensor, probs_strong: torch.Tensor,
                         sharpen_T: float = 0.5) -> torch.Tensor:
    """L1 (Manhattan) distance between sharpened weak-view target and
    strong-view prediction. Bounded in [0, 2] for probability vectors --
    unlike cross-entropy, a badly-wrong prediction can't produce an
    unbounded gradient spike."""
    with torch.no_grad():
        target = sharpen(probs_weak, T=sharpen_T)
    return (target - probs_strong).abs().sum(dim=1).mean()


def consistency_loss_l2(probs_weak: torch.Tensor, probs_strong: torch.Tensor,
                         sharpen_T: float = 0.5) -> torch.Tensor:
    """Squared L2 distance (multiclass Brier score) -- MixMatch's actual
    choice (Berthelot et al., 2019), for the same bounded/robust reasoning."""
    with torch.no_grad():
        target = sharpen(probs_weak, T=sharpen_T)
    return ((target - probs_strong) ** 2).sum(dim=1).mean()

def consistency_loss(probs_weak: torch.Tensor, probs_strong: torch.Tensor,
                      sharpen_T: float = 0.5, eps: float = 1e-8) -> torch.Tensor:
    """Weak view (sharpened, no-grad) is the target; strong view is
    trained to match it -- same weak-generates-target asymmetry as the
    FixMatch/FreeMatch scripts, applied to soft, non-parametric labels."""
    with torch.no_grad():
        target = sharpen(probs_weak, T=sharpen_T)
    return -(target * torch.log(probs_strong.clamp(min=eps))).sum(dim=1).mean()


def mean_entropy_max_loss(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Penalizes the BATCH-AVERAGED prediction for deviating from uniform
    -- directly targets class-collapse, independent of per-sample
    confidence. Already sign-flipped: minimizing this MAXIMIZES entropy."""
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
    return model(f_lab.to(device), s_lab.to(device))


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
        emb = model(f_batch, s_batch)
        preds = knn_classify(emb, ref_emb, ref_labels, n_classes, k=k).cpu()
        all_preds.append(preds)
        all_labels.append(y_batch)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


# ==========================================================================
# Pretrained-encoder loading / freezing
# ==========================================================================

def load_pretrained_encoders(model: ProtoModel, path: str, device: str, strict: bool = True):
    encoders_state = torch.load(path, map_location=device)
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
    SHARPEN_T = 0.5           # weak-view target sharpening temperature
    K_PER_CLASS = 1           # support-set size per class, resampled every step
    K_NEIGHBORS = 5           # k for FINAL evaluation k-NN (against full labeled set)
    LAMBDA_U = 1.0            # weight of the whole unsupervised block
    LAMBDA_ME = 1.0           # weight of mean-entropy-max within the unsupervised block

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

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
    scaler = GradScaler('cuda')
    print("model created")
    sys.stdout.flush()

    ema_weights = None
    for epoch in range(EPOCHS):
        model.train()
        use_ssl = epoch >= WARM_UP_EPOCH_SSL
        total_loss = torch.zeros((), device=device)
        n_batches = 0
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
                    # ---- labeled: SupCon, full batch, no augmentation ----
                    emb_lab = model(f_lab_b, s_lab_b)
                    loss_sup = supervised_contrastive_loss(emb_lab, y_lab_b, temperature=SUP_TEMPERATURE)

                    # ---- unlabeled: support-set consistency + me-max ----
                    #f_weak, s_weak = weak_augment_pair(f_unl_b, s_unl_b)
                    f_weak, s_weak = f_unl_b, s_unl_b
                    f_strong, s_strong = strong_augment_pair(f_unl_b, s_unl_b)

                    f_sup, s_sup, y_sup = sample_support_set(
                        f_lab_b, s_lab_b, y_lab_b, n_classes, k_per_class=K_PER_CLASS)
                    support_emb = model(f_sup, s_sup)          # grad-enabled (student path uses this)

                    with torch.no_grad():
                        emb_weak = model(f_weak, s_weak)
                        probs_weak = soft_knn_probs(emb_weak, support_emb.detach(), y_sup,
                                                     n_classes, temperature=KNN_TEMPERATURE)

                    emb_strong = model(f_strong, s_strong)
                    probs_strong = soft_knn_probs(emb_strong, support_emb, y_sup,
                                                   n_classes, temperature=KNN_TEMPERATURE)

                    #loss_consistency = consistency_loss(probs_weak, probs_strong, sharpen_T=SHARPEN_T)
                    #loss_consistency = consistency_loss_l1(probs_weak, probs_strong, sharpen_T=SHARPEN_T)
                    #loss_consistency = consistency_loss_l2(probs_weak, probs_strong, sharpen_T=SHARPEN_T)
                    loss_consistency = consistency_loss_embedding_cosine(emb_weak, emb_strong)


                    loss_me = mean_entropy_max_loss(probs_strong)

                    #loss = loss_sup + LAMBDA_U * (loss_consistency + LAMBDA_ME * loss_me)
                    loss = loss_sup + LAMBDA_U * loss_consistency

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.detach()
                n_batches += 1

                with torch.no_grad():
                    hard_preds = probs_strong.argmax(dim=1)
                    majority_frac = (hard_preds.bincount(minlength=n_classes).max() / hard_preds.numel()).item()
                    majority_frac_sum += majority_frac
                    majority_frac_batches += 1
        else:
            # warmup: labeled SupCon only, no unlabeled pool touched
            for f_lab_b, s_lab_b, y_lab_b in dataloader_lab_train:
                optimizer.zero_grad(set_to_none=True)
                f_lab_b = f_lab_b.to(device, non_blocking=True)
                s_lab_b = s_lab_b.to(device, non_blocking=True)
                y_lab_b = y_lab_b.to(device, non_blocking=True)

                with autocast('cuda'):
                    emb_lab = model(f_lab_b, s_lab_b)
                    loss = supervised_contrastive_loss(emb_lab, y_lab_b, temperature=SUP_TEMPERATURE)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.detach()
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
                model.load_state_dict(current_state_dict)
            else:
                ref_emb = compute_reference_embeddings(model, x_tensor_f_lab, x_tensor_s_lab, device)
                predictions, test_labels = evaluate_with_knn(
                    model, ref_emb, y_tensor.to(device), dataloader_test, n_classes, device, k=K_NEIGHBORS)

            f1_val = f1_score(test_labels, predictions, average="weighted")
            avg_majority_frac = majority_frac_sum / max(majority_frac_batches, 1)
            print(f"epoch {epoch} total={total_loss.item() / max(n_batches, 1):.4f} "
                  f"F1-score={(f1_val * 100):.2f} majority_frac={avg_majority_frac:.3f} "
                  f"ssl={'on' if use_ssl else 'off (warmup)'}")
            sys.stdout.flush()

    if ema_weights is not None:
        model.load_state_dict(ema_weights)
    torch.save(model.state_dict(), output_file)
    print("Saved to %s" % output_file)