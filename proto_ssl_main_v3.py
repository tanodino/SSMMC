"""
Non-parametric semi-supervised classifier: support-set consistency +
InfoNCE + a jointly-trained classification head.

--------------------------------------------------------------------------
Components (see conversation for full derivation of each)
--------------------------------------------------------------------------
1. LABELED LOSS -- cross-entropy from the classifier head baked directly
   into forward() (SupCon has been removed; see conversation for why
   DETACH-style gradient protection is no longer optional once SupCon is
   gone -- CE is now the ONLY path for labels to reach h, so it must flow
   through un-detached, which is exactly what this version does).

2. UNLABELED LOSS, two parts:
   a. SUPPORT-SET machinery (PAWS-style; Assran et al., ICCV 2021) is
      still computed (soft_knn_probs, majority_frac diagnostic), but
      mean-entropy-max (loss_me) is NOT currently included in the
      optimized loss -- it's diagnostic-only right now.
   b. InfoNCE / NT-Xent consistency between weak/strong augmented views,
      computed in z-space (project_for_ssl(h) -> z), not h directly --
      SimCLR-style h/z split, protects h from InfoNCE's
      instance-discrimination-specific pressure.

3. EVALUATION: k-NN (distance-weighted vote), a post-hoc linear probe, and
   the jointly-trained classification head -- all evaluated side by side,
   all reading h/classif from the SAME forward() call.

--------------------------------------------------------------------------
forward() returns a TUPLE: (classif, h). classif = raw classifier logits
(gradient flows straight through, no detach -- see above). h = L2-normalized
embedding, used everywhere else (support-set comparisons, k-NN, linear
probe, InfoNCE via project_for_ssl). Every call site must unpack both
values, even where only one is used (`_, h = model(...)` or
`classif, _ = model(...)`).
--------------------------------------------------------------------------

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

    forward() returns (classif, h) -- BOTH computed every call. classif
    is the classifier's raw logits (self.classifier(emb), gradient flows
    normally -- SupCon has been removed, so CE via classif is now the
    ONLY path for labels to shape h; there is no detach option anymore
    since classify() was merged directly into forward()). h is the
    L2-normalized embedding used by everything else.

    ALSO includes a small, SEPARATE ssl_head (SimCLR-style h/z split;
    Chen et al., 2020): project_for_ssl(h) -> z, called explicitly ONLY
    at the InfoNCE call site, on an ALREADY-COMPUTED h (never on raw
    modality inputs)."""

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
            nn.Linear(512, proj_dim), nn.BatchNorm1d(proj_dim)
        )

        # SSL-only head: h -> z, called explicitly via project_for_ssl(h).
        # Never read by support-set comparisons or evaluation.
        #
        # NOTE (from conversation): BatchNorm vs LayerNorm was tested
        # directly. LayerNorm removes the running-statistics train/eval
        # discrepancy and produces a genuine stable plateau, but that
        # plateau sits below where BatchNorm's still-declining trajectory
        # was at the same epoch -- BatchNorm's noise appears to confer
        # real regularization benefit here (same pattern seen earlier
        # comparing BatchNorm vs GroupNorm on a ResNet18 SF baseline).
        # Kept BatchNorm; weight decay + gradient clipping (see main loop)
        # are being used instead to address the drift without giving up
        # BatchNorm's higher ceiling. Swap to nn.LayerNorm here if you
        # want the stability-first tradeoff instead.
        self.ssl_head = nn.Sequential(
            nn.LazyLinear(512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, proj_dim), nn.BatchNorm1d(proj_dim)
        )

        # Linear classification head. Computed directly inside forward()
        # (not a separate method anymore) -- gradient flows into h
        # un-detached, since SupCon has been removed and CE is now the
        # only source of label-aware gradient for h.
        self.classifier = nn.Linear(proj_dim, config.num_classes)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        cls_m1 = self.modality_1_encoder(x1)
        cls_m2 = self.modality_2_encoder(x2)
        concat = torch.cat([cls_m1, cls_m2], dim=1)
        emb = self.fusion(concat)
        h = F.normalize(emb, dim=1)
        classif = self.classifier(emb)
        return classif, h                                  # ALWAYS a 2-tuple

    def project_for_ssl(self, h: torch.Tensor) -> torch.Tensor:
        """h -> z (L2-normalized embedding, NOT logits -- used only as the
        InfoNCE consistency space). Takes an ALREADY-COMPUTED h, not raw
        modality inputs."""
        z = self.ssl_head(h)
        return F.normalize(z, dim=1)


# ==========================================================================
# InfoNCE consistency (unlabeled)
# ==========================================================================

def consistency_loss_infonce(emb_weak: torch.Tensor, emb_strong: torch.Tensor,
                              temperature: float = 0.2) -> torch.Tensor:
    """InfoNCE / NT-Xent style consistency. Called with z (project_for_ssl
    output), NOT h directly.

    strong-view embedding i is the ANCHOR, weak-view embedding i (same
    sample, detached) is the POSITIVE, every OTHER sample's weak-view
    embedding in the batch is a NEGATIVE.

    NOTE (from conversation): this loss showed a "never saturates"
    over-optimization pattern -- loss_consistency keeps falling slowly
    while F1 doesn't track it 1:1. Weight decay + gradient clipping are
    being used as the primary countermeasure (see main loop) rather than
    best-checkpoint selection on the test set, which would be unfair
    given this setup has no separate validation set."""
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
# Support-set machinery (currently diagnostic-only -- see loss_me below)
# ==========================================================================

def sample_support_set(f_lab, s_lab, y_lab, n_classes, k_per_class=1):
    """Fresh random support set: k_per_class labeled samples per class,
    without replacement, resampled every call."""
    idx = []
    for c in range(n_classes):
        class_idx = (y_lab == c).nonzero(as_tuple=True)[0]
        n_take = min(k_per_class, len(class_idx))
        chosen = class_idx[torch.randperm(len(class_idx), device=class_idx.device)[:n_take]]
        idx.append(chosen)
    idx = torch.cat(idx)
    return f_lab[idx], s_lab[idx], y_lab[idx]


def logit_standardize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / (std + eps)


def soft_knn_probs(query_emb: torch.Tensor, ref_emb: torch.Tensor, ref_labels: torch.Tensor,
                    n_classes: int, temperature: float = 0.1, standardize: bool = False) -> torch.Tensor:
    sims = query_emb @ ref_emb.T
    if standardize:
        sims = logit_standardize(sims)
    sims = sims / temperature
    weights = F.softmax(sims, dim=1)
    one_hot_labels = F.one_hot(ref_labels, n_classes).float()
    return weights @ one_hot_labels


def mean_entropy_max_loss(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Penalizes the BATCH-AVERAGED prediction for deviating from uniform.
    Already sign-flipped: minimizing this MAXIMIZES entropy. NOTE:
    computed every step for diagnostics (majority_frac) but NOT currently
    included in the optimized loss -- see main loop."""
    mean_p = probs.mean(dim=0)
    entropy = -(mean_p * torch.log(mean_p.clamp(min=eps))).sum()
    return -entropy


# ==========================================================================
# Evaluation: k-NN, post-hoc linear probe, jointly-trained classifier
# ==========================================================================

@torch.no_grad()
def compute_reference_embeddings(model: ProtoModel, f_lab: torch.Tensor, s_lab: torch.Tensor,
                                  device) -> torch.Tensor:
    model.eval()
    _, h = model(f_lab.to(device), s_lab.to(device))       # unpack the (classif, h) tuple, keep h
    return h


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
        _, emb = model(f_batch, s_batch)                     # unpack (classif, h), keep h
        preds = knn_classify(emb, ref_emb, ref_labels, n_classes, k=k).cpu()
        all_preds.append(preds)
        all_labels.append(y_batch)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def evaluate_with_linear_probe(model: ProtoModel, f_lab: torch.Tensor, s_lab: torch.Tensor,
                                y_lab: torch.Tensor, dataloader, device, C: float = 1.0):
    """"Linear probe" evaluation, refit from scratch each call (standard
    SSL protocol -- SimCLR, MoCo, BYOL all report this)."""
    model.eval()
    with torch.no_grad():
        _, ref_emb_t = model(f_lab.to(device), s_lab.to(device))
        ref_emb = ref_emb_t.cpu().numpy()
    ref_labels = y_lab.numpy()

    clf = LogisticRegression(max_iter=1000, C=C)
    clf.fit(ref_emb, ref_labels)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for f_batch, s_batch, y_batch in dataloader:
            f_batch = f_batch.to(device, non_blocking=True)
            s_batch = s_batch.to(device, non_blocking=True)
            _, emb_t = model(f_batch, s_batch)
            emb = emb_t.cpu().numpy()
            preds = clf.predict(emb)
            all_preds.append(preds)
            all_labels.append(y_batch.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


@torch.no_grad()
def evaluate_with_classifier(model: ProtoModel, dataloader, device):
    """Reads predictions directly from forward()'s classif output -- the
    head trained jointly, continuously, during the SSL loop. classify()
    is no longer a separate method; forward() already computes it."""
    model.eval()
    all_preds, all_labels = [], []
    for f_batch, s_batch, y_batch in dataloader:
        f_batch = f_batch.to(device, non_blocking=True)
        s_batch = s_batch.to(device, non_blocking=True)
        classif, _ = model(f_batch, s_batch)                 # unpack (classif, h), keep classif
        preds = classif.argmax(dim=1).cpu()
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
    KNN_TEMPERATURE = 0.1     # soft-label similarity temperature (support-set comparisons)
    INFO_NCE_TEMPERATURE = 1.0  # NOTE: SimCLR-typical range is 0.1-0.5; 1.0 is much flatter --
                                 # weaker anti-collapse signal. Worth sweeping down (e.g. 0.2).
    K_PER_CLASS = 1           # support-set size per class, resampled every step (diagnostic only right now)
    K_NEIGHBORS = 5           # k for FINAL evaluation k-NN (against full labeled set)
    LAMBDA_U = 1.0            # weight of the InfoNCE consistency term
    LAMBDA_CLS = 1.0          # weight of the classifier's CE loss

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
    scaler = GradScaler('cuda')
    print("model created")
    sys.stdout.flush()

    ema_weights = None
    for epoch in range(EPOCHS):
        model.train()
        use_ssl = epoch >= WARM_UP_EPOCH_SSL
        total_loss = torch.zeros((), device=device)
        loss_cls_sum = torch.zeros((), device=device)
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
                    # ---- labeled: classifier CE (SupCon removed), full batch, no augmentation ----
                    classif, emb_lab = model(f_lab_b, s_lab_b)
                    loss_cls = F.cross_entropy(classif, y_lab_b)

                    # ---- unlabeled: support-set diagnostics + InfoNCE ----
                    f_weak, s_weak = weak_augment_pair(f_unl_b, s_unl_b)
                    f_strong, s_strong = strong_augment_pair(f_unl_b, s_unl_b)

                    f_sup, s_sup, y_sup = sample_support_set(
                        f_lab_b, s_lab_b, y_lab_b, n_classes, k_per_class=K_PER_CLASS)
                    _, support_emb = model(f_sup, s_sup)          # grad-enabled

                    with torch.no_grad():
                        _, emb_weak = model(f_weak, s_weak)
                        proj_weak = model.project_for_ssl(emb_weak)
                        probs_weak = soft_knn_probs(emb_weak, support_emb.detach(), y_sup,
                                                     n_classes, temperature=KNN_TEMPERATURE)

                    _, emb_strong = model(f_strong, s_strong)
                    proj_strong = model.project_for_ssl(emb_strong)
                    probs_strong = soft_knn_probs(emb_strong, support_emb, y_sup,
                                                   n_classes, temperature=KNN_TEMPERATURE)

                    loss_consistency = consistency_loss_infonce(proj_weak, proj_strong, temperature=INFO_NCE_TEMPERATURE)
                    loss_me = mean_entropy_max_loss(probs_strong)   # diagnostic only, not in loss

                    loss = LAMBDA_U * loss_consistency + LAMBDA_CLS * loss_cls

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.detach()
                loss_cls_sum += loss_cls.detach()
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
            # warmup: classifier CE only, no unlabeled pool touched
            for f_lab_b, s_lab_b, y_lab_b in dataloader_lab_train:
                optimizer.zero_grad(set_to_none=True)
                f_lab_b = f_lab_b.to(device, non_blocking=True)
                s_lab_b = s_lab_b.to(device, non_blocking=True)
                y_lab_b = y_lab_b.to(device, non_blocking=True)

                with autocast('cuda'):
                    classif, emb_lab = model(f_lab_b, s_lab_b)
                    loss_cls = F.cross_entropy(classif, y_lab_b)
                    loss = LAMBDA_CLS * loss_cls

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.detach()
                loss_cls_sum += loss_cls.detach()
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
            avg_loss_cls = loss_cls_sum.item() / max(n_batches, 1)
            avg_loss_consistency = loss_consistency_sum.item() / max(n_ssl_batches, 1)
            avg_loss_me = loss_me_sum.item() / max(n_ssl_batches, 1)

            print(f"epoch {epoch} total={total_loss.item() / max(n_batches, 1):.4f} "
                  f"loss_cls={avg_loss_cls:.4f} loss_consistency={avg_loss_consistency:.4f} "
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