"""
Resume full pretrained model (encoders + projectors), continue the ORIGINAL
contrastive SSL objective, add SUPERVISED CONTRASTIVE loss for labeled
supervision, classify via k-NN. No trained classifier head anywhere.

--------------------------------------------------------------------------
How this differs from resume_pretrain_supervised_main.py
--------------------------------------------------------------------------
Same philosophy (resume the FULL pretrained model, continue the exact
validated contrastive objective, never generate a pseudo-label), but the
labeled-data supervision mechanism is swapped:

  resume_pretrain_supervised_main.py : classifier(concat(cls_m1,cls_m2)) + CE
  THIS script                         : SupCon on concat(cls_m1,cls_m2),
                                         classification via k-NN against the
                                         labeled set (no trained head at all)

Consequence: since nothing is freshly initialized anymore (encoders AND
both projectors are all resumed from the checkpoint), there's no more
"pretrained vs fresh" distinction driving a differential learning rate --
a single, gentle LR is used for the whole model.

--------------------------------------------------------------------------
Loss terms
--------------------------------------------------------------------------
  loss_m1, loss_m2 : UNCHANGED from pretrain.py -- NTXentLoss instance-
                      discrimination on each modality's own projector output
  loss_cross        : UNCHANGED -- NTXentLoss on the shared/invariant
                       subspace, operates DIRECTLY on encoder output
  loss_sup           : NEW -- Supervised Contrastive (Khosla et al., 2020)
                        on concat(cls_m1, cls_m2) for the labeled batch,
                        full-batch (no augmentation, matching proto_ssl_main.py's
                        reasoning: with only ~50 labeled samples, full-batch
                        guarantees every anchor sees its same-class peers)

CAVEAT: loss_sup operates on RAW CLS TOKENS (same space as loss_cross), so
if you freeze the encoders, loss_sup becomes gradient-inert too (same
consequence loss_cross already has when frozen) -- only loss_m1/loss_m2
(via the still-trainable projectors) do real work in frozen mode. See
conversation for the alternative (SupCon on projector outputs instead)
if you want supervision to remain live under freezing.

Usage:
    python resume_pretrain_supcon_knn_main.py EUROSAT SAR MS 5 0 \\
        EUROSAT/PRETRAIN/checkpoint_latest.pth [freeze]
"""

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
from functions import (strong_augment_pair, NTXentLoss, MOMENTUM_EMA, cumulate_EMA,
                        WARM_UP_EPOCH_EMA, EPOCHS, RATIO_LABELED_UNLABELED_BATCHES)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ==========================================================================
# Model: EXACTLY pretrain.py's PretrainModel -- no new modules at all
# ==========================================================================

class PretrainModel(nn.Module):
    """Identical to pretrain.py's PretrainModel -- no classifier, no new
    modules of any kind. The checkpoint's state_dict loads with
    strict=True since architecture matches exactly."""

    def __init__(self, config: SFFCConfig):
        super().__init__()
        self.modality_1_encoder = ViTEncoder(
            img_size=config.img_size_m1, patch_size=config.patch_size_m1,
            in_chans=config.in_chans_m1,
        )
        self.modality_2_encoder = ViTEncoder(
            img_size=config.img_size_m2, patch_size=config.patch_size_m2,
            in_chans=config.in_chans_m2,
        )
        self.projector_m1 = nn.Sequential(
            nn.LazyLinear(512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 128), nn.BatchNorm1d(128),
        )
        self.projector_m2 = nn.Sequential(
            nn.LazyLinear(512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 128), nn.BatchNorm1d(128),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        cls_token_m1 = self.modality_1_encoder(x1)   # [B, embed_dim]
        cls_token_m2 = self.modality_2_encoder(x2)
        proj_m1 = self.projector_m1(cls_token_m1)
        proj_m2 = self.projector_m2(cls_token_m2)
        return cls_token_m1, cls_token_m2, proj_m1, proj_m2


def load_full_pretrained_checkpoint(model: PretrainModel, path: str, device: str):
    """Loads the FULL periodic checkpoint (encoders + both projectors).
    strict=True -- model architecture exactly matches the checkpoint, no
    new modules to reconcile (unlike the classifier version of this script)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    print("Loaded full pretrained model (encoders + projectors) from %s" % path)


def freeze_pretrained_backbone(model: PretrainModel, freeze_projectors: bool = False):
    for p in model.modality_1_encoder.parameters():
        p.requires_grad = False
    for p in model.modality_2_encoder.parameters():
        p.requires_grad = False
    if freeze_projectors:
        for p in model.projector_m1.parameters():
            p.requires_grad = False
        for p in model.projector_m2.parameters():
            p.requires_grad = False


# ==========================================================================
# Supervised Contrastive loss (Khosla et al., 2020) -- labeled supervision
# ==========================================================================

def supervised_contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor,
                                 temperature: float = 0.1) -> torch.Tensor:
    """Same-class = positive pair, different-class = negative pair.
    embeddings assumed L2-normalized."""
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
# Evaluation: hard top-k k-NN AND soft (full-reference-set) k-NN, compared side by side
# ==========================================================================

@torch.no_grad()
def compute_reference_embedding(model: PretrainModel, f_lab: torch.Tensor, s_lab: torch.Tensor, device):
    model.eval()
    cls_m1, cls_m2, _, _ = model(f_lab.to(device), s_lab.to(device))
    return F.normalize(torch.cat([cls_m1, cls_m2], dim=1), dim=1)


@torch.no_grad()
def knn_classify_hard(query_emb: torch.Tensor, ref_emb: torch.Tensor, ref_labels: torch.Tensor,
                       n_classes: int, k: int = 5) -> torch.Tensor:
    """Distance-weighted vote among the TOP-K nearest labeled samples."""
    k = min(k, ref_emb.shape[0])
    sims = query_emb @ ref_emb.T
    topk_sims, topk_idx = sims.topk(k, dim=1)
    topk_labels = ref_labels[topk_idx]
    class_scores = torch.zeros(query_emb.shape[0], n_classes, device=query_emb.device)
    class_scores.scatter_add_(1, topk_labels, topk_sims.clamp(min=0))
    return class_scores.argmax(dim=1)


@torch.no_grad()
def soft_knn_probs(query_emb: torch.Tensor, ref_emb: torch.Tensor, ref_labels: torch.Tensor,
                    n_classes: int, temperature: float = 0.1) -> torch.Tensor:
    """[Bq, C] soft distribution: softmax over similarity to ALL reference
    samples (not just top-k), aggregated by class via a one-hot matmul."""
    sims = query_emb @ ref_emb.T / temperature
    weights = F.softmax(sims, dim=1)
    one_hot_labels = F.one_hot(ref_labels, n_classes).float()
    return weights @ one_hot_labels


@torch.no_grad()
def evaluate(model: PretrainModel, ref_emb: torch.Tensor, ref_labels: torch.Tensor,
             dataloader, n_classes: int, device, k: int = 5, soft_temperature: float = 0.1):
    model.eval()
    hard_preds, soft_preds, all_labels = [], [], []
    for f_batch, s_batch, y_batch in dataloader:
        f_batch = f_batch.to(device, non_blocking=True)
        s_batch = s_batch.to(device, non_blocking=True)
        cls_m1, cls_m2, _, _ = model(f_batch, s_batch)
        emb = F.normalize(torch.cat([cls_m1, cls_m2], dim=1), dim=1)

        hard_preds.append(knn_classify_hard(emb, ref_emb, ref_labels, n_classes, k=k).cpu())
        probs = soft_knn_probs(emb, ref_emb, ref_labels, n_classes, temperature=soft_temperature)
        soft_preds.append(probs.argmax(dim=1).cpu())
        all_labels.append(y_batch)
    return (torch.cat(hard_preds).numpy(), torch.cat(soft_preds).numpy(), torch.cat(all_labels).numpy())


if __name__ == "__main__":
    batch_size = 16
    dataset_path = sys.argv[1]
    first_prefix = sys.argv[2]
    second_prefix = sys.argv[3]
    perc = sys.argv[4]
    run_id = sys.argv[5]
    checkpoint_path = sys.argv[6]
    freeze_encoder = "freeze" in sys.argv
    print(sys.argv)

    # ---- tunables ----
    SHARED_UNSHARED = 50      # invariant/specific split, matches pretrain.py's default
    SUP_TEMPERATURE = 0.1     # SupCon loss temperature
    SOFT_KNN_TEMPERATURE = 0.1  # eval-time soft-knn temperature
    LAMBDA_SUP = 1.0          # weight of the SupCon loss
    K_NEIGHBORS = 5           # k for the hard top-k k-NN comparison metric
    MODEL_LR = 1e-5           # single LR for the whole model -- everything trainable is
                               # pretrained (no fresh classifier anymore), so this replaces
                               # the previous BACKBONE_LR/CLASSIFIER_LR split. Still gentle
                               # relative to earlier from-scratch runs (5e-5), given the
                               # goal is careful refinement, not aggressive relearning.

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

    dir_name = dataset_path + "/RESUME_PRETRAIN_SUPCON_KNN"
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

    model = PretrainModel(config).to(device)
    load_full_pretrained_checkpoint(model, checkpoint_path, device)

    if freeze_encoder:
        freeze_pretrained_backbone(model, freeze_projectors=False)
        print("Encoders FROZEN (projectors still trainable)")
        print("NOTE: loss_sup and loss_cross both operate on raw CLS tokens -- "
              "with encoders frozen, BOTH become gradient-inert. Only loss_m1/"
              "loss_m2 (via the still-trainable projectors) do real work in "
              "this mode. See script docstring.")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=MODEL_LR, weight_decay=1e-4,
    )
    scaler = GradScaler('cuda')
    print("model created")
    sys.stdout.flush()

    ema_weights = None
    for epoch in range(EPOCHS):
        model.train()
        total_loss = torch.zeros((), device=device)
        loss_m1_sum = torch.zeros((), device=device)
        loss_m2_sum = torch.zeros((), device=device)
        loss_cross_sum = torch.zeros((), device=device)
        loss_sup_sum = torch.zeros((), device=device)
        n_batches = 0

        for f_batch_unl, s_batch_unl in dataloader_unl_train:
            optimizer.zero_grad(set_to_none=True)
            f_batch_unl = f_batch_unl.to(device, non_blocking=True)
            s_batch_unl = s_batch_unl.to(device, non_blocking=True)

            # labeled batch, cycled independently (full labeled set every step)
            f_lab_b, s_lab_b, y_lab_b = next(iter(dataloader_lab_train))
            f_lab_b = f_lab_b.to(device, non_blocking=True)
            s_lab_b = s_lab_b.to(device, non_blocking=True)
            y_lab_b = y_lab_b.to(device, non_blocking=True)

            with autocast('cuda'):
                # ---- unlabeled: ORIGINAL contrastive objective, unchanged ----
                f_strong, s_strong = strong_augment_pair(f_batch_unl, s_batch_unl)
                cls_token_m1, cls_token_m2, proj_m1, proj_m2 = model(f_batch_unl, s_batch_unl)
                _, _, proj_m1_aug, proj_m2_aug = model(f_strong, s_strong)

                n_feat = cls_token_m1.shape[-1]
                shared_n_feat = int(n_feat * SHARED_UNSHARED / 100)

                emb_m1_inv = cls_token_m1[:, :shared_n_feat]
                emb_m2_inv = cls_token_m2[:, :shared_n_feat]
                emb_inv = F.normalize(torch.cat([emb_m1_inv, emb_m2_inv], dim=0), dim=1)

                repr_m1 = F.normalize(torch.cat([proj_m1, proj_m1_aug], dim=0), dim=1)
                repr_m2 = F.normalize(torch.cat([proj_m2, proj_m2_aug], dim=0), dim=1)

                labels_cls_loss = torch.arange(f_batch_unl.shape[0]).repeat(2).to(device)

                loss_m1 = NTXentLoss(repr_m1, labels_cls_loss, temperature=1.0)
                loss_m2 = NTXentLoss(repr_m2, labels_cls_loss, temperature=1.0)
                loss_cross = NTXentLoss(emb_inv, labels_cls_loss, temperature=1.0)

                # ---- labeled: SupCon, NOT a classifier -- no pseudo-labeling ----
                cls_token_m1_lab, cls_token_m2_lab, _, _ = model(f_lab_b, s_lab_b)
                emb_lab = F.normalize(torch.cat([cls_token_m1_lab, cls_token_m2_lab], dim=1), dim=1)
                loss_sup = supervised_contrastive_loss(emb_lab, y_lab_b, temperature=SUP_TEMPERATURE)

                loss = 0.5 * (loss_m1 + loss_m2) + loss_cross + LAMBDA_SUP * loss_sup

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.detach()
            loss_m1_sum += loss_m1.detach()
            loss_m2_sum += loss_m2.detach()
            loss_cross_sum += loss_cross.detach()
            loss_sup_sum += loss_sup.detach()
            n_batches += 1

        if epoch >= WARM_UP_EPOCH_EMA:
            ema_weights = cumulate_EMA(model, ema_weights, MOMENTUM_EMA)

        if epoch % 5 == 0:
            if epoch >= WARM_UP_EPOCH_EMA:
                current_state_dict = copy.deepcopy(model.state_dict())
                model.load_state_dict(ema_weights)
                ref_emb = compute_reference_embedding(model, x_tensor_f_lab, x_tensor_s_lab, device)
                hard_preds, soft_preds, test_labels = evaluate(
                    model, ref_emb, y_tensor.to(device), dataloader_test, n_classes, device,
                    k=K_NEIGHBORS, soft_temperature=SOFT_KNN_TEMPERATURE)
                model.load_state_dict(current_state_dict)
            else:
                ref_emb = compute_reference_embedding(model, x_tensor_f_lab, x_tensor_s_lab, device)
                hard_preds, soft_preds, test_labels = evaluate(
                    model, ref_emb, y_tensor.to(device), dataloader_test, n_classes, device,
                    k=K_NEIGHBORS, soft_temperature=SOFT_KNN_TEMPERATURE)

            f1_hard = f1_score(test_labels, hard_preds, average="weighted")
            f1_soft = f1_score(test_labels, soft_preds, average="weighted")

            print(f"epoch {epoch} total={total_loss.item() / max(n_batches, 1):.4f} "
                  f"loss_m1={loss_m1_sum.item() / max(n_batches, 1):.4f} "
                  f"loss_m2={loss_m2_sum.item() / max(n_batches, 1):.4f} "
                  f"loss_cross={loss_cross_sum.item() / max(n_batches, 1):.4f} "
                  f"loss_sup={loss_sup_sum.item() / max(n_batches, 1):.4f} "
                  f"F1-knn-hard={(f1_hard * 100):.2f} F1-knn-soft={(f1_soft * 100):.2f}")
            sys.stdout.flush()

    if ema_weights is not None:
        model.load_state_dict(ema_weights)
    torch.save(model.state_dict(), output_file)
    print("Saved to %s" % output_file)