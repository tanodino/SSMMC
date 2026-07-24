"""
Resume full pretrained model, continue original SSL objective, classify
via a LIGHTWEIGHT multi-layer fusion head instead of last-layer-only.

--------------------------------------------------------------------------
What changed relative to resume_pretrain_supervised_main.py
--------------------------------------------------------------------------
Motivated by "Attentive Multi-Layer Fusion for Vision Transformers"
(Ciernik et al., 2026): task-relevant information is distributed across
ALL transformer layers, not just the last one -- their method (ALF) fuses
CLS (+AP) tokens from every layer via learned cross-attention, gaining
5.54pp average over last-layer linear probing across 20 datasets,
including EuroSAT specifically.

TWO deliberate departures from the paper, both due to your ~50-sample
labeled budget (see conversation for the parameter-count correction):

1. QUARTERLY layers only (not all 12) -- cheaper attention computation.
   NOTE: this does NOT reduce the attentive fusion module's PARAMETER
   count (see LightweightLayerFusion vs the commented-out
   AttentiveLayerFusion below) -- only the linear/concatenation fusion
   strategy's parameter count scales with layer count. Quarterly layers
   here is about picking a representative subset of depths (paper's own
   validated cheaper variant, Fig. 2's "+1/4 and 3/4 blocks"), not about
   parameter savings.

2. LIGHTWEIGHT fusion (a learned scalar weight per layer, softmax-
   normalized, weighted sum -- O(L) parameters) instead of the paper's
   full multi-head cross-attention (O(d^2) parameters, ~1.18M for
   embed_dim=384, PER MODALITY -- see conversation). Still implements the
   paper's core idea (learn which layers matter) at a parameter count
   that doesn't dwarf your labeled set by 15-300x.

Backbone is UNFROZEN by default (per your request) and continues to be
updated by loss_m1/loss_m2/loss_cross, exactly as in
resume_pretrain_supervised_main.py -- nothing about the original
contrastive objective changes.

Usage:
    python resume_pretrain_alf_main.py EUROSAT SAR MS 5 0 \\
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
# ViTEncoder extension: expose intermediate-layer CLS tokens
# ==========================================================================
# NOT modifying model.py's ViTEncoder (used by every other script today) --
# this is a free function that reimplements the forward pass manually,
# leaving ViTEncoder itself untouched. Functionally identical to
# self.transformer(x) for the parts it shares (nn.TransformerEncoder here
# has no final norm of its own -- ViTEncoder applies self.norm separately
# -- so manually chaining layer(x) calls is exactly equivalent).

def get_quarterly_layer_indices(depth: int) -> list:
    """0-based indices into encoder.transformer.layers for the paper's
    "quarterly" variant: ~1/4, ~1/2, ~3/4, and the last block."""
    idx = sorted(set([
        max(0, depth // 4 - 1),
        max(0, depth // 2 - 1),
        max(0, (3 * depth) // 4 - 1),
        depth - 1,
    ]))
    return idx


def encoder_forward_all_layers(encoder: ViTEncoder, x: torch.Tensor, layer_indices: list) -> torch.Tensor:
    """Returns CLS tokens from the SPECIFIED layers (0-based indices into
    encoder.transformer.layers), stacked as [B, len(layer_indices), embed_dim].

    NOTE: manually iterates encoder.transformer.layers instead of calling
    encoder.transformer(x) as a black box, since nn.TransformerEncoder's
    standard forward only exposes the FINAL layer's output.

    NOTE: encoder.norm (meant only for the true final layer in the
    original forward()) is applied to EVERY collected layer here, to give
    all collected representations a consistent scale before fusion --
    simplest choice for a first test; a separate per-layer norm is a
    reasonable alternative if this scale-sharing turns out to matter."""
    B = x.shape[0]
    x = encoder.patch_embed(x)
    x = x + encoder.pos_embed[:, 1:, :]
    cls_token = encoder.cls_token + encoder.pos_embed[:, :1, :]
    cls_tokens = cls_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    x = encoder.dropout(x)

    layer_indices_set = set(layer_indices)
    collected = {}
    for i, layer in enumerate(encoder.transformer.layers):
        x = layer(x)
        if i in layer_indices_set:
            collected[i] = encoder.norm(x)[:, 0, :]   # CLS token, normalized

    return torch.stack([collected[i] for i in layer_indices], dim=1)   # [B, L, D]


# ==========================================================================
# Multi-layer fusion modules
# ==========================================================================

class LightweightLayerFusion(nn.Module):
    """A learned weight per layer combining per-layer CLS tokens.

    gating='softmax' : ORIGINAL design. Non-negative, weights sum to 1 --
                        layers COMPETE for a shared budget (raising one
                        layer's weight necessarily lowers another's).
                        Implicitly bounds the fused output's scale to at
                        most the largest individual layer's magnitude.
    gating='sigmoid'  : RECOMMENDED (see conversation). Each layer's
                        weight in (0,1), decided INDEPENDENTLY -- no
                        competition, so genuinely complementary layers
                        (per the paper's own finding) can both be used
                        fully rather than trading off. Still non-negative
                        (no "subtracting" a layer), keeping the
                        combination easy to reason about.
    gating='tanh'     : most expressive -- allows NEGATIVE weights (a
                        layer can be subtracted, not just included/
                        excluded). Bigger, more speculative departure;
                        nothing tested so far validates that subtracting
                        layer representations helps. Try only if sigmoid's
                        ceiling turns out to be the actual bottleneck.

    post_norm: LayerNorm on the fused output. Sigmoid/tanh lack softmax's
    implicit scale bound (fused magnitude can grow up to ~L x a single
    layer's scale if several gates open near 1/-1 simultaneously) -- cheap
    insurance against the kind of scale-driven instability seen elsewhere
    today (BatchNorm drift, InfoNCE). On by default for anything other
    than softmax; off by default for softmax to match the already-tested
    version exactly."""

    #def __init__(self, num_layers: int, embed_dim: int, gating: str = "sigmoid", post_norm: bool = None):
    def __init__(self, num_layers: int, embed_dim: int, gating: str = "sigmoid", post_norm: bool = None):
        super().__init__()
        assert gating in ("softmax", "sigmoid", "tanh")
        self.gating = gating
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))

        if post_norm is None:
            post_norm = (gating != "softmax")
        self.norm = nn.LayerNorm(embed_dim) if post_norm else None

    def forward(self, layer_tokens: torch.Tensor) -> torch.Tensor:
        # layer_tokens: [B, L, D]
        if self.gating == "softmax":
            weights = F.softmax(self.layer_logits, dim=0)
        elif self.gating == "sigmoid":
            weights = torch.sigmoid(self.layer_logits)
        else:
            weights = torch.tanh(self.layer_logits)

        fused = (layer_tokens * weights.view(1, -1, 1)).sum(dim=1)   # [B, D]
        if self.norm is not None:
            fused = self.norm(fused)
        return fused

    def current_weights(self) -> torch.Tensor:
        """Returns the actual weights currently in use (for logging) --
        matches whichever gating function is active, unlike hardcoding
        F.softmax in the caller."""
        with torch.no_grad():
            if self.gating == "softmax":
                return F.softmax(self.layer_logits, dim=0)
            elif self.gating == "sigmoid":
                return torch.sigmoid(self.layer_logits)
            else:
                return torch.tanh(self.layer_logits)


# class AttentiveLayerFusion(nn.Module):
#     """Faithful to the paper: learnable query token, cross-attention over
#     per-layer CLS tokens (used as K/V). ~1.18M parameters for
#     embed_dim=384 REGARDLESS of how many layers are fused (O(d^2), see
#     module docstring) -- likely too large for a ~50-sample labeled set.
#     Kept here, unused, for a later comparison once more labels are
#     available or if the lightweight version underperforms enough to
#     justify the added capacity."""
#     def __init__(self, embed_dim: int, num_layers: int, dropout: float = 0.3):
#         super().__init__()
#         self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
#         assert embed_dim % num_layers == 0, "embed_dim must be divisible by num_layers (used as num_heads)"
#         self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=num_layers,
#                                                  dropout=dropout, batch_first=True)
#         self.norm = nn.BatchNorm1d(embed_dim)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, layer_tokens: torch.Tensor) -> torch.Tensor:
#         B = layer_tokens.shape[0]
#         q = self.query.expand(B, -1, -1)
#         fused, _ = self.cross_attn(q, layer_tokens, layer_tokens)
#         fused = self.norm(fused.squeeze(1))
#         return self.dropout(fused)


# ==========================================================================
# Model
# ==========================================================================

class PretrainModel(nn.Module):
    """Same encoders/projectors as pretrain.py (checkpoint loads with
    strict=False; only the NEW fusion + classifier modules are missing).
    forward() is UNCHANGED (used by loss_m1/loss_m2/loss_cross, exactly as
    before). classify_alf() is NEW -- multi-layer fusion instead of
    last-layer-only classification."""

    def __init__(self, config: SFFCConfig, num_classes: int, layer_indices: list, embed_dim: int = 384):
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

        self.layer_indices = layer_indices   # shared across both modalities (same depth assumed)

        # NEW -- not present in the pretraining checkpoint.
        self.alf_m1 = LightweightLayerFusion(num_layers=len(layer_indices), embed_dim=embed_dim)
        self.alf_m2 = LightweightLayerFusion(num_layers=len(layer_indices), embed_dim=embed_dim)
        self.classifier = nn.Linear(embed_dim * 2, num_classes)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """UNCHANGED -- used by the unlabeled contrastive losses exactly
        as in resume_pretrain_supervised_main.py."""
        cls_token_m1 = self.modality_1_encoder(x1)
        cls_token_m2 = self.modality_2_encoder(x2)
        proj_m1 = self.projector_m1(cls_token_m1)
        proj_m2 = self.projector_m2(cls_token_m2)
        return cls_token_m1, cls_token_m2, proj_m1, proj_m2

    def classify_alf(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """NEW classification path: multi-layer fusion instead of the
        final-layer-only CLS token. Runs its OWN forward pass through each
        encoder (via encoder_forward_all_layers) rather than reusing
        forward()'s output, since forward() only returns the final layer."""
        layers_m1 = encoder_forward_all_layers(self.modality_1_encoder, x1, self.layer_indices)  # [B, L, D]
        layers_m2 = encoder_forward_all_layers(self.modality_2_encoder, x2, self.layer_indices)
        fused_m1 = self.alf_m1(layers_m1)   # [B, D]
        fused_m2 = self.alf_m2(layers_m2)
        concat = torch.cat([fused_m1, fused_m2], dim=1)
        return self.classifier(concat), fused_m1, fused_m2   # logits + embeddings (for optional k-NN eval)


def load_full_pretrained_checkpoint(model: PretrainModel, path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    result = model.load_state_dict(state_dict, strict=False)

    unexpected = list(result.unexpected_keys)
    missing_non_new = [k for k in result.missing_keys
                        if not (k.startswith("classifier.") or k.startswith("alf_m1.") or k.startswith("alf_m2."))]

    if unexpected:
        print("WARNING: unexpected keys in checkpoint (not used): %s" % unexpected)
    if missing_non_new:
        raise RuntimeError(
            "Checkpoint is missing non-fusion/classifier keys -- something "
            "else doesn't match: %s" % missing_non_new
        )
    print("Loaded full pretrained model (encoders + projectors) from %s" % path)
    print("  (classifier + alf_m1 + alf_m2 are new, randomly initialized -- expected)")


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
# Evaluation: classifier readout + k-NN on the ALF-fused embedding
# ==========================================================================

@torch.no_grad()
def compute_reference_embedding(model: PretrainModel, f_lab: torch.Tensor, s_lab: torch.Tensor, device):
    model.eval()
    _, fused_m1, fused_m2 = model.classify_alf(f_lab.to(device), s_lab.to(device))
    return F.normalize(torch.cat([fused_m1, fused_m2], dim=1), dim=1)


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
def evaluate(model: PretrainModel, ref_emb: torch.Tensor, ref_labels: torch.Tensor,
             dataloader, n_classes: int, device, k: int = 5):
    model.eval()
    cls_preds, knn_preds, all_labels = [], [], []
    for f_batch, s_batch, y_batch in dataloader:
        f_batch = f_batch.to(device, non_blocking=True)
        s_batch = s_batch.to(device, non_blocking=True)
        logits, fused_m1, fused_m2 = model.classify_alf(f_batch, s_batch)
        cls_preds.append(logits.argmax(dim=1).cpu())
        emb = F.normalize(torch.cat([fused_m1, fused_m2], dim=1), dim=1)
        knn_preds.append(knn_classify(emb, ref_emb, ref_labels, n_classes, k=k).cpu())
        all_labels.append(y_batch)
    return (torch.cat(cls_preds).numpy(), torch.cat(knn_preds).numpy(), torch.cat(all_labels).numpy())


if __name__ == "__main__":
    batch_size = 16
    dataset_path = sys.argv[1]
    first_prefix = sys.argv[2]
    second_prefix = sys.argv[3]
    perc = sys.argv[4]
    run_id = sys.argv[5]
    checkpoint_path = sys.argv[6]
    freeze_encoder = "freeze" in sys.argv   # NOT the default recommendation here -- see conversation
    print(sys.argv)

    # ---- tunables ----
    SHARED_UNSHARED = 50
    LAMBDA_CLS = 1.0
    K_NEIGHBORS = 5
    BACKBONE_LR = 5e-6        # encoders + projectors -- pretrained, move slowly
    FRESH_LR = 5e-5           # classifier + alf_m1 + alf_m2 -- freshly initialized

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

    dir_name = dataset_path + "/RESUME_PRETRAIN_ALF"
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

    # depth probed from a throwaway encoder BEFORE building the real model,
    # just to compute layer_indices -- ViTEncoder's default depth=12
    _probe_encoder = ViTEncoder(img_size=config.img_size_m1, patch_size=config.patch_size_m1,
                                 in_chans=config.in_chans_m1)
    depth = len(_probe_encoder.transformer.layers)
    layer_indices = get_quarterly_layer_indices(depth)
    del _probe_encoder
    print("ViT depth=%d, using quarterly layer_indices (0-based)=%s" % (depth, layer_indices))

    model = PretrainModel(config, num_classes=n_classes, layer_indices=layer_indices).to(device)
    load_full_pretrained_checkpoint(model, checkpoint_path, device)

    if freeze_encoder:
        freeze_pretrained_backbone(model, freeze_projectors=False)
        print("Encoders FROZEN (projectors still trainable)")
    else:
        print("Encoders UNFROZEN -- continuing to update via loss_m1/loss_m2/loss_cross")

    backbone_params = [p for p in (list(model.modality_1_encoder.parameters())
                                    + list(model.modality_2_encoder.parameters())
                                    + list(model.projector_m1.parameters())
                                    + list(model.projector_m2.parameters())) if p.requires_grad]
    fresh_params = [p for p in (list(model.classifier.parameters())
                                 + list(model.alf_m1.parameters())
                                 + list(model.alf_m2.parameters())) if p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": BACKBONE_LR, "weight_decay": 1e-4},
        {"params": fresh_params, "lr": FRESH_LR, "weight_decay": 1e-4},
    ])
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
        loss_cls_sum = torch.zeros((), device=device)
        n_batches = 0

        for f_batch_unl, s_batch_unl in dataloader_unl_train:
            optimizer.zero_grad(set_to_none=True)
            f_batch_unl = f_batch_unl.to(device, non_blocking=True)
            s_batch_unl = s_batch_unl.to(device, non_blocking=True)

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

                # ---- labeled: multi-layer fusion classification, no pseudo-labeling ----
                logits_lab, _, _ = model.classify_alf(f_lab_b, s_lab_b)
                loss_cls = F.cross_entropy(logits_lab, y_lab_b)

                loss = 0.5 * (loss_m1 + loss_m2) + loss_cross + LAMBDA_CLS * loss_cls

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
            loss_cls_sum += loss_cls.detach()
            n_batches += 1

        if epoch >= WARM_UP_EPOCH_EMA:
            ema_weights = cumulate_EMA(model, ema_weights, MOMENTUM_EMA)

        if epoch % 5 == 0:
            if epoch >= WARM_UP_EPOCH_EMA:
                current_state_dict = copy.deepcopy(model.state_dict())
                model.load_state_dict(ema_weights)
                ref_emb = compute_reference_embedding(model, x_tensor_f_lab, x_tensor_s_lab, device)
                cls_preds, knn_preds, test_labels = evaluate(
                    model, ref_emb, y_tensor.to(device), dataloader_test, n_classes, device, k=K_NEIGHBORS)
                model.load_state_dict(current_state_dict)
            else:
                ref_emb = compute_reference_embedding(model, x_tensor_f_lab, x_tensor_s_lab, device)
                cls_preds, knn_preds, test_labels = evaluate(
                    model, ref_emb, y_tensor.to(device), dataloader_test, n_classes, device, k=K_NEIGHBORS)

            f1_cls = f1_score(test_labels, cls_preds, average="weighted")
            f1_knn = f1_score(test_labels, knn_preds, average="weighted")

            layer_weights_m1 = model.alf_m1.current_weights().cpu().numpy().round(3)
            layer_weights_m2 = model.alf_m2.current_weights().cpu().numpy().round(3)

            print(f"epoch {epoch} total={total_loss.item() / max(n_batches, 1):.4f} "
                  f"loss_m1={loss_m1_sum.item() / max(n_batches, 1):.4f} "
                  f"loss_m2={loss_m2_sum.item() / max(n_batches, 1):.4f} "
                  f"loss_cross={loss_cross_sum.item() / max(n_batches, 1):.4f} "
                  f"loss_cls={loss_cls_sum.item() / max(n_batches, 1):.4f} "
                  f"F1-classifier={(f1_cls * 100):.2f} F1-knn={(f1_knn * 100):.2f} "
                  f"layer_w_m1={layer_weights_m1} layer_w_m2={layer_weights_m2}")
            sys.stdout.flush()

    if ema_weights is not None:
        model.load_state_dict(ema_weights)
    torch.save(model.state_dict(), output_file)
    print("Saved to %s" % output_file)