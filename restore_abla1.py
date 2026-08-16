import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.checkpoint import checkpoint
from sklearn.metrics import f1_score

from model import SFFCConfig, ViTEncoder
from functions import get_quarterly_layer_indices

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ==========================================================================
# Restores OURS_ABLA1_NOSSL checkpoints (ablation: no continual SSL during
# fine-tuning -- training loss was `loss = loss_cls` only).
#
# Kept self-contained (encoder_forward_all_layers, LightweightLayerFusion,
# PretrainModelV5 all defined here) for the same reason the training
# script gives: avoids a circular import through model.py/functions.py.
#
# NOTE: two versions of the ablation-1 training script were provided --
# one defining PretrainModelV5 (with a `layer_dropout` field on
# LightweightLayerFusion) and an older one defining PretrainModelV4
# (without it). `layer_dropout` is a plain float, not an nn.Parameter, so
# it adds no keys to state_dict() and has no effect once model.eval() is
# called (the stochastic-depth branch is gated on self.training). This
# restore script is therefore compatible with checkpoints from EITHER
# version -- the class is named PretrainModelV5 below to match the more
# recent script, but functionally it restores the same weights either way.
#
# What MUST match training, or load_state_dict(strict=True) will fail:
#   - layer_indices (quarterly vs. all layers -- see --all_layers_combination)
#   - gating ("sigmoid" by default in both training scripts)
# What does NOT need to match (no effect at eval time):
#   - layer_dropout value
# ==========================================================================


def encoder_forward_all_layers(encoder: ViTEncoder, x: torch.Tensor, layer_indices: list,
                                use_checkpointing: bool = True) -> torch.Tensor:
    """Returns CLS tokens from the SPECIFIED layers, stacked as
    [B, len(layer_indices), embed_dim]. use_checkpointing is a no-op here
    since restore always runs under torch.no_grad()."""
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
        if use_checkpointing and torch.is_grad_enabled():
            x = checkpoint(layer, x, use_reentrant=False)
        else:
            x = layer(x)
        if i in layer_indices_set:
            collected[i] = encoder.norm(x)[:, 0, :]   # CLS token, normalized

    return torch.stack([collected[i] for i in layer_indices], dim=1)   # [B, L, D]


class LightweightLayerFusion(nn.Module):
    """A learned weight per layer combining per-layer CLS tokens.
    See training script for full docstring; only the forward-pass shape
    (which keys land in state_dict) matters for restore."""

    def __init__(self, num_layers: int, embed_dim: int, gating: str = "sigmoid",
                 post_norm: bool = None, layer_dropout: float = 0.5):
        super().__init__()
        assert gating in ("softmax", "sigmoid", "tanh")
        self.gating = gating
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.layer_dropout = layer_dropout   # inert at eval time -- kept for signature parity only

        if post_norm is None:
            post_norm = (gating != "softmax")
        self.norm = nn.LayerNorm(embed_dim) if post_norm else None

    def forward(self, layer_tokens: torch.Tensor) -> torch.Tensor:
        B, L, D = layer_tokens.shape

        if self.gating == "softmax":
            base_weights = F.softmax(self.layer_logits, dim=0)
        elif self.gating == "sigmoid":
            base_weights = torch.sigmoid(self.layer_logits)
        else:
            base_weights = torch.tanh(self.layer_logits)

        weights = base_weights.view(1, L, 1).expand(B, L, 1).clone()

        if self.training and self.layer_dropout > 0:
            keep_mask = (torch.rand(B, L, device=layer_tokens.device) >= self.layer_dropout).float()
            all_dropped = keep_mask.sum(dim=1) == 0
            if all_dropped.any():
                fallback_idx = torch.randint(0, L, (int(all_dropped.sum().item()),), device=layer_tokens.device)
                keep_mask[all_dropped, fallback_idx] = 1.0
            weights = weights * keep_mask.unsqueeze(-1)
            scale = L / keep_mask.sum(dim=1, keepdim=True).clamp(min=1e-8)
            weights = weights * scale.unsqueeze(-1)

        fused = (layer_tokens * weights).sum(dim=1)
        if self.norm is not None:
            fused = self.norm(fused)
        return fused


class PretrainModelV5(nn.Module):
    def __init__(self, config: SFFCConfig, num_classes: int, layer_indices: list,
                 embed_dim: int = 384, layer_dropout: float = 0.5, gating: str = "sigmoid"):
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

        self.layer_indices = layer_indices

        self.alf_m1 = LightweightLayerFusion(num_layers=len(layer_indices), embed_dim=embed_dim,
                                              gating=gating, layer_dropout=layer_dropout)
        self.alf_m2 = LightweightLayerFusion(num_layers=len(layer_indices), embed_dim=embed_dim,
                                              gating=gating, layer_dropout=layer_dropout)
        self.classifier = nn.Linear(embed_dim * 2, num_classes)

    def classify_alf(self, x1: torch.Tensor, x2: torch.Tensor, use_checkpointing: bool = True):
        layers_m1 = encoder_forward_all_layers(self.modality_1_encoder, x1, self.layer_indices,
                                                use_checkpointing=use_checkpointing)
        layers_m2 = encoder_forward_all_layers(self.modality_2_encoder, x2, self.layer_indices,
                                                use_checkpointing=use_checkpointing)
        fused_m1 = self.alf_m1(layers_m1)
        fused_m2 = self.alf_m2(layers_m2)
        concat = torch.cat([fused_m1, fused_m2], dim=1)
        return self.classifier(concat), fused_m1, fused_m2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Restore OURS_ABLA1_NOSSL (no continual SSL during fine-tuning) checkpoints "
                    "and evaluate weighted F1 on the test split."
    )
    parser.add_argument("dataset_path", type=str, help="e.g. EUROSAT")
    parser.add_argument("first_prefix", type=str, help="e.g. SAR")
    parser.add_argument("second_prefix", type=str, help="e.g. MS")
    parser.add_argument("perc", type=str, help="labeled percentage/count identifier, e.g. 5")

    parser.add_argument("--all_layers_combination", action="store_true", default=False,
                    help="use all ViT layers instead of the quarterly subset -- MUST match training")
    parser.add_argument("--gating", type=str, default="sigmoid", choices=["softmax", "sigmoid", "tanh"],
                    help="layer-fusion gating function -- MUST match training (default: sigmoid)")
    parser.add_argument("--layer-dropout", type=float, default=0.5,
                    help="constructor parity only -- has no effect at eval time")

    parser.add_argument("--output_dir", type=str, default="OURS_ABLA1_NOSSL",
                        help="output directory, default OURS_ABLA1_NOSSL")

    return parser.parse_args()


if __name__ == "__main__":
    batch_size = 16
    args = parse_args()

    dataset_path = args.dataset_path
    first_prefix = args.first_prefix
    second_prefix = args.second_prefix
    perc = args.perc
    output_dir = args.output_dir
    all_layer_combination = args.all_layers_combination
    n_splits = 5
    run_ids = range(n_splits)

    first_data = np.load("%s/%s_data_normalized.npy" % (dataset_path, first_prefix))
    second_data = np.load("%s/%s_data_normalized.npy" % (dataset_path, second_prefix))
    full_labels = np.load("%s/labels.npy" % dataset_path)
    train_idx = np.load("%s/train_idx.npy" % dataset_path)

    test_idx = np.setdiff1d(np.arange(full_labels.shape[0]), train_idx)
    f_data_test = first_data[test_idx]
    s_data_test = second_data[test_idx]
    labels_test = full_labels[test_idx]

    n_classes = len(np.unique(full_labels))

    x_tensor_f_test = torch.tensor(f_data_test, dtype=torch.float32)
    x_tensor_s_test = torch.tensor(s_data_test, dtype=torch.float32)
    y_tensor_test = torch.tensor(labels_test, dtype=torch.int64)
    test_dataset = TensorDataset(x_tensor_f_test, x_tensor_s_test, y_tensor_test)

    dataloader_test = DataLoader(
        test_dataset, shuffle=False, batch_size=batch_size * 8,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=False
    )

    print("TEST DATA built (%d samples)" % len(test_dataset))
    sys.stdout.flush()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = SFFCConfig(
        img_size_m1=f_data_test.shape[2], img_size_m2=s_data_test.shape[2],
        patch_size_m1=8, patch_size_m2=8,
        in_chans_m1=f_data_test.shape[1], in_chans_m2=s_data_test.shape[1],
        num_classes=n_classes, hidden_dim=256, dropout=0.1
    )

    _probe_encoder = ViTEncoder(img_size=config.img_size_m1, patch_size=config.patch_size_m1,
                                 in_chans=config.in_chans_m1)
    depth = len(_probe_encoder.transformer.layers)
    if all_layer_combination:
        layer_indices = np.arange(depth)
    else:
        layer_indices = get_quarterly_layer_indices(depth)
    del _probe_encoder

    print("ViT depth=%d, layer_indices (0-based)=%s" % (depth, layer_indices))

    dir_name = os.path.join(dataset_path, output_dir)
    f1_scores = []
    for run_id in run_ids:
        ckpt_path = os.path.join(dir_name, "%s_%s.pth" % (perc, run_id))

        if not os.path.exists(ckpt_path):
            print("WARNING: checkpoint not found, skipping: %s" % ckpt_path)
            continue

        print("Loading checkpoint: %s" % ckpt_path)
        sys.stdout.flush()

        model = PretrainModelV5(config, num_classes=n_classes, layer_indices=layer_indices,
                                 layer_dropout=args.layer_dropout, gating=args.gating).to(device)

        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)   # strict=True (default) -- will error loudly on any mismatch
        model.eval()

        all_preds, all_labels = [], []
        with torch.no_grad():
            for f_batch, s_batch, y_batch in dataloader_test:
                f_batch = f_batch.to(device, non_blocking=True)
                s_batch = s_batch.to(device, non_blocking=True)
                logits, _, _ = model.classify_alf(f_batch, s_batch)
                preds = logits.argmax(dim=1).cpu()
                all_preds.append(preds)
                all_labels.append(y_batch)

        predictions = torch.cat(all_preds).numpy()
        test_labels = torch.cat(all_labels).numpy()

        f1_val = f1_score(test_labels, predictions, average="weighted")
        f1_scores.append(f1_val)

        print("Split %s -> F1 = %.4f" % (str(run_id), f1_val))
        sys.stdout.flush()

        del model
        torch.cuda.empty_cache()

    f1_scores = np.array(f1_scores)
    print("\n===== Summary over %d splits =====" % len(f1_scores))
    print("%.2f $\pm$ %.2f" % (f1_scores.mean() * 100, f1_scores.std() * 100))