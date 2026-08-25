#ABLATION WITHOUT multi-level Layer Fusion

import sys
import os
import copy
import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
from sklearn.metrics import f1_score

from model import SFFCConfig, ViTEncoder
from functions import strong_augment_pair, NTXentLoss, MOMENTUM_EMA, cumulate_EMA, WARM_UP_EPOCH_EMA, EPOCHS, RATIO_LABELED_UNLABELED_BATCHES

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ==========================================================================
# Model -- NO ALF, last-layer CLS token only
# ==========================================================================

class PretrainModelV4_NoALF(nn.Module):
    """Same encoders/projectors as ssl_pretrained_classif_v4.py.
    classify_last_layer() replaces classify_alf() -- uses the final-layer
    CLS token from each encoder directly, no multi-layer fusion."""

    def __init__(self, config: SFFCConfig, num_classes: int, embed_dim: int = 384):
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

        # Classifier on top of concatenated last-layer CLS tokens
        # embed_dim * 2 : one CLS per modality
        self.classifier = nn.Linear(embed_dim * 2, num_classes)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """UNCHANGED -- used by the unlabeled contrastive losses."""
        cls_token_m1 = self.modality_1_encoder(x1)
        cls_token_m2 = self.modality_2_encoder(x2)
        proj_m1 = self.projector_m1(cls_token_m1)
        proj_m2 = self.projector_m2(cls_token_m2)
        return cls_token_m1, cls_token_m2, proj_m1, proj_m2

    def classify_last_layer(self, x1: torch.Tensor, x2: torch.Tensor):
        """Ablation classification path: last-layer CLS token only.
        No multi-layer fusion -- direct concatenation of final CLS tokens."""
        cls_m1 = self.modality_1_encoder(x1)   # [B, D] -- last-layer CLS
        cls_m2 = self.modality_2_encoder(x2)   # [B, D]
        concat = torch.cat([cls_m1, cls_m2], dim=1)   # [B, 2D]
        return self.classifier(concat), cls_m1, cls_m2


# ==========================================================================
# Gradient checkpointing wrapper (ported from ssl_pretrained_classif_v5's
# ablation-1 script) -- trades recomputation for activation memory on the
# plain forward() / classify_last_layer() path. Needed here because,
# unlike ablation 3 (loss = loss_cls only), this ablation backpropagates
# through ALL THREE forward passes per step (original unlabeled, strongly
# augmented, and labeled) -- there's no graph that can be dropped for
# free, so memory has to be cut structurally instead.
# ==========================================================================

def wrap_gradient_checkpointing(transformer_encoder):
    original_layers = transformer_encoder.layers

    class CheckpointedTransformerEncoder(nn.Module):
        def __init__(self, layers, norm):
            super().__init__()
            self.layers = layers
            self.norm = norm

        def forward(self, src, mask=None, src_key_padding_mask=None):
            x = src
            for layer in self.layers:
                if torch.is_grad_enabled():
                    x = checkpoint(layer, x, mask, src_key_padding_mask, use_reentrant=False)
                else:
                    x = layer(x, mask, src_key_padding_mask)
            if self.norm is not None:
                x = self.norm(x)
            return x

    return CheckpointedTransformerEncoder(original_layers, transformer_encoder.norm)


# ==========================================================================
# Checkpoint loading / freezing (identical to ssl_pretrained_classif_v4.py)
# ==========================================================================

def load_full_pretrained_checkpoint(model: PretrainModelV4_NoALF, path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    result = model.load_state_dict(state_dict, strict=False)

    unexpected = list(result.unexpected_keys)
    missing_non_new = [k for k in result.missing_keys
                       if not k.startswith("classifier.")]

    if unexpected:
        print("WARNING: unexpected keys in checkpoint (not used): %s" % unexpected)
    if missing_non_new:
        raise RuntimeError(
            "Checkpoint is missing non-classifier keys -- something "
            "else doesn't match: %s" % missing_non_new
        )
    print("Loaded pretrained model (encoders + projectors) from %s" % path)
    print("  (classifier is new, randomly initialized -- expected)")


def freeze_pretrained_backbone(model: PretrainModelV4_NoALF, freeze_projectors: bool = False):
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
# Evaluation (identical to ssl_pretrained_classif_v4.py)
# ==========================================================================

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
def evaluate(model: PretrainModelV4_NoALF, ref_emb: torch.Tensor, ref_labels: torch.Tensor,
             dataloader, n_classes: int, device, k: int = 5):
    model.eval()
    cls_preds, knn_preds, all_labels = [], [], []
    for f_batch, s_batch, y_batch in dataloader:
        f_batch = f_batch.to(device, non_blocking=True)
        s_batch = s_batch.to(device, non_blocking=True)
        logits, cls_m1, cls_m2 = model.classify_last_layer(f_batch, s_batch)
        cls_preds.append(logits.argmax(dim=1).cpu())
        emb = F.normalize(torch.cat([cls_m1, cls_m2], dim=1), dim=1)
        knn_preds.append(knn_classify(emb, ref_emb, ref_labels, n_classes, k=k).cpu())
        all_labels.append(y_batch)
    return (torch.cat(cls_preds).numpy(), torch.cat(knn_preds).numpy(), torch.cat(all_labels).numpy())


@torch.no_grad()
def compute_reference_embedding(model: PretrainModelV4_NoALF, f_lab: torch.Tensor,
                                 s_lab: torch.Tensor, device):
    model.eval()
    _, cls_m1, cls_m2 = model.classify_last_layer(f_lab.to(device), s_lab.to(device))
    return F.normalize(torch.cat([cls_m1, cls_m2], dim=1), dim=1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resume full pretrained model, continue original SSL objective, "
                    "classify via layer-dropout-regularized multi-layer fusion."
    )
    # ---- required positional arguments (same order as the old sys.argv[1:6]) ----
    parser.add_argument("dataset_path", type=str, help="e.g. SUNRGBD")
    parser.add_argument("first_prefix", type=str, help="e.g. RGB")
    parser.add_argument("second_prefix", type=str, help="e.g. DEPTH")
    parser.add_argument("perc", type=str, help="labeled percentage/count identifier, e.g. 5")
    parser.add_argument("run_id", type=str, help="split id, e.g. 0")
    parser.add_argument("checkpoint_path", type=str, help="path to the full pretrained checkpoint")

    # ---- optional flags ----
    parser.add_argument("--freeze", action="store_true",
                        help="freeze the encoders (projectors still trainable)")
    parser.add_argument("--no_crossmodal_ssl", action="store_false", default=True,
                    help="deactivate the crossmodal ssl loss")
    parser.add_argument("--grad-checkpointing", action=argparse.BooleanOptionalAction, default=True,
                        help="gradient checkpointing on both encoders (default: on); "
                             "pass --no-grad-checkpointing to disable")

    # ---- optional tunables (previously hardcoded constants) ----
    parser.add_argument("--output_dir", type=str, default="OURS_ABLA2_NoMLA",
                        help="output directory, default OURS_ABLA2_NoMLA")
    parser.add_argument("--shared-unshared", type=int, default=50,
                        help="invariant/specific split %% for loss_cross (default: 50)")
    parser.add_argument("--lambda-cls", type=float, default=1.0,
                        help="weight of the classifier CE loss (default: 1.0)")
    parser.add_argument("--k-neighbors", type=int, default=5,
                        help="k for the k-NN evaluation metric (default: 5)")
    parser.add_argument("--backbone-lr", type=float, default=5e-6,
                        help="LR for encoders + projectors (default: 5e-6)")
    parser.add_argument("--fresh-lr", type=float, default=5e-5,
                        help="LR for classifier + alf_m1 + alf_m2 (default: 5e-5)")
    parser.add_argument("--gating", type=str, default="sigmoid",
                        choices=["softmax", "sigmoid", "tanh"],
                        help="layer-fusion gating function (default: sigmoid)")
    parser.add_argument("--layer-dropout", type=float, default=0.5,
                        help="per-layer dropout probability in the fusion module (default: 0.2)")

    return parser.parse_args()


# ==========================================================================
# Main
# ==========================================================================

if __name__ == "__main__":
    batch_size = 16
    args = parse_args()
    print(vars(args))

    dataset_path = args.dataset_path
    first_prefix = args.first_prefix
    second_prefix = args.second_prefix
    perc = args.perc
    run_id = args.run_id
    checkpoint_path = args.checkpoint_path
    freeze_encoder = args.freeze
    output_dir = args.output_dir
    crossmodal_ssl = args.no_crossmodal_ssl

    '''
    dataset_path = sys.argv[1]
    first_prefix = sys.argv[2]
    second_prefix = sys.argv[3]
    perc = sys.argv[4]
    run_id = sys.argv[5]
    checkpoint_path = sys.argv[6]
    freeze_encoder = "freeze" if "freeze" in sys.argv else None
    print(sys.argv)
    '''

    # ---- tunables (identical to ssl_pretrained_classif_v4.py) ----
    SHARED_UNSHARED = 50
    LAMBDA_CLS = 1.0
    K_NEIGHBORS = 5
    BACKBONE_LR = 5e-6
    FRESH_LR = 1e-4

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

    # ✅ distinct output dir to avoid overwriting the MLA results
    dir_name = dataset_path + "/"+output_dir  #Multi-Layer Aggregation
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
    dataloader_test = DataLoader(test_dataset, shuffle=False,
        batch_size=batch_size * RATIO_LABELED_UNLABELED_BATCHES,
        num_workers=6, pin_memory=True, persistent_workers=True,
        prefetch_factor=4, drop_last=False)
    print("TEST DATA built")
    sys.stdout.flush()

    # ---------------- LABELED DATA ----------------
    x_tensor_f_lab = torch.tensor(f_lab_data_train, dtype=torch.float32)
    x_tensor_s_lab = torch.tensor(s_lab_data_train, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.int64)
    lab_dataset = TensorDataset(x_tensor_f_lab, x_tensor_s_lab, y_tensor)
    dataloader_lab_train = DataLoader(lab_dataset, shuffle=True,
        batch_size=len(lab_dataset), num_workers=0, pin_memory=True, drop_last=False)

    # ---------------- UNLABELED DATA ----------------
    x_tensor_f_unl = torch.tensor(f_unlab_data_train, dtype=torch.float32)
    x_tensor_s_unl = torch.tensor(s_unlab_data_train, dtype=torch.float32)
    unl_dataset = TensorDataset(x_tensor_f_unl, x_tensor_s_unl)
    dataloader_unl_train = DataLoader(unl_dataset, shuffle=True,
        batch_size=batch_size * RATIO_LABELED_UNLABELED_BATCHES,
        num_workers=6, pin_memory=True, persistent_workers=True,
        prefetch_factor=4, drop_last=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("all dataloaders built")
    sys.stdout.flush()

    config = SFFCConfig(
        img_size_m1=f_lab_data_train.shape[2], img_size_m2=s_lab_data_train.shape[2],
        patch_size_m1=8, patch_size_m2=8,
        in_chans_m1=f_lab_data_train.shape[1], in_chans_m2=s_lab_data_train.shape[1],
        num_classes=n_classes, hidden_dim=256, dropout=0.1
    )

    model = PretrainModelV4_NoALF(config, num_classes=n_classes).to(device)
    load_full_pretrained_checkpoint(model, checkpoint_path, device)

    # ---- NEW: gradient checkpointing on both encoders (fixes the CUDA OOM
    # in classify_last_layer's forward -- see module docstring above) ----
    model.modality_1_encoder.transformer = wrap_gradient_checkpointing(model.modality_1_encoder.transformer)
    model.modality_2_encoder.transformer = wrap_gradient_checkpointing(model.modality_2_encoder.transformer)
    print("Gradient checkpointing active on both encoders")

    if freeze_encoder:
        freeze_pretrained_backbone(model, freeze_projectors=False)
        print("Encoders FROZEN")
    else:
        print("Encoders UNFROZEN")

    backbone_params = [p for p in (list(model.modality_1_encoder.parameters())
                                   + list(model.modality_2_encoder.parameters())
                                   + list(model.projector_m1.parameters())
                                   + list(model.projector_m2.parameters()))
                       if p.requires_grad]
    fresh_params = [p for p in model.classifier.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": BACKBONE_LR, "weight_decay": 1e-4},
        {"params": fresh_params,    "lr": FRESH_LR,    "weight_decay": 1e-4},
    ])
    scaler = GradScaler('cuda')
    print("model created")
    sys.stdout.flush()

    ema_weights = None
    for epoch in range(EPOCHS):
        model.train()
        total_loss      = torch.zeros((), device=device)
        loss_m1_sum     = torch.zeros((), device=device)
        loss_m2_sum     = torch.zeros((), device=device)
        loss_cross_sum  = torch.zeros((), device=device)
        loss_cls_sum    = torch.zeros((), device=device)
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
                # ---- unlabeled: contrastive objective (unchanged) ----
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

                loss_m1    = NTXentLoss(repr_m1, labels_cls_loss, temperature=1.0)
                loss_m2    = NTXentLoss(repr_m2, labels_cls_loss, temperature=1.0)
                loss_cross = NTXentLoss(emb_inv, labels_cls_loss, temperature=1.0)

                # ---- labeled: last-layer CLS classification (no ALF) ----
                logits_lab, _, _ = model.classify_last_layer(f_lab_b, s_lab_b)
                loss_cls = F.cross_entropy(logits_lab, y_lab_b)

                loss = 0.5 * (loss_m1 + loss_m2) + LAMBDA_CLS * loss_cls
                if crossmodal_ssl:
                    loss = loss + loss_cross

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss     += loss.detach()
            loss_m1_sum    += loss_m1.detach()
            loss_m2_sum    += loss_m2.detach()
            loss_cross_sum += loss_cross.detach()
            loss_cls_sum   += loss_cls.detach()
            n_batches += 1

        if epoch >= WARM_UP_EPOCH_EMA:
            ema_weights = cumulate_EMA(model, ema_weights, MOMENTUM_EMA)

        if epoch % 5 == 0:
            if epoch >= WARM_UP_EPOCH_EMA and ema_weights is not None:
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

            print(f"epoch {epoch} total={total_loss.item() / max(n_batches, 1):.4f} "
                  f"loss_m1={loss_m1_sum.item() / max(n_batches, 1):.4f} "
                  f"loss_m2={loss_m2_sum.item() / max(n_batches, 1):.4f} "
                  f"loss_cross={loss_cross_sum.item() / max(n_batches, 1):.4f} "
                  f"loss_cls={loss_cls_sum.item() / max(n_batches, 1):.4f} "
                  f"F1-classifier={(f1_cls * 100):.2f} F1-knn={(f1_knn * 100):.2f}")
            sys.stdout.flush()

    if ema_weights is not None:
        model.load_state_dict(ema_weights)
    torch.save(model.state_dict(), output_file)
    print("Saved to %s" % output_file)