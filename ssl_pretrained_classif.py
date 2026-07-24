"""
Resume full pretrained model (encoders + projectors), continue the ORIGINAL
contrastive SSL objective, add a directly-supervised classification head.

--------------------------------------------------------------------------
Why this design, instead of proto_ssl_main.py's approach
--------------------------------------------------------------------------
Everything in proto_ssl_main.py (h/z split, InfoNCE, support-set resampling,
BatchNorm-vs-LayerNorm, classifier-overfitting fixes) exists to make a NEW,
never-before-validated combination of losses behave well. That machinery
never clearly beat the very first, much simpler result: a frozen pretrained
encoder + a plain CE-trained classification head, which was immediately
stable at F1~80 with no tuning at all.

This script does the more conservative thing instead: resume the FULL
pretrained model (encoders AND both projectors -- the encoders-only save
file, pretrained_backbones.pth, does NOT contain the projectors; this
script needs the full periodic checkpoint, e.g. checkpoint_latest.pth,
which does) and CONTINUE the exact contrastive objective that already
produced a good representation, while adding a classification head trained
with PLAIN, DIRECT cross-entropy on the labeled set.

Nothing here ever generates a pseudo-label. The contrastive terms
(loss_m1, loss_m2, loss_cross) never touch a label. The classifier's CE
loss is direct supervision, never used to produce a target for anything
else. This sidesteps every confirmation-bias failure mode diagnosed today
(FixMatch, SoftMatch, MSC's collapse), because none of those failures were
possible in a design with no pseudo-labeling step at all.

--------------------------------------------------------------------------
Loss terms (identical to pretrain.py's original three, see that script for
full derivation):
  loss_m1, loss_m2 : NTXentLoss instance-discrimination on each modality's
                      OWN projector output (raw vs. strongly-augmented view)
  loss_cross        : NTXentLoss on the shared/invariant subspace (first
                       shared_unshared% of each modality's CLS token,
                       concatenated) -- operates DIRECTLY on encoder output,
                       not through a projector, so its gradient reaches the
                       encoders directly whenever they're not frozen
  loss_cls           : plain cross-entropy, classifier(concat(cls_m1, cls_m2))
                       vs. true labels -- NEW in this script, the only
                       labeled-data term

Usage:
    python resume_pretrain_supervised_main.py EUROSAT SAR MS 5 0 \\
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
# Model: same as pretrain.py's PretrainModel, PLUS a classification head
# ==========================================================================

class PretrainModel(nn.Module):
    """Identical structure to pretrain.py's PretrainModel (two encoders,
    two per-modality projectors) so the full checkpoint's state_dict loads
    cleanly, PLUS a new self.classifier not present in that checkpoint --
    load with strict=False and expect ONLY classifier.* in missing_keys."""

    def __init__(self, config: SFFCConfig, num_classes: int, embed_dim: int = 384,
                 classifier_hidden_dim: int = 256, classifier_dropout: float = 0.3):
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

        # NEW -- not present in the pretraining checkpoint. Small MLP on
        # concatenated CLS tokens (2 hidden layers), rather than a single
        # linear layer. LayerNorm (not BatchNorm): this classifier trains
        # on the SAME fixed 50-sample labeled batch every step (no
        # augmentation), so unlike proto_ssl_main's fusion/ssl_head (which
        # saw a heterogeneous mix of labeled + varying augmented unlabeled
        # data -- the actual driver of that script's BatchNorm drift),
        # BatchNorm's running stats here would likely converge to something
        # stable. Still defaulting to LayerNorm since 50 samples is a small
        # population to trust for running statistics, and it removes the
        # question entirely at no real cost -- swap to nn.BatchNorm1d if
        # you want to test that instead.
        #
        # Dropout is deliberately present and non-trivial (0.3 default):
        # a deeper head has MORE capacity to memorize a fixed 50-sample
        # batch than the single-linear version did, and that version
        # already showed real overfitting (F1-classifier trailing
        # F1-knn/F1-linprobe by several points in proto_ssl_main.py).
        # Making the head more expressive without adding regularization
        # would likely make that gap worse, not better.
        '''
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, classifier_hidden_dim),
            nn.LayerNorm(classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden_dim, classifier_hidden_dim // 2),
            nn.LayerNorm(classifier_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden_dim // 2, num_classes),
        )
        '''

        self.classifier = nn.Linear(embed_dim * 2, num_classes)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        cls_token_m1 = self.modality_1_encoder(x1)   # [B, embed_dim]
        cls_token_m2 = self.modality_2_encoder(x2)
        proj_m1 = self.projector_m1(cls_token_m1)
        proj_m2 = self.projector_m2(cls_token_m2)
        return cls_token_m1, cls_token_m2, proj_m1, proj_m2

    def classify(self, cls_token_m1: torch.Tensor, cls_token_m2: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([cls_token_m1, cls_token_m2], dim=1)
        return self.classifier(concat)                  # raw logits


def load_full_pretrained_checkpoint(model: PretrainModel, path: str, device: str):
    """Loads the FULL periodic checkpoint (encoders + both projectors) --
    NOT the encoders-only save file. strict=False because self.classifier
    is new; asserts that classifier.* is the ONLY thing missing, as a
    safety check against a real key mismatch being silently swallowed."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    result = model.load_state_dict(state_dict, strict=False)

    unexpected = list(result.unexpected_keys)
    missing_non_classifier = [k for k in result.missing_keys if not k.startswith("classifier.")]

    if unexpected:
        print("WARNING: unexpected keys in checkpoint (not used): %s" % unexpected)
    if missing_non_classifier:
        raise RuntimeError(
            "Checkpoint is missing non-classifier keys -- this is NOT just "
            "'classifier is new', something else doesn't match: %s" % missing_non_classifier
        )
    print("Loaded full pretrained model (encoders + projectors) from %s" % path)
    print("  (classifier head is new, randomly initialized -- expected)")


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
# Evaluation: classifier readout + k-NN on concatenated CLS-token space
# ==========================================================================

@torch.no_grad()
def compute_reference_cls_tokens(model: PretrainModel, f_lab: torch.Tensor, s_lab: torch.Tensor, device):
    model.eval()
    cls_m1, cls_m2, _, _ = model(f_lab.to(device), s_lab.to(device))
    return F.normalize(torch.cat([cls_m1, cls_m2], dim=1), dim=1)


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
    knn_preds, cls_preds, all_labels = [], [], []
    for f_batch, s_batch, y_batch in dataloader:
        f_batch = f_batch.to(device, non_blocking=True)
        s_batch = s_batch.to(device, non_blocking=True)
        cls_m1, cls_m2, _, _ = model(f_batch, s_batch)
        emb = F.normalize(torch.cat([cls_m1, cls_m2], dim=1), dim=1)
        knn_preds.append(knn_classify(emb, ref_emb, ref_labels, n_classes, k=k).cpu())
        logits = model.classify(cls_m1, cls_m2)
        cls_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(y_batch)
    return (torch.cat(knn_preds).numpy(), torch.cat(cls_preds).numpy(), torch.cat(all_labels).numpy())


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
    SHARED_UNSHARED = 50      # invariant/specific split, matches pretrain.py's default -- set
                               # to whatever value your actual pretraining run used, if different
    LAMBDA_CLS = 1.0          # weight of the classifier CE loss
    K_NEIGHBORS = 5
    BACKBONE_LR = 5e-6        # encoders + projectors -- ALL pretrained, move slowly so
                               # continued SSL can refine them without risking the kind
                               # of drift that hurt every unfrozen-pretrained-encoder
                               # experiment earlier today (e.g. pretrained+unfrozen
                               # FixMatch collapsing worse than from-scratch FixMatch)
    #CLASSIFIER_LR = 5e-5      # freshly initialized, needs to actually learn from scratch
    CLASSIFIER_LR = 1e-4      # freshly initialized, needs to actually learn from scratch

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

    dir_name = dataset_path + "/RESUME_PRETRAIN_SUPERVISED"
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

    model = PretrainModel(config, num_classes=n_classes).to(device)
    load_full_pretrained_checkpoint(model, checkpoint_path, device)

    if freeze_encoder:
        freeze_pretrained_backbone(model, freeze_projectors=False)
        print("Encoders FROZEN (projectors still trainable)")

    optimizer = torch.optim.AdamW([
        {"params": [p for p in list(model.modality_1_encoder.parameters())
                    + list(model.modality_2_encoder.parameters())
                    + list(model.projector_m1.parameters())
                    + list(model.projector_m2.parameters()) if p.requires_grad],
         "lr": BACKBONE_LR, "weight_decay": 1e-4},
        {"params": [p for p in model.classifier.parameters() if p.requires_grad],
         "lr": CLASSIFIER_LR, "weight_decay": 1e-4},
    ])
    # NOTE: encoders AND projectors are all PRETRAINED (loaded from checkpoint),
    # only self.classifier is freshly initialized. Grouping "encoder vs
    # everything else" would be wrong -- projectors need the same gentle
    # treatment as the encoders, not the classifier's faster rate. With
    # freeze_encoder=True, the encoder params simply have requires_grad=False
    # and drop out of the backbone group automatically; the projectors still
    # train (at BACKBONE_LR) since they're never frozen by default.
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

                # ---- labeled: NEW, direct supervision, no pseudo-labeling ----
                cls_token_m1_lab, cls_token_m2_lab, _, _ = model(f_lab_b, s_lab_b)
                logits_lab = model.classify(cls_token_m1_lab, cls_token_m2_lab)
                loss_cls = F.cross_entropy(logits_lab, y_lab_b)

                loss = 0.5 * (loss_m1 + loss_m2) + loss_cross + LAMBDA_CLS * loss_cls

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
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
                ref_emb = compute_reference_cls_tokens(model, x_tensor_f_lab, x_tensor_s_lab, device)
                knn_preds, cls_preds, test_labels = evaluate(
                    model, ref_emb, y_tensor.to(device), dataloader_test, n_classes, device, k=K_NEIGHBORS)
                model.load_state_dict(current_state_dict)
            else:
                ref_emb = compute_reference_cls_tokens(model, x_tensor_f_lab, x_tensor_s_lab, device)
                knn_preds, cls_preds, test_labels = evaluate(
                    model, ref_emb, y_tensor.to(device), dataloader_test, n_classes, device, k=K_NEIGHBORS)

            f1_knn = f1_score(test_labels, knn_preds, average="weighted")
            f1_cls = f1_score(test_labels, cls_preds, average="weighted")

            print(f"epoch {epoch} total={total_loss.item() / max(n_batches, 1):.4f} "
                  f"loss_m1={loss_m1_sum.item() / max(n_batches, 1):.4f} "
                  f"loss_m2={loss_m2_sum.item() / max(n_batches, 1):.4f} "
                  f"loss_cross={loss_cross_sum.item() / max(n_batches, 1):.4f} "
                  f"loss_cls={loss_cls_sum.item() / max(n_batches, 1):.4f} "
                  f"F1-knn={(f1_knn * 100):.2f} F1-classifier={(f1_cls * 100):.2f}")
            sys.stdout.flush()

    if ema_weights is not None:
        model.load_state_dict(ema_weights)
    torch.save(model.state_dict(), output_file)
    print("Saved to %s" % output_file)