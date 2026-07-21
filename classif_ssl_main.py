"""
Classification-head SSL: CE supervised loss + weak/strong consistency +
mean-entropy-maximization (me-max), NO confidence thresholding.

This is the "simple" alternative to proto_ssl_main.py's non-parametric
(k-NN / support-set) approach: back to a standard trained classification
head (ScoreFusion or FusionConcat), but keeping the one mechanism today's
evidence identified as necessary for stability under label scarcity --
me-max, which directly penalizes the batch-averaged unlabeled prediction
for collapsing onto a subset of classes, independent of and in addition to
per-sample confidence.

Explicitly NOT included: any hard or soft confidence-based masking/
weighting (no FixMatch-style threshold, no SoftMatch-style continuous
weight). This isolates whether me-max ALONE is sufficient, since:
  - plain classification-head + consistency + confidence weighting, NO
    me-max, IS what SoftMatch already is -- and SoftMatch-frozen-FC
    declined steadily from ~83 to ~61 with no recovery (see today's runs).
  - proto_ssl_main.py (k-NN/support-set + consistency + me-max, frozen
    encoder) reached ~85 F1 with a flat, stable trajectory from SSL onset.
This script tests the missing cell: classification head + consistency +
me-max, no thresholding -- to see whether me-max alone (without the
non-parametric machinery) reproduces that stability.

Consistency loss defaults to L2 (Brier score) on softmax probabilities,
matching what worked best in proto_ssl_main.py's own ablation.
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

from model import SFFCConfig, ScoreFusion, FusionConcat

from functions import (evaluation, weak_augment_pair, strong_augment_pair, MOMENTUM_EMA, cumulate_EMA,
                        WARM_UP_EPOCH_EMA, EPOCHS, WARM_UP_EPOCH_SSL, RATIO_LABELED_UNLABELED_BATCHES,
                        load_pretrained_encoders)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ==========================================================================
# Unlabeled loss: consistency (L2 / Brier) + mean-entropy-maximization
# ==========================================================================

def sharpen(p: torch.Tensor, T: float = 0.5) -> torch.Tensor:
    p_sharp = p ** (1.0 / T)
    return p_sharp / p_sharp.sum(dim=1, keepdim=True)


def consistency_loss_l2(probs_weak: torch.Tensor, probs_strong: torch.Tensor,
                         sharpen_T: float = 0.5) -> torch.Tensor:
    """Squared L2 (Brier score) between sharpened weak-view target
    (no-grad) and strong-view prediction. Bounded, unlike cross-entropy --
    a badly-wrong prediction can't produce an unbounded gradient spike."""
    with torch.no_grad():
        target = sharpen(probs_weak, T=sharpen_T)
    return ((target - probs_strong) ** 2).sum(dim=1).mean()


def mean_entropy_max_loss(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Penalizes the BATCH-AVERAGED prediction for deviating from uniform
    across classes -- targets class-collapse directly, independent of
    per-sample confidence. Already sign-flipped: minimizing this loss
    MAXIMIZES the entropy of the batch-mean prediction."""
    mean_p = probs.mean(dim=0)
    entropy = -(mean_p * torch.log(mean_p.clamp(min=eps))).sum()
    return -entropy


# ==========================================================================
# Pretrained-encoder freezing (load_pretrained_encoders already in functions.py)
# ==========================================================================

def freeze_pretrained_encoders(model):
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
    sf_or_fc = sys.argv[6]  # SF = score fusion / FC = Feature Concat
    pretrained_path = sys.argv[7] if len(sys.argv) > 7 else None
    freeze_encoder = "freeze" in sys.argv
    print(sys.argv)

    # ---- tunables ----
    SHARPEN_T = 1.0#0.5      # weak-view target sharpening temperature
    LAMBDA_U = 1.0        # weight of consistency term
    LAMBDA_ME = .3#1.0        # weight of me-max term

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

    dir_name = dataset_path + "/HEAD_MEMAX_%s" % sf_or_fc
    os.makedirs(dir_name, exist_ok=True)
    output_file = dir_name + "/%s_%s.pth" % (perc, run_id)

    # ---------------- TEST DATA ----------------
    test_idx = np.setdiff1d(np.arange(full_labels.shape[0]), train_idx)
    f_data_test = first_data[test_idx]
    s_data_test = second_data[test_idx]
    labels_test = full_labels[test_idx]

    x_tensor_f_test = torch.tensor(f_data_test, dtype=torch.float32)
    x_tensor_s_test = torch.tensor(s_data_test, dtype=torch.float32)
    y_tensor_test = torch.tensor(labels_test, dtype=torch.int64)
    test_dataset = TensorDataset(x_tensor_f_test, x_tensor_s_test, y_tensor_test)
    dataloader_test = DataLoader(test_dataset, shuffle=False, batch_size=batch_size * RATIO_LABELED_UNLABELED_BATCHES,
        num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=4, drop_last=False)
    print("TEST DATA built")
    sys.stdout.flush()

    # ---------------- LABELED / UNLABELED DATA ----------------
    x_tensor_f_lab = torch.tensor(f_lab_data_train, dtype=torch.float32)
    x_tensor_s_lab = torch.tensor(s_lab_data_train, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.int64)
    lab_dataset = TensorDataset(x_tensor_f_lab, x_tensor_s_lab, y_tensor)

    x_tensor_f_unl = torch.tensor(f_unlab_data_train, dtype=torch.float32)
    x_tensor_s_unl = torch.tensor(s_unlab_data_train, dtype=torch.float32)
    unl_dataset = TensorDataset(x_tensor_f_unl, x_tensor_s_unl)

    dataloader_lab_train = DataLoader(lab_dataset, shuffle=True, batch_size=batch_size,
        num_workers=0, pin_memory=True, drop_last=False)

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

    if sf_or_fc == "SF":
        model = ScoreFusion(config).to(device)
        loss_fn = nn.NLLLoss()
    elif sf_or_fc == "FC":
        model = FusionConcat(config).to(device)
        loss_fn = nn.CrossEntropyLoss()
    else:
        print("NO METHOD DEFINED")
        exit(0)

    if pretrained_path is not None:
        load_pretrained_encoders(model, pretrained_path, device)
        if freeze_encoder:
            freeze_pretrained_encoders(model)
            print("Pretrained encoders FROZEN")

    model.compile()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
    scaler = GradScaler('cuda')
    print("model created and compiled")
    sys.stdout.flush()

    ema_weights = None
    for epoch in range(EPOCHS):
        start_time = __import__("time").time()
        total_loss = torch.zeros((), device=device)
        use_ssl = epoch >= WARM_UP_EPOCH_SSL
        n_batches = 0
        majority_frac_sum = 0.0
        majority_frac_batches = 0

        for (f_batch, s_batch, y_batch), (f_batch_unl, s_batch_unl) in zip(
                itertools.cycle(dataloader_lab_train), dataloader_unl_train):

            model.train()
            optimizer.zero_grad(set_to_none=True)
            f_batch = f_batch.to(device, non_blocking=True)
            s_batch = s_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            f_batch_unl = f_batch_unl.to(device, non_blocking=True)
            s_batch_unl = s_batch_unl.to(device, non_blocking=True)

            with autocast('cuda'):
                pred_lab = model(f_batch, s_batch)
                if sf_or_fc == "SF":
                    sup_loss = loss_fn(torch.log(pred_lab.clamp(min=1e-8)), y_batch)
                else:
                    sup_loss = loss_fn(pred_lab, y_batch)

                if use_ssl:
                    f_weak, s_weak = weak_augment_pair(f_batch_unl, s_batch_unl)
                    f_strong, s_strong = strong_augment_pair(f_batch_unl, s_batch_unl)

                    with torch.no_grad():
                        pred_weak = model(f_weak, s_weak)
                        probs_weak = F.softmax(pred_weak, dim=-1) if sf_or_fc == "FC" else pred_weak

                    pred_strong = model(f_strong, s_strong)
                    probs_strong = F.softmax(pred_strong, dim=-1) if sf_or_fc == "FC" else pred_strong

                    # NO thresholding/masking of any kind -- every sample
                    # contributes to both terms below.
                    loss_consistency = consistency_loss_l2(probs_weak, probs_strong, sharpen_T=SHARPEN_T)
                    loss_me = mean_entropy_max_loss(probs_strong)

                    loss = sup_loss + LAMBDA_U * loss_consistency + LAMBDA_ME * loss_me
                else:
                    loss = sup_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.detach()
            n_batches += 1

            if use_ssl:
                with torch.no_grad():
                    hard_preds = probs_strong.argmax(dim=1)
                    majority_frac = (hard_preds.bincount(minlength=n_classes).max() / hard_preds.numel()).item()
                    majority_frac_sum += majority_frac
                    majority_frac_batches += 1

        elapsed_time = __import__("time").time() - start_time

        if epoch >= WARM_UP_EPOCH_EMA:
            ema_weights = cumulate_EMA(model, ema_weights, MOMENTUM_EMA)

        if epoch % 5 == 0:
            if epoch >= WARM_UP_EPOCH_EMA:
                current_state_dict = copy.deepcopy(model.state_dict())
                model.load_state_dict(ema_weights)
                predictions, test_labels = evaluation(model, dataloader_test, device)
                model.load_state_dict(current_state_dict)
            else:
                predictions, test_labels = evaluation(model, dataloader_test, device)

            f1_val = f1_score(test_labels, predictions, average="weighted")
            avg_majority_frac = majority_frac_sum / max(majority_frac_batches, 1)
            print(f"epoch {epoch} total={total_loss.item() / max(n_batches, 1):.4f} "
                  f"elapsed_time={elapsed_time:.2f} F1-score={(f1_val * 100):.2f} "
                  f"majority_frac={avg_majority_frac:.3f} ssl={'on' if use_ssl else 'off (warmup)'}")
            sys.stdout.flush()

    if ema_weights is not None:
        model.load_state_dict(ema_weights)
    torch.save(model.state_dict(), output_file)
    print("Saved to %s" % output_file)