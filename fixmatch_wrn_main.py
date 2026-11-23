"""
fixmatch_wrn_main.py -- FixMatch (Sohn et al., 2020) with its ORIGINAL backbone and recipe, from scratch
(new script; the ViT FixMatch scripts are left unchanged). Agreed with the user 2026-11-21:
  * backbone: one WRN-28-2 per modality (wrn.py), score fusion = mean of the two softmax outputs
  * from scratch (no pretraining, no supervised init, no warm-up), as in the paper
  * optimiser and lr: the SAME as all the other approaches (user, 2026-11-21): AdamW, lr 1e-4 (constant),
    torch default weight decay -- replaces the original SGD 0.03 + cosine recipe (first launch stopped at epoch 0)
  * threshold 0.95, lambda_u 1, unlabelled ratio mu = 7; EMA of the weights (0.999 per step) for evaluation
  * single forward of [labelled weak, unlabelled weak, unlabelled strong] in train mode (BN over the whole batch),
    pseudo-labels from the (detached) weak predictions, as in the reference implementation
  * batches: 16 labelled + 112 unlabelled (user's choice; original 64 + 448 does not fit our labelled sets / memory)
  * labelled images get the weak augmentation, as in the paper (user's choice)
  * training length: EPOCHS (400) passes over the unlabelled set, as for our other methods
  * augmentations: our weak_augment_pair / strong_augment_pair (same as the other competitors)
Checkpoint (EMA weights): <dataset>/FIXMATCH_WRN_SF/<perc>_<run>.pth
    python fixmatch_wrn_main.py SUNRGBD_STD RGB DEPTH 5 0
"""
import itertools, math, os, sys, time, copy, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score
from functions import evaluation, weak_augment_pair, strong_augment_pair, EPOCHS
from wrn import WRNScoreFusion
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

THRESHOLD = 0.95
LAMBDA_U = 1.0
MU = 7
BATCH_LAB = 16
LR = 1e-4
EMA_DECAY = 0.999


class ModelEMA:
    """EMA of the parameters (decay per step); buffers (BN statistics) copied, as in the reference code."""

    def __init__(self, model, decay):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.param_names = set(n for n, _ in self.ema.named_parameters())

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if k in self.param_names:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(msd[k])


if __name__ == "__main__":
    dataset_path, first_prefix, second_prefix, perc, run_id = sys.argv[1:6]
    print(sys.argv)
    first_data = np.load("%s/%s_data_normalized.npy" % (dataset_path, first_prefix))
    second_data = np.load("%s/%s_data_normalized.npy" % (dataset_path, second_prefix))
    full_labels = np.load("%s/labels.npy" % dataset_path)
    train_idx = np.load("%s/train_idx.npy" % dataset_path)
    labelled_idx = np.load("%s/labelled_samples_%s_%s.npy" % (dataset_path, perc, run_id))
    unlabelled_idx = np.setdiff1d(np.arange(len(train_idx)), labelled_idx)

    f_lab, s_lab = first_data[train_idx][labelled_idx], second_data[train_idx][labelled_idx]
    f_unl, s_unl = first_data[train_idx][unlabelled_idx], second_data[train_idx][unlabelled_idx]
    labels = full_labels[train_idx][labelled_idx]
    n_classes = len(np.unique(labels))
    print("labelled %d unlabelled %d classes %d" % (len(f_lab), len(f_unl), n_classes))

    dir_name = dataset_path + "/FIXMATCH_WRN_SF"
    os.makedirs(dir_name, exist_ok=True)
    output_file = dir_name + "/%s_%s.pth" % (perc, run_id)

    test_idx = np.setdiff1d(np.arange(full_labels.shape[0]), train_idx)
    t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt)
    dataloader_test = DataLoader(TensorDataset(t(first_data[test_idx]), t(second_data[test_idx]),
                                               t(full_labels[test_idx], torch.int64)),
                                 shuffle=False, batch_size=128, num_workers=6, pin_memory=True,
                                 persistent_workers=True, prefetch_factor=4)
    dataloader_lab = DataLoader(TensorDataset(t(f_lab), t(s_lab), t(labels, torch.int64)), shuffle=True,
                                batch_size=BATCH_LAB, num_workers=0, pin_memory=True, drop_last=False)
    dataloader_unl = DataLoader(TensorDataset(t(f_unl), t(s_unl)), shuffle=True, batch_size=BATCH_LAB * MU,
                                num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=4,
                                drop_last=True)
    device = "cuda"

    model = WRNScoreFusion(n_classes, f_lab.shape[1], s_lab.shape[1]).to(device)
    print("parameters: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)   # same as the other approaches
    steps_per_epoch = len(dataloader_unl)
    total_steps = EPOCHS * steps_per_epoch
    ema = ModelEMA(model, EMA_DECAY)
    scaler = GradScaler()
    print("steps/epoch %d, total steps %d" % (steps_per_epoch, total_steps))
    sys.stdout.flush()

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        total_loss = torch.zeros((), device=device)
        n_batches, mask_sum, maj_sum, maj_n = 0, 0.0, 0.0, 0
        for (f_b, s_b, y_b), (f_u, s_u) in zip(itertools.cycle(dataloader_lab), dataloader_unl):
            f_b, s_b, y_b = f_b.to(device, non_blocking=True), s_b.to(device, non_blocking=True), y_b.to(device, non_blocking=True)
            f_u, s_u = f_u.to(device, non_blocking=True), s_u.to(device, non_blocking=True)
            with torch.no_grad():
                f_lw, s_lw = weak_augment_pair(f_b, s_b)          # labelled: weak aug (as in the paper)
                f_w, s_w = weak_augment_pair(f_u, s_u)
                f_s, s_s = strong_augment_pair(f_u, s_u)
            nl, nu = f_b.shape[0], f_u.shape[0]
            with autocast():
                probs = model(torch.cat([f_lw, f_w, f_s]), torch.cat([s_lw, s_w, s_s])).float()
                p_lab, p_w, p_s = probs[:nl], probs[nl:nl + nu], probs[nl + nu:]
                sup_loss = F.nll_loss(torch.log(p_lab.clamp(min=1e-8)), y_b)
                max_probs, pseudo = p_w.detach().max(dim=-1)
                mask = (max_probs >= THRESHOLD).float()
                unsup_loss = (F.nll_loss(torch.log(p_s.clamp(min=1e-8)), pseudo, reduction="none") * mask).mean()
                loss = sup_loss + LAMBDA_U * unsup_loss
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            total_loss += loss.detach()
            n_batches += 1
            mask_sum += mask.mean().item()
            if mask.sum() > 0:
                acc = pseudo[mask.bool()]
                maj_sum += (acc.bincount(minlength=n_classes).max() / acc.numel()).item()
                maj_n += 1
        elapsed_time = time.time() - start_time
        if epoch % 5 == 0:
            predictions, test_labels = evaluation(ema.ema, dataloader_test, device)
            f1_val = f1_score(test_labels, predictions, average="weighted")
            print(f"epoch {epoch} total={total_loss.item() / max(n_batches, 1):.4f} "
                  f"elapsed_time={elapsed_time:.2f} F1-score={f1_val * 100:.2f} "
                  f"mask_rate={mask_sum / max(n_batches, 1):.3f} majority_frac={maj_sum / max(maj_n, 1):.3f} "
                  )
            sys.stdout.flush()

    torch.save(ema.ema.state_dict(), output_file)
