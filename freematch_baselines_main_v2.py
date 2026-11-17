"""
freematch_baselines_main_v2.py -- FreeMatch baseline, corrected version of freematch_baselines_main.py.

Same data, model, protocol, optimiser, EMA and augmentations as v1. Only the FreeMatch
components are changed to follow the official USB implementation
(microsoft/Semi-supervised-learning, semilearn/algorithms/freematch/{freematch.py,utils.py}):

  * Self-adaptive thresholding (SAT): global tau_t (EMA of mean max-prob), local p_t
    (EMA of mean class probabilities) AND the pseudo-label histogram h_t
    (EMA of the normalised bincount of argmax), all initialised to 1/C, momentum 0.999,
    updated before masking. Mask: max_p >= tau_t * p_t[c] / max(p_t).
  * Self-adaptive fairness (SAF): computed only on the strong-view predictions of the
    samples accepted by the mask (skipped when none is accepted):
        target = SumNorm(p_t / h_t)            (no gradient)
        pred   = SumNorm(mean(q_s) / hist(q_s))
        L_f    = sum(target * log(pred))       (= -H(target, pred), as in USB entropy_loss)
    with weight lambda_f = 0.01 (USB ent_loss_ratio default).

v1 used a fairness target that was always uniform (p_t / local_tau is constant), no label
histogram, all strong samples and a KL with the opposite sign.

Checkpoints: <dataset>/FREEMATCH_V2_<SF|FC>/<perc>_<run>.pth (v1 results are not overwritten).
Usage (same arguments as v1):
    python freematch_baselines_main_v2.py EUROSAT_STD SAR MS 5 0 FC
"""
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
import numpy as np
from model import ViTEncoder
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingWarmRestarts
import time
import sys
from torch.utils.data import TensorDataset, DataLoader
from functions import evaluation, weak_augment_pair, strong_augment_pair, MOMENTUM_EMA, cumulate_EMA, WARM_UP_EPOCH_EMA, EPOCHS, WARM_UP_EPOCH_SSL, RATIO_LABELED_UNLABELED_BATCHES, load_pretrained_encoders, freeze_pretrained_encoders
import random
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint
import os
import warnings
#from msc import MSCConfig, MSCModel, MSCLoss
from model import SFFCConfig, ScoreFusion, FusionConcat
import time
from sklearn.metrics import f1_score
import copy
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def replace_inf_to_zero(val):
    val[val == float("inf")] = 0.0
    return val


class FreeMatchSAT:
    """Self-adaptive thresholding of FreeMatch (USB FreeMatchThresingHook, use_quantile=False,
    clip_thresh=False). Keeps tau_t (global), p_t (local) and h_t (label histogram)."""

    def __init__(self, num_classes, momentum=0.999, device="cuda"):
        self.m = momentum
        self.num_classes = num_classes
        self.p_t = torch.ones(num_classes, device=device) / num_classes
        self.h_t = torch.ones(num_classes, device=device) / num_classes
        self.tau_t = self.p_t.mean()

    @torch.no_grad()
    def update(self, probs_weak):
        probs_weak = probs_weak.float()
        max_probs, max_idx = probs_weak.max(dim=-1)
        self.tau_t = self.tau_t * self.m + (1 - self.m) * max_probs.mean()
        self.p_t = self.p_t * self.m + (1 - self.m) * probs_weak.mean(dim=0)
        hist = torch.bincount(max_idx, minlength=self.num_classes).to(self.p_t.dtype)
        self.h_t = self.h_t * self.m + (1 - self.m) * (hist / hist.sum())

    @torch.no_grad()
    def mask(self, probs_weak):
        """update the statistics, then return the hard 0/1 mask (as USB masking())."""
        self.update(probs_weak)
        max_probs, max_idx = probs_weak.float().max(dim=-1)
        mod = self.p_t / self.p_t.max()
        return max_probs.ge(self.tau_t * mod[max_idx]).float()


def freematch_fairness_loss(mask, probs_strong, p_t, h_t):
    """USB entropy_loss with probabilities as input (SF outputs probabilities; for FC pass
    softmax(logits)). Only the samples accepted by the mask are used. Returns sum(target*log(pred))."""
    probs_s = probs_strong[mask.bool()].float()
    num_classes = probs_s.shape[1]
    hist_s = torch.bincount(probs_s.argmax(dim=-1), minlength=num_classes).to(probs_s.dtype)
    hist_s = hist_s / hist_s.sum()

    target = p_t.reshape(1, -1) * replace_inf_to_zero(1 / h_t.reshape(1, -1)).detach()
    target = target / target.sum(dim=-1, keepdim=True)

    pred = probs_s.mean(dim=0, keepdim=True) * replace_inf_to_zero(1 / hist_s).detach()
    pred = pred / pred.sum(dim=-1, keepdim=True)

    return (target * torch.log(pred + 1e-12)).sum(dim=1).mean()


if __name__ == "__main__":
    batch_size = 16
    dataset_path = sys.argv[1]
    first_prefix = sys.argv[2]
    second_prefix = sys.argv[3]
    perc = sys.argv[4]
    run_id = sys.argv[5]
    sf_or_fc = sys.argv[6] # SF = score fusion / FC = Feature Concat
    pretrained_path = sys.argv[7] if len(sys.argv) > 7 else None   # <-- new, optional
    print(sys.argv)
    
    first_data = np.load("%s/%s_data_normalized.npy"%(dataset_path, first_prefix) )
    second_data = np.load("%s/%s_data_normalized.npy"%(dataset_path, second_prefix) )
    full_labels = np.load("%s/labels.npy"%dataset_path)
    train_idx = np.load("%s/train_idx.npy"%dataset_path)
    labelled_idx = np.load("%s/labelled_samples_%s_%s.npy"%(dataset_path, perc, run_id))
    
    full_train_idx = np.arange(len(train_idx))
    unlabelled_idx = np.setdiff1d(full_train_idx, labelled_idx)
    
    f_lab_data_train = first_data[train_idx][labelled_idx]
    s_lab_data_train = second_data[train_idx][labelled_idx]
    f_unlab_data_train = first_data[train_idx][unlabelled_idx]
    s_unlab_data_train = second_data[train_idx][unlabelled_idx]

    labels = full_labels[train_idx][labelled_idx]

    print("f_lab_data_train %d"%len(f_lab_data_train))
    print("f_unlab_data_train %d"%len(f_unlab_data_train))

    n_classes = len(np.unique(labels))

    dir_name = dataset_path+"/FREEMATCH_V2_%s"%sf_or_fc
    os.makedirs(dir_name, exist_ok=True)
    output_file = dir_name+"/%s_%s.pth"%(perc, run_id)

    #batch_size = pretraining_batch_size
    print("batch_size %d"%batch_size)

    ########## TEST DATA ##########
    test_idx = np.setdiff1d(np.arange(full_labels.shape[0]), train_idx)
    f_data_test = first_data[test_idx]
    s_data_test = second_data[test_idx]
    labels_test = full_labels[test_idx]

    x_tensor_f_test = torch.tensor(f_data_test, dtype=torch.float32)
    x_tensor_s_test = torch.tensor(s_data_test, dtype=torch.float32)
    y_tensor_test = torch.tensor(labels_test, dtype=torch.int64)
    test_dataset = TensorDataset(x_tensor_f_test, x_tensor_s_test, y_tensor_test)
    dataloader_test = DataLoader(test_dataset, shuffle=False, batch_size=batch_size*RATIO_LABELED_UNLABELED_BATCHES, 
        num_workers=6,           # parallel CPU data loading
        pin_memory=True,         # faster CPU→GPU transfer
        persistent_workers=True, # keeps workers alive between epochs
        prefetch_factor=4,        # prefetch batches ahead of time
        drop_last=False
    )
    ########## TEST DATA ##########
    print("TEST DATA built")
    sys.stdout.flush()

    x_tensor_f_lab = torch.tensor(f_lab_data_train, dtype=torch.float32)
    x_tensor_s_lab = torch.tensor(s_lab_data_train, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.int64)
    lab_dataset = TensorDataset(x_tensor_f_lab, x_tensor_s_lab, y_tensor)
    
    x_tensor_f_unl = torch.tensor(f_unlab_data_train, dtype=torch.float32)
    x_tensor_s_unl = torch.tensor(s_unlab_data_train, dtype=torch.float32)
    unl_dataset = TensorDataset(x_tensor_f_unl, x_tensor_s_unl)

    dataloader_lab_train = DataLoader(lab_dataset, shuffle=True, batch_size=batch_size, 
        num_workers=0,           # parallel CPU data loading
        pin_memory=True,         # faster CPU→GPU transfer
        drop_last=False
    )

    dataloader_unl_train = DataLoader(unl_dataset, shuffle=True, batch_size=batch_size*RATIO_LABELED_UNLABELED_BATCHES, 
        num_workers=6,           # parallel CPU data loading
        pin_memory=True,         # faster CPU→GPU transfer
        persistent_workers=True, # keeps workers alive between epochs
        prefetch_factor=4,        # prefetch batches ahead of time
        drop_last=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("all dataloader built")
    sys.stdout.flush()

    config = SFFCConfig(
        img_size_m1 = f_lab_data_train.shape[2], img_size_m2 = s_lab_data_train.shape[2], 
        patch_size_m1 = 8, patch_size_m2 = 8, 
        in_chans_m1 = f_lab_data_train.shape[1], in_chans_m2 = s_lab_data_train.shape[1], 
        num_classes=n_classes, hidden_dim=256, dropout=0.1
    )

    if sf_or_fc == "SF":
        model = ScoreFusion(config).to(device)
        loss_fn = nn.NLLLoss()
    elif sf_or_fc == "FC":
        model = FusionConcat(config).to(device)
        loss_fn = nn.CrossEntropyLoss() #MSCLoss(config)
    else:
        print("NO METHOD DEFINED")
        exit(0)
    
    if pretrained_path is not None:                                    # <-- new
        load_pretrained_encoders(model, pretrained_path, device)       # <-- new
        freeze_pretrained_encoders(model)
        
    model.compile()
    loss_fn_none = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()
    print("model created and compiled")
    sys.stdout.flush()
    
    freematch = FreeMatchSAT(num_classes=n_classes, momentum=0.999, device=device)
    lambda_u = 1.0
    lambda_f = 0.01  # USB ent_loss_ratio default

    #for epoch in range(start_epoch, max_num_epochs):
    step = 0
    ema_weights = None
    for epoch in range(EPOCHS):
        start_time = time.time()
        total_loss = torch.zeros((), device=device)
        n_batches = 0
        use_ssl = epoch >= WARM_UP_EPOCH_SSL
        mask_rate_sum = 0.0
        mask_batches = 0
        majority_frac_sum = 0.0
        majority_frac_batches = 0

        tau_t_sum = 0.0
        p_t_max_sum = 0.0
        p_t_min_sum = 0.0
        h_t_max_sum = 0.0
        fair_sum = 0.0
        state_batches = 0

        for (f_batch, s_batch, y_batch), (f_batch_unl, s_batch_unl) in zip(
            itertools.cycle(dataloader_lab_train), dataloader_unl_train):           

            model.train()
            optimizer.zero_grad(set_to_none=True)
            f_batch = f_batch.to(device, non_blocking=True)
            s_batch = s_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            
            f_batch_unl = f_batch_unl.to(device, non_blocking=True)
            s_batch_unl = s_batch_unl.to(device, non_blocking=True)
            with autocast():
                pred_lab = model(f_batch, s_batch)
                if sf_or_fc == "SF":
                    sup_loss = loss_fn(torch.log(pred_lab.clamp(min=1e-8)), y_batch)
                else:
                    sup_loss = loss_fn(pred_lab, y_batch)

                if use_ssl:
                    f_weak, s_weak = weak_augment_pair(f_batch_unl, s_batch_unl)
                    f_strong, s_strong = strong_augment_pair(f_batch_unl, s_batch_unl)

                    with torch.no_grad():
                        model.eval()
                        pred_weak = model(f_weak, s_weak)
                        if sf_or_fc == "FC":
                            probs_weak = F.softmax(pred_weak, dim=-1)
                        else:
                            probs_weak = pred_weak  # SF already outputs a distribution
                        max_probs, pseudo_labels = probs_weak.max(dim=-1)
                        model.train()

                    sample_mask = freematch.mask(probs_weak)   # updates tau_t, p_t, h_t, then hard 0/1 mask [B]

                    mask_rate_sum += sample_mask.float().mean().item()
                    mask_batches += 1

                    if sample_mask.sum() > 0:
                        accepted_labels = pseudo_labels[sample_mask.bool()]
                        majority_frac = (accepted_labels.bincount(minlength=n_classes).max() / accepted_labels.numel()).item()
                        majority_frac_sum += majority_frac
                        majority_frac_batches += 1

                    tau_t_sum += freematch.tau_t.item()
                    p_t_max_sum += freematch.p_t.max().item()
                    p_t_min_sum += freematch.p_t.min().item()
                    h_t_max_sum += freematch.h_t.max().item()
                    state_batches += 1


                    pred_strong = model(f_strong, s_strong)
                    if sf_or_fc == "SF":
                        log_probs_strong = torch.log(pred_strong.clamp(min=1e-8))
                        unsup_loss_per_sample = F.nll_loss(log_probs_strong, pseudo_labels, reduction="none")
                        probs_strong = pred_strong  # already probabilities, still differentiable
                    else:
                        unsup_loss_per_sample = loss_fn_none(pred_strong, pseudo_labels)
                        probs_strong = F.softmax(pred_strong, dim=-1)

                    unsup_loss = (unsup_loss_per_sample * sample_mask).mean()

                    if sample_mask.sum() > 0:
                        fairness_loss = freematch_fairness_loss(sample_mask, probs_strong, freematch.p_t, freematch.h_t)
                    else:
                        fairness_loss = torch.zeros((), device=device)
                    fair_sum += fairness_loss.item()

                    loss = sup_loss + lambda_u * unsup_loss + lambda_f * fairness_loss
                else:
                    loss = sup_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.detach()
            n_batches += 1

        end = time.time()
        elapsed_time = time.time() - start_time

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
            total_loss = total_loss.item()

            avg_mask_rate = mask_rate_sum / max(mask_batches, 1)
            avg_majority_frac = majority_frac_sum / max(majority_frac_batches, 1)
            avg_tau_t = tau_t_sum / max(state_batches, 1)
            avg_p_t_max = p_t_max_sum / max(state_batches, 1)
            avg_p_t_min = p_t_min_sum / max(state_batches, 1)
            avg_h_t_max = h_t_max_sum / max(state_batches, 1)
            avg_fair = fair_sum / max(state_batches, 1)
            print(f"epoch {epoch} "
                f"total={np.mean(total_loss / max(n_batches, 1)):.4f} "
                f"elapsed_time={elapsed_time:.2f} "
                f"F1-score={(f1_val * 100):.2f} "
                f"mask_rate={avg_mask_rate:.3f} "
                f"majority_frac={avg_majority_frac:.3f} "
                f"tau_t={avg_tau_t:.3f} "
                f"p_t_max={avg_p_t_max:.3f} "
                f"p_t_min={avg_p_t_min:.3f} "
                f"h_t_max={avg_h_t_max:.3f} "
                f"fairness={avg_fair:.4f}")
            sys.stdout.flush()

    model.load_state_dict(ema_weights)
    torch.save(model.state_dict(), output_file)