# =============================================================================
# msc_main_v5.py -- copy of msc_main_v4.py (left unchanged). v5 (agreed 2026-11-23): the two ViT encoders are
# initialised with the per-modality MAE encoders <dataset>/MAE/<MOD>_encoder.pth (mae_pretrain.py) ONLY -- no
# supervised stage (that stage belongs to the Semi-ViT / Semi-SST pipelines, not to MSC, which was introduced with
# pretrained encoders); MSC heads random; same differential AdamW lr as the other pretrained-start runs (encoders
# 5e-6, heads 1e-4, wd 1e-4). Rest identical to v4/v3. Checkpoints: <dataset>/MSC_V5/<perc>_<run>.pth
# =============================================================================
# msc_main_v4.py -- copy of msc_main_v3.py (left unchanged). v4 (agreed 2026-11-22): the two ViT encoders come
# from the ImageNet-initialised supervised SF checkpoint <dataset>/SF_IN/<perc>_<run>.pth (baselines_main_v4.py;
# MSC was introduced with pretrained encoders), MSC heads random; differential AdamW lr (encoders 5e-6, heads 1e-4,
# wd 1e-4) as the other ImageNet-start runs. Rest identical to v3. Checkpoints: <dataset>/MSC_V4/<perc>_<run>.pth
# =============================================================================
# msc_main_v3.py -- MSC v2 started from the supervised baseline (copy of msc_main_v2.py, left unchanged).
# v3 changes (agreed 2026-11-20): the two ViT encoders are initialised from the supervised SF checkpoint
# <dataset>/SF/<perc>_<run>.pth (no pretraining, labelled data only; MSC heads stay random);
# no supervised warm-up (unlabelled loss from epoch 0); EMA seeded with the initial weights and
# updated from epoch 0. Everything else unchanged. Checkpoints: <dataset>/MSC_V3/<perc>_<run>.pth
#
# msc_main_v2.py -- MSC (FixMatch+MSC variant), v2 of msc_main.py (v1 left unchanged).
# Only change: AdamW lr 1e-4 (v1: 5e-5, the paper's fine-tuning value), the same as
# Supervised / FixMatch / FreeMatch / SoftMatch trained from scratch.
# Checkpoints: <dataset>/MSC_V2/<perc>_<run>.pth   (evaluate with msc_restore_v2.py)
# Usage: python msc_main_v2.py EUROSAT_STD SAR MS 5 0
# =============================================================================
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
from functions import evaluation, weak_augment_pair, strong_augment_pair, MOMENTUM_EMA, cumulate_EMA, WARM_UP_EPOCH_EMA, EPOCHS, WARM_UP_EPOCH_SSL, RATIO_LABELED_UNLABELED_BATCHES
import random
import bitsandbytes as bnb
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint
import os
import warnings
from msc import MSCConfig, MSCModel, MSCLoss, weighted_nll_loss
from sup_init import load_supervised_init
from mae_init import load_mae_encoders
import time
from sklearn.metrics import f1_score
from torch.cuda.amp import autocast, GradScaler
import gc
import copy

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True




# ==========================================================================
# 9. Example: plugging in arbitrary backbones + a runnable dummy training loop
# ==========================================================================

if __name__ == "__main__":
    batch_size = 16
    dataset_path = sys.argv[1]
    first_prefix = sys.argv[2]
    second_prefix = sys.argv[3]
    perc = sys.argv[4]
    run_id = sys.argv[5]
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

    dir_name = dataset_path + "/MSC_V5"
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
    dataloader_test = DataLoader(test_dataset, shuffle=False, batch_size=batch_size*6, 
        num_workers=6,           # parallel CPU data loading
        pin_memory=True,         # faster CPU→GPU transfer
        persistent_workers=True, # keeps workers alive between epochs
        prefetch_factor=4,        # prefetch batches ahead of time
        drop_last=False
        #drop_last=True
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

    config = MSCConfig(
        img_size_m1 = f_lab_data_train.shape[2], img_size_m2 = s_lab_data_train.shape[2], 
        patch_size_m1 = 8, patch_size_m2 = 8, 
        in_chans_m1 = f_lab_data_train.shape[1], in_chans_m2 = s_lab_data_train.shape[1], 
        num_classes=n_classes, hidden_dim=256, dropout=0.1, tau=0.95, eta=0.95, lambda_u=1.0,
        beta1=1.0, beta2=1.0, beta3=1.0, detach_teacher=True,
    )

    model = MSCModel(config).to(device)
    load_mae_encoders(model, dataset_path, first_prefix, second_prefix, device)   # v5: raw MAE encoders (no supervised stage)
    # v3 fix: the MLPHead heads use nn.LazyLinear (no weights until the first forward); materialise them
    # with one no-grad forward on a labelled batch, before the optimizer is built and the EMA is seeded.
    with torch.no_grad():
        model.eval()
        _f, _s, _y = next(iter(dataloader_lab_train))
        model(_f.to(device), _s.to(device))
    assert not any(isinstance(p, nn.parameter.UninitializedParameter) for p in model.parameters())
    #model.compile()
    loss_fn = MSCLoss(config)
    loss_ce = nn.CrossEntropyLoss() #MSCLoss(config)
    special_params, normal_params = [], []   # v4: same differential lr as the other ImageNet-start runs
    for name, param in model.named_parameters():
        (special_params if ("modality_1_encoder" in name or "modality_2_encoder" in name) else normal_params).append(param)
    optimizer = torch.optim.AdamW([{"params": normal_params, "lr": 1e-4}, {"params": special_params, "lr": 5e-6}], weight_decay=1e-4)
    scaler = GradScaler()

    print("model created and compiled")
    sys.stdout.flush()

    #for epoch in range(start_epoch, max_num_epochs):
    step = 0
    ema_weights = cumulate_EMA(model, None, MOMENTUM_EMA)   # v3: EMA seeded with the initial weights
    for epoch in range(EPOCHS):
        use_ssl = True   # v3: no warm-up (encoders already trained on the labels)
        total_loss = torch.zeros((), device=device) 
        start_time = time.time()
        n_batches = 0
        for (f_batch, s_batch, y_batch), (f_batch_unl, s_batch_unl) in zip(
            itertools.cycle(dataloader_lab_train), dataloader_unl_train):           

            model.train()
            optimizer.zero_grad(set_to_none=True)
            f_batch = f_batch.to(device, non_blocking=True)
            s_batch = s_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            if use_ssl:
                f_batch_unl = f_batch_unl.to(device, non_blocking=True)
                s_batch_unl = s_batch_unl.to(device, non_blocking=True)
                with autocast():
                    unl_f_weak, unl_s_weak = weak_augment_pair(f_batch_unl, s_batch_unl)
                    unl_f_strong, unl_s_strong = strong_augment_pair(f_batch_unl, s_batch_unl)

                    out = loss_fn(
                        model,
                        f_batch, s_batch, y_batch,
                        unl_f_weak, unl_s_weak,
                        unl_f_strong, unl_s_strong, epoch, 0   # v3: SSL terms active from epoch 0
                    )
                scaler.scale(out["loss"]).backward()
                scaler.step(optimizer)
                scaler.update()
                #total_loss += loss_s1.item()
                total_loss += out["loss"].detach()
            else:
                with autocast():
                    out = model(f_batch, s_batch)
                    loss = (weighted_nll_loss(out["p_S"], y_batch, config.eps) + weighted_nll_loss(out["p_C"], y_batch, config.eps))                
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.detach()
            n_batches += 1
        
        
        end = time.time()
        elapsed_time = time.time() - start_time

        if True:   # v3: EMA from epoch 0
            ema_weights = cumulate_EMA(model, ema_weights, MOMENTUM_EMA)


        if epoch % 5 == 0:
            if True:   # v3: EMA from epoch 0
                current_state_dict = copy.deepcopy(model.state_dict())
                model.load_state_dict(ema_weights)
                predictions, test_labels = evaluation(model, dataloader_test, device)
                model.load_state_dict(current_state_dict)
            else:
                predictions, test_labels = evaluation(model, dataloader_test, device)

            f1_val = f1_score(test_labels, predictions, average="weighted")
            total_loss = total_loss.item()  
            print(f"epoch {epoch} "
                f"total={np.mean(total_loss / max(n_batches, 1)):.4f} "
                f"elapsed_time={elapsed_time:.2f} "
                f"F1-score={(f1_val * 100):.2f}")

        sys.stdout.flush()
    
    model.load_state_dict(ema_weights)
    torch.save(model.state_dict(), output_file)