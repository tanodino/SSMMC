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
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint
import os
import warnings
#from msc import MSCConfig, MSCModel, MSCLoss
from model import SFFCConfig, ScoreFusion, FusionConcat
import time
from sklearn.metrics import f1_score
import copy

# FixMatch (Sohn et al., 2020) uses a single FIXED confidence threshold to
# hard-mask pseudo-labels, unlike SoftMatch's EMA-tracked continuous
# weighting -- no stateful class needed, just this one constant.
FIXMATCH_THRESHOLD = 0.95

if __name__ == "__main__":
    batch_size = 16
    dataset_path = sys.argv[1]
    first_prefix = sys.argv[2]
    second_prefix = sys.argv[3]
    perc = sys.argv[4]
    run_id = sys.argv[5]
    sf_or_fc = sys.argv[6] # SF = score fusion / FC = Feature Concat
    
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

    dir_name = dataset_path+"/FIXMATCH_%s"%sf_or_fc
    os.makedirs(dir_name, exist_ok=True)
    output_file = dir_name+"%s_%s.pth"%(perc, run_id)

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

    dataloader_unl_train = DataLoader(unl_dataset, shuffle=True, batch_size=batch_size*8, 
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
    elif sf_or_fc == "FC":
        model = FusionConcat(config).to(device)
    else:
        print("NO METHOD DEFINED")
        exit(0)
    
    model.compile()
    loss_fn = nn.CrossEntropyLoss() #MSCLoss(config)
    loss_fn_none = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scaler = GradScaler()
    print("model created and compiled")
    sys.stdout.flush()

    lambda_u = 1.0  # weight of the unsupervised loss term

    #for epoch in range(start_epoch, max_num_epochs):
    step = 0
    ema_weights = None
    for epoch in range(EPOCHS):
        #for f_batch, s_batch, y_batch in dataloader_lab_train:
        start_time = time.time()
        total_loss = torch.zeros((), device=device) 
        use_ssl = epoch > WARM_UP_EPOCH_SSL
        n_batches = 0
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
                sup_loss = loss_fn(pred_lab, y_batch)
                
                if use_ssl:
                    f_weak, s_weak = weak_augment_pair(f_batch_unl, s_batch_unl)
                    f_strong, s_strong = strong_augment_pair(f_batch_unl, s_batch_unl)

                    with torch.no_grad():
                        pred_weak = model(f_weak, s_weak)
                        probs_weak = F.softmax(pred_weak, dim=-1)
                        max_probs, pseudo_labels = probs_weak.max(dim=-1)

                    # FixMatch: fixed threshold, hard 0/1 mask (Sohn et al., 2020, Eq. 6)
                    # -- replaces SoftMatch's EMA-tracked continuous sample_weights.
                    mask = (max_probs >= FIXMATCH_THRESHOLD).float()

                    pred_strong = model(f_strong, s_strong)
                    unsup_loss_per_sample = loss_fn_none(pred_strong, pseudo_labels)
                    unsup_loss = (unsup_loss_per_sample * mask).mean()                   
                    loss = sup_loss + lambda_u * unsup_loss
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
            current_state_dict = copy.deepcopy(model.state_dict())
            model.load_state_dict(ema_weights)
            predictions, test_labels = evaluation(model, dataloader_test, device)    
            model.load_state_dict(current_state_dict)
        else:
            predictions, test_labels = evaluation(model, dataloader_test, device)

        f1_val = f1_score(test_labels, predictions, average="weighted")
        total_loss = total_loss.item()  
        if epoch % 5 == 0:
            print(f"epoch {epoch} "
                f"total={np.mean(total_loss / max(n_batches, 1)):.4f} "
                f"elapsed_time={elapsed_time:.2f} "
                f"F1-score={(f1_val * 100):.2f}")
            sys.stdout.flush()
    
    model.load_state_dict(ema_weights)
    torch.save(model.state_dict(), output_file)