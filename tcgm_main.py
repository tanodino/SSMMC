"""
tcgm_main.py -- TCGM competitor (Sun et al., ECCV 2020), adapted to the common baseline protocol.

Agreed adaptation (2026-11-17):
  * unified protocol, the same as the other baselines: each step = CE on the labelled batch
    (one CE per modality classifier, summed) + L_TC on the unlabelled batch, weight 1
    (official code alpha = 1). AdamW lr 1e-4, per-epoch EMA, EPOCHS, batch 16 labelled x
    RATIO unlabelled. The 10 supervised warm-up epochs (WARM_UP_EPOCH_SSL) replace the
    code's "train accuracy > 0.9" gate. The paper's alternating CE/TC epochs with separate
    learning rates are not used;
  * prior p = empirical class frequencies of the labelled set, used in L_TC and in the
    aggregator (paper Sec. 4.3);
  * no augmentation (as in the paper and code), on labelled or unlabelled data;
  * prediction = native aggregator zeta ~ h1*h2/p (no SF/FC variants).

Usage: python tcgm_main.py <DATASET> <MOD1> <MOD2> <PERC> <RUN_ID>
Checkpoint: <DATASET>/TCGM/<PERC>_<RUN_ID>.pth (final EMA weights, as the other baselines)
"""

import itertools
import copy
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score

from functions import (evaluation, MOMENTUM_EMA, cumulate_EMA, WARM_UP_EPOCH_EMA, EPOCHS,
                       WARM_UP_EPOCH_SSL, RATIO_LABELED_UNLABELED_BATCHES)
from tcgm import TCGMModel, SFFCConfig, tcgm_loss

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

LAMBDA_TC = 1.0   # weight of L_TC (official code: alpha = 1.0)

if __name__ == "__main__":
    batch_size = 16
    dataset_path, first_prefix, second_prefix, perc, run_id = sys.argv[1:6]
    print(sys.argv)

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

    print("f_lab_data_train %d" % len(f_lab_data_train))
    print("f_unlab_data_train %d" % len(f_unlab_data_train))

    n_classes = len(np.unique(full_labels))
    prior = np.bincount(labels, minlength=n_classes).astype(np.float64)
    prior = prior / prior.sum()
    print("empirical prior p (labelled set):", np.round(prior, 4).tolist())

    dir_name = dataset_path + "/TCGM"
    os.makedirs(dir_name, exist_ok=True)
    output_file = dir_name + "/%s_%s.pth" % (perc, run_id)

    ########## TEST DATA ##########
    test_idx = np.setdiff1d(np.arange(full_labels.shape[0]), train_idx)
    test_dataset = TensorDataset(torch.tensor(first_data[test_idx], dtype=torch.float32),
                                 torch.tensor(second_data[test_idx], dtype=torch.float32),
                                 torch.tensor(full_labels[test_idx], dtype=torch.int64))
    dataloader_test = DataLoader(test_dataset, shuffle=False, batch_size=batch_size * RATIO_LABELED_UNLABELED_BATCHES,
                                 num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=4,
                                 drop_last=False)
    print("TEST DATA built")
    sys.stdout.flush()

    lab_dataset = TensorDataset(torch.tensor(f_lab_data_train, dtype=torch.float32),
                                torch.tensor(s_lab_data_train, dtype=torch.float32),
                                torch.tensor(labels, dtype=torch.int64))
    unl_dataset = TensorDataset(torch.tensor(f_unlab_data_train, dtype=torch.float32),
                                torch.tensor(s_unlab_data_train, dtype=torch.float32))
    dataloader_lab_train = DataLoader(lab_dataset, shuffle=True, batch_size=batch_size, num_workers=0,
                                      pin_memory=True, drop_last=False)
    dataloader_unl_train = DataLoader(unl_dataset, shuffle=True, batch_size=batch_size * RATIO_LABELED_UNLABELED_BATCHES,
                                      num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=4,
                                      drop_last=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("all dataloader built")
    sys.stdout.flush()

    config = SFFCConfig(img_size_m1=f_lab_data_train.shape[2], img_size_m2=s_lab_data_train.shape[2],
                        patch_size_m1=8, patch_size_m2=8,
                        in_chans_m1=f_lab_data_train.shape[1], in_chans_m2=s_lab_data_train.shape[1],
                        num_classes=n_classes, hidden_dim=256, dropout=0.1)
    model = TCGMModel(config, prior=torch.tensor(prior, dtype=torch.float32)).to(device)
    model.compile()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()
    print("model created and compiled")
    sys.stdout.flush()

    ema_weights = None
    for epoch in range(EPOCHS):
        start_time = time.time()
        use_ssl = epoch >= WARM_UP_EPOCH_SSL
        total_loss = torch.zeros((), device=device)
        tc_sum = torch.zeros((), device=device)
        a_sum = torch.zeros((), device=device)
        b_sum = torch.zeros((), device=device)
        n_batches, n_ssl = 0, 0
        model.train()
        for (f_batch, s_batch, y_batch), (f_batch_unl, s_batch_unl) in zip(
                itertools.cycle(dataloader_lab_train), dataloader_unl_train):
            optimizer.zero_grad(set_to_none=True)
            f_batch = f_batch.to(device, non_blocking=True)
            s_batch = s_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            with autocast():
                l1, l2 = model(f_batch, s_batch)
                # one CE per modality classifier (paper: L_CE for each h_m)
                sup_loss = F.cross_entropy(l1.float(), y_batch) + F.cross_entropy(l2.float(), y_batch)
                if use_ssl:
                    f_batch_unl = f_batch_unl.to(device, non_blocking=True)
                    s_batch_unl = s_batch_unl.to(device, non_blocking=True)
                    u1, u2 = model(f_batch_unl, s_batch_unl)          # no augmentation
                    tc, term_a, term_b = tcgm_loss(F.softmax(u1.float(), dim=-1), F.softmax(u2.float(), dim=-1),
                                                   model.prior)
                    loss = sup_loss + LAMBDA_TC * tc
                    tc_sum += tc.detach(); a_sum += term_a; b_sum += term_b; n_ssl += 1
                else:
                    loss = sup_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.detach()
            n_batches += 1

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
            k = max(n_ssl, 1)
            print(f"epoch {epoch} "
                  f"total={(total_loss / max(n_batches, 1)).item():.4f} "
                  f"L_TC={(tc_sum / k).item():.4f} "
                  f"same={(a_sum / k).item():.4f} "
                  f"diff={(b_sum / k).item():.4f} "
                  f"elapsed_time={elapsed_time:.2f} "
                  f"F1-score={(f1_val * 100):.2f}")
            sys.stdout.flush()

    model.load_state_dict(ema_weights)
    torch.save(model.state_dict(), output_file)
