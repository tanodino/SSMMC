import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score

from model import SFFCConfig, ViTEncoder, encoder_forward_all_layers, LightweightLayerFusion, PretrainModelV4

from functions import get_quarterly_layer_indices

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ==========================================================================
# Reproduced from resume_pretrain_alf_main.py -- self-contained, must match
# the architecture that produced the checkpoints being restored here. If
# you changed layer_indices / gating / embed_dim during training,
# load_state_dict(strict=True) below will fail with a clear key mismatch
# rather than silently loading something wrong.
# ==========================================================================

if __name__ == "__main__":
    batch_size = 16
    dataset_path = sys.argv[1]
    first_prefix = sys.argv[2]
    second_prefix = sys.argv[3]
    perc = sys.argv[4]
    n_splits = 5
    run_ids = range(n_splits)           # ASSUMPTION: run_id in training was "0","1","2","3","4"
                                         # -> adjust `run_ids` below if your run_ids are named differently

    # ---------------- Load data (mirrors training script) ----------------
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

    # ---------------- Model config (must match training exactly) ----------------
    config = SFFCConfig(
        img_size_m1=f_data_test.shape[2], img_size_m2=s_data_test.shape[2],
        patch_size_m1=8, patch_size_m2=8,
        in_chans_m1=f_data_test.shape[1], in_chans_m2=s_data_test.shape[1],
        num_classes=n_classes, hidden_dim=256, dropout=0.1
    )

    # depth probed the same way training did, to reproduce identical layer_indices
    _probe_encoder = ViTEncoder(img_size=config.img_size_m1, patch_size=config.patch_size_m1,
                                 in_chans=config.in_chans_m1)
    depth = len(_probe_encoder.transformer.layers)
    layer_indices = get_quarterly_layer_indices(depth)
    del _probe_encoder
    print("ViT depth=%d, layer_indices (0-based)=%s" % (depth, layer_indices))

    dir_name = os.path.join(dataset_path, "RESUME_PRETRAIN_ALF")
    f1_scores = []
    for run_id in run_ids:
        ckpt_path = os.path.join(dir_name, "%s_%s.pth" % (perc, run_id))

        if not os.path.exists(ckpt_path):
            print("WARNING: checkpoint not found, skipping: %s" % ckpt_path)
            continue

        print("Loading checkpoint: %s" % ckpt_path)
        sys.stdout.flush()

        model = PretrainModelV4(config, num_classes=n_classes, layer_indices=layer_indices).to(device)

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