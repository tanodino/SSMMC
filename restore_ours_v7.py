"""
restore_ours_v7.py -- evaluate the checkpoints saved by ssl_pretrained_classif_v7.py.

Mirrors restore_ours_v5.py: for one dataset / label budget it loads the
checkpoints <dataset>/<output_dir>/<perc>_<run_id>.pth for run_id 0..4,
predicts the test split (all samples not in train_idx) with the multi-layer
fusion classifier and prints the weighted F1 per split and mean +- std.

The model is imported from ssl_pretrained_classif_v7.py (same architecture
as training) and loaded with strict=True, so any mismatch raises an error.
--all_layers_combination and --gating must match the training run.

Usage:
    python restore_ours_v7.py TRISTAR_STD DEPTH THERMAL 5 [--all_layers_combination] [--output_dir OURS_V7]
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score

from model import SFFCConfig
from ssl_pretrained_classif_v7 import PretrainModelV7, get_quarterly_layer_indices

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ssl_pretrained_classif_v7.py checkpoints on the test split.")
    parser.add_argument("dataset_path", type=str, help="e.g. TRISTAR_STD")
    parser.add_argument("first_prefix", type=str, help="e.g. DEPTH")
    parser.add_argument("second_prefix", type=str, help="e.g. THERMAL")
    parser.add_argument("perc", type=str, help="labelled samples per class, e.g. 5")
    parser.add_argument("--all_layers_combination", action="store_true", default=False,
                        help="must match training: fuse all ViT layers instead of the quarterly ones")
    parser.add_argument("--gating", type=str, default="sigmoid", choices=["softmax", "sigmoid", "tanh"],
                        help="must match training (default: sigmoid, as in v7)")
    parser.add_argument("--output_dir", type=str, default="OURS_V7",
                        help="folder (inside dataset_path) holding the checkpoints, default OURS_V7")
    parser.add_argument("--n_splits", type=int, default=5, help="run ids 0..n_splits-1")
    return parser.parse_args()


if __name__ == "__main__":
    batch_size = 16
    args = parse_args()
    print(vars(args))

    # ---------------- test data (as in training: all samples not in train_idx) ----------------
    first_data = np.load("%s/%s_data_normalized.npy" % (args.dataset_path, args.first_prefix), mmap_mode="r")
    second_data = np.load("%s/%s_data_normalized.npy" % (args.dataset_path, args.second_prefix), mmap_mode="r")
    full_labels = np.load("%s/labels.npy" % args.dataset_path)
    train_idx = np.load("%s/train_idx.npy" % args.dataset_path)
    test_idx = np.setdiff1d(np.arange(full_labels.shape[0]), train_idx)
    f_test = np.asarray(first_data[test_idx], dtype=np.float32)
    s_test = np.asarray(second_data[test_idx], dtype=np.float32)
    n_classes = len(np.unique(full_labels))

    test_dataset = TensorDataset(torch.from_numpy(f_test), torch.from_numpy(s_test),
                                 torch.tensor(full_labels[test_idx], dtype=torch.int64))
    dataloader_test = DataLoader(test_dataset, shuffle=False, batch_size=batch_size * 8,
                                 num_workers=6, pin_memory=True, drop_last=False)
    print("TEST DATA built (%d samples, %d classes)" % (len(test_dataset), n_classes))
    sys.stdout.flush()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------- model config (must match training) ----------------
    config = SFFCConfig(img_size_m1=f_test.shape[2], img_size_m2=s_test.shape[2],
                        patch_size_m1=8, patch_size_m2=8,
                        in_chans_m1=f_test.shape[1], in_chans_m2=s_test.shape[1],
                        num_classes=n_classes, hidden_dim=256, dropout=0.1)
    depth = 12
    layer_indices = list(range(depth)) if args.all_layers_combination else get_quarterly_layer_indices(depth)
    print("layer_indices (0-based) = %s" % layer_indices)

    dir_name = os.path.join(args.dataset_path, args.output_dir)
    f1_scores = []
    for run_id in range(args.n_splits):
        ckpt_path = os.path.join(dir_name, "%s_%s.pth" % (args.perc, run_id))
        if not os.path.exists(ckpt_path):
            print("WARNING: checkpoint not found, skipping: %s" % ckpt_path)
            continue

        model = PretrainModelV7(config, num_classes=n_classes, layer_indices=layer_indices,
                                gating=args.gating).to(device)
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)   # strict=True -- errors on any mismatch
        model.eval()                        # layer dropout off: all fused layers used

        all_preds, all_labels = [], []
        with torch.no_grad():
            for f_batch, s_batch, y_batch in dataloader_test:
                logits, _, _ = model.classify_alf(f_batch.to(device, non_blocking=True),
                                                  s_batch.to(device, non_blocking=True),
                                                  use_checkpointing=False)
                all_preds.append(logits.argmax(dim=1).cpu())
                all_labels.append(y_batch)

        f1_val = f1_score(torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy(), average="weighted")
        f1_scores.append(f1_val)
        print("Split %d (%s) -> F1 = %.4f" % (run_id, ckpt_path, f1_val))
        sys.stdout.flush()
        del model
        torch.cuda.empty_cache()

    f1_scores = np.array(f1_scores)
    print("\n===== Summary over %d splits =====" % len(f1_scores))
    if len(f1_scores):
        print("%.2f $\\pm$ %.2f" % (f1_scores.mean() * 100, f1_scores.std() * 100))
