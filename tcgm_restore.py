# tcgm_restore.py -- evaluate the TCGM checkpoints of tcgm_main.py (<dataset>/TCGM), splits 0-4.
# Same test split and weighted F1 as the other restore scripts; prediction = TCGM aggregator
# (h1*h2/p, with the prior p stored in the checkpoint).
# Usage: python tcgm_restore.py EUROSAT_STD SAR MS 5
import os
import sys

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score

from functions import evaluation
from tcgm import TCGMModel, SFFCConfig

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if __name__ == "__main__":
    dataset_path, first_prefix, second_prefix, perc = sys.argv[1:5]

    first_data = np.load("%s/%s_data_normalized.npy" % (dataset_path, first_prefix), mmap_mode="r")
    second_data = np.load("%s/%s_data_normalized.npy" % (dataset_path, second_prefix), mmap_mode="r")
    full_labels = np.load("%s/labels.npy" % dataset_path)
    train_idx = np.load("%s/train_idx.npy" % dataset_path)
    test_idx = np.setdiff1d(np.arange(full_labels.shape[0]), train_idx)

    f = torch.tensor(np.asarray(first_data[test_idx]), dtype=torch.float32)
    s = torch.tensor(np.asarray(second_data[test_idx]), dtype=torch.float32)
    y = torch.tensor(full_labels[test_idx], dtype=torch.int64)
    loader = DataLoader(TensorDataset(f, s, y), shuffle=False, batch_size=16 * 6, num_workers=6,
                        pin_memory=True, drop_last=False)
    print("TEST DATA built (%d samples)" % len(y))

    config = SFFCConfig(img_size_m1=f.shape[2], img_size_m2=s.shape[2], patch_size_m1=8, patch_size_m2=8,
                        in_chans_m1=f.shape[1], in_chans_m2=s.shape[1],
                        num_classes=len(np.unique(full_labels)), hidden_dim=256, dropout=0.1)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    f1_scores = []
    for run_id in range(5):
        ckpt = os.path.join(dataset_path, "TCGM", "%s_%d.pth" % (perc, run_id))
        if not os.path.exists(ckpt):
            print("WARNING: checkpoint not found, skipping: %s" % ckpt)
            continue
        model = TCGMModel(config).to(device)
        state = torch.load(ckpt, map_location=device)
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state)          # strict; also restores the prior p
        model.eval()
        with torch.no_grad():
            preds, labels = evaluation(model, loader, device)
        f1 = f1_score(labels, preds, average="weighted")
        f1_scores.append(f1)
        print("Split %d -> F1 = %.4f" % (run_id, f1))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    f1_scores = np.array(f1_scores)
    print("\n===== Summary over %d splits =====" % len(f1_scores))
    print("%.2f $\\pm$ %.2f" % (f1_scores.mean() * 100, f1_scores.std() * 100))
