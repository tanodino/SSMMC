"""
export_predictions.py -- per-sample test predictions for every method / checkpoint, in one format.

For each dataset, method folder, label budget and split it loads <DATASET>/<FOLDER>/<perc>_<run>.pth,
predicts the test split (all samples not in train_idx, in the order of np.setdiff1d, the same as
every restore script) and saves:

  <out>/<DATASET>/test_labels.npz             test_idx, y_true (written once per dataset)
  <out>/<DATASET>/<FOLDER>/<perc>_<run>.npz   y_pred [N] int16, scores [N, C] float16
                                              (the model's output used for the argmax:
                                              probabilities for SF, logits for FC/KDMvC/MSC/OURS,
                                              log-aggregator for TCGM)
  <out>/per_split.csv                         dataset, folder, perc, run, n_test, acc, f1_weighted, f1_macro
                                              (rows are appended; existing (dataset,folder,perc,run) rows are kept)

Since every method is evaluated on the same test samples in the same order, paired tests
(McNemar per split, Wilcoxon / t-test over splits, bootstrap) and other metrics can be
computed afterwards from these files.

Supported folders: SF, FC, <PREFIX>_SF, <PREFIX>_FC (all SF/FC baselines), KDMvC_V2, MSC_V2, TCGM, OURS_V7.

    python export_predictions.py --out results/preds_jeanzay \
        --methods SF FC FIXMATCH_SF FIXMATCH_FC FREEMATCH_V2_SF FREEMATCH_V2_FC \
                  SOFTMATCH_V2_SF SOFTMATCH_V2_FC KDMvC_V2 MSC_V2
    python export_predictions.py --out results/preds_local --methods OURS_V7 TCGM
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score

from model import SFFCConfig

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEFAULT_DATASETS = ["SUNRGBD_STD:RGB:DEPTH", "EUROSAT_STD:SAR:MS", "TRISTAR_STD:DEPTH:THERMAL"]
FIELDS = ["dataset", "folder", "perc", "run", "n_test", "acc", "f1_weighted", "f1_macro"]


def parse_args():
    p = argparse.ArgumentParser(description="Export per-sample test predictions of all checkpoints.")
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, help="DATASET:MOD1:MOD2 entries")
    p.add_argument("--methods", nargs="+", required=True, help="checkpoint folder names")
    p.add_argument("--percs", nargs="+", default=["5", "25", "50"])
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--out", required=True, help="output directory")
    return p.parse_args()


def build_model(folder, config, n_classes):
    """Model + a function batch -> scores [B, C] (argmax = prediction), exactly as in the restore scripts."""
    if folder == "OURS_V7":
        from ssl_pretrained_classif_v7 import PretrainModelV7, get_quarterly_layer_indices
        model = PretrainModelV7(config, num_classes=n_classes, layer_indices=get_quarterly_layer_indices(12),
                                gating="sigmoid")
        return model, lambda m, a, b: m.classify_alf(a, b, use_checkpointing=False)[0]
    if folder == "TCGM":
        from tcgm import TCGMModel
        return TCGMModel(config), lambda m, a, b: m.predict(a, b)
    if folder == "KDMvC_V2":
        from model import ViTEncoder, KDMvCModel
        vit = lambda s, p, c: ViTEncoder(img_size=s, patch_size=p, in_chans=c)
        model = KDMvCModel(vit_encoder_m1=vit(config.img_size_m1, config.patch_size_m1, config.in_chans_m1),
                           vit_encoder_m2=vit(config.img_size_m2, config.patch_size_m2, config.in_chans_m2),
                           embed_dim=384, feat_dim=128, hidden_dim=128, num_classes=n_classes)
        return model, lambda m, a, b: m.predict(a, b)
    if folder == "MSC_V2":
        from msc import MSCConfig, MSCModel
        mcfg = MSCConfig(img_size_m1=config.img_size_m1, img_size_m2=config.img_size_m2,
                         patch_size_m1=8, patch_size_m2=8,
                         in_chans_m1=config.in_chans_m1, in_chans_m2=config.in_chans_m2,
                         num_classes=n_classes, hidden_dim=256, dropout=0.1, tau=0.95, eta=0.95, lambda_u=1.0,
                         beta1=1.0, beta2=1.0, beta3=1.0, detach_teacher=True)
        return MSCModel(mcfg), lambda m, a, b: m.predict(a, b)
    fusion = folder.split("_")[-1]
    if fusion in ("SF", "FC"):
        from model import ScoreFusion, FusionConcat
        return (ScoreFusion if fusion == "SF" else FusionConcat)(config), lambda m, a, b: m.predict(a, b)
    raise ValueError(f"unknown method folder {folder}")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "per_split.csv")
    rows = {}
    if os.path.exists(csv_path):
        with open(csv_path) as fh:
            for r in csv.DictReader(fh):
                rows[(r["dataset"], r["folder"], r["perc"], r["run"])] = r
    missing = []

    for entry in args.datasets:
        dataset, m1, m2 = entry.split(":")
        first = np.load(f"{dataset}/{m1}_data_normalized.npy", mmap_mode="r")
        second = np.load(f"{dataset}/{m2}_data_normalized.npy", mmap_mode="r")
        # labels / train_idx: in the dataset folder, or (Jean Zay layout) in the folder without the
        # _STD suffix -- the files are identical in both (checked on the workstation, cmp)
        aux = dataset if os.path.exists(f"{dataset}/labels.npy") else dataset.replace("_STD", "")
        if aux != dataset:
            print(f"   labels.npy / train_idx.npy read from {aux}/ (not in {dataset}/)", flush=True)
        labels = np.load(f"{aux}/labels.npy")
        train_idx = np.load(f"{aux}/train_idx.npy")
        test_idx = np.setdiff1d(np.arange(labels.shape[0]), train_idx)
        f = torch.tensor(np.asarray(first[test_idx]), dtype=torch.float32)
        s = torch.tensor(np.asarray(second[test_idx]), dtype=torch.float32)
        y = labels[test_idx]
        n_classes = len(np.unique(labels))
        loader = DataLoader(TensorDataset(f, s), shuffle=False, batch_size=128, num_workers=4,
                            pin_memory=True, drop_last=False)
        config = SFFCConfig(img_size_m1=f.shape[2], img_size_m2=s.shape[2], patch_size_m1=8, patch_size_m2=8,
                            in_chans_m1=f.shape[1], in_chans_m2=s.shape[1],
                            num_classes=n_classes, hidden_dim=256, dropout=0.1)
        ds_out = os.path.join(args.out, dataset)
        os.makedirs(ds_out, exist_ok=True)
        np.savez_compressed(os.path.join(ds_out, "test_labels.npz"), test_idx=test_idx, y_true=y)
        print(f"== {dataset} ({m1}+{m2}): {len(y)} test samples, {n_classes} classes", flush=True)

        for folder in args.methods:
            os.makedirs(os.path.join(ds_out, folder), exist_ok=True)
            for perc in args.percs:
                for run in range(args.n_splits):
                    ckpt = os.path.join(dataset, folder, f"{perc}_{run}.pth")
                    if not os.path.exists(ckpt):
                        missing.append(ckpt)
                        continue
                    npz = os.path.join(ds_out, folder, f"{perc}_{run}.npz")
                    if (dataset, folder, perc, str(run)) in rows and os.path.exists(npz) and \
                            os.path.getmtime(npz) >= os.path.getmtime(ckpt):
                        continue                          # already exported (resume after an interrupted session)
                    model, score_fn = build_model(folder, config, n_classes)
                    model = model.to(device)
                    state = torch.load(ckpt, map_location=device)
                    state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
                    model.load_state_dict(state)      # strict: errors on any mismatch
                    model.eval()
                    scores = []
                    with torch.no_grad():
                        for fb, sb in loader:
                            scores.append(score_fn(model, fb.to(device, non_blocking=True),
                                                   sb.to(device, non_blocking=True)).float().cpu())
                    scores = torch.cat(scores).numpy()
                    pred = scores.argmax(1)
                    np.savez_compressed(os.path.join(ds_out, folder, f"{perc}_{run}.npz"),
                                        y_pred=pred.astype(np.int16), scores=scores.astype(np.float16))
                    r = dict(dataset=dataset, folder=folder, perc=perc, run=str(run), n_test=len(y),
                             acc=f"{accuracy_score(y, pred):.6f}",
                             f1_weighted=f"{f1_score(y, pred, average='weighted'):.6f}",
                             f1_macro=f"{f1_score(y, pred, average='macro'):.6f}")
                    rows[(dataset, folder, perc, str(run))] = r
                    print(f"  {folder:<16s} {perc:>2s} split {run}: F1w = {float(r['f1_weighted']) * 100:.2f}",
                          flush=True)
                    del model
                    if device == "cuda":
                        torch.cuda.empty_cache()
            # write after each method, so a crash keeps what is done
            with open(csv_path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(rows[k] for k in sorted(rows))

    print(f"\nwrote {csv_path} ({len(rows)} rows)")
    if missing:
        print(f"{len(missing)} missing checkpoints:")
        for m in missing:
            print("  " + m)


if __name__ == "__main__":
    main()
