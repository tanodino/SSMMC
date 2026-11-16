"""
eval_baselines_table.py -- evaluate all baseline checkpoints and build the results table.

Same evaluation as restore_baselines.py (same model classes, same test split =
all samples not in train_idx, same functions.evaluation, weighted F1), looped
over datasets x methods x fusion x labels-per-class, splits 0..4.

    python eval_baselines_table.py                      # defaults below
    python eval_baselines_table.py --methods SF FC SOFTMATCH_SF SOFTMATCH_FC --out results/main
    python eval_baselines_table.py --datasets EUROSAT_STD:SAR:MS

Checkpoints: <DATASET>/<FOLDER>/<perc>_<run>.pth, FOLDER in {SF, FC, SOFTMATCH_SF,
SOFTMATCH_FC, FIXMATCH_SF, FIXMATCH_FC, FREEMATCH_SF, FREEMATCH_FC}; the fusion
model (ScoreFusion / FusionConcat) is taken from the folder suffix.

Outputs (<out> = --out prefix):
  <out>_per_split.csv   one row per checkpoint: dataset, method, fusion, perc, run, f1
  <out>_summary.csv     one row per configuration: mean, std (x100, over splits), n_splits
  <out>_table.tex       LaTeX tabular: rows = method/fusion, columns = dataset x perc
Missing checkpoints are listed at the end and shown as "--" (or with the number
of splits found, e.g. "(3/5)") in the table.
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score

from model import SFFCConfig, ScoreFusion, FusionConcat
from functions import evaluation

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEFAULT_DATASETS = ["SUNRGBD_STD:RGB:DEPTH", "EUROSAT_STD:SAR:MS", "TRISTAR_STD:DEPTH:THERMAL"]
DEFAULT_METHODS = ["SF", "FC", "FIXMATCH_SF", "FIXMATCH_FC", "FREEMATCH_SF", "FREEMATCH_FC",
                   "SOFTMATCH_SF", "SOFTMATCH_FC"]
METHOD_NAMES = {"": "Supervised", "FIXMATCH": "FixMatch", "FREEMATCH": "FreeMatch", "SOFTMATCH": "SoftMatch"}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate all baseline checkpoints and build the results table.")
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, help="DATASET:MOD1:MOD2 entries")
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, help="checkpoint folder names")
    p.add_argument("--percs", nargs="+", default=["5", "25", "50"])
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--out", default="results/baselines", help="output path prefix")
    return p.parse_args()


def split_folder(folder):
    """'SOFTMATCH_FC' -> ('SoftMatch', 'FC'); 'SF' -> ('Supervised', 'SF')."""
    fusion = folder.split("_")[-1]
    prefix = folder[:-len(fusion)].rstrip("_")
    assert fusion in ("SF", "FC"), f"cannot read the fusion type from folder name {folder}"
    return METHOD_NAMES.get(prefix, prefix), fusion


def short(dataset):
    """Display name: folder name without the _STD suffix (EUROSAT_STD -> EUROSAT)."""
    return os.path.basename(dataset.rstrip("/")).replace("_STD", "")


def load_test(dataset, m1, m2):
    """Test split exactly as restore_baselines.py."""
    first = np.load(f"{dataset}/{m1}_data_normalized.npy", mmap_mode="r")
    second = np.load(f"{dataset}/{m2}_data_normalized.npy", mmap_mode="r")
    labels = np.load(f"{dataset}/labels.npy")
    train_idx = np.load(f"{dataset}/train_idx.npy")
    test_idx = np.setdiff1d(np.arange(labels.shape[0]), train_idx)
    f = torch.tensor(np.asarray(first[test_idx]), dtype=torch.float32)
    s = torch.tensor(np.asarray(second[test_idx]), dtype=torch.float32)
    y = torch.tensor(labels[test_idx], dtype=torch.int64)
    loader = DataLoader(TensorDataset(f, s, y), shuffle=False, batch_size=16 * 8, num_workers=6,
                        pin_memory=True, drop_last=False)
    config = SFFCConfig(img_size_m1=f.shape[2], img_size_m2=s.shape[2], patch_size_m1=8, patch_size_m2=8,
                        in_chans_m1=f.shape[1], in_chans_m2=s.shape[1],
                        num_classes=len(np.unique(labels)), hidden_dim=256, dropout=0.1)
    return loader, config, len(y)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows, missing = [], []

    for entry in args.datasets:
        dataset, m1, m2 = entry.split(":")
        loader, config, n_test = load_test(dataset, m1, m2)
        print(f"== {dataset} ({m1}+{m2}): {n_test} test samples, {config.num_classes} classes", flush=True)
        for folder in args.methods:
            method, fusion = split_folder(folder)
            for perc in args.percs:
                for run in range(args.n_splits):
                    ckpt = os.path.join(dataset, folder, f"{perc}_{run}.pth")
                    if not os.path.exists(ckpt):
                        missing.append(ckpt)
                        continue
                    model = (ScoreFusion if fusion == "SF" else FusionConcat)(config).to(device)
                    state = torch.load(ckpt, map_location=device)
                    state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
                    model.load_state_dict(state)          # strict: errors on any mismatch
                    model.eval()
                    with torch.no_grad():
                        preds, labels = evaluation(model, loader, device)
                    f1 = f1_score(labels, preds, average="weighted")
                    rows.append(dict(dataset=dataset, method=method, folder=folder, fusion=fusion,
                                     perc=perc, run=run, f1=f1))
                    print(f"  {folder:<13s} {perc:>2s} labels/class split {run}: F1 = {f1 * 100:.2f}", flush=True)
                    del model
                    if device == "cuda":
                        torch.cuda.empty_cache()

    # ---- per split ----
    with open(args.out + "_per_split.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "method", "folder", "fusion", "perc", "run", "f1"])
        w.writeheader()
        w.writerows(rows)

    # ---- summary ----
    datasets = [e.split(":")[0] for e in args.datasets]
    summary = {}
    for d in datasets:
        for folder in args.methods:
            for perc in args.percs:
                f1s = [r["f1"] for r in rows if r["dataset"] == d and r["folder"] == folder and r["perc"] == perc]
                summary[(d, folder, perc)] = f1s
    with open(args.out + "_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "method", "fusion", "folder", "perc", "mean_f1", "std_f1", "n_splits"])
        for (d, folder, perc), f1s in summary.items():
            method, fusion = split_folder(folder)
            mean = f"{np.mean(f1s) * 100:.2f}" if f1s else ""
            std = f"{np.std(f1s) * 100:.2f}" if f1s else ""
            w.writerow([d, method, fusion, folder, perc, mean, std, len(f1s)])

    # ---- LaTeX table: rows = method/fusion, columns = dataset x perc ----
    def cell(f1s):
        if not f1s:
            return "--"
        s = f"{np.mean(f1s) * 100:.2f} $\\pm$ {np.std(f1s) * 100:.2f}"
        return s if len(f1s) == args.n_splits else s + f" ({len(f1s)}/{args.n_splits})"

    ncol = len(datasets) * len(args.percs)
    lines = ["\\begin{tabular}{ll" + "c" * ncol + "}", "\\toprule",
             " & & " + " & ".join(f"\\multicolumn{{{len(args.percs)}}}{{c}}{{{short(d).replace('_', '-')}}}"
                                  for d in datasets) + " \\\\",
             "Method & Fusion & " + " & ".join(p for _ in datasets for p in args.percs) + " \\\\", "\\midrule"]
    for folder in args.methods:
        method, fusion = split_folder(folder)
        lines.append(f"{method} & {fusion} & " +
                     " & ".join(cell(summary[(d, folder, p)]) for d in datasets for p in args.percs) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(args.out + "_table.tex", "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # ---- console table ----
    print("\nWeighted F1 (mean ± std over splits, x100); columns = labels per class")
    header = f"{'method':<11s}{'fus':<4s}" + "".join(f"{short(d)[:8] + ' ' + p:>18s}"
                                                  for d in datasets for p in args.percs)
    print(header)
    for folder in args.methods:
        method, fusion = split_folder(folder)
        vals = []
        for d in datasets:
            for p in args.percs:
                f1s = summary[(d, folder, p)]
                vals.append(f"{np.mean(f1s) * 100:6.2f}±{np.std(f1s) * 100:5.2f}"
                            + ("" if len(f1s) == args.n_splits else f"({len(f1s)})") if f1s else "--")
        print(f"{method:<11s}{fusion:<4s}" + "".join(f"{v:>18s}" for v in vals))
    print(f"\nwrote {args.out}_per_split.csv, {args.out}_summary.csv, {args.out}_table.tex")
    if missing:
        print(f"\n{len(missing)} missing checkpoints, e.g.:")
        for m in missing[:10]:
            print("  " + m)


if __name__ == "__main__":
    main()
