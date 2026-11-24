"""summarize_per_split.py -- results table from a per_split.csv written by export_predictions.py.

For every dataset x method folder x label budget: weighted F1 (x100) mean +- std over the available splits, the number
of splits, and the per-split values. Writes <dir>/summary_table.csv and prints a readable table (also saved as
<dir>/summary_table.txt).
    python summarize_per_split.py results/preds_jz_full
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np

out_dir = sys.argv[1] if len(sys.argv) > 1 else "results/preds_jz_full"
rows = list(csv.DictReader(open(os.path.join(out_dir, "per_split.csv"))))
groups = defaultdict(dict)
for r in rows:
    groups[(r["dataset"], r["folder"], int(r["perc"]))][int(r["run"])] = 100 * float(r["f1_weighted"])

table = []
for (ds, folder, perc) in sorted(groups):
    vals = groups[(ds, folder, perc)]
    v = np.array([vals[k] for k in sorted(vals)])
    table.append(dict(dataset=ds, folder=folder, perc=perc, n_splits=len(v),
                      f1_mean=f"{v.mean():.2f}", f1_std=f"{v.std():.2f}",
                      per_split=" ".join(f"{k}:{vals[k]:.2f}" for k in sorted(vals))))

with open(os.path.join(out_dir, "summary_table.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
    w.writeheader()
    w.writerows(table)

lines = []
for ds in sorted({t["dataset"] for t in table}):
    lines.append(f"\n== {ds}   (weighted F1 x100, mean +- std over n splits)")
    lines.append(f"{'method':<18}" + "".join(f"{p:>22}" for p in (5, 25, 50)))
    for folder in sorted({t["folder"] for t in table if t["dataset"] == ds}):
        cells = []
        for p in (5, 25, 50):
            t = [x for x in table if x["dataset"] == ds and x["folder"] == folder and x["perc"] == p]
            cells.append(f"{t[0]['f1_mean']} +- {t[0]['f1_std']} ({t[0]['n_splits']})" if t else "-")
        lines.append(f"{folder:<18}" + "".join(f"{c:>22}" for c in cells))
text = "\n".join(lines)
print(text)
open(os.path.join(out_dir, "summary_table.txt"), "w").write(text + "\n")
print(f"\nwrote {out_dir}/summary_table.csv and summary_table.txt")
