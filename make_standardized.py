"""
make_standardized.py -- create a standardised copy of a dataset folder.

    python make_standardized.py EUROSAT SAR MS
    python make_standardized.py TRISTAR DEPTH THERMAL
    python make_standardized.py SUNRGBD RGB DEPTH

For <DATASET>/ it creates <DATASET>_STD/ containing
  * <MOD>_data_normalized.npy  -- float32, (x - mean_c) / std_c per channel c,
                                  mean/std computed on the TRAIN split only
                                  (train_idx.npy), applied to ALL samples.
                                  Same file name as the original, so every
                                  script works unchanged: pass <DATASET>_STD
                                  instead of <DATASET>.
  * relative symlinks to ../<DATASET>/labels.npy, train_idx.npy and every
    labelled_samples_*.npy  -> splits are identical to the original folder.
  * stats.json                -- per-channel mean/std used, for the record.

Existing files are never overwritten (the script stops instead).
"""

import glob
import json
import os
import sys

import numpy as np

CHUNK = 1024   # samples per chunk (keeps memory low for large arrays)


def channel_stats(x, idx):
    """Per-channel mean/std over the samples `idx` (float64 accumulation)."""
    c = x.shape[1]
    s, s2, n = np.zeros(c), np.zeros(c), 0
    for i in range(0, len(idx), CHUNK):
        b = np.asarray(x[idx[i:i + CHUNK]], dtype=np.float64)
        s += b.sum(axis=(0, 2, 3))
        s2 += (b ** 2).sum(axis=(0, 2, 3))
        n += b.shape[0] * b.shape[2] * b.shape[3]
    mean = s / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 0.0))
    return mean, std


def main(dataset, mods):
    src = dataset.rstrip("/")
    dst = src + "_STD"
    os.makedirs(dst, exist_ok=True)
    train_idx = np.sort(np.load(os.path.join(src, "train_idx.npy")))
    stats = {"source": os.path.abspath(src), "train_samples": int(len(train_idx)), "modalities": {}}

    # ---- standardised arrays ----
    for m in mods:
        fname = f"{m}_data_normalized.npy"
        out_path = os.path.join(dst, fname)
        if os.path.lexists(out_path):
            sys.exit(f"{out_path} already exists -- not overwriting. Remove it first if intended.")
        x = np.load(os.path.join(src, fname), mmap_mode="r")
        mean, std = channel_stats(x, train_idx)
        if (std == 0).any():
            print(f"WARNING {m}: zero-std channels {np.where(std == 0)[0].tolist()} -> std set to 1")
            std[std == 0] = 1.0
        out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=x.shape)
        mean_b, std_b = mean.reshape(1, -1, 1, 1), std.reshape(1, -1, 1, 1)
        for i in range(0, x.shape[0], CHUNK):
            out[i:i + CHUNK] = ((np.asarray(x[i:i + CHUNK], dtype=np.float64) - mean_b) / std_b).astype(np.float32)
        out.flush()
        del out
        stats["modalities"][m] = {"shape": list(x.shape), "source_dtype": str(x.dtype),
                                  "mean": mean.tolist(), "std": std.tolist()}
        print(f"{m}: {x.shape} {x.dtype} -> float32 | train mean {np.round(mean, 4).tolist()} "
              f"std {np.round(std, 4).tolist()}")

    # ---- symlinks to labels and splits (relative, so the pair of folders is portable) ----
    base = os.path.basename(src)
    links = ["labels.npy", "train_idx.npy"] + sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(src, "labelled_samples_*.npy")))
    for f in links:
        link = os.path.join(dst, f)
        if os.path.lexists(link):
            sys.exit(f"{link} already exists -- not overwriting.")
        os.symlink(os.path.join("..", base, f), link)
    print(f"linked {len(links)} split/label files")

    with open(os.path.join(dst, "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=1)

    # ---- verification ----
    for m in mods:
        y = np.load(os.path.join(dst, f"{m}_data_normalized.npy"), mmap_mode="r")
        mean, std = channel_stats(y, train_idx)
        assert y.shape == tuple(stats["modalities"][m]["shape"]), "shape mismatch"
        assert np.allclose(mean, 0, atol=1e-3) and np.allclose(std, 1, atol=1e-3), (mean, std)
        assert np.isfinite(np.asarray(y[:CHUNK])).all()
        print(f"verified {m}: train-split mean ~0, std ~1 (max |mean| {np.abs(mean).max():.1e}, "
              f"max |std-1| {np.abs(std - 1).max():.1e})")
    for f in links:
        assert np.array_equal(np.load(os.path.join(dst, f)), np.load(os.path.join(src, f))), f
    print(f"verified {len(links)} links resolve to the original files\ndone: {dst}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
