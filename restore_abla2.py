import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score

from model import SFFCConfig, ViTEncoder

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ==========================================================================
# Restores OURS_ABLA2_NoMLA checkpoints (ablation: no multi-layer fusion --
# classification uses each encoder's FINAL-layer CLS token directly,
# concatenated and passed to a linear classifier).
#
# Self-contained (PretrainModelV4_NoALF defined here) to mirror the
# training script's own structure. Unlike ablation 1, there is no ALF
# module here -- no layer_indices, no gating, no --all_layers_combination.
# That machinery simply doesn't exist in this ablation, so it isn't in
# this restore script either.
# ==========================================================================


class PretrainModelV4_NoALF(nn.Module):
    """Same encoders/projectors as ssl_pretrained_classif_v4.py.
    classify_last_layer() uses the final-layer CLS token from each
    encoder directly, no multi-layer fusion."""

    def __init__(self, config: SFFCConfig, num_classes: int, embed_dim: int = 384):
        super().__init__()
        self.modality_1_encoder = ViTEncoder(
            img_size=config.img_size_m1, patch_size=config.patch_size_m1,
            in_chans=config.in_chans_m1,
        )
        self.modality_2_encoder = ViTEncoder(
            img_size=config.img_size_m2, patch_size=config.patch_size_m2,
            in_chans=config.in_chans_m2,
        )
        self.projector_m1 = nn.Sequential(
            nn.LazyLinear(512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 128), nn.BatchNorm1d(128),
        )
        self.projector_m2 = nn.Sequential(
            nn.LazyLinear(512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 128), nn.BatchNorm1d(128),
        )

        # Classifier on top of concatenated last-layer CLS tokens
        self.classifier = nn.Linear(embed_dim * 2, num_classes)

    def classify_last_layer(self, x1: torch.Tensor, x2: torch.Tensor):
        """Ablation classification path: last-layer CLS token only."""
        cls_m1 = self.modality_1_encoder(x1)   # [B, D] -- last-layer CLS
        cls_m2 = self.modality_2_encoder(x2)   # [B, D]
        concat = torch.cat([cls_m1, cls_m2], dim=1)   # [B, 2D]
        return self.classifier(concat), cls_m1, cls_m2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Restore OURS_ABLA2_NoMLA (no multi-layer fusion) checkpoints "
                    "and evaluate weighted F1 on the test split."
    )
    parser.add_argument("dataset_path", type=str, help="e.g. EUROSAT")
    parser.add_argument("first_prefix", type=str, help="e.g. SAR")
    parser.add_argument("second_prefix", type=str, help="e.g. MS")
    parser.add_argument("perc", type=str, help="labeled percentage/count identifier, e.g. 5")

    parser.add_argument("--output_dir", type=str, default="OURS_ABLA2_NoMLA",
                        help="output directory, default OURS_ABLA2_NoMLA")

    return parser.parse_args()


if __name__ == "__main__":
    batch_size = 16
    args = parse_args()

    dataset_path = args.dataset_path
    first_prefix = args.first_prefix
    second_prefix = args.second_prefix
    perc = args.perc
    output_dir = args.output_dir
    n_splits = 5
    run_ids = range(n_splits)           # ASSUMPTION: run_id in training was "0","1","2","3","4"

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

    dir_name = os.path.join(dataset_path, output_dir)
    f1_scores = []
    for run_id in run_ids:
        ckpt_path = os.path.join(dir_name, "%s_%s.pth" % (perc, run_id))

        if not os.path.exists(ckpt_path):
            print("WARNING: checkpoint not found, skipping: %s" % ckpt_path)
            continue

        print("Loading checkpoint: %s" % ckpt_path)
        sys.stdout.flush()

        model = PretrainModelV4_NoALF(config, num_classes=n_classes).to(device)

        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)   # strict=True (default) -- will error loudly on any mismatch
        model.eval()

        all_preds, all_labels = [], []
        with torch.no_grad():
            for f_batch, s_batch, y_batch in dataloader_test:
                f_batch = f_batch.to(device, non_blocking=True)
                s_batch = s_batch.to(device, non_blocking=True)
                logits, _, _ = model.classify_last_layer(f_batch, s_batch)
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