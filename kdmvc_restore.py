import sys
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score

from model import SFFCConfig, ViTEncoder, KDMvCModel
from functions import evaluation

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------------- KDMvC-specific constants (must match training script) ----------------
# CONFIRMED against kdmvc_main.py: ViTEncoder's constructor default is
# embed_dim=384, and KDMvC's build_vit_encoder mirrors ScoreFusion/
# FusionConcat's instantiation (img_size/patch_size/in_chans only, no
# embed_dim/dropout override) -- so encoder capacity matches every other
# baseline. Keep in sync if kdmvc_main.py's constants ever change.
VIT_EMBED_DIM = 384
FEAT_DIM = 128  # dimensionality of z^v (view-specific feature)


def build_vit_encoder(img_size, patch_size, in_chans):
    """CONFIRMED against kdmvc_main.py -- identical to the training script's
    build_vit_encoder, reproduced here so this restore script is self-contained."""
    return ViTEncoder(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
    )


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

    dir_name = os.path.join(dataset_path, "KDMvC")
    f1_scores = []
    for run_id in run_ids:
        ckpt_path = os.path.join(dir_name, "%s_%s.pth" % (perc, run_id))

        if not os.path.exists(ckpt_path):
            print("WARNING: checkpoint not found, skipping: %s" % ckpt_path)
            continue

        print("Loading checkpoint: %s" % ckpt_path)
        sys.stdout.flush()

        vit_m1 = build_vit_encoder(config.img_size_m1, config.patch_size_m1, config.in_chans_m1)
        vit_m2 = build_vit_encoder(config.img_size_m2, config.patch_size_m2, config.in_chans_m2)

        model = KDMvCModel(
            vit_encoder_m1=vit_m1, vit_encoder_m2=vit_m2,
            embed_dim=VIT_EMBED_DIM, feat_dim=FEAT_DIM, hidden_dim=FEAT_DIM,
            num_classes=n_classes,
        ).to(device)

        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        with torch.no_grad():
            predictions, test_labels = evaluation(model, dataloader_test, device)

        f1_val = f1_score(test_labels, predictions, average="weighted")
        f1_scores.append(f1_val)

        print("Split %s -> F1 = %.4f" % (str(run_id), f1_val))
        sys.stdout.flush()

        del model
        torch.cuda.empty_cache()

    f1_scores = np.array(f1_scores)
    print("\n===== Summary over %d splits =====" % len(f1_scores))
    print("%.2f $\pm$ %.2f" % (f1_scores.mean() * 100, f1_scores.std() * 100))