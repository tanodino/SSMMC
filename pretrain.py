import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
import numpy as np
from model import ViTEncoder
import time
import sys
from torch.utils.data import TensorDataset, DataLoader
from functions import evaluation, strong_augment_pair, MOMENTUM_EMA, cumulate_EMA, WARM_UP_EPOCH_EMA, EPOCHS, NTXentLoss
import random
import bitsandbytes as bnb
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint
import os
import warnings
from model import SFFCConfig, PretrainModel
from sklearn.metrics import f1_score
import copy

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Flash attention toggle -- HARDWARE-SPECIFIC. The source script disables
# flash/mem-efficient SDP as a V100-compatibility fallback. Leave these
# lines OUT (or set both flash/mem_efficient to True) if you're on a newer
# GPU (A100/H100/etc), where flash attention is faster AND lower-memory --
# disabling it there would work against the "memory decreasing" goal.
# torch.backends.cuda.enable_flash_sdp(False)
# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_math_sdp(True)


def _strip_prefix(state_dict, prefix="_orig_mod."):
    return {(k[len(prefix):] if k.startswith(prefix) else k): v
            for k, v in state_dict.items()}


def save_pretrained_encoders(model_raw: "PretrainModel", path: str):
    """model_raw must be the UNCOMPILED module (model._orig_mod if compiled)."""
    encoders_state = {
        "modality_1": _strip_prefix(model_raw.modality_1_encoder.state_dict()),
        "modality_2": _strip_prefix(model_raw.modality_2_encoder.state_dict()),
    }
    torch.save(encoders_state, path)


def wrap_gradient_checkpointing(transformer_encoder):
    """Wraps a nn.TransformerEncoder so each layer runs under
    torch.utils.checkpoint -- trades recomputation for activation memory."""
    original_layers = transformer_encoder.layers

    class CheckpointedTransformerEncoder(nn.Module):
        def __init__(self, layers, norm):
            super().__init__()
            self.layers = layers
            self.norm = norm

        def forward(self, src, mask=None, src_key_padding_mask=None):
            x = src
            for layer in self.layers:
                x = checkpoint(layer, x, mask, src_key_padding_mask, use_reentrant=False)
            if self.norm is not None:
                x = self.norm(x)
            return x

    return CheckpointedTransformerEncoder(original_layers, transformer_encoder.norm)


if __name__ == "__main__":
    pretrain_batch_size = 512
    dataset_path = sys.argv[1]
    first_prefix = sys.argv[2]
    second_prefix = sys.argv[3]
    perc = sys.argv[4]
    run_id = sys.argv[5]
    sf_or_fc = sys.argv[6]  # unused by pretraining itself, kept for CLI consistency
    shared_unshared = 50
    USE_GRAD_CHECKPOINTING = True  # set False to disable

    first_data = np.load("%s/%s_data_normalized.npy" % (dataset_path, first_prefix))
    second_data = np.load("%s/%s_data_normalized.npy" % (dataset_path, second_prefix))
    full_labels = np.load("%s/labels.npy" % dataset_path)
    train_idx = np.load("%s/train_idx.npy" % dataset_path)

    full_train_data_first = first_data[train_idx]
    full_train_data_second = second_data[train_idx]

    dir_name = dataset_path + "/PRETRAIN"
    os.makedirs(dir_name, exist_ok=True)
    output_file = dir_name + "/pretrained_backbones.pth"
    checkpoint_path = dir_name + "/checkpoint_latest.pth"

    print("batch_size %d" % pretrain_batch_size)

    x_tensor_f = torch.tensor(full_train_data_first, dtype=torch.float32)
    x_tensor_s = torch.tensor(full_train_data_second, dtype=torch.float32)
    train_dataset = TensorDataset(x_tensor_f, x_tensor_s)

    dataloader_train = DataLoader(train_dataset, shuffle=True,
        batch_size=pretrain_batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=True   # avoids a possible batch-of-1 hitting BatchNorm1d in the projectors
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("all dataloader built")
    sys.stdout.flush()

    config = SFFCConfig(
        img_size_m1=full_train_data_first.shape[2], img_size_m2=full_train_data_second.shape[2],
        patch_size_m1=8, patch_size_m2=8,
        in_chans_m1=full_train_data_first.shape[1], in_chans_m2=full_train_data_second.shape[1],
        num_classes=1,  # unused by PretrainModel; placeholder only if SFFCConfig requires it
        hidden_dim=256, dropout=0.1
    )

    model = PretrainModel(config).to(device)

    if USE_GRAD_CHECKPOINTING:
        model.modality_1_encoder.transformer = wrap_gradient_checkpointing(model.modality_1_encoder.transformer)
        model.modality_2_encoder.transformer = wrap_gradient_checkpointing(model.modality_2_encoder.transformer)
        print("Gradient checkpointing active on both encoders")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scaler = GradScaler()

    # ---- load checkpoint BEFORE torch.compile (compile must wrap the
    # already-resumed model, not the other way around) ----
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")
        sys.stdout.flush()

    model = torch.compile(model, mode="reduce-overhead")
    print("model created and compiled")
    sys.stdout.flush()

    for epoch in range(start_epoch, EPOCHS):
        start_time = time.time()
        total_loss = torch.zeros((), device=device)
        n_batches = 0

        for f_batch, s_batch in dataloader_train:

            model.train()
            optimizer.zero_grad(set_to_none=True)
            f_batch = f_batch.to(device, non_blocking=True)
            s_batch = s_batch.to(device, non_blocking=True)

            with autocast():
                f_strong, s_strong = strong_augment_pair(f_batch, s_batch)
                cls_token_m1, cls_token_m2, proj_m1, proj_m2 = model(f_batch, s_batch)
                _, _, proj_m1_aug, proj_m2_aug = model(f_strong, s_strong)

                n_feat = cls_token_m1.shape[-1]
                shared_n_feat = int(n_feat * shared_unshared / 100)

                emb_m1_inv = cls_token_m1[:, :shared_n_feat]
                emb_m2_inv = cls_token_m2[:, :shared_n_feat]
                emb_inv = F.normalize(torch.cat([emb_m1_inv, emb_m2_inv], dim=0), dim=1)

                repr_m1 = F.normalize(torch.cat([proj_m1, proj_m1_aug], dim=0), dim=1)
                repr_m2 = F.normalize(torch.cat([proj_m2, proj_m2_aug], dim=0), dim=1)

                labels_cls_loss = torch.arange(f_batch.shape[0]).repeat(2).to(device)

                loss_m1 = NTXentLoss(repr_m1, labels_cls_loss, temperature=1.0)
                loss_m2 = NTXentLoss(repr_m2, labels_cls_loss, temperature=1.0)
                loss_cross = NTXentLoss(emb_inv, labels_cls_loss, temperature=1.0)

                loss = .5 * (loss_m1 + loss_m2) + loss_cross

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.detach()
            n_batches += 1

        elapsed_time = time.time() - start_time
        total_loss = total_loss.item()

        if epoch % 5 == 0:
            print(f"epoch {epoch} "
                f"total={np.mean(total_loss / max(n_batches, 1)):.4f} "
                f"elapsed_time={elapsed_time:.2f} ")
            sys.stdout.flush()

            torch.save({
                "epoch": epoch,
                "model_state_dict": model._orig_mod.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "loss": total_loss / max(n_batches, 1),
            }, checkpoint_path)

        if epoch % 20 == 0 and epoch > 0:
            save_pretrained_encoders(model._orig_mod, output_file)
            print("SAVED encoders at epoch %d" % epoch)
            sys.stdout.flush()

    save_pretrained_encoders(model._orig_mod, output_file)
    print("Final encoders saved")