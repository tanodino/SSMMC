"""
v7: semi-supervised fine-tuning ALIGNED WITH pretrain_v2.py (cross-modal SimCLR).

--------------------------------------------------------------------------
What changed relative to v5 (everything else is identical / imported from v5)
--------------------------------------------------------------------------
Unlabelled data -- the SAME self-supervised loss as pretrain_v2.py 'standard',
computed by the SAME function (pretrain_v2.compute_losses):
    two augmented views per batch (pretrain_v2.make_views, AUG_STANDARD)
    L1 = SimCLR on modality 1   (projector_m1,  T = 0.2)
    L2 = SimCLR on modality 2   (projector_m2,  T = 0.2)
    L3 = cross-modal contrastive on the cross-modal heads (cross_head_m1/m2,
         T = 0.1), positive = the two modalities of a sample, mean over views
    (v5: clean + 1 strong view, T = 1.0, L3 on the first 50% CLS dims)

Labelled data -- unchanged from v5: CE on the multi-layer-fusion classifier
(classify_alf, layer dropout), no pseudo-labels.

Total loss:  lambda_ssl * (L1 + L2 + L3) + lambda_cls * CE     (defaults 0.1, 1.0)
    (v5: 0.5 * (L1 + L2) + L3 + lambda_cls * CE)

Model -- v5 model + the two cross-modal heads, loaded from the pretrain_v2
checkpoint (the loader stops if any pretrained module is missing). The heads
are trained with the backbone learning rate.

Unchanged from v5: data loading and splits, batch sizes (16 labelled, 96
unlabelled), fusion module, optimiser and learning rates, fp16 autocast,
gradient clipping, EMA, evaluation every 10 epochs (classifier + k-NN on the
fused embedding), saved EMA checkpoint.

Usage (standardised data, checkpoint from pretrain_v2.py):
    python ssl_pretrained_classif_v7.py TRISTAR_STD DEPTH THERMAL 5 0 \\
        TRISTAR_STD/PRETRAIN_V2/checkpoint_latest.pth [--freeze] [--all_layers_combination]
    python ssl_pretrained_classif_v7.py --self_check
"""

import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import f1_score

from model import SFFCConfig, ViTEncoder
from functions import MOMENTUM_EMA, cumulate_EMA, WARM_UP_EPOCH_EMA, EPOCHS, RATIO_LABELED_UNLABELED_BATCHES
# unchanged pieces of v5
from ssl_pretrained_classif_v5 import (get_quarterly_layer_indices, encoder_forward_all_layers,
                                       LightweightLayerFusion, wrap_gradient_checkpointing,
                                       compute_reference_embedding, evaluate)
# self-supervised objective, views and heads: the SAME code as the pretraining stage
from pretrain_v2 import compute_losses, make_views, make_projector, AUGS, RECIPES

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

SSL = RECIPES["standard"]        # temperatures / views / augmentation of the pretraining stage


# ==========================================================================
# Model = v5 model + cross-modal heads
# ==========================================================================
class PretrainModelV7(nn.Module):
    """Same attribute names as pretrain_v2.PretrainModel (encoders, projectors,
    cross-modal heads) so that pretrain_v2.compute_losses works on it, plus
    the v5 fusion modules and classifier."""

    def __init__(self, config: SFFCConfig, num_classes: int, layer_indices: list,
                 embed_dim: int = 384, layer_dropout: float = 0.2, gating: str = "sigmoid"):
        super().__init__()
        self.modality_1_encoder = ViTEncoder(img_size=config.img_size_m1, patch_size=config.patch_size_m1,
                                             in_chans=config.in_chans_m1)
        self.modality_2_encoder = ViTEncoder(img_size=config.img_size_m2, patch_size=config.patch_size_m2,
                                             in_chans=config.in_chans_m2)
        self.projector_m1 = make_projector(embed_dim)     # L1
        self.projector_m2 = make_projector(embed_dim)     # L2
        self.cross_head_m1 = make_projector(embed_dim)    # L3
        self.cross_head_m2 = make_projector(embed_dim)

        self.layer_indices = list(layer_indices)
        # new modules (not in the pretraining checkpoint) -- as in v5
        self.alf_m1 = LightweightLayerFusion(num_layers=len(layer_indices), embed_dim=embed_dim,
                                             gating=gating, layer_dropout=layer_dropout)
        self.alf_m2 = LightweightLayerFusion(num_layers=len(layer_indices), embed_dim=embed_dim,
                                             gating=gating, layer_dropout=layer_dropout)
        self.classifier = nn.Linear(embed_dim * 2, num_classes)

    def classify_alf(self, x1, x2, use_checkpointing: bool = True):
        """Identical to v5's PretrainModelV5.classify_alf."""
        layers_m1 = encoder_forward_all_layers(self.modality_1_encoder, x1, self.layer_indices,
                                               use_checkpointing=use_checkpointing)
        layers_m2 = encoder_forward_all_layers(self.modality_2_encoder, x2, self.layer_indices,
                                               use_checkpointing=use_checkpointing)
        fused_m1, fused_m2 = self.alf_m1(layers_m1), self.alf_m2(layers_m2)
        return self.classifier(torch.cat([fused_m1, fused_m2], dim=1)), fused_m1, fused_m2


NEW_MODULES = ("classifier.", "alf_m1.", "alf_m2.")


def load_pretrain_v2_checkpoint(model: PretrainModelV7, path: str, device):
    """Loads encoders + projectors + cross-modal heads. Stops if any of them is
    missing (e.g. a checkpoint from the old pretrain.py, which has no
    cross-modal heads) or if the checkpoint contains unknown keys."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    result = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in result.missing_keys if not k.startswith(NEW_MODULES)]
    if missing or result.unexpected_keys:
        raise RuntimeError(f"checkpoint {path} does not match PretrainModelV7.\n"
                           f"  missing pretrained keys: {missing[:8]}{' ...' if len(missing) > 8 else ''}\n"
                           f"  unexpected keys: {list(result.unexpected_keys)[:8]}\n"
                           f"  (v7 needs a checkpoint from pretrain_v2.py --recipe standard)")
    if isinstance(ckpt, dict) and "args" in ckpt:
        a = ckpt["args"]
        print(f"pretraining checkpoint: recipe={a.get('recipe')} dataset={a.get('dataset')} "
              f"step={ckpt.get('step')}")
    print(f"Loaded encoders + projectors + cross-modal heads from {path} "
          f"(classifier, alf_m1, alf_m2 are new)")


def freeze_pretrained_backbone(model: PretrainModelV7):
    """As v5: freezes the encoders only (projectors and cross-modal heads stay trainable)."""
    for enc in (model.modality_1_encoder, model.modality_2_encoder):
        for p in enc.parameters():
            p.requires_grad = False


# ==========================================================================
# Arguments
# ==========================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Semi-supervised fine-tuning aligned with pretrain_v2.py "
                    "(L1 + L2 + L3 on unlabelled data, multi-layer fusion CE on labelled data).")
    parser.add_argument("dataset_path", type=str, nargs="?", help="e.g. TRISTAR_STD")
    parser.add_argument("first_prefix", type=str, nargs="?", help="e.g. DEPTH")
    parser.add_argument("second_prefix", type=str, nargs="?", help="e.g. THERMAL")
    parser.add_argument("perc", type=str, nargs="?", help="labelled samples per class, e.g. 5")
    parser.add_argument("run_id", type=str, nargs="?", help="split id, e.g. 0")
    parser.add_argument("checkpoint_path", type=str, nargs="?", help="pretrain_v2.py checkpoint_latest.pth")

    parser.add_argument("--freeze", action="store_true",
                        help="freeze the encoders (projectors and cross-modal heads still trainable)")
    parser.add_argument("--all_layers_combination", action="store_true", default=False,
                        help="fuse all the internal ViT layers instead of the quarterly ones")
    parser.add_argument("--no_crossmodal_ssl", action="store_true",
                        help="drop L3 (cross-modal term) from the self-supervised loss")
    parser.add_argument("--grad-checkpointing", action=argparse.BooleanOptionalAction, default=True,
                        help="gradient checkpointing on both encoders (default: on)")

    parser.add_argument("--output_dir", type=str, default="OURS_V7")
    parser.add_argument("--lambda-ssl", type=float, default=0.1, help="weight of (L1 + L2 + L3)")
    parser.add_argument("--lambda-cls", type=float, default=1.0, help="weight of the classifier CE loss")
    parser.add_argument("--k-neighbors", type=int, default=5)
    parser.add_argument("--backbone-lr", type=float, default=5e-6,
                        help="LR for encoders + projectors + cross-modal heads")
    parser.add_argument("--fresh-lr", type=float, default=5e-5, help="LR for classifier + alf_m1 + alf_m2")
    parser.add_argument("--gating", type=str, default="sigmoid", choices=["softmax", "sigmoid", "tanh"])
    parser.add_argument("--layer-dropout", type=float, default=0.5, help="as v5 (default 0.5)")
    parser.add_argument("--self_check", action="store_true", help="run consistency tests and exit")
    args = parser.parse_args()
    if not args.self_check and None in (args.dataset_path, args.first_prefix, args.second_prefix,
                                         args.perc, args.run_id, args.checkpoint_path):
        parser.error("dataset_path first_prefix second_prefix perc run_id checkpoint_path are required")
    return args


# ==========================================================================
# Self-check
# ==========================================================================
def self_check():
    import tempfile
    from pretrain_v2 import PretrainModel as PretrainModelV2
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ok = True

    cfg = SFFCConfig(img_size_m1=32, img_size_m2=32, patch_size_m1=8, patch_size_m2=8,
                     in_chans_m1=3, in_chans_m2=1, num_classes=4)
    pre = PretrainModelV2(32, 32, 3, 1, patch_size=8, cross_heads=True).to(dev)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pth")
        torch.save({"model_state_dict": pre.state_dict(), "args": {"recipe": "standard"}}, path)
        ft = PretrainModelV7(cfg, num_classes=4, layer_indices=get_quarterly_layer_indices(12)).to(dev)
        load_pretrain_v2_checkpoint(ft, path, dev)
    same = all(torch.equal(v, ft.state_dict()[k]) for k, v in pre.state_dict().items())
    ok &= same
    print(f"[{'OK  ' if same else 'FAIL'}] all pretrained tensors (encoders, projectors, cross heads) loaded exactly")

    # old pretrain.py checkpoints (no cross-modal heads) must be refused
    old = PretrainModelV2(32, 32, 3, 1, patch_size=8, cross_heads=False).to(dev)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "old.pth")
        torch.save({"model_state_dict": old.state_dict()}, path)
        try:
            load_pretrain_v2_checkpoint(PretrainModelV7(cfg, 4, get_quarterly_layer_indices(12)).to(dev), path, dev)
            refused = False
        except RuntimeError:
            refused = True
    ok &= refused
    print(f"[{'OK  ' if refused else 'FAIL'}] checkpoint without cross-modal heads is refused")

    # SSL loss in fine-tuning == pretraining loss for the same weights / views
    for m in (pre, ft):
        m.train(); m.modality_1_encoder.eval(); m.modality_2_encoder.eval()
    x1, x2 = torch.randn(16, 3, 32, 32, device=dev), torch.randn(16, 1, 32, 32, device=dev)
    v = make_views(x1, x2, SSL["views"], AUGS[SSL["aug"]])
    kw = dict(shared_dim=192, temp_intra=SSL["temp_intra"], temp_cross=SSL["temp_cross"], use_cross=True,
              views=SSL["views"], cross_mode=SSL["cross_mode"], intra_weight=SSL["intra_weight"])
    l_pre, _ = compute_losses(pre, *v, **kw)
    l_ft, _ = compute_losses(ft, *v, **kw)
    err = (l_pre - l_ft).abs().item()
    ok &= err < 1e-5
    print(f"[{'OK  ' if err < 1e-5 else 'FAIL'}] fine-tuning SSL loss == pretraining loss (L1+L2+L3)   |diff| = {err:.1e}")
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return ok


# ==========================================================================
# Main
# ==========================================================================
if __name__ == "__main__":
    batch_size = 16
    args = parse_args()
    if args.self_check:
        sys.exit(0 if self_check() else 1)
    print(vars(args))

    dataset_path, first_prefix, second_prefix = args.dataset_path, args.first_prefix, args.second_prefix
    perc, run_id = args.perc, args.run_id
    LAMBDA_SSL, LAMBDA_CLS = args.lambda_ssl, args.lambda_cls
    K_NEIGHBORS = args.k_neighbors
    USE_GRAD_CHECKPOINTING = args.grad_checkpointing
    A = AUGS[SSL["aug"]]
    ssl_kw = dict(shared_dim=192, temp_intra=SSL["temp_intra"], temp_cross=SSL["temp_cross"],
                  use_cross=not args.no_crossmodal_ssl, views=SSL["views"],
                  cross_mode=SSL["cross_mode"], intra_weight=SSL["intra_weight"])
    print(f"SSL loss = {LAMBDA_SSL} * (L1 + L2{'' if args.no_crossmodal_ssl else ' + L3'}) "
          f"| T_intra={SSL['temp_intra']} T_cross={SSL['temp_cross']} views={SSL['views']} aug={SSL['aug']} "
          f"| + {LAMBDA_CLS} * CE")

    # ---------------- data (as v5) ----------------
    first_data = np.load("%s/%s_data_normalized.npy" % (dataset_path, first_prefix))
    second_data = np.load("%s/%s_data_normalized.npy" % (dataset_path, second_prefix))
    full_labels = np.load("%s/labels.npy" % dataset_path)
    train_idx = np.load("%s/train_idx.npy" % dataset_path)
    labelled_idx = np.load("%s/labelled_samples_%s_%s.npy" % (dataset_path, perc, run_id))

    unlabelled_idx = np.setdiff1d(np.arange(len(train_idx)), labelled_idx)
    f_lab_data_train = first_data[train_idx][labelled_idx]
    s_lab_data_train = second_data[train_idx][labelled_idx]
    f_unlab_data_train = first_data[train_idx][unlabelled_idx]
    s_unlab_data_train = second_data[train_idx][unlabelled_idx]
    labels = full_labels[train_idx][labelled_idx]
    n_classes = len(np.unique(labels))
    print("labelled %d | unlabelled %d | classes %d" % (len(f_lab_data_train), len(f_unlab_data_train), n_classes))

    dir_name = os.path.join(dataset_path, args.output_dir)
    os.makedirs(dir_name, exist_ok=True)
    output_file = os.path.join(dir_name, "%s_%s.pth" % (perc, run_id))

    test_idx = np.setdiff1d(np.arange(full_labels.shape[0]), train_idx)
    test_dataset = TensorDataset(torch.tensor(first_data[test_idx], dtype=torch.float32),
                                 torch.tensor(second_data[test_idx], dtype=torch.float32),
                                 torch.tensor(full_labels[test_idx], dtype=torch.int64))
    dataloader_test = DataLoader(test_dataset, shuffle=False, batch_size=batch_size * RATIO_LABELED_UNLABELED_BATCHES,
                                 num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=4,
                                 drop_last=False)

    x_tensor_f_lab = torch.tensor(f_lab_data_train, dtype=torch.float32)
    x_tensor_s_lab = torch.tensor(s_lab_data_train, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.int64)
    dataloader_lab_train = DataLoader(TensorDataset(x_tensor_f_lab, x_tensor_s_lab, y_tensor), shuffle=True,
                                      batch_size=batch_size, num_workers=0, pin_memory=True, drop_last=False)
    dataloader_unl_train = DataLoader(TensorDataset(torch.tensor(f_unlab_data_train, dtype=torch.float32),
                                                    torch.tensor(s_unlab_data_train, dtype=torch.float32)),
                                      shuffle=True, batch_size=batch_size * RATIO_LABELED_UNLABELED_BATCHES,
                                      num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=4,
                                      drop_last=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------- model ----------------
    config = SFFCConfig(img_size_m1=f_lab_data_train.shape[2], img_size_m2=s_lab_data_train.shape[2],
                        patch_size_m1=8, patch_size_m2=8,
                        in_chans_m1=f_lab_data_train.shape[1], in_chans_m2=s_lab_data_train.shape[1],
                        num_classes=n_classes, hidden_dim=256, dropout=0.1)
    depth = 12   # ViTEncoder default depth (checked below)
    layer_indices = list(range(depth)) if args.all_layers_combination else get_quarterly_layer_indices(depth)
    model = PretrainModelV7(config, num_classes=n_classes, layer_indices=layer_indices,
                            layer_dropout=args.layer_dropout, gating=args.gating).to(device)
    assert len(model.modality_1_encoder.transformer.layers) == depth
    print("layer_indices (0-based) = %s" % layer_indices)
    load_pretrain_v2_checkpoint(model, args.checkpoint_path, device)

    if USE_GRAD_CHECKPOINTING:
        model.modality_1_encoder.transformer = wrap_gradient_checkpointing(model.modality_1_encoder.transformer)
        model.modality_2_encoder.transformer = wrap_gradient_checkpointing(model.modality_2_encoder.transformer)
    if args.freeze:
        freeze_pretrained_backbone(model)
        print("Encoders FROZEN (projectors and cross-modal heads still trainable)")

    backbone_modules = (model.modality_1_encoder, model.modality_2_encoder, model.projector_m1,
                        model.projector_m2, model.cross_head_m1, model.cross_head_m2)
    backbone_params = [p for m in backbone_modules for p in m.parameters() if p.requires_grad]
    fresh_params = [p for m in (model.classifier, model.alf_m1, model.alf_m2) for p in m.parameters()]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_lr, "weight_decay": 1e-4},
        {"params": fresh_params, "lr": args.fresh_lr, "weight_decay": 1e-4},
    ])
    scaler = GradScaler('cuda')
    ssl_loss_fn = torch.compile(compute_losses)
    sys.stdout.flush()

    # ---------------- training (as v5, with the new SSL term) ----------------
    ema_weights = None
    for epoch in range(EPOCHS):
        model.train()
        start_time = time.time()
        sums = torch.zeros(5, device=device)   # total, L1, L2, L3, CE
        n_batches = 0

        for f_unl, s_unl in dataloader_unl_train:
            optimizer.zero_grad(set_to_none=True)
            f_unl = f_unl.to(device, non_blocking=True)
            s_unl = s_unl.to(device, non_blocking=True)
            f_lab_b, s_lab_b, y_lab_b = next(iter(dataloader_lab_train))
            f_lab_b, s_lab_b, y_lab_b = f_lab_b.to(device), s_lab_b.to(device), y_lab_b.to(device)

            views = make_views(f_unl, s_unl, SSL["views"], A)          # outside autocast, fp32
            with autocast('cuda'):
                # ---- unlabelled: L1 + L2 + L3, same code as pretraining ----
                loss_ssl, parts = ssl_loss_fn(model, *views, **ssl_kw)
                # ---- labelled: multi-layer fusion classification (as v5) ----
                logits_lab, _, _ = model.classify_alf(f_lab_b, s_lab_b, use_checkpointing=USE_GRAD_CHECKPOINTING)
                loss_cls = F.cross_entropy(logits_lab, y_lab_b)
                loss = LAMBDA_SSL * loss_ssl + LAMBDA_CLS * loss_cls

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            sums += torch.stack([loss.detach().float(), parts[0].float(), parts[1].float(),
                                 parts[2].float(), loss_cls.detach().float()])
            n_batches += 1
        elapsed_time = time.time() - start_time

        if epoch >= WARM_UP_EPOCH_EMA:
            ema_weights = cumulate_EMA(model, ema_weights, MOMENTUM_EMA)

        if epoch % 10 == 0:
            if epoch >= WARM_UP_EPOCH_EMA:
                current_state_dict = copy.deepcopy(model.state_dict())
                model.load_state_dict(ema_weights)
            ref_emb = compute_reference_embedding(model, x_tensor_f_lab, x_tensor_s_lab, device)
            cls_preds, knn_preds, test_labels = evaluate(model, ref_emb, y_tensor.to(device), dataloader_test,
                                                         n_classes, device, k=K_NEIGHBORS)
            if epoch >= WARM_UP_EPOCH_EMA:
                model.load_state_dict(current_state_dict)
            f1_cls = f1_score(test_labels, cls_preds, average="weighted")
            f1_knn = f1_score(test_labels, knn_preds, average="weighted")
            m = (sums / max(n_batches, 1)).tolist()
            print(f"epoch {epoch} total={m[0]:.4f} L1={m[1]:.4f} L2={m[2]:.4f} L3={m[3]:.4f} CE={m[4]:.4f} "
                  f"F1-classifier={f1_cls * 100:.2f} F1-knn={f1_knn * 100:.2f} "
                  f"layer_w_m1={model.alf_m1.current_weights().cpu().numpy().round(3)} "
                  f"layer_w_m2={model.alf_m2.current_weights().cpu().numpy().round(3)} "
                  f"one_epoch_time={elapsed_time:.2f}")
            sys.stdout.flush()

    if ema_weights is not None:
        model.load_state_dict(ema_weights)
    torch.save(model.state_dict(), output_file)
    print("Saved to %s" % output_file)
