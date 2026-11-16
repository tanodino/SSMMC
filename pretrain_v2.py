"""
pretrain_v2.py -- multimodal contrastive pretraining (clean + faster rewrite of pretrain.py)
==========================================================================================

MODEL
-----
Two ViT-S encoders (model.ViTEncoder, one per modality) + per modality:
  projector_m{1,2}   SimCLR head        Linear-BN-ReLU-Linear-BN, 384 -> 512 -> 128
  cross_head_m{1,2}  cross-modal head   same architecture (recipe 'standard' only)
Encoders and projectors keep the parameter names of model.PretrainModel, so
checkpoints load into ssl_pretrained_classif_v5*.py (`<out_dir>/checkpoint_latest.pth`,
key "model_state_dict"); the cross-modal heads are reported there as unused keys.

OBJECTIVE  (recipe 'standard': cross-modal SimCLR, three contrastive terms)
---------
For a batch of paired images (x1, x2), two views (a, b) are produced:

    view a : (x1a, x2a)        view b : (x1b, x2b)

The spatial part of an augmentation (crop, flip) is SHARED by the two
modalities of the same view (keeps them aligned); everything else is drawn
independently. With h = CLS token (384-d):

    L1 = NT-Xent( projector_m1(h1a),  projector_m1(h1b) ; T_intra=0.2 )   SimCLR, modality 1
    L2 = NT-Xent( projector_m2(h2a),  projector_m2(h2b) ; T_intra=0.2 )   SimCLR, modality 2
    L3 = mean over views v in {a, b} of
         NT-Xent( cross_head_m1(h1v), cross_head_m2(h2v) ; T_cross=0.1 )  cross-modal:
         positive pair = the two modalities of the same sample
    loss = L1 + L2 + L3

Each head is applied to both views at once (BatchNorm over the two views, as SimCLR).

Recipe 'original' (= pretrain.py) instead uses L3 on the first k = shared_pct% CLS
dims (no cross-modal heads) and loss = 0.5 * (L1 + L2) + L3. With
--cross_mode shared_dims an optional shared/specific decorrelation term is
available (--decorr_weight; mean squared batch correlation between h[:, :k] and h[:, k:]).

Diagnostics logged at every log line: std_sh / std_sp = mean per-dimension batch
std of the first / second half of the CLS token (collapse -> std -> 0); with
cross_mode shared_dims also |C| = mean absolute correlation between the halves.

NT-Xent: for 2N embeddings, the positive of each one is its partner, the
other 2N-2 are negatives (cosine similarity / T, cross-entropy).

RECIPES  (--recipe; any individual flag overrides the recipe value)
-------------------------------------------------------------------
                        standard (default)                 original (= pretrain.py)
  views                 2 augmented views (SimCLR)         clean + 1 augmented view
  head BN               over both views together (SimCLR)  separately per view
  cross-modal loss      cross-modal heads,                 first 50% CLS dims,
                        mean over both views               clean view only
  loss weights          L1 + L2 + L3                       0.5 (L1 + L2) + L3
  T_intra / T_cross     0.2 / 0.1                          1.0 / 1.0
  learning rate         1.5e-4 * batch/256 (MoCo v3 ViT)   5e-5
  schedule              linear warm-up (10% of the run),   constant
                        cosine decay to 1e-6
  weight decay          0.1, not on bias/norm/pos/cls      0.01 on all parameters
  length                1000 epochs                        5000 epochs
  augmentation          AUG_STANDARD (SimCLR-like)         AUG_ORIGINAL (functions.py)

Both recipes keep gradient clipping at 1.0 and patch size 8.

SPEED (vs pretrain.py, same maths -- verified by --self_check)
-------------------------------------------------------------
* each encoder runs once on the concatenation of the two views (exact: the
  ViT has no batch-dependent layer)
* training set kept on the GPU when it fits, no DataLoader
* vectorised augmentation without GPU->CPU syncs
* bf16 autocast, fused AdamW, torch.compile, losses synced only when logging
* gradient checkpointing on a configurable number of layers (--ckpt_layers)

USAGE
-----
    python pretrain_v2.py --self_check            # equivalence tests, ~1 min
    # Batch sizes = exact divisors of the training-set size (no sample dropped,
    # no duplicates, constant batch), close to the 24 GB RTX 3090 memory limit
    # (measured at --ckpt_layers 12: 2048 @64px -> 20.3 GB, 512 @128px -> 18.8 GB):
    #   EUROSAT 18900 train -> 1890 (10 batches/epoch)
    #   TRISTAR  9940 train ->  497 (20 batches/epoch)
    #   SUNRGBD  1473 train ->  491 ( 3 batches/epoch)
    # Use the standardised copies (<DATASET>_STD, created by make_standardized.py):
    CUDA_VISIBLE_DEVICES=0 python pretrain_v2.py EUROSAT_STD SAR MS --batch_size 1890
    CUDA_VISIBLE_DEVICES=1 python pretrain_v2.py TRISTAR_STD DEPTH THERMAL --batch_size 497
    CUDA_VISIBLE_DEVICES=1 python pretrain_v2.py SUNRGBD_STD RGB DEPTH --batch_size 491
    python pretrain_v2.py EUROSAT SAR MS --no_cross        # ablation (= pretrain_abla.py)
    python pretrain_v2.py -h
Resumes automatically from <out_dir>/checkpoint_latest.pth (default out_dir:
<DATASET>/PRETRAIN_V2, never the old PRETRAIN/).
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less fragmentation

import argparse
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model import ViTEncoder

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


# =============================================================================
# 1. CONFIGURATION
# =============================================================================
RECIPES = {
    "standard": dict(views="two_aug", aug="standard", temp_intra=0.2, temp_cross=0.1,
                     cross_mode="heads", intra_weight=1.0,
                     base_lr=1.5e-4, lr=None, warmup_frac=0.1, min_lr=1e-6, schedule="cosine",
                     weight_decay=0.1, decay_all=False, total_steps=None, epochs=1000),
    "original": dict(views="clean_aug", aug="original", temp_intra=1.0, temp_cross=1.0,
                     cross_mode="shared_dims", intra_weight=0.5,
                     base_lr=None, lr=5e-5, warmup_frac=0.0, min_lr=0.0, schedule="constant",
                     weight_decay=0.01, decay_all=True, total_steps=None, epochs=5000),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Multimodal contrastive pretraining (v2)",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("dataset", nargs="?", help="dataset folder, e.g. EUROSAT")
    p.add_argument("mod1", nargs="?", help="first modality prefix, e.g. SAR")
    p.add_argument("mod2", nargs="?", help="second modality prefix, e.g. MS")
    p.add_argument("--recipe", choices=list(RECIPES), default="standard",
                   help="set of defaults (see module docstring); flags below override it")

    g = p.add_argument_group("objective")
    g.add_argument("--views", choices=["two_aug", "clean_aug"])
    g.add_argument("--aug", choices=["standard", "original"])
    g.add_argument("--aug_scale", choices=["absolute", "relative"], default="absolute",
                   help="relative: additive shift / noise scaled by each image's per-channel std, "
                        "channel dropout fills with the channel mean instead of 0")
    g.add_argument("--standardize", action="store_true",
                   help="standardise inputs with per-channel mean/std computed on the train split")
    g.add_argument("--temp_intra", type=float, help="temperature of the within-modality losses")
    g.add_argument("--temp_cross", type=float, help="temperature of the cross-modal loss")
    g.add_argument("--cross_mode", choices=["heads", "shared_dims"],
                   help="cross-modal loss on dedicated cross-modal projection heads (heads) "
                        "or directly on the first shared_pct%% CLS dims (shared_dims, pretrain.py)")
    g.add_argument("--intra_weight", type=float,
                   help="weight of each within-modality loss: total = w*(L1 + L2) + L3")
    g.add_argument("--shared_pct", type=int, default=50,
                   help="%% of CLS dims used by the cross-modal loss (cross_mode=shared_dims only)")
    g.add_argument("--no_cross", action="store_true", help="drop the cross-modal loss (= pretrain_abla.py)")
    g.add_argument("--decorr_weight", type=float, default=0.0,
                   help="weight of the shared/specific decorrelation loss (0 = off)")

    g = p.add_argument_group("optimisation")
    g.add_argument("--batch_size", type=int, default=512)
    g.add_argument("--base_lr", type=float, help="peak lr = base_lr * batch_size / 256")
    g.add_argument("--lr", type=float, help="peak lr, absolute (overrides base_lr)")
    g.add_argument("--schedule", choices=["cosine", "constant"])
    g.add_argument("--warmup_frac", type=float, help="linear warm-up length as a fraction of the run")
    g.add_argument("--warmup_steps", type=int, help="warm-up length in steps (overrides --warmup_frac)")
    g.add_argument("--min_lr", type=float, help="final lr of the cosine decay")
    g.add_argument("--weight_decay", type=float)
    g.add_argument("--decay_all", action=argparse.BooleanOptionalAction,
                   help="apply weight decay also to bias / norm / pos_embed / cls_token")
    g.add_argument("--total_steps", type=int, help="training length in optimisation steps")
    g.add_argument("--epochs", type=int, help="training length in epochs (used if total_steps is unset)")
    g.add_argument("--clip_grad", type=float, default=1.0)
    g.add_argument("--patch_size", type=int, default=8)
    g.add_argument("--seed", type=int, default=0)

    g = p.add_argument_group("performance")
    g.add_argument("--amp", choices=["bf16", "fp16", "none"], default="bf16")
    g.add_argument("--ckpt_layers", type=int, default=12,
                   help="transformer layers (of 12) under gradient checkpointing: 0 = fastest, 12 = least memory")
    g.add_argument("--no_compile", action="store_true", help="disable torch.compile")
    g.add_argument("--data_device", choices=["auto", "cuda", "cpu"], default="auto",
                   help="where the training set lives; auto = GPU if it takes < 10%% of GPU memory")

    g = p.add_argument_group("output / logging (in optimisation steps)")
    g.add_argument("--out_dir", default=None, help="default: <dataset>/PRETRAIN_V2")
    g.add_argument("--log_every", type=int, default=200)
    g.add_argument("--save_every", type=int, default=1000, help="checkpoint_latest.pth")
    g.add_argument("--backbone_every", type=int, default=10_000, help="pretrained_backbones.pth")
    g.add_argument("--max_steps", type=int, default=0, help="stop after N steps of THIS run (benchmarking)")

    p.add_argument("--self_check", action="store_true",
                   help="run equivalence tests against the original implementation and exit")
    args = p.parse_args(argv)

    for k, v in RECIPES[args.recipe].items():          # fill unset flags from the recipe
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.lr is None:
        args.lr = args.base_lr * args.batch_size / 256
    if args.decorr_weight > 0 and args.cross_mode != "shared_dims":
        p.error("--decorr_weight needs --cross_mode shared_dims (it acts on the shared/specific split)")

    if not args.self_check and not (args.dataset and args.mod1 and args.mod2):
        p.error("dataset, mod1 and mod2 are required (unless --self_check)")
    return args


# =============================================================================
# 2. DATA
# =============================================================================
def load_npy(path):
    try:
        return np.load(path, mmap_mode="r")
    except ValueError as e:
        raise SystemExit(f"Cannot read {path} as a .npy array ({e}). Check the file (`file {path}`).")


def load_train_data(dataset, mod1, mod2):
    """Train split only (train_idx.npy) as float32 CPU tensors."""
    x1 = load_npy(f"{dataset}/{mod1}_data_normalized.npy")
    x2 = load_npy(f"{dataset}/{mod2}_data_normalized.npy")
    train_idx = np.sort(np.load(f"{dataset}/train_idx.npy"))   # sorted -> sequential reads; order irrelevant
    x1 = torch.from_numpy(np.ascontiguousarray(x1[train_idx], dtype=np.float32))
    x2 = torch.from_numpy(np.ascontiguousarray(x2[train_idx], dtype=np.float32))
    assert x1.shape[0] == x2.shape[0], "modalities have different number of samples"
    return x1, x2


def place_data(x1, x2, mode, device):
    nbytes = (x1.numel() + x2.numel()) * 4
    if mode == "auto":
        mode = "cuda" if nbytes < 0.10 * torch.cuda.get_device_properties(device).total_memory else "cpu"
    if mode == "cuda":
        return x1.to(device), x2.to(device), mode, nbytes
    return x1.pin_memory(), x2.pin_memory(), mode, nbytes


def batches(x1, x2, batch_size, device):
    """Endless stream of shuffled batches (drop_last within each epoch).
    Yields (epoch, b1, b2)."""
    n, epoch = x1.shape[0], 0
    while True:
        perm = torch.randperm(n, device=x1.device)
        for i in range(n // batch_size):
            idx = perm[i * batch_size:(i + 1) * batch_size]
            b1, b2 = x1[idx], x2[idx]
            if b1.device != device:
                b1, b2 = b1.to(device, non_blocking=True), b2.to(device, non_blocking=True)
            yield epoch, b1, b2
        epoch += 1


# =============================================================================
# 3. AUGMENTATION
# =============================================================================
# All decisions are drawn per SAMPLE. Spatial ops are shared by the two
# modalities of a view; radiometric ops are independent per modality.
#
# AUG_ORIGINAL reproduces functions.strong_augment_pair (+ functions._strong_aug):
#   crop p=0.5 scale [0.5,1] | hflip 0.5 | vflip 0.5 | shift p=0.8 +-0.1 |
#   channel dropout p=0.2 (15%/channel) | noise p=0.5 std [0.01,0.10] |
#   blur p=0.25 3x3 sigma [0.1,2]   (random contrast is commented out there)
#
# AUG_STANDARD follows the SimCLR recipe (Chen et al. 2020, appendix A), with
# modality-agnostic analogues of the RGB colour operations:
#   random resized crop  p=1.0, scale [0.08, 1], ratio [3/4, 4/3]
#   horizontal flip      p=0.5           (SimCLR uses no vertical flip)
#   colour jitter        p=0.8  -> brightness = additive shift +-0.4,
#                                  contrast = per-channel x' = m + c (x - m), c in [0.6, 1.4]
#                                  (strength 0.5, the SimCLR setting for small images)
#   grayscale            p=0.2  -> channel dropout (15% of channels, >= 1 kept)
#   gaussian blur        p=0.5, kernel ~10% of image side (odd), sigma [0.1, 2.0]
#   (no additive noise: not part of SimCLR)
AUG_ORIGINAL = dict(crop_p=0.5, crop_scale=(0.5, 1.0), crop_ratio=(3 / 4, 4 / 3), hflip_p=0.5, vflip_p=0.5,
                    shift_p=0.8, shift_delta=0.1, contrast_range=(1.0, 1.0),
                    drop_p=0.2, drop_frac=0.15, noise_p=0.5, noise_std=(0.01, 0.10),
                    blur_p=0.25, blur_sigma=(0.1, 2.0), blur_ksize=3)
AUG_STANDARD = dict(crop_p=1.0, crop_scale=(0.08, 1.0), crop_ratio=(3 / 4, 4 / 3), hflip_p=0.5, vflip_p=0.0,
                    shift_p=0.8, shift_delta=0.4, contrast_range=(0.6, 1.4),
                    drop_p=0.2, drop_frac=0.15, noise_p=0.0, noise_std=(0.01, 0.10),
                    blur_p=0.5, blur_sigma=(0.1, 2.0), blur_ksize=None)   # None -> 10% of image side
AUGS = {"original": AUG_ORIGINAL, "standard": AUG_STANDARD}


def _coin(b, p, device):
    """(B,1,1,1) boolean mask, True with probability p."""
    return torch.rand(b, 1, 1, 1, device=device) < p


def sample_crop_fractions(b, device, scale, ratio):
    """Crop box as fractions of the image (= functions._sample_crop_fractions)."""
    area = torch.empty(b, device=device).uniform_(*scale)
    aspect = torch.exp(torch.empty(b, device=device).uniform_(math.log(ratio[0]), math.log(ratio[1])))
    w = torch.sqrt(area * aspect).clamp(max=1.0)
    h = torch.sqrt(area / aspect).clamp(max=1.0)
    x0 = torch.rand(b, device=device) * (1.0 - w)
    y0 = torch.rand(b, device=device) * (1.0 - h)
    return x0, y0, w, h


def crop_resize(x, x0, y0, w, h):
    """Crop the box (fractions) and resize back to the input size (bilinear).
    Equivalent to kornia crop_and_resize(align_corners=True) used in functions.py."""
    b = x.shape[0]
    theta = torch.zeros(b, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0], theta[:, 0, 2] = w, 2 * x0 + w - 1
    theta[:, 1, 1], theta[:, 1, 2] = h, 2 * y0 + h - 1
    grid = F.affine_grid(theta, list(x.shape), align_corners=True)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def gaussian_blur(x, sigma, ksize):
    """Per-sample isotropic gaussian blur, reflect padding (as kornia)."""
    b, c, hgt, wid = x.shape
    r = ksize // 2
    t = torch.arange(ksize, device=x.device, dtype=x.dtype) - r
    k = torch.exp(-(t[None, :] ** 2) / (2 * sigma[:, None] ** 2))
    k = (k / k.sum(dim=1, keepdim=True)).repeat_interleave(c, dim=0)      # (B*C, K)
    y = F.pad(x.reshape(1, b * c, hgt, wid), (r, r, r, r), mode="reflect")
    y = F.conv2d(y, k.view(b * c, 1, 1, ksize), groups=b * c)             # horizontal
    y = F.conv2d(y, k.view(b * c, 1, ksize, 1), groups=b * c)             # vertical
    return y.view(b, c, hgt, wid)


def radiometric_aug(x, A):
    """Per-modality part: [shift + contrast] -> channel dropout -> noise -> blur."""
    b, c, hgt, _ = x.shape
    dev = x.device
    relative = A.get("relative", False)
    # reference scale: 1 (absolute) or each image's per-channel std (relative)
    ref = x.std(dim=(2, 3), keepdim=True) if relative else 1.0
    # colour-jitter analogue (one coin for brightness + contrast, as SimCLR's ColorJitter)
    jitter = _coin(b, A["shift_p"], dev)
    delta = torch.empty(b, 1, 1, 1, device=dev).uniform_(-A["shift_delta"], A["shift_delta"])
    y = x + delta * ref
    if A["contrast_range"] != (1.0, 1.0):
        cf = torch.empty(b, 1, 1, 1, device=dev).uniform_(*A["contrast_range"])
        m = y.mean(dim=(2, 3), keepdim=True)
        y = m + cf * (y - m)
    x = torch.where(jitter, y, x)
    # channel dropout ("grayscale" analogue), at least one channel survives
    keep = torch.rand(b, c, device=dev) >= A["drop_frac"]
    fallback = F.one_hot(torch.randint(0, c, (b,), device=dev), c).bool()
    keep = keep | (fallback & ~keep.any(dim=1, keepdim=True))
    keep = keep.view(b, c, 1, 1)
    fill = x.mean(dim=(2, 3), keepdim=True) if relative else 0.0     # dropped channel -> mean (relative) or 0
    x = torch.where(_coin(b, A["drop_p"], dev), torch.where(keep, x, fill), x)
    # additive gaussian noise
    if A["noise_p"] > 0:
        std = torch.empty(b, 1, 1, 1, device=dev).uniform_(*A["noise_std"]) * ref
        x = torch.where(_coin(b, A["noise_p"], dev), x + torch.randn_like(x) * std, x)
    # gaussian blur
    if A["blur_p"] > 0:
        ksize = A["blur_ksize"] or (int(0.1 * hgt) // 2 * 2 + 1)
        sigma = torch.empty(b, device=dev).uniform_(*A["blur_sigma"])
        x = torch.where(_coin(b, A["blur_p"], dev), gaussian_blur(x, sigma, ksize), x)
    return x


@torch.no_grad()
def augment_pair(x1, x2, A):
    """One augmented view of a paired batch: shared spatial ops, independent radiometric ops."""
    b, dev = x1.shape[0], x1.device
    x0, y0, w, h = sample_crop_fractions(b, dev, A["crop_scale"], A["crop_ratio"])
    do = _coin(b, A["crop_p"], dev)
    x1 = torch.where(do, crop_resize(x1, x0, y0, w, h), x1)
    x2 = torch.where(do, crop_resize(x2, x0, y0, w, h), x2)
    for p, dim in ((A["hflip_p"], -1), (A["vflip_p"], -2)):
        if p > 0:
            do = _coin(b, p, dev)
            x1, x2 = torch.where(do, x1.flip(dim), x1), torch.where(do, x2.flip(dim), x2)
    return radiometric_aug(x1, A), radiometric_aug(x2, A)


def make_views(x1, x2, views, A):
    if views == "two_aug":                      # SimCLR: two independent augmentations
        x1a, x2a = augment_pair(x1, x2, A)
        x1b, x2b = augment_pair(x1, x2, A)
        return x1a, x2a, x1b, x2b
    x1b, x2b = augment_pair(x1, x2, A)          # original: clean view + one augmented view
    return x1, x2, x1b, x2b


# =============================================================================
# 4. MODEL  (same parameter names as model.PretrainModel)
# =============================================================================
def make_projector(in_dim=384):
    return nn.Sequential(nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
                         nn.Linear(512, 128), nn.BatchNorm1d(128))


class CheckpointedLayers(nn.Module):
    """Drop-in for nn.TransformerEncoder (same `layers` -> same state_dict keys);
    the first `n_ckpt` layers are recomputed during the backward pass."""

    def __init__(self, layers, n_ckpt):
        super().__init__()
        self.layers = layers
        self.n_ckpt = n_ckpt

    def forward(self, x, mask=None, src_key_padding_mask=None):
        for i, layer in enumerate(self.layers):
            x = checkpoint(layer, x, use_reentrant=False) if (self.training and i < self.n_ckpt) else layer(x)
        return x


class PretrainModel(nn.Module):
    def __init__(self, img_size_m1, img_size_m2, in_chans_m1, in_chans_m2, patch_size=8, ckpt_layers=0,
                 cross_heads=False):
        super().__init__()
        self.modality_1_encoder = ViTEncoder(img_size=img_size_m1, patch_size=patch_size, in_chans=in_chans_m1)
        self.modality_2_encoder = ViTEncoder(img_size=img_size_m2, patch_size=patch_size, in_chans=in_chans_m2)
        self.projector_m1 = make_projector(self.modality_1_encoder.embed_dim)   # SimCLR heads (L1, L2)
        self.projector_m2 = make_projector(self.modality_2_encoder.embed_dim)
        if cross_heads:   # cross-modal heads (L3), same architecture as the SimCLR heads
            self.cross_head_m1 = make_projector(self.modality_1_encoder.embed_dim)
            self.cross_head_m2 = make_projector(self.modality_2_encoder.embed_dim)
        for enc in (self.modality_1_encoder, self.modality_2_encoder):
            assert enc.transformer.norm is None
            enc.transformer = CheckpointedLayers(enc.transformer.layers, ckpt_layers)

    def forward(self, x1, x2):
        """Same signature/outputs as model.PretrainModel.forward."""
        h1, h2 = self.modality_1_encoder(x1), self.modality_2_encoder(x2)
        return h1, h2, self.projector_m1(h1), self.projector_m2(h2)


# =============================================================================
# 5. LOSSES
# =============================================================================
def nt_xent(a, b, temperature):
    """NT-Xent over 2N embeddings, positive pairs (a_i, b_i). Identical to
    functions.NTXentLoss(cat[a, b], arange(N).repeat(2), temperature)."""
    z = F.normalize(torch.cat([a, b], dim=0).float(), dim=1)
    n = a.shape[0]
    sim = (z @ z.T) / temperature
    sim = sim.masked_fill(torch.eye(2 * n, device=z.device, dtype=torch.bool), float("-inf"))
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


def shared_specific_corr(h, k, eps=1e-6):
    """Cross-correlation matrix (k x (D-k)) over the batch between the shared
    dims h[:, :k] and the specific dims h[:, k:]. Also returns the mean
    per-dimension std of each part (collapse diagnostic)."""
    h = h.float()
    s, p = h[:, :k], h[:, k:]
    std_s, std_p = s.std(dim=0, unbiased=False), p.std(dim=0, unbiased=False)
    s = (s - s.mean(dim=0)) / (std_s + eps)
    p = (p - p.mean(dim=0)) / (std_p + eps)
    c = s.T @ p / h.shape[0]
    return c, std_s.mean(), std_p.mean()


def compute_losses(model, x1a, x2a, x1b, x2b, *, shared_dim, temp_intra, temp_cross, use_cross, views,
                   decorr_weight=0.0, cross_mode="shared_dims", intra_weight=0.5):
    """views='two_aug'  : heads' BN over both views, cross loss averaged over both views (SimCLR)
       views='clean_aug': (a = clean) heads per view, cross loss on the clean view (pretrain.py)
       cross_mode='heads'      : L3 on cross_head_m1(h1) vs cross_head_m2(h2)
       cross_mode='shared_dims': L3 on h1[:, :k] vs h2[:, :k]  (pretrain.py)
       total = intra_weight * (L1 + L2) + L3 (+ decorr_weight * L_decorr)"""
    n = x1a.shape[0]
    h1a, h1b = model.modality_1_encoder(torch.cat([x1a, x1b])).split(n)
    h2a, h2b = model.modality_2_encoder(torch.cat([x2a, x2b])).split(n)

    def head(f, ha, hb):
        if views == "two_aug":
            return f(torch.cat([ha, hb])).split(n)
        return f(ha), f(hb)

    z1a, z1b = head(model.projector_m1, h1a, h1b)       # L1 / L2: SimCLR within each modality
    z2a, z2b = head(model.projector_m2, h2a, h2b)
    loss_m1 = nt_xent(z1a, z1b, temp_intra)
    loss_m2 = nt_xent(z2a, z2b, temp_intra)

    k = shared_dim                                       # L3: cross-modal, positive = (m1, m2) of a sample
    if not use_cross:
        loss_cross = torch.zeros((), device=x1a.device)
    else:
        if cross_mode == "heads":
            c1a, c1b = head(model.cross_head_m1, h1a, h1b)
            c2a, c2b = head(model.cross_head_m2, h2a, h2b)
        else:
            c1a, c1b, c2a, c2b = h1a[:, :k], h1b[:, :k], h2a[:, :k], h2b[:, :k]
        if views == "two_aug":
            loss_cross = 0.5 * (nt_xent(c1a, c2a, temp_cross) + nt_xent(c1b, c2b, temp_cross))
        else:
            loss_cross = nt_xent(c1a, c2a, temp_cross)

    # shared/specific decorrelation, per modality and per view (+ diagnostics)
    corr = [shared_specific_corr(h, k) for h in (h1a, h1b, h2a, h2b)]
    loss_decorr = torch.stack([(c ** 2).mean() for c, _, _ in corr]).mean()
    abs_c = torch.stack([c.abs().mean() for c, _, _ in corr]).mean()
    std_sh = torch.stack([s for _, s, _ in corr]).mean()
    std_sp = torch.stack([s for _, _, s in corr]).mean()

    total = intra_weight * (loss_m1 + loss_m2) + loss_cross
    if decorr_weight > 0:
        total = total + decorr_weight * loss_decorr
    return total, torch.stack([loss_m1, loss_m2, loss_cross, loss_decorr, abs_c, std_sh, std_sp]).detach()


# =============================================================================
# 6. OPTIMISER, LR SCHEDULE, CHECKPOINTS
# =============================================================================
def param_groups(model, weight_decay, decay_all):
    """No weight decay on biases, norm weights (all 1-d params), pos_embed and cls_token."""
    if decay_all:
        return [{"params": list(model.parameters()), "weight_decay": weight_decay}]
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim <= 1 or name.endswith(("pos_embed", "cls_token")) else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def lr_at(step, args, total_steps):
    """Linear warm-up from 0 to args.lr, then cosine decay to args.min_lr (or constant)."""
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    if args.schedule == "constant":
        return args.lr
    t = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * min(t, 1.0)))


def atomic_save(obj, path):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def save_backbones(model, path):
    """Same format as pretrain.save_pretrained_encoders."""
    atomic_save({"modality_1": model.modality_1_encoder.state_dict(),
                 "modality_2": model.modality_2_encoder.state_dict()}, path)


# =============================================================================
# 7. TRAINING
# =============================================================================
def train(args):
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = args.out_dir or os.path.join(args.dataset, "PRETRAIN_V2")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "checkpoint_latest.pth")
    backbone_path = os.path.join(out_dir, "pretrained_backbones.pth")

    # ---- data ----
    t0 = time.time()
    x1, x2 = load_train_data(args.dataset, args.mod1, args.mod2)
    if args.standardize:   # per-channel statistics of the train split (saved in the checkpoint args)
        stats = {}
        for name, x in ((args.mod1, x1), (args.mod2, x2)):
            mean = x.double().mean(dim=(0, 2, 3))
            std = x.double().std(dim=(0, 2, 3))
            x.sub_(mean.float().view(1, -1, 1, 1)).div_(std.float().view(1, -1, 1, 1))
            stats[name] = {"mean": mean.tolist(), "std": std.tolist()}
            print(f"standardised {name}: mean {[round(v, 4) for v in mean.tolist()]} "
                  f"std {[round(v, 4) for v in std.tolist()]}")
        args.standardize_stats = stats
    x1, x2, data_dev, nbytes = place_data(x1, x2, args.data_device, device)
    n = x1.shape[0]
    steps_per_epoch = n // args.batch_size
    assert steps_per_epoch > 0, f"batch_size {args.batch_size} > training set size {n}"
    total_steps = args.total_steps or args.epochs * steps_per_epoch
    if args.warmup_steps is None:
        args.warmup_steps = int(round(args.warmup_frac * total_steps))
    print(f"data: {args.mod1} {tuple(x1.shape)} {args.mod2} {tuple(x2.shape)} "
          f"({nbytes / 1e9:.2f} GB on {data_dev}, {time.time() - t0:.0f}s)")
    print(f"train samples {n} | batch {args.batch_size} | {steps_per_epoch} steps/epoch | "
          f"{total_steps} steps total (~{total_steps / steps_per_epoch:.0f} epochs)")

    # ---- model / optimiser ----
    model = PretrainModel(x1.shape[-1], x2.shape[-1], x1.shape[1], x2.shape[1],
                          patch_size=args.patch_size, ckpt_layers=args.ckpt_layers,
                          cross_heads=(args.cross_mode == "heads")).to(device)
    optimizer = torch.optim.AdamW(param_groups(model, args.weight_decay, args.decay_all),
                                  lr=args.lr, fused=True)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[args.amp]
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp == "fp16"))
    shared_dim = int(model.modality_1_encoder.embed_dim * args.shared_pct / 100)
    A = dict(AUGS[args.aug], relative=(args.aug_scale == "relative"))

    step = 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if args.amp == "fp16" and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        torch.set_rng_state(ckpt["rng_cpu"])
        torch.cuda.set_rng_state(ckpt["rng_cuda"])
        step = ckpt["step"]
        print(f"resumed from {ckpt_path} at step {step}")

    loss_fn = compute_losses if args.no_compile else torch.compile(compute_losses)
    loss_kw = dict(shared_dim=shared_dim, temp_intra=args.temp_intra, temp_cross=args.temp_cross,
                   use_cross=not args.no_cross, views=args.views, decorr_weight=args.decorr_weight,
                   cross_mode=args.cross_mode, intra_weight=args.intra_weight)

    print(f"recipe={args.recipe} | views={args.views} aug={args.aug} aug_scale={args.aug_scale} "
          f"standardize={args.standardize} | T_intra={args.temp_intra} "
          f"T_cross={args.temp_cross} cross={'off' if args.no_cross else ('on (cross-modal heads)' if args.cross_mode == 'heads' else f'on (first {shared_dim} dims)')} "
          f"total={args.intra_weight}*(L1+L2)+L3 "
          f"decorr={args.decorr_weight if args.decorr_weight > 0 else 'off'} | "
          f"peak_lr={args.lr:.2e} {args.schedule} warmup={args.warmup_steps} wd={args.weight_decay} | out={out_dir}")
    print("args: " + " ".join(f"{k}={v}" for k, v in vars(args).items()))
    sys.stdout.flush()

    def save(step, epoch):
        atomic_save({"step": step, "epoch": epoch,
                     "model_state_dict": model.state_dict(),
                     "optimizer_state_dict": optimizer.state_dict(),
                     "scaler_state_dict": scaler.state_dict(),
                     "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(),
                     "args": vars(args)}, ckpt_path)

    # ---- loop ----
    model.train()
    log_sum = torch.zeros(8, device=device)        # total, m1, m2, cross, decorr, |C|, std_sh, std_sp
    log_steps, log_t0, run_steps = 0, time.time(), 0
    torch.cuda.reset_peak_memory_stats()
    stream = batches(x1, x2, args.batch_size, device)

    while step < total_steps:
        epoch, b1, b2 = next(stream)
        lr = lr_at(step, args, total_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x1a, x2a, x1b, x2b = make_views(b1, b2, args.views, A)
        with torch.autocast("cuda", dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            loss, parts = loss_fn(model, x1a, x2a, x1b, x2b, **loss_kw)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        scaler.step(optimizer)
        scaler.update()

        log_sum += torch.cat([loss.detach().float().view(1), parts.float()])
        log_steps += 1
        step += 1
        run_steps += 1
        last = step >= total_steps or (args.max_steps and run_steps >= args.max_steps)

        if step % args.log_every == 0 or last:
            m = (log_sum / log_steps).tolist()      # single GPU->CPU sync
            dt = time.time() - log_t0
            print(f"step {step}/{total_steps} ep {epoch} lr {lr:.2e} | total={m[0]:.4f} m1={m[1]:.4f} "
                  f"m2={m[2]:.4f} cross={m[3]:.4f} | "
                  + (f"decorr={m[4]:.4f} |C|={m[5]:.3f} " if args.cross_mode == "shared_dims" else "")
                  + f"std_sh={m[6]:.3f} std_sp={m[7]:.3f} | {dt / log_steps * 1000:.0f} ms/step "
                  f"{log_steps * args.batch_size / dt:.0f} samples/s "
                  f"peak_mem={torch.cuda.max_memory_allocated() / 2**30:.1f}GB")
            sys.stdout.flush()
            log_sum.zero_()
            log_steps, log_t0 = 0, time.time()
        if step % args.save_every == 0 or last:
            save(step, epoch)
        if step % args.backbone_every == 0 or last:
            save_backbones(model, backbone_path)
        if last:
            break

    print(f"done at step {step} -- {ckpt_path}")


# =============================================================================
# 8. SELF-CHECK  (python pretrain_v2.py --self_check)
# =============================================================================
def _reference_ntxent(embeddings, labels, temperature=1.0):
    """Verbatim copy of functions.NTXentLoss (functions.py needs scikit-image,
    so it is not imported here)."""
    embeddings = embeddings.float()
    similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    batch_size = embeddings.shape[0]
    diagonal_mask = torch.eye(batch_size, device=embeddings.device, dtype=torch.bool)
    positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    positive_mask.fill_diagonal_(0)
    similarity_matrix = similarity_matrix - similarity_matrix.max(dim=1, keepdim=True).values.detach()
    similarity_matrix_masked = similarity_matrix.masked_fill(diagonal_mask, float('-inf'))
    log_prob = F.log_softmax(similarity_matrix_masked, dim=1)
    log_prob = torch.nan_to_num(log_prob, nan=0.0, posinf=0.0, neginf=0.0)
    loss = -(log_prob * positive_mask).sum(dim=1) / (positive_mask.sum(dim=1) + 1e-8)
    return loss.mean()


def self_check():
    import model as original
    from kornia.geometry.transform import crop_and_resize
    from kornia.filters import gaussian_blur2d

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False   # exact fp32 for the comparisons
    torch.backends.cudnn.allow_tf32 = False
    ok = True

    def report(name, a, b, tol, relative=False):
        nonlocal ok
        err = (a - b).abs().max().item()
        if relative:
            err /= b.abs().max().item()
        ok &= err < tol
        print(f"[{'OK  ' if err < tol else 'FAIL'}] {name:<58s} max|diff| = {err:.2e}")

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'OK  ' if cond else 'FAIL'}] {name}")

    # (a) NT-Xent == functions.NTXentLoss
    a, b = torch.randn(64, 128, device=dev), torch.randn(64, 128, device=dev)
    lab = torch.arange(64, device=dev).repeat(2)
    for T in (1.0, 0.2, 0.1):
        report(f"nt_xent vs functions.NTXentLoss (T={T})", nt_xent(a, b, T),
               _reference_ntxent(F.normalize(torch.cat([a, b]), dim=1), lab, T), 1e-4)

    # (b) crop == kornia.crop_and_resize with the box convention of functions.py
    x = torch.randn(16, 3, 40, 40, device=dev)
    x0, y0, w, h = sample_crop_fractions(16, dev, (0.08, 1.0), (3 / 4, 4 / 3))
    H = W = 40
    boxes = torch.stack([torch.stack([x0 * (W - 1), y0 * (H - 1)], 1),
                         torch.stack([(x0 + w) * (W - 1), y0 * (H - 1)], 1),
                         torch.stack([(x0 + w) * (W - 1), (y0 + h) * (H - 1)], 1),
                         torch.stack([x0 * (W - 1), (y0 + h) * (H - 1)], 1)], dim=1)
    report("crop_resize vs kornia.crop_and_resize", crop_resize(x, x0, y0, w, h),
           crop_and_resize(x, boxes, size=(H, W)), 1e-4)

    # (c) blur == kornia.gaussian_blur2d (per-sample sigma, reflect)
    sigma = torch.empty(16, device=dev).uniform_(0.1, 2.0)
    for k in (3, 7):
        report(f"gaussian_blur {k}x{k} vs kornia.gaussian_blur2d", gaussian_blur(x, sigma, k),
               gaussian_blur2d(x, (k, k), torch.stack([sigma, sigma], 1)), 1e-5)

    # (d) recipe 'original' == pretrain.py (same weights & batch)
    new = PretrainModel(32, 32, 3, 1, patch_size=8, ckpt_layers=6).to(dev)
    sffc = original.SFFCConfig(img_size_m1=32, img_size_m2=32, patch_size_m1=8, patch_size_m2=8,
                               in_chans_m1=3, in_chans_m2=1, num_classes=1)
    old = original.PretrainModel(sffc).to(dev)
    with torch.no_grad():   # materialise LazyLinear
        old(torch.zeros(2, 3, 32, 32, device=dev), torch.zeros(2, 1, 32, 32, device=dev))
    old.load_state_dict(new.state_dict(), strict=True)
    check(f"state_dict keys identical to model.PretrainModel ({len(new.state_dict())} tensors, strict load)", True)

    for m in (new, old):   # dropout off (stochastic); BatchNorm in train mode as during training
        m.train()
        m.modality_1_encoder.eval()
        m.modality_2_encoder.eval()
    x1, x2 = torch.randn(32, 3, 32, 32, device=dev), torch.randn(32, 1, 32, 32, device=dev)
    _, _, x1a, x2a = make_views(x1, x2, "clean_aug", AUG_ORIGINAL)
    loss_new, _ = compute_losses(new, x1, x2, x1a, x2a, shared_dim=192, temp_intra=1.0, temp_cross=1.0,
                                 use_cross=True, views="clean_aug")
    c1, c2, p1, p2 = old(x1, x2)                     # --- pretrain.py, verbatim logic ---
    _, _, p1a, p2a = old(x1a, x2a)
    emb_inv = F.normalize(torch.cat([c1[:, :192], c2[:, :192]], dim=0), dim=1)
    r1 = F.normalize(torch.cat([p1, p1a], dim=0), dim=1)
    r2 = F.normalize(torch.cat([p2, p2a], dim=0), dim=1)
    lab = torch.arange(32, device=dev).repeat(2)
    loss_old = .5 * (_reference_ntxent(r1, lab) + _reference_ntxent(r2, lab)) + _reference_ntxent(emb_inv, lab)
    report("recipe 'original': loss vs pretrain.py", loss_new, loss_old, 1e-4)
    loss_new.backward()
    loss_old.backward()
    report("recipe 'original': gradients vs pretrain.py (rel.)",
           torch.cat([p.grad.flatten() for p in new.parameters()]),
           torch.cat([p.grad.flatten() for p in old.parameters()]), 1e-4, relative=True)

    # (e) two views, cross loss on the shared dims (previous 'standard'): loss written out by hand
    new.zero_grad()
    x1a, x2a, x1b, x2b = make_views(x1, x2, "two_aug", AUG_STANDARD)
    loss_new, _ = compute_losses(new, x1a, x2a, x1b, x2b, shared_dim=192, temp_intra=0.2, temp_cross=0.1,
                                 use_cross=True, views="two_aug", cross_mode="shared_dims", intra_weight=0.5)
    h1a, h2a, _, _ = new(x1a, x2a)
    h1b, h2b, _, _ = new(x1b, x2b)
    z1 = new.projector_m1(torch.cat([h1a, h1b]))      # BN over both views
    z2 = new.projector_m2(torch.cat([h2a, h2b]))
    ref = (0.5 * (_reference_ntxent(F.normalize(z1, dim=1), lab, 0.2) +
                  _reference_ntxent(F.normalize(z2, dim=1), lab, 0.2)) +
           0.5 * (_reference_ntxent(F.normalize(torch.cat([h1a[:, :192], h2a[:, :192]]), dim=1), lab, 0.1) +
                  _reference_ntxent(F.normalize(torch.cat([h1b[:, :192], h2b[:, :192]]), dim=1), lab, 0.1)))
    report("two views + shared_dims cross loss vs explicit formula", loss_new, ref, 1e-4)

    # (e1) recipe 'standard': L1 + L2 + L3 with cross-modal heads, written out by hand
    hm = PretrainModel(32, 32, 3, 1, patch_size=8, ckpt_layers=6, cross_heads=True).to(dev)
    hm.train(); hm.modality_1_encoder.eval(); hm.modality_2_encoder.eval()
    loss_h, _ = compute_losses(hm, x1a, x2a, x1b, x2b, shared_dim=192, temp_intra=0.2, temp_cross=0.1,
                               use_cross=True, views="two_aug", cross_mode="heads", intra_weight=1.0)
    g1a, g2a, _, _ = hm(x1a, x2a)
    g1b, g2b, _, _ = hm(x1b, x2b)
    q1 = hm.projector_m1(torch.cat([g1a, g1b])); q2 = hm.projector_m2(torch.cat([g2a, g2b]))
    k1a, k1b = hm.cross_head_m1(torch.cat([g1a, g1b])).split(32)
    k2a, k2b = hm.cross_head_m2(torch.cat([g2a, g2b])).split(32)
    ref_h = (_reference_ntxent(F.normalize(q1, dim=1), lab, 0.2) + _reference_ntxent(F.normalize(q2, dim=1), lab, 0.2)
             + 0.5 * (_reference_ntxent(F.normalize(torch.cat([k1a, k2a]), dim=1), lab, 0.1) +
                      _reference_ntxent(F.normalize(torch.cat([k1b, k2b]), dim=1), lab, 0.1)))
    report("recipe 'standard': L1+L2+L3 (cross heads) vs explicit formula", loss_h, ref_h, 1e-4)
    extra = sorted({k.split('.')[0] for k in hm.state_dict()} - {k.split('.')[0] for k in old.state_dict()})
    check(f"recipe 'standard': only extra modules vs pretrain.py are {extra}", extra == ["cross_head_m1", "cross_head_m2"])

    # (e2) decorrelation loss vs torch.corrcoef
    loss_d, parts = compute_losses(new, x1a, x2a, x1b, x2b, shared_dim=192, temp_intra=0.2, temp_cross=0.1,
                                   use_cross=True, views="two_aug", decorr_weight=1.0,
                                   cross_mode="shared_dims", intra_weight=0.5)
    ref_d = torch.stack([(torch.corrcoef(h.T)[:192, 192:] ** 2).mean() for h in (h1a, h1b, h2a, h2b)]).mean()
    report("decorrelation term vs torch.corrcoef", parts[3], ref_d, 1e-4)
    report("total loss with decorr_weight=1 = base + decorr", loss_d, loss_new + ref_d, 1e-4)

    # (f) augmentation: spatial part shared by both modalities, views independent
    xa = torch.randn(256, 1, 32, 32, device=dev)
    for name, A in AUGS.items():
        A_spatial = dict(A, shift_p=0.0, drop_p=0.0, noise_p=0.0, blur_p=0.0)
        y1, y2 = augment_pair(xa, xa.clone(), A_spatial)
        report(f"aug '{name}': crop/flips shared by the two modalities", y1, y2, 1e-6)
    v = make_views(xa, xa.clone(), "two_aug", AUG_STANDARD)
    check("aug 'standard': the two views differ", (v[0] - v[2]).abs().max().item() > 0.1)

    # (f2) relative augmentation is scale-equivariant: aug(k * x) == k * aug(x) (same random draws)
    A_rel = dict(AUG_STANDARD, relative=True, noise_p=0.5)
    xr = torch.randn(64, 3, 32, 32, device=dev) * 0.07 + 0.3
    torch.manual_seed(123)
    y_small, _ = augment_pair(xr, xr.clone(), A_rel)
    torch.manual_seed(123)
    y_big, _ = augment_pair(xr * 10, xr.clone() * 10, A_rel)
    report("aug_scale 'relative': aug(10 x) == 10 aug(x)", y_big, 10 * y_small, 1e-3)

    # (g) lr schedule
    sargs = argparse.Namespace(lr=6e-4, min_lr=1e-6, warmup_steps=10_000, schedule="cosine")
    check("lr schedule: 0 -> peak at end of warm-up -> min_lr at the end",
          lr_at(0, sargs, 100_000) < 1e-7 and abs(lr_at(9_999, sargs, 100_000) - 6e-4) < 1e-12
          and abs(lr_at(55_000, sargs, 100_000) - (1e-6 + 0.5 * (6e-4 - 1e-6))) < 1e-9
          and abs(lr_at(100_000, sargs, 100_000) - 1e-6) < 1e-12)

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    args = parse_args()
    if args.self_check:
        sys.exit(0 if self_check() else 1)
    train(args)
