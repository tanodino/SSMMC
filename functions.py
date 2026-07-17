from typing import Sequence
import random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
from collections import OrderedDict
import numpy as np
import math
import torch.nn.functional as F
import torchvision.transforms as T 
import torchvision.transforms.functional as TF
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader
import kornia.augmentation as K
import kornia.filters as KF
from kornia.augmentation import AugmentationSequential

from kornia.geometry.transform import crop_and_resize


from typing import Sequence
import random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
from collections import OrderedDict
import numpy as np
import math
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader
import kornia.augmentation as K
import kornia.filters as KF
from kornia.augmentation import AugmentationSequential

from kornia.geometry.transform import crop_and_resize

'''
def _select_by_choice(options: list, choice: torch.Tensor) -> torch.Tensor:
    """
    Pick one tensor per sample out of `options` (a list of K same-shaped
    (B, C, H, W) tensors), according to `choice` (a (B,) int64 tensor of
    indices in [0, K)). Sample b's output is options[choice[b]][b].

    This computes every candidate for the whole batch, then selects --
    simpler and more vectorized than branching per sample, at the cost of
    some wasted compute on the discarded candidates (fine at this batch
    size / augmentation cost).
    """
    stacked = torch.stack(options, dim=0)  # (K, B, C, H, W)
    batch_idx = torch.arange(choice.shape[0], device=choice.device)
    return stacked[choice, batch_idx]


def _transpose(x: torch.Tensor) -> torch.Tensor:
    """Swap H/W. Requires square spatial dims."""
    assert x.shape[-1] == x.shape[-2], "transpose augmentation requires square H == W"
    return x.transpose(-1, -2)


def _rot90(x: torch.Tensor, k: int) -> torch.Tensor:
    """Rotate by k*90 degrees. Requires square spatial dims (same constraint as transpose),
    since k=1/3 swap H and W."""
    assert x.shape[-1] == x.shape[-2], "rot90 augmentation requires square H == W"
    return torch.rot90(x, k, dims=[-2, -1])


def weak_augment_pair(x1: torch.Tensor, x2: torch.Tensor) -> tuple:
    """
    Sample ONE weak op per sample -- horizontal flip, vertical flip, transpose, or a
    90/180/270 degree rotation -- and apply it to both modalities. The same op type is
    used for both, so spatial alignment is preserved.
    """
    batch_size = x1.shape[0]

    x1_options = [
        torch.flip(x1, dims=[-1]),  # hflip
        torch.flip(x1, dims=[-2]),  # vflip
        _transpose(x1),             # transpose
        _rot90(x1, 1),               # rot90
        _rot90(x1, 2),               # rot180
        _rot90(x1, 3),               # rot270
    ]
    x2_options = [
        torch.flip(x2, dims=[-1]),
        torch.flip(x2, dims=[-2]),
        _transpose(x2),
        _rot90(x2, 1),
        _rot90(x2, 2),
        _rot90(x2, 3),
    ]
    choice = torch.randint(0, len(x1_options), (batch_size,), device=x1.device)
    return _select_by_choice(x1_options, choice), _select_by_choice(x2_options, choice)


def _sample_crop_fractions(
    batch_size: int,
    device: torch.device,
    scale: tuple = (0.5, 1.0),
    ratio: tuple = (3 / 4, 4 / 3),
) -> tuple:
    """Random-resized-crop region as fractions of the image, in [0, 1].
    Fractions (not pixels) so the SAME region can be applied to modalities
    with different spatial resolutions."""
    area_frac = torch.empty(batch_size, device=device).uniform_(*scale)
    log_ratio = torch.empty(batch_size, device=device).uniform_(math.log(ratio[0]), math.log(ratio[1]))
    aspect = torch.exp(log_ratio)

    w_frac = torch.sqrt(area_frac * aspect).clamp(max=1.0)
    h_frac = torch.sqrt(area_frac / aspect).clamp(max=1.0)

    x0_frac = torch.rand(batch_size, device=device) * (1.0 - w_frac)
    y0_frac = torch.rand(batch_size, device=device) * (1.0 - h_frac)
    return x0_frac, y0_frac, w_frac, h_frac


def _fractions_to_boxes(x0_f, y0_f, w_f, h_f, height: int, width: int) -> torch.Tensor:
    """Convert normalized crop fractions to kornia's expected box format:
    (B, 4, 2) corners in (x, y) pixel coords, clockwise from top-left."""
    x0 = x0_f * (width - 1)
    y0 = y0_f * (height - 1)
    x1 = (x0_f + w_f) * (width - 1)
    y1 = (y0_f + h_f) * (height - 1)
    return torch.stack(
        [
            torch.stack([x0, y0], dim=1),
            torch.stack([x1, y0], dim=1),
            torch.stack([x1, y1], dim=1),
            torch.stack([x0, y1], dim=1),
        ],
        dim=1,
    )


def _shared_random_resized_crop(x1: torch.Tensor, x2: torch.Tensor) -> tuple:
    """Same relative crop region applied to both modalities, each resized
    back to its own native H/W."""
    x0_f, y0_f, w_f, h_f = _sample_crop_fractions(x1.shape[0], x1.device)

    h1, w1 = x1.shape[-2], x1.shape[-1]
    h2, w2 = x2.shape[-2], x2.shape[-1]

    boxes1 = _fractions_to_boxes(x0_f, y0_f, w_f, h_f, h1, w1)
    boxes2 = _fractions_to_boxes(x0_f, y0_f, w_f, h_f, h2, w2)

    x1 = crop_and_resize(x1, boxes1, size=(h1, w1))
    x2 = crop_and_resize(x2, boxes2, size=(h2, w2))
    return x1, x2


def _random_gaussian_blur(kernel_size: int = 5, sigma_range: tuple = (0.1, 2.0)) -> K.RandomGaussianBlur:
    # same_on_batch=False (default) samples a fresh sigma per SAMPLE per call,
    # already channel-agnostic since kornia blurs each channel independently.
    return K.RandomGaussianBlur((kernel_size, kernel_size), sigma_range, p=1.0)


_blur_m1 = _random_gaussian_blur()
_blur_m2 = _random_gaussian_blur()


def _channel_mean_grayscale(x: torch.Tensor) -> torch.Tensor:
    """Generalization of 'grayscale' to arbitrary channel counts: collapse to
    a per-pixel mean across channels, then broadcast back to the original
    channel count. Kornia's RandomGrayscale assumes 3-channel RGB, so it
    doesn't apply here.

    NOTE: for single-channel modalities (your Depth/Thermal) this is a no-op
    (mean of 1 channel = itself) -- harmless, but it means the effective
    strong-augmentation pool for those modalities is smaller than for
    multi-channel ones (dual-pol SAR, multispectral).
    """
    mean = x.mean(dim=1, keepdim=True)
    return mean.expand_as(x)


def _random_brightness(x: torch.Tensor, magnitude: float = 0.2) -> torch.Tensor:
    """Additive brightness jitter: x + delta, delta ~ U(-magnitude, magnitude), sampled
    per sample (not per channel, so relative structure across channels/bands is preserved).

    magnitude is in the same units as your (already normalized/relative) input, so keep it
    conservative -- default 0.2 assumes roughly unit-scale normalized data. Tune per modality
    if SAR/Depth/Thermal end up on different normalized scales.
    """
    b = x.shape[0]
    delta = (torch.rand(b, 1, 1, 1, device=x.device) * 2 - 1) * magnitude
    return x + delta


def _random_contrast(x: torch.Tensor, magnitude: tuple = (0.8, 1.2)) -> torch.Tensor:
    """Contrast jitter: scale each sample around its own per-channel spatial mean,
    (x - mean) * factor + mean, factor ~ U(*magnitude) sampled per sample.

    Per-channel mean (not per-pixel or global) keeps this a plausible radiometric/gain
    perturbation rather than a spatial distortion.
    """
    b = x.shape[0]
    factor = torch.empty(b, 1, 1, 1, device=x.device).uniform_(*magnitude)
    mean = x.mean(dim=[-2, -1], keepdim=True)  # per-sample, per-channel mean
    return (x - mean) * factor + mean


def strong_augment_pair(
    x1: torch.Tensor,
    x2: torch.Tensor,
    brightness_magnitude: float = 0.2,
    contrast_magnitude: tuple = (0.8, 1.2),
) -> tuple:
    """
    Sample ONE strong op per sample -- random-resized-crop, gaussian blur, 'grayscale'
    (channel-mean), brightness jitter, or contrast jitter -- and apply it to both
    modalities. The same op *type* is shared across modalities; crop uses a shared
    spatial region (alignment), while blur/grayscale/brightness/contrast are computed
    independently per modality even though the choice of which op is shared.
    """
    batch_size = x1.shape[0]
    choice = torch.randint(0, 5, (batch_size,), device=x1.device)

    cropped_x1, cropped_x2 = _shared_random_resized_crop(x1, x2)

    x1_options = [
        cropped_x1,
        _blur_m1(x1),
        _channel_mean_grayscale(x1),
        _random_brightness(x1, brightness_magnitude),
        _random_contrast(x1, contrast_magnitude),
    ]
    x2_options = [
        cropped_x2,
        _blur_m2(x2),
        _channel_mean_grayscale(x2),
        _random_brightness(x2, brightness_magnitude),
        _random_contrast(x2, contrast_magnitude),
    ]

    return _select_by_choice(x1_options, choice), _select_by_choice(x2_options, choice)



'''
def _select_by_choice(options: list, choice: torch.Tensor) -> torch.Tensor:
    """
    Pick one tensor per sample out of `options` (a list of K same-shaped
    (B, C, H, W) tensors), according to `choice` (a (B,) int64 tensor of
    indices in [0, K)). Sample b's output is options[choice[b]][b].

    This computes every candidate for the whole batch, then selects --
    simpler and more vectorized than branching per sample, at the cost of
    some wasted compute on the discarded candidates (fine at this batch
    size / augmentation cost).
    """
    stacked = torch.stack(options, dim=0)  # (K, B, C, H, W)
    batch_idx = torch.arange(choice.shape[0], device=choice.device)
    return stacked[choice, batch_idx]


def _transpose(x: torch.Tensor) -> torch.Tensor:
    """Swap H/W. Requires square spatial dims."""
    assert x.shape[-1] == x.shape[-2], "transpose augmentation requires square H == W"
    return x.transpose(-1, -2)


def weak_augment_pair(x1: torch.Tensor, x2: torch.Tensor) -> tuple:
    """
    Sample ONE weak op per sample -- horizontal flip, vertical flip, or
    transpose -- and apply it to both modalities. The same op type is used
    for both, so spatial alignment is preserved.
    """
    batch_size = x1.shape[0]
    choice = torch.randint(0, 3, (batch_size,), device=x1.device)

    x1_options = [torch.flip(x1, dims=[-1]), torch.flip(x1, dims=[-2]), _transpose(x1)]
    x2_options = [torch.flip(x2, dims=[-1]), torch.flip(x2, dims=[-2]), _transpose(x2)]

    return _select_by_choice(x1_options, choice), _select_by_choice(x2_options, choice)


def _sample_crop_fractions(
    batch_size: int,
    device: torch.device,
    scale: tuple = (0.5, 1.0),
    ratio: tuple = (3 / 4, 4 / 3),
) -> tuple:
    """Random-resized-crop region as fractions of the image, in [0, 1].
    Fractions (not pixels) so the SAME region can be applied to modalities
    with different spatial resolutions."""
    area_frac = torch.empty(batch_size, device=device).uniform_(*scale)
    log_ratio = torch.empty(batch_size, device=device).uniform_(math.log(ratio[0]), math.log(ratio[1]))
    aspect = torch.exp(log_ratio)

    w_frac = torch.sqrt(area_frac * aspect).clamp(max=1.0)
    h_frac = torch.sqrt(area_frac / aspect).clamp(max=1.0)

    x0_frac = torch.rand(batch_size, device=device) * (1.0 - w_frac)
    y0_frac = torch.rand(batch_size, device=device) * (1.0 - h_frac)
    return x0_frac, y0_frac, w_frac, h_frac


def _fractions_to_boxes(x0_f, y0_f, w_f, h_f, height: int, width: int) -> torch.Tensor:
    """Convert normalized crop fractions to kornia's expected box format:
    (B, 4, 2) corners in (x, y) pixel coords, clockwise from top-left."""
    x0 = x0_f * (width - 1)
    y0 = y0_f * (height - 1)
    x1 = (x0_f + w_f) * (width - 1)
    y1 = (y0_f + h_f) * (height - 1)
    return torch.stack(
        [
            torch.stack([x0, y0], dim=1),
            torch.stack([x1, y0], dim=1),
            torch.stack([x1, y1], dim=1),
            torch.stack([x0, y1], dim=1),
        ],
        dim=1,
    )


def _shared_random_resized_crop(x1: torch.Tensor, x2: torch.Tensor) -> tuple:
    """Same relative crop region applied to both modalities, each resized
    back to its own native H/W."""
    x0_f, y0_f, w_f, h_f = _sample_crop_fractions(x1.shape[0], x1.device)

    h1, w1 = x1.shape[-2], x1.shape[-1]
    h2, w2 = x2.shape[-2], x2.shape[-1]

    boxes1 = _fractions_to_boxes(x0_f, y0_f, w_f, h_f, h1, w1)
    boxes2 = _fractions_to_boxes(x0_f, y0_f, w_f, h_f, h2, w2)

    x1 = crop_and_resize(x1, boxes1, size=(h1, w1))
    x2 = crop_and_resize(x2, boxes2, size=(h2, w2))
    return x1, x2


def _random_gaussian_blur(kernel_size: int = 5, sigma_range: tuple = (0.1, 2.0)) -> K.RandomGaussianBlur:
    # same_on_batch=False (default) samples a fresh sigma per SAMPLE per call,
    # already channel-agnostic since kornia blurs each channel independently.
    return K.RandomGaussianBlur((kernel_size, kernel_size), sigma_range, p=1.0)


_blur_m1 = _random_gaussian_blur()
_blur_m2 = _random_gaussian_blur()


def _channel_mean_grayscale(x: torch.Tensor) -> torch.Tensor:
    """Generalization of 'grayscale' to arbitrary channel counts: collapse to
    a per-pixel mean across channels, then broadcast back to the original
    channel count. Kornia's RandomGrayscale assumes 3-channel RGB, so it
    doesn't apply here."""
    mean = x.mean(dim=1, keepdim=True)
    return mean.expand_as(x)


def strong_augment_pair(x1: torch.Tensor, x2: torch.Tensor) -> tuple:
    """
    Sample ONE strong op per sample -- random-resized-crop, gaussian blur, or
    'grayscale' (channel-mean) -- and apply it to both modalities. The same
    op type is used for both: crop uses a shared region (spatial alignment),
    while blur/grayscale are computed independently per modality even though
    the choice of *which* op is shared.
    """
    batch_size = x1.shape[0]
    choice = torch.randint(0, 3, (batch_size,), device=x1.device)

    cropped_x1, cropped_x2 = _shared_random_resized_crop(x1, x2)
    x1_options = [cropped_x1, _blur_m1(x1), _channel_mean_grayscale(x1)]
    x2_options = [cropped_x2, _blur_m2(x2), _channel_mean_grayscale(x2)]

    return _select_by_choice(x1_options, choice), _select_by_choice(x2_options, choice)

###########



TRAIN_BATCH_SIZE = 64
TRAIN_BATCH_SIZE_2 = 32
VALID_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
LEARNING_RATE = 0.0001
MOMENTUM_EMA = .95
EPOCHS = 500
TH_FIXMATCH = .95
WARM_UP_EPOCH_EMA = 30
warmup_epochs = 5
max_epochs = 500
pretraining_max_epochs = 900
pretraining_batch_size = 512
decur_lambd = 0.0051
WARM_UP_EPOCH_SSL = 10
RATIO_LABELED_UNLABELED_BATCHES = 6

import torch
import torch.nn.functional as F

'''
class PairedSentinelDataset(torch.utils.data.Dataset):
    def __init__(self, data_m1, data_m2, augment=True):
        """
        Args:
            data_m1: numpy array (N, C1, H, W) — Sentinel-1 data
            data_m2: numpy array (N, C2, H, W) — Sentinel-2 data
            augment: whether to apply geometric augmentations
        """
        self.data_m1 = torch.tensor(data_m1, dtype=torch.float32)
        self.data_m2 = torch.tensor(data_m2, dtype=torch.float32)
        self.augment = augment

    def __len__(self):
        return len(self.data_m1)

    def __getitem__(self, idx):
        x1 = self.data_m1[idx].clone()  # (C1, H, W)
        x2 = self.data_m2[idx].clone()  # (C2, H, W)

        if self.augment:
            # --- Shared geometric augmentations ---
            # Same operation applied to both modalities
            # preserves spatial alignment between S1 and S2

            # Random 90° rotation — satellite imagery has no canonical orientation
            k = random.randint(0, 3)
            if k > 0:
                x1 = torch.rot90(x1, k, dims=[-2, -1])
                x2 = torch.rot90(x2, k, dims=[-2, -1])

            # Random horizontal flip
            if random.random() < 0.5:
                x1 = torch.flip(x1, dims=[-1])
                x2 = torch.flip(x2, dims=[-1])

            # Random vertical flip
            if random.random() < 0.5:
                x1 = torch.flip(x1, dims=[-2])
                x2 = torch.flip(x2, dims=[-2])

        return x1, x2
'''

def get_maps(model, dataloader, device):
    model.eval()
    tot_pred = []
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                x_batch, y_batch = batch
                x_batch = x_batch.to(device, non_blocking=True)
                logits = model(x_batch)
            elif len(batch) == 3:
                x_batch_m1, x_batch_m2, y_batch = batch
                x_batch_m1 = x_batch_m1.to(device, non_blocking=True)
                x_batch_m2 = x_batch_m2.to(device, non_blocking=True)
                logits = model(x_batch_m1, x_batch_m2)

            preds = logits.argmax(dim=1) 
            tot_pred.append(preds.cpu())
    
    return np.concatenate(tot_pred, axis=0)


###########################################
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

############# DECUR ######################

def bt_loss_cross(z1, z2, pretraining_batch_size, lambd=0.005):
    #EL, D = z1.size()
    # empirical cross-correlation matrix
    #z1 and z2 are alreayd batch normalized with affine=False
    z1 = (z1 - z1.mean(0)) / z1.std(0)  # should already be done
    z2 = (z2 - z2.mean(0)) / z2.std(0)
    c = z1.T @ z2
    c.div_(pretraining_batch_size)
    dim_c = z1.shape[1] // 2
    #dim_c = self.args.dim_common
    c_c = c[:dim_c,:dim_c]
    c_u = c[dim_c:,dim_c:]
    D_c_c = c_c.shape[1]
    D_c_u = c_u.shape[1]


    on_diag_c = torch.diagonal(c_c).add_(-1).pow_(2).sum()
    off_diag_c = off_diagonal(c_c).pow_(2).sum()
    
    on_diag_u = torch.diagonal(c_u).pow_(2).sum()
    off_diag_u = off_diagonal(c_u).pow_(2).sum()
    
    loss_c = on_diag_c + lambd * off_diag_c
    loss_u = on_diag_u + lambd * off_diag_u
    
    return loss_c, on_diag_c, off_diag_c, loss_u, on_diag_u, off_diag_u

'''
#INVARIANCE LOSS / BARLOW TWINS
def bt_loss_single_v2(z1, z2, lambd=0.005):
    N, D = z1.size()
    # empirical cross-correlation matrix
    #z1 and z2 are alreayd batch normalized with affine=False
    z1 = (z1 - z1.mean(0)) / z1.std(0)  # should already be done
    z2 = (z2 - z2.mean(0)) / z2.std(0)
    c = z1.T @ z2

    c.div_(N)

    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum() / D
    off_diag = off_diagonal(c).pow_(2).sum() / (D * (D - 1))
    loss = on_diag + lambd * off_diag
    return loss
'''


def bt_loss_single(z1, z2, batch_size, lambd=0.005):
    _, D = z1.size()
    # empirical cross-correlation matrix
    #z1 and z2 are alreayd batch normalized with affine=False
    z1 = (z1 - z1.mean(0)) / z1.std(0)  # should already be done
    z2 = (z2 - z2.mean(0)) / z2.std(0)
    c = z1.T @ z2

    c.div_(batch_size)

    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    loss = on_diag + lambd * off_diag
    return loss

##########################################



# ─────────────────────────────────────────────
#  TRANSFORMATIONS
# ─────────────────────────────────────────────
'''
def random_brightness(x: np.ndarray, max_delta: float = 0.2) -> np.ndarray:
    """Additive brightness jitter, applied uniformly across all channels."""
    delta = np.random.uniform(-max_delta, max_delta)
    return np.clip(x + delta, 0.0, 1.0)


def random_contrast(x: np.ndarray, factor_range: tuple = (0.8, 1.2)) -> np.ndarray:
    """Per-image contrast scaling around the per-channel mean."""
    factor = np.random.uniform(*factor_range)
    mean = x.mean(axis=(0, 1), keepdims=True)  # (1, 1, C)
    return np.clip((x - mean) * factor + mean, 0.0, 1.0)


def to_gray(x: np.ndarray) -> np.ndarray:
    """Averages all channels into one and broadcasts back to (H, W, C)."""
    gray = x.mean(axis=-1, keepdims=True)      # (H, W, 1)
    return np.broadcast_to(gray, x.shape).copy()


def gaussian_blur(x: np.ndarray, sigma_range: tuple = (0.1, 2.0)) -> np.ndarray:
    """Gaussian blur applied spatially, independently per channel."""
    sigma = np.random.uniform(*sigma_range)
    return gaussian_filter(x, sigma=[sigma, sigma, 0])  # no blur across channels


def random_resized_crop(x: np.ndarray, size: int, scale: tuple = (0.2, 1.0)) -> np.ndarray:
    """Random crop of a scaled region, resized back to (size, size, C)."""
    h, w = x.shape[:2]
    crop_area = np.random.uniform(*scale) * h * w
    aspect_ratio = np.random.uniform(3/4, 4/3)
    crop_h = int(np.round(np.sqrt(crop_area / aspect_ratio)))
    crop_w = int(np.round(np.sqrt(crop_area * aspect_ratio)))
    crop_h = np.clip(crop_h, 1, h)
    crop_w = np.clip(crop_w, 1, w)
    top  = np.random.randint(0, h - crop_h + 1)
    left = np.random.randint(0, w - crop_w + 1)
    cropped = x[top:top + crop_h, left:left + crop_w, :]
    return resize(cropped, (size, size), anti_aliasing=False).astype(np.float32)
'''

# ─────────────────────────────────────────────
#  AUGMENTATION CLASS
# ─────────────────────────────────────────────
class ProbabilisticApply(nn.Module):
    """Applies a module with probability p."""
    def __init__(self, module: nn.Module, p: float = 0.5):
        super().__init__()
        self.module = module
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            return self.module(x)
        return x

class ChannelMeanGrayscale(nn.Module):
    """Channel-agnostic grayscale: replaces all channels with their mean."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1, keepdim=True).expand_as(x)

class AdditiveBrightness(nn.Module):
    def __init__(self, max_delta: float = 0.2):
        super().__init__()
        self.max_delta = max_delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(
            -self.max_delta, self.max_delta
        )
        return (x + delta).clamp(0.0, 1.0)


class AugmentationGPU(nn.Module):
    def __init__(self, image_size: int = 224):
        super().__init__()
        self.spatial = K.AugmentationSequential(
            K.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.2, 1.0),
                ratio=(3/4, 4/3),
                p=1.0,
            ),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
        )
        self.contrast = K.RandomContrast(contrast=(0.8, 1.2), p=0.8)
        self.blur = K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.5)
        self.brightness = ProbabilisticApply(AdditiveBrightness(max_delta=0.2), p=0.8)
        self.grayscale = ProbabilisticApply(ChannelMeanGrayscale(), p=0.2)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial(x)
        x = self.brightness(x)
        x = self.contrast(x)
        x = self.grayscale(x)
        x = self.blur(x)
        return x

    def forward(self, x: torch.Tensor):
        return self._augment(x)


class MSorSARAugmentationGPU(nn.Module):
    def __init__(self, image_size: int = 224):
        super().__init__()
        self.spatial = K.AugmentationSequential(
            K.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.2, 1.0),
                ratio=(3/4, 4/3),
                p=1.0,
            ),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
        )
        self.contrast = K.RandomContrast(contrast=(0.8, 1.2), p=0.8)
        self.blur = K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.5)
        self.brightness = ProbabilisticApply(AdditiveBrightness(max_delta=0.2), p=0.8)
        self.grayscale = ProbabilisticApply(ChannelMeanGrayscale(), p=0.2)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial(x)
        x = self.brightness(x)
        x = self.contrast(x)
        x = self.grayscale(x)
        x = self.blur(x)
        return x

    def forward(self, x: torch.Tensor):
        return self._augment(x), self._augment(x)

class ContrastiveDatasetGPU(Dataset):
    """
    Wraps a numpy array of images for SimCLR contrastive pre-training.

    Args:
        images:       np.ndarray of shape (N, C, H, W) float32, already normalised
        augmentation: SimCLRAugmentation instance
    """
    def __init__(self, images):
        self.images = images

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        return self.images[idx]


class DecurDatasetGPU(Dataset):
    def __init__(self, images_m1, images_m2):
        assert len(images_m1) == len(images_m2)
        self.images_m1 = images_m1
        self.images_m2 = images_m2

    def __len__(self) -> int:
        return len(self.images_m1)

    def __getitem__(self, idx: int):
        # Just return raw images, augmentation happens on GPU in the training loop
        return self.images_m1[idx], self.images_m2[idx]  # each (C, H, W) float32

def build_MS_or_SAR_dataloaderGPU(
    data: np.ndarray,
    batch_size: int = 512,
    num_workers: int = 2,
) -> DataLoader:
    dataset = ContrastiveDatasetGPU(data)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
        drop_last=True,    # avoids a batch of size 1 at the end, which breaks BatchNorm
        prefetch_factor=2,
        persistent_workers=True,
    )

def build_decur_dataloaderGPU(
    data_m1: np.ndarray,
    data_m2: np.ndarray,
    batch_size: int = 512,
    num_workers: int = 2,
) -> DataLoader:
    assert len(data_m1) == len(data_m2)
    dataset = DecurDatasetGPU(data_m1, data_m2)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
        prefetch_factor=2,
    )


##############################
'''
class MSorSARAugmentation:
    def __init__(self, image_size: int = 224):
        self.image_size = image_size

    def apply(self, x: np.ndarray) -> torch.Tensor:
        # (C, H, W) → (H, W, C) for numpy augmentations
        x = x.transpose(1, 2, 0)

        # ── spatial ──────────────────────────────────────────────────
        #x = random_resized_crop(x, self.image_size)

        if np.random.rand() < 0.5:
            x = x[::-1, :, :].copy()       # horizontal flip

        if np.random.rand() < 0.5:
            x = x[:, ::-1, :].copy()       # vertical flip

        if np.random.rand() < 0.5:
            x = x.transpose(1, 0, 2).copy()  # swap H and W axes (transpose)

        # ── photometric ──────────────────────────────────────────────
        
        #if np.random.rand() < 0.8:
        #    x = random_brightness(x)

        #if np.random.rand() < 0.8:
        #    x = random_contrast(x)

        #if np.random.rand() < 0.2:
        #    x = to_gray(x)

        #if np.random.rand() < 0.5:
        #    x = gaussian_blur(x)

        # (H, W, C) → (C, H, W), then to tensor
        return torch.from_numpy(x.transpose(2, 0, 1).copy())

    def __call__(self, x: np.ndarray):
        return self.apply(x), self.apply(x)
'''

# ─────────────────────────────────────────────
#  DATASET & DATALOADER
# ─────────────────────────────────────────────
'''
class DecurDataset(Dataset):
    def __init__(self, images_m1, images_m2, augmentation):
        self.images_m1 = images_m1
        self.images_m2 = images_m2
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.images_m1)

    def __getitem__(self, idx: int):
        view1_1, view1_2 = self.augmentation(self.images_m1[idx])  # (C, H, W) each
        view2_1, view2_2 = self.augmentation(self.images_m2[idx])  # (C, H, W) each
        return view1_1, view1_2, view2_1, view2_2
'''

'''
class ContrastiveDataset(Dataset):
    """
    Wraps a numpy array of images for SimCLR contrastive pre-training.

    Args:
        images:       np.ndarray of shape (N, C, H, W) float32, already normalised
        augmentation: SimCLRAugmentation instance
    """
    def __init__(self, images, augmentation):
        self.images = images
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        view1, view2 = self.augmentation(self.images[idx])  # (C, H, W) each
        return view1, view2
'''

'''
def build_MS_or_SAR_dataloader(
    data: np.ndarray,
    image_size: int = 224,
    batch_size: int = 256,
    num_workers: int = 3,
) -> DataLoader:
    """
    Convenience function to build the contrastive DataLoader in one call.

    Args:
        images:      np.ndarray of shape (N, C, H, W) float32, already normalised
        image_size:  spatial size passed to SimCLRAugmentation
        batch_size:  number of samples per batch (use largest that fits in VRAM)
        num_workers: parallel workers for data loading
    """
    #augmentation = MSorSARAugmentation(image_size=image_size)
    dataset = ContrastiveDataset(data, augmentation)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,    # avoids a batch of size 1 at the end, which breaks BatchNorm
        prefetch_factor=4
    )
'''

def built_decur_dataloader(data_m1: np.ndarray, data_m2: np.ndarray, image_size: int = 224, batch_size: int = 256, num_workers: int = 5) -> DataLoader:
    assert len(data_m1) == len(data_m2)
    augmentation = MSorSARAugmentation(image_size=image_size)
    dataset = DecurDataset(data_m1, data_m2, augmentation)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,    # avoids a batch of size 1 at the end, which breaks BatchNorm
        prefetch_factor=4
    )
############################################################################
def variance_loss(emb, gamma=1.0):
    std = torch.sqrt(emb.var(dim=0) + 1e-8)
    return torch.mean(F.relu(gamma - std))


def redundancy_loss(emb_a, emb_b):
    # Normalize across batch dimension (not feature dimension)
    a_norm = (emb_a - emb_a.mean(0)) / (emb_a.std(0) + 1e-8)
    b_norm = (emb_b - emb_b.mean(0)) / (emb_b.std(0) + 1e-8)
    
    # Cross-correlation matrix between inv and spc feature dimensions
    N, D = a_norm.shape
    cross_corr = torch.matmul(a_norm.T, b_norm) / N  # (D/2, D/2)

    # Penalize all cross-correlations — we want this matrix to be zero
    loss = torch.sum(cross_corr ** 2) / (D * D)
    return loss

'''
def NTXentLoss_patchlevel(embeddings, patch_labels, img_labels, temperature=0.07):
    """
    Patch-level contrastive loss with within-tile exclusion.
    
    For each anchor patch token (m, i):
      - Positive  : same patch index, other modality (strict spatial correspondence)
      - Ignored   : same image, different patch index (within-tile, potential false negatives)
      - Negatives : different image entirely (cross-image only)
    
    Args:
        embeddings:    (2*B*P, D) — normalized patch embeddings, M1 first then M2
        patch_labels:  (2*B*P,)  — unique ID per patch, same index = spatial counterpart
                                   e.g. [0,1,...,B*P-1, 0,1,...,B*P-1]
        img_labels:    (2*B*P,)  — image ID per patch, groups patches by image
                                   e.g. [0,0,...,0, 1,1,...,1, ..., 0,0,...,0, 1,1,...,1]
        temperature:   scalar
    """
    embeddings = embeddings.float()
    batch_size = embeddings.shape[0]  # 2*B*P

    # Full pairwise cosine similarity matrix (already normalized input)
    similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature  # (2*B*P, 2*B*P)

    # --- Mask 1: diagonal (self-similarity) ---
    diagonal_mask = torch.eye(batch_size, device=embeddings.device, dtype=torch.bool)

    # --- Mask 2: positives — same patch index, different modality ---
    # patch_labels[i] == patch_labels[j] AND i != j
    positive_mask = (patch_labels.unsqueeze(0) == patch_labels.unsqueeze(1))
    positive_mask = positive_mask & ~diagonal_mask  # exclude diagonal
    positive_mask_float = positive_mask.float()

    # --- Mask 3: within-tile ignore — same image, different patch ---
    # same img_label but NOT a positive and NOT diagonal
    same_image_mask = (img_labels.unsqueeze(0) == img_labels.unsqueeze(1))
    ignore_mask = same_image_mask & ~positive_mask & ~diagonal_mask

    # --- Apply masks to similarity matrix ---
    # Mask out diagonal
    similarity_matrix = similarity_matrix.masked_fill(diagonal_mask, float('-inf'))
    # Mask out within-tile non-positive pairs (ignored, not negatives)
    similarity_matrix = similarity_matrix.masked_fill(ignore_mask, float('-inf'))

    # Numerical stability
    similarity_matrix = similarity_matrix - similarity_matrix.max(dim=1, keepdim=True).values.detach()

    log_prob = F.log_softmax(similarity_matrix, dim=1)
    log_prob = torch.nan_to_num(log_prob, nan=0.0, posinf=0.0, neginf=0.0)

    # Loss: average negative log-likelihood over positive pairs
    loss = -(log_prob * positive_mask_float).sum(dim=1) / (positive_mask_float.sum(dim=1) + 1e-8)
    return loss.mean()
'''

def NTXentLoss(embeddings, labels, temperature=1.0):
    embeddings = embeddings.float()
    
    similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    
    batch_size = embeddings.shape[0]
    diagonal_mask = torch.eye(batch_size, device=embeddings.device, dtype=torch.bool)
    
    positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    positive_mask.fill_diagonal_(0)
    
    # Numerical stability
    similarity_matrix = similarity_matrix - similarity_matrix.max(dim=1, keepdim=True).values.detach()
    similarity_matrix_masked = similarity_matrix.masked_fill(diagonal_mask, float('-inf'))
    
    log_prob = F.log_softmax(similarity_matrix_masked, dim=1)
    
    # FIX: replace -inf with 0 in log_prob BEFORE multiplying with positive_mask
    # -inf * 0 = nan, but 0 * 0 = 0 which is correct since those are diagonal entries
    log_prob = torch.nan_to_num(log_prob, nan=0.0, posinf=0.0, neginf=0.0)
    
    loss = -(log_prob * positive_mask).sum(dim=1) / (positive_mask.sum(dim=1) + 1e-8)
    
    return loss.mean()

def cumulate_EMA(model, ema_weights, alpha):
    current_weights = OrderedDict()
    current_weights_npy = OrderedDict()
    state_dict = model.state_dict()
    for k in state_dict:
        current_weights_npy[k] = state_dict[k].cpu().detach().numpy()

    if ema_weights is not None:
        for k in state_dict:
            current_weights_npy[k] = alpha * ema_weights[k].cpu().detach().numpy() + (1-alpha) * current_weights_npy[k]

    for k in state_dict:
        current_weights[k] = torch.tensor( current_weights_npy[k] )

    return current_weights

def modify_weights(model, ema_weights, alpha):
    current_weights = OrderedDict()
    current_weights_npy = OrderedDict()
    state_dict = model.state_dict()
    
    for k in state_dict:
        current_weights_npy[k] = state_dict[k].cpu().detach().numpy()

    if ema_weights is not None:
        for k in state_dict:
            current_weights_npy[k] = alpha * ema_weights[k] + (1-alpha) * current_weights_npy[k]
    
    for k in state_dict:
        current_weights[k] = torch.tensor( current_weights_npy[k] )
    
    return current_weights, current_weights_npy

'''
class MyDatasetDouble(Dataset):
    def __init__(self, data1, data2, targets, transform=None):
        self.data1 = data1
        self.data2 = data2
        #self.targets = torch.LongTensor(targets)
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        x1 = self.data1[index]
        x2 = self.data2[index]
        y = self.targets[index]
        
        if self.transform:
            if np.random.uniform() > .5:
                x1 = self.transform(x1)
                x2 = self.transform(x2)
        
        return x1, x2, y

    def __len__(self):
        return len(self.data1)
'''

class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        #self.targets = torch.LongTensor(targets)
        self.targets = targets
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform:
            if np.random.uniform() > .5:
                x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)
    
class MyRotateTransform():
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

angle = [0, 90, 180, 270]
transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(degrees=(-45, 45)),
    T.RandomApply([MyRotateTransform(angles=angle)], p=0.5),
    T.GaussianBlur(kernel_size=3)
    #T.RandomApply([T.ColorJitter()], p=0.5)
    ])

'''
def createDataLoaderDouble(x1, x2, y, tobeshuffled, BATCH_SIZE):
    #DATALOADER TRAIN
    x1_tensor = torch.tensor(x1, dtype=torch.float32)
    x2_tensor = torch.tensor(x2, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.int64)

    #dataset = TensorDataset(x_ms_tensor, y_tensor)
    dataset = MyDatasetDouble(x1_tensor, x2_tensor, y_tensor, transform=transform)
    dataloader = DataLoader(dataset, shuffle=tobeshuffled, batch_size=BATCH_SIZE)
    return dataloader
'''

def createDataLoader(x, y, tobeshuffled, BATCH_SIZE, activte_transform=True, is_multilabel=False):
    #DATALOADER TRAIN
    x_tensor = torch.tensor(x, dtype=torch.float32)
    if is_multilabel:
        y_tensor = torch.tensor(y, dtype=torch.float32)
    else:
        y_tensor = torch.tensor(y, dtype=torch.int64)

    #dataset = TensorDataset(x_ms_tensor, y_tensor)
    if activte_transform:
        dataset = MyDataset(x_tensor, y_tensor, transform=transform)
    else:
        dataset = MyDataset(x_tensor, y_tensor, transform=None)
    dataloader = DataLoader(dataset, shuffle=tobeshuffled, batch_size=BATCH_SIZE)
    return dataloader

def evaluation(model, dataloader, device, multilabel=False):
    model.eval()
    tot_pred = []
    tot_labels = []
    with torch.no_grad():
        for f_batch, s_batch, y_batch in dataloader:
            f_batch = f_batch.to(device, non_blocking=True)
            s_batch = s_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            pred = model.predict(f_batch, s_batch)            
            preds = torch.argmax(pred, dim=1).cpu()
            tot_pred.append( preds )
            tot_labels.append( y_batch.cpu())
    tot_pred = torch.cat(tot_pred).cpu().numpy()
    tot_labels = torch.cat(tot_labels).numpy()
    
    return tot_pred, tot_labels




