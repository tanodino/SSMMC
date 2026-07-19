import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import math
from typing import Tuple, Optional
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
from torchvision import models
from torchvision.models import resnet18, resnet50, AlexNet
from torch.autograd import Function
from collections import OrderedDict
from typing import Dict, List, Union
import torch.autograd as autograd
from dataclasses import dataclass
import os
import functools

import socket


class ResNet18Encoder(nn.Module):
    def __init__(self, img_size: int = None, in_chans: int = 3, gn_groups: int = 32):
        super().__init__()

        #norm_layer = functools.partial(nn.GroupNorm, gn_groups)
        #backbone = resnet18(weights=None, norm_layer=norm_layer)
        backbone = resnet18(weights=None)

        backbone.conv1 = nn.Conv2d(in_chans, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(backbone.conv1.weight, mode="fan_out", nonlinearity="relu")

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).flatten(1)


############## KDMVC ####################


class ViewSpecificExtractor(nn.Module):
    """
    Implements E^v(.), WITHOUT the gate/sparsity mechanism (see module
    docstring for why it's dropped):
 
        h^v = ViTEncoder(x^v)          # pooled embedding
        z^v = proj_net(h^v)            # projected feature
    """
 
    def __init__(self, vit_encoder: nn.Module, embed_dim: int, feat_dim: int):
        super().__init__()
        self.encoder = vit_encoder
        self.proj_net = nn.Sequential(
            nn.Linear(embed_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )
 
    def forward(self, x):
        h = self.encoder(x)
        if h.dim() == 3:
            h = h.mean(dim=1)
        z = self.proj_net(h)
        return z
 
 
class FusionHead(nn.Module):
    """
    Implements F_fusion(.): concatenates view-specific features and projects
    them to the unified representation h (paper Sec. 3.2, "Multi-view
    Unified Feature Extractor"). The paper does not specify the internal
    architecture of F_fusion(.) beyond "concatenation + fusion layer"; a
    2-layer MLP is used here as a reasonable, simple default.
    """
 
    def __init__(self, feat_dim: int, num_views: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim * num_views, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
 
    def forward(self, z_list):
        z_cat = torch.cat(z_list, dim=1)
        return self.net(z_cat)
 
 
class KDMvCModel(nn.Module):
    """
    Full KDMvC model, specialized to V=2 views (your two modalities).
    No gate / L1 sparsity mechanism (see module docstring).
 
    Forward pass returns a dict with:
      p_fusion : [B, C]        unified-representation classification logits
      p_views  : [Tensor, ...] per-view classification logits (share g_shared)
      z_views  : [Tensor, ...] per-view features (used in contrastive loss)
      h        : [B, hidden_dim]  unified representation (contrastive loss)
 
    Parameter grouping matches Algorithm 1's alternating schedule:
      - `specific_parameters()`: E(.) and F(.), i.e. both ViewSpecificExtractor
        instances plus the shared head g_shared -- trained in Phase 1
        ("multi-view unified to specific"), frozen in Phase 2.
      - `unified_parameters()`: F_fusion(.) and g_fusion(.) -- trained in
        both phases (Phase 1 via L^s1_x which includes the fusion path;
        Phase 2 exclusively).
    """
 
    def __init__(self, vit_encoder_m1: nn.Module, vit_encoder_m2: nn.Module,
                 embed_dim: int, feat_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.extractors = nn.ModuleList([
            ViewSpecificExtractor(vit_encoder_m1, embed_dim, feat_dim),
            ViewSpecificExtractor(vit_encoder_m2, embed_dim, feat_dim),
        ])
        # Sec. 3.3: "the multiple view-specific features share the same
        # classification head g(.)"
        self.g_shared = nn.Linear(feat_dim, num_classes)
        self.fusion = FusionHead(feat_dim, num_views=2, hidden_dim=hidden_dim)
        self.g_fusion = nn.Linear(hidden_dim, num_classes)
 
    def forward(self, x_m1, x_m2):
        z1 = self.extractors[0](x_m1)
        z2 = self.extractors[1](x_m2)
        z_views = [z1, z2]
 
        p_views = [self.g_shared(z) for z in z_views]
 
        h = self.fusion(z_views)
        p_fusion = self.g_fusion(h)
 
        return {
            "p_fusion": p_fusion,
            "p_views": p_views,
            "z_views": z_views,
            "h": h,
        }
 
    # ---- parameter groups for the alternating training schedule ----
    def specific_parameters(self):
        params = list(self.extractors.parameters())
        params += list(self.g_shared.parameters())
        return params
 
    def unified_parameters(self):
        return list(self.fusion.parameters()) + list(self.g_fusion.parameters())
 
    def set_specific_trainable(self, trainable: bool):
        """Freeze/unfreeze E(.) and F(.) -- Algorithm 1, line 13."""
        for p in self.specific_parameters():
            p.requires_grad = trainable

    def predict(self, x_m1, x_m2):
        out = self.forward(x_m1, x_m2)
        return out["p_fusion"]
  

##############

class FreeMatchThresholding:
    def __init__(self, num_classes, momentum=0.999, device="cuda"):
        self.momentum = momentum
        self.num_classes = num_classes
        self.device = device

        # global threshold: EMA of mean confidence, same init logic as SoftMatch's mu
        self.tau_t = torch.tensor(1.0 / num_classes, device=device)

        # per-class EMA of average predicted probability mass for each class
        self.p_t = torch.ones(num_classes, device=device) / num_classes

    @torch.no_grad()
    def update(self, probs_weak):
        # probs_weak: [B, C] full softmax distribution (NOT just the max) from the weak pass
        max_probs, _ = probs_weak.max(dim=-1)
        batch_tau = max_probs.mean()
        self.tau_t = self.momentum * self.tau_t + (1 - self.momentum) * batch_tau

        batch_p = probs_weak.mean(dim=0)  # [C] — average prob assigned to each class this batch
        self.p_t = self.momentum * self.p_t + (1 - self.momentum) * batch_p

    @torch.no_grad()
    def local_thresholds(self):
        # scale global threshold per class: classes the model is more confident about
        # (relative to the most confident class) get a threshold closer to tau_t;
        # harder classes get a proportionally lower bar
        max_p = self.p_t.max()
        return (self.p_t / max_p) * self.tau_t  # [C]

    @torch.no_grad()
    def mask(self, max_probs, pseudo_labels):
        local_tau = self.local_thresholds()           # [C]
        sample_tau = local_tau[pseudo_labels]          # [B] — gather each sample's class-specific bar
        return (max_probs >= sample_tau).float()       # [B] — hard 0/1 mask, unlike SoftMatch's continuous weight

    @torch.no_grad()
    def fairness_target(self):
        local_tau = self.local_thresholds()
        w = self.p_t / (local_tau + 1e-8)
        return w / w.sum()   # normalized target distribution, shape [C]


class SoftMatchWeighting:
    def __init__(self, num_classes, momentum=0.999, n_sigma=2, device="cuda"):
        self.momentum = momentum
        self.n_sigma = n_sigma
        self.num_classes = num_classes
        self.device = device
        self.prob_max_mu = torch.tensor(1.0 / num_classes, device=device)
        self.prob_max_var = torch.tensor(1.0, device=device)


    @torch.no_grad()
    def update(self, max_probs):
        batch_mu = max_probs.mean()
        batch_var = max_probs.var(unbiased=True)  # note: paper uses unbiased
        self.prob_max_mu = self.momentum * self.prob_max_mu + (1 - self.momentum) * batch_mu
        self.prob_max_var = self.momentum * self.prob_max_var + (1 - self.momentum) * batch_var

    @torch.no_grad()
    def weight(self, max_probs):
        # truncated Gaussian, capped at 1.0 for confidences above the running mean
        diff = torch.clamp(max_probs - self.prob_max_mu, max=0.0)
        #denom = 2 * (self.prob_max_std ** 2) + 1e-8
        denom = 2 * (self.prob_max_var) / (self.n_sigma ** 2) + 1e-8
        w = torch.exp(-(diff ** 2) / denom)
        return w



@dataclass
class SFFCConfig:
    img_size_m1: int 
    img_size_m2: int
    patch_size_m1: int
    patch_size_m2: int
    in_chans_m1: int
    in_chans_m2: int

    num_classes: int
    hidden_dim: int = 256
    dropout: float = 0.1

class MLPHead(nn.Module):
    """Generic single-hidden-layer MLP classifier used for every head
    (modality_1, modality_2, Feature Concat, reliability generator)."""

    def __init__(self, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            #nn.Linear(in_dim, hidden_dim),
            nn.LazyLinear(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            #nn.Linear(hidden_dim, out_dim),
            nn.LazyLinear(out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # returns logits

class ScoreFusion(nn.Module):
    def __init__(self, config: SFFCConfig):
        super().__init__()
        '''
        self.modality_1_encoder = ResNet18Encoder(in_chans=config.in_chans_m1)
        self.modality_2_encoder = ResNet18Encoder(in_chans=config.in_chans_m2)

        '''
        self.modality_1_encoder = ViTEncoder(
            img_size = config.img_size_m1,
            patch_size = config.patch_size_m1,
            in_chans = config.in_chans_m1
        )
        
        self.modality_2_encoder = ViTEncoder(
            img_size = config.img_size_m2,
            patch_size = config.patch_size_m2,
            in_chans = config.in_chans_m2
        )
        
        self.head_M1 = MLPHead(config.hidden_dim,
                                config.num_classes, config.dropout)          # Eq. 1
        self.head_M2 = MLPHead(config.hidden_dim,
                                config.num_classes, config.dropout)          # Eq. 2
        
    def forward(self, modality_1_input: torch.Tensor, modality_2_input: torch.Tensor):
        f_1 = self.modality_1_encoder(modality_1_input)
        f_2 = self.modality_2_encoder(modality_2_input)
        logits_M1 = self.head_M1(f_1)
        logits_M2 = self.head_M2(f_2)
        prob_1 = F.softmax(logits_M1, dim=-1)
        prob_2 = F.softmax(logits_M2, dim=-1)
        #return prob_2
        return (prob_1 + prob_2) / 2

    def predict(self, f_1: torch.Tensor, f_2: torch.Tensor):
        return self.forward(f_1, f_2)


class PretrainModel(nn.Module):
    def __init__(self, config: SFFCConfig):
        super().__init__()
        self.modality_1_encoder = ViTEncoder(
            img_size = config.img_size_m1,
            patch_size = config.patch_size_m1,
            in_chans = config.in_chans_m1
        )
        
        self.modality_2_encoder = ViTEncoder(
            img_size = config.img_size_m2,
            patch_size = config.patch_size_m2,
            in_chans = config.in_chans_m2
        )
        
        self.projector_m1 = nn.Sequential(nn.LazyLinear(512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 128), nn.BatchNorm1d(128))
        self.projector_m2 = nn.Sequential(nn.LazyLinear(512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 128), nn.BatchNorm1d(128))

    def forward(self, x1, x2):        
        cls_token_m1 = self.modality_1_encoder(x1)
        cls_token_m2 = self.modality_2_encoder(x2)
        return cls_token_m1, cls_token_m2, self.projector_m1(cls_token_m1), self.projector_m2(cls_token_m2)



class FusionConcat(nn.Module):
    def __init__(self, config: SFFCConfig):
        super().__init__()
        self.modality_1_encoder = ViTEncoder(
            img_size = config.img_size_m1,
            patch_size = config.patch_size_m1,
            in_chans = config.in_chans_m1
        )
        
        self.modality_2_encoder = ViTEncoder(
            img_size = config.img_size_m2,
            patch_size = config.patch_size_m2,
            in_chans = config.in_chans_m2
        )
        self.head_Concat = MLPHead(config.hidden_dim,
                                config.num_classes, config.dropout)          # Eq. 1
        
    def forward(self, modality_1_input: torch.Tensor, modality_2_input: torch.Tensor):
        f_1 = self.modality_1_encoder(modality_1_input)
        f_2 = self.modality_2_encoder(modality_2_input)
        f_1_2 = torch.cat((f_1, f_2), dim=-1)
        return self.head_Concat(f_1_2)

    def predict(self, f_1: torch.Tensor, f_2: torch.Tensor):
        return self.forward(f_1, f_2)




class PatchEmbed(nn.Module):
    """Image to Patch Embedding using Conv2d"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Use unfold to extract patches, then linear projection
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, H//patch_size, W//patch_size)
        x = self.proj(x)
        # Flatten spatial dimensions: (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x

class ViTEncoder(nn.Module):
    """Complete ViT-S MAE using PyTorch built-in functions"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384, 
                 depth=12, num_heads=6, mlp_ratio=4., dropout=0.1):
        super(ViTEncoder, self).__init__()

        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        
        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Use PyTorch's built-in TransformerEncoderLayer
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,  # Important: use batch_first=True
            norm_first=True    # Pre-norm like in modern transformers
        )
        
        self.transformer = TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize position embeddings and cls token
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize patch embedding projection
        nn.init.trunc_normal_(self.patch_embed.proj.weight, std=0.02)
        if self.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.patch_embed.proj.bias)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # (B, N, C)
        #print("x.shape ",x.shape)
        
        # Add position embedding (without cls token)
        x = x + self.pos_embed[:, 1:, :]
        
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Pass through transformer
        x = self.transformer(x)

        # Final norm
        emb = self.norm(x)
        return emb[:,0,:]
