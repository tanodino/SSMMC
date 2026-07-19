"""
MSC: Modal and Strategic Complementarity framework for Semi-Supervised
Multimodal Classification (SSMC).

Reimplementation of:
  Chen, J.; Zhang, R.; Chen, J. "Semi-Supervised Multimodal Classification
  Through Learning from Modal and Strategic Complementarities." AAAI-25.

This file is a GENERIC, encoder-agnostic implementation: you plug in any two
`nn.Module` encoders (ResNet, ViT, MLP, BERT, a CNN, a sensor-specific
backbone, ...) that each map a raw input to a fixed-size feature vector, and
this module builds the two modality-specific classifiers, the Feature-Concat
classifier, the reliability generator, and all three guidance losses (LCG,
MRG, SCG) on top of them.

--------------------------------------------------------------------------
Equation -> code map (equation numbers refer to the AAAI-25 paper, where
modality_1 corresponds to the paper's "text" (t) slot and modality_2 to the
paper's "image" (v) slot)
--------------------------------------------------------------------------
Eq. 1-2   unimodal predictions p_M1, p_M2               -> MSCModel.forward
Eq. 3     averaged Score Fusion (baseline, not used
          once reliability weights are learned)        -> not used directly
Eq. 4     Feature Concat p_C                            -> MSCModel.forward
Eq. 5     supervised CE loss                            -> weighted_nll_loss
Eq. 6     FixMatch/FreeMatch-style pseudo-label loss     -> pseudo_label_loss
Eq. 7     Modal Reliability Generator w(z|m1,m2)          -> MSCModel.forward
Eq. 8     weighted Score Fusion p_S                       -> MSCModel.forward
Eq. 9     r_1, r_2 from strong-augmented predictions       -> caller (see
                                                              MSCLoss.forward)
Eq. 10    Label Consistency Guidance target G              -> label_consistency_targets
Eq. 11    L_lcg                                            -> MSCLoss.forward
Eq. 12    L_mrg (Modal Reliability Guidance)                -> mrg_masks / mrg_loss
Eq. 13    Consistent Pseudo-label Selection (D_sub)          -> consistent_pseudo_label_selection
Eq. 14    L_scg (Strategic Complementarity Guidance)          -> scg_masks / scg_loss
Eq. 15    total training objective                            -> MSCLoss.forward

--------------------------------------------------------------------------
Things the paper does NOT fully specify, and the choices made here
--------------------------------------------------------------------------
1. MLP head depth/width for the M1, M2, C and reliability heads are not
   given beyond "an MLP". We use a single hidden layer of configurable
   width (default 256) with GELU + dropout. Change `MLPHead` or the
   `hidden_dim` argument freely.
2. Whether the "teacher" side of the MRG (Eq. 12) and SCG (Eq. 14) KL terms
   is stop-gradient (detached) is not stated explicitly. The paper's prose
   ("the unreliable modality learns from the more reliable one") reads as a
   one-directional guidance signal, so we detach the teacher by default.
   Set `detach_teacher=False` in MSCConfig to make both branches trainable
   through every KL term instead.
3. The unsupervised loss weight lambda in "L_sup + lambda*L_u" (used to
   build L_S and L_C in Eq. 15) is not given a value in this paper (it is a
   FixMatch/FreeMatch convention). Default is 1.0, exposed as `lambda_u`.
4. Data augmentation (e.g. RandAugment-style perturbation for images,
   swap/synonym + SBERT-similarity selection for text, or whatever the
   appropriate weak/strong augmentation is for your two modalities) is
   modality-specific and is intentionally left OUT of this file. You must
   supply weakly- and strongly-augmented views of each unlabeled sample
   yourself (as tensors your encoders can consume); this is what keeps the
   framework backbone- and modality-agnostic.
5. FreeMatch's self-adaptive threshold is not reimplemented; a fixed
   confidence threshold `tau` (paper: not explicitly stated for MSC itself,
   only eta=0.95 is given for Eq. 13) is used for Eq. 6. Swap in an adaptive
   threshold scheduler if you want to reproduce the FreeMatch variant
   exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ViTEncoder, MLPHead, ResNet18Encoder


# ==========================================================================
# 0. Config
# ==========================================================================

@dataclass
class MSCConfig:
    img_size_m1: int 
    img_size_m2: int
    patch_size_m1: int
    patch_size_m2: int
    in_chans_m1: int
    in_chans_m2: int

    num_classes: int
    hidden_dim: int = 256
    dropout: float = 0.1

    # Eq. 6: confidence threshold for FixMatch/FreeMatch-style pseudo-labeling
    tau: float = 0.95
    # Eq. 13: confidence threshold for Consistent Pseudo-label Selection
    eta: float = 0.95
    # weight of the unsupervised term inside L_S / L_C (L_sup + lambda_u * L_u)
    lambda_u: float = 1.0

    # Eq. 15 loss weights
    beta1: float = 1.0  # L_lcg
    beta2: float = 1.0  # L_mrg
    beta3: float = 1.0  # L_scg

    # whether MRG/SCG KL "teacher" side is stop-gradient (see module docstring, point 2)
    detach_teacher: bool = True

    eps: float = 1e-8


# ==========================================================================
# 1. Building blocks
# ==========================================================================




class MSCModel(nn.Module):
    """
    Wraps two arbitrary, pluggable encoders and builds every classifier /
    reliability head the MSC framework needs on top of them.

    modality_1_encoder / modality_2_encoder: any nn.Module such that
        modality_1_encoder(modality_1_input) -> Tensor[B, modality_1_dim]
        modality_2_encoder(modality_2_input) -> Tensor[B, modality_2_dim]

    The two slots are interchangeable and modality-agnostic: plug in any
    pair (depth/thermal, SAR/multispectral, audio/video, tabular/image,
    text/image, ...) as long as each encoder returns a fixed-size feature
    vector.
    """

    #def __init__(self, modality_1_encoder: nn.Module, modality_2_encoder: nn.Module, config: MSCConfig):
    def __init__(self, config: MSCConfig):
        super().__init__()
        #self.modality_1_encoder = modality_1_encoder
        #self.modality_2_encoder = modality_2_encoder
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
        
        

        self.cfg = config

        self.head_M1 = MLPHead(config.hidden_dim,
                                config.num_classes, config.dropout)          # Eq. 1
        self.head_M2 = MLPHead(config.hidden_dim,
                                config.num_classes, config.dropout)          # Eq. 2
        self.head_C = MLPHead(config.hidden_dim,
                               config.num_classes, config.dropout)          # Eq. 4
        self.reliability_net = MLPHead(config.hidden_dim, 2, config.dropout)  # Eq. 7

    def encode(self, modality_1_input, modality_2_input) -> Tuple[torch.Tensor, torch.Tensor]:
        f_1 = self.modality_1_encoder(modality_1_input)
        f_2 = self.modality_2_encoder(modality_2_input)
        return f_1, f_2

    def forward(self, modality_1_input, modality_2_input) -> dict:
        f_1, f_2 = self.encode(modality_1_input, modality_2_input)
        return self.forward_from_features(f_1, f_2)

    def forward_from_features(self, f_1: torch.Tensor, f_2: torch.Tensor) -> dict:
        eps = self.cfg.eps
        concat = torch.cat([f_1, f_2], dim=-1)

        logits_M1 = self.head_M1(f_1)
        logits_M2 = self.head_M2(f_2)
        logits_C = self.head_C(concat)
        logits_w = self.reliability_net(concat)

        p_M1 = F.softmax(logits_M1, dim=-1)          # Eq. 1
        p_M2 = F.softmax(logits_M2, dim=-1)           # Eq. 2
        p_C = F.softmax(logits_C, dim=-1)              # Eq. 4
        w = F.softmax(logits_w, dim=-1)                 # Eq. 7, w[:,0]=modality_1 reliability, w[:,1]=modality_2 reliability

        p_S = w[:, 0:1] * p_M1 + w[:, 1:2] * p_M2         # Eq. 8, weighted Score Fusion

        return {
            "f_1": f_1, "f_2": f_2,
            "logits_M1": logits_M1, "logits_M2": logits_M2, "logits_C": logits_C,
            "p_M1": p_M1.clamp(min=eps), "p_M2": p_M2.clamp(min=eps),
            "p_C": p_C.clamp(min=eps), "p_S": p_S.clamp(min=eps),
            "w": w.clamp(min=eps),
        }
    
    @torch.no_grad()
    def predict(self, modality_1_input, modality_2_input) -> torch.Tensor:
        out = self.forward(modality_1_input, modality_2_input)
        p_final = 0.5 * (out["p_S"] + out["p_C"])
        return p_final



# ==========================================================================
# 2. Generic loss primitives
# ==========================================================================

def weighted_nll_loss(p: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Cross-entropy computed from probabilities (needed because p_S is a
    weighted mixture of two softmaxes, not a single softmax output, so
    F.cross_entropy on logits isn't directly applicable). Implements Eq. 5.
    """
    if y.numel() == 0:
        return p.new_zeros(())
    logp = torch.log(p.clamp(min=eps))
    return F.nll_loss(logp, y)


def pseudo_label_loss(p_strong: torch.Tensor, p_weak: torch.Tensor, tau: float,
                       eps: float = 1e-8) -> torch.Tensor:
    """FixMatch/FreeMatch-style pseudo-label loss, Eq. 6.
    Normalization is over the FULL unlabeled batch B_u (per Eq. 6), not over
    the selected subset — masked-out samples contribute 0.
    """
    with torch.no_grad():
        conf, yhat = p_weak.max(dim=-1)
        mask = (conf >= tau).to(p_strong.dtype)
    logp = torch.log(p_strong.clamp(min=eps))
    per_sample = F.nll_loss(logp, yhat, reduction="none")
    return (per_sample * mask).mean()

'''
def pseudo_label_loss(p_strong: torch.Tensor, p_weak: torch.Tensor, tau: float,
                       eps: float = 1e-8) -> torch.Tensor:
    """FixMatch/FreeMatch-style pseudo-label loss, Eq. 6.

    p_weak:   predictions on the weakly-augmented view, used to select
              high-confidence pseudo-labels.
    p_strong: predictions on the strongly-augmented view, trained against
              those pseudo-labels.
    """
    with torch.no_grad():
        conf, yhat = p_weak.max(dim=-1)
        mask = conf >= tau
    if mask.sum() == 0:
        return p_strong.new_zeros(())
    return weighted_nll_loss(p_strong[mask], yhat[mask], eps=eps)
'''

def kl_divergence(p_target: torch.Tensor, q_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample KL(p_target || q_pred), both [B, C] probability tensors."""
    p = p_target.clamp(min=eps)
    q = q_pred.clamp(min=eps)
    return (p * (p.log() - q.log())).sum(dim=-1)


# ==========================================================================
# 3. Consistent Pseudo-label Selection (Eq. 13)
# ==========================================================================

def consistent_pseudo_label_selection(p_S_weak: torch.Tensor, p_C_weak: torch.Tensor,
                                       eta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Selects unlabeled samples on which weighted Score Fusion and Feature
    Concat agree on the predicted class AND at least one branch is
    confident above `eta`. Implements Eq. 13.

    Returns
    -------
    mask : BoolTensor [B]   -- True where the sample is accepted into D_sub
    yhat : LongTensor [B]   -- pseudo-label (== yhat_s == yhat_c on accepted samples)
    """
    conf_s, yhat_s = p_S_weak.max(dim=-1)
    conf_c, yhat_c = p_C_weak.max(dim=-1)
    consistent = yhat_s == yhat_c
    confident = (conf_s >= eta) | (conf_c >= eta)
    mask = consistent & confident
    return mask, yhat_s


# ==========================================================================
# 4. Label Consistency Guidance (Eq. 9-11)
# ==========================================================================

def label_consistency_targets(r_1: torch.Tensor, r_2: torch.Tensor, y: torch.Tensor,
                               p_M1: torch.Tensor, p_M2: torch.Tensor,
                               eps: float = 1e-8) -> torch.Tensor:
    """
    Builds the guidance distribution G(z|m1,m2) of Eq. 10.

    r_1, r_2 : LongTensor [B]  -- argmax predictions of modality_1/modality_2 branches
    y        : LongTensor [B]  -- ground-truth label (labeled data) or
                                   pseudo-label (unlabeled data, from Eq. 13)
    p_M1, p_M2 : FloatTensor [B, C]  -- the corresponding class probabilities

    Returns
    -------
    G : FloatTensor [B, 2]  -- G[:,0] = G(z=0|m1,m2) (modality_1 reliable),
                                G[:,1] = G(z=1|m1,m2) (modality_2 reliable)
    """
    B = y.shape[0]
    device = y.device
    G0 = torch.zeros(B, device=device, dtype=p_M1.dtype)

    m1_ok = r_1 == y
    m2_ok = r_2 == y

    only_m1 = m1_ok & ~m2_ok
    only_m2 = ~m1_ok & m2_ok
    neither = ~m1_ok & ~m2_ok
    both = m1_ok & m2_ok

    G0[only_m1] = 1.0
    G0[only_m2] = 0.0
    G0[neither] = 0.5

    if both.any():
        idx = both.nonzero(as_tuple=True)[0]
        y_b = y[idx]
        p1_y = p_M1[idx, y_b]
        p2_y = p_M2[idx, y_b]
        G0[idx] = p1_y / (p1_y + p2_y + eps)

    G1 = 1.0 - G0
    return torch.stack([G0, G1], dim=-1)


# ==========================================================================
# 5. Modal Reliability Guidance (Eq. 12)
# ==========================================================================

def mrg_masks(r_1: torch.Tensor, r_2: torch.Tensor, y: torch.Tensor,
              p_M1: torch.Tensor, p_M2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Splits D_sub into D_1to2 (modality_1 teaches modality_2) and
    D_2to1 (modality_2 teaches modality_1)."""
    m1_ok = r_1 == y
    m2_ok = r_2 == y
    p1_max = p_M1.max(dim=-1).values
    p2_max = p_M2.max(dim=-1).values

    mask_1to2 = (m1_ok & ~m2_ok) | (m1_ok & m2_ok & (p1_max > p2_max))
    mask_2to1 = (~m1_ok & m2_ok) | (m1_ok & m2_ok & (p2_max > p1_max))
    return mask_1to2, mask_2to1


def mrg_loss(p_M1: torch.Tensor, p_M2: torch.Tensor, mask_1to2: torch.Tensor,
             mask_2to1: torch.Tensor, detach_teacher: bool = True,
             eps: float = 1e-8) -> torch.Tensor:
    """Eq. 12: L_mrg = mean_{D_1to2} KL(p_M1||p_M2) + mean_{D_2to1} KL(p_M2||p_M1)."""
    zero = p_M1.new_zeros(())
    term_1to2 = zero
    term_2to1 = zero

    if mask_1to2.any():
        teacher = p_M1[mask_1to2].detach() if detach_teacher else p_M1[mask_1to2]
        student = p_M2[mask_1to2]
        term_1to2 = kl_divergence(teacher, student, eps=eps).mean()

    if mask_2to1.any():
        teacher = p_M2[mask_2to1].detach() if detach_teacher else p_M2[mask_2to1]
        student = p_M1[mask_2to1]
        term_2to1 = kl_divergence(teacher, student, eps=eps).mean()

    return term_1to2 + term_2to1


# ==========================================================================
# 6. Strategic Complementarity Guidance (Eq. 14)
# ==========================================================================

def scg_masks(r_s: torch.Tensor, r_c: torch.Tensor, y: torch.Tensor,
              p_S: torch.Tensor, p_C: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Splits D_sub into D_s2c (Score Fusion teaches Feature Concat) and D_c2s (reverse)."""
    s_ok = r_s == y
    c_ok = r_c == y
    pS_max = p_S.max(dim=-1).values
    pC_max = p_C.max(dim=-1).values

    mask_s2c = (s_ok & ~c_ok) | (s_ok & c_ok & (pS_max > pC_max))
    mask_c2s = (~s_ok & c_ok) | (s_ok & c_ok & (pC_max > pS_max))
    return mask_s2c, mask_c2s


def scg_loss(p_S: torch.Tensor, p_C: torch.Tensor, mask_s2c: torch.Tensor,
             mask_c2s: torch.Tensor, detach_teacher: bool = True,
             eps: float = 1e-8) -> torch.Tensor:
    """Eq. 14: L_scg = mean_{D_s2c} KL(p_S||p_C) + mean_{D_c2s} KL(p_C||p_S)."""
    zero = p_S.new_zeros(())
    term_s2c = zero
    term_c2s = zero

    if mask_s2c.any():
        teacher = p_S[mask_s2c].detach() if detach_teacher else p_S[mask_s2c]
        student = p_C[mask_s2c]
        term_s2c = kl_divergence(teacher, student, eps=eps).mean()

    if mask_c2s.any():
        teacher = p_C[mask_c2s].detach() if detach_teacher else p_C[mask_c2s]
        student = p_S[mask_c2s]
        term_c2s = kl_divergence(teacher, student, eps=eps).mean()

    return term_s2c + term_c2s


# ==========================================================================
# 7. Full training objective (Eq. 15)
# ==========================================================================

class MSCLoss(nn.Module):
    """
    Orchestrates one full MSC training step: builds L_S, L_C (Eq. 5-6 style
    pseudo-labeling losses on the two fusion branches), D_sub (Eq. 13), and
    the three guidance losses L_lcg (Eq. 11), L_mrg (Eq. 12), L_scg (Eq. 14),
    then combines them via Eq. 15.
    """

    def __init__(self, config: MSCConfig):
        super().__init__()
        self.cfg = config

    def forward(self, model: MSCModel,
                labeled_m1, labeled_m2, labeled_y: torch.Tensor,
                unlabeled_m1_weak, unlabeled_m2_weak,
                unlabeled_m1_strong, unlabeled_m2_strong, current_epoch, warm_up_epoch_ssl) -> dict:
        cfg = self.cfg

        # ---- labeled forward pass (no augmentation) ----------------------
        out_l = model(labeled_m1, labeled_m2)

        L_S_sup = weighted_nll_loss(out_l["p_S"], labeled_y, cfg.eps)   # Eq. 5, p = p_S
        L_C_sup = weighted_nll_loss(out_l["p_C"], labeled_y, cfg.eps)    # Eq. 5, p = p_C

        # ---- unlabeled forward passes (weak & strong augmentation) -------
        '''
        out_uw = model(unlabeled_m1_weak, unlabeled_m2_weak)
        out_us = model(unlabeled_m1_strong, unlabeled_m2_strong)
        '''

        was_training = model.training
        model.eval()
        with torch.no_grad():
            out_uw = model(unlabeled_m1_weak, unlabeled_m2_weak)
        if was_training:
            model.train()
        
        out_us = model(unlabeled_m1_strong, unlabeled_m2_strong)


        # Eq. 6 style pseudo-label losses for each fusion branch
        L_S_unsup = pseudo_label_loss(out_us["p_S"], out_uw["p_S"], cfg.tau, cfg.eps)
        L_C_unsup = pseudo_label_loss(out_us["p_C"], out_uw["p_C"], cfg.tau, cfg.eps)

        if current_epoch >= warm_up_epoch_ssl:
            L_S = L_S_sup + cfg.lambda_u * L_S_unsup
            L_C = L_C_sup + cfg.lambda_u * L_C_unsup
        else:
            L_S = L_S_sup
            L_C = L_C_sup

        # ---- Eq. 13: Consistent Pseudo-label Selection --------------------
        sub_mask, yhat_sub = consistent_pseudo_label_selection(out_uw["p_S"], out_uw["p_C"], cfg.eta)

        # ---- build D_sub = X_sub (labeled) U U_sub (selected unlabeled) ---
        # labeled part uses non-augmented predictions + ground truth (paper,
        # "Learning from Modal Complementarity" section: "we replace yhat
        # with ground-truth labels y and use no-augmented data pairs for
        # labeled data").
        # unlabeled part uses STRONG-augmented predictions (paper: "we first
        # obtain predicted results with strong-augment of unlabeled input"),
        # paired with the pseudo-label from Eq. 13.
        if sub_mask.any():
            p_M1_sub = torch.cat([out_l["p_M1"], out_us["p_M1"][sub_mask]], dim=0)
            p_M2_sub = torch.cat([out_l["p_M2"], out_us["p_M2"][sub_mask]], dim=0)
            p_S_sub = torch.cat([out_l["p_S"], out_us["p_S"][sub_mask]], dim=0)
            p_C_sub = torch.cat([out_l["p_C"], out_us["p_C"][sub_mask]], dim=0)
            w_sub = torch.cat([out_l["w"], out_us["w"][sub_mask]], dim=0)
            y_sub = torch.cat([labeled_y, yhat_sub[sub_mask]], dim=0)
        else:
            p_M1_sub, p_M2_sub = out_l["p_M1"], out_l["p_M2"]
            p_S_sub, p_C_sub, w_sub = out_l["p_S"], out_l["p_C"], out_l["w"]
            y_sub = labeled_y

        r_1 = p_M1_sub.argmax(dim=-1)
        r_2 = p_M2_sub.argmax(dim=-1)
        r_s = p_S_sub.argmax(dim=-1)
        r_c = p_C_sub.argmax(dim=-1)

        # ---- Eq. 9-11: Label Consistency Guidance -------------------------
        G = label_consistency_targets(r_1, r_2, y_sub, p_M1_sub, p_M2_sub, cfg.eps).detach()
        L_lcg = kl_divergence(G, w_sub, cfg.eps).mean()

        # ---- Eq. 12: Modal Reliability Guidance ----------------------------
        mask_1to2, mask_2to1 = mrg_masks(r_1, r_2, y_sub, p_M1_sub, p_M2_sub)
        L_mrg = mrg_loss(p_M1_sub, p_M2_sub, mask_1to2, mask_2to1, cfg.detach_teacher, cfg.eps)

        # ---- Eq. 14: Strategic Complementarity Guidance ---------------------
        mask_s2c, mask_c2s = scg_masks(r_s, r_c, y_sub, p_S_sub, p_C_sub)
        L_scg = scg_loss(p_S_sub, p_C_sub, mask_s2c, mask_c2s, cfg.detach_teacher, cfg.eps)

        # ---- Eq. 15: total objective -----------------------------------------
        total = L_S + L_C + cfg.beta1 * L_lcg + cfg.beta2 * L_mrg + cfg.beta3 * L_scg

        return {
            "loss": total,
            "L_S": L_S.detach(), "L_C": L_C.detach(),
            "L_S_sup": L_S_sup.detach(), "L_C_sup": L_C_sup.detach(),
            "L_S_unsup": L_S_unsup.detach(), "L_C_unsup": L_C_unsup.detach(),
            "L_lcg": L_lcg.detach(), "L_mrg": L_mrg.detach(), "L_scg": L_scg.detach(),
            "n_selected_unlabeled": int(sub_mask.sum().item()),
        }


# ==========================================================================
# 8. Inference
# ==========================================================================

@torch.no_grad()
def predict(model: MSCModel, modality_1_input, modality_2_input) -> torch.Tensor:
    """Final prediction = average of p_S and p_C, as stated in the Training
    Objective section ("we take the average of predicted probabilities p_S
    and p_C from the MSC framework as final integration results")."""
    out = model(modality_1_input, modality_2_input)
    p_final = 0.5 * (out["p_S"] + out["p_C"])
    return p_final


