"""
tcgm.py -- TCGM model and loss for two modalities.

Sun, X., Xu, Y., Cao, P., Kong, Y., Hu, L., Zhang, S., Wang, Y. (2020).
"TCGM: An Information-Theoretic Framework for Semi-Supervised Multi-Modality Learning." ECCV 2020.
Official code (20news only): github.com/Newbeeer/Semi-supervised-Multi-Modalities-Learning

Model: one classifier h_m per modality. The encoders and heads are the same as ScoreFusion
(model.py), so the backbone is identical to the SF baselines.

Total Correlation gain for M = 2, paper Eq. 6, with a batch of B unlabelled samples:
    TCg = 1 + 1/B sum_i log sum_c h1(x1_i)_c h2(x2_i)_c / p_c
            - 1/(B(B-1)) sum_{i != j} sum_c h1(x1_i)_c h2(x2_j)_c / p_c
    L_TC = -TCg
This is the same as mig_loss_function in the official code.

Aggregator used for prediction (Sec. 4.2 / 4.3):
    zeta(x)_c  proportional to  h1(x1)_c h2(x2)_c / p_c     (M - 1 = 1)
p is the empirical class prior of the labelled set. It is stored in the checkpoint as a
buffer and used both in L_TC and in the aggregator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ViTEncoder, MLPHead, SFFCConfig  # noqa: F401  (SFFCConfig re-exported)


class TCGMModel(nn.Module):
    def __init__(self, config, prior=None):
        super().__init__()
        # same encoders / heads as model.ScoreFusion
        self.modality_1_encoder = ViTEncoder(img_size=config.img_size_m1, patch_size=config.patch_size_m1,
                                             in_chans=config.in_chans_m1)
        self.modality_2_encoder = ViTEncoder(img_size=config.img_size_m2, patch_size=config.patch_size_m2,
                                             in_chans=config.in_chans_m2)
        self.head_M1 = MLPHead(config.hidden_dim, config.num_classes, config.dropout)
        self.head_M2 = MLPHead(config.hidden_dim, config.num_classes, config.dropout)
        if prior is None:
            prior = torch.full((config.num_classes,), 1.0 / config.num_classes)
        self.register_buffer("prior", torch.as_tensor(prior, dtype=torch.float32))

    def forward(self, x1, x2):
        """Per-modality logits (logits_1, logits_2)."""
        return self.head_M1(self.modality_1_encoder(x1)), self.head_M2(self.modality_2_encoder(x2))

    def predict(self, x1, x2):
        """Aggregator zeta: log h1 + log h2 - log p (argmax-equivalent to the normalised product)."""
        l1, l2 = self.forward(x1, x2)
        return (F.log_softmax(l1.float(), dim=-1) + F.log_softmax(l2.float(), dim=-1)
                - torch.log(self.prior.clamp(min=1e-12)))


def tcgm_loss(h1, h2, prior, eps=1e-12):
    """
    L_TC = -TCg (paper Eq. 6, M = 2).
    h1, h2 : [B, C] probabilities of the two modality classifiers on the SAME B samples
    prior  : [C] class prior p
    Returns (loss, term_a, term_b): term_a is the agreement on the same sample, term_b the
    agreement on different samples (the penalty).
    """
    h1, h2, prior = h1.float(), h2.float(), prior.float()
    B = h1.size(0)
    S = (h1 / prior.clamp(min=eps)) @ h2.t()          # S_ij = sum_c h1_ic h2_jc / p_c
    diag = torch.diagonal(S)
    term_a = torch.log(diag + eps).mean()
    term_b = (S.sum() - diag.sum()) / (B * (B - 1))
    loss = -(1.0 + term_a - term_b)
    return loss, term_a.detach(), term_b.detach()
