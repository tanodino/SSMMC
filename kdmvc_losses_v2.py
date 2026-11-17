"""
kdmvc_losses_v2.py -- KDMvC class-aware contrastive loss following the paper (Eq. 9-12).

Wang, X., Wang, Y., Ke, G., Wang, Y., Hong, X. (2024). "Knowledge distillation-driven
semi-supervised multi-view classification." Information Fusion, 103, 102098.

Everything else (KDMvCWeighting, soft_cross_entropy) is unchanged and re-exported
from kdmvc_losses.py.

Eq. 12 (contrastive loss between the unified representation h and view v):
    L^v_c = sum_{i=1}^{2B_U} -1/|A(i)| sum_{a in A(i)} w_ia
            * log( exp(s(i,a)/tau) / sum_{j=1}^{2B_U} 1[j != i] exp(s(i,j)/tau) )
with s the cosine similarity (Eq. 9) and the indicator/weight matrix W^con (Eq. 11):
    w_ij = 1                      if j is the same instance as i (h_i <-> z^v_i),
           lambda(p_i)*lambda(p_j) if i and j have the same (pseudo-)category,
           0                      otherwise.

Implementation (differences from kdmvc_losses.ClassAwareContrastiveLoss, v1):
  * pool of 2B_U embeddings P = [h_1..h_B ; z^v_1..z^v_B] -- every element is an anchor
    (v1: only h anchors against the z^v pool, a B x B matrix);
  * the anchor itself is excluded from the denominator (1[j != i]) and from A(i)
    (v1: denominator included the anchor's own column);
  * A(i) = all other pool elements that are positives (counterpart + same pseudo-class,
    both h and z^v members of the pool), normalised by the COUNT |A(i)| (v1: by the sum
    of the weights);
  * the sum over the 2B_U anchors is averaged (mean over anchors), as in the reference
    SupCon / CCSSL implementations the paper builds on ([36], [38]); the paper's Sigma is
    the usual SupCon notation. This only rescales the loss by 1/(2B_U).
lambda(p) comes from KDMvCWeighting.weight (no gradient), pseudo-labels from the
fusion head (argmax), exactly as in v1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from kdmvc_losses import KDMvCWeighting, soft_cross_entropy  # noqa: F401  (re-exported, unchanged)


class ClassAwareContrastiveLossV2(nn.Module):
    """Eq. 9-12 over the 2B_U pool [h ; z^v]."""

    def __init__(self, temperature: float = 0.4):
        super().__init__()
        self.tau = temperature

    def forward(self, h: torch.Tensor, z_v: torch.Tensor,
                pseudo_labels: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        """
        h             : [B, D] unified representation (unlabelled batch)
        z_v           : [B, D] v-th view-specific representation (unlabelled batch)
        pseudo_labels : [B]    hard pseudo-label per sample (fusion-head argmax)
        lam           : [B]    lambda(p_i) confidence weight per sample (Eq. 6), no grad
        returns       : scalar L^v_c (mean over the 2B anchors)
        """
        B = h.size(0)
        pool = F.normalize(torch.cat([h, z_v], dim=0).float(), dim=1)        # [2B, D]
        sim = pool @ pool.t() / self.tau                                       # Eq. 9, [2B, 2B]

        inst = torch.arange(B, device=h.device).repeat(2)                      # instance id of each pool element
        lab = pseudo_labels.repeat(2)
        lam2 = lam.detach().float().repeat(2)

        self_mask = torch.eye(2 * B, device=h.device, dtype=torch.bool)
        same_inst = (inst.unsqueeze(0) == inst.unsqueeze(1)) & ~self_mask      # counterpart h_i <-> z_i
        same_cls = (lab.unsqueeze(0) == lab.unsqueeze(1)) & ~self_mask & ~same_inst

        # Eq. 11 weights on the positives A(i)
        w = torch.zeros(2 * B, 2 * B, device=h.device)
        w[same_inst] = 1.0
        lam_outer = lam2.unsqueeze(1) * lam2.unsqueeze(0)
        w[same_cls] = lam_outer[same_cls]
        pos_mask = same_inst | same_cls                                        # A(i)

        # log( exp(s_ia/tau) / sum_{j != i} exp(s_ij/tau) )
        logits = sim.masked_fill(self_mask, float("-inf"))
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        log_prob = log_prob.masked_fill(self_mask, 0.0)                        # avoid 0 * -inf

        n_pos = pos_mask.sum(dim=1).clamp(min=1)                               # |A(i)| >= 1 (counterpart)
        loss_i = -(w * log_prob).sum(dim=1) / n_pos
        return loss_i.mean()
