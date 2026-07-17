"""
KDMvC loss functions: distribution-aligned confidence weighting (Eq. 6-8),
class-aware contrastive loss (Eq. 9-13), and small helpers for the
supervised / knowledge-distillation losses (Eq. 4, 5, 15-17).

Adapted from: Wang, X., Wang, Y., Ke, G., Wang, Y., Hong, X. (2024).
"Knowledge distillation-driven semi-supervised multi-view classification."
Information Fusion, 103, 102098.

============================================================================
DOCUMENTED AMBIGUITIES / LITERAL-READING CHOICES
============================================================================
1. Eq. 6-8 mismatch: Eq. 7-8 define mu_t/sigma_t^2 as EMA statistics of the
   RAW confidence max(p_i), while Eq. 6 (lambda(p)) compares max(DA(p)) --
   the *distribution-aligned* confidence -- against those same mu_t/sigma_t.
   This is what the paper literally specifies (not an error on our part);
   implemented literally below rather than "corrected", since we have no
   reference code to check which one is intended.
2. Eq. 9-13 (class-aware contrastive loss): the paper describes a combined
   pool of 2*B_U embeddings (unified h_i's + view-specific z^v_j's) with an
   indicator matrix in R^{2B_U x 2B_U}, but doesn't fully specify whether
   positives are drawn only across the h/z^v boundary or also within h-h /
   z-z pairs. We implement the cross-modal case only (h_i's positives come
   from the z^v pool and vice versa) since that matches the paper's stated
   motivation most directly ("a contrastive constraint between the
   multi-view unified representations and the v-th view-specific
   representations"). Flag this if you want the full-pool interpretation
   instead -- it's a different (larger) loss, not just a scaling factor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KDMvCWeighting(nn.Module):
    """
    Implements Eq. 6-8: distribution alignment (DA) + EMA-tracked confidence
    mean/std, producing the per-sample weight lambda(p) used both in the
    unified-to-specific knowledge distillation loss (Eq. 5) and to reweight
    positive pairs in the class-aware contrastive loss (Eq. 11).

    Requires per-step updates of:
      - p_l_avg / p_u_avg : moving-average class-marginal distributions of
        labeled / unlabeled predictions (the DA(.) normalization term)
      - mu_t / sigma2_t   : EMA mean/variance of raw prediction confidence
    """

    def __init__(self, num_classes: int, beta: float = 0.99, device: str = "cuda"):
        super().__init__()
        self.beta = beta
        self.num_classes = num_classes
        self.register_buffer("mu_t", torch.tensor(1.0 / num_classes, device=device))
        self.register_buffer("sigma2_t", torch.tensor(1.0, device=device))
        self.register_buffer("p_l_avg", torch.ones(num_classes, device=device) / num_classes)
        self.register_buffer("p_u_avg", torch.ones(num_classes, device=device) / num_classes)
        self._init_l = False
        self._init_u = False

    @torch.no_grad()
    def update_label_distributions(self, probs_labeled: torch.Tensor = None,
                                    probs_unlabeled: torch.Tensor = None):
        if probs_labeled is not None:
            batch_avg = probs_labeled.mean(dim=0)
            if not self._init_l:
                self.p_l_avg.copy_(batch_avg)
                self._init_l = True
            else:
                self.p_l_avg.mul_(self.beta).add_(batch_avg, alpha=1 - self.beta)
        if probs_unlabeled is not None:
            batch_avg = probs_unlabeled.mean(dim=0)
            if not self._init_u:
                self.p_u_avg.copy_(batch_avg)
                self._init_u = True
            else:
                self.p_u_avg.mul_(self.beta).add_(batch_avg, alpha=1 - self.beta)

    @torch.no_grad()
    def distribution_align(self, probs: torch.Tensor) -> torch.Tensor:
        ratio = self.p_l_avg / (self.p_u_avg + 1e-8)
        aligned = probs * ratio.unsqueeze(0)
        aligned = aligned / (aligned.sum(dim=1, keepdim=True) + 1e-8)
        return aligned

    @torch.no_grad()
    def update_confidence_stats(self, probs: torch.Tensor):
        """Eq. 7-8. NOTE: uses RAW (non-DA) confidence, as literally specified."""
        max_p = probs.max(dim=1).values
        mu_e = max_p.mean()
        B = probs.size(0)
        sigma2_e = ((max_p - mu_e) ** 2).sum() / max(B - 1, 1)
        self.mu_t.mul_(self.beta).add_(mu_e, alpha=1 - self.beta)
        self.sigma2_t.mul_(self.beta).add_(sigma2_e, alpha=1 - self.beta)

    @torch.no_grad()
    def weight(self, probs: torch.Tensor) -> torch.Tensor:
        """Eq. 6: lambda(p), evaluated on DA(p) against mu_t/sigma2_t."""
        aligned = self.distribution_align(probs)
        max_da = aligned.max(dim=1).values
        below = max_da < self.mu_t
        lam = torch.where(
            below,
            torch.exp(-((max_da - self.mu_t) ** 2) / (2 * self.sigma2_t + 1e-8)),
            torch.ones_like(max_da),
        )
        return lam


class ClassAwareContrastiveLoss(nn.Module):
    """
    Implements Eq. 9, 11, 12, 13: a SupCon-style loss between the unified
    representation h and the v-th view-specific representation z^v, with
    positive pairs reweighted by lambda(p_i)*lambda(p_j) (Eq. 11) rather
    than uniformly (Eq. 10), so low-confidence pseudo-labels contribute
    less and confirmation bias is reduced.

    See module docstring for the documented cross-modal-only interpretation.
    """

    def __init__(self, temperature: float = 0.4):
        super().__init__()
        self.tau = temperature

    def forward(self, h: torch.Tensor, z_v: torch.Tensor,
                pseudo_labels: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        """
        h             : [B, D] unified representation (unlabeled batch)
        z_v           : [B, D] v-th view-specific representation (unlabeled batch)
        pseudo_labels : [B]    hard pseudo-label per sample (teacher argmax)
        lam           : [B]    lambda(p_i) confidence weight per sample (Eq. 6)
        """
        B = h.size(0)
        h_n = F.normalize(h, dim=1)
        z_n = F.normalize(z_v, dim=1)

        sim = (h_n @ z_n.t()) / self.tau  # [B, B]: s(h_i, z_v_j), Eq. 9

        same_class = pseudo_labels.unsqueeze(0) == pseudo_labels.unsqueeze(1)
        eye = torch.eye(B, device=h.device, dtype=torch.bool)

        # Eq. 11: 1 on the diagonal (same-instance cross-modal pair),
        # lambda_i*lambda_j on same-category off-diagonal pairs, 0 otherwise
        w_pos = torch.zeros(B, B, device=h.device)
        w_pos[eye] = 1.0
        lam_outer = lam.unsqueeze(1) * lam.unsqueeze(0)
        off_diag_same = same_class & (~eye)
        w_pos[off_diag_same] = lam_outer[off_diag_same]

        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        num_pos = w_pos.sum(dim=1).clamp(min=1e-8)

        loss = -(w_pos * log_prob).sum(dim=1) / num_pos
        return loss.mean()

'''
def sparsity_loss(gates) -> torch.Tensor:
    """Eq. 2 (L1 approximation): sum of L1 norms of the gating vectors w^v."""
    terms = [g.abs().mean() for g in gates if g is not None]
    if len(terms) == 0:
        return torch.tensor(0.0)
    return sum(terms)
'''

def soft_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor,
                        reduction: str = "mean", sample_mask: torch.Tensor = None) -> torch.Tensor:
    """
    H(target_probs, softmax(logits)) for a soft (probability-vector) target,
    used in Eq. 16 (L^s2_kd, teacher target p^t_i is a sum-pooled soft
    distribution rather than a hard label).
    """
    log_probs = F.log_softmax(logits, dim=1)
    per_sample = -(target_probs * log_probs).sum(dim=1)
    if sample_mask is not None:
        per_sample = per_sample * sample_mask
    if reduction == "mean":
        return per_sample.mean()
    elif reduction == "sum":
        return per_sample.sum()
    return per_sample