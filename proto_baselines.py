"""
Prototype-based semi-supervised classifier.

Design (see conversation for full reasoning):
  1. Fine-tune the pretrained encoders with a SUPERVISED CONTRASTIVE loss
     (Khosla et al. 2020) on the labeled set only -- same-class embeddings
     are positive pairs (pulled together), different-class embeddings are
     negative pairs (pushed apart), in the SAME fused embedding space the
     contrastive pretraining already built. NO augmentation is used here --
     positive pairs come from distinct real labeled samples sharing a
     class, not from two augmented views of the same sample. Because of
     this, the labeled set is trained as ONE FULL BATCH per step (rather
     than shuffled mini-batches): with only ~50 labeled samples spread
     across classes, a small mini-batch risks anchors whose class doesn't
     appear elsewhere in that batch, giving them zero positives and zero
     learning signal for that step. Full-batch guarantees every anchor
     sees every same-class sample, every step.
  2. Classification is NON-PARAMETRIC: a test sample's label is decided by
     k-NEAREST-NEIGHBOR (distance-weighted) vote against the labeled
     embeddings directly -- no averaging into per-class prototypes, no
     trained linear head, no random initialization to bootstrap from.
     This is the key difference from every pseudo-labeling baseline
     (FixMatch/FreeMatch/SoftMatch/MSC/KDMvC) tried so far, all of which
     rely on a classifier head that starts uncalibrated and can generate
     wrong, confidently-held pseudo-labels.

This is the "simple" version: NO pseudo-labeling of the unlabeled pool yet.
It's meant as a direct, apples-to-apples comparison against
"frozen pretrained encoder + FC fine-tune, no SSL" (your best result so
far), to test whether SupCon fine-tuning + KNN classification alone --
without touching the unlabeled pool at all -- already matches or beats it.
If it does, extending this to a PAWS-style soft-pseudo-label consistency
loss on the unlabeled pool (comparing unlabeled embeddings to the SAME
labeled reference set) is the natural next step -- see the note at the
bottom of this file.

Related work: this is close in spirit to PAWS (Assran et al., ICCV 2021,
"Semi-Supervised Learning of Visual Features by Non-Parametrically
Predicting View Assignments with Support Samples"), which extends
self-supervised distance-metric learning (BYOL/SwAV-style) to the
semi-supervised setting via non-parametric comparisons against labeled
support samples rather than a trained classifier head.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score

from model import SFFCConfig, ViTEncoder
from functions import EPOCHS

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ==========================================================================
# Model: two encoders -> per-modality projection -> fused, L2-normalized embedding
# ==========================================================================

class ProtoModel(nn.Module):
    """Same encoder/projector structure as PretrainModel, but forward()
    returns a single fused, L2-normalized embedding (concatenation of the
    two modality projections) -- the space SupCon and prototypes both
    operate in."""

    def __init__(self, config: SFFCConfig, proj_dim: int = 128):
        super().__init__()
        self.modality_1_encoder = ViTEncoder(
            img_size=config.img_size_m1, patch_size=config.patch_size_m1,
            in_chans=config.in_chans_m1,
        )
        self.modality_2_encoder = ViTEncoder(
            img_size=config.img_size_m2, patch_size=config.patch_size_m2,
            in_chans=config.in_chans_m2,
        )
        self.projector_m1 = nn.Sequential(
            nn.LazyLinear(512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, proj_dim),
        )
        self.projector_m2 = nn.Sequential(
            nn.LazyLinear(512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, proj_dim),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        cls_m1 = self.modality_1_encoder(x1)   # [B, embed_dim], already CLS token
        cls_m2 = self.modality_2_encoder(x2)
        emb = torch.cat([self.projector_m1(cls_m1), self.projector_m2(cls_m2)], dim=1)
        return F.normalize(emb, dim=1)          # [B, 2*proj_dim], unit norm


# ==========================================================================
# Loss: Supervised Contrastive (Khosla et al., 2020)
# ==========================================================================

def supervised_contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor,
                                 temperature: float = 0.1) -> torch.Tensor:
    """embeddings: [N, D], ALREADY L2-normalized. labels: [N] int64.
    Every same-class sample (excluding self) is a positive; loss is the
    mean, per anchor, of -log P(positive) over all its positives."""
    device = embeddings.device
    N = embeddings.shape[0]

    sim = embeddings @ embeddings.T / temperature          # [N, N]
    self_mask = torch.eye(N, dtype=torch.bool, device=device)

    sim_max = sim.masked_fill(self_mask, float("-inf")).max(dim=1, keepdim=True).values
    sim = sim - sim_max.detach()                             # numerical stability
    exp_sim = torch.exp(sim) * (~self_mask)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.T) & (~self_mask)
    pos_counts = pos_mask.sum(dim=1).clamp(min=1)

    loss = -(pos_mask * log_prob).sum(dim=1) / pos_counts
    return loss.mean()


# ==========================================================================
# k-NN classification against labeled embeddings (no prototypes, no head)
# ==========================================================================

@torch.no_grad()
def compute_reference_embeddings(model: ProtoModel, f_lab: torch.Tensor, s_lab: torch.Tensor,
                                  device) -> torch.Tensor:
    """Embeds the (small) labeled set once -- reused as the KNN reference
    bank. Cheap: one forward pass over 50 samples."""
    model.eval()
    return model(f_lab.to(device), s_lab.to(device))          # [L, D], L2-normalized


@torch.no_grad()
def knn_classify(query_emb: torch.Tensor, ref_emb: torch.Tensor, ref_labels: torch.Tensor,
                  n_classes: int, k: int = 5) -> torch.Tensor:
    """query_emb: [Nq, D], ref_emb: [Nr, D], both L2-normalized -> cosine
    similarity via dot product. Distance-WEIGHTED majority vote: each of
    the k nearest labeled samples contributes its similarity score to its
    class's total, rather than a flat +1 -- ties broken naturally, and
    closer neighbors count more, which matters when k spans samples of
    noticeably different similarity."""
    k = min(k, ref_emb.shape[0])
    sims = query_emb @ ref_emb.T                                # [Nq, Nr]
    topk_sims, topk_idx = sims.topk(k, dim=1)                    # [Nq, k]
    topk_labels = ref_labels[topk_idx]                            # [Nq, k]

    class_scores = torch.zeros(query_emb.shape[0], n_classes, device=query_emb.device)
    class_scores.scatter_add_(1, topk_labels, topk_sims.clamp(min=0))
    return class_scores.argmax(dim=1)


@torch.no_grad()
def evaluate_with_knn(model: ProtoModel, ref_emb: torch.Tensor, ref_labels: torch.Tensor,
                       dataloader, n_classes: int, device, k: int = 5):
    model.eval()
    all_preds, all_labels = [], []
    for f_batch, s_batch, y_batch in dataloader:
        f_batch = f_batch.to(device, non_blocking=True)
        s_batch = s_batch.to(device, non_blocking=True)
        emb = model(f_batch, s_batch)
        preds = knn_classify(emb, ref_emb, ref_labels, n_classes, k=k).cpu()
        all_preds.append(preds)
        all_labels.append(y_batch)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


# ==========================================================================
# Pretrained-encoder loading / freezing (same pattern as baselines_main.py)
# ==========================================================================

def load_pretrained_encoders(model: ProtoModel, path: str, device: str, strict: bool = True):
    encoders_state = torch.load(path, map_location=device)
    model.modality_1_encoder.load_state_dict(encoders_state["modality_1"], strict=strict)
    model.modality_2_encoder.load_state_dict(encoders_state["modality_2"], strict=strict)
    print("Loaded pretrained encoder weights from %s" % path)


def freeze_pretrained_encoders(model: ProtoModel):
    for p in model.modality_1_encoder.parameters():
        p.requires_grad = False
    for p in model.modality_2_encoder.parameters():
        p.requires_grad = False


if __name__ == "__main__":
    batch_size = 16
    dataset_path = sys.argv[1]
    first_prefix = sys.argv[2]
    second_prefix = sys.argv[3]
    perc = sys.argv[4]
    run_id = sys.argv[5]
    pretrained_path = sys.argv[6] if len(sys.argv) > 6 else None
    freeze_encoder = "freeze" in sys.argv
    K_NEIGHBORS = 5  # with 5 labels/class, k=5 ~= "vote across roughly one full class"; tune freely

    first_data = np.load("%s/%s_data_normalized.npy" % (dataset_path, first_prefix))
    second_data = np.load("%s/%s_data_normalized.npy" % (dataset_path, second_prefix))
    full_labels = np.load("%s/labels.npy" % dataset_path)
    train_idx = np.load("%s/train_idx.npy" % dataset_path)
    labelled_idx = np.load("%s/labelled_samples_%s_%s.npy" % (dataset_path, perc, run_id))

    f_lab_data_train = first_data[train_idx][labelled_idx]
    s_lab_data_train = second_data[train_idx][labelled_idx]
    labels = full_labels[train_idx][labelled_idx]
    n_classes = len(np.unique(labels))

    dir_name = dataset_path + "/PROTO"
    os.makedirs(dir_name, exist_ok=True)
    output_file = dir_name + "/%s_%s.pth" % (perc, run_id)

    print("f_lab_data_train %d" % len(f_lab_data_train))
    print("n_classes %d" % n_classes)

    # ---------------- TEST DATA ----------------
    test_idx = np.setdiff1d(np.arange(full_labels.shape[0]), train_idx)
    f_data_test = first_data[test_idx]
    s_data_test = second_data[test_idx]
    labels_test = full_labels[test_idx]

    test_dataset = TensorDataset(
        torch.tensor(f_data_test, dtype=torch.float32),
        torch.tensor(s_data_test, dtype=torch.float32),
        torch.tensor(labels_test, dtype=torch.int64),
    )
    dataloader_test = DataLoader(test_dataset, shuffle=False, batch_size=batch_size * 8,
        num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=4, drop_last=False)
    print("TEST DATA built")
    sys.stdout.flush()

    # ---------------- LABELED DATA (only -- no unlabeled loader in this simple version) ----------------
    x_tensor_f_lab = torch.tensor(f_lab_data_train, dtype=torch.float32)
    x_tensor_s_lab = torch.tensor(s_lab_data_train, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.int64)
    lab_dataset = TensorDataset(x_tensor_f_lab, x_tensor_s_lab, y_tensor)

    dataloader_lab_train = DataLoader(lab_dataset, shuffle=True, batch_size=len(lab_dataset),
        num_workers=0, pin_memory=True, drop_last=False)
    # NOTE: batch_size == full labeled set size -- every training step sees
    # ALL labeled samples at once (no augmentation to guarantee positives
    # otherwise, see module docstring). shuffle=True is a no-op with only
    # one possible batch, kept for consistency with the other scripts.

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("all dataloader built")
    sys.stdout.flush()

    config = SFFCConfig(
        img_size_m1=f_lab_data_train.shape[2], img_size_m2=s_lab_data_train.shape[2],
        patch_size_m1=8, patch_size_m2=8,
        in_chans_m1=f_lab_data_train.shape[1], in_chans_m2=s_lab_data_train.shape[1],
        num_classes=n_classes, hidden_dim=256, dropout=0.1
    )

    model = ProtoModel(config).to(device)

    if pretrained_path is not None:
        load_pretrained_encoders(model, pretrained_path, device)
        if freeze_encoder:
            freeze_pretrained_encoders(model)
            print("Pretrained encoders FROZEN")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

    print("model created")
    sys.stdout.flush()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for f_batch, s_batch, y_batch in dataloader_lab_train:
            optimizer.zero_grad(set_to_none=True)
            f_batch = f_batch.to(device, non_blocking=True)
            s_batch = s_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            # No augmentation: embed the real labeled batch once, and use
            # the actual class labels directly for SupCon's positive/
            # negative structure (same class = positive, different = negative).
            emb = model(f_batch, s_batch)
            loss = supervised_contrastive_loss(emb, y_batch, temperature=0.1)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if epoch % 5 == 0:
            ref_emb = compute_reference_embeddings(model, x_tensor_f_lab, x_tensor_s_lab, device)
            ref_labels = y_tensor.to(device)
            predictions, test_labels = evaluate_with_knn(
                model, ref_emb, ref_labels, dataloader_test, n_classes, device, k=K_NEIGHBORS)
            f1_val = f1_score(test_labels, predictions, average="weighted")
            print(f"epoch {epoch} total={total_loss / max(n_batches, 1):.4f} F1-score={(f1_val * 100):.2f}")
            sys.stdout.flush()

    torch.save(model.state_dict(), output_file)
    print("Saved to %s" % output_file)

# ==========================================================================
# Next step, once this version is validated: extend to a PAWS-style
# consistency signal on unlabeled data. Sketch (no augmentation, per this
# script's design):
#   - compute soft pseudo-labels for an unlabeled batch as a temperature-
#     scaled softmax over similarity-to-each-labeled-reference-sample
#     (aggregated by class, as in knn_classify, but kept soft/unargmax'd)
#   - if you want a consistency signal without augmented views, one option
#     is enforcing agreement between the k-NN vote and a second, independent
#     estimate (e.g. the OTHER modality's embedding alone, or a dropout-
#     perturbed forward pass) rather than two image augmentations
#   - crucially: keep it non-parametric (still comparing against the
#     labeled reference set, not a trained head) to preserve the property
#     that made this version robust in the first place
# ==========================================================================