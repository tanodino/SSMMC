"""
KDMvC training script -- adapted implementation of:
Wang, X., Wang, Y., Ke, G., Wang, Y., Hong, X. (2024).
"Knowledge distillation-driven semi-supervised multi-view classification."
Information Fusion, 103, 102098.

============================================================================
CONFIRMED AGAINST YOUR ACTUAL model.py (2024-07)
============================================================================
1. ViTEncoder's constructor signature is `(img_size, patch_size, in_chans,
   embed_dim=384, depth=12, num_heads=6, mlp_ratio=4., dropout=0.1)`.
   `build_vit_encoder` below mirrors exactly how your ScoreFusion/
   FusionConcat instantiate it (img_size/patch_size/in_chans only, no
   embed_dim/dropout override), so KDMvC's backbones run at the SAME
   capacity (embed_dim=384) as your other baselines. IMPORTANT: your
   SFFCConfig.hidden_dim is NOT the encoder's embedding dimension in your
   existing baselines either -- it only sizes MLPHead's hidden layer.
   VIT_EMBED_DIM=384 below must be kept in sync if that class default in
   model.py ever changes.
2. ViTEncoder.forward returns `emb[:,0,:]` -- the CLS token only, always a
   2D [B, embed_dim] tensor (not a token sequence). The `if h.dim()==3`
   mean-pool fallback in `ViewSpecificExtractor` (kdmvc_model.py) is
   therefore dead code for this encoder -- left in place defensively, but
   it will never trigger.

============================================================================
DELIBERATE, DOCUMENTED DEVIATION: NO GATE / NO L1 SPARSITY TERM
============================================================================
The paper's E^v(.) (Eq. 1-3) includes a sigmoid informativeness gate and an
L1 sparsity regularizer on it (the "+eta*l1" term in Eq. 14). Both are
motivated by redundancy in hand-crafted feature vectors (GIST, color
moments, etc.) -- a justification that doesn't transfer to a ViT CLS-token
embedding, which is already a trained, compressed representation rather
than a raw descriptor with known-redundant dimensions. This implementation
drops the gate and sparsity term entirely (see kdmvc_model.py's module
docstring for the full reasoning). Everything else in the paper (shared
classification head, fusion, both self-distillation phases, class-aware
contrastive loss) is unaffected, since none of it depends on the gate.

============================================================================
STILL OPEN -- DESIGN DECISIONS, NOT UNKNOWNS
============================================================================
3. No weak/strong augmentation is used on unlabeled data in Phase 1 or
   Phase 2, because the paper does not describe one (unlike FixMatch/
   FreeMatch). If you want consistency-style augmentation added anyway
   (forward on a weakly-augmented view for the teacher signal, strongly-
   augmented for the student loss), this is a deliberate addition beyond
   the paper and should be flagged as such in your writeup.
4. This script trains supervised-only for `WARM_UP_EPOCH_SSL` epochs before
   starting the self-distillation phases (mirroring the warm-up discussion
   from your FreeMatch script) -- the KDMvC paper doesn't specify a warm-up,
   this is an addition for training stability. Set to 0 to disable.

============================================================================
ALGORITHM 1 MAPPING
============================================================================
for epoch in range(max_epochs):
    ---- Phase 1: "multi-view unified to specific" ----
    for batch in (labelled + unlabelled):
        compute L_s1 = L^s1_x + gamma*L^s1_kd + delta*L^ctr_u   [no +eta*l1 -- see above]
        backward, step (all parameters trainable)
    ---- Phase 2: "multi-view specific to unified" ----
    freeze E(.) & F(.) [model.set_specific_trainable(False)]
    for batch in (labelled + unlabelled):
        compute L_s2 = L^s2_x + L^s2_kd
        backward, step (only fusion params receive gradient)
    unfreeze E(.) & F(.) for the next epoch's Phase 1
"""

import itertools
import sys
import time
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score

from functions import evaluation, MOMENTUM_EMA, cumulate_EMA, WARM_UP_EPOCH_EMA, EPOCHS, WARM_UP_EPOCH_SSL, RATIO_LABELED_UNLABELED_BATCHES, load_pretrained_encoders_kdmvc
from model import SFFCConfig, ViTEncoder, KDMvCModel  # ViTEncoder import assumed available here
from kdmvc_losses import KDMvCWeighting, ClassAwareContrastiveLoss, soft_cross_entropy
import copy
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda.amp")

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ----------------------------------------------------------------------
# KDMvC-specific hyperparameters (paper Sec. 4.1.3, adapted where noted)
# ----------------------------------------------------------------------
GAMMA = 0.6          # weight of L^s1_kd in L_s1 (paper: from {1,.8,.6,.4,.2})
DELTA = 1.0          # weight of L^ctr_u in L_s1
# NOTE: no ETA / sparsity weight here -- the gate + L1 sparsity term (Eq. 1-3,
# and the "+eta*l1" part of Eq. 14) is deliberately dropped in this
# implementation. See kdmvc_model.py's module docstring for why.
DA_BETA = 0.99       # EMA momentum for KDMvCWeighting (paper's "beta")
CONTRASTIVE_TAU = 0.4  # paper: tau = 0.4
PHASE2_THRESHOLD = 0.95  # paper: T = 0.95, confidence threshold in Eq. 16
FEAT_DIM = 128           # dimensionality of z^v (view-specific feature)

# CONFIRMED against model.py: ViTEncoder's constructor default is embed_dim=384.
# Your existing ScoreFusion/FusionConcat baselines instantiate ViTEncoder WITHOUT
# passing embed_dim or dropout, so their backbones run at this default (384) --
# config.hidden_dim is only ever used for MLPHead's hidden width in those classes,
# never for the encoder itself. To keep encoder capacity identical across all
# baselines (KDMvC vs. FreeMatch/FreeMatch+SF/FC vs. MSC), we deliberately mirror
# that same instantiation pattern below -- NOT overriding embed_dim/dropout --
# rather than tying it to config.hidden_dim, which would silently under- or
# over-power KDMvC's backbone relative to the others.
VIT_EMBED_DIM = 384  # must stay in sync with ViTEncoder's default in model.py
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def build_vit_encoder(img_size, patch_size, in_chans):
    """
    CONFIRMED against model.py. Mirrors exactly how ScoreFusion/FusionConcat
    construct ViTEncoder (img_size, patch_size, in_chans only), so all
    baselines share the same encoder capacity (embed_dim=384, depth=12,
    num_heads=6, dropout=0.1 -- all class defaults, left untouched).
    """
    return ViTEncoder(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
    )


if __name__ == "__main__":
    batch_size = 16
    dataset_path = sys.argv[1]
    first_prefix = sys.argv[2]
    second_prefix = sys.argv[3]
    perc = sys.argv[4]
    run_id = sys.argv[5]
    pretrained_path = sys.argv[6] if len(sys.argv) > 7 else None   # <-- new, optional
    print(sys.argv)

    first_data = np.load("%s/%s_data_normalized.npy" % (dataset_path, first_prefix))
    second_data = np.load("%s/%s_data_normalized.npy" % (dataset_path, second_prefix))
    print("first_data ",first_data.shape)
    print("second_data ",second_data.shape)
    full_labels = np.load("%s/labels.npy" % dataset_path)
    train_idx = np.load("%s/train_idx.npy" % dataset_path)
    labelled_idx = np.load("%s/labelled_samples_%s_%s.npy" % (dataset_path, perc, run_id))

    full_train_idx = np.arange(len(train_idx))
    unlabelled_idx = np.setdiff1d(full_train_idx, labelled_idx)

    f_lab_data_train = first_data[train_idx][labelled_idx]
    s_lab_data_train = second_data[train_idx][labelled_idx]
    f_unlab_data_train = first_data[train_idx][unlabelled_idx]
    s_unlab_data_train = second_data[train_idx][unlabelled_idx]

    labels = full_labels[train_idx][labelled_idx]

    print("f_lab_data_train %d" % len(f_lab_data_train))
    print("f_unlab_data_train %d" % len(f_unlab_data_train))

    n_classes = len(np.unique(labels))

    dir_name = dataset_path + "/KDMvC"
    os.makedirs(dir_name, exist_ok=True)
    output_file = dir_name+"/%s_%s.pth"%(perc, run_id)

    print("batch_size %d" % batch_size)

    # ---------------- TEST DATA ----------------
    test_idx = np.setdiff1d(np.arange(full_labels.shape[0]), train_idx)
    f_data_test = first_data[test_idx]
    s_data_test = second_data[test_idx]
    labels_test = full_labels[test_idx]

    x_tensor_f_test = torch.tensor(f_data_test, dtype=torch.float32)
    x_tensor_s_test = torch.tensor(s_data_test, dtype=torch.float32)
    y_tensor_test = torch.tensor(labels_test, dtype=torch.int64)
    test_dataset = TensorDataset(x_tensor_f_test, x_tensor_s_test, y_tensor_test)
    dataloader_test = DataLoader(test_dataset, shuffle=False, batch_size=batch_size*8,
        num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=4, drop_last=False)
    print("TEST DATA built")
    sys.stdout.flush()

    # ---------------- TRAIN DATA ----------------
    x_tensor_f_lab = torch.tensor(f_lab_data_train, dtype=torch.float32)
    x_tensor_s_lab = torch.tensor(s_lab_data_train, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.int64)
    lab_dataset = TensorDataset(x_tensor_f_lab, x_tensor_s_lab, y_tensor)

    x_tensor_f_unl = torch.tensor(f_unlab_data_train, dtype=torch.float32)
    x_tensor_s_unl = torch.tensor(s_unlab_data_train, dtype=torch.float32)
    unl_dataset = TensorDataset(x_tensor_f_unl, x_tensor_s_unl)

    dataloader_lab_train = DataLoader(lab_dataset, shuffle=True, batch_size=batch_size,
        num_workers=0, pin_memory=True, drop_last=False)

    dataloader_unl_train = DataLoader(unl_dataset, shuffle=True, batch_size=batch_size * RATIO_LABELED_UNLABELED_BATCHES,
        num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=4, drop_last=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("all dataloader built")
    sys.stdout.flush()

    config = SFFCConfig(
        img_size_m1=f_lab_data_train.shape[2], img_size_m2=s_lab_data_train.shape[2],
        patch_size_m1=8, patch_size_m2=8,
        in_chans_m1=f_lab_data_train.shape[1], in_chans_m2=s_lab_data_train.shape[1],
        num_classes=n_classes, hidden_dim=256, dropout=0.1
    )
    vit_m1 = build_vit_encoder(config.img_size_m1, config.patch_size_m1, config.in_chans_m1)
    vit_m2 = build_vit_encoder(config.img_size_m2, config.patch_size_m2, config.in_chans_m2)

    model = KDMvCModel(
        vit_encoder_m1=vit_m1, vit_encoder_m2=vit_m2,
        embed_dim=VIT_EMBED_DIM, feat_dim=FEAT_DIM, hidden_dim=FEAT_DIM,
        num_classes=n_classes,
    ).to(device)
    print("model created")
    sys.stdout.flush()

    if pretrained_path is not None:                                    # <-- new
        load_pretrained_encoders_kdmvc(model, pretrained_path, device)       # <-- new

    model.compile()
    weighting = KDMvCWeighting(num_classes=n_classes, beta=DA_BETA, device=device)
    contrastive_loss_fn = ClassAwareContrastiveLoss(temperature=CONTRASTIVE_TAU)
    loss_fn = nn.CrossEntropyLoss()

    # single optimizer over ALL parameters; Phase 2's freeze is enforced via
    # requires_grad (frozen params simply receive no gradient / don't move),
    # matching Algorithm 1's "frozen parameters in E(.) & F(.)" instruction.
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scaler = GradScaler()

    step = 0
    ema_weights = None

    for epoch in range(EPOCHS):
        start_time = time.time()
        total_loss = torch.zeros((), device=device) 
        use_ssl = epoch >= WARM_UP_EPOCH_SSL

        # ================================================================
        # PHASE 1: "multi-view unified to specific" (Algorithm 1, lines 2-12)
        # ================================================================
        model.train()
        model.set_specific_trainable(True)

        n_batches = 0
        for (f_batch, s_batch, y_batch), (f_batch_unl, s_batch_unl) in zip(
                itertools.cycle(dataloader_lab_train), dataloader_unl_train):

            optimizer.zero_grad(set_to_none=True)
            f_batch = f_batch.to(device, non_blocking=True)
            s_batch = s_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            with autocast():
                out_l = model(f_batch, s_batch)
                # Eq. 4: L^s1_x -- supervised CE on both fused and per-view predictions
                loss_s1_x = loss_fn(out_l["p_fusion"], y_batch)
                for p_v in out_l["p_views"]:
                    loss_s1_x = loss_s1_x + loss_fn(p_v, y_batch)

                loss_s1 = loss_s1_x

                if use_ssl:
                    f_batch_unl = f_batch_unl.to(device, non_blocking=True)
                    s_batch_unl = s_batch_unl.to(device, non_blocking=True)

                    out_u = model(f_batch_unl, s_batch_unl)
                    p_fusion_u_probs = torch.softmax(out_u["p_fusion"].detach(), dim=1)

                    # update DA + confidence-EMA trackers (Eq. 6-8)
                    with torch.no_grad():
                        labeled_probs = torch.softmax(out_l["p_fusion"].detach(), dim=1)
                    weighting.update_label_distributions(
                        probs_labeled=labeled_probs, probs_unlabeled=p_fusion_u_probs)
                    weighting.update_confidence_stats(p_fusion_u_probs)
                    lam = weighting.weight(p_fusion_u_probs)          # [B_u], Eq. 6
                    pseudo_labels_u = torch.argmax(p_fusion_u_probs, dim=1)  # y_hat_i

                    # Eq. 5: L^s1_kd -- teacher (fusion) -> student (each view)
                    loss_s1_kd = 0.0
                    for p_v_u in out_u["p_views"]:
                        ce_v = F.cross_entropy(p_v_u, pseudo_labels_u, reduction="none")
                        loss_s1_kd = loss_s1_kd + (lam * ce_v).mean()

                    # Eq. 9-13: class-aware contrastive loss between h and each z^v
                    loss_ctr_u = 0.0
                    for z_v_u in out_u["z_views"]:
                        loss_ctr_u = loss_ctr_u + contrastive_loss_fn(
                            out_u["h"], z_v_u, pseudo_labels_u, lam)

                    loss_s1 = loss_s1 + GAMMA * loss_s1_kd + DELTA * loss_ctr_u


            scaler.scale(loss_s1).backward()
            scaler.step(optimizer)
            scaler.update()
            #total_loss += loss_s1.item()
            total_loss += loss_s1.detach()
            n_batches += 1


        # ================================================================
        # PHASE 2: "multi-view specific to unified" (Algorithm 1, lines 13-17)
        # ================================================================
        if use_ssl:
            model.set_specific_trainable(False)

            for (f_batch, s_batch, y_batch), (f_batch_unl, s_batch_unl) in zip(
                    itertools.cycle(dataloader_lab_train), dataloader_unl_train):

                optimizer.zero_grad(set_to_none=True)
                f_batch = f_batch.to(device, non_blocking=True)
                s_batch = s_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                f_batch_unl = f_batch_unl.to(device, non_blocking=True)
                s_batch_unl = s_batch_unl.to(device, non_blocking=True)
                with autocast():
                    out_l = model(f_batch, s_batch)
                    loss_s2_x = loss_fn(out_l["p_fusion"], y_batch)  # L^s2_x

                    out_u = model(f_batch_unl, s_batch_unl)
                    with torch.no_grad():
                        # Eq. 15: sum-pooled multi-teacher target, normalized by
                        # the number of views so p_t is a valid probability
                        # distribution (raw sum-of-softmaxes sums to V, not 1,
                        # which would make PHASE2_THRESHOLD too permissive and
                        # inflate L^s2_kd's scale relative to L^s2_x).
                        num_views = len(out_u["p_views"])
                        p_t = sum(torch.softmax(p_v, dim=1) for p_v in out_u["p_views"]) / num_views
                        max_p_t = p_t.max(dim=1).values
                        mask_T = (max_p_t >= PHASE2_THRESHOLD).float()

                    # Eq. 16: L^s2_kd -- soft target p_t supervises the fusion head
                    loss_s2_kd = soft_cross_entropy(out_u["p_fusion"], p_t, sample_mask=mask_T)

                    loss_s2 = loss_s2_x + loss_s2_kd  # Eq. 17
                
                scaler.scale(loss_s2).backward()
                scaler.step(optimizer)
                scaler.update()
                #total_loss += loss_s2.item()
                total_loss += loss_s2.detach()
                n_batches += 1

            model.set_specific_trainable(True)  # unfreeze for next epoch's Phase 1

        elapsed_time = time.time() - start_time

        # ---------------- evaluation (unchanged pattern from your other scripts) ----------------
        if epoch >= WARM_UP_EPOCH_EMA:
            ema_weights = cumulate_EMA(model, ema_weights, MOMENTUM_EMA)

        if epoch % 5 == 0:
            if epoch >= WARM_UP_EPOCH_EMA:
                current_state_dict = copy.deepcopy(model.state_dict())
                model.load_state_dict(ema_weights)
                predictions, test_labels = evaluation(model, dataloader_test, device)
                model.load_state_dict(current_state_dict)
            else:
                predictions, test_labels = evaluation(model, dataloader_test, device)

            f1_val = f1_score(test_labels, predictions, average="weighted")
            total_loss = total_loss.item()  

            print(f"epoch {epoch} | phase2={'on' if use_ssl else 'off (warmup)'} "
                f"total={np.mean(total_loss / max(n_batches, 1)):.4f} "
                f"elapsed_time={elapsed_time:.2f} "
                f"F1-score={(f1_val * 100):.2f}")
            sys.stdout.flush()
    model.load_state_dict(ema_weights)
    torch.save(model.state_dict(), output_file)