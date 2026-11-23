"""
semivit_main.py -- Semi-ViT (Cai et al., NeurIPS 2022) stage 3: semi-supervised fine-tuning with an EMA teacher and
probabilistic pseudo mixup, for our two-modality SF / FC models (agreed with the user 2026-11-22).
Pipeline: (1) mae_pretrain.py (per-modality MAE)  (2) baselines_main_v5.py (supervised <SF|FC>_MAE)  (3) this script.
  * student initialised with the full supervised checkpoint <DS>/<SF|FC>_MAE/<perc>_<run>.pth; teacher = EMA of the
    student (decay 0.999 per step, seeded with the same weights), eval mode (no dropout)
  * pseudo-labels: teacher on the weak view (fused prediction); confidence o = max prob, label y = argmax
  * probabilistic pseudo mixup (Semi-ViT Sec. 3.3): strong views of the unlabelled batch mixed with a random partner j,
    lambda_i = o_i / (o_i + o_j) (same lambda and same partner for both modalities), target = lambda*onehot(y_i) +
    (1-lambda)*onehot(y_j), kept if max(o_i, o_j) >= 0.6; soft cross-entropy
  * loss = supervised CE (labelled, not augmented, as in our other ViT competitors; labelled mixup off as in the
    paper's 1% runs) + 5 * unlabelled loss; batches 16 labelled + 80 unlabelled (mu = 5)
  * optimiser = the ImageNet/MAE-start runs: AdamW, encoders 5e-6 / heads 1e-4, weight decay 1e-4, constant
  * 400 epochs, our weak_augment_pair / strong_augment_pair; evaluation and checkpoint = EMA teacher
Output: <DS>/SEMIVIT_<SF|FC>/<perc>_<run>.pth
    python semivit_main.py SUNRGBD_STD RGB DEPTH 5 0 SF
"""
import itertools, os, sys, time, copy, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score
from functions import evaluation, weak_augment_pair, strong_augment_pair, EPOCHS
from model import SFFCConfig, ScoreFusion, FusionConcat
from sup_init import load_supervised_init
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

THRESHOLD = 0.6
LAMBDA_U = 5.0
MU = 5
BATCH_LAB = 16
EMA_DECAY = 0.999


def log_probs_of(out, sf):
    return torch.log(out.float().clamp(min=1e-8)) if sf else F.log_softmax(out.float(), dim=-1)


@torch.no_grad()
def ema_update(teacher, student, decay):
    for pt, ps in zip(teacher.parameters(), student.parameters()):
        pt.mul_(decay).add_(ps.detach(), alpha=1 - decay)
    for bt, bs in zip(teacher.buffers(), student.buffers()):
        bt.copy_(bs)


if __name__ == "__main__":
    dataset_path, first_prefix, second_prefix, perc, run_id, sf_or_fc = sys.argv[1:7]
    print(sys.argv, flush=True)
    sf = sf_or_fc == "SF"
    first_data = np.load("%s/%s_data_normalized.npy" % (dataset_path, first_prefix))
    second_data = np.load("%s/%s_data_normalized.npy" % (dataset_path, second_prefix))
    full_labels = np.load("%s/labels.npy" % dataset_path)
    train_idx = np.load("%s/train_idx.npy" % dataset_path)
    labelled_idx = np.load("%s/labelled_samples_%s_%s.npy" % (dataset_path, perc, run_id))
    unlabelled_idx = np.setdiff1d(np.arange(len(train_idx)), labelled_idx)
    f_lab, s_lab = first_data[train_idx][labelled_idx], second_data[train_idx][labelled_idx]
    f_unl, s_unl = first_data[train_idx][unlabelled_idx], second_data[train_idx][unlabelled_idx]
    labels = full_labels[train_idx][labelled_idx]
    n_classes = len(np.unique(labels))
    print("labelled %d unlabelled %d classes %d" % (len(f_lab), len(f_unl), n_classes), flush=True)

    dir_name = dataset_path + "/SEMIVIT_%s" % sf_or_fc
    os.makedirs(dir_name, exist_ok=True)
    output_file = dir_name + "/%s_%s.pth" % (perc, run_id)

    test_idx = np.setdiff1d(np.arange(full_labels.shape[0]), train_idx)
    t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt)
    dataloader_test = DataLoader(TensorDataset(t(first_data[test_idx]), t(second_data[test_idx]),
                                               t(full_labels[test_idx], torch.int64)),
                                 shuffle=False, batch_size=96, num_workers=6, pin_memory=True,
                                 persistent_workers=True, prefetch_factor=4)
    dataloader_lab = DataLoader(TensorDataset(t(f_lab), t(s_lab), t(labels, torch.int64)), shuffle=True,
                                batch_size=BATCH_LAB, num_workers=0, pin_memory=True, drop_last=False)
    dataloader_unl = DataLoader(TensorDataset(t(f_unl), t(s_unl)), shuffle=True, batch_size=BATCH_LAB * MU,
                                num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=4,
                                drop_last=True)
    device = "cuda"

    config = SFFCConfig(img_size_m1=f_lab.shape[2], img_size_m2=s_lab.shape[2], patch_size_m1=8, patch_size_m2=8,
                        in_chans_m1=f_lab.shape[1], in_chans_m2=s_lab.shape[1], num_classes=n_classes,
                        hidden_dim=256, dropout=0.1)
    student = (ScoreFusion(config) if sf else FusionConcat(config)).to(device)
    load_supervised_init(student, dataset_path, sf_or_fc + "_MAE", perc, run_id, device)
    teacher = copy.deepcopy(student).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    predictions, test_labels = evaluation(teacher, dataloader_test, device)
    print(f"loaded supervised model: F1-score={f1_score(test_labels, predictions, average='weighted') * 100:.2f}",
          flush=True)

    enc, head = [], []
    for name, p in student.named_parameters():
        (enc if ("modality_1_encoder" in name or "modality_2_encoder" in name) else head).append(p)
    optimizer = torch.optim.AdamW([{"params": head, "lr": 1e-4}, {"params": enc, "lr": 5e-6}], weight_decay=1e-4)
    scaler = GradScaler()

    for epoch in range(EPOCHS):
        start_time = time.time()
        total_loss = torch.zeros((), device=device)
        n_batches, mask_sum, maj_sum, maj_n, conf_sum = 0, 0.0, 0.0, 0, 0.0
        for (f_b, s_b, y_b), (f_u, s_u) in zip(itertools.cycle(dataloader_lab), dataloader_unl):
            student.train()
            f_b, s_b, y_b = f_b.to(device, non_blocking=True), s_b.to(device, non_blocking=True), y_b.to(device, non_blocking=True)
            f_u, s_u = f_u.to(device, non_blocking=True), s_u.to(device, non_blocking=True)
            with torch.no_grad():
                f_w, s_w = weak_augment_pair(f_u, s_u)
                f_s, s_s = strong_augment_pair(f_u, s_u)
                with autocast():
                    out_t = teacher(f_w, s_w)
                probs_t = out_t.float() if sf else F.softmax(out_t.float(), dim=-1)
                conf, pseudo = probs_t.max(dim=-1)
                # probabilistic pseudo mixup
                perm = torch.randperm(f_u.shape[0], device=device)
                lam = conf / (conf + conf[perm])
                lam_x = lam.view(-1, 1, 1, 1)
                f_mix = lam_x * f_s + (1 - lam_x) * f_s[perm]
                s_mix = lam_x * s_s + (1 - lam_x) * s_s[perm]
                onehot = F.one_hot(pseudo, n_classes).float()
                target = lam.view(-1, 1) * onehot + (1 - lam.view(-1, 1)) * onehot[perm]
                mask = (torch.maximum(conf, conf[perm]) >= THRESHOLD).float()
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                out_l = student(f_b, s_b)
                sup_loss = F.nll_loss(log_probs_of(out_l, sf), y_b)
                out_u = student(f_mix, s_mix)
                unsup_loss = ((-(target * log_probs_of(out_u, sf)).sum(-1)) * mask).mean()
                loss = sup_loss + LAMBDA_U * unsup_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema_update(teacher, student, EMA_DECAY)
            total_loss += loss.detach()
            n_batches += 1
            mask_sum += mask.mean().item()
            conf_sum += conf.mean().item()
            acc = pseudo[conf >= THRESHOLD]
            if acc.numel() > 0:
                maj_sum += (acc.bincount(minlength=n_classes).max() / acc.numel()).item()
                maj_n += 1
        elapsed_time = time.time() - start_time
        if epoch % 5 == 0:
            predictions, test_labels = evaluation(teacher, dataloader_test, device)
            f1_val = f1_score(test_labels, predictions, average="weighted")
            print(f"epoch {epoch} total={total_loss.item() / max(n_batches, 1):.4f} "
                  f"elapsed_time={elapsed_time:.2f} F1-score={f1_val * 100:.2f} "
                  f"mask_rate={mask_sum / max(n_batches, 1):.3f} majority_frac={maj_sum / max(maj_n, 1):.3f} "
                  f"mean_conf={conf_sum / max(n_batches, 1):.3f}", flush=True)

    torch.save(teacher.state_dict(), output_file)
