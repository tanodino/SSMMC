"""
semisst_main.py -- Semi-SST (SST: Self-training with Self-adaptive Thresholding, Zhao et al., IP&M 2025; the
headline variant of the SST framework) for our two-modality SF / FC models. New script (semivit_main.py unchanged).
Recipe agreed with the user 2026-11-23:
  * start: supervised checkpoint <DS>/<SF|FC>_MAE/<perc>_<run>.pth (MAE per modality -> supervised), as Semi-ViT
  * K = 4 cycles x 100 epochs (400 epochs in total, same budget as Semi-ViT); the student and its EMA teacher are
    initialised ONCE from the supervised checkpoint (paper Algorithm 1, step 1) and CONTINUE from one cycle to the
    next (Algorithm 1 trains the same model at every cycle; agreed 2026-11-23, replaces the restart option)
  * at the start of each cycle, the current EMA teacher (cycle 1: the supervised start) predicts the whole
    unlabelled set (clean view, fused prediction) and
      - Self-Adaptive Thresholding (SAT, paper Eqs. 3-6): for each class j, tau_j = S * mean{ P[i,j] : P[i,j] > C },
        C = 0.5, S = 0.8 (paper values for ViT-S); a class with no probability > C gets no pseudo-label (tau_j = inf)
      - offline pseudo-labels: unlabelled samples with max prob > tau_(argmax) get that hard label for the whole
        cycle and are merged into the labelled set ("treated equally")
  * online (Semi-SST): EMA teacher (0.999 per step) on the weak view -> hard pseudo-labels, kept if conf > tau_(label)
    (same per-class thresholds of the cycle); student on the strong view; hard cross-entropy (paper Eq. 11, no pseudo
    mixup); loss = CE(combined labelled batch) + mu * L_u, mu = 5; batches 16 (combined labelled) + 80 unlabelled
  * optimiser as the other MAE/ImageNet-start runs: AdamW, encoders 5e-6, heads 1e-4, weight decay 1e-4, constant
  * our weak_augment_pair / strong_augment_pair; evaluation and checkpoint = EMA teacher of the last cycle
Output: <DS>/SEMISST_<SF|FC>/<perc>_<run>.pth
    python semisst_main.py SUNRGBD_STD RGB DEPTH 5 0 SF
"""
import itertools, os, sys, time, copy, warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score
from functions import evaluation, weak_augment_pair, strong_augment_pair
from model import SFFCConfig, ScoreFusion, FusionConcat
from sup_init import load_supervised_init
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

N_CYCLES = 4
EPOCHS_PER_CYCLE = 100
SAT_C = 0.5
SAT_S = 0.8
LAMBDA_U = 5.0
MU = 5
BATCH_LAB = 16
EMA_DECAY = 0.999


def probs_of(out, sf):
    return out.float() if sf else F.softmax(out.float(), dim=-1)


def log_probs_of(out, sf):
    return torch.log(out.float().clamp(min=1e-8)) if sf else F.log_softmax(out.float(), dim=-1)


@torch.no_grad()
def ema_update(teacher, student, decay):
    for pt, ps in zip(teacher.parameters(), student.parameters()):
        pt.mul_(decay).add_(ps.detach(), alpha=1 - decay)
    for bt, bs in zip(teacher.buffers(), student.buffers()):
        bt.copy_(bs)


@torch.no_grad()
def predict_probs(model, f, s, sf, device, bs=96):
    model.eval()
    out = []
    for i in range(0, len(f), bs):
        fb = torch.tensor(f[i:i + bs], dtype=torch.float32, device=device)
        sb = torch.tensor(s[i:i + bs], dtype=torch.float32, device=device)
        with autocast():
            o = model(fb, sb)
        out.append(probs_of(o, sf).cpu())
    return torch.cat(out)


def sat_thresholds(P, C, S):
    """SAT (paper Eqs. 3-6): per-class threshold S * mean of the class-j probabilities above C (inf if none)."""
    tau = torch.full((P.shape[1],), float("inf"))
    for j in range(P.shape[1]):
        col = P[:, j]
        sel = col[col > C]
        if sel.numel() > 0:
            tau[j] = S * sel.mean()
    return tau


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
    y_unl_true = full_labels[train_idx][unlabelled_idx]          # logging only (accuracy of offline pseudo-labels)
    n_classes = len(np.unique(labels))
    print("labelled %d unlabelled %d classes %d" % (len(f_lab), len(f_unl), n_classes), flush=True)

    dir_name = dataset_path + "/SEMISST_%s" % sf_or_fc
    os.makedirs(dir_name, exist_ok=True)
    output_file = dir_name + "/%s_%s.pth" % (perc, run_id)

    test_idx = np.setdiff1d(np.arange(full_labels.shape[0]), train_idx)
    t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt)
    dataloader_test = DataLoader(TensorDataset(t(first_data[test_idx]), t(second_data[test_idx]),
                                               t(full_labels[test_idx], torch.int64)),
                                 shuffle=False, batch_size=96, num_workers=6, pin_memory=True,
                                 persistent_workers=True, prefetch_factor=4)
    dataloader_unl = DataLoader(TensorDataset(t(f_unl), t(s_unl)), shuffle=True, batch_size=BATCH_LAB * MU,
                                num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=4,
                                drop_last=True)
    device = "cuda"
    config = SFFCConfig(img_size_m1=f_lab.shape[2], img_size_m2=s_lab.shape[2], patch_size_m1=8, patch_size_m2=8,
                        in_chans_m1=f_lab.shape[1], in_chans_m2=s_lab.shape[1], num_classes=n_classes,
                        hidden_dim=256, dropout=0.1)

    def fresh_model():
        m = (ScoreFusion(config) if sf else FusionConcat(config)).to(device)
        load_supervised_init(m, dataset_path, sf_or_fc + "_MAE", perc, run_id, device)
        return m

    student = fresh_model()                                       # Algorithm 1, step 1: supervised initialisation
    teacher = copy.deepcopy(student).eval()                       # EMA teacher = cycle-1 pseudo-labeller
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

    epoch_global = 0
    for cycle in range(N_CYCLES):
        # ---- offline step: SAT thresholds + offline pseudo-labels from the current EMA teacher ----
        P = predict_probs(teacher, f_unl, s_unl, sf, device)
        tau = sat_thresholds(P, SAT_C, SAT_S)
        conf, yhat = P.max(dim=1)
        keep = conf > tau[yhat]
        n_keep = int(keep.sum())
        acc_keep = float((yhat[keep].numpy() == y_unl_true[keep.numpy()]).mean() * 100) if n_keep else float("nan")
        finite = tau[torch.isfinite(tau)]
        print(f"cycle {cycle} offline: selected {n_keep}/{len(f_unl)} ({100 * n_keep / len(f_unl):.1f}%) "
              f"pseudo-label acc {acc_keep:.1f}% | tau mean {finite.mean().item() if finite.numel() else float('nan'):.3f} "
              f"classes without threshold {int((~torch.isfinite(tau)).sum())} | tau {[round(x, 3) for x in tau.tolist()]}",
              flush=True)
        keep_np = keep.numpy()
        f_comb = np.concatenate([f_lab, f_unl[keep_np]])
        s_comb = np.concatenate([s_lab, s_unl[keep_np]])
        y_comb = np.concatenate([labels, yhat[keep].numpy()])
        dataloader_lab = DataLoader(TensorDataset(t(f_comb), t(s_comb), t(y_comb, torch.int64)), shuffle=True,
                                    batch_size=BATCH_LAB, num_workers=0, pin_memory=True, drop_last=False)
        tau_dev = tau.to(device)

        # ---- online step: student / teacher / optimiser CONTINUE from the previous cycle ----
        for epoch in range(EPOCHS_PER_CYCLE):
            start_time = time.time()
            total_loss = torch.zeros((), device=device)
            n_batches, mask_sum, maj_sum, maj_n = 0, 0.0, 0.0, 0
            for (f_b, s_b, y_b), (f_u, s_u) in zip(itertools.cycle(dataloader_lab), dataloader_unl):
                student.train()
                f_b, s_b, y_b = f_b.to(device, non_blocking=True), s_b.to(device, non_blocking=True), y_b.to(device, non_blocking=True)
                f_u, s_u = f_u.to(device, non_blocking=True), s_u.to(device, non_blocking=True)
                with torch.no_grad():
                    f_w, s_w = weak_augment_pair(f_u, s_u)
                    f_s, s_s = strong_augment_pair(f_u, s_u)
                    with autocast():
                        out_t = teacher(f_w, s_w)
                    c_t, pseudo = probs_of(out_t, sf).max(dim=-1)
                    mask = (c_t > tau_dev[pseudo]).float()
                optimizer.zero_grad(set_to_none=True)
                with autocast():
                    sup_loss = F.nll_loss(log_probs_of(student(f_b, s_b), sf), y_b)
                    unsup_loss = (F.nll_loss(log_probs_of(student(f_s, s_s), sf), pseudo, reduction="none") * mask).mean()
                    loss = sup_loss + LAMBDA_U * unsup_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                ema_update(teacher, student, EMA_DECAY)
                total_loss += loss.detach()
                n_batches += 1
                mask_sum += mask.mean().item()
                acc = pseudo[mask.bool()]
                if acc.numel() > 0:
                    maj_sum += (acc.bincount(minlength=n_classes).max() / acc.numel()).item()
                    maj_n += 1
            elapsed_time = time.time() - start_time
            if epoch_global % 5 == 0 or epoch == EPOCHS_PER_CYCLE - 1:
                predictions, test_labels = evaluation(teacher, dataloader_test, device)
                f1_val = f1_score(test_labels, predictions, average="weighted")
                print(f"epoch {epoch_global} cycle {cycle} total={total_loss.item() / max(n_batches, 1):.4f} "
                      f"elapsed_time={elapsed_time:.2f} F1-score={f1_val * 100:.2f} "
                      f"mask_rate={mask_sum / max(n_batches, 1):.3f} majority_frac={maj_sum / max(maj_n, 1):.3f}",
                      flush=True)
            epoch_global += 1

    torch.save(teacher.state_dict(), output_file)
    print("saved", output_file, flush=True)
