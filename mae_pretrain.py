"""
mae_pretrain.py -- generic single-modality MAE pretraining (He et al., 2022) of OUR ViT-S encoder (model.ViTEncoder,
patch 8), used ONLY for the Semi-ViT competitor (stage 1 of its pipeline). Agreed with the user 2026-11-22:
  * one run per dataset x modality, on ALL training images (labelled + unlabelled, no labels; test excluded);
    split-independent, reused for every split
  * mask ratio 0.75, normalised-pixel loss (MSE on masked patches), official decoder: 512-d, 8 blocks, 16 heads,
    fixed 2-D sin-cos decoder position embeddings
  * MAE augmentations: random-resized-crop (scale 0.2-1, ratio 3/4-4/3) + horizontal flip
  * AdamW betas (0.9, 0.95), weight decay 0.05 (not on biases / norms / pos-embed / cls / mask token),
    lr = 1.5e-4 * batch / 256, linear warm-up 10% of the run, cosine to 1e-6
  * SAME budget as our pretraining (pretrain_v2): 1000 epochs, batch EUROSAT 1890 / TRISTAR 497 / SUNRGBD 491
  * encoder dropout 0 during MAE (as in MAE); bf16 autocast; gradient accumulation over micro-batches
    (128 images at 128 px, 512 at 64 px) to fit memory -- exact, the effective batch is the agreed one
Output: <dataset>/MAE/<MODALITY>_encoder.pth  (ViTEncoder state_dict, loadable into modality_{1,2}_encoder)
    CUDA_VISIBLE_DEVICES=0 python mae_pretrain.py EUROSAT_STD SAR 1890
"""
import math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from model import ViTEncoder

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

MASK_RATIO = 0.75
DEC_DIM, DEC_DEPTH, DEC_HEADS = 512, 8, 16
EPOCHS = 1000
BASE_LR, MIN_LR, WD, WARMUP_FRAC = 1.5e-4, 1e-6, 0.05, 0.1
PATCH = 8


def sincos_2d(embed_dim, grid):
    """Fixed 2-D sin-cos position embedding (MAE util), with a zero row for the cls token."""
    gh, gw = np.meshgrid(np.arange(grid, dtype=np.float32), np.arange(grid, dtype=np.float32), indexing="xy")

    def one(d, pos):
        omega = 1.0 / 10000 ** (np.arange(d // 2, dtype=np.float64) / (d / 2.0))
        out = np.einsum("m,d->md", pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    emb = np.concatenate([one(embed_dim // 2, gh), one(embed_dim // 2, gw)], axis=1)
    emb = np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)
    return torch.from_numpy(emb).float().unsqueeze(0)


class MAE(nn.Module):
    def __init__(self, img_size, in_chans):
        super().__init__()
        self.encoder = ViTEncoder(img_size=img_size, patch_size=PATCH, in_chans=in_chans, dropout=0.0)
        self.grid = img_size // PATCH
        self.in_chans = in_chans
        E = self.encoder.embed_dim
        self.decoder_embed = nn.Linear(E, DEC_DIM)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, DEC_DIM))
        self.register_buffer("decoder_pos_embed", sincos_2d(DEC_DIM, self.grid))
        layer = nn.TransformerEncoderLayer(DEC_DIM, DEC_HEADS, DEC_DIM * 4, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(layer, num_layers=DEC_DEPTH)
        self.decoder_norm = nn.LayerNorm(DEC_DIM)
        self.decoder_pred = nn.Linear(DEC_DIM, PATCH * PATCH * in_chans)
        nn.init.normal_(self.mask_token, std=0.02)
        for m in [self.decoder_embed, self.decoder_pred, *self.decoder.modules()]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def patchify(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, self.grid, PATCH, self.grid, PATCH)
        return x.permute(0, 2, 4, 3, 5, 1).reshape(B, self.grid * self.grid, PATCH * PATCH * C)

    def forward(self, x):
        enc = self.encoder
        t = enc.patch_embed(x) + enc.pos_embed[:, 1:, :]
        B, N, E = t.shape
        keep = int(N * (1 - MASK_RATIO))
        ids_shuffle = torch.argsort(torch.rand(B, N, device=x.device), dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        t = torch.gather(t, 1, ids_shuffle[:, :keep, None].expand(-1, -1, E))
        cls = (enc.cls_token + enc.pos_embed[:, :1, :]).expand(B, -1, -1)
        z = enc.norm(enc.transformer(torch.cat([cls, t], 1)))
        mask = torch.ones(B, N, device=x.device)
        mask[:, :keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        # decoder
        d = self.decoder_embed(z)
        mt = self.mask_token.expand(B, N - keep, -1)
        d_ = torch.cat([d[:, 1:, :], mt], 1)
        d_ = torch.gather(d_, 1, ids_restore[:, :, None].expand(-1, -1, DEC_DIM))
        d = torch.cat([d[:, :1, :], d_], 1) + self.decoder_pos_embed
        pred = self.decoder_pred(self.decoder_norm(self.decoder(d)))[:, 1:, :].float()
        target = self.patchify(x).float()
        mean, var = target.mean(-1, keepdim=True), target.var(-1, keepdim=True)
        target = (target - mean) / (var + 1e-6) ** 0.5            # norm_pix_loss
        loss = ((pred - target) ** 2).mean(-1)
        return (loss * mask).sum() / mask.sum()


def rrc_flip(x, scale=(0.2, 1.0), ratio=(3 / 4, 4 / 3)):
    """Per-sample random-resized-crop (bilinear, output = input size) + horizontal flip p=0.5, on GPU."""
    B, C, H, W = x.shape
    area = H * W * torch.empty(B, device=x.device).uniform_(*scale)
    logr = torch.empty(B, device=x.device).uniform_(math.log(ratio[0]), math.log(ratio[1]))
    r = torch.exp(logr)
    w = torch.sqrt(area * r).clamp(4, W)
    h = torch.sqrt(area / r).clamp(4, H)
    x0 = torch.rand(B, device=x.device) * (W - w)
    y0 = torch.rand(B, device=x.device) * (H - h)
    boxes = torch.stack([torch.arange(B, device=x.device, dtype=x.dtype), x0, y0, x0 + w, y0 + h], 1)
    out = roi_align(x, boxes, output_size=(H, W), spatial_scale=1.0, sampling_ratio=2, aligned=True)
    flip = torch.rand(B, device=x.device) < 0.5
    out[flip] = out[flip].flip(-1)
    return out


if __name__ == "__main__":
    dataset_path, modality, batch_size = sys.argv[1], sys.argv[2], int(sys.argv[3])
    print(sys.argv, flush=True)
    data = np.load("%s/%s_data_normalized.npy" % (dataset_path, modality))
    train_idx = np.load("%s/train_idx.npy" % dataset_path)
    x_all = torch.tensor(data[train_idx], dtype=torch.float32)
    n, C, H, W = x_all.shape
    device = "cuda"
    x_all = x_all.to(device)
    out_dir = dataset_path + "/MAE"
    os.makedirs(out_dir, exist_ok=True)
    out_file = "%s/%s_encoder.pth" % (out_dir, modality)

    model = MAE(H, C).to(device)
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if (p.ndim < 2 or "pos_embed" in name or "cls_token" in name or "mask_token" in name)
         else decay).append(p)
    lr = BASE_LR * batch_size / 256
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": WD}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.95))
    steps_per_epoch = n // batch_size
    total = EPOCHS * steps_per_epoch
    warm = int(WARMUP_FRAC * total)
    sched = lambda s: (s + 1) / warm if s < warm else MIN_LR / lr + (1 - MIN_LR / lr) * 0.5 * (
        1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, sched)
    print("n=%d C=%d H=%d batch=%d steps/epoch=%d total=%d lr=%.2e enc params %.1fM" % (
        n, C, H, batch_size, steps_per_epoch, total, lr,
        sum(p.numel() for p in model.encoder.parameters()) / 1e6), flush=True)

    micro = 128 if H >= 128 else 512        # micro-batch for memory (~8 GB); effective batch = batch_size
    step, t0 = 0, time.time()
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=device)
        run = torch.zeros((), device=device)
        for i in range(steps_per_epoch):
            idx = perm[i * batch_size:(i + 1) * batch_size]
            opt.zero_grad(set_to_none=True)
            for chunk in idx.split(micro):      # gradient accumulation: exact (no BN), same effective batch
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(rrc_flip(x_all[chunk])) * (len(chunk) / batch_size)
                loss.backward()
                run += loss.detach()
            opt.step()
            scheduler.step()
            step += 1
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print("epoch %d step %d loss %.4f lr %.2e elapsed %.0fs" % (
                epoch, step, run.item() / steps_per_epoch, scheduler.get_last_lr()[0], time.time() - t0), flush=True)
        if epoch % 100 == 99 or epoch == EPOCHS - 1:
            torch.save(model.encoder.state_dict(), out_file)
    print("saved", out_file, flush=True)
