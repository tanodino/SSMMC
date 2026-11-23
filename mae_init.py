"""mae_init.py -- load the per-modality MAE encoders (mae_pretrain.py, <DS>/MAE/<MOD>_encoder.pth) into
modality_1_encoder / modality_2_encoder of a ScoreFusion / FusionConcat model. Semi-ViT competitor only
(agreed 2026-11-22). Strict loading: the MAE encoder is our ViTEncoder (patch 8), same shapes."""
import torch


def load_mae_encoders(model, dataset_path, first_prefix, second_prefix, device="cuda"):
    for enc, mod in ((model.modality_1_encoder, first_prefix), (model.modality_2_encoder, second_prefix)):
        path = "%s/MAE/%s_encoder.pth" % (dataset_path, mod)
        enc.load_state_dict(torch.load(path, map_location=device), strict=True)
        print("MAE encoder loaded from %s" % path, flush=True)
