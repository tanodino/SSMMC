"""
sup_init.py -- start a semi-supervised competitor from the trained supervised baseline of the same split
(agreed protocol, 2026-11-20: no pretraining for the competitors, labelled data only).

    <dataset>/<SF|FC>/<perc>_<run>.pth   (EMA weights saved by baselines_main.py)

full model    : FixMatch / FreeMatch / SoftMatch SF|FC (same ScoreFusion / FusionConcat class)
encoders_only : KDMvC / MSC (own fusion heads): only modality_1_encoder / modality_2_encoder, from SF
"""
import torch

ENCODER_PREFIXES = ("modality_1_encoder.", "modality_2_encoder.")


def supervised_checkpoint(dataset_path, fusion, perc, run_id):
    return "%s/%s/%s_%s.pth" % (dataset_path, fusion, perc, run_id)


def load_supervised_init(model, dataset_path, fusion, perc, run_id, device, encoders_only=False):
    """Load the supervised checkpoint into `model` (call BEFORE model.compile()). Strict: any missing or
    unexpected key is an error. Returns the checkpoint path."""
    path = supervised_checkpoint(dataset_path, fusion, perc, run_id)
    state = torch.load(path, map_location=device)
    state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    if encoders_only:
        for prefix in ENCODER_PREFIXES:
            sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            if not sub:
                raise KeyError("no %s* weights in %s" % (prefix, path))
            getattr(model, prefix[:-1]).load_state_dict(sub, strict=True)
        what = "ViT encoders only"
    else:
        model.load_state_dict(state, strict=True)
        what = "full model"
    print("supervised init (%s) loaded from %s" % (what, path), flush=True)
    return path
