"""
Ablation A1 (agreed 2026-11-26): our fine-tuning (ssl_pretrained_classif_v7.py) WITHOUT pretraining.

Runs the source of ssl_pretrained_classif_v7.py UNCHANGED except one line: the pretrain_v2 checkpoint is not
loaded, so encoders, projectors, cross-modal heads, fusion and classifier all start from random init
(v7's own initialisation, same seed). Losses, views, multi-layer fusion, EMA, epochs, batch sizes, evaluation
and saving are exactly those of v7. The learning rates are v7's flags (--backbone-lr, --fresh-lr); the pilot
compares v7's rates (5e-6 / 5e-5) with 1e-4 / 1e-4 (rate of the supervised from-scratch baselines).

Usage (no checkpoint argument):
    python ssl_pretrained_classif_v7_scratch.py TRISTAR_STD DEPTH THERMAL 5 0 [v7 flags]
        default --output_dir OURS_V7_SCRATCH, e.g. for the pilot:
    ... --output_dir A1_V7LR                                         (v7 rates)
    ... --backbone-lr 1e-4 --fresh-lr 1e-4 --output_dir A1_LR1E4     (scratch rates)
"""
import os
import sys

V7 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ssl_pretrained_classif_v7.py")
LOAD_LINE = "    load_pretrain_v2_checkpoint(model, args.checkpoint_path, device)\n"
NO_LOAD = ('    print("A1 ablation: NO pretraining -- pretrain_v2 checkpoint NOT loaded, '
           'all modules from random init", flush=True)\n')

src = open(V7).read()
if src.count(LOAD_LINE) != 1:
    sys.exit(f"cannot build the A1 variant: the checkpoint-loading line occurs {src.count(LOAD_LINE)} times in {V7}")
src = src.replace(LOAD_LINE, NO_LOAD)

# v7's parser needs a checkpoint_path positional: insert a placeholder after the 5 positionals
argv = sys.argv[1:]
if "--self_check" not in argv:
    if len(argv) < 5 or any(a.startswith("-") for a in argv[:5]):
        sys.exit("usage: ssl_pretrained_classif_v7_scratch.py DATASET MOD1 MOD2 PERC RUN [v7 flags]")
    argv = argv[:5] + ["NO_PRETRAINING"] + argv[5:]
    if not any(a.startswith("--output_dir") for a in argv):
        argv += ["--output_dir", "OURS_V7_SCRATCH"]
sys.argv = [V7] + argv
exec(compile(src, V7 + " (A1: no pretraining)", "exec"), {"__name__": "__main__", "__file__": V7})
