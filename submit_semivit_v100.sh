#!/bin/bash
# Submit the Semi-ViT competitor (Cai et al., NeurIPS 2022; MAE per modality -> supervised -> EMA teacher +
# probabilistic pseudo mixup; agreed 2026-11-22) on Jean Zay (V100). Each sbatch = one configuration x 5 splits
# (array 0-4), run by semivit_array_v100.sh (stage 2 + stage 3):
#   3 datasets x 2 fusions (SF, FC) x 3 label budgets = 18 jobs = 90 tasks.
# Needs the 6 MAE encoders <DATASET>_STD/MAE/<MOD>_encoder.pth (mae_encoders.tar, see jz_commands_full.csv).
#
#   cd $WORK/SSMMC/SSMMC && bash submit_semivit_v100.sh --dry   # only print the commands
#   bash submit_semivit_v100.sh                                 # submit
#
# Results: <DATASET>_STD/{SF,FC}_MAE/<perc>_<run>.pth (supervised, MAE start)
#          <DATASET>_STD/SEMIVIT_{SF,FC}/<perc>_<run>.pth (Semi-ViT, EMA teacher)
# Any sbatch refused (e.g. per-user job limit) is listed at the end: re-run those later.

DRY=0; [ "$1" == "--dry" ] && DRY=1
mkdir -p slurm_logs

DATASETS=("SUNRGBD_STD RGB DEPTH" "EUROSAT_STD SAR MS" "TRISTAR_STD DEPTH THERMAL")

n=0; failed=()
for ds in "${DATASETS[@]}"; do
  for fusion in SF FC; do
    for perc in 5 25 50; do
      cmd="sbatch --job-name=semivit_${fusion}_${ds%%_*}_${perc} semivit_array_v100.sh $ds $perc $fusion"
      if [ $DRY -eq 1 ]; then echo "$cmd"; else $cmd || failed+=("$cmd"); fi
      n=$((n + 1))
    done
  done
done
echo "$n array jobs ($((n * 5)) tasks), ${#failed[@]} refused"
for c in "${failed[@]}"; do echo "REFUSED: $c"; done
