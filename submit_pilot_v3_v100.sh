#!/bin/bash
# PILOT of the agreed protocol (2026-11-20): semi-supervised competitors started from the supervised
# baseline of the same split (no pretraining). FixMatch SF (full SF model) and MSC (ViT encoders from SF),
# 5 labelled samples per class, splits 0 and 1, on the three datasets: 6 sbatch x 2 tasks = 12 tasks.
# Needs the supervised checkpoints <DS>_STD/SF/5_{0,1}.pth (already on Jean Zay).
#   cd $WORK/SSMMC/SSMMC && bash submit_pilot_v3_v100.sh --dry   # only print the commands
#   bash submit_pilot_v3_v100.sh                                 # submit
# Results: <DS>_STD/FIXMATCH_V3_SF/5_{0,1}.pth and <DS>_STD/MSC_V3/5_{0,1}.pth
# Logs: slurm_logs/fixmatch_v3_SF_<DS>_5_*.out / msc_v3_<DS>_5_*.out (F1, mask_rate, majority_frac every 5 epochs)
cd $WORK/SSMMC/SSMMC
DRY=0; [ "$1" == "--dry" ] && DRY=1
mkdir -p slurm_logs
DATASETS=("SUNRGBD_STD RGB DEPTH" "EUROSAT_STD SAR MS" "TRISTAR_STD DEPTH THERMAL")
for ds in "${DATASETS[@]}"; do
  d=${ds%% *}
  for r in 0 1; do [ -f $d/SF/5_$r.pth ] || echo "WARNING: missing supervised checkpoint $d/SF/5_$r.pth"; done
  for cmd in "sbatch --array=0,1 --job-name=fixmatch_v3_SF_${d%%_*}_5 baselines_array_v100.sh fixmatch_baselines_main_v3.py $ds 5 SF" \
             "sbatch --array=0,1 --job-name=msc_v3_${d%%_*}_5 baselines_array_v100.sh msc_main_v3.py $ds 5"; do
    echo "$cmd"; [ $DRY -eq 0 ] && $cmd
  done
done
