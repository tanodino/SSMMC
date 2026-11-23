#!/bin/bash
# Submit FixMatch-WRN (FixMatch with its original backbone, one WRN-28-2 per modality, score fusion, from scratch,
# AdamW lr 1e-4 as all approaches; fixmatch_wrn_main.py, agreed 2026-11-21/22) on Jean Zay (V100).
# Each sbatch = one configuration x 5 splits (array 0-4), run by baselines_array_v100.sh (no fusion argument):
#   3 datasets x 3 label budgets = 9 jobs = 45 tasks.
#
#   cd $WORK/SSMMC/SSMMC && bash submit_fixmatch_wrn_v100.sh --dry   # only print the commands
#   bash submit_fixmatch_wrn_v100.sh                                 # submit
#
# Results: <DATASET>_STD/FIXMATCH_WRN_SF/<perc>_<run>.pth (EMA weights)
# Any sbatch refused (e.g. per-user job limit) is listed at the end: re-run those later.

DRY=0; [ "$1" == "--dry" ] && DRY=1
mkdir -p slurm_logs

DATASETS=("SUNRGBD_STD RGB DEPTH" "EUROSAT_STD SAR MS" "TRISTAR_STD DEPTH THERMAL")

n=0; failed=()
for ds in "${DATASETS[@]}"; do
  for perc in 5 25 50; do
    cmd="sbatch --job-name=fmwrn_${ds%%_*}_${perc} baselines_array_v100.sh fixmatch_wrn_main.py $ds $perc"
    if [ $DRY -eq 1 ]; then echo "$cmd"; else $cmd || failed+=("$cmd"); fi
    n=$((n + 1))
  done
done
echo "$n array jobs ($((n * 5)) tasks), ${#failed[@]} refused"
for c in "${failed[@]}"; do echo "REFUSED: $c"; done
