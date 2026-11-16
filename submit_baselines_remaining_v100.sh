#!/bin/bash
# Submit the REMAINING main-table baselines on Jean Zay (V100): FixMatch and FreeMatch,
# SF and FC, standardised data, from scratch (Supervised + SoftMatch are in
# submit_baselines_v100.sh). Each sbatch = one configuration x 5 splits (array 0-4):
#   2 methods x 3 datasets x 2 fusions x 3 label budgets = 36 jobs = 180 tasks.
#
#   cd $WORK/SSMMC/SSMMC && bash submit_baselines_remaining_v100.sh     # submit
#   bash submit_baselines_remaining_v100.sh --dry                       # only print the commands
#
# Results: <DATASET>_STD/{FIXMATCH_SF,FIXMATCH_FC,FREEMATCH_SF,FREEMATCH_FC}/<perc>_<run>.pth
# Evaluate with: python restore_baselines.py <DATASET>_STD <FOLDER> <MOD1> <MOD2> <PERC> <SF|FC>

DRY=0; [ "$1" == "--dry" ] && DRY=1
mkdir -p slurm_logs

SCRIPTS="fixmatch_baselines_main.py freematch_baselines_main.py"
DATASETS=("SUNRGBD_STD RGB DEPTH" "EUROSAT_STD SAR MS" "TRISTAR_STD DEPTH THERMAL")

n=0
for script in $SCRIPTS; do
  for ds in "${DATASETS[@]}"; do
    for fusion in SF FC; do
      for perc in 5 25 50; do
        cmd="sbatch --job-name=${script%%_*}_${fusion}_${ds%%_*}_${perc} baselines_array_v100.sh $script $ds $perc $fusion"
        if [ $DRY -eq 1 ]; then echo "$cmd"; else $cmd; fi
        n=$((n + 1))
      done
    done
  done
done
echo "$n array jobs ($((n * 5)) tasks)"
