#!/bin/bash
# Submit the Supervised and FixMatch baselines on Jean Zay (V100), SF and FC,
# standardised data, from scratch. Each sbatch = one configuration x 5 splits (array 0-4):
#   2 methods x 3 datasets x 2 fusions x 3 label budgets = 36 jobs = 180 tasks.
# (FreeMatch v2 / SoftMatch v2 are in submit_baselines_v2_v100.sh.)
#
#   cd $WORK/SSMMC/SSMMC && bash submit_baselines_sup_fixmatch_v100.sh     # submit
#   bash submit_baselines_sup_fixmatch_v100.sh --dry                       # only print the commands
#
# Results: <DATASET>_STD/{SF,FC,FIXMATCH_SF,FIXMATCH_FC}/<perc>_<run>.pth
# Any sbatch refused (e.g. per-user job limit) is listed at the end: re-run those later.

DRY=0; [ "$1" == "--dry" ] && DRY=1
mkdir -p slurm_logs

SCRIPTS="baselines_main.py fixmatch_baselines_main.py"
DATASETS=("SUNRGBD_STD RGB DEPTH" "EUROSAT_STD SAR MS" "TRISTAR_STD DEPTH THERMAL")

n=0; failed=()
for script in $SCRIPTS; do
  for ds in "${DATASETS[@]}"; do
    for fusion in SF FC; do
      for perc in 5 25 50; do
        cmd="sbatch --job-name=${script%%_*}_${fusion}_${ds%%_*}_${perc} baselines_array_v100.sh $script $ds $perc $fusion"
        if [ $DRY -eq 1 ]; then echo "$cmd"; else $cmd || failed+=("$cmd"); fi
        n=$((n + 1))
      done
    done
  done
done
echo "$n array jobs ($((n * 5)) tasks), ${#failed[@]} refused"
for c in "${failed[@]}"; do echo "REFUSED: $c"; done
