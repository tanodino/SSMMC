#!/bin/bash
# Submit the multimodal competitors on Jean Zay (V100), v2 versions:
#   kdmvc_training_v2.py  (KDMvC: contrastive loss as in the paper Eq. 9-12, lr 1e-4)
#   msc_main_v2.py        (MSC, FixMatch+MSC variant: lr 1e-4)
# standardised data, from scratch. Each sbatch = one configuration x 5 splits (array 0-4),
# run by baselines_array_v100.sh (no fusion argument for these two methods):
#   2 methods x 3 datasets x 3 label budgets = 18 jobs = 90 tasks.
#
#   cd $WORK/SSMMC/SSMMC && bash submit_competitors_v2_v100.sh     # submit
#   bash submit_competitors_v2_v100.sh --dry                       # only print the commands
#
# Results: <DATASET>_STD/{KDMvC_V2,MSC_V2}/<perc>_<run>.pth
# Evaluate: python kdmvc_restore_v2.py <DATASET>_STD <MOD1> <MOD2> <PERC>
#           python msc_restore_v2.py   <DATASET>_STD <MOD1> <MOD2> <PERC>
# Any sbatch refused (e.g. per-user job limit) is listed at the end: re-run those later.

DRY=0; [ "$1" == "--dry" ] && DRY=1
mkdir -p slurm_logs

SCRIPTS="kdmvc_training_v2.py msc_main_v2.py"
DATASETS=("SUNRGBD_STD RGB DEPTH" "EUROSAT_STD SAR MS" "TRISTAR_STD DEPTH THERMAL")

n=0; failed=()
for script in $SCRIPTS; do
  for ds in "${DATASETS[@]}"; do
    for perc in 5 25 50; do
      cmd="sbatch --job-name=${script%%_*}_v2_${ds%%_*}_${perc} baselines_array_v100.sh $script $ds $perc"
      if [ $DRY -eq 1 ]; then echo "$cmd"; else $cmd || failed+=("$cmd"); fi
      n=$((n + 1))
    done
  done
done
echo "$n array jobs ($((n * 5)) tasks), ${#failed[@]} refused"
for c in "${failed[@]}"; do echo "REFUSED: $c"; done
