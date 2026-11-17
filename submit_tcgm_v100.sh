#!/bin/bash
# Submit the TCGM competitor (Sun et al., ECCV 2020) on Jean Zay (V100): tcgm_main.py,
# standardised data, from scratch. Each sbatch = one configuration x 5 splits (array 0-4),
# run by baselines_array_v100.sh (no fusion argument: TCGM uses its own aggregator):
#   3 datasets x 3 label budgets = 9 jobs = 45 tasks.
#
#   cd $WORK/SSMMC/SSMMC && bash submit_tcgm_v100.sh     # submit
#   bash submit_tcgm_v100.sh --dry                       # only print the commands
#
# Results: <DATASET>_STD/TCGM/<perc>_<run>.pth
# Evaluate: python tcgm_restore.py <DATASET>_STD <MOD1> <MOD2> <PERC>
# Any sbatch refused (e.g. per-user job limit) is listed at the end: re-run those later.

DRY=0; [ "$1" == "--dry" ] && DRY=1
mkdir -p slurm_logs

DATASETS=("SUNRGBD_STD RGB DEPTH" "EUROSAT_STD SAR MS" "TRISTAR_STD DEPTH THERMAL")

n=0; failed=()
for ds in "${DATASETS[@]}"; do
  for perc in 5 25 50; do
    cmd="sbatch --job-name=tcgm_${ds%%_*}_${perc} baselines_array_v100.sh tcgm_main.py $ds $perc"
    if [ $DRY -eq 1 ]; then echo "$cmd"; else $cmd || failed+=("$cmd"); fi
    n=$((n + 1))
  done
done
echo "$n array jobs ($((n * 5)) tasks), ${#failed[@]} refused"
for c in "${failed[@]}"; do echo "REFUSED: $c"; done
