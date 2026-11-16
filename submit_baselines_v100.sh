#!/bin/bash
# Submit the main-table baselines on Jean Zay (V100), standardised data, from scratch.
# Each sbatch = one configuration x 5 splits (array 0-4)  ->  4 x 3 x 3 = 36 jobs = 180 tasks.
#
#   cd $WORK/SSMMC/SSMMC && bash submit_baselines_v100.sh          # submit
#   bash submit_baselines_v100.sh --dry                            # only print the sbatch commands
#
# Results: <DATASET>_STD/{SF,FC,SOFTMATCH_SF,SOFTMATCH_FC}/<perc>_<run>.pth (as before);
# evaluate with restore_baselines.py as for the previous results.

DRY=0; [ "$1" == "--dry" ] && DRY=1
mkdir -p slurm_logs

SCRIPTS="baselines_main.py softmatch_baselines_main.py"
# later, for the full table:  SCRIPTS="$SCRIPTS fixmatch_baselines_main.py freematch_baselines_main.py"

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
