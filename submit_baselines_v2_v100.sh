#!/bin/bash
# Submit the CORRECTED FreeMatch and SoftMatch baselines on Jean Zay (V100):
#   freematch_baselines_main_v2.py  (fairness term as in the official USB code)
#   softmatch_baselines_main_v2.py  (adds Uniform Alignment, as in the official USB code)
# SF and FC, standardised data, from scratch. Each sbatch = one configuration x 5 splits (array 0-4):
#   2 methods x 3 datasets x 2 fusions x 3 label budgets = 36 jobs = 180 tasks.
#
#   cd $WORK/SSMMC/SSMMC && bash submit_baselines_v2_v100.sh     # submit
#   bash submit_baselines_v2_v100.sh --dry                       # only print the commands
#
# Results: <DATASET>_STD/{FREEMATCH_V2_SF,FREEMATCH_V2_FC,SOFTMATCH_V2_SF,SOFTMATCH_V2_FC}/<perc>_<run>.pth
# Evaluate with:
#   python eval_baselines_table.py --methods FREEMATCH_V2_SF FREEMATCH_V2_FC SOFTMATCH_V2_SF SOFTMATCH_V2_FC --out results/baselines_v2

DRY=0; [ "$1" == "--dry" ] && DRY=1
mkdir -p slurm_logs

SCRIPTS="freematch_baselines_main_v2.py softmatch_baselines_main_v2.py"
DATASETS=("SUNRGBD_STD RGB DEPTH" "EUROSAT_STD SAR MS" "TRISTAR_STD DEPTH THERMAL")

n=0
for script in $SCRIPTS; do
  for ds in "${DATASETS[@]}"; do
    for fusion in SF FC; do
      for perc in 5 25 50; do
        cmd="sbatch --job-name=${script%%_*}_v2_${fusion}_${ds%%_*}_${perc} baselines_array_v100.sh $script $ds $perc $fusion"
        if [ $DRY -eq 1 ]; then echo "$cmd"; else $cmd; fi
        n=$((n + 1))
      done
    done
  done
done
echo "$n array jobs ($((n * 5)) tasks)"
