#!/bin/bash
# Before tomorrow's maintenance: submit as INDIVIDUAL jobs (< 10 h each) only the tasks that surely fit
# (agreed 2026-11-23), so they can start before the maintenance window:
#   FixMatch-WRN, all 3 datasets x 3 label budgets x 5 splits = 45 jobs
#        (SUNRGBD --time 4h; TRISTAR / EUROSAT --time 9:59 -- estimated 4-9 h on a V100, the longest may not fit)
#   Semi-ViT SUNRGBD, SF + FC x 3 label budgets x 5 splits      = 30 jobs (--time 6h; stage 2 + stage 3)
# The long Semi-ViT TRISTAR / EUROSAT tasks stay in the 20 h arrays (submit_semivit_v100.sh) and run after the
# maintenance. Cancel the pending duplicate arrays FIRST (see jz_commands_full.csv), then run this script.
#
#   cd $WORK/SSMMC/SSMMC && bash submit_short_tasks_v100.sh --dry   # only print the commands
#   bash submit_short_tasks_v100.sh                                 # submit
# Results: same folders as the arrays. Refused sbatch (per-user job limit) are listed at the end.

DRY=0; [ "$1" == "--dry" ] && DRY=1
mkdir -p slurm_logs
DATASETS=("SUNRGBD_STD RGB DEPTH" "EUROSAT_STD SAR MS" "TRISTAR_STD DEPTH THERMAL")

n=0; failed=()
sub() { if [ $DRY -eq 1 ]; then echo "$*"; else "$@" || failed+=("$*"); fi; n=$((n + 1)); }

for ds in "${DATASETS[@]}"; do
  name=${ds%%_*}
  t=09:59:00; [ "$name" == "SUNRGBD" ] && t=04:00:00
  for perc in 5 25 50; do
    for run in 0 1 2 3 4; do
      sub sbatch --time=$t --job-name=fmwrn_t_${name}_${perc}_${run} task_v100.sh plain fixmatch_wrn_main.py $ds $perc $run
    done
  done
done
ds="SUNRGBD_STD RGB DEPTH"
for fusion in SF FC; do
  for perc in 5 25 50; do
    for run in 0 1 2 3 4; do
      sub sbatch --time=06:00:00 --job-name=semivit_t_${fusion}_SUNRGBD_${perc}_${run} task_v100.sh semivit $ds $perc $run $fusion
    done
  done
done
echo "$n individual jobs, ${#failed[@]} refused"
for c in "${failed[@]}"; do echo "REFUSED: $c"; done
