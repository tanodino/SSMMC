#!/bin/bash
# Resubmit ONLY the baseline tasks whose checkpoint <DS>_STD/<FOLDER>/<perc>_<run>.pth is missing:
# one sbatch per configuration, with --array set to the missing split numbers (e.g. --array=0,1,4).
# Same job names and same baselines_array_v100.sh as the original submissions.
# Configurations that still have a job in the queue are skipped.
#   cd $WORK/SSMMC/SSMMC && bash jz_resubmit_missing.sh --dry    # only print the sbatch commands
#   bash jz_resubmit_missing.sh                                  # submit
# Everything printed is also written to jz_resubmit_missing.txt.
cd $WORK/SSMMC/SSMMC
DRY=0; [ "$1" == "--dry" ] && DRY=1
mkdir -p slurm_logs
DATASETS=("SUNRGBD_STD RGB DEPTH" "EUROSAT_STD SAR MS" "TRISTAR_STD DEPTH THERMAL")
# folder prefix : training script : job-name prefix
METHODS=(":baselines_main.py:baselines" "FIXMATCH_:fixmatch_baselines_main.py:fixmatch"
         "FREEMATCH_V2_:freematch_baselines_main_v2.py:freematch_v2"
         "SOFTMATCH_V2_:softmatch_baselines_main_v2.py:softmatch_v2")
queued=$(squeue -u $USER -h -o "%j" 2>/dev/null)
{
n=0; ntask=0; failed=()
for m in "${METHODS[@]}"; do
  prefix=$(echo "$m" | cut -d: -f1); script=$(echo "$m" | cut -d: -f2); jprefix=$(echo "$m" | cut -d: -f3)
  for ds in "${DATASETS[@]}"; do
    dsdir=${ds%% *}
    for fusion in SF FC; do
      for perc in 5 25 50; do
        runs=""
        for run in 0 1 2 3 4; do
          [ -f $dsdir/${prefix}${fusion}/${perc}_${run}.pth ] || runs="$runs,$run"
        done
        [ -z "$runs" ] && continue
        runs=${runs#,}
        jname=${jprefix}_${fusion}_${dsdir%%_*}_${perc}
        if echo "$queued" | grep -qx "$jname"; then echo "SKIPPED (still in queue): $jname runs $runs"; continue; fi
        cmd="sbatch --array=$runs --job-name=$jname baselines_array_v100.sh $script $ds $perc $fusion"
        if [ $DRY -eq 1 ]; then echo "$cmd"; else echo "$cmd"; $cmd || failed+=("$cmd"); fi
        n=$((n + 1)); ntask=$((ntask + $(echo $runs | tr ',' '\n' | wc -l)))
      done
    done
  done
done
echo "$n array jobs, $ntask tasks$([ $DRY -eq 1 ] && echo ' (dry run, nothing submitted)'), ${#failed[@]} refused"
for c in "${failed[@]}"; do echo "REFUSED: $c"; done
} 2>&1 | tee jz_resubmit_missing.txt
