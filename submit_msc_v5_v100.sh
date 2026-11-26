#!/bin/bash
# MSC competitor (decision 2026-11-26: reported, with its collapse explained) = MSC v5: starts from the RAW per-modality
# MAE encoders (<DS>/MAE/<MOD>_encoder.pth), differential lr (encoders 5e-6, heads 1e-4, wd 1e-4), as INDIVIDUAL jobs:
#   task_v100.sh plain msc_main_v5.py <DS> <m1> <m2> <perc> <run>   (one retry); no torch.compile in msc_main_v5.py
#   3 datasets x 3 label budgets x 5 splits = 45 jobs; --time SUNRGBD 4 h, TRISTAR / EUROSAT 20 h
#   (local pilot, RTX 3090 shared by 2 jobs: SUNRGBD ~1.5 h, TRISTAR ~5 h, EUROSAT ~10 h for 400 epochs)
# Tasks whose result exists or that are already queued are skipped. Safe to run again.
#   cd $WORK/SSMMC/SSMMC && bash submit_msc_v5_v100.sh --dry   # only print the commands
#   bash submit_msc_v5_v100.sh                                 # submit
# Results: <DS>_STD/MSC_V5/<perc>_<run>.pth
cd $WORK/SSMMC/SSMMC
DRY=0; [ "$1" == "--dry" ] && DRY=1
mkdir -p slurm_logs
queued=$(squeue -u $USER -h -o "%j" 2>/dev/null)
n=0; failed=()
for ds in "SUNRGBD_STD RGB DEPTH 04:00:00" "TRISTAR_STD DEPTH THERMAL 20:00:00" "EUROSAT_STD SAR MS 20:00:00"; do
  set -- $ds; d=$1; m1=$2; m2=$3; t=$4
  for m in $m1 $m2; do [ -f $d/MAE/${m}_encoder.pth ] || echo "WARNING: missing MAE encoder $d/MAE/${m}_encoder.pth"; done
  for perc in 5 25 50; do for run in 0 1 2 3 4; do
    [ -f $d/MSC_V5/${perc}_${run}.pth ] && continue
    name=msc5_t_${d%%_*}_${perc}_${run}
    echo "$queued" | grep -qx "$name" && continue
    cmd="sbatch --time=$t --job-name=$name task_v100.sh plain msc_main_v5.py $d $m1 $m2 $perc $run"
    if [ $DRY -eq 1 ]; then echo "$cmd"; else $cmd || failed+=("$cmd"); fi
    n=$((n + 1))
  done; done
done
echo "$n job(s)$([ $DRY -eq 1 ] && echo ' (dry run, nothing submitted)'), ${#failed[@]} refused"
for c in "${failed[@]}"; do echo "REFUSED: $c"; done
