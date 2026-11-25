#!/bin/bash
# Semi-SST competitor (FC start, agreed 2026-11-25) on Jean Zay as INDIVIDUAL jobs: semisst_main.py <DS> <m1> <m2> <perc> <run> FC
# (4 cycles x 100 epochs, continue across cycles, SAT thresholds C=0.5 S=0.8), run by task_v100.sh plain (one retry).
#   3 datasets x 3 label budgets x 5 splits = 45 jobs; --time SUNRGBD 4 h, TRISTAR / EUROSAT 20 h
# Semi-SST starts from the supervised FC-from-MAE checkpoint <DS>/FC_MAE/<perc>_<run>.pth (produced by the Semi-ViT FC
# jobs): a task is submitted ONLY when that checkpoint exists; the others are listed as "waiting" -> run the script again
# later. Tasks whose result exists or that are already queued are skipped. Safe to run again.
#   cd $WORK/SSMMC/SSMMC && bash submit_semisst_v100.sh --dry   # only print the commands
#   bash submit_semisst_v100.sh                                 # submit
# Results: <DS>_STD/SEMISST_FC/<perc>_<run>.pth
cd $WORK/SSMMC/SSMMC
DRY=0; [ "$1" == "--dry" ] && DRY=1
mkdir -p slurm_logs
queued=$(squeue -u $USER -h -o "%j" 2>/dev/null)
n=0; nwait=0; waiting=""; failed=()
for ds in "SUNRGBD_STD RGB DEPTH 04:00:00" "TRISTAR_STD DEPTH THERMAL 20:00:00" "EUROSAT_STD SAR MS 20:00:00"; do
  set -- $ds; d=$1; m1=$2; m2=$3; t=$4
  for perc in 5 25 50; do for run in 0 1 2 3 4; do
    [ -f $d/SEMISST_FC/${perc}_${run}.pth ] && continue
    name=semisst_t_${d%%_*}_${perc}_${run}
    echo "$queued" | grep -qx "$name" && continue
    if [ ! -f $d/FC_MAE/${perc}_${run}.pth ]; then waiting="$waiting ${d%%_*}_${perc}_${run}"; nwait=$((nwait + 1)); continue; fi
    cmd="sbatch --time=$t --job-name=$name task_v100.sh plain semisst_main.py $d $m1 $m2 $perc $run FC"
    if [ $DRY -eq 1 ]; then echo "$cmd"; else $cmd || failed+=("$cmd"); fi
    n=$((n + 1))
  done; done
done
echo "$n job(s)$([ $DRY -eq 1 ] && echo ' (dry run, nothing submitted)'), ${#failed[@]} refused"
[ $nwait -gt 0 ] && echo "$nwait task(s) waiting for their FC_MAE checkpoint (run this script again later):$waiting"
for c in "${failed[@]}"; do echo "REFUSED: $c"; done
