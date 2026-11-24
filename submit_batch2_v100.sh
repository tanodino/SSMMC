#!/bin/bash
# Jean Zay batch 2 (agreed 2026-11-24): full runs, as INDIVIDUAL jobs (task_v100.sh plain, one retry on failure),
# of the multimodal competitors in their kept settings:
#   TCGM from scratch      tcgm_main.py          -> <DS>_STD/TCGM/      (both TCGM variants are reported)
#   TCGM, MAE start        tcgm_main_v2.py       -> <DS>_STD/TCGM_V2/
#   KDMvC, MAE start       kdmvc_training_v3.py  -> <DS>_STD/KDMvC_V3/
#   3 methods x 3 datasets x 3 label budgets x 5 splits = 135 jobs
#   --time: SUNRGBD 4 h; TCGM TRISTAR/EUROSAT 10 h; KDMvC TRISTAR/EUROSAT 20 h (about 6-10 h expected on a V100)
# Needs the MAE encoders <DS>_STD/MAE/<MOD>_encoder.pth (already on Jean Zay). Tasks whose checkpoint exists or that
# are already queued (same job name) are skipped, so running it again resubmits only what is missing/failed.
#   cd $WORK/SSMMC/SSMMC && bash submit_batch2_v100.sh --dry   # only print the commands
#   bash submit_batch2_v100.sh                                 # submit
cd $WORK/SSMMC/SSMMC
DRY=0; [ "$1" == "--dry" ] && DRY=1
mkdir -p slurm_logs
queued=$(squeue -u $USER -h -o "%j" 2>/dev/null)
n=0; failed=()
# method folder : script : job-name prefix : time for TRISTAR/EUROSAT
for m in "TCGM:tcgm_main.py:tcgm_t:10:00:00" "TCGM_V2:tcgm_main_v2.py:tcgm2_t:10:00:00" "KDMvC_V3:kdmvc_training_v3.py:kdmvc3_t:20:00:00"; do
  folder=$(echo $m | cut -d: -f1); script=$(echo $m | cut -d: -f2); jp=$(echo $m | cut -d: -f3); tlong=$(echo $m | cut -d: -f4-6)
  for ds in "SUNRGBD_STD RGB DEPTH" "EUROSAT_STD SAR MS" "TRISTAR_STD DEPTH THERMAL"; do
    set -- $ds; d=$1; name=${d%%_*}
    t=$tlong; [ "$name" == "SUNRGBD" ] && t=04:00:00
    for perc in 5 25 50; do for run in 0 1 2 3 4; do
      [ -f $d/$folder/${perc}_${run}.pth ] && continue
      jname=${jp}_${name}_${perc}_${run}
      echo "$queued" | grep -qx "$jname" && continue
      cmd="sbatch --time=$t --job-name=$jname task_v100.sh plain $script $ds $perc $run"
      if [ $DRY -eq 1 ]; then echo "$cmd"; else $cmd || failed+=("$cmd"); fi
      n=$((n + 1))
    done; done
  done
done
echo "$n job(s)$([ $DRY -eq 1 ] && echo ' (dry run, nothing submitted)'), ${#failed[@]} refused"
for c in "${failed[@]}"; do echo "REFUSED: $c"; done
