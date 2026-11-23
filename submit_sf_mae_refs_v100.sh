#!/bin/bash
# Supervised SF from the MAE encoders (reference of the main table) for TRISTAR and EUROSAT, as INDIVIDUAL jobs running
# Semi-ViT stage 2 only (baselines_main_v5.py ... SF). Needed because Semi-ViT is reported with FC only (agreed
# 2026-11-23), so the Semi-ViT SF tasks that used to produce <DS>/SF_MAE are not run. SUNRGBD SF_MAE already exists.
#   2 datasets x 3 label budgets x 5 splits = 30 jobs, --time 2 h (stage 2 = 400 supervised epochs on labelled data)
# Tasks whose checkpoint exists or that are already queued are skipped. Safe to run again.
#   cd $WORK/SSMMC/SSMMC && bash submit_sf_mae_refs_v100.sh --dry   # only print the commands
#   bash submit_sf_mae_refs_v100.sh                                 # submit
# Results: <DS>_STD/SF_MAE/<perc>_<run>.pth
cd $WORK/SSMMC/SSMMC
DRY=0; [ "$1" == "--dry" ] && DRY=1
mkdir -p slurm_logs
queued=$(squeue -u $USER -h -o "%j" 2>/dev/null)
n=0; failed=()
for ds in "EUROSAT_STD SAR MS" "TRISTAR_STD DEPTH THERMAL"; do
  set -- $ds; d=$1
  for perc in 5 25 50; do for run in 0 1 2 3 4; do
    [ -f $d/SF_MAE/${perc}_${run}.pth ] && continue
    name=sfmae_t_${d%%_*}_${perc}_${run}
    echo "$queued" | grep -qx "$name" && continue
    cmd="sbatch --time=02:00:00 --job-name=$name task_v100.sh stage2 $ds $perc $run SF"
    if [ $DRY -eq 1 ]; then echo "$cmd"; else $cmd || failed+=("$cmd"); fi
    n=$((n + 1))
  done; done
done
echo "$n job(s)$([ $DRY -eq 1 ] && echo ' (dry run)'), ${#failed[@]} refused"
for c in "${failed[@]}"; do echo "REFUSED: $c"; done
