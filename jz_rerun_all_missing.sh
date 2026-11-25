#!/bin/bash
# Resubmit, as individual jobs, EVERY task of the reported methods whose final checkpoint is missing and which is not
# already pending/running. Safe to run several times (done or queued tasks are skipped).
#   cd $WORK/SSMMC/SSMMC && bash jz_rerun_all_missing.sh --dry   # only print what would be submitted
#   bash jz_rerun_all_missing.sh                                 # submit
# Covers:
#   FixMatch-WRN (all datasets) + Semi-ViT SUNRGBD     -> jz_resubmit_missing_full.sh
#   supervised SF from MAE, EUROSAT/TRISTAR            -> submit_sf_mae_refs_v100.sh
#   Semi-ViT FC EUROSAT/TRISTAR (+ its stage 2 = FC_MAE)  -> below, task_v100.sh semivit, 20 h
#   TCGM, TCGM_V2, KDMvC_V3                            -> submit_batch2_v100.sh
#   Semi-SST FC (only splits whose FC_MAE exists)      -> submit_semisst_v100.sh (run again after Semi-ViT FC)
# NB: a checkpoint that exists but is truncated (job killed while saving) counts as done here; such files are
# listed as FAILED in export_log.txt of the export -> delete them and run this script again.
cd $WORK/SSMMC/SSMMC
ARG=$1
DRY=0; [ "$ARG" == "--dry" ] && DRY=1
mkdir -p slurm_logs
{
echo "== FixMatch-WRN + Semi-ViT SUNRGBD";           bash jz_resubmit_missing_full.sh $ARG
echo "== supervised SF from MAE (EUROSAT/TRISTAR)";  bash submit_sf_mae_refs_v100.sh $ARG
echo "== Semi-ViT FC (EUROSAT/TRISTAR)"
queued=$(squeue -u $USER -h -o "%j" 2>/dev/null)
n=0
for ds in "EUROSAT_STD SAR MS" "TRISTAR_STD DEPTH THERMAL"; do
  set -- $ds; d=$1; name=${d%%_*}
  for perc in 5 25 50; do for run in 0 1 2 3 4; do
    [ -f $d/SEMIVIT_FC/${perc}_${run}.pth ] && continue
    jname=semivit_t_FC_${name}_${perc}_${run}
    echo "$queued" | grep -qx -e "$jname" -e "semivit_FC_${name}_${perc}" && continue   # individual job or old array
    cmd="sbatch --time=20:00:00 --job-name=$jname task_v100.sh semivit $ds $perc $run FC"
    if [ $DRY -eq 1 ]; then echo "$cmd"; else $cmd; fi
    n=$((n + 1))
  done; done
done
echo "$n Semi-ViT FC job(s)$([ $DRY -eq 1 ] && echo ' (dry run)')"
echo "== TCGM / TCGM_V2 / KDMvC_V3";                 bash submit_batch2_v100.sh $ARG
echo "== Semi-SST FC";                               bash submit_semisst_v100.sh $ARG
} 2>&1 | tee jz_rerun_all_missing.txt
