#!/bin/bash
# Resubmit, as individual jobs (task_v100.sh), every FixMatch-WRN or Semi-ViT SUNRGBD task whose final checkpoint is
# missing AND which is not already pending/running (matched by job name). Safe to run several times.
#   cd $WORK/SSMMC/SSMMC && bash jz_resubmit_missing_full.sh --dry   # only print what would be resubmitted
#   bash jz_resubmit_missing_full.sh                                 # resubmit
# (Semi-ViT TRISTAR/EUROSAT are handled by the 20 h arrays of step 15, after the maintenance.)
cd $WORK/SSMMC/SSMMC
DRY=0; [ "$1" == "--dry" ] && DRY=1
mkdir -p slurm_logs
queued=$(squeue -u $USER -h -o "%j")
n=0
sub() {   # $1 = job name, $2 = time, rest = task_v100.sh arguments
  local name=$1 t=$2; shift 2
  if echo "$queued" | grep -qx "$name"; then return; fi
  if [ $DRY -eq 1 ]; then echo "sbatch --time=$t --job-name=$name task_v100.sh $*"; else sbatch --time=$t --job-name=$name task_v100.sh "$@"; fi
  n=$((n + 1))
}
for ds in "SUNRGBD_STD RGB DEPTH" "EUROSAT_STD SAR MS" "TRISTAR_STD DEPTH THERMAL"; do
  set -- $ds; d=$1; name=${d%%_*}
  t=09:59:00; [ "$name" == "SUNRGBD" ] && t=04:00:00
  for perc in 5 25 50; do for run in 0 1 2 3 4; do
    [ -f $d/FIXMATCH_WRN_SF/${perc}_${run}.pth ] || sub fmwrn_t_${name}_${perc}_${run} $t plain fixmatch_wrn_main.py $ds $perc $run
  done; done
done
for fusion in SF FC; do for perc in 5 25 50; do for run in 0 1 2 3 4; do
  [ -f SUNRGBD_STD/SEMIVIT_${fusion}/${perc}_${run}.pth ] || \
    sub semivit_t_${fusion}_SUNRGBD_${perc}_${run} 06:00:00 semivit SUNRGBD_STD RGB DEPTH $perc $run $fusion
done; done; done
echo "$n job(s) $( [ $DRY -eq 1 ] && echo 'would be' ) resubmitted"
