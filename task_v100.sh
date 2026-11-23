#!/bin/bash
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --time=09:59:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --constraint=v100-32g
#SBATCH --account=xfp@v100

# ONE task (one split) as an individual job, with a time limit < 10 h (set with sbatch --time, default 9:59) so it
# can start before a maintenance window. Same settings as baselines_array_v100.sh / semivit_array_v100.sh.
#   $1 = plain   : python $2 <dataset> <mod1> <mod2> <perc> <run> [<fusion>]      (e.g. fixmatch_wrn_main.py)
#   $1 = semivit : Semi-ViT stage 2 (baselines_main_v5.py, skipped if its checkpoint exists) then stage 3
#                  (semivit_main.py); arguments <dataset> <mod1> <mod2> <perc> <run> <SF|FC>
# Examples: sbatch --time=04:00:00 task_v100.sh plain fixmatch_wrn_main.py SUNRGBD_STD RGB DEPTH 5 0
#           sbatch --time=04:00:00 task_v100.sh semivit SUNRGBD_STD RGB DEPTH 5 0 SF

module purge
module load pytorch-gpu/py3/2.3.0
export PYTHONUSERBASE=$WORK/.local
export TMPDIR=${JOBSCRATCH:-$SCRATCH/tmp_${SLURM_JOB_ID}}
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor
export TRITON_CACHE_DIR=$TMPDIR/triton
mkdir -p $TMPDIR $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR
cd $WORK/SSMMC/SSMMC

kind=$1; shift
echo "job $SLURM_JOB_ID: $kind $*"
if [ "$kind" == "plain" ]; then
  srun python -u "$@"
elif [ "$kind" == "semivit" ]; then
  ds=$1; m1=$2; m2=$3; perc=$4; run=$5; fusion=$6
  for m in $m1 $m2; do [ -f $ds/MAE/${m}_encoder.pth ] || { echo "missing MAE encoder $ds/MAE/${m}_encoder.pth"; exit 1; }; done
  if [ -f $ds/${fusion}_MAE/${perc}_${run}.pth ]; then
    echo "stage 2 checkpoint $ds/${fusion}_MAE/${perc}_${run}.pth exists, skipping stage 2"
  else
    srun python -u baselines_main_v5.py $ds $m1 $m2 $perc $run $fusion || exit 1
  fi
  srun python -u semivit_main.py $ds $m1 $m2 $perc $run $fusion
else
  echo "unknown kind $kind"; exit 1
fi
