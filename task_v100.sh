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
#   $1 = stage2  : same as semivit but stage 2 only (supervised SF/FC from MAE = reference of the main table)
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
# torch.compile (used by stage 2, baselines_main_v5.py) compiles with parallel worker processes that can race on
# the Triton cache (FileNotFoundError on triton_*.json.tmp, seen on Jean Zay 2026-11-23): compile in a single process.
export TORCHINDUCTOR_COMPILE_THREADS=1
# no core dumps: a crashing python process otherwise writes several GB (core-pt_main_thread-* folders) into $WORK,
# which can fill the quota and make the final torch.save of other jobs write empty / truncated checkpoints (2026-11-26)
ulimit -c 0
cd $WORK/SSMMC/SSMMC
marker=$TMPDIR/job_start_marker; touch $marker

# ok_ckpts <ds> <perc> <run>: every <ds>/<folder>/<perc>_<run>.pth written by THIS job must load with torch.load;
# a broken one is deleted (so it is never counted as done by the submit scripts) and the function fails
ok_ckpts() {
  local bad=0 f
  for f in $(find $1 -mindepth 2 -maxdepth 2 -name "${2}_${3}.pth" -newer $marker 2>/dev/null); do
    if python -c "import sys, torch; torch.load(sys.argv[1], map_location='cpu')" $f > /dev/null 2>&1; then
      echo "checkpoint OK: $f ($(stat -c %s $f) bytes)"
    else
      echo "BROKEN checkpoint (does not load), deleted: $f ($(stat -c %s $f) bytes)"; rm -f $f; bad=1
    fi
  done
  return $bad
}
# run_checked <ds> <perc> <run> <python args...>: run, verify the checkpoint, one automatic retry if either fails
run_checked() {
  local ds=$1 perc=$2 run=$3; shift 3
  { srun python -u "$@" && ok_ckpts $ds $perc $run; } && return 0
  echo "failed or broken checkpoint, retrying once"; touch $marker
  srun python -u "$@" && ok_ckpts $ds $perc $run
}

kind=$1; shift
echo "job $SLURM_JOB_ID: $kind $*"
if [ "$kind" == "plain" ]; then
  # $1 = script, $2 = dataset, $5 = perc, $6 = run
  run_checked $2 $5 $6 "$@"
elif [ "$kind" == "semivit" ] || [ "$kind" == "stage2" ]; then
  ds=$1; m1=$2; m2=$3; perc=$4; run=$5; fusion=$6
  for m in $m1 $m2; do [ -f $ds/MAE/${m}_encoder.pth ] || { echo "missing MAE encoder $ds/MAE/${m}_encoder.pth"; exit 1; }; done
  if [ -f $ds/${fusion}_MAE/${perc}_${run}.pth ] && \
     python -c "import sys, torch; torch.load(sys.argv[1], map_location='cpu')" $ds/${fusion}_MAE/${perc}_${run}.pth > /dev/null 2>&1; then
    echo "stage 2 checkpoint $ds/${fusion}_MAE/${perc}_${run}.pth exists and loads, skipping stage 2"
  else
    rm -f $ds/${fusion}_MAE/${perc}_${run}.pth      # absent or broken
    run_checked $ds $perc $run baselines_main_v5.py $ds $m1 $m2 $perc $run $fusion || exit 1
  fi
  [ "$kind" == "stage2" ] && exit 0          # stage2: supervised-from-MAE reference only, no Semi-ViT stage
  touch $marker
  srun python -u semivit_main.py $ds $m1 $m2 $perc $run $fusion && ok_ckpts $ds $perc $run
else
  echo "unknown kind $kind"; exit 1
fi
