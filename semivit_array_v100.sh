#!/bin/bash
#SBATCH --job-name=semivit_v100
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --constraint=v100-32g
#SBATCH --account=xfp@v100
#SBATCH --array=0-4

# Semi-ViT competitor, one configuration, the 5 splits (run ids 0-4) as 5 array tasks. Each task runs the two
# Semi-ViT stages that depend on the labels (agreed 2026-11-22):
#   stage 2: baselines_main_v5.py  -> supervised SF/FC from the per-modality MAE encoders  -> <DS>/<SF|FC>_MAE/
#            (also the "supervised, MAE start" reference of the main table)
#   stage 3: semivit_main.py       -> EMA teacher + probabilistic pseudo mixup              -> <DS>/SEMIVIT_<SF|FC>/
# Stage 1 (MAE pretraining, split-independent) is done once: <DS>/MAE/<MOD>_encoder.pth must exist (mae_encoders.tar).
# Stage 2 is skipped if its checkpoint already exists (e.g. task re-submitted after a timeout in stage 3).
#
# $1 - dataset folder (e.g. EUROSAT_STD)   $2 - first modality   $3 - second modality
# $4 - labelled samples per class (5 / 25 / 50)   $5 - fusion: SF or FC
# Example:  sbatch semivit_array_v100.sh EUROSAT_STD SAR MS 5 FC

module purge
module load pytorch-gpu/py3/2.3.0

export PYTHONUSERBASE=$WORK/.local
export TMPDIR=${JOBSCRATCH:-$SCRATCH/tmp_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}}
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor
export TRITON_CACHE_DIR=$TMPDIR/triton
mkdir -p $TMPDIR $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR

cd $WORK/SSMMC/SSMMC
RUN=${SLURM_ARRAY_TASK_ID}
echo "task $RUN: Semi-ViT $1 $2 $3 perc=$4 run=$RUN fusion=$5"

for m in $2 $3; do
  [ -f $1/MAE/${m}_encoder.pth ] || { echo "missing MAE encoder $1/MAE/${m}_encoder.pth"; exit 1; }
done

if [ -f $1/$5_MAE/$4_$RUN.pth ]; then
  echo "stage 2 checkpoint $1/$5_MAE/$4_$RUN.pth exists, skipping stage 2"
else
  srun python -u baselines_main_v5.py $1 $2 $3 $4 $RUN $5 || exit 1
fi
srun python -u semivit_main.py $1 $2 $3 $4 $RUN $5
