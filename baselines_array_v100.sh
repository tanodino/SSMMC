#!/bin/bash
#SBATCH --job-name=bl_v100
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

# One baseline configuration, the 5 splits (run ids 0-4) as 5 array tasks that
# run in parallel, one GPU each. Same settings as the previous V100 scripts.
#
# $1 - training script: baselines_main.py | softmatch_baselines_main.py |
#      fixmatch_baselines_main.py | freematch_baselines_main.py
# $2 - dataset folder, e.g. EUROSAT_STD (standardised data)
# $3 - first modality      $4 - second modality
# $5 - labelled samples per class (5 / 25 / 50)
# $6 - fusion: SF (score fusion) or FC (feature concatenation)
#
# Example:  sbatch baselines_array_v100.sh softmatch_baselines_main.py EUROSAT_STD SAR MS 5 FC
# (slurm_logs/ must exist before submitting; submit_baselines_v100.sh creates it)

module purge
module load pytorch-gpu/py3/2.3.0

export PYTHONUSERBASE=$WORK/.local

# Temporary files (torch.compile / Triton caches, multiprocessing pymp-* folders) go to the per-job
# scratch folder, private to this job and deleted at its end. The node's shared /tmp filled up and
# concurrent tasks overwrote each other's compile cache (Triton PTX codegen / No space left errors).
export TMPDIR=${JOBSCRATCH:-$SCRATCH/tmp_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}}
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor
export TRITON_CACHE_DIR=$TMPDIR/triton
mkdir -p $TMPDIR $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR

cd $WORK/SSMMC/SSMMC

echo "task ${SLURM_ARRAY_TASK_ID}: $1 $2 $3 $4 perc=$5 run=${SLURM_ARRAY_TASK_ID} fusion=$6"
srun python -u $1 $2 $3 $4 $5 ${SLURM_ARRAY_TASK_ID} $6
