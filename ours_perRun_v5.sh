#!/bin/bash
#SBATCH --job-name=ours
#SBATCH --output=ours_%j.out
#SBATCH --error=ours_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --constraint=v100-32g
#SBATCH --account=xfp@v100

module purge
module load pytorch-gpu/py3/2.3.0

export PYTHONUSERBASE=$WORK/.local

cd $WORK/SSMMC/SSMMC

# $1 - Dataset
# $2 - First Modality
# $3 - Second Modality
# $4 - per-class labels
# $5 - runID

srun python ssl_pretrained_classif_v5.py $1 $2 $3 $4 $5 $1/PRETRAIN/checkpoint_latest.pth
