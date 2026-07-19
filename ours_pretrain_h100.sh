#!/bin/bash
#SBATCH --job-name=pretrain_ours
#SBATCH --output=pretrain_ours_%j.out
#SBATCH --error=pretrain_ours_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --constraint=h100
#SBATCH --account=xfp@h100



module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0

export PYTHONUSERBASE=$WORK/.local

cd $WORK/SSMMC/SSMMC

# EUROSAT SAR MS
#$1: directory of the data
#$2: First Data Modality
#$3: Second Data Modality

srun python pretrain.py EUROSAT SAR MS
