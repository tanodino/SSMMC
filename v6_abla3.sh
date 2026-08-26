#!/bin/bash
#SBATCH --job-name=ours
#SBATCH --output=v6_abla3_%j.out
#SBATCH --error=v6_abla3_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --constraint=v100-32g
#SBATCH --account=xfp@v100

module purge
module load pytorch-gpu/py3/2.3.0

#module load arch/h100
#module load pytorch-gpu/py3/2.4.0

export PYTHONUSERBASE=$WORK/.local

cd $WORK/SSMMC/SSMMC

# $1 - Dataset
# $2 - First Modality
# $3 - Second Modality
# $4 - per-class labels
# $5 - runID

# ABLATION WITHOUT MULTI-LAYER AGGREGATION/FUSION and WITHOUT SELF SUPERVISED LEARNING

srun python ssl_pretrained_classif_v5_abla3.py $1 $2 $3 $4 $5 $1/PRETRAIN_ABLA/checkpoint_latest.pth --output_dir V6_ABLA3
