#!/bin/bash
#SBATCH --job-name=ours
#SBATCH --output=ours_V5_long_%j.out
#SBATCH --error=ours_V5_long_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --constraint=h100
#SBATCH --account=xfp@h100

module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0

export PYTHONUSERBASE=$WORK/.local

cd $WORK/SSMMC/SSMMC

# $1 - Dataset
# $2 - First Modality
# $3 - Second Modality
# $4 - per-class labels
# $5 - runID

#ABLATION WITHOUT CONTINUAL SSL
srun python ssl_pretrained_classif_v5_abla1.py $1 $2 $3 $4 $5 $1/PRETRAIN_ABLA/checkpoint_latest.pth --all_layers_combination --output_dir V6_ABLA1
