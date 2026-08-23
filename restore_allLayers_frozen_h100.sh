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

srun python restore_abla1.py EUROSAT SAR MS 5 --output_dir OURS_FROZEN_MLA --all_layers_combination
srun python restore_abla1.py EUROSAT SAR MS 25 --output_dir OURS_FROZEN_MLA --all_layers_combination
srun python restore_abla1.py EUROSAT SAR MS 50 --output_dir OURS_FROZEN_MLA --all_layers_combination
