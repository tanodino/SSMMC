#!/bin/bash
#SBATCH --job-name=kdmvc_mm
#SBATCH --output=kdmvc_mm_%j.out
#SBATCH --error=kdmvc_mm_%j.err
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
# $5 - Fusion Model (SF / FC)

for i in $(seq 0 4)
do
    srun python kdmvc_training.py $1 $2 $3 $4 $i $5
done
