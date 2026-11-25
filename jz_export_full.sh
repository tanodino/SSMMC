#!/bin/bash
#SBATCH --job-name=export_full
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --constraint=v100-32g
#SBATCH --account=xfp@v100

# Export the per-sample test predictions + per-split scores of every finished competitor checkpoint on Jean Zay
# (export_predictions.py, same format as the earlier exports), then pack them in a small archive to download:
#   results/preds_jz_full/per_split.csv, results/preds_jz_full/<DS>/<FOLDER>/<perc>_<run>.npz, plus a results table
#   results/preds_jz_full/summary_table.txt / .csv (summarize_per_split.py) -> all packed in preds_jz_full.tar.gz
# Checkpoints that do not exist yet are skipped; already-exported ones are not redone, so it can be re-submitted
# later to add the newly finished runs.   Submit with:  sbatch jz_export_full.sh
module purge
module load pytorch-gpu/py3/2.3.0
export PYTHONUSERBASE=$WORK/.local
cd $WORK/SSMMC/SSMMC
srun python -u export_predictions.py --out results/preds_jz_full \
  --methods SF FC FIXMATCH_WRN_SF SF_MAE FC_MAE SEMIVIT_SF SEMIVIT_FC SEMISST_FC TCGM TCGM_V2 KDMvC_V2 KDMvC_V3
srun python -u summarize_per_split.py results/preds_jz_full      # summary_table.csv / .txt (mean +- std over splits)
tar czf preds_jz_full.tar.gz results/preds_jz_full && md5sum preds_jz_full.tar.gz > preds_jz_full.md5
ls -la preds_jz_full.tar.gz preds_jz_full.md5
