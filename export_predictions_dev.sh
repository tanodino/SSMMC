#!/bin/bash
# Export the per-sample test predictions of the methods trained on Jean Zay (export_predictions.py),
# from an INTERACTIVE dev session (no batch queue). Output: results/preds_jeanzay
#   per_split.csv  +  <DATASET>/test_labels.npz  +  <DATASET>/<FOLDER>/<perc>_<run>.npz
#
# 1) get a 2h interactive GPU session (dev QoS, max 2h):
#    srun --pty --ntasks=1 --cpus-per-task=10 --gres=gpu:1 --hint=nomultithread \
#         --qos=qos_gpu-dev --time=02:00:00 -C v100-32g -A xfp@v100 bash
# 2) inside the session:
#    cd $WORK/SSMMC/SSMMC && bash export_predictions_dev.sh                 # the 8 baselines
#    bash export_predictions_dev.sh KDMvC_V2 MSC_V2                         # later, other folders
# 3) tar czf preds_jeanzay.tgz -C results preds_jeanzay
#
# Re-running is safe: checkpoints already exported are skipped, so a second session resumes where the first stopped.

module purge
module load pytorch-gpu/py3/2.3.0
export PYTHONUSERBASE=$WORK/.local
cd $WORK/SSMMC/SSMMC

METHODS="$@"
[ -z "$METHODS" ] && METHODS="SF FC FIXMATCH_SF FIXMATCH_FC FREEMATCH_V2_SF FREEMATCH_V2_FC SOFTMATCH_V2_SF SOFTMATCH_V2_FC"
echo "methods: $METHODS"
python -u export_predictions.py --out results/preds_jeanzay --methods $METHODS
