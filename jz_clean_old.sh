#!/bin/bash
# Delete the previous results on Jean Zay (agreed 2026-11-20):
#   * from-scratch competitor checkpoints: <DS>_STD/{FIXMATCH_SF,FIXMATCH_FC,FREEMATCH_V2_SF,FREEMATCH_V2_FC,
#     SOFTMATCH_V2_SF,SOFTMATCH_V2_FC,KDMvC_V2,MSC_V2}/   (their predictions are saved on the workstation)
#   * slurm_logs/*  and the pymp-* folders in $WORK/SSMMC/SSMMC
# KEPT: <DS>_STD/SF and <DS>_STD/FC (supervised checkpoints, needed by the v3 protocol), PRETRAIN_V2, data,
# split files, results/, the plain SUNRGBD/ EUROSAT/ TRISTAR/ folders.
# Refuses to run while you have jobs in the queue (their logs and temp folders are in use).
#   cd $WORK/SSMMC/SSMMC && bash jz_clean_old.sh          # only shows what would be deleted
#   bash jz_clean_old.sh --yes                            # deletes
cd $WORK/SSMMC/SSMMC || exit 1
YES=0; [ "$1" == "--yes" ] && YES=1
if [ -n "$(squeue -u $USER -h 2>/dev/null)" ]; then
  echo "You still have jobs in the queue: nothing deleted. Wait for them to end, or cancel them (scancel <jobid>)."
  squeue -u $USER -o "%.12i %.40j %.8T %.10M"; exit 1
fi
TARGETS=()
for d in SUNRGBD_STD EUROSAT_STD TRISTAR_STD; do
  for f in FIXMATCH_SF FIXMATCH_FC FREEMATCH_V2_SF FREEMATCH_V2_FC SOFTMATCH_V2_SF SOFTMATCH_V2_FC KDMvC_V2 MSC_V2; do
    [ -d $d/$f ] && TARGETS+=("$d/$f")
  done
done
for t in "${TARGETS[@]}"; do echo "checkpoint folder: $t ($(ls $t | wc -l) files)"; done
echo "slurm_logs: $(ls slurm_logs 2>/dev/null | wc -l) files"
echo "pymp-* folders: $(ls -d pymp-* 2>/dev/null | wc -l)"
for d in SUNRGBD_STD EUROSAT_STD TRISTAR_STD; do echo "KEPT: $d/SF ($(ls $d/SF 2>/dev/null | wc -l) files), $d/FC ($(ls $d/FC 2>/dev/null | wc -l) files)"; done
if [ $YES -eq 0 ]; then echo; echo "Nothing deleted. To delete the items above: bash jz_clean_old.sh --yes"; exit 0; fi
rm -rf "${TARGETS[@]}"
rm -f slurm_logs/*
rm -rf pymp-*
echo; echo "Deleted. Remaining folders in the *_STD directories:"; ls -d *_STD/*/
