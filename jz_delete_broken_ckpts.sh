#!/bin/bash
# Delete the checkpoints that the last export could NOT load (empty or truncated .pth, listed as "FAILED" in
# results/preds_jz_full/export_log.txt), so that the submit scripts / jz_rerun_all_missing.sh resubmit those tasks.
# A file modified AFTER the export log (rewritten since by a running job) is kept.
#   cd $WORK/SSMMC/SSMMC && bash jz_delete_broken_ckpts.sh --dry   # only list what would be deleted
#   bash jz_delete_broken_ckpts.sh                                 # delete
cd $WORK/SSMMC/SSMMC
DRY=0; [ "$1" == "--dry" ] && DRY=1
LOG=results/preds_jz_full/export_log.txt
[ -f $LOG ] || { echo "no $LOG: run the export first (step 37)"; exit 1; }
n=0; kept=0
for f in $(grep -a "^  FAILED " $LOG | awk '{print $2}' | sort -u); do
  [ -f "$f" ] || continue
  if [ "$f" -nt $LOG ]; then echo "KEPT (rewritten after the export): $f"; kept=$((kept + 1)); continue; fi
  if [ $DRY -eq 1 ]; then echo "would delete $f ($(stat -c %s $f) bytes)"; else rm -f "$f" && echo "deleted $f"; fi
  n=$((n + 1))
done
echo "$n broken checkpoint(s) $([ $DRY -eq 1 ] && echo 'would be ')deleted, $kept kept"
