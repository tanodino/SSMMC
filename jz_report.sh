#!/bin/bash
# Diagnostic report on Jean Zay (login node is fine): where are the checkpoints, did the jobs fail?
#   cd $WORK/SSMMC/SSMMC && bash jz_report.sh        -> writes jz_report.txt
cd $WORK/SSMMC/SSMMC
{
echo "== date"; date
echo "== number of .pth files under SSMMC"; find . -name "*.pth" | wc -l
echo "== .pth files per folder"; find . -name "*.pth" -printf "%h\n" | sort | uniq -c
echo "== slurm logs: count and most recent"; ls slurm_logs | wc -l; ls -t slurm_logs | head -5
echo "== last lines of the 5 most recent .err files"
for f in $(ls -t slurm_logs/*.err 2>/dev/null | head -5); do echo "-- $f"; tail -8 "$f"; done
echo "== last lines of the 3 most recent .out files"
for f in $(ls -t slurm_logs/*.out 2>/dev/null | head -3); do echo "-- $f"; tail -8 "$f"; done
echo "== count of .err files mentioning an error"; grep -l -E "Error|Traceback" slurm_logs/*.err 2>/dev/null | wc -l
echo "== most common error lines"; grep -h -E "Error:" slurm_logs/*.err 2>/dev/null | sort | uniq -c | sort -rn | head -10
for d in SUNRGBD EUROSAT TRISTAR; do
  echo "== files in ${d}_STD (without data)"; ls ${d}_STD | grep -v data_normalized
  echo "== files in ${d} (first 10)"; ls ${d} 2>/dev/null | head -10
done
} > jz_report.txt 2>&1
echo "written: $PWD/jz_report.txt"
