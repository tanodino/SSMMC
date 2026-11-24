#!/bin/bash
# Progress of the full competitor runs on Jean Zay (login node is fine), written to jz_full_report.txt:
#   cd $WORK/SSMMC/SSMMC && bash jz_full_report.sh
# For each method folder x dataset x label budget: number of finished checkpoints out of 5 splits (and the missing
# run ids), then the running/pending jobs and the slurm tasks whose .err mentions an error.
cd $WORK/SSMMC/SSMMC
FOLDERS="FIXMATCH_WRN_SF SF_MAE FC_MAE SEMIVIT_SF SEMIVIT_FC TCGM TCGM_V2 KDMvC_V3"
{
echo "== date"; date
echo "== MAE encoders"; ls -la {SUNRGBD,EUROSAT,TRISTAR}_STD/MAE/*_encoder.pth 2>&1 | awk '{print $5, $NF}'
echo "== checkpoints per method / dataset / labels (done out of 5, missing run ids)"
for f in $FOLDERS; do
  for d in SUNRGBD_STD EUROSAT_STD TRISTAR_STD; do
    line="$f $d:"
    for p in 5 25 50; do
      n=0; miss=""
      for r in 0 1 2 3 4; do
        if [ -f $d/$f/${p}_${r}.pth ]; then n=$((n + 1)); else miss="$miss$r"; fi
      done
      line="$line  $p=$n/5${miss:+(missing $miss)}"
    done
    echo "$line"
  done
done
echo "== jobs in the queue"; squeue -u $USER -o "%.18i %.40j %.8T %.10M" 2>/dev/null
echo "== tasks with an error (fmwrn / semivit / sfmae / tcgm / tcgm2 / kdmvc3), most recent first"
for e in $(ls -t slurm_logs/fmwrn_*.err slurm_logs/semivit_*.err slurm_logs/sfmae_*.err slurm_logs/tcgm_t_*.err slurm_logs/tcgm2_t_*.err slurm_logs/kdmvc3_t_*.err 2>/dev/null); do
  if grep -a -q -E "Error|Traceback|CANCELLED|DUE TO TIME LIMIT" "$e"; then
    echo "-- $e: $(grep -a -h -E 'Error|CANCELLED|DUE TO TIME LIMIT' "$e" | tail -1)"
  fi
done
} > jz_full_report.txt 2>&1
echo "written: $PWD/jz_full_report.txt"
