#!/bin/bash
# Why are some baseline checkpoints missing? For every missing <DS>_STD/<FOLDER>/<perc>_<run>.pth it
# finds the most recent slurm log of that task and prints the cause (time limit, OOM, Python error,
# still running...) plus the last lines. Run on the login node:
#   cd $WORK/SSMMC/SSMMC && bash jz_missing.sh        -> writes jz_missing.txt
cd $WORK/SSMMC/SSMMC
{
echo "== date"; date
echo "== my jobs still in the queue"; squeue -u $USER -o "%.12i %.40j %.8T %.10M" 2>/dev/null
for ds in SUNRGBD EUROSAT TRISTAR; do
  for folder in FIXMATCH_SF FIXMATCH_FC FREEMATCH_V2_SF FREEMATCH_V2_FC SOFTMATCH_V2_SF SOFTMATCH_V2_FC SF FC; do
    fusion=${folder##*_}
    case $folder in
      FIXMATCH_*)  job=fixmatch_${fusion}_${ds} ;;
      FREEMATCH_*) job=freematch_v2_${fusion}_${ds} ;;
      SOFTMATCH_*) job=softmatch_v2_${fusion}_${ds} ;;
      *)           job=baselines_${fusion}_${ds} ;;
    esac
    for perc in 5 25 50; do
      for run in 0 1 2 3 4; do
        [ -f ${ds}_STD/$folder/${perc}_${run}.pth ] && continue
        err=$(ls -t slurm_logs/${job}_${perc}_*_${run}.err 2>/dev/null | head -1)
        out=${err%.err}.out
        echo
        echo "#### MISSING ${ds}_STD/$folder/${perc}_${run}.pth   log: ${err:-NO LOG FOUND for ${job}_${perc}_*_${run}}"
        [ -z "$err" ] && continue
        cause=$(grep -h -o -E "DUE TO TIME LIMIT|CANCELLED[^*]*|[Oo]ut [Oo]f [Mm]emory|oom-kill[a-z]*|NODE_FAIL|[A-Za-z]*Error: .*|nan|NaN" $err $out 2>/dev/null | sort | uniq -c | sort -rn | head -3 | tr -s " \n" " ")
        echo "  cause: ${cause:-none found in the logs}"
        echo "  last epoch line: $(grep -E -i '^epoch|ep ' $out 2>/dev/null | tail -1 | cut -c1-150)"
        echo "  last lines of .err:"; tail -4 $err | sed 's/^/    /'
      done
    done
  done
done
} > jz_missing.txt 2>&1
grep -c "#### MISSING" jz_missing.txt | sed 's/^/missing checkpoints: /'
grep "^  cause:" jz_missing.txt | sort | uniq -c | sort -rn
echo "full report: $PWD/jz_missing.txt"
