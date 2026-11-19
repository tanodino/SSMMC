#!/bin/bash
# Copy the auxiliary split files (train_idx.npy, labelled_samples_*.npy) from SUNRGBD/, EUROSAT/,
# TRISTAR/ into the *_STD folders on Jean Zay, so the training scripts (which read everything from
# the *_STD folder) find them. labels.npy is already in the *_STD folders: it is only compared.
# Existing files in *_STD are never overwritten (cp -n). The result is written to jz_fix_aux.txt.
#   cd $WORK/SSMMC/SSMMC && bash jz_fix_aux.sh
cd $WORK/SSMMC/SSMMC
{
for d in SUNRGBD EUROSAT TRISTAR; do
  echo "== $d -> ${d}_STD"
  cp -n -v $d/train_idx.npy $d/labelled_samples_*.npy ${d}_STD/
  for f in labels.npy train_idx.npy; do
    if cmp -s $d/$f ${d}_STD/$f; then echo "  $f identical in $d and ${d}_STD"; else echo "  WARNING: $f DIFFERS (or missing) between $d and ${d}_STD"; fi
  done
  n=$(ls ${d}_STD/labelled_samples_*.npy 2>/dev/null | wc -l)
  echo "  ${d}_STD now has: labels.npy=$([ -f ${d}_STD/labels.npy ] && echo yes || echo NO)" \
       "train_idx.npy=$([ -f ${d}_STD/train_idx.npy ] && echo yes || echo NO) labelled_samples files=$n (expected 15)"
done
} > jz_fix_aux.txt 2>&1
cat jz_fix_aux.txt | grep -E "==|identical|WARNING|now has"
echo "full log: $PWD/jz_fix_aux.txt"
