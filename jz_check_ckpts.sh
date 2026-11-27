#!/bin/bash
# Count, per method / dataset / label budget, the checkpoints that are present AND readable, and how many of them
# are NEW (written after the last export, i.e. newer than results/preds_jz_full/export_log.txt). Login node is fine:
# only the zip directory at the end of each .pth is read (an empty or truncated file -- the failure seen on
# 2026-11-26 -- has none), not the tensors.
#   cd $WORK/SSMMC/SSMMC && bash jz_check_ckpts.sh        # -> jz_check_ckpts.txt
cd $WORK/SSMMC/SSMMC
REF=results/preds_jz_full/export_log.txt
python - "$REF" > jz_check_ckpts.txt 2>&1 <<'EOF'
import os, sys, zipfile
ref = sys.argv[1]
t_ref = os.path.getmtime(ref) if os.path.exists(ref) else 0
folders = "FIXMATCH_WRN_SF SF_MAE FC_MAE SEMIVIT_SF SEMIVIT_FC SEMISST_FC TCGM TCGM_V2 KDMvC_V3 MSC_V5".split()
datasets = ["SUNRGBD_STD", "EUROSAT_STD", "TRISTAR_STD"]
print("new = written after the last export (" + (ref if t_ref else "no export log: every file counts as new") + ")")
print("per budget: ok/5 (new) ; x = broken (unreadable) split ids ; - = missing split ids")
broken_all, tot_ok, tot_new = [], 0, 0
for f in folders:
    for d in datasets:
        cells = []
        for p in (5, 25, 50):
            ok = new = 0; bad = miss = ""
            for r in range(5):
                path = f"{d}/{f}/{p}_{r}.pth"
                if not os.path.exists(path):
                    miss += str(r); continue
                try:
                    with zipfile.ZipFile(path) as z:
                        z.namelist()
                    ok += 1
                    new += os.path.getmtime(path) > t_ref
                except Exception:
                    bad += str(r); broken_all.append(f"{path} ({os.path.getsize(path)} bytes)")
            tot_ok += ok; tot_new += new
            cells.append(f"{p}={ok}/5({new} new)" + (f" x{bad}" if bad else "") + (f" -{miss}" if miss else ""))
        print(f"{f:16s} {d:12s} " + "  ".join(cells))
print(f"\nTOTAL readable checkpoints: {tot_ok} ({tot_new} new since the last export)")
print(f"broken (unreadable) checkpoints: {len(broken_all)}")
for b in broken_all:
    print("  BROKEN " + b)
EOF
echo "== jobs still in the queue: $(squeue -u $USER -h | wc -l)" >> jz_check_ckpts.txt
cat jz_check_ckpts.txt
