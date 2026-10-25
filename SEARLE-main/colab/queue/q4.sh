#!/bin/bash
# Queue 4 (after train queue 2): D11 robust phi eval (E3.8c), cosine eval (E8.2a), CI, vector drift (E3.9)
source $DRIVE/queue/common.sh
while pgrep -f "queue/t2.sh" > /dev/null || pgrep -f "queue/t3.sh" > /dev/null; do sleep 120; done
echo "===== queue 4 start $(date)"
ev() { local P=$1 C=$2; shift 2; run $R/$C $V --eval-type phi --exp-name $P --phi-checkpoint-name phi_last.pt $CACHE "$@" --output-dir $R/$C; }
if [ -f $R/E3.8b-train-robust/train.done ]; then
  ev phi_B32_robust E3.8c-robust-clean-cirr  $CIRR
  ev phi_B32_robust E3.8c-robust-clean-circo $CIRCO
  ev phi_B32_robust E3.8c-robust-clean-fiq   $FIQ
  for D in gauss_noise:2 gauss_blur:2 jpeg:2 low_light:2 salt_pepper:2 motion_blur:2 low_res:2 color_cast:2; do
    ev phi_B32_robust E3.8c-robust-${D/:/}-cirr  $CIRR  --degrade $D
    ev phi_B32_robust E3.8c-robust-${D/:/}-circo $CIRCO --degrade $D
  done
  run $R/E3.9-drift-robust-cirr $PY tools/vector_analysis.py drift $CIRR $CACHE --method phi:phi_B32_robust --runs $R/E3.8c-robust-*-cirr --out $R/E3.9-drift-robust-cirr
fi
if [ -f $R/E8.2a-train-cosine/train.done ]; then
  ev phi_B32_cosine E8.2a-cosine-cirr  $CIRR
  ev phi_B32_cosine E8.2a-cosine-circo $CIRCO
fi
run $R/E3.9-drift-cirr  $PY tools/vector_analysis.py drift $CIRR  $CACHE --method phi:phi_B32 --runs $R/E3.1-*-cirr $R/E3.2-* $R/E3.5-*-cirr --out $R/E3.9-drift-cirr
run $R/E3.9-drift-circo $PY tools/vector_analysis.py drift $CIRCO $CACHE --method phi:phi_B32 --runs $R/E3.1-*-circo --out $R/E3.9-drift-circo
run $R/E3.8c-ci-cirr $PY tools/bootstrap_ci.py --baseline $R/E1.7d-phiB32-cirr --runs $(ls -d $R/E3.8c-robust-clean-cirr $R/E8.2a-cosine-cirr $R/E1.14-*-cirr 2>/dev/null) --out $R/E3.8c-ci-cirr
$PY tools/collect_results.py --results-dir $R --out $R/summary.md > /dev/null 2>&1
echo "===== queue 4 done $(date)"
