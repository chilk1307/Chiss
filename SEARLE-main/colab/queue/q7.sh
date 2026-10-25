#!/bin/bash
# Queue 7 (24/10, after the extended E2.3): larger attribute weights and other K for colour / brightness.
# (Written in the notebook at the time; this copy is rebuilt from queue/q7.log, same runs and names.)
source $DRIVE/queue/common.sh
echo "===== queue 7 start $(date)"
for FAM in color brightness; do
  for W in 4.0 8.0; do
    run $R/E2.3-$FAM-w$W-cirr $V $PHI $CIRR --attr-rerank --attr-families $FAM --attr-weight $W --output-dir $R/E2.3-$FAM-w$W-cirr
    run $R/E2.3-$FAM-w$W-fiq  $V $PHI $FIQ --exclude-reference --attr-rerank --attr-families $FAM --attr-weight $W --output-dir $R/E2.3-$FAM-w$W-fiq
  done
done
for K in 50 200; do
  run $R/E2.3-color-w2.0-k$K-cirr $V $PHI $CIRR --attr-rerank --attr-families color --attr-weight 2.0 --attr-topk $K --output-dir $R/E2.3-color-w2.0-k$K-cirr
done
echo "===== queue 7 done $(date)"
