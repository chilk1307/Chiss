#!/bin/bash
# Queue 8 (C18 chosen 25/10): final configuration = phi_B32 + attribute re-ranking with the E2.7 flags
# ($R/E2-final-flags.txt), no restoration on clean test images; robust phi submitted too (D13).
# H13 test submissions, D13, H12 qualitative figures, extra CIs (robust vs phi per degradation).
source $DRIVE/queue/common.sh
echo "===== queue 8 start $(date)"
FINAL=$(cat $R/E2-final-flags.txt)
G="$PY src/generate_test_submission.py"
SUB=$DRIVE/outputs/submissions; mkdir -p $SUB
run $R/H13-final-cirr  $G $PHI $CIRR  $FINAL --submission-name final_cirr  --output-dir $R/H13-final-cirr
run $R/H13-final-circo $G $PHI $CIRCO $FINAL --submission-name final_circo --output-dir $R/H13-final-circo
RB="--eval-type phi --exp-name phi_B32_robust --phi-checkpoint-name phi_last.pt $CACHE"
run $R/D13-robust-cirr  $G $RB $CIRR  --submission-name robust_cirr  --output-dir $R/D13-robust-cirr
run $R/D13-robust-circo $G $RB $CIRCO --submission-name robust_circo --output-dir $R/D13-robust-circo
cp -v $R/H13-final-*/*.json $R/D13-robust-*/*.json $SUB/ 2>/dev/null | grep -v metrics
ls -la $SUB

# H12: qualitative examples, occlusion maps, explained attribute re-ranking
run $R/E1.12-examples-cirr  $PY tools/visualize_queries.py --run $R/E1.7d-phiB32-cirr --compare $R/E2.7-final-cirr $CIRR --n 20 --seed 0 $CACHE --out $R/E1.12-examples-cirr
run $R/E1.13-occlusion-cirr $PY tools/occlusion_map.py --run $R/E1.7d-phiB32-cirr $CIRR --n 10 --seed 0 $CACHE --out $R/E1.13-occlusion-cirr
run $R/E2.8-explained-fiq   $PY tools/visualize_queries.py --explain-attributes --run $R/E2.7-final-fiq-ex $FIQ --n 20 --seed 0 $CACHE --out $R/E2.8-explained-fiq

# CIs: robust phi vs phi_B32 on the same degraded queries; E2.7 on CIRCO with the reference in the gallery
for D in gauss_noise salt_pepper gauss_blur motion_blur jpeg low_light low_res color_cast; do
  run $R/E3.8c-ci-$D-cirr $PY tools/bootstrap_ci.py --baseline $R/E3.1-${D}2-cirr --runs $R/E3.8c-robust-${D}2-cirr --out $R/E3.8c-ci-$D-cirr
done
run $R/E3.8c-ci-clean-circo $PY tools/bootstrap_ci.py --baseline $R/E1.7e-phiB32-circo --runs $R/E3.8c-robust-clean-circo $R/E2.7-final-circo --out $R/E3.8c-ci-clean-circo
$PY tools/collect_results.py --results-dir $R --out $R/summary.md > /dev/null 2>&1
echo "===== queue 8 done $(date)"
