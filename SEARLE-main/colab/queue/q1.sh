#!/bin/bash
# Queue 1: H5 (E1.7b-f), H4 + C4 (test files), H9 (E2.0, E2.2), H6 (E1.8, E1.9), D5 (E3.1), D6 (E3.5), D3/D4 (E4.1-E4.3), H7
source $DRIVE/queue/common.sh
echo "===== queue 1 start $(date)"

# --- H5: no-regression baselines
run $R/E1.7b-searle-circo $V --eval-type searle $CIRCO $CACHE --output-dir $R/E1.7b-searle-circo
run $R/E1.7c-searle-fiq   $V --eval-type searle $FIQ $CACHE --output-dir $R/E1.7c-searle-fiq
run $R/E1.7d-phiB32-cirr  $V $PHI $CIRR  --output-dir $R/E1.7d-phiB32-cirr
run $R/E1.7e-phiB32-circo $V $PHI $CIRCO --output-dir $R/E1.7e-phiB32-circo
run $R/E1.7f-phiB32-fiq   $V $PHI $FIQ   --output-dir $R/E1.7f-phiB32-fiq

# --- H4 + C4: test submission files (uploaded to the servers by hand)
S=$R/H4-submissions
run $S/searle_cirr  $PY src/generate_test_submission.py --submission-name searle_official_cirr  --eval-type searle $CIRR  $CACHE --output-dir $S/searle_cirr
run $S/searle_circo $PY src/generate_test_submission.py --submission-name searle_official_circo --eval-type searle $CIRCO $CACHE --output-dir $S/searle_circo
run $S/check_phiB32 $PY src/generate_test_submission.py --submission-name check_phi_B32_cirr $PHI $CIRR --output-dir $S/check_phiB32
mkdir -p $DRIVE/outputs/submissions && cp -n data/test_submissions/cirr/*.json data/test_submissions/circo/*.json $DRIVE/outputs/submissions/ 2>/dev/null
$PY - <<'EOF' > $S/C4_compare.txt 2>&1
import json
a = json.load(open("/content/XLA_SEARLE/SEARLE-main/data/test_submissions/cirr/check_phi_B32_cirr.json"))
b = json.load(open("/content/XLA_SEARLE/outputs/submissions/phi_B32_cirr.json"))
keys = [k for k in a if k not in ("version", "metric")]
same = sum(a[k][:10] == b[k][:10] for k in keys)
print(same, "/", len(keys), "queries with identical top-10 (%.2f%%)" % (100 * same / len(keys)))
EOF
echo "[$(date +%H:%M)] C4: $(cat $S/C4_compare.txt)"

# --- H9: attribute statistics and meter agreement (decision M3)
for D in cirr fiq circo; do
  case $D in cirr) A=$CIRR;; fiq) A=$FIQ;; circo) A=$CIRCO;; esac
  run $R/E2.0-stats-$D $PY tools/attribute_stats.py $A --out $R/E2.0-stats-$D
  run $R/E2.2-agree-$D $PY tools/attribute_stats.py --agreement --attr-mask auto $A $CACHE --out $R/E2.2-agree-$D
done

# --- H6: CLIP baselines and classical CBIR
for T in image text sum; do
  run $R/E1.8-$T-cirr     $V --eval-type $T $CIRR  $CACHE --output-dir $R/E1.8-$T-cirr
  run $R/E1.8-$T-circo-ex $V --eval-type $T $CIRCO $CACHE --exclude-reference --output-dir $R/E1.8-$T-circo-ex
  run $R/E1.8-$T-fiq-ex   $V --eval-type $T $FIQ   $CACHE --exclude-reference --output-dir $R/E1.8-$T-fiq-ex
done
run $R/E1.8-searle-circo-ex $V --eval-type searle $CIRCO $CACHE --exclude-reference --output-dir $R/E1.8-searle-circo-ex
run $R/E1.8-searle-fiq-ex   $V --eval-type searle $FIQ   $CACHE --exclude-reference --output-dir $R/E1.8-searle-fiq-ex
run $R/E1.8-phiB32-circo-ex $V $PHI $CIRCO --exclude-reference --output-dir $R/E1.8-phiB32-circo-ex
run $R/E1.8-phiB32-fiq-ex   $V $PHI $FIQ   --exclude-reference --output-dir $R/E1.8-phiB32-fiq-ex
for F in color lbp hog gabor all; do
  run $R/E1.9-cbir-$F-cirr   $V --eval-type cbir --cbir-features $F $CIRR $CACHE --output-dir $R/E1.9-cbir-$F-cirr
  run $R/E1.9-cbir-$F-fiq-ex $V --eval-type cbir --cbir-features $F $FIQ  $CACHE --exclude-reference --output-dir $R/E1.9-cbir-$F-fiq-ex
done
run $R/E1.9-cbir-text-cirr $V --eval-type cbir --cbir-features all --cbir-text-weight 1.0 $CIRR $CACHE --output-dir $R/E1.9-cbir-text-cirr

# --- D5: robustness to degraded queries
for D in gauss_noise salt_pepper gauss_blur motion_blur jpeg low_light low_res color_cast; do
  for L in 1 2; do
    run $R/E3.1-$D$L-cirr  $V $PHI $CIRR  --degrade $D:$L --save-quality --output-dir $R/E3.1-$D$L-cirr
    run $R/E3.1-$D$L-circo $V $PHI $CIRCO --degrade $D:$L --save-quality --output-dir $R/E3.1-$D$L-circo
  done
done

# --- D6: frequency domain
for RT in 0.05 0.1 0.2 0.3 0.5; do
  run $R/E3.5-lp$RT-cirr $V $PHI $CIRR --freq-filter lowpass:$RT  --output-dir $R/E3.5-lp$RT-cirr
  run $R/E3.5-hp$RT-cirr $V $PHI $CIRR --freq-filter highpass:$RT --output-dir $R/E3.5-hp$RT-cirr
done

# --- D3, D4: padding, interpolation, flip TTA
for RATIO in 1.0 1.25 1.5 2.0; do
  run $R/E4.1-pad$RATIO-cirr $V $PHI $CIRR --targetpad-ratio $RATIO --output-dir $R/E4.1-pad$RATIO-cirr
done
run $R/E4.1-clipcrop-cirr $V $PHI $CIRR --preprocess-type clip --output-dir $R/E4.1-clipcrop-cirr
for I in nearest bilinear bicubic lanczos; do
  run $R/E4.2-interp-$I-cirr $V $PHI $CIRR --interp $I --output-dir $R/E4.2-interp-$I-cirr
done
for T in flip-ref flip-gallery flip-both; do
  run $R/E4.3-$T-cirr $V $PHI $CIRR --tta $T --output-dir $R/E4.3-$T-cirr
done

# --- H7: confidence intervals and summary
run $R/E1.4-ci-cirr  $PY tools/bootstrap_ci.py --baseline $R/E1.7d-phiB32-cirr --runs $R/E1.7a-searle-cirr $R/E1.8-*-cirr $R/E1.9-*-cirr --out $R/E1.4-ci-cirr
run $R/E1.4-ci-circo $PY tools/bootstrap_ci.py --baseline $R/E1.8-phiB32-circo-ex --runs $R/E1.8-*-circo-ex --out $R/E1.4-ci-circo
run $R/E1.4-ci-fiq   $PY tools/bootstrap_ci.py --baseline $R/E1.8-phiB32-fiq-ex --runs $R/E1.8-*-fiq-ex $R/E1.9-*-fiq-ex --out $R/E1.4-ci-fiq
$PY tools/collect_results.py --results-dir $R --out $R/summary.md > /dev/null 2>&1
echo "===== queue 1 done $(date)"
