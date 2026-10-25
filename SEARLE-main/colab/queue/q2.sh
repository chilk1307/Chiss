#!/bin/bash
# Queue 2 (starts when queue 1 ends): C14 calibration, E3.2 fixed restoration, E4.3b, E2.3 attribute sweep,
# E4.4 fusion, E4.5 prompt ensemble, E3.6/E3.7 meter robustness + figures, H8 analyses, E3.4
source $DRIVE/queue/common.sh
while pgrep -f "queue/q1.sh" > /dev/null; do sleep 60; done
echo "===== queue 2 start $(date)"

# --- C14: calibrate the thresholds of --restore auto on 300 CIRR train images (copy kept on Drive)
mkdir -p $DRIVE/code_changes
run $R/E3.3-calibration $PY tools/calibrate_restore_rules.py --images-dir $DATA/CIRR/train --n 300 --out src/restore_rules.json --report-dir $R/E3.3-calibration
cp src/restore_rules.json $DRIVE/code_changes/restore_rules.json

# --- D8: fixed restoration after a level-2 degradation (E3.2)
e32() { run $R/E3.2-${1/:/}-$2-cirr $V $PHI $CIRR --degrade $1 --restore $2 --save-quality --output-dir $R/E3.2-${1/:/}-$2-cirr; }
for M in gaussian bilateral nlm; do e32 gauss_noise:2 $M; done
e32 salt_pepper:2 median
for M in unsharp wiener richardson_lucy; do e32 gauss_blur:2 $M; done
for M in wiener richardson_lucy; do e32 motion_blur:2 $M; done
for M in deblock bilateral; do e32 jpeg:2 $M; done
for M in histeq clahe gamma retinex; do e32 low_light:2 $M; done
e32 low_res:2 unsharp
for M in gray_world white_patch; do e32 color_cast:2 $M; done

# --- E4.3b: left/right queries under flip TTA
run $R/E4.3b-leftright-cirr $PY tools/query_taxonomy.py $CIRR --categories spatial --runs $R/E1.7d-phiB32-cirr $R/E4.3-*-cirr --out $R/E4.3b-leftright-cirr

# --- H10: attribute re-ranking, one family x weight (E2.3)
for FAM in color brightness colorfulness pattern zoom; do
  for W in 0.05 0.1 0.2 0.3; do
    run $R/E2.3-$FAM-w$W-cirr $V $PHI $CIRR --attr-rerank --attr-families $FAM --attr-weight $W --output-dir $R/E2.3-$FAM-w$W-cirr
    run $R/E2.3-$FAM-w$W-fiq  $V $PHI $FIQ --exclude-reference --attr-rerank --attr-families $FAM --attr-weight $W --output-dir $R/E2.3-$FAM-w$W-fiq
  done
done

# --- H11: score fusion and prompt ensemble (E4.4, E4.5)
for A in 0.05 0.1 0.2 0.3 0.5; do
  run $R/E4.4-fusion-a$A-cirr $V $PHI $CIRR --fusion-weight $A --output-dir $R/E4.4-fusion-a$A-cirr
done
run $R/E4.5-ensemble-cirr $V $PHI $CIRR --prompt-set ensemble --output-dir $R/E4.5-ensemble-cirr

# --- D12: robustness of the attribute meters, figures (E3.6, E3.7)
for D in gauss_noise:2 jpeg:2 low_light:2 color_cast:2; do
  run $R/E3.6-agree-${D/:/}-fiq $PY tools/attribute_stats.py --agreement --attr-mask auto $FIQ --degrade $D $CACHE --out $R/E3.6-agree-${D/:/}-fiq
done
run $R/E3.6-agree-colorcast2-grayworld-fiq $PY tools/attribute_stats.py --agreement --attr-mask auto $FIQ --degrade color_cast:2 --restore gray_world $CACHE --out $R/E3.6-agree-colorcast2-grayworld-fiq
run $R/E3.7-figures $PY tools/visualize_degradations.py $CIRR --run $R/E1.7d-phiB32-cirr --degrade gauss_noise:2,low_light:2,jpeg:2,color_cast:2 --restore auto --n 8 $CACHE --out $R/E3.7-figures

# --- H8: query taxonomy, keep/edit, image-word vectors (E1.10, E1.11, E1.15)
run $R/E1.10-taxonomy-cirr  $PY tools/query_taxonomy.py $CIRR  --runs $R/E1.7a-searle-cirr $R/E1.7d-phiB32-cirr $R/E1.8-*-cirr --out $R/E1.10-taxonomy-cirr
run $R/E1.10-taxonomy-circo $PY tools/query_taxonomy.py $CIRCO --runs $R/E1.8-*-circo-ex --out $R/E1.10-taxonomy-circo
run $R/E1.10-taxonomy-fiq   $PY tools/query_taxonomy.py $FIQ   --runs $R/E1.8-*-fiq-ex --out $R/E1.10-taxonomy-fiq
run $R/E1.11-keepedit-cirr  $PY tools/keep_edit_analysis.py $CIRR  --runs $R/E1.7d-phiB32-cirr $R/E1.8-sum-cirr $CACHE --out $R/E1.11-keepedit-cirr
run $R/E1.11-keepedit-circo $PY tools/keep_edit_analysis.py $CIRCO --runs $R/E1.7e-phiB32-circo $R/E1.8-sum-circo-ex $CACHE --out $R/E1.11-keepedit-circo
run $R/E1.15a-tokens-cirr   $PY tools/vector_analysis.py tokens $CIRR  $CACHE --sources searle phi:phi_B32 --snap-eval --out $R/E1.15a-tokens-cirr
run $R/E1.15a-tokens-circo  $PY tools/vector_analysis.py tokens $CIRCO $CACHE --sources searle phi:phi_B32 --snap-eval --out $R/E1.15a-tokens-circo
run $R/E1.15b-gap-cirr      $PY tools/vector_analysis.py gap $CIRR  $CACHE --sources phi:phi_B32 --out $R/E1.15b-gap-cirr
run $R/E1.15b-gap-circo     $PY tools/vector_analysis.py gap $CIRCO $CACHE --sources phi:phi_B32 --out $R/E1.15b-gap-circo

# --- E3.4: image quality vs retrieval
run $R/E3.4-quality-cirr $PY tools/quality_vs_retrieval.py --runs $R/E3.1-*-cirr $R/E3.2-* --out $R/E3.4-quality-cirr
$PY tools/collect_results.py --results-dir $R --out $R/summary.md > /dev/null 2>&1
echo "===== queue 2 done $(date)"
