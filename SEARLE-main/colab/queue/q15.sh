#!/bin/bash
# Queue 15 (E13): combine the improvements that so far were only tested one at a time. Evaluation only, no training
# (about 30-60 min on an A100 once the gallery caches exist).
#   E13.0  baselines re-evaluated here (phi_B32, phi_B32_P3oti), so the CIs do not depend on which account ran E12
#   E13.1  phi_B32 and phi_B32_P3oti + attribute re-ranking (final flags of E2.7) on CIRR, CIRCO, FashionIQ val
#   E13.2  degraded reference images (mixed): phi_B32 / phi_B32_P3oti, without and with automatic restoration,
#          and everything combined (P3oti + auto restoration + attribute re-ranking), CIRR val
#   E13.3  bootstrap CIs against phi_B32 and against phi_B32_P3oti alone
#   E13.4  test files of phi_B32_P3oti + attribute re-ranking (CIRR, CIRCO) for the evaluation servers
# Run after boot (cells 0-5 of SEARLE_runner.ipynb or boot.sh):
#   source /content/env.sh; nohup bash $SRC/colab/queue/q15.sh >> $DRIVE/queue/q15.log 2>&1 &
source /content/env.sh; source $SRC/colab/queue/common.sh
# second Google account (see q14.sh): results go to its own Drive folder
if [ -n "${SEARLE_WORK:-}" ]; then R=$SEARLE_WORK/results; mkdir -p $R; fi
echo "===== queue 15 start $(date)"

# --- phi_B32_P3oti: E12 may have saved it in the second account's folder; copy the two files phi needs
PHIX=phi_B32_P3oti
if [ ! -f data/phi_models/$PHIX/checkpoints/phi_last.pt ]; then
  for src in ${SEARLE_WORK:-} /content/drive/MyDrive/SEARLE_work $DRIVE; do
    if [ -f $src/outputs/phi_models/$PHIX/checkpoints/phi_last.pt ]; then
      mkdir -p data/phi_models/$PHIX/checkpoints
      cp $src/outputs/phi_models/$PHIX/checkpoints/phi_last.pt data/phi_models/$PHIX/checkpoints/
      cp $src/outputs/phi_models/$PHIX/hyperparameters.json data/phi_models/$PHIX/ 2>/dev/null
      echo "copied $PHIX from $src"; break
    fi
  done
fi
if [ ! -f data/phi_models/$PHIX/checkpoints/phi_last.pt ]; then
  echo "missing $PHIX: put checkpoints/phi_last.pt and hyperparameters.json in \$DRIVE/outputs/phi_models/$PHIX/"
  exit 1
fi

# --- attribute re-ranking flags chosen in E2.7 (q5 writes them; the fallback is the same configuration)
FINAL="--attr-rerank --attr-families color,brightness,zoom --attr-weight 1.0 --attr-topk 50 --attr-mask grabcut --attr-negation off"
[ -f $DRIVE/results/E2-final-flags.txt ] && FINAL=$(cat $DRIVE/results/E2-final-flags.txt)
echo "attribute flags: $FINAL"

X="--eval-type phi --exp-name $PHIX --phi-checkpoint-name phi_last.pt $CACHE"
declare -A SETS=( [cirr]="$CIRR" [circo]="$CIRCO" [circo-ex]="$CIRCO --exclude-reference" [fiq-ex]="$FIQ --exclude-reference" )
for k in cirr circo circo-ex fiq-ex; do
  A=${SETS[$k]}
  run $R/E13.0-phiB32-$k         $V $PHI $A --output-dir $R/E13.0-phiB32-$k
  run $R/E13.0-$PHIX-$k          $V $X   $A --output-dir $R/E13.0-$PHIX-$k
  run $R/E13.1-phiB32-attr-$k    $V $PHI $A $FINAL --output-dir $R/E13.1-phiB32-attr-$k
  run $R/E13.1-$PHIX-attr-$k     $V $X   $A $FINAL --output-dir $R/E13.1-$PHIX-attr-$k
  run $R/E13.3-ci-$k $PY tools/bootstrap_ci.py --baseline $R/E13.0-phiB32-$k \
    --runs $R/E13.1-phiB32-attr-$k $R/E13.0-$PHIX-$k $R/E13.1-$PHIX-attr-$k --out $R/E13.3-ci-$k
  run $R/E13.3-ci-attr-on-P3oti-$k $PY tools/bootstrap_ci.py --baseline $R/E13.0-$PHIX-$k \
    --runs $R/E13.1-$PHIX-attr-$k --out $R/E13.3-ci-attr-on-P3oti-$k
done

# --- degraded reference images (each image gets one random degradation and level, fixed by its name)
run $R/E13.2-phiB32-mixed-cirr          $V $PHI $CIRR --degrade mixed --output-dir $R/E13.2-phiB32-mixed-cirr
run $R/E13.2-phiB32-mixed-auto-cirr     $V $PHI $CIRR --degrade mixed --restore auto --output-dir $R/E13.2-phiB32-mixed-auto-cirr
run $R/E13.2-$PHIX-mixed-cirr           $V $X   $CIRR --degrade mixed --output-dir $R/E13.2-$PHIX-mixed-cirr
run $R/E13.2-$PHIX-mixed-auto-cirr      $V $X   $CIRR --degrade mixed --restore auto --output-dir $R/E13.2-$PHIX-mixed-auto-cirr
run $R/E13.2-$PHIX-mixed-auto-attr-cirr $V $X   $CIRR --degrade mixed --restore auto $FINAL --output-dir $R/E13.2-$PHIX-mixed-auto-attr-cirr
run $R/E13.3-ci-mixed-cirr $PY tools/bootstrap_ci.py --baseline $R/E13.2-phiB32-mixed-cirr \
  --runs $R/E13.2-phiB32-mixed-auto-cirr $R/E13.2-$PHIX-mixed-cirr $R/E13.2-$PHIX-mixed-auto-cirr \
         $R/E13.2-$PHIX-mixed-auto-attr-cirr --out $R/E13.3-ci-mixed-cirr

# --- test files (clean test images: no restoration)
G="$PY src/generate_test_submission.py"
SUB=${SEARLE_WORK:-$DRIVE}/outputs/submissions; mkdir -p $SUB
run $R/E13.4-test-cirr  $G $X $CIRR  $FINAL --submission-name phiP3oti_attr_cirr  --output-dir $R/E13.4-test-cirr
run $R/E13.4-test-circo $G $X $CIRCO $FINAL --submission-name phiP3oti_attr_circo --output-dir $R/E13.4-test-circo
for d in E13.4-test-cirr E13.4-test-circo; do
  for f in $R/$d/*.json; do case $(basename $f) in metrics.json) ;; *) cp $f $SUB/ ;; esac; done
done
$PY tools/collect_results.py --results-dir $R --out $R/summary.md > /dev/null 2>&1
echo "files to submit: $SUB/phiP3oti_attr_cirr.json, subset_phiP3oti_attr_cirr.json (CIRR server), phiP3oti_attr_circo.json (CIRCO server)"
echo "===== queue 15 done $(date)"
