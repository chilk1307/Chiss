#!/bin/bash
# Queue 13 (15/11, E11): train phi again with the regenerated GPT phrases. The P3 set (Qwen2.5-1.5B-Instruct, one
# visible detail per sentence; best set of E10) is completed for every concept of data/oidv7-class-descriptions.csv
# (the phi GPT loss draws from the top-150 concepts of the 100k ImageNet images, almost all of the 20,931), then phi is
# trained exactly like phi_B32 (same OTI targets oti_B32, same hyper-parameters, seed 42), only --gpt-exp-name
# changes, and evaluated against phi_B32 with paired bootstrap CIs. Test files for the two servers at the end.
source /content/env.sh; source $DRIVE/queue/common.sh
echo "===== queue 13 start $(date)"
SYS=/usr/bin/python3
GP=$DRIVE/outputs/GPT_phrases
RAW=$GP/P3_qwen15i_all_raw
SET=P3_qwen15i_all
EXP=phi_B32_P3
LIC="--license Apache-2.0"

# --- 1. phrases: start from the 7,514 concepts of E10 (32 phrases for the CIRCO ones, 16 for the others)
mkdir -p $RAW
[ -s $RAW/concept_to_phrases.pkl ] || cp $GP/P3_qwen15i_val_raw/concept_to_phrases.pkl $RAW/
run $R/E11.0-gen-$SET $SYS tools/generate_phrases_llm.py --mode instruct --model Qwen/Qwen2.5-1.5B-Instruct \
  --n-phrases 16 --batch-concepts 48 --rounds 1 $LIC --out-dir $RAW
run $R/E11.0-clean-$SET $PY tools/clean_phrases.py --phrases $RAW/concept_to_phrases.pkl --out-exp $SET --out-root $GP
run $R/E11.0-stats-$SET $PY tools/phrases_stats.py --phrases $GP/$SET/concept_to_phrases.pkl \
  --similar-concepts $DRIVE/outputs/similar_concepts/imagenet/test/ViT-B32_top_250_ensembling_image_texts.csv \
  --out $R/E11.0-stats-$SET

# --- 2. phi, same recipe as phi_B32 (hyper-parameters copied from its hyperparameters.json when they exist)
HP=$DRIVE/outputs/phi_models/phi_B32/hyperparameters.json
EXTRA=$($PY - "$HP" <<'PYEOF'
import json, sys
try:
    hp = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
keys = {'batch_size': '--batch-size', 'learning_rate': '--learning-rate', 'weight_decay': '--weight-decay',
        'lambda_distil': '--lambda-distil', 'lambda_gpt': '--lambda-gpt', 'temperature': '--temperature',
        'phi_dropout': '--phi-dropout', 'top_k_concepts': '--top-k-concepts', 'seed': '--seed',
        'preprocess_type': '--preprocess-type'}
print(' '.join(f"{flag} {hp[k]}" for k, flag in keys.items() if k in hp))
PYEOF
)
echo "phi_B32 hyper-parameters reused: $EXTRA"
BASE="--clip-model-name ViT-B/32 --imagenet-dataset-path $DATA/ImageNet1K --cirr-dataset-path $DATA/CIRR --save-training"
RESUME=""; [ -f data/phi_models/$EXP/checkpoints/resume.pt ] && RESUME="--resume"
run $R/E11.1-train-$EXP $PY src/train_phi.py --exp-name $EXP $BASE --oti-exp-name oti_B32 --gpt-exp-name $SET \
  --num-epochs 100 --validation-frequency 5 --features-cache-dir $DRIVE/cache/train_features $EXTRA $RESUME

# --- 3. validation and CIs against phi_B32 (same settings as E1.7d / E1.7e / E1.8)
P="--eval-type phi --exp-name $EXP --phi-checkpoint-name phi_last.pt $CACHE"
run $R/E11.2-$EXP-cirr     $V $P $CIRR  --output-dir $R/E11.2-$EXP-cirr
run $R/E11.2-$EXP-circo    $V $P $CIRCO --output-dir $R/E11.2-$EXP-circo
run $R/E11.2-$EXP-circo-ex $V $P $CIRCO --exclude-reference --output-dir $R/E11.2-$EXP-circo-ex
run $R/E11.2-$EXP-fiq-ex   $V $P $FIQ   --exclude-reference --output-dir $R/E11.2-$EXP-fiq-ex
run $R/E11.3-ci-cirr     $PY tools/bootstrap_ci.py --baseline $R/E1.7d-phiB32-cirr     --runs $R/E11.2-$EXP-cirr     --out $R/E11.3-ci-cirr
run $R/E11.3-ci-circo    $PY tools/bootstrap_ci.py --baseline $R/E1.7e-phiB32-circo    --runs $R/E11.2-$EXP-circo    --out $R/E11.3-ci-circo
run $R/E11.3-ci-circo-ex $PY tools/bootstrap_ci.py --baseline $R/E1.8-phiB32-circo-ex --runs $R/E11.2-$EXP-circo-ex --out $R/E11.3-ci-circo-ex
run $R/E11.3-ci-fiq      $PY tools/bootstrap_ci.py --baseline $R/E1.8-phiB32-fiq-ex    --runs $R/E11.2-$EXP-fiq-ex   --out $R/E11.3-ci-fiq

# --- 4. test files (CIRR and CIRCO servers)
G="$PY src/generate_test_submission.py"
SUB=$DRIVE/outputs/submissions
run $R/E11.4-$EXP-test-cirr  $G $P $CIRR  --submission-name phiP3_cirr  --output-dir $R/E11.4-$EXP-test-cirr
run $R/E11.4-$EXP-test-circo $G $P $CIRCO --submission-name phiP3_circo --output-dir $R/E11.4-$EXP-test-circo
for d in E11.4-$EXP-test-cirr E11.4-$EXP-test-circo; do
  for f in $R/$d/*.json; do case $(basename $f) in metrics.json) ;; *) cp $f $SUB/ ;; esac; done
done
$PY tools/collect_results.py --results-dir $R --out $R/summary.md > /dev/null 2>&1
echo "===== queue 13 done $(date)"
