#!/bin/bash
# Queue 14 (E12): OTI on the 100k ImageNet images again, with the P3 GPT phrases, in 3 parts that can run in separate
# Colab sessions:   bash q14.sh 1   |   bash q14.sh 2   |   bash q14.sh 3
# Part N inverts the images of the old shard oti_B32_s<N-1> (about 33,000 images, ~12 h on an A100, ~24 h on an L4)
# into oti_B32_P3_s<N-1>, with the same settings as oti_B32 (random init, lr 0.02, 350 steps, seed 42) except
# --gpt-exp-name. A cut session is resumed by running the same part again (tokens are saved every 5 batches).
# The part that finds the 3 shards complete also merges them (oti_B32_P3), trains phi_B32_P3oti exactly like
# phi_B32 (only the OTI targets and the GPT phrases change), evaluates it against phi_B32 and phi_B32_P3 with CIs and
# writes the test files.
PART=${1:?usage: bash q14.sh 1|2|3}
case $PART in 1|2|3) ;; *) echo "part must be 1, 2 or 3"; exit 1 ;; esac
source /content/env.sh; source $DRIVE/queue/common.sh
S=s$((PART - 1))
SET=P3_qwen15i_allf
OTI_ROOT=data/oti_pseudo_tokens/imagenet/test
echo "===== queue 14 part $PART ($S) start $(date)"

# --- 1. OTI of this part
EXP=oti_B32_P3_$S
# same OTI settings as the old shard (read from its hyperparameters.json when it exists)
OHP=$OTI_ROOT/oti_B32_$S/hyperparameters.json
OEXTRA=$($PY - "$OHP" <<'PYEOF2'
import json, sys
try:
    hp = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
keys = {'learning_rate': '--learning-rate', 'weight_decay': '--weight-decay', 'batch_size': '--batch-size',
        'top_k': '--top-k', 'oti_steps': '--oti-steps', 'lambda_gpt': '--lambda_gpt', 'lambda_cos': '--lambda_cos',
        'ema_decay': '--ema-decay', 'seed': '--seed', 'preprocess_type': '--preprocess-type'}
print(' '.join(f"{flag} {hp[k]}" for k, flag in keys.items() if k in hp))
PYEOF2
)
echo "OTI settings of oti_B32_$S reused: ${OEXTRA:-defaults}"
RES=""; [ -f $OTI_ROOT/$EXP/image_names.pkl ] && RES="--resume-experiment"
run $R/E12.0-oti-$S $PY src/oti_inversion.py --exp-name $EXP --clip-model-name ViT-B/32 --dataset imagenet \
  --dataset-path $DATA/ImageNet1K --split test --gpt-exp-name $SET --image-names $OTI_ROOT/oti_B32_$S/image_names.pkl \
  --init random --learning-rate 0.02 --save-frequency 5 $OEXTRA $RES

# --- 2. once the 3 parts are done (whichever part finishes last does this; the parts can also run at the same
#        time on one GPU, OTI alone does not use all of an A100)
for s in s0 s1 s2; do
  if [ ! -f $R/E12.0-oti-$s/.done ]; then
    echo "part $PART done; waiting for $s (run its part in another session)"
    echo "===== queue 14 part $PART done $(date)"; exit 0
  fi
done
# several parts may run at the same time (same runtime or not): only one of them goes on (delete
# $R/E12.1-lock to retry after a failure)
mkdir $R/E12.1-lock 2>/dev/null || { echo "another part does the merge and the training"; echo "===== queue 14 part $PART done $(date)"; exit 0; }
run $R/E12.1-merge $PY tools/merge_oti.py --shards oti_B32_P3_s0,oti_B32_P3_s1,oti_B32_P3_s2 --out oti_B32_P3

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
PHIX=phi_B32_P3oti
BASE="--clip-model-name ViT-B/32 --imagenet-dataset-path $DATA/ImageNet1K --cirr-dataset-path $DATA/CIRR --save-training"
RESUME=""; [ -f data/phi_models/$PHIX/checkpoints/resume.pt ] && RESUME="--resume"
run $R/E12.2-train-$PHIX $PY src/train_phi.py --exp-name $PHIX $BASE --oti-exp-name oti_B32_P3 --gpt-exp-name $SET \
  --num-epochs 100 --validation-frequency 5 --features-cache-dir $DRIVE/cache/train_features $EXTRA $RESUME

P="--eval-type phi --exp-name $PHIX --phi-checkpoint-name phi_last.pt $CACHE"
run $R/E12.3-$PHIX-cirr     $V $P $CIRR  --output-dir $R/E12.3-$PHIX-cirr
run $R/E12.3-$PHIX-circo    $V $P $CIRCO --output-dir $R/E12.3-$PHIX-circo
run $R/E12.3-$PHIX-circo-ex $V $P $CIRCO --exclude-reference --output-dir $R/E12.3-$PHIX-circo-ex
run $R/E12.3-$PHIX-fiq-ex   $V $P $FIQ   --exclude-reference --output-dir $R/E12.3-$PHIX-fiq-ex
run $R/E12.4-ci-cirr     $PY tools/bootstrap_ci.py --baseline $R/E1.7d-phiB32-cirr \
  --runs $R/E11.2-phi_B32_P3-cirr $R/E12.3-$PHIX-cirr --out $R/E12.4-ci-cirr
run $R/E12.4-ci-circo    $PY tools/bootstrap_ci.py --baseline $R/E1.7e-phiB32-circo \
  --runs $R/E11.2-phi_B32_P3-circo $R/E12.3-$PHIX-circo --out $R/E12.4-ci-circo
run $R/E12.4-ci-circo-ex $PY tools/bootstrap_ci.py --baseline $R/E1.8-phiB32-circo-ex \
  --runs $R/E11.2-phi_B32_P3-circo-ex $R/E12.3-$PHIX-circo-ex --out $R/E12.4-ci-circo-ex
run $R/E12.4-ci-fiq      $PY tools/bootstrap_ci.py --baseline $R/E1.8-phiB32-fiq-ex \
  --runs $R/E11.2-phi_B32_P3-fiq-ex $R/E12.3-$PHIX-fiq-ex --out $R/E12.4-ci-fiq

G="$PY src/generate_test_submission.py"
SUB=$DRIVE/outputs/submissions
run $R/E12.5-test-cirr  $G $P $CIRR  --submission-name phiP3oti_cirr  --output-dir $R/E12.5-test-cirr
run $R/E12.5-test-circo $G $P $CIRCO --submission-name phiP3oti_circo --output-dir $R/E12.5-test-circo
for d in E12.5-test-cirr E12.5-test-circo; do
  for f in $R/$d/*.json; do case $(basename $f) in metrics.json) ;; *) cp $f $SUB/ ;; esac; done
done
$PY tools/collect_results.py --results-dir $R --out $R/summary.md > /dev/null 2>&1
echo "===== queue 14 part $PART ($S) done, all parts merged and evaluated $(date)"
