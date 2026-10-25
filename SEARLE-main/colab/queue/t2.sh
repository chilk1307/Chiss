#!/bin/bash
# Train queue 2: D10 phi_B32_robust (full 100k, 100 epochs, chosen variant); H16 cosine runs in t3.sh.
# Waits for: train queue 1 (precompute_full) and queue 3 (variant choice).
source $DRIVE/queue/common.sh
BASE="--clip-model-name ViT-B/32 --imagenet-dataset-path $DATA/ImageNet1K --cirr-dataset-path $DATA/CIRR --save-training"
while pgrep -f "queue/t1.sh" > /dev/null || pgrep -f "queue/q3.sh" > /dev/null; do sleep 120; done
echo "===== train queue 2 start $(date)"
CHOICE=$(grep '^CHOICE' $R/E3.8a-choice.txt 2>/dev/null | awk '{print $2}')
[ -f $R/E3.8a-train/precompute_full.done ] || { echo "precompute_full missing: D10 not started"; CHOICE=none; }
echo "variant: $CHOICE"
case $CHOICE in
  phi_s0_degaug) ROBUST="--aug-feature-dir $DRIVE/cache/imagenet_full --aug-prob 0.5" ;;
  phi_s0_noise)  ROBUST="--feature-noise 0.1" ;;
  *)             ROBUST="" ;;
esac

train() {  # train <exp name> <log dir> <flags...>
  local name=$1 dir=$2; shift 2; mkdir -p $dir
  if [ -f $dir/train.done ]; then echo "SKIP $name"; return; fi
  local resume=""; [ -f data/phi_models/$name/checkpoints/resume.pt ] && resume="--resume"
  echo "[$(date +%H:%M)] START $name $resume"
  ( $PY src/train_phi.py --exp-name $name $BASE --oti-exp-name oti_B32 --num-epochs 100 --validation-frequency 5 "$@" $resume \
      >> $dir/train.log 2>&1 && touch $dir/train.done
    echo "[$(date +%H:%M)] END   $name ($( [ -f $dir/train.done ] && echo OK || echo FAIL))" ) &
}
[ -n "$ROBUST" ] && train phi_B32_robust $R/E3.8b-train-robust $ROBUST
wait
echo "===== train queue 2 done $(date)"
