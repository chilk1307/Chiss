#!/bin/bash
# Train queue 1: wait for T1, then D7-D8 screening of the robust-phi variants on shard s0 (E3.8a),
# then D9 degraded features of the 100k images (needed by D10).
source $DRIVE/queue/common.sh
BASE="--clip-model-name ViT-B/32 --imagenet-dataset-path $DATA/ImageNet1K --cirr-dataset-path $DATA/CIRR --save-training"
FCACHE="--features-cache-dir $DRIVE/cache/train_features"
TRAIN="$BASE --oti-exp-name oti_B32_s0 --train-on-oti-names --num-epochs 50 --validation-frequency 5 $FCACHE"
L=$R/E3.8a-train; mkdir -p $L
echo "===== train queue 1 start $(date)"

while pgrep -f "exp-name phi_scale_s0" > /dev/null; do sleep 60; done
# T1 cut by a Colab reset: resume it first (validation every 5 epochs, so the last line is epoch 100 when complete)
T1_LAST=$(tail -n 1 data/phi_models/phi_scale_s0/validation_metrics.csv 2>/dev/null | cut -d, -f1)
if [ "$T1_LAST" != "100" ] && [ -f data/phi_models/phi_scale_s0/checkpoints/resume.pt ]; then
  echo "[$(date +%H:%M)] RESUME T1 (last validated epoch $T1_LAST)"
  $PY src/train_phi.py --exp-name phi_scale_s0 $BASE --oti-exp-name oti_B32_s0 --train-on-oti-names --num-epochs 100 \
    --validation-frequency 5 $FCACHE --resume >> $R/E1.14a-train-s0/train.log 2>&1
fi
echo "[$(date +%H:%M)] T1 finished: $(tail -n 1 data/phi_models/phi_scale_s0/validation_metrics.csv 2>/dev/null)"

start() {  # start <exp name> <extra flags...>: in the background unless its phi_last.pt exists
  local name=$1; shift
  if [ -f data/phi_models/$name/checkpoints/phi_last.pt ] && [ -f $L/$name.done ]; then echo "SKIP $name"; return; fi
  local resume=""; [ -f data/phi_models/$name/checkpoints/resume.pt ] && resume="--resume"
  echo "[$(date +%H:%M)] START $name $resume"
  ( $PY src/train_phi.py --exp-name $name $TRAIN "$@" $resume >> $L/$name.log 2>&1 && touch $L/$name.done
    echo "[$(date +%H:%M)] END   $name ($( [ -f $L/$name.done ] && echo OK || echo FAIL))" ) &
}

start phi_s0_clean
start phi_s0_degaug --aug-feature-dir $DRIVE/cache/imagenet_s0 --aug-prob 0.5
wait
start phi_s0_noise --feature-noise 0.1
echo "[$(date +%H:%M)] START precompute_full"
$PY tools/precompute_degraded_features.py --imagenet-path $DATA/ImageNet1K --oti-exp-name oti_B32 \
  --degrade none,gauss_noise:1,gauss_noise:2,gauss_blur:1,gauss_blur:2,jpeg:1,jpeg:2,low_light:1,low_light:2 \
  --out-dir $DRIVE/cache/imagenet_full > $L/precompute_full.log 2>&1 && touch $L/precompute_full.done
echo "[$(date +%H:%M)] END   precompute_full ($( [ -f $L/precompute_full.done ] && echo OK || echo FAIL))"
wait
echo "===== train queue 1 done $(date)"
