#!/bin/bash
# Train queue 3: H16 phi_B32_cosine (E8.2a), independent of the robust-variant choice, so it starts at once.
source $DRIVE/queue/common.sh
BASE="--clip-model-name ViT-B/32 --imagenet-dataset-path $DATA/ImageNet1K --cirr-dataset-path $DATA/CIRR --save-training"
dir=$R/E8.2a-train-cosine; mkdir -p $dir
echo "===== train queue 3 start $(date)"
if [ -f $dir/train.done ]; then echo "SKIP phi_B32_cosine"; exit 0; fi
resume=""; [ -f data/phi_models/phi_B32_cosine/checkpoints/resume.pt ] && resume="--resume"
echo "[$(date +%H:%M)] START phi_B32_cosine $resume"
$PY src/train_phi.py --exp-name phi_B32_cosine $BASE --oti-exp-name oti_B32 --num-epochs 100 --validation-frequency 5 \
  --lr-scheduler cosine --warmup-epochs 5 $resume >> $dir/train.log 2>&1 && touch $dir/train.done
echo "[$(date +%H:%M)] END   phi_B32_cosine ($( [ -f $dir/train.done ] && echo OK || echo FAIL))"
