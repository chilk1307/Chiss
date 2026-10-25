#!/bin/bash
# Session bootstrap after a Colab reset (Drive must be mounted first, cell 1):
# code (cell 2), venv (cell 3), data (cell 4) incl. the local CIRCO copy, saved code changes, then every queue.
#   bash /content/drive/MyDrive/SEARLE/queue/boot.sh > /content/boot.log 2>&1
set -u
DRIVE=/content/drive/MyDrive/SEARLE
Q=$DRIVE/queue
REPO=/content/XLA_SEARLE
echo "[$(date +%H:%M)] boot start"
if [ -d $REPO/.git ]; then git -C $REPO pull -q; else git clone -q https://github.com/chilk1307/Chiss.git $REPO; fi
git -C $REPO log -1 --format="code %h %s"
[ -f /content/env.sh ] || bash $REPO/SEARLE-main/colab/setup_env.sh Chi
source /content/env.sh
bash $SRC/colab/prepare_data.sh cirr circo fiq imagenet

# Saved code changes (not on GitHub yet): copied over the clone
if [ -d $DRIVE/code_changes/SEARLE-main ]; then cp -r $DRIVE/code_changes/SEARLE-main/. $SRC/; echo "code_changes applied"; fi
[ -f $DRIVE/code_changes/restore_rules.json ] && cp $DRIVE/code_changes/restore_rules.json $SRC/src/restore_rules.json
[ -f $DRIVE/code_changes/restore_oracle.json ] && cp $DRIVE/code_changes/restore_oracle.json $SRC/src/restore_oracle.json

# CIRCO gallery on the local disk (Drive cannot list 123k files reliably)
L=$DATA/CIRCO_local
if [ "$(ls -f $L/COCO2017_unlabeled/unlabeled2017 2>/dev/null | grep -c jpg)" -lt 123403 ]; then
  mkdir -p $L/COCO2017_unlabeled
  cp -rn $DRIVE/datasets/CIRCO/annotations $L/
  cp -rn $DRIVE/datasets/CIRCO/COCO2017_unlabeled/annotations $L/COCO2017_unlabeled/
  cd /content && { [ -s unlabeled2017.zip ] || wget -q -c http://images.cocodataset.org/zips/unlabeled2017.zip; } \
    && unzip -q -o unlabeled2017.zip -d $L/COCO2017_unlabeled/ && rm -f unlabeled2017.zip
fi
echo "CIRCO_local: $(ls -f $L/COCO2017_unlabeled/unlabeled2017 | grep -c jpg) images"

for s in t1 t3 q1 q2 q3 t2 q4 q5 q6; do
  pgrep -f "queue/$s.sh" > /dev/null && echo "$s already running" || { nohup bash $Q/$s.sh >> $Q/$s.log 2>&1 & echo "$s started"; sleep 2; }
done
echo "[$(date +%H:%M)] boot done"
