#!/bin/bash
# Queue 12c (E10): more seeds on CIRCO val. The seed control E10.1a (GPTNeo27B, seed 43: mAP@10 5.95) is 1.3 points
# below E7.1a (seed 42: 7.22), so one run per phrase set says little with 220 queries: seeds 43-45 for every set.
source /content/env.sh; source /content/drive/MyDrive/SEARLE/queue/q12env.sh
echo "===== queue 12c start $(date)"
for S in 44 45; do
  oti E10.1a-gptneo-seed$S-circo $CIRCOV --gpt-exp-name GPTNeo27B --seed $S; ev E10.1a-gptneo-seed$S-circo $CIRCO
done
for S in 43 44 45; do
  oti E10.1b-P2-seed$S-circo $CIRCOV --gpt-exp-name P2_qwen15_val --seed $S; ev E10.1b-P2-seed$S-circo $CIRCO
done
while [ ! -f $R/E10.0-clean-P3_qwen15i_val/.done ]; do sleep 60; done
for S in 43 44 45; do
  oti E10.1c-P3-seed$S-circo $CIRCOV --gpt-exp-name P3_qwen15i_val --seed $S; ev E10.1c-P3-seed$S-circo $CIRCO
done
echo "===== queue 12c done $(date)"
