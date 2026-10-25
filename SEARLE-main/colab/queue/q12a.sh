#!/bin/bash
# Queue 12a (E10, split from q12 so that P2 does not wait for the slow P3 generation): P2 set, seed controls with
# the original GPTNeo27B phrases, OTI + evaluation on CIRCO val and CIRR val.
source /content/env.sh; source /content/drive/MyDrive/SEARLE/queue/q12env.sh
echo "===== queue 12a start $(date)"
run $R/E10.0-gen-P2-cirr $SYS tools/generate_phrases_llm.py $P2 $SR --out-dir $GP/P2_qwen15_val_raw
clean_set P2_qwen15_val
run $R/E10.0-stats-GPTNeo27B $PY tools/phrases_stats.py --phrases $GP/GPTNeo27B/concept_to_phrases.pkl \
  --similar-concepts $SIM/cirr/val/ViT-B32_top_250_ensembling_image_texts.csv --top-k-phi 0 --out $R/E10.0-stats-GPTNeo27B
oti E10.1a-gptneo-seed43-circo $CIRCOV --gpt-exp-name GPTNeo27B --seed 43
ev E10.1a-gptneo-seed43-circo $CIRCO
oti E10.1b-P2-circo $CIRCOV --gpt-exp-name P2_qwen15_val
ev E10.1b-P2-circo $CIRCO
oti E10.2b-P2-cirr $CIRRV --gpt-exp-name P2_qwen15_val
ev E10.2b-P2-cirr $CIRR
oti E10.2a-gptneo-seed43-cirr $CIRRV --gpt-exp-name GPTNeo27B --seed 43
ev E10.2a-gptneo-seed43-cirr $CIRR
echo "===== queue 12a done $(date)"
