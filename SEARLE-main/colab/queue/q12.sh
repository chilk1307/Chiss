#!/bin/bash
# Queue 12 (25/10): small test of regenerated GPT phrases (E10). Two new phrase sets for the concepts of the CIRCO val
# and CIRR val reference images (top-15 per image, E7.0), both with Qwen2.5-1.5B (Apache-2.0):
#   P2 = base model, the original prompt "a photo of {concept} that" (newer language model, same task)
#   P3 = instruct model, sentences about one visible detail (tools/prompts/p3.txt)
# then OTI from random init (lr 0.02, 350 steps, as E7.1a / E7.2a with GPTNeo27B), a seed control with the
# original phrases, and paired CIs against E7.1a / E7.2a. Generation uses Colab's own Python (recent transformers).
source $DRIVE/queue/common.sh
echo "===== queue 12 start $(date)"
SYS=/usr/bin/python3
SIM=$DRIVE/outputs/similar_concepts
GP=$DRIVE/outputs/GPT_phrases
CIRCOV="--dataset circo --dataset-path $DATA/CIRCO_local --split val"
CIRRV="--dataset cirr --dataset-path $DATA/CIRR --split val"
SC="--similar-concepts $SIM/circo/val/ViT-B32_top_250_ensembling_image_texts.csv --top-k-phi 0"
SR="--similar-concepts $SIM/cirr/val/ViT-B32_top_250_ensembling_image_texts.csv --top-k-phi 0"
LIC="--license Apache-2.0"
P2="--mode continue --model Qwen/Qwen2.5-1.5B --n-phrases 64 --batch-concepts 8 $LIC"
P3="--mode instruct --model Qwen/Qwen2.5-1.5B-Instruct --n-phrases 32 --batch-concepts 32 --rounds 2 $LIC"

# --- generation (CIRCO concepts first, then the CIRR ones are added to the same set), cleaning, statistics
run $R/E10.0-gen-P2-circo $SYS tools/generate_phrases_llm.py $P2 $SC --out-dir $GP/P2_qwen15_val_raw
run $R/E10.0-gen-P3-circo $SYS tools/generate_phrases_llm.py $P3 $SC --out-dir $GP/P3_qwen15i_val_raw
run $R/E10.0-gen-P2-cirr $SYS tools/generate_phrases_llm.py $P2 $SR --out-dir $GP/P2_qwen15_val_raw
run $R/E10.0-gen-P3-cirr $SYS tools/generate_phrases_llm.py $P3 $SR --out-dir $GP/P3_qwen15i_val_raw
for N in P2_qwen15_val P3_qwen15i_val; do
  run $R/E10.0-clean-$N $PY tools/clean_phrases.py --phrases $GP/${N}_raw/concept_to_phrases.pkl --out-exp $N --out-root $GP
  run $R/E10.0-stats-$N $PY tools/phrases_stats.py --phrases $GP/$N/concept_to_phrases.pkl \
    --similar-concepts $SIM/cirr/val/ViT-B32_top_250_ensembling_image_texts.csv --top-k-phi 0 --out $R/E10.0-stats-$N
done
run $R/E10.0-stats-GPTNeo27B $PY tools/phrases_stats.py --phrases $GP/GPTNeo27B/concept_to_phrases.pkl \
  --similar-concepts $SIM/cirr/val/ViT-B32_top_250_ensembling_image_texts.csv --top-k-phi 0 --out $R/E10.0-stats-GPTNeo27B

oti() { local E=$1; shift; run $R/$E-oti $PY src/oti_inversion.py --exp-name $E --clip-model-name ViT-B/32 \
  --init random --learning-rate 0.02 --snapshot-steps 50,350 "$@"; }
ev() { local E=$1; shift; run $R/$E-ema350 $V --eval-type oti --exp-name $E --token-file ema_oti_pseudo_tokens_step350.pt "$@" $CACHE --output-dir $R/$E-ema350; }

# --- CIRCO val (220 images, ~5 min each): seed control, P2, P3
oti E10.1a-gptneo-seed43-circo $CIRCOV --gpt-exp-name GPTNeo27B --seed 43
oti E10.1b-P2-circo $CIRCOV --gpt-exp-name P2_qwen15_val
oti E10.1c-P3-circo $CIRCOV --gpt-exp-name P3_qwen15i_val
for E in E10.1a-gptneo-seed43-circo E10.1b-P2-circo E10.1c-P3-circo; do ev $E $CIRCO; done
run $R/E10.1-ci-circo $PY tools/bootstrap_ci.py --baseline $R/E7.1a-random-lr0.02-circo-step350-ema_oti_pseudo_tokens \
  --runs $R/E10.1a-gptneo-seed43-circo-ema350 $R/E10.1b-P2-circo-ema350 $R/E10.1c-P3-circo-ema350 --out $R/E10.1-ci-circo

# --- CIRR val (~2.2k images, ~48 min each): P2, seed control, P3
oti E10.2b-P2-cirr $CIRRV --gpt-exp-name P2_qwen15_val
ev E10.2b-P2-cirr $CIRR
oti E10.2a-gptneo-seed43-cirr $CIRRV --gpt-exp-name GPTNeo27B --seed 43
ev E10.2a-gptneo-seed43-cirr $CIRR
oti E10.2c-P3-cirr $CIRRV --gpt-exp-name P3_qwen15i_val
ev E10.2c-P3-cirr $CIRR
run $R/E10.2-ci-cirr $PY tools/bootstrap_ci.py --baseline $R/E7.2a-random-lr0.02-cirr-step350-ema_oti_pseudo_tokens \
  --runs $(ls -d $R/E10.2*-cirr-ema350 2>/dev/null) --out $R/E10.2-ci-cirr
$PY tools/collect_results.py --results-dir $R --out $R/summary.md > /dev/null 2>&1
echo "===== queue 12 done $(date)"
