# Shared definitions of queues 12a / 12p3 (source it). See q12.sh for the design of E10.
source $DRIVE/queue/common.sh
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
oti() { local E=$1; shift; run $R/$E-oti $PY src/oti_inversion.py --exp-name $E --clip-model-name ViT-B/32 \
  --init random --learning-rate 0.02 --snapshot-steps 50,350 "$@"; }
ev() { local E=$1; shift; run $R/$E-ema350 $V --eval-type oti --exp-name $E --token-file ema_oti_pseudo_tokens_step350.pt "$@" $CACHE --output-dir $R/$E-ema350; }
clean_set() { local N=$1
  run $R/E10.0-clean-$N $PY tools/clean_phrases.py --phrases $GP/${N}_raw/concept_to_phrases.pkl --out-exp $N --out-root $GP
  run $R/E10.0-stats-$N $PY tools/phrases_stats.py --phrases $GP/$N/concept_to_phrases.pkl \
    --similar-concepts $SIM/cirr/val/ViT-B32_top_250_ensembling_image_texts.csv --top-k-phi 0 --out $R/E10.0-stats-$N; }
