#!/bin/bash
# Queue 12p3 (E10): P3 set (instruct, visual sentences). The CIRCO part of the generation may still be running from
# q12 (its python process was left alive): wait for it, then the CIRR concepts with fewer phrases (16, one round,
# to keep the generation near one hour), OTI + evaluation, and at the end (after q12a) the CIs.
source /content/env.sh; source /content/drive/MyDrive/SEARLE/queue/q12env.sh
echo "===== queue 12p3 start $(date)"
while pgrep -f "P3_qwen15i_val_raw" > /dev/null; do sleep 60; done
run $R/E10.0-gen-P3-circo2 $SYS tools/generate_phrases_llm.py $P3 $SC --out-dir $GP/P3_qwen15i_val_raw
run $R/E10.0-gen-P3-cirr $SYS tools/generate_phrases_llm.py --mode instruct --model Qwen/Qwen2.5-1.5B-Instruct \
  --n-phrases 16 --batch-concepts 48 --rounds 1 $LIC $SR --out-dir $GP/P3_qwen15i_val_raw
clean_set P3_qwen15i_val
oti E10.1c-P3-circo $CIRCOV --gpt-exp-name P3_qwen15i_val
ev E10.1c-P3-circo $CIRCO
oti E10.2c-P3-cirr $CIRRV --gpt-exp-name P3_qwen15i_val
ev E10.2c-P3-cirr $CIRR
while pgrep -f "queue/q12a.sh" > /dev/null; do sleep 60; done
run $R/E10.1-ci-circo $PY tools/bootstrap_ci.py --baseline $R/E7.1a-random-lr0.02-circo-step350-ema_oti_pseudo_tokens \
  --runs $(ls -d $R/E10.1*-circo-ema350 2>/dev/null) --out $R/E10.1-ci-circo
run $R/E10.2-ci-cirr $PY tools/bootstrap_ci.py --baseline $R/E7.2a-random-lr0.02-cirr-step350-ema_oti_pseudo_tokens \
  --runs $(ls -d $R/E10.2*-cirr-ema350 2>/dev/null) --out $R/E10.2-ci-cirr
$PY tools/collect_results.py --results-dir $R --out $R/summary.md > /dev/null 2>&1
echo "===== queue 12p3 done $(date)"
