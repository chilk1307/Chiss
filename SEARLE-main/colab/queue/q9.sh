#!/bin/bash
# Queue 9 (25/10, plan 07): OTI warm start. H17 (E7.0 concepts, E6.2-Plambda0), H15 (E7.1a-d on CIRCO val),
# D15 (E7.0 + E7.2 on CIRR val). Every OTI run saves raw and EMA tokens at steps 0,10,25,50,100,200,350 and each
# snapshot is evaluated; CIs against phi_B32 at the end.
source $DRIVE/queue/common.sh
echo "===== queue 9 start $(date)"
if [ ! -s data/GPT_phrases/GPTNeo27B/concept_to_phrases.pkl ]; then
  echo "FAIL  GPT phrases missing: data/GPT_phrases/GPTNeo27B/concept_to_phrases.pkl"; exit 1
fi
CIRCOV="--dataset circo --dataset-path $DATA/CIRCO_local --split val"
CIRRV="--dataset cirr --dataset-path $DATA/CIRR --split val"
SNAP=0,10,25,50,100,200,350
WARM="--init-phi-exp-name phi_B32 --init-phi-checkpoint phi_last.pt"
oti() { local E=$1; shift; run $R/$E-oti $PY src/oti_inversion.py --exp-name $E --clip-model-name ViT-B/32 "$@"; }
snaps() {  # snaps <exp> <dataset flags...>: evaluate every snapshot, raw and EMA
  local E=$1; shift
  for S in 0 10 25 50 100 200 350; do
    for TK in oti_pseudo_tokens ema_oti_pseudo_tokens; do
      run $R/$E-step$S-$TK $V --eval-type oti --exp-name $E --token-file ${TK}_step$S.pt "$@" $CACHE --output-dir $R/$E-step$S-$TK
    done
  done
}

# --- H17: concepts of the CIRCO val reference images (E7.0), OTI without GPT phrases (E6.2-Plambda0)
run $R/E7.0-concepts-circo $PY src/image_concepts_association.py --clip-model-name ViT-B/32 $CIRCOV --dataset-mode relative
oti E6.2-Plambda0-circo $CIRCOV --lambda_gpt 0
run $R/E6.2-Plambda0-circo $V --eval-type oti --exp-name E6.2-Plambda0-circo $CIRCO $CACHE --output-dir $R/E6.2-Plambda0-circo

# --- H15: warm start on CIRCO val (E7.1a = SEARLE-OTI of the paper)
oti E7.1a-random-lr0.02-circo  $CIRCOV --snapshot-steps $SNAP --init random --learning-rate 0.02
oti E7.1b-phi-lr0.02-circo     $CIRCOV --snapshot-steps $SNAP --init phi $WARM --learning-rate 0.02
oti E7.1c-phi-lr0.005-circo    $CIRCOV --snapshot-steps $SNAP --init phi $WARM --learning-rate 0.005
oti E7.1d-searle-lr0.02-circo  $CIRCOV --snapshot-steps $SNAP --init searle --learning-rate 0.02
oti E7.1d-searle-lr0.005-circo $CIRCOV --snapshot-steps $SNAP --init searle --learning-rate 0.005
for E in E7.1a-random-lr0.02-circo E7.1b-phi-lr0.02-circo E7.1c-phi-lr0.005-circo E7.1d-searle-lr0.02-circo E7.1d-searle-lr0.005-circo; do
  snaps $E $CIRCO
done

# --- D15: CIRR val (E7.0 + E7.2): random and the two phi initialisations
run $R/E7.0-concepts-cirr $PY src/image_concepts_association.py --clip-model-name ViT-B/32 $CIRRV --dataset-mode relative
oti E7.2a-random-lr0.02-cirr $CIRRV --snapshot-steps $SNAP --init random --learning-rate 0.02
oti E7.2b-phi-lr0.02-cirr    $CIRRV --snapshot-steps $SNAP --init phi $WARM --learning-rate 0.02
oti E7.2c-phi-lr0.005-cirr   $CIRRV --snapshot-steps $SNAP --init phi $WARM --learning-rate 0.005
for E in E7.2a-random-lr0.02-cirr E7.2b-phi-lr0.02-cirr E7.2c-phi-lr0.005-cirr; do
  snaps $E $CIRR
done

# --- CIs against phi_B32 (CIRCO: reference image in the gallery, as E1.7e) at 50 and 350 steps, EMA tokens
run $R/E7.1-ci-circo $PY tools/bootstrap_ci.py --baseline $R/E1.7e-phiB32-circo \
  --runs $(ls -d $R/E7.1*-circo-step50-ema_oti_pseudo_tokens $R/E7.1*-circo-step350-ema_oti_pseudo_tokens 2>/dev/null) --out $R/E7.1-ci-circo
run $R/E7.2-ci-cirr $PY tools/bootstrap_ci.py --baseline $R/E1.7d-phiB32-cirr \
  --runs $(ls -d $R/E7.2*-cirr-step50-ema_oti_pseudo_tokens $R/E7.2*-cirr-step350-ema_oti_pseudo_tokens 2>/dev/null) --out $R/E7.2-ci-cirr
$PY tools/collect_results.py --results-dir $R --out $R/summary.md > /dev/null 2>&1
echo "===== queue 9 done $(date)"
