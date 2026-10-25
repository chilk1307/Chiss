# Shared definitions for the run queues (source it; never run it directly).
source /content/env.sh
cd $SRC
R=$DRIVE/results
CACHE="--cache-dir $DRIVE/cache_A100"
PHI="--eval-type phi --exp-name phi_B32 --phi-checkpoint-name phi_last.pt $CACHE"
CIRR="--dataset cirr --dataset-path $DATA/CIRR"
CIRCO="--dataset circo --dataset-path $DATA/CIRCO_local"
FIQ="--dataset fashioniq --dataset-path $DATA/FashionIQ"
V="$PY src/validate.py"

# run <out dir> <command...>: skip when <out>/.done exists, log to <out>/stdout.log
run() {
  local out=$1; shift
  if [ -f "$out/.done" ]; then echo "[$(date +%H:%M)] SKIP $(basename $out)"; return 0; fi
  mkdir -p "$out"
  echo "[$(date +%H:%M)] START $(basename $out)"
  if "$@" > "$out/stdout.log" 2>&1; then
    touch "$out/.done"; echo "[$(date +%H:%M)] OK    $(basename $out)"
  else
    echo "[$(date +%H:%M)] FAIL  $(basename $out): $(tail -n 2 $out/stdout.log | tr '\n' ' ' | cut -c1-200)"
  fi
}
