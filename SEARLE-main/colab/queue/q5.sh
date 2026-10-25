#!/bin/bash
# Queue 5 (after queue 2): CIRCO robustness without the reference image (E3.1-*-circo-ex), then the attribute
# re-ranking choices of plan 02 on top of E2.3: E2.6 family combination, E2.3 top-K, E2.4 masks, E2.5 negation,
# E2.7 final configuration on CIRCO (no re-tuning). Choices are written to $R/E2-choices.json.
source $DRIVE/queue/common.sh
while pgrep -f "queue/q2.sh" > /dev/null; do sleep 60; done
echo "===== queue 5 start $(date)"

for D in gauss_noise salt_pepper gauss_blur motion_blur jpeg low_light low_res color_cast; do
  for L in 1 2; do
    run $R/E3.1-$D$L-circo-ex $V $PHI $CIRCO --exclude-reference --degrade $D:$L --output-dir $R/E3.1-$D$L-circo-ex
  done
done

choose() {  # choose <stage>: prints the shell assignments of the current choice
$PY - "$1" <<'EOF'
import json, os, sys, glob
R = os.environ['DRIVE'] + '/results'
stage = sys.argv[1]
KEY = {'cirr': 'cirr_recall_at1', 'fiq': 'average_fiq_recall_at10'}
BASE = {'cirr': 'E1.7d-phiB32-cirr', 'fiq': 'E1.8-phiB32-fiq-ex'}
def m(run, ds):
    f = f'{R}/{run}/metrics.json'
    if not os.path.exists(f): return None
    d = json.load(open(f)); d = d.get('metrics', d); return d.get(KEY[ds])
path = f'{R}/E2-choices.json'
ch = json.load(open(path)) if os.path.exists(path) else {}
for ds in ['cirr', 'fiq']:
    base = m(BASE[ds], ds); c = ch.setdefault(ds, {'baseline': base})
    if stage == 'families':
        fam = {}
        for F in ['color', 'brightness', 'colorfulness', 'pattern', 'zoom']:
            vals = {W: m(f'E2.3-{F}-w{W}-{ds}', ds) for W in ['0.05', '0.1', '0.2', '0.3', '0.5', '1.0', '2.0']}
            vals = {W: v for W, v in vals.items() if v is not None}
            if vals:
                W = max(vals, key=vals.get); fam[F] = {'weight': W, 'value': vals[W], 'gain': vals[W] - base}
        c['per_family'] = fam
        pos = [F for F, v in fam.items() if v['gain'] > 0]
        c['families'] = ','.join(pos) if pos else (max(fam, key=lambda F: fam[F]['gain']) if fam else 'color')
    elif stage == 'weight':
        vals = {W: m(f'E2.6-comb-w{W}-{ds}', ds) for W in ['0.1', '0.3', '0.5', '1.0', '2.0']}
        vals = {W: v for W, v in vals.items() if v is not None}
        c['weight'] = max(vals, key=vals.get) if vals else '0.1'; c['comb_value'] = vals.get(c['weight'])
    elif stage == 'final':
        opts = {'topk': {'50': f'E2.3-topk50-{ds}', '100': f'E2.6-comb-w{c["weight"]}-{ds}'},
                'mask': {m_: f'E2.4-mask-{m_}-{ds}' for m_ in ['none', 'otsu', 'grabcut', 'saliency']},
                'negation': {'off': f'E2.5-neg-off-{ds}'}}
        auto = m(f'E2.6-comb-w{c["weight"]}-{ds}', ds)
        opts['mask']['auto'] = f'E2.6-comb-w{c["weight"]}-{ds}'; opts['negation']['on'] = f'E2.6-comb-w{c["weight"]}-{ds}'
        for k, o in opts.items():
            vals = {v: m(r, ds) for v, r in o.items()}; vals = {v: x for v, x in vals.items() if x is not None}
            c[k] = max(vals, key=vals.get) if vals else {'topk': '100', 'mask': 'auto', 'negation': 'on'}[k]
            c[k + '_values'] = vals
json.dump(ch, open(path, 'w'), indent=2)
for ds in ['cirr', 'fiq']:
    c = ch[ds]
    print(f"FAM_{ds}={c.get('families', 'color')}; W_{ds}={c.get('weight', '0.1')}; "
          f"K_{ds}={c.get('topk', '100')}; MASK_{ds}={c.get('mask', 'auto')}; NEG_{ds}={c.get('negation', 'on')}")
EOF
}

# E2.3 extended: weights <= 0.3 barely move the ranking (+0.05 CIRR R@1 at best), so larger weights too
for FAM in color brightness colorfulness pattern zoom; do
  for W in 0.5 1.0 2.0; do
    run $R/E2.3-$FAM-w$W-cirr $V $PHI $CIRR --attr-rerank --attr-families $FAM --attr-weight $W --output-dir $R/E2.3-$FAM-w$W-cirr
    run $R/E2.3-$FAM-w$W-fiq  $V $PHI $FIQ --exclude-reference --attr-rerank --attr-families $FAM --attr-weight $W --output-dir $R/E2.3-$FAM-w$W-fiq
  done
done
eval "$(choose families)"; echo "[$(date +%H:%M)] families cirr=$FAM_cirr fiq=$FAM_fiq"
for W in 0.1 0.3 0.5 1.0 2.0; do
  run $R/E2.6-comb-w$W-cirr $V $PHI $CIRR --attr-rerank --attr-families $FAM_cirr --attr-weight $W --output-dir $R/E2.6-comb-w$W-cirr
  run $R/E2.6-comb-w$W-fiq  $V $PHI $FIQ --exclude-reference --attr-rerank --attr-families $FAM_fiq --attr-weight $W --output-dir $R/E2.6-comb-w$W-fiq
done
eval "$(choose weight)"; echo "[$(date +%H:%M)] weight cirr=$W_cirr fiq=$W_fiq"
for DS in cirr fiq; do
  if [ $DS = cirr ]; then A="$CIRR"; F=$FAM_cirr; W=$W_cirr; else A="$FIQ --exclude-reference"; F=$FAM_fiq; W=$W_fiq; fi
  C="$V $PHI $A --attr-rerank --attr-families $F --attr-weight $W"
  run $R/E2.3-topk50-$DS $C --attr-topk 50 --output-dir $R/E2.3-topk50-$DS
  for M in none otsu grabcut saliency; do run $R/E2.4-mask-$M-$DS $C --attr-mask $M --output-dir $R/E2.4-mask-$M-$DS; done
  run $R/E2.5-neg-off-$DS $C --attr-negation off --output-dir $R/E2.5-neg-off-$DS
done
eval "$(choose final)"; echo "[$(date +%H:%M)] final cirr: $FAM_cirr w$W_cirr k$K_cirr $MASK_cirr neg-$NEG_cirr | fiq: $FAM_fiq w$W_fiq k$K_fiq $MASK_fiq neg-$NEG_fiq"
FINAL_CIRR="--attr-rerank --attr-families $FAM_cirr --attr-weight $W_cirr --attr-topk $K_cirr --attr-mask $MASK_cirr --attr-negation $NEG_cirr"
echo "$FINAL_CIRR" > $R/E2-final-flags.txt
run $R/E2.7-final-circo    $V $PHI $CIRCO $FINAL_CIRR --output-dir $R/E2.7-final-circo
run $R/E2.7-final-circo-ex $V $PHI $CIRCO --exclude-reference $FINAL_CIRR --output-dir $R/E2.7-final-circo-ex
run $R/E2.7-final-cirr     $V $PHI $CIRR $FINAL_CIRR --output-dir $R/E2.7-final-cirr
run $R/E2.7-final-fiq-ex   $V $PHI $FIQ --exclude-reference $FINAL_CIRR --output-dir $R/E2.7-final-fiq-ex
run $R/E2.7-ci-cirr  $PY tools/bootstrap_ci.py --baseline $R/E1.7d-phiB32-cirr --runs $R/E2.7-final-cirr $R/E2.6-comb-*-cirr --out $R/E2.7-ci-cirr
run $R/E2.7-ci-fiq   $PY tools/bootstrap_ci.py --baseline $R/E1.8-phiB32-fiq-ex --runs $R/E2.7-final-fiq-ex $R/E2.6-comb-*-fiq --out $R/E2.7-ci-fiq
run $R/E2.7-ci-circo $PY tools/bootstrap_ci.py --baseline $R/E1.8-phiB32-circo-ex --runs $R/E2.7-final-circo-ex --out $R/E2.7-ci-circo
$PY tools/collect_results.py --results-dir $R --out $R/summary.md > /dev/null 2>&1
echo "===== queue 5 done $(date)"
