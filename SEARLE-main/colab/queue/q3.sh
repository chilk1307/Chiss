#!/bin/bash
# Queue 3: D9 evaluation of the 3 screening variants (E3.8a) under 9 degradations + data-scale eval of T1 (E1.14),
# then the automatic choice of the robust variant (rule of nhiem_vu_train_Colab.md, D9).
source $DRIVE/queue/common.sh
L=$R/E3.8a-train
for P in phi_s0_clean phi_s0_degaug phi_s0_noise; do
  while ! [ -f $L/$P.done ] && { pgrep -f "exp-name $P" > /dev/null || ! [ -f $L/$P.log ]; }; do sleep 120; done
done
echo "===== queue 3 start $(date)"

ev() {  # ev <phi exp> <out code> <flags...>
  local P=$1 C=$2; shift 2
  run $R/$C $V --eval-type phi --exp-name $P --phi-checkpoint-name phi_last.pt $CACHE "$@" --output-dir $R/$C
}
for P in phi_s0_clean phi_s0_degaug phi_s0_noise; do
  [ -f $L/$P.done ] || { echo "[$(date +%H:%M)] SKIP $P (training failed)"; continue; }
  for D in none gauss_noise:2 gauss_blur:2 jpeg:2 low_light:2 salt_pepper:2 motion_blur:2 low_res:2 color_cast:2; do
    ev $P E3.8a-$P-${D/:/}-cirr $CIRR --degrade $D
  done
done
ev phi_scale_s0 E1.14-phi_scale_s0-cirr  $CIRR
ev phi_scale_s0 E1.14-phi_scale_s0-circo $CIRCO

# Choice: best mean R@1 over the 8 degraded versions, clean R@1 at most 0.5 below phi_s0_clean
$PY - <<'EOF' | tee $R/E3.8a-choice.txt
import json, os
R = os.environ['DRIVE'] + '/results'
degs = ['gauss_noise2', 'gauss_blur2', 'jpeg2', 'low_light2', 'salt_pepper2', 'motion_blur2', 'low_res2', 'color_cast2']
def r1(p, d):
    f = f'{R}/E3.8a-{p}-{d}-cirr/metrics.json'
    if not os.path.exists(f): return None
    m = json.load(open(f)); m = m.get('metrics', m); return m['cirr_recall_at1']
rows = {}
for p in ['phi_s0_clean', 'phi_s0_degaug', 'phi_s0_noise']:
    clean = r1(p, 'none'); vals = [r1(p, d) for d in degs]
    if clean is None or None in vals: continue
    rows[p] = (clean, sum(vals) / len(vals), sum(vals[:4]) / 4, sum(vals[4:]) / 4)
    print(f"{p:15s} clean {clean:.2f} | degraded mean {rows[p][1]:.2f} (seen {rows[p][2]:.2f}, unseen {rows[p][3]:.2f})")
base = rows.get('phi_s0_clean', (0,))[0]
ok = {p: v for p, v in rows.items() if p != 'phi_s0_clean' and v[0] >= base - 0.5}
choice = max(ok, key=lambda p: ok[p][1]) if ok else 'phi_s0_degaug'
if ok and ok[choice][1] <= rows['phi_s0_clean'][1]: print('NOTE: no variant beats phi_s0_clean on degraded images')
print('CHOICE', choice)
EOF
$PY tools/collect_results.py --results-dir $R --out $R/summary.md > /dev/null 2>&1
echo "===== queue 3 done $(date)"
