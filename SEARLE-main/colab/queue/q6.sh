#!/bin/bash
# Queue 6 (after queue 2): D10 automatic restoration E3.3 on mixed degradations (CIRR), clean-image check, best on CIRCO
source $DRIVE/queue/common.sh
while pgrep -f "queue/q2.sh" > /dev/null; do sleep 60; done
echo "===== queue 6 start $(date)"
for RS in none auto auto_iqa oracle; do
  run $R/E3.3-mixed-$RS-cirr $V $PHI $CIRR --degrade mixed --restore $RS --save-quality --output-dir $R/E3.3-mixed-$RS-cirr
done
for RS in auto auto_iqa; do
  run $R/E3.3-clean-$RS-cirr $V $PHI $CIRR --restore $RS --output-dir $R/E3.3-clean-$RS-cirr
done
BEST=$($PY - <<'EOF'
import json, os
R = os.environ['DRIVE'] + '/results'
def r1(rs):
    f = f'{R}/E3.3-mixed-{rs}-cirr/metrics.json'
    if not os.path.exists(f): return -1
    m = json.load(open(f)); m = m.get('metrics', m); return m['cirr_recall_at1']
print(max(['auto', 'auto_iqa'], key=r1))
EOF
)
echo "[$(date +%H:%M)] best automatic restoration on CIRR: $BEST"
for RS in none $BEST oracle; do
  run $R/E3.3-mixed-$RS-circo-ex $V $PHI $CIRCO --exclude-reference --degrade mixed --restore $RS --output-dir $R/E3.3-mixed-$RS-circo-ex
done
run $R/E3.3-ci-cirr $PY tools/bootstrap_ci.py --baseline $R/E3.3-mixed-none-cirr --runs $R/E3.3-mixed-auto-cirr $R/E3.3-mixed-auto_iqa-cirr $R/E3.3-mixed-oracle-cirr --out $R/E3.3-ci-cirr
$PY tools/collect_results.py --results-dir $R --out $R/summary.md > /dev/null 2>&1
echo "===== queue 6 done $(date)"
