#!/usr/bin/env bash
# Build the Python 3.8 environment of the team on a fresh Colab runtime and write /content/env.sh.
#
#   bash /content/XLA_SEARLE/SEARLE-main/colab/setup_env.sh <RUNNER>      # RUNNER = Chi, Hoa or Doanh
#
# Same recipe as outputs/scripts/run_eval2.sh: uv creates a Python 3.8 venv from outputs/pip_freeze.txt
# (torch 1.11.0+cu113, CLIP d05afc43, ...), then colab/requirements-xla.txt adds the image-processing libraries.
# Running it again in the same session only rewrites /content/env.sh.
set -euo pipefail

RUNNER="${1:-${RUNNER:-unknown}}"
DRIVE=/content/drive/MyDrive/SEARLE
DATA=/content/data
REPO=/content/XLA_SEARLE
SRC=$REPO/SEARLE-main
VENV=/content/venv
PY=$VENV/bin/python

FREEZE=$REPO/outputs/pip_freeze.txt
[ -f "$FREEZE" ] || FREEZE=$DRIVE/outputs/pip_freeze.txt
[ -f "$FREEZE" ] || { echo "pip_freeze.txt not found in the repo nor on Drive"; exit 1; }

if [ -x "$PY" ] && "$PY" -c "import torch, clip, cv2, skimage, scipy" 2>/dev/null; then
  echo "venv already ready: $VENV"
else
  echo "Creating the Python 3.8 venv (a few minutes)..."
  python3 -m pip install -q uv
  python3 -m uv venv --python 3.8 "$VENV"
  python3 -m uv pip install -q --python "$PY" --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu113 -r "$FREEZE"
  python3 -m uv pip install -q --python "$PY" -r "$SRC/colab/requirements-xla.txt"
fi

cat > /content/env.sh <<EOF
# Written by colab/setup_env.sh. Usage in a %%bash cell: source /content/env.sh
export DRIVE=$DRIVE
export DATA=$DATA
export REPO=$REPO
export SRC=$SRC
export PY=$PY
export RUNNER=$RUNNER
export PYTHONUNBUFFERED=1
EOF

mkdir -p "$DATA"
[ -d "$DRIVE" ] && mkdir -p "$DRIVE/results" "$DRIVE/cache"

"$PY" - <<'EOF'
import sys, torch, torchvision, clip, numpy, cv2, skimage, scipy
print("python", sys.version.split()[0], "| torch", torch.__version__, "| torchvision", torchvision.__version__,
      "| numpy", numpy.__version__, "| opencv", cv2.__version__, "| scikit-image", skimage.__version__,
      "| scipy", scipy.__version__)
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
EOF
echo "Wrote /content/env.sh (RUNNER=$RUNNER)"
