#!/usr/bin/env bash
# One-time (or idempotent) environment setup for AquaVision.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
VENV="$ROOT/.venv"

say() { printf '\n\033[1;36m%s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m%s\033[0m\n' "$*"; }

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python 3 is required but '$PYTHON' was not found." >&2
  exit 1
fi

say "AquaVision setup"
"$PYTHON" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"Python 3.10+ is required (found {sys.version.split()[0]})")
print(f"  Python {sys.version.split()[0]}")
PY

if [ ! -x "$VENV/bin/python" ]; then
  say "Creating virtual environment at .venv"
  "$PYTHON" -m venv "$VENV"
else
  echo "  Reusing virtual environment at .venv"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel

GPU=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  GPU=1
fi

if [ "$GPU" -eq 1 ]; then
  say "NVIDIA GPU detected"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  if ! python - <<'PY'
import sys
try:
    import torch
    ok = torch.cuda.is_available()
except Exception:
    ok = False
sys.exit(0 if ok else 1)
PY
  then
    say "Installing CUDA-enabled PyTorch (cu130 wheels)"
    python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision
  else
    echo "  CUDA-enabled PyTorch is already installed"
  fi
else
  warn "No NVIDIA GPU detected; installing CPU PyTorch (BioCLIP is slow on CPU)."
  if ! python - <<'PY'
try:
    import torch, torchvision  # noqa: F401
except Exception:
    raise SystemExit(1)
PY
  then
    python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
  fi
fi

say "Installing application dependencies"
python -m pip install -r requirements.txt

say "Fetching and preparing models (about 1.7 GiB on first run)"
python scripts/download_models.py

say "Checking installed runtime"
python - <<'PY'
import cv2, torch, torchvision
print(f"  torch       {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  gpu         {torch.cuda.get_device_name(0)}")
print(f"  torchvision {torchvision.__version__}")
print(f"  opencv      {cv2.__version__}")
PY

say "Setup complete"
cat <<'EOF'
Start the dashboard:
  ./run.sh

WSL camera note:
  The Windows camera must be attached to WSL before the feed will work:
    usbipd list
    usbipd attach --wsl --busid <BUSID>
  Run those commands in PowerShell as Administrator.

Open http://127.0.0.1:8000 in your browser.
EOF
