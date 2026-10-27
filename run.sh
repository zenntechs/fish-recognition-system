#!/usr/bin/env bash
# Start the AquaVision dashboard.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [ ! -x "$ROOT/.venv/bin/python" ]; then
  echo "The virtual environment is missing. Run ./setup.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

export HF_HOME="${HF_HOME:-$ROOT/models/hf_cache}"
if [ -d "$HF_HOME" ]; then
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
fi
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

if [ ! -f "$ROOT/models/face_detection_yunet_2023mar.onnx" ] \
  || [ ! -f "$ROOT/models/fish_detector.pt" ] \
  || [ ! -f "$ROOT/models/convnext_tiny_imagenet.pth" ] \
  || [ ! -d "$ROOT/models/hf_cache/hub/models--imageomics--bioclip-2" ]; then
  echo "Model assets are missing. Run: .venv/bin/python scripts/download_models.py" >&2
  exit 1
fi

if ! compgen -G "/dev/video*" >/dev/null; then
  printf '\033[1;33mNo /dev/video* device found.\033[0m\n'
  echo "The dashboard will open, but the camera needs to be attached in WSL:"
  echo "  PowerShell (Admin): usbipd list"
  echo "  PowerShell (Admin): usbipd attach --wsl --busid <BUSID>"
  echo
fi

exec python -m app "$@"
