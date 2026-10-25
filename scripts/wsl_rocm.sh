#!/bin/bash
# Like wsl.sh but with the ROCm PyTorch venv first on PATH (GPU learner). Same usage.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/dotasim-rocm-venv/bin:$HOME/.local/bin:$PATH"
cd "$REPO"
export PYTHONPATH="$REPO"   # the ROCm venv has no editable install of the package
exec "$@"
