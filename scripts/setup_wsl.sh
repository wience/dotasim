#!/bin/bash
# One-time Python setup inside WSL2: uv, Python 3.12, the project venv at ~/dotasim-venv.
# Usage (from Windows): wsl.exe -d Ubuntu -- bash /mnt/c/Users/ADMIN/Documents/GitHub/dotasim/scripts/setup_wsl.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi
uv --version
uv python install 3.12 >/dev/null
[ -x "$HOME/dotasim-venv/bin/python" ] || uv venv -q "$HOME/dotasim-venv" -p 3.12
uv pip install -q -p "$HOME/dotasim-venv/bin/python" -e "$REPO[dev]"
"$HOME/dotasim-venv/bin/python" -c "import google.protobuf, psutil, grpc_tools, pytest; print('venv ok', google.protobuf.__version__)"
