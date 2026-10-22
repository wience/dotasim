#!/bin/bash
# Run a command inside WSL2 from the repo root with the project venv on PATH.
# Usage (from Windows): wsl.exe -d Ubuntu -- bash /mnt/c/Users/ADMIN/Documents/GitHub/dotasim/scripts/wsl.sh <command...>
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/dotasim-venv/bin:$HOME/.local/bin:$PATH"
cd "$REPO"
exec "$@"
