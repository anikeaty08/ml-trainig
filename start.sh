#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
  echo "Missing .venv. Run ./install.sh first."
  exit 1
fi

source "$VENV_DIR/bin/activate"
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)}"
python "$ROOT_DIR/scripts/start_server.py"
