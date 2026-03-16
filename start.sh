#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
  echo "Missing .venv. Run ./install.sh first."
  exit 1
fi

source "$VENV_DIR/bin/activate"
python "$ROOT_DIR/scripts/start_server.py"
