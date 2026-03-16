#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

ensure_python() {
  if command_exists python3; then
    PYTHON_BIN="$(command -v python3)"
  elif command_exists python; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "Python 3.10+ is required."
    exit 1
  fi
}

ensure_node() {
  if ! command_exists node; then
    echo "Node.js 18+ is required."
    exit 1
  fi
  if ! command_exists npm; then
    echo "npm is required."
    exit 1
  fi
}

open_browser() {
  if command_exists open; then
    open "http://localhost:3000" >/dev/null 2>&1 || true
  elif command_exists xdg-open; then
    xdg-open "http://localhost:3000" >/dev/null 2>&1 || true
  fi
}

ensure_python
ensure_node

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$ROOT_DIR/requirements.txt"
python "$ROOT_DIR/scripts/download_models.py"
python "$ROOT_DIR/scripts/init_db.py"

pushd "$ROOT_DIR/frontend" >/dev/null
npm install
npm run build
popd >/dev/null

python "$ROOT_DIR/scripts/start_server.py" &
SERVER_PID=$!
sleep 5
open_browser
echo "ML Pipeline Agent ready at http://localhost:3000"
wait $SERVER_PID
