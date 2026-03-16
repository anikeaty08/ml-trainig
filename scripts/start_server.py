from __future__ import annotations

import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.request import urlopen

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.config import APP_HOST, APP_PORT

FRONTEND_BUILD_DIR = ROOT_DIR / "frontend" / "build"
APP_ORIGIN = f"http://{APP_HOST}:{APP_PORT}"


def start_backend() -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", APP_HOST, "--port", str(APP_PORT)],
        cwd=ROOT_DIR,
    )


def wait_for_backend(timeout_seconds: int = 30) -> None:
    started = time.time()
    while time.time() - started < timeout_seconds:
        try:
            with urlopen(f"{APP_ORIGIN}/api/health", timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("Backend did not become healthy in time")


if __name__ == "__main__":
    if not FRONTEND_BUILD_DIR.exists():
        raise SystemExit("Frontend build is missing. Run `npm run build` in frontend/ first.")
    backend_process = start_backend()
    try:
        wait_for_backend()
        webbrowser.open(APP_ORIGIN)
        backend_process.wait()
    finally:
        if backend_process.poll() is None:
            backend_process.terminate()
