from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen


ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_BUILD_DIR = ROOT_DIR / "frontend" / "build"


def start_backend() -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=ROOT_DIR,
    )


def start_frontend_server() -> ThreadingHTTPServer:
    os.chdir(FRONTEND_BUILD_DIR)
    server = ThreadingHTTPServer(("0.0.0.0", 3000), SimpleHTTPRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def wait_for_backend(timeout_seconds: int = 30) -> None:
    started = time.time()
    while time.time() - started < timeout_seconds:
        try:
            with urlopen("http://127.0.0.1:8000/api/health", timeout=2) as response:
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
        frontend_server = start_frontend_server()
        webbrowser.open("http://localhost:3000")
        backend_process.wait()
    finally:
        if "frontend_server" in locals():
            frontend_server.shutdown()
