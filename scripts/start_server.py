from __future__ import annotations

import os
import subprocess
import sys
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


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


if __name__ == "__main__":
    backend_process = start_backend()
    frontend_server = start_frontend_server()
    try:
        webbrowser.open("http://localhost:3000")
        backend_process.wait()
    finally:
        frontend_server.shutdown()
