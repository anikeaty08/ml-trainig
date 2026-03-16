from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
FRONTEND_DIR = ROOT_DIR / "frontend"
FRONTEND_BUILD_DIR = FRONTEND_DIR / "build"

DATA_DIR = BACKEND_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
MODELS_DIR = DATA_DIR / "models"
REPORTS_DIR = DATA_DIR / "reports"
LOGS_DIR = DATA_DIR / "logs"
BUILTIN_MODELS_DIR = BACKEND_DIR / "models_builtin"

DB_PATH = BACKEND_DIR / "db.sqlite"
API_HOST = "0.0.0.0"
API_PORT = 8000
FRONTEND_PORT = 3000
SUPPORTED_FILE_EXTENSIONS = {".csv", ".tsv", ".txt"}


def ensure_directories() -> None:
    for path in (
        DATA_DIR,
        UPLOADS_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        LOGS_DIR,
        BUILTIN_MODELS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
