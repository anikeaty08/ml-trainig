from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import REPORTS_DIR, UPLOADS_DIR, ensure_directories


def save_upload(job_id: str, filename: str, content: bytes) -> Path:
    ensure_directories()
    job_dir = UPLOADS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / filename
    path.write_bytes(content)
    return path


def prepare_job_artifact_dirs(job_id: str) -> dict[str, Path]:
    ensure_directories()
    base = REPORTS_DIR / job_id
    paths = {
        "base": base,
        "model": base / "model",
        "code": base / "code",
        "data": base / "data",
        "reports": base / "reports",
        "logs": base / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return path


def save_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def save_csv(path: Path, dataframe: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)
    return path


def save_comparison_csv(reports_dir: Path, rows: list[dict[str, Any]]) -> Path:
    path = reports_dir / "model_comparison.csv"
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def create_bundle_archive(base_dir: Path) -> Path:
    archive_base = base_dir.parent / f"{base_dir.name}_bundle"
    archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=base_dir)
    return Path(archive_path)
