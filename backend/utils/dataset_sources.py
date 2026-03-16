from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import httpx

from ..config import UPLOADS_DIR, ensure_directories


def _download_dir(job_id: str) -> Path:
    ensure_directories()
    path = UPLOADS_DIR / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _filename_from_headers(url: str, response: httpx.Response) -> str:
    disposition = response.headers.get("content-disposition", "")
    match = re.search(r'filename="?([^";]+)"?', disposition)
    if match:
        return match.group(1)
    parsed = urlparse(url)
    name = Path(parsed.path).name
    return name or "dataset.bin"


def _download_http_source(job_id: str, url: str) -> Path:
    download_dir = _download_dir(job_id)
    with httpx.Client(follow_redirects=True, timeout=120.0) as client:
        response = client.get(url)
        response.raise_for_status()
        filename = _filename_from_headers(url, response)
        destination = download_dir / filename
        destination.write_bytes(response.content)
    return destination


def _resolve_kaggle_reference(url: str) -> tuple[str, str]:
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    if "datasets" in parts:
        index = parts.index("datasets")
        if len(parts) > index + 2:
            return "dataset", f"{parts[index + 1]}/{parts[index + 2]}"
    if "competitions" in parts:
        index = parts.index("competitions")
        if len(parts) > index + 1:
            return "competition", parts[index + 1]
    raise ValueError("Unsupported Kaggle URL. Use a dataset or competition URL.")


def _download_kaggle_source(job_id: str, url: str) -> Path:
    kaggle_type, reference = _resolve_kaggle_reference(url)
    if shutil.which("kaggle") is None:
        raise ValueError("Kaggle CLI is not installed or not on PATH. Install it and configure kaggle.json first.")

    download_dir = _download_dir(job_id)
    if kaggle_type == "dataset":
        command = ["kaggle", "datasets", "download", "-d", reference, "-p", str(download_dir), "--force"]
    else:
        command = ["kaggle", "competitions", "download", "-c", reference, "-p", str(download_dir), "--force"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ValueError(result.stderr.strip() or result.stdout.strip() or "Kaggle download failed.")

    candidates = sorted(download_dir.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True)
    if not candidates:
        raise ValueError("Kaggle download did not produce a local file.")
    return candidates[0]


def download_dataset_source(job_id: str, url: str) -> Path:
    normalized = url.strip()
    if not normalized:
        raise ValueError("Dataset URL is required.")
    if "kaggle.com/" in normalized:
        return _download_kaggle_source(job_id, normalized)
    return _download_http_source(job_id, normalized)
