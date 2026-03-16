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


def _extract_kagglehub_reference(value: str) -> str | None:
    match = re.search(r'kagglehub\.dataset_download\(\s*["\']([^"\']+)["\']\s*\)', value)
    if match:
        return match.group(1).strip()
    match = re.fullmatch(r"\s*([a-z0-9_.-]+/[a-z0-9_.-]+)\s*", value, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _pick_downloaded_dataset(path: Path) -> Path:
    if path.is_file():
        return path
    supported = {".csv", ".tsv", ".txt", ".zip"}
    candidates = sorted(
        [item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in supported],
        key=lambda item: (item.suffix.lower() != ".csv", item.suffix.lower() != ".tsv", item.name.lower()),
    )
    if not candidates:
        raise ValueError("The dataset source was downloaded, but no supported CSV/TSV/TXT/ZIP files were found.")
    return candidates[0]


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
    return _pick_downloaded_dataset(candidates[0])


def _download_kagglehub_source(job_id: str, reference: str) -> Path:
    download_dir = _download_dir(job_id)

    try:
        import kagglehub  # type: ignore

        dataset_path = Path(kagglehub.dataset_download(reference))
        return _pick_downloaded_dataset(dataset_path)
    except Exception:
        pass

    if shutil.which("kaggle") is None:
        raise ValueError(
            "Could not use kagglehub locally and Kaggle CLI is not installed. "
            "Install kagglehub or Kaggle CLI, then try again."
        )

    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", reference, "-p", str(download_dir), "--force"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(result.stderr.strip() or result.stdout.strip() or "Kaggle download failed.")

    candidates = sorted(download_dir.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True)
    if not candidates:
        raise ValueError("Kaggle dataset download did not produce a local file.")
    return _pick_downloaded_dataset(candidates[0])


def download_dataset_source(job_id: str, url: str) -> Path:
    normalized = url.strip()
    if not normalized:
        raise ValueError("Dataset URL is required.")
    kagglehub_reference = _extract_kagglehub_reference(normalized)
    if kagglehub_reference:
        return _download_kagglehub_source(job_id, kagglehub_reference)
    if "kaggle.com/" in normalized:
        return _download_kaggle_source(job_id, normalized)
    return _download_http_source(job_id, normalized)
