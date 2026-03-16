from __future__ import annotations

from pathlib import Path
from typing import Any
import zipfile

import pandas as pd
from pandas.api.types import is_numeric_dtype

from .audio_pipeline import AUDIO_EXTENSIONS, detect_audio_archive
from .image_pipeline import detect_image_archive


LIKELY_TARGET_NAMES = ("target", "label", "class", "y", "output", "prediction", "response")


def _read_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _datetime_candidates(df: pd.DataFrame) -> list[str]:
    candidates: list[str] = []
    for column in df.columns:
        series = df[column]
        if is_numeric_dtype(series):
            continue
        sample = series.dropna().astype(str).head(250)
        if sample.empty:
            continue
        parsed = pd.to_datetime(sample, errors="coerce", utc=False, format="mixed")
        success_ratio = parsed.notna().mean()
        if success_ratio >= 0.8 or any(token in column.lower() for token in ("date", "time", "timestamp")):
            candidates.append(column)
    return candidates


def _text_candidates(df: pd.DataFrame, target_column: str | None) -> list[str]:
    candidates: list[str] = []
    for column in df.columns:
        if column == target_column:
            continue
        series = df[column]
        if is_numeric_dtype(series):
            continue
        sample = series.dropna().astype(str)
        if sample.empty:
            continue
        average_length = sample.str.len().mean()
        unique_ratio = sample.nunique(dropna=True) / max(len(sample), 1)
        if average_length >= 24 and unique_ratio >= 0.25:
            candidates.append(column)
    return candidates


def _looks_like_identifier(series: pd.Series, name: str) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    unique_ratio = non_null.nunique(dropna=True) / max(len(non_null), 1)
    return "id" in name.lower() or unique_ratio > 0.98


def infer_target_column(df: pd.DataFrame) -> str | None:
    columns = list(df.columns)
    preferred = [
        column
        for column in columns
        if column.lower() in LIKELY_TARGET_NAMES and not _looks_like_identifier(df[column], column)
    ]
    if preferred:
        return preferred[0]

    candidates = [
        column
        for index, column in enumerate(columns)
        if index == len(columns) - 1 or not _looks_like_identifier(df[column], column)
    ]
    if not candidates:
        return columns[-1] if columns else None

    for column in reversed(candidates):
        unique_count = df[column].nunique(dropna=True)
        if unique_count <= max(20, int(len(df) * 0.25)) or is_numeric_dtype(df[column]):
            return column
    return candidates[-1]


def infer_problem_type(df: pd.DataFrame, target_column: str | None) -> str:
    if not target_column:
        return "unknown"
    target = df[target_column]
    unique_count = target.nunique(dropna=True)
    if not is_numeric_dtype(target):
        return "classification"
    if unique_count <= max(10, int(len(target) * 0.05)):
        return "classification"
    return "regression"


def _archive_dataset_type(path: Path) -> str:
    image_count = 0
    audio_count = 0
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            suffix = Path(name).suffix.lower()
            if suffix in AUDIO_EXTENSIONS:
                audio_count += 1
            if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                image_count += 1
    if audio_count and audio_count >= image_count:
        return "audio"
    return "image"


def detect_dataset(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".zip":
        archive_type = _archive_dataset_type(path)
        if archive_type == "audio":
            return detect_audio_archive(path)
        return detect_image_archive(path)

    if path.suffix.lower() not in {".csv", ".tsv", ".txt"}:
        return {
            "dataset_type": "unsupported",
            "problem_type": "unknown",
            "target_column": None,
            "preview_summary": {"filename": path.name, "reason": "Unsupported file format"},
        }

    dataframe = _read_dataframe(path)
    target_column = infer_target_column(dataframe)
    problem_type = infer_problem_type(dataframe, target_column)
    datetime_columns = _datetime_candidates(dataframe)
    text_columns = _text_candidates(dataframe, target_column)

    dataset_type = "tabular"
    dataset_context: dict[str, Any] = {
        "datetime_columns": datetime_columns,
        "text_columns": text_columns,
    }
    if target_column and problem_type == "regression" and datetime_columns:
        dataset_type = "timeseries"
        dataset_context["time_column"] = datetime_columns[0]
    elif target_column and text_columns:
        dataset_type = "text"
        dataset_context["text_column"] = text_columns[0]

    return {
        "dataset_type": dataset_type,
        "problem_type": problem_type,
        "target_column": target_column,
        "dataframe": dataframe,
        "source_path": path,
        "dataset_context": dataset_context,
        "preview_summary": {
            "filename": path.name,
            "rows": int(dataframe.shape[0]),
            "columns": int(dataframe.shape[1]),
            "target_column": target_column,
            "problem_type": problem_type,
            "dataset_type": dataset_type,
            "time_column": dataset_context.get("time_column"),
            "text_column": dataset_context.get("text_column"),
        },
    }
