from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype


LIKELY_TARGET_NAMES = ("target", "label", "class", "y", "output", "prediction", "response")


def _read_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _looks_like_identifier(series: pd.Series, name: str) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    unique_ratio = non_null.nunique(dropna=True) / max(len(non_null), 1)
    return "id" in name.lower() or unique_ratio > 0.98


def infer_target_column(df: pd.DataFrame) -> str | None:
    preferred = [
        column
        for column in df.columns
        if column.lower() in LIKELY_TARGET_NAMES and not _looks_like_identifier(df[column], column)
    ]
    if preferred:
        return preferred[0]

    candidates = [column for column in df.columns if not _looks_like_identifier(df[column], column)]
    if not candidates:
        return None

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


def detect_dataset(path: Path) -> dict[str, Any]:
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
    return {
        "dataset_type": "tabular",
        "problem_type": problem_type,
        "target_column": target_column,
        "dataframe": dataframe,
        "preview_summary": {
            "filename": path.name,
            "rows": int(dataframe.shape[0]),
            "columns": int(dataframe.shape[1]),
            "target_column": target_column,
            "problem_type": problem_type,
        },
    }
