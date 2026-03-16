from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .audio_pipeline import clean_audio_dataset
from .image_pipeline import clean_image_dataset


def _quality_score(df: pd.DataFrame) -> float:
    missing_ratio = float(df.isna().mean().mean()) if len(df.columns) else 0.0
    duplicate_ratio = float(df.duplicated().mean()) if len(df) else 0.0
    zero_variance_ratio = float((df.nunique(dropna=False) <= 1).mean()) if len(df.columns) else 0.0
    score = 10 - (missing_ratio * 5) - (duplicate_ratio * 3) - (zero_variance_ratio * 2)
    return round(max(score, 1.0), 2)


def clean_tabular_dataset(df: pd.DataFrame, target_column: str | None) -> dict[str, Any]:
    original_df = df.copy()
    cleaned = df.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    actions: list[str] = []

    duplicate_count = int(cleaned.duplicated().sum())
    if duplicate_count:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        actions.append(f"Removed {duplicate_count} duplicate rows")

    high_missing_columns = [
        column
        for column in cleaned.columns
        if column != target_column and cleaned[column].isna().mean() >= 0.5
    ]
    if high_missing_columns:
        cleaned = cleaned.drop(columns=high_missing_columns)
        actions.append(f"Dropped high-missing columns: {', '.join(high_missing_columns)}")

    cleaned = cleaned.replace(r"^\s*$", np.nan, regex=True)

    coerced_numeric: list[str] = []
    for column in cleaned.columns:
        if column == target_column:
            continue
        if cleaned[column].dtype == object:
            converted = pd.to_numeric(cleaned[column], errors="coerce")
            if converted.notna().sum() >= max(5, int(len(cleaned) * 0.7)):
                cleaned[column] = converted
                coerced_numeric.append(column)
    if coerced_numeric:
        actions.append(f"Coerced likely numeric text columns: {', '.join(coerced_numeric)}")

    zero_variance_columns = [
        column for column in cleaned.columns if column != target_column and cleaned[column].nunique(dropna=False) <= 1
    ]
    if zero_variance_columns:
        cleaned = cleaned.drop(columns=zero_variance_columns)
        actions.append(f"Dropped zero-variance columns: {', '.join(zero_variance_columns)}")

    if target_column and target_column in cleaned.columns and cleaned[target_column].isna().any():
        missing_target = int(cleaned[target_column].isna().sum())
        cleaned = cleaned.dropna(subset=[target_column]).reset_index(drop=True)
        actions.append(f"Dropped {missing_target} rows with missing target")

    for column in cleaned.columns:
        if column == target_column:
            continue
        if is_numeric_dtype(cleaned[column]):
            if cleaned[column].isna().any():
                fill_value = float(cleaned[column].median())
                missing = int(cleaned[column].isna().sum())
                cleaned[column] = cleaned[column].fillna(fill_value)
                actions.append(f"Imputed {missing} numeric values in {column} with median {fill_value:.3f}")
        else:
            if cleaned[column].isna().any():
                fill_value = cleaned[column].mode(dropna=True)
                chosen = str(fill_value.iloc[0]) if not fill_value.empty else "unknown"
                missing = int(cleaned[column].isna().sum())
                cleaned[column] = cleaned[column].fillna(chosen)
                actions.append(f"Imputed {missing} categorical values in {column} with mode '{chosen}'")
            cleaned[column] = cleaned[column].astype(str).str.strip()

    return {
        "dataframe": cleaned,
        "target_column": target_column,
        "actions": actions,
        "summary": {
            "rows_before": int(original_df.shape[0]),
            "rows_after": int(cleaned.shape[0]),
            "columns_before": int(original_df.shape[1]),
            "columns_after": int(cleaned.shape[1]),
            "missing_before": int(original_df.isna().sum().sum()),
            "missing_after": int(cleaned.isna().sum().sum()),
            "quality_before": _quality_score(original_df),
            "quality_after": _quality_score(cleaned),
        },
    }


def clean_text_dataset(
    df: pd.DataFrame,
    *,
    target_column: str,
    text_column: str,
) -> dict[str, Any]:
    cleaned = clean_tabular_dataset(df, target_column=target_column)
    text_df = cleaned["dataframe"].copy()
    before_rows = len(text_df)
    text_df[text_column] = text_df[text_column].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    text_df = text_df[text_df[text_column].str.len() > 0].reset_index(drop=True)
    removed_empty = before_rows - len(text_df)
    if removed_empty:
        cleaned["actions"].append(f"Removed {removed_empty} rows with empty text in {text_column}")
    cleaned["dataframe"] = text_df
    cleaned["summary"]["rows_after"] = int(text_df.shape[0])
    cleaned["summary"]["quality_after"] = _quality_score(text_df)
    return cleaned


def clean_timeseries_dataset(
    df: pd.DataFrame,
    *,
    target_column: str,
    time_column: str,
) -> dict[str, Any]:
    cleaned = clean_tabular_dataset(df, target_column=target_column)
    ts_df = cleaned["dataframe"].copy()
    parsed_time = pd.to_datetime(ts_df[time_column], errors="coerce", utc=False)
    invalid_time = int(parsed_time.isna().sum())
    ts_df[time_column] = parsed_time
    ts_df = ts_df.dropna(subset=[time_column]).sort_values(time_column).reset_index(drop=True)
    if invalid_time:
        cleaned["actions"].append(f"Dropped {invalid_time} rows with invalid timestamps in {time_column}")
    duplicate_timestamps = int(ts_df.duplicated(subset=[time_column]).sum())
    if duplicate_timestamps:
        ts_df = ts_df.drop_duplicates(subset=[time_column], keep="last").reset_index(drop=True)
        cleaned["actions"].append(f"Collapsed {duplicate_timestamps} duplicate timestamps in {time_column}")
    cleaned["dataframe"] = ts_df
    cleaned["summary"]["rows_after"] = int(ts_df.shape[0])
    cleaned["summary"]["quality_after"] = _quality_score(ts_df)
    return cleaned


def clean_dataset(
    df: pd.DataFrame | None,
    *,
    target_column: str | None,
    dataset_type: str,
    dataset_context: dict[str, Any] | None = None,
    source_path: str | None = None,
) -> dict[str, Any]:
    context = dataset_context or {}
    if dataset_type == "audio":
        if not source_path:
            raise ValueError("Audio datasets require a source_path")
        return clean_audio_dataset(source_path, target_column=target_column or "label")
    if dataset_type == "image":
        if not source_path:
            raise ValueError("Image datasets require a source_path")
        return clean_image_dataset(source_path, target_column=target_column or "label")
    if dataset_type == "text" and target_column and context.get("text_column"):
        assert df is not None
        return clean_text_dataset(df, target_column=target_column, text_column=context["text_column"])
    if dataset_type == "timeseries" and target_column and context.get("time_column"):
        assert df is not None
        return clean_timeseries_dataset(df, target_column=target_column, time_column=context["time_column"])
    assert df is not None
    return clean_tabular_dataset(df, target_column=target_column)
