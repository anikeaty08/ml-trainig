from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


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
