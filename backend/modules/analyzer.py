from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def analyze_dataset(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    dataset_type: str = "tabular",
    dataset_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = dataset_context or {}
    numeric_columns = [column for column in df.columns if column != target_column and is_numeric_dtype(df[column])]
    categorical_columns = [column for column in df.columns if column not in numeric_columns and column != target_column]
    target = df[target_column]

    if task_type == "classification":
        distribution = target.value_counts(normalize=True, dropna=False).round(4).to_dict()
        target_summary = {
            "classes": int(target.nunique(dropna=True)),
            "distribution": distribution,
            "imbalance_ratio": round(float(max(distribution.values()) / max(min(distribution.values()), 1e-9)), 3)
            if distribution
            else 1.0,
        }
    else:
        target_summary = {
            "min": float(target.min()),
            "max": float(target.max()),
            "mean": float(target.mean()),
            "std": float(target.std(ddof=0)),
        }

    numeric_correlations: list[dict[str, Any]] = []
    if numeric_columns and is_numeric_dtype(target):
        correlations = df[numeric_columns + [target_column]].corr(numeric_only=True)[target_column].drop(target_column)
        numeric_correlations = [
            {"feature": feature, "correlation": round(float(value), 4)}
            for feature, value in correlations.abs().sort_values(ascending=False).items()
        ]

    fitness = []
    if task_type == "classification":
        fitness.append("Good fit for classification workflows")
    elif task_type == "regression":
        fitness.append("Good fit for regression workflows")
    if categorical_columns:
        fitness.append("Categorical signal detected; encoded pipelines will be used")
    if not any("date" in column.lower() or "time" in column.lower() for column in df.columns):
        fitness.append("Not ideal for time-series forecasting because temporal columns were not detected")
    if dataset_type == "text" and context.get("text_column"):
        text_column = context["text_column"]
        text_lengths = df[text_column].astype(str).str.len()
        fitness.append("Text column detected; TF-IDF style modeling is available")
        text_summary = {
            "text_column": text_column,
            "avg_char_length": round(float(text_lengths.mean()), 2),
            "max_char_length": int(text_lengths.max()),
        }
    else:
        text_summary = {}

    if dataset_type == "timeseries" and context.get("time_column"):
        time_column = context["time_column"]
        time_series = pd.to_datetime(df[time_column], errors="coerce")
        deltas = time_series.sort_values().diff().dropna()
        timeseries_summary = {
            "time_column": time_column,
            "start": str(time_series.min()),
            "end": str(time_series.max()),
            "median_step_seconds": float(deltas.median().total_seconds()) if not deltas.empty else None,
        }
        fitness.append("Temporal ordering detected; sequential forecasting is available")
    else:
        timeseries_summary = {}

    if dataset_type == "image":
        image_summary = {
            "image_count": context.get("image_count", int(df.shape[0])),
            "class_distribution": context.get("class_distribution", {}),
            "label_strategy": context.get("label_strategy", "derived"),
        }
        fitness.append("Image archive detected; local handcrafted visual features were extracted for classification")
    else:
        image_summary = {}

    return {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "dataset_type": dataset_type,
        "numeric_feature_count": len(numeric_columns),
        "categorical_feature_count": len(categorical_columns),
        "target_summary": target_summary,
        "top_numeric_correlations": numeric_correlations[:10],
        "text_summary": text_summary,
        "timeseries_summary": timeseries_summary,
        "image_summary": image_summary,
        "numeric_profile": [
            {
                "feature": column,
                "mean": round(float(df[column].mean()), 4),
                "std": round(float(df[column].std(ddof=0)), 4),
                "min": round(float(df[column].min()), 4),
                "max": round(float(df[column].max()), 4),
                "skew": round(float(df[column].skew()), 4) if len(df[column]) > 2 else 0.0,
            }
            for column in numeric_columns[:20]
        ],
        "categorical_profile": [
            {
                "feature": column,
                "unique_values": int(df[column].nunique(dropna=True)),
                "top_values": df[column].value_counts(dropna=False).head(5).to_dict(),
            }
            for column in categorical_columns[:10]
        ],
        "dataset_fitness": fitness,
        "difficulty_score": round(
            min(10.0, max(2.0, 5.0 + np.log1p(len(df.columns)) - (0.4 if len(df) > 10000 else 0.0))),
            2,
        ),
    }
