from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def primary_metric_name(task_type: str) -> str:
    return "f1" if task_type == "classification" else "r2"


def classification_metrics(y_true: Any, y_pred: Any, y_prob: Any | None) -> dict[str, float]:
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
    }
    try:
        if y_prob is not None and len(np.unique(y_true)) == 2:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
    except Exception:
        pass
    return metrics


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }
