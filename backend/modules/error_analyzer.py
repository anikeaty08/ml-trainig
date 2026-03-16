from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def analyze_errors(
    *,
    task_type: str,
    target_column: str,
    evaluation: dict[str, Any],
    top_features: list[dict[str, Any]],
) -> dict[str, Any]:
    X_test = evaluation["X_test"].copy()
    y_test = evaluation["y_test"]
    predictions = np.asarray(evaluation["predictions"])

    if task_type == "classification":
        labels = sorted(pd.Series(y_test).dropna().unique().tolist())
        matrix = confusion_matrix(y_test, predictions, labels=labels).tolist()
        errors = X_test.copy()
        errors[target_column] = y_test.values
        errors["prediction"] = predictions
        errors["is_error"] = errors[target_column] != errors["prediction"]
        return {
            "confusion_matrix": {"labels": labels, "matrix": matrix},
            "error_count": int(errors["is_error"].sum()),
            "error_rate": round(float(errors["is_error"].mean()), 4),
            "top_features_considered": top_features,
            "worst_cases": errors[errors["is_error"]].head(10).to_dict(orient="records"),
        }

    residuals = y_test - predictions
    errors = X_test.copy()
    errors[target_column] = y_test.values
    errors["prediction"] = predictions
    errors["absolute_error"] = np.abs(residuals)
    return {
        "mae_like": round(float(np.mean(np.abs(residuals))), 4),
        "rmse_like": round(float(np.sqrt(np.mean(np.square(residuals)))), 4),
        "top_features_considered": top_features,
        "worst_cases": errors.sort_values("absolute_error", ascending=False).head(10).to_dict(orient="records"),
    }
