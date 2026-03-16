from __future__ import annotations

from typing import Any


def provider_catalog() -> list[dict[str, Any]]:
    return [
        {
            "id": "ollama",
            "label": "Ollama",
            "provider_type": "local",
            "auth_modes": ["local"],
            "default_base_url": "http://127.0.0.1:11434",
        },
        {
            "id": "lmstudio",
            "label": "LM Studio",
            "provider_type": "local",
            "auth_modes": ["local", "api_key"],
            "default_base_url": "http://127.0.0.1:1234/v1",
        },
        {
            "id": "openai_compatible",
            "label": "OpenAI-Compatible Endpoint",
            "provider_type": "custom",
            "auth_modes": ["api_key"],
            "default_base_url": "http://127.0.0.1:4000/v1",
        },
        {
            "id": "openai",
            "label": "OpenAI API",
            "provider_type": "cloud_optional",
            "auth_modes": ["api_key", "browser_login"],
            "default_base_url": "https://api.openai.com/v1",
        },
    ]


def recommend_model_families(analysis: dict[str, Any], task_type: str) -> dict[str, Any]:
    rows = analysis["shape"]["rows"]
    categorical_count = analysis["categorical_feature_count"]

    if task_type == "classification":
        recommended = [
            {"name": "Logistic Regression", "reason": "Strong baseline and sanity check"},
            {"name": "Random Forest", "reason": "Good mixed-feature default"},
            {"name": "Extra Trees", "reason": "Fast strong tabular ensemble"},
            {"name": "Histogram Gradient Boosting", "reason": "Strong boosted-tree candidate"},
            {"name": "SVM", "reason": "Useful comparison when the feature space is manageable"},
        ]
        not_recommended = []
        if rows > 20000:
            not_recommended.append({"name": "KNN", "reason": "Prediction gets slow on large datasets"})
        if categorical_count > 20:
            not_recommended.append({"name": "SVM", "reason": "Large encoded spaces can be slow"})
    else:
        recommended = [
            {"name": "Linear Regression", "reason": "Baseline for signal calibration"},
            {"name": "Ridge", "reason": "Regularized linear baseline"},
            {"name": "Random Forest Regressor", "reason": "Robust non-linear default"},
            {"name": "Histogram Gradient Boosting Regressor", "reason": "Strong larger-data candidate"},
        ]
        not_recommended = []

    return {
        "recommended": recommended,
        "not_recommended": not_recommended,
        "dataset_fit": analysis["dataset_fitness"],
    }
