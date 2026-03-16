from pathlib import Path

import pandas as pd

from backend.modules.code_generator import generate_training_artifacts


class DummyPipeline:
    pass


def _artifact_dirs(base: Path) -> dict[str, Path]:
    paths = {
        "base": base,
        "model": base / "model",
        "code": base / "code",
        "data": base / "data",
        "reports": base / "reports",
        "logs": base / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def test_generate_timeseries_artifacts_uses_timeseries_template(tmp_path):
    artifact_dirs = _artifact_dirs(tmp_path / "artifacts")
    cleaned_df = pd.DataFrame(
        {
            "timestamp": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "target": [10.0, 12.0, 13.5],
        }
    )
    best_model = {
        "name": "ARIMA",
        "params": {"framework": "statsmodels", "order": [1, 1, 1], "time_column": "timestamp"},
        "pipeline": DummyPipeline(),
        "test_metrics": {"mae": 1.2, "rmse": 1.4, "r2": 0.71},
        "feature_importance": [],
        "evaluation": {
            "X_test": pd.DataFrame({"timestamp": ["2025-01-03"]}),
            "y_test": pd.Series([13.5]),
            "predictions": [13.1],
            "probabilities": [],
        },
    }

    outputs = generate_training_artifacts(
        job_id="job123",
        cleaned_df=cleaned_df,
        target_column="target",
        task_type="regression",
        dataset_type="timeseries",
        best_model=best_model,
        artifact_dirs=artifact_dirs,
    )

    script = outputs["training_script_path"].read_text(encoding="utf-8")
    requirements_text = (artifact_dirs["code"] / "requirements.txt").read_text(encoding="utf-8")

    assert "from statsmodels.tsa.arima.model import ARIMA" in script
    assert "statsmodels" in requirements_text


def test_generate_transformer_artifacts_include_transformers_requirements(tmp_path):
    artifact_dirs = _artifact_dirs(tmp_path / "artifacts")
    cleaned_df = pd.DataFrame(
        {
            "text": ["good", "bad"],
            "target": [1, 0],
        }
    )
    best_model = {
        "name": "DistilBERT Embeddings + Logistic Regression",
        "params": {"framework": "transformers", "model_id": "distilbert-base-uncased", "text_column": "text"},
        "pipeline": DummyPipeline(),
        "test_metrics": {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0},
        "feature_importance": [],
        "evaluation": {
            "X_test": pd.DataFrame({"text": ["good"]}),
            "y_test": pd.Series([1]),
            "predictions": [1],
            "probabilities": [0.91],
        },
    }

    generate_training_artifacts(
        job_id="job124",
        cleaned_df=cleaned_df,
        target_column="target",
        task_type="classification",
        dataset_type="text",
        best_model=best_model,
        artifact_dirs=artifact_dirs,
    )

    requirements_text = (artifact_dirs["code"] / "requirements.txt").read_text(encoding="utf-8")
    assert "transformers" in requirements_text
    assert "torch" in requirements_text
