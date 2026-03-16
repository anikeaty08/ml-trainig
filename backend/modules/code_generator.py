from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from ..utils.storage import save_csv, save_text


def generate_training_artifacts(
    *,
    job_id: str,
    cleaned_df: pd.DataFrame,
    target_column: str,
    task_type: str,
    best_model: dict[str, Any],
    artifact_dirs: dict[str, Path],
) -> dict[str, Path]:
    template_dir = Path(__file__).resolve().parent.parent / "templates"
    env = Environment(loader=FileSystemLoader(template_dir))

    training_template = env.get_template("sklearn_training.jinja2")
    report_template = env.get_template("html_report.jinja2")

    cleaned_dataset_path = save_csv(artifact_dirs["data"] / "cleaned_dataset.csv", cleaned_df)
    predictions_df = best_model["evaluation"]["X_test"].copy()
    predictions_df[target_column] = best_model["evaluation"]["y_test"].values
    predictions_df["prediction"] = best_model["evaluation"]["predictions"]
    if best_model["evaluation"]["probabilities"]:
        predictions_df["probability"] = best_model["evaluation"]["probabilities"]
    predictions_path = save_csv(artifact_dirs["reports"] / "predictions.csv", predictions_df)

    model_path = artifact_dirs["model"] / "best_model.joblib"
    joblib.dump(best_model["pipeline"], model_path)

    training_script = training_template.render(
        target_column=target_column,
        task_type=task_type,
        best_model_name=best_model["name"],
        best_params=best_model["params"],
    )
    training_script_path = save_text(artifact_dirs["code"] / "train.py", training_script)

    report_html = report_template.render(
        job_id=job_id,
        task_type=task_type,
        target_column=target_column,
        best_model_name=best_model["name"],
        test_metrics=best_model["test_metrics"],
        params=best_model["params"],
        feature_importance=best_model["feature_importance"],
    )
    report_path = save_text(artifact_dirs["reports"] / "report.html", report_html)

    save_text(artifact_dirs["code"] / "README.md", f"Reproduce the {best_model['name']} pipeline with train.py\n")
    save_text(artifact_dirs["code"] / "requirements.txt", "pandas\nnumpy\nscikit-learn\njoblib\n")

    return {
        "cleaned_dataset_path": cleaned_dataset_path,
        "predictions_path": predictions_path,
        "model_path": model_path,
        "training_script_path": training_script_path,
        "report_path": report_path,
    }
