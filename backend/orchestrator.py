from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from . import database
from .modules.analyzer import analyze_dataset
from .modules.cleaner import clean_dataset
from .modules.code_generator import generate_training_artifacts
from .modules.detector import detect_dataset
from .modules.error_analyzer import analyze_errors
from .modules.recommender import recommend_model_families
from .modules.trainer import train_candidate_models
from .utils.logger import get_job_logger
from .utils.storage import prepare_job_artifact_dirs, save_comparison_csv, save_json


class Orchestrator:
    def __init__(self, state_manager: Any) -> None:
        self.state_manager = state_manager

    def update_progress(
        self,
        job_id: str,
        *,
        stage: str,
        progress: int,
        message: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        payload = {"job_id": job_id, "stage": stage, "progress": progress, "message": message}
        if extra:
            payload.update(extra)
        database.update_job(job_id, stage=stage, progress=progress, message=message, status="processing")
        database.log_job(job_id, stage, message)
        self.state_manager.set_state(job_id, payload)

    def mark_failed(self, job_id: str, message: str, exc: Exception | None = None) -> None:
        details = f"{message}: {exc}" if exc else message
        database.update_job(
            job_id,
            status="failed",
            stage="failed",
            progress=100,
            message=message,
            error_message=details,
        )
        database.log_job(job_id, "failed", details)
        self.state_manager.set_state(
            job_id,
            {"job_id": job_id, "stage": "failed", "progress": 100, "message": message, "error_message": details},
        )

    def mark_completed(self, job_id: str, message: str, extra: dict[str, Any]) -> None:
        database.update_job(
            job_id,
            status="completed",
            stage="completed",
            progress=100,
            message=message,
            completed_at=database.utc_now(),
            **extra,
        )
        database.log_job(job_id, "completed", message)
        self.state_manager.set_state(job_id, {"job_id": job_id, "stage": "completed", "progress": 100, "message": message, **extra})

    def process_job(self, job_id: str, file_path: str) -> None:
        logger = get_job_logger(job_id)
        artifact_dirs = prepare_job_artifact_dirs(job_id)

        try:
            self.update_progress(job_id, stage="detecting_type", progress=5, message="Analyzing uploaded dataset")
            detection = detect_dataset(Path(file_path))
            if not detection["target_column"] or detection["problem_type"] == "unknown":
                self.mark_failed(
                    job_id,
                    "Could not confidently infer a usable target column. Add a target/label column or rename the intended target.",
                )
                return

            database.update_job(
                job_id,
                dataset_type=detection["dataset_type"],
                problem_type=detection["problem_type"],
                target_column=detection["target_column"],
            )

            self.update_progress(job_id, stage="cleaning", progress=20, message="Cleaning dataset automatically")
            cleaned = clean_dataset(
                detection["dataframe"],
                target_column=detection["target_column"],
                dataset_type=detection["dataset_type"],
                dataset_context=detection.get("dataset_context"),
            )
            cleaned_df = cleaned["dataframe"]

            self.update_progress(job_id, stage="analyzing", progress=40, message="Profiling dataset quality and signal")
            analysis = analyze_dataset(
                cleaned_df,
                target_column=cleaned["target_column"],
                task_type=detection["problem_type"],
                dataset_type=detection["dataset_type"],
                dataset_context=detection.get("dataset_context"),
            )
            recommendation = recommend_model_families(analysis=analysis, task_type=detection["problem_type"])

            self.update_progress(job_id, stage="training", progress=55, message="Training multiple candidate models")

            def progress_callback(message: str, progress: int, extra: dict[str, Any] | None = None) -> None:
                self.update_progress(job_id, stage="training", progress=progress, message=message, extra=extra)

            training = train_candidate_models(
                cleaned_df,
                target_column=cleaned["target_column"],
                task_type=detection["problem_type"],
                dataset_type=detection["dataset_type"],
                dataset_context=detection.get("dataset_context"),
                progress_callback=progress_callback,
            )

            self.update_progress(job_id, stage="analyzing_errors", progress=92, message="Analyzing failure patterns")
            error_analysis = analyze_errors(
                task_type=detection["problem_type"],
                target_column=cleaned["target_column"],
                evaluation=training["best_model"]["evaluation"],
                top_features=training["best_model"]["feature_importance"][:5],
            )

            self.update_progress(job_id, stage="generating_code", progress=95, message="Generating artifacts and report")
            generated = generate_training_artifacts(
                job_id=job_id,
                cleaned_df=cleaned_df,
                target_column=cleaned["target_column"],
                task_type=detection["problem_type"],
                best_model=training["best_model"],
                artifact_dirs=artifact_dirs,
            )

            comparison_path = save_comparison_csv(artifact_dirs["reports"], training["comparison"])
            save_json(artifact_dirs["reports"] / "analysis.json", analysis)
            save_json(artifact_dirs["reports"] / "recommendation.json", recommendation)
            save_json(artifact_dirs["reports"] / "error_analysis.json", error_analysis)
            save_json(
                artifact_dirs["reports"] / "cleaning_log.json",
                {
                    "target_column": cleaned["target_column"],
                    "actions": cleaned["actions"],
                    "summary": cleaned["summary"],
                },
            )
            save_json(artifact_dirs["reports"] / "training_summary.json", training["summary"])

            database.save_job_result(
                job_id,
                data_type=detection["dataset_type"],
                problem_type=detection["problem_type"],
                n_samples=int(cleaned_df.shape[0]),
                n_features=int(cleaned_df.shape[1] - 1),
                metrics=training["best_model"]["test_metrics"],
                recommendation=recommendation,
                comparison=training["comparison"],
                summary={
                    "detection": detection["preview_summary"],
                    "dataset_context": detection.get("dataset_context", {}),
                    "cleaning": cleaned["summary"],
                    "analysis": analysis,
                    "training": training["summary"],
                    "error_analysis": error_analysis,
                },
            )

            self.mark_completed(
                job_id,
                "Pipeline complete",
                {
                    "report_path": str(generated["report_path"]),
                    "model_path": str(generated["model_path"]),
                    "predictions_path": str(generated["predictions_path"]),
                    "code_path": str(generated["training_script_path"]),
                    "comparison_path": str(comparison_path),
                    "cleaned_data_path": str(generated["cleaned_dataset_path"]),
                },
            )
        except Exception as exc:
            logger.error("Job %s failed\n%s", job_id, traceback.format_exc())
            self.mark_failed(job_id, "Pipeline failed", exc)
