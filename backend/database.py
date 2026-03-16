from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterable

from .config import DB_PATH, ensure_directories


DB_LOCK = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_connection() -> Iterable[sqlite3.Connection]:
    ensure_directories()
    connection = sqlite3.connect(DB_PATH, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_database() -> None:
    with DB_LOCK:
        with get_connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress INTEGER NOT NULL DEFAULT 0,
                    stage TEXT NOT NULL,
                    message TEXT,
                    dataset_type TEXT,
                    problem_type TEXT,
                    target_column TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    error_message TEXT,
                    llm_provider TEXT,
                    llm_model TEXT,
                    report_path TEXT,
                    model_path TEXT,
                    predictions_path TEXT,
                    code_path TEXT,
                    comparison_path TEXT,
                    cleaned_data_path TEXT
                );

                CREATE TABLE IF NOT EXISTS job_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS job_results (
                    job_id TEXT PRIMARY KEY,
                    data_type TEXT NOT NULL,
                    problem_type TEXT NOT NULL,
                    n_samples INTEGER NOT NULL,
                    n_features INTEGER NOT NULL,
                    metrics_json TEXT NOT NULL,
                    recommendation_json TEXT NOT NULL,
                    comparison_json TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS agent_settings (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    provider TEXT NOT NULL,
                    model TEXT,
                    base_url TEXT,
                    auth_mode TEXT NOT NULL DEFAULT 'local',
                    api_key TEXT,
                    browser_session_hint TEXT,
                    updated_at TEXT NOT NULL
                );
                """
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO agent_settings
                (id, provider, model, base_url, auth_mode, api_key, browser_session_hint, updated_at)
                VALUES (1, 'ollama', '', 'http://127.0.0.1:11434', 'local', '', '', ?)
                """,
                (utc_now(),),
            )


def _serialize(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str)
    return value


def execute(query: str, params: tuple[Any, ...] = ()) -> None:
    with DB_LOCK:
        with get_connection() as connection:
            connection.execute(query, tuple(_serialize(item) for item in params))


def fetch_one(query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    with DB_LOCK:
        with get_connection() as connection:
            row = connection.execute(query, tuple(_serialize(item) for item in params)).fetchone()
    return dict(row) if row else None


def fetch_all(query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with DB_LOCK:
        with get_connection() as connection:
            rows = connection.execute(query, tuple(_serialize(item) for item in params)).fetchall()
    return [dict(row) for row in rows]


def create_job(job_id: str, filename: str, filepath: str, provider: str = "", model: str = "") -> None:
    now = utc_now()
    execute(
        """
        INSERT INTO jobs (
            id, filename, filepath, status, progress, stage, message,
            created_at, updated_at, llm_provider, llm_model
        )
        VALUES (?, ?, ?, 'queued', 0, 'queued', 'Job queued', ?, ?, ?, ?)
        """,
        (job_id, filename, filepath, now, now, provider, model),
    )


def update_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = utc_now()
    assignments = ", ".join(f"{column} = ?" for column in fields)
    params = tuple(fields.values()) + (job_id,)
    execute(f"UPDATE jobs SET {assignments} WHERE id = ?", params)


def log_job(job_id: str, stage: str, message: str) -> None:
    execute(
        "INSERT INTO job_logs (job_id, stage, message, created_at) VALUES (?, ?, ?, ?)",
        (job_id, stage, message, utc_now()),
    )


def list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    return fetch_all("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))


def get_job(job_id: str) -> dict[str, Any] | None:
    return fetch_one("SELECT * FROM jobs WHERE id = ?", (job_id,))


def get_job_logs(job_id: str) -> list[dict[str, Any]]:
    return fetch_all(
        "SELECT stage, message, created_at FROM job_logs WHERE job_id = ? ORDER BY id ASC",
        (job_id,),
    )


def save_job_result(
    job_id: str,
    *,
    data_type: str,
    problem_type: str,
    n_samples: int,
    n_features: int,
    metrics: dict[str, Any],
    recommendation: dict[str, Any],
    comparison: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    execute(
        """
        INSERT OR REPLACE INTO job_results (
            job_id, data_type, problem_type, n_samples, n_features,
            metrics_json, recommendation_json, comparison_json, summary_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            job_id,
            data_type,
            problem_type,
            n_samples,
            n_features,
            metrics,
            recommendation,
            comparison,
            summary,
            utc_now(),
        ),
    )


def get_job_result(job_id: str) -> dict[str, Any] | None:
    row = fetch_one("SELECT * FROM job_results WHERE job_id = ?", (job_id,))
    if not row:
        return None
    for key in ("metrics_json", "recommendation_json", "comparison_json", "summary_json"):
        row[key] = json.loads(row[key])
    return row


def get_agent_settings(include_secrets: bool = False) -> dict[str, Any]:
    settings = fetch_one("SELECT * FROM agent_settings WHERE id = 1")
    if not settings:
        init_database()
        settings = fetch_one("SELECT * FROM agent_settings WHERE id = 1")
    assert settings is not None
    api_key = settings.get("api_key") or ""
    settings["has_api_key"] = bool(api_key)
    if include_secrets:
        return settings
    settings["api_key"] = f"{api_key[:4]}..." if api_key else ""
    return settings


def save_agent_settings(
    provider: str,
    model: str,
    base_url: str,
    auth_mode: str,
    api_key: str = "",
    browser_session_hint: str = "",
) -> None:
    execute(
        """
        UPDATE agent_settings
        SET provider = ?, model = ?, base_url = ?, auth_mode = ?, api_key = ?, browser_session_hint = ?, updated_at = ?
        WHERE id = 1
        """,
        (provider, model, base_url, auth_mode, api_key, browser_session_hint, utc_now()),
    )
