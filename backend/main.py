from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import database
from .config import APP_PORT, BUILTIN_MODELS_DIR, FRONTEND_BUILD_DIR, SUPPORTED_FILE_EXTENSIONS, ensure_directories
from .jobs.queue import JobQueue
from .jobs.state import JobStateManager
from .modules.recommender import provider_catalog
from .orchestrator import Orchestrator
from .utils.agent_console import dispatch_provider_command, list_remote_models, onboard_text, resolve_console_command
from .utils.agent_routing import normalize_agent_settings, resolve_agent_policy
from .utils.helpers import read_json_file
from .utils.storage import create_bundle_archive, delete_job_artifacts, save_upload
from .utils.validators import validate_upload


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_directories()
    database.init_database()
    state_manager.bootstrap_from_database(database.list_jobs(limit=200))
    yield


app = FastAPI(title="ML Pipeline Agent", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

state_manager = JobStateManager()
orchestrator = Orchestrator(state_manager)
job_queue = JobQueue(orchestrator.process_job)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models/available")
def available_models() -> dict[str, Any]:
    manifest_path = BUILTIN_MODELS_DIR / "models_manifest.json"
    manifest = read_json_file(manifest_path) if manifest_path.exists() else {"models": []}
    return {
        "builtin_models": manifest.get("models", []),
        "candidate_families": [
            "Logistic Regression / Linear Regression",
            "Random Forest",
            "Extra Trees",
            "Gradient Boosting",
            "Histogram Gradient Boosting",
            "SVM",
            "KNN",
            "Naive Bayes",
            "Voting Ensemble",
            "TF-IDF Text Pipelines",
            "Lag-based Time-Series Forecasting",
        ],
    }


@app.get("/api/jobs")
def list_jobs() -> list[dict[str, Any]]:
    return database.list_jobs()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job["logs"] = database.get_job_logs(job_id)
    result = database.get_job_result(job_id)
    if result:
        job["result"] = result
    return job


@app.get("/api/jobs/{job_id}/results")
def get_job_results(job_id: str) -> dict[str, Any]:
    result = database.get_job_result(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Job results not available yet")
    return result


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict[str, str]:
    job = database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] == "processing":
        raise HTTPException(status_code=400, detail="Processing jobs cannot be deleted in this MVP")
    delete_job_artifacts(job)
    database.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    database.execute("DELETE FROM job_logs WHERE job_id = ?", (job_id,))
    database.execute("DELETE FROM job_results WHERE job_id = ?", (job_id,))
    return {"status": "deleted"}


@app.post("/api/jobs/submit")
async def submit_job(
    file: UploadFile = File(...),
    provider: str = Form("ollama"),
    model: str = Form(""),
    settings_json: str = Form("{}"),
) -> JSONResponse:
    validate_upload(file.filename or "", SUPPORTED_FILE_EXTENSIONS)
    content = await file.read()
    job_id = uuid.uuid4().hex[:12]
    upload_path = save_upload(job_id, file.filename or "dataset.csv", content)
    try:
        raw_settings = json.loads(settings_json or "{}")
    except json.JSONDecodeError:
        raw_settings = {}
    normalized_settings = normalize_agent_settings({**raw_settings, "provider": provider, "model": model})
    database.create_job(
        job_id,
        file.filename or "dataset.csv",
        str(upload_path),
        normalized_settings["provider"],
        normalized_settings["model"],
    )
    state_manager.set_state(job_id, {"job_id": job_id, "stage": "queued", "progress": 0, "message": "Job queued"})
    job_queue.enqueue(job_id, str(upload_path))
    return JSONResponse({"job_id": job_id, "status": "queued"})


@app.websocket("/api/jobs/{job_id}/progress")
async def job_progress(job_id: str, websocket: WebSocket) -> None:
    await websocket.accept()
    last_payload: dict[str, Any] | None = None
    try:
        while True:
            payload = state_manager.get_state(job_id) or {}
            if payload and payload != last_payload:
                await websocket.send_json(payload)
                last_payload = dict(payload)
                if payload.get("stage") in {"completed", "failed"}:
                    await asyncio.sleep(0.5)
                    break
            await asyncio.sleep(0.75)
    except WebSocketDisconnect:
        return


def _download_path(job_id: str, field: str) -> FileResponse:
    job = database.get_job(job_id)
    if not job or not job.get(field):
        raise HTTPException(status_code=404, detail="File not available")
    path = Path(job[field])
    if not path.exists():
        raise HTTPException(status_code=404, detail="File missing on disk")
    return FileResponse(path)


@app.get("/api/downloads/{job_id}/model")
def download_model(job_id: str) -> FileResponse:
    return _download_path(job_id, "model_path")


@app.get("/api/downloads/{job_id}/code")
def download_code(job_id: str) -> FileResponse:
    return _download_path(job_id, "code_path")


@app.get("/api/downloads/{job_id}/report")
def download_report(job_id: str) -> FileResponse:
    return _download_path(job_id, "report_path")


@app.get("/api/downloads/{job_id}/predictions")
def download_predictions(job_id: str) -> FileResponse:
    return _download_path(job_id, "predictions_path")


@app.get("/api/downloads/{job_id}/cleaned-data")
def download_cleaned_data(job_id: str) -> FileResponse:
    return _download_path(job_id, "cleaned_data_path")


@app.get("/api/downloads/{job_id}/bundle")
def download_bundle(job_id: str) -> FileResponse:
    job = database.get_job(job_id)
    if not job or not job.get("report_path"):
        raise HTTPException(status_code=404, detail="Bundle not available")
    reports_dir = Path(job["report_path"]).parent.parent
    if not reports_dir.exists():
        raise HTTPException(status_code=404, detail="Bundle source folder missing")
    archive_path = create_bundle_archive(reports_dir)
    return FileResponse(archive_path)


@app.get("/api/agent/providers")
def get_agent_providers() -> dict[str, Any]:
    return {
        "providers": provider_catalog(),
        "routing_style": "openclaw-like provider/model refs with fallback chain and provider auth rotation",
        "commands": [
            "/help",
            "/onboard [provider]",
            "/model",
            "/model list",
            "/model status",
            "/model status --probe",
            "/model scan",
            "/model set <provider/model>",
            "/model image <provider/model>",
            "/model fallback add <provider/model>",
            "/model fallback remove <provider/model>",
            "/jobs",
            "/job show <job_id>",
            "/dataset summary",
            "/dataset columns",
            "/dataset head",
            "/dataset stats",
            "/analysis",
            "/cleaning",
            "/search <query>",
        ],
    }


@app.get("/api/agent/config")
def get_agent_config() -> dict[str, Any]:
    return resolve_agent_policy(database.get_agent_config())


@app.put("/api/agent/config")
def save_agent_config(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = resolve_agent_policy(payload)
    database.save_agent_config(normalized)
    return normalized


@app.post("/api/agent/policy")
def get_agent_policy(payload: dict[str, Any]) -> dict[str, Any]:
    return resolve_agent_policy(payload)

@app.post("/api/agent/models")
async def get_agent_models(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        normalized = resolve_agent_policy(payload)
        models = await list_remote_models(normalized)
    except Exception as exc:
        return {"models": [], "error": str(exc)}
    return {
        "models": models,
        "model_refs": [f"{normalized['provider']}/{model_name}" for model_name in models],
        "normalized": normalized,
    }

@app.post("/api/agent/command")
async def run_agent_command(payload: dict[str, Any]) -> dict[str, str]:
    command = str(payload.get("command", "")).strip()
    settings = payload.get("settings") or database.get_agent_config()
    job_id = payload.get("job_id")
    if not command:
        raise HTTPException(status_code=400, detail="Command is required")
    local_response = await resolve_console_command(command, settings, job_id)
    if local_response:
        return {"response": local_response}

    try:
        if settings:
            response = await dispatch_provider_command(command, settings, job_id)
            if response:
                return {"response": response}
    except Exception:
        pass
    return {"response": onboard_text(settings)}


@app.get("/api/meta/runtime")
def runtime_meta() -> dict[str, Any]:
    return {"app_port": APP_PORT, "app_origin": f"http://127.0.0.1:{APP_PORT}"}


if FRONTEND_BUILD_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_BUILD_DIR, html=True), name="frontend")
