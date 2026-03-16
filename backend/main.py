from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import database
from .config import BUILTIN_MODELS_DIR, FRONTEND_BUILD_DIR, SUPPORTED_FILE_EXTENSIONS, ensure_directories
from .jobs.queue import JobQueue
from .jobs.state import JobStateManager
from .modules.recommender import provider_catalog
from .orchestrator import Orchestrator
from .utils.agent_routing import model_chain, normalize_agent_settings
from .utils.helpers import read_json_file
from .utils.storage import save_upload
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


@app.get("/api/agent/providers")
def get_agent_providers() -> dict[str, Any]:
    return {"providers": provider_catalog(), "routing_style": "openclaw-like provider/model refs with fallback chain"}


async def _list_remote_models(settings: dict[str, Any]) -> list[str]:
    normalized = normalize_agent_settings(settings)
    provider = normalized.get("provider", "")
    base_url = normalized.get("base_url", "")
    api_key = normalized.get("api_key", "")
    headers: dict[str, str] = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    async with httpx.AsyncClient(timeout=10.0) as client:
        if provider == "ollama":
            response = await client.get(f"{base_url.rstrip('/')}/api/tags")
            response.raise_for_status()
            return [item["name"] for item in response.json().get("models", [])]

        if provider in {"lmstudio", "openai_compatible", "openai"}:
            url = f"{base_url.rstrip('/')}/models" if not base_url.endswith("/v1") else f"{base_url}/models"
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            payload = response.json()
            return [item["id"] for item in payload.get("data", [])]

    return []


@app.post("/api/agent/models")
async def get_agent_models(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        normalized = normalize_agent_settings(payload)
        models = await _list_remote_models(normalized)
    except Exception as exc:
        return {"models": [], "error": str(exc)}
    return {
        "models": models,
        "model_refs": [f"{normalized['provider']}/{model_name}" for model_name in models],
        "normalized": normalized,
    }


def _rule_based_console_response(command: str, job_id: str | None) -> str:
    normalized = command.lower().strip()
    if normalized in {"help", "/help"}:
        return "Commands: help, list jobs, latest result, cleaning summary, recommended model, provider status"
    if normalized in {"list jobs", "/jobs"}:
        jobs = database.list_jobs(limit=5)
        if not jobs:
            return "No jobs yet."
        return "\n".join(f"{job['id']} | {job['status']} | {job['filename']}" for job in jobs)
    if normalized in {"latest result", "recommended model", "/model"}:
        jobs = database.list_jobs(limit=1)
        if not jobs:
            return "No completed jobs available yet."
        latest_id = job_id or jobs[0]["id"]
        result = database.get_job_result(latest_id)
        if not result:
            return "Latest job is not finished yet."
        winner = result["comparison_json"][0]["name"] if result["comparison_json"] else "unknown"
        metrics = result["metrics_json"]
        return f"Best model: {winner}. Key metrics: {json.dumps(metrics)}"
    if normalized in {"cleaning summary", "/cleaning"}:
        jobs = database.list_jobs(limit=1)
        if not jobs:
            return "No jobs available yet."
        latest_id = job_id or jobs[0]["id"]
        result = database.get_job_result(latest_id)
        if not result:
            return "Latest job is not finished yet."
        return json.dumps(result["summary_json"].get("cleaning", {}), indent=2)
    if normalized in {"provider status", "/providers"}:
        return json.dumps(provider_catalog(), indent=2)
    return "Command not recognized. Type help."


async def _dispatch_provider_command(command: str, settings: dict[str, Any]) -> str:
    normalized = normalize_agent_settings(settings)
    auth_mode = normalized.get("auth_mode", "local")
    if auth_mode == "browser_login":
        return "Browser-login mode is represented in the selector, but execution currently uses local or API-key-backed routes."

    async with httpx.AsyncClient(timeout=30.0) as client:
        for route in model_chain(normalized):
            provider = route["provider"]
            model = route["model"]
            if not model:
                continue
            base_url = route["base_url"]
            headers: dict[str, str] = {"Authorization": f"Bearer {normalized['api_key']}"} if normalized.get("api_key") else {}
            try:
                if provider == "ollama":
                    response = await client.post(
                        f"{base_url.rstrip('/')}/api/chat",
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": command}],
                            "stream": False,
                        },
                    )
                    response.raise_for_status()
                    payload = response.json()
                    content = payload.get("message", {}).get("content", "")
                    if content:
                        return f"[{route['ref']}]\n{content}"

                if provider in {"lmstudio", "openai_compatible", "openai"}:
                    url = f"{base_url.rstrip('/')}/chat/completions" if not base_url.endswith("/v1") else f"{base_url}/chat/completions"
                    response = await client.post(
                        url,
                        headers=headers,
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": command}],
                            "temperature": 0.2,
                        },
                    )
                    response.raise_for_status()
                    payload = response.json()
                    choices = payload.get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                        if content:
                            return f"[{route['ref']}]\n{content}"
            except Exception:
                continue
    return ""


@app.post("/api/agent/command")
async def run_agent_command(payload: dict[str, Any]) -> dict[str, str]:
    command = str(payload.get("command", "")).strip()
    settings = payload.get("settings", {})
    job_id = payload.get("job_id")
    if not command:
        raise HTTPException(status_code=400, detail="Command is required")
    try:
        if settings:
            response = await _dispatch_provider_command(command, settings)
            if response:
                return {"response": response}
    except Exception:
        pass
    return {"response": _rule_based_console_response(command, job_id)}


if FRONTEND_BUILD_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_BUILD_DIR, html=True), name="frontend")
