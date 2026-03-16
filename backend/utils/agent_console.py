from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .. import database
from ..modules.recommender import provider_catalog
from .agent_routing import auth_profile_chain, model_chain, resolve_agent_policy
from .local_search import search_local_knowledge


OPENAI_COMPATIBLE_PROVIDERS = {
    "lmstudio",
    "openai",
    "openai-codex",
    "openai_compatible",
    "openrouter",
    "kimi",
}

CURATED_REMOTE_MODELS = {
    "openai": ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini"],
    "openai-codex": ["gpt-5.3-codex", "gpt-5.1-codex"],
    "anthropic": ["claude-3-7-sonnet-latest", "claude-3-5-haiku-latest"],
    "google": ["gemini-2.5-flash", "gemini-2.5-pro"],
    "openrouter": ["openai/gpt-4o-mini", "moonshotai/kimi-k2"],
    "kimi": ["moonshot-v1-8k", "kimi-k2"],
}

HELP_TEXT = """Commands
  /help
  /onboard [provider]
  /model status
  /model status --probe
  /model list
  /model scan
  /jobs
  /job show <job_id>
  /dataset summary
  /dataset columns
  /dataset head
  /dataset stats
  /analysis
  /cleaning
  /search <query>

Any other text is sent to the configured model chain with local job context attached.
""".strip()


def _json_block(payload: Any) -> str:
    return json.dumps(payload, indent=2, default=str)


def _provider_lookup(provider_id: str) -> dict[str, Any] | None:
    return next((item for item in provider_catalog() if item["id"] == provider_id), None)


def format_policy(settings: dict[str, Any]) -> str:
    policy = resolve_agent_policy(settings)
    chat = " -> ".join(item["ref"] for item in policy.get("chat_chain", [])) or "none"
    image = " -> ".join(item["ref"] for item in policy.get("image_chain", [])) or "none"
    auth_summary = "\n".join(
        f"{provider}: {', '.join(labels) or 'default'}"
        for provider, labels in sorted(policy.get("auth_summary", {}).items())
    ) or "none"
    return "\n".join(
        [
            f"primary: {policy.get('primary_model_ref') or 'unset'}",
            f"image: {policy.get('image_model_ref') or 'unset'}",
            f"fallbacks: {', '.join(policy.get('fallback_model_refs', [])) or 'none'}",
            f"allowlist: {', '.join(policy.get('model_allowlist', [])) or 'none'}",
            f"chat chain: {chat}",
            f"image chain: {image}",
            f"auth profiles:\n{auth_summary}",
        ]
    )


def _format_catalog(settings: dict[str, Any]) -> str:
    policy = resolve_agent_policy(settings)
    catalog = policy.get("model_catalog", [])
    provider = _provider_lookup(policy["provider"])
    lines = []
    if provider and provider.get("suggested_models"):
        lines.append("suggested:")
        lines.extend(f"  {model_ref}" for model_ref in provider["suggested_models"])
    if catalog:
        lines.append("catalog:")
        lines.extend(
            f"  {(item.get('alias') or item['ref'])} -> {item['ref']} [{item.get('capability', 'chat')}]"
            for item in catalog
        )
    return "\n".join(lines) if lines else "No local catalog entries. Try /model scan."


def _resolve_job(job_id: str | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    job = database.get_job(job_id) if job_id else None
    if not job:
        jobs = database.list_jobs(limit=1)
        job = jobs[0] if jobs else None
    result = database.get_job_result(job["id"]) if job else None
    return job, result


def _dataset_frame(job: dict[str, Any] | None) -> Any:
    if not job:
        return None
    import pandas as pd

    path_candidates = [job.get("cleaned_data_path"), job.get("filepath")]
    for candidate in path_candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if not path.exists() or path.suffix.lower() not in {".csv", ".tsv", ".txt"}:
            continue
        if path.suffix.lower() == ".tsv":
            return pd.read_csv(path, sep="\t")
        return pd.read_csv(path)
    return None


def _job_summary(job_id: str | None) -> str:
    job, result = _resolve_job(job_id)
    if not job:
        return "No jobs available yet."
    lines = [
        f"job: {job['id']}",
        f"file: {job['filename']}",
        f"status: {job['status']}",
        f"dataset_type: {job.get('dataset_type') or 'unknown'}",
        f"problem_type: {job.get('problem_type') or 'unknown'}",
        f"target: {job.get('target_column') or 'unknown'}",
    ]
    if result:
        winner = result["comparison_json"][0]["name"] if result["comparison_json"] else "unknown"
        lines.append(f"winner: {winner}")
        lines.append(f"metrics: {_json_block(result['metrics_json'])}")
    return "\n".join(lines)


def _dataset_summary(job_id: str | None) -> str:
    job, result = _resolve_job(job_id)
    if not job:
        return "No jobs available yet."
    frame = _dataset_frame(job)
    if frame is None:
        if result:
            return _json_block(result.get("summary_json", {}))
        return "No dataset table is available for summary."
    summary = {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "column_names": list(frame.columns),
        "missing_values": int(frame.isna().sum().sum()),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
    }
    if result:
        summary["pipeline_summary"] = result.get("summary_json", {})
    return _json_block(summary)


def _dataset_columns(job_id: str | None) -> str:
    job, _ = _resolve_job(job_id)
    frame = _dataset_frame(job)
    if frame is None:
        return "No CSV/TSV dataset is available for column inspection."
    return "\n".join(f"{column}: {frame[column].dtype}" for column in frame.columns)


def _dataset_head(job_id: str | None) -> str:
    job, _ = _resolve_job(job_id)
    frame = _dataset_frame(job)
    if frame is None:
        return "No CSV/TSV dataset is available for preview."
    return frame.head(5).to_string(index=False)


def _dataset_stats(job_id: str | None) -> str:
    job, result = _resolve_job(job_id)
    frame = _dataset_frame(job)
    if frame is None:
        if result:
            return _json_block(result.get("summary_json", {}).get("analysis", {}))
        return "No dataset table is available for statistics."
    numeric = frame.select_dtypes(include=["number"])
    if numeric.empty:
        return "The active dataset has no numeric columns to summarize."
    return numeric.describe(include="all").transpose().round(4).to_string()


def _analysis_text(job_id: str | None) -> str:
    _, result = _resolve_job(job_id)
    if not result:
        return "No completed results are available yet."
    return _json_block(result["summary_json"].get("analysis", {}))


def _cleaning_text(job_id: str | None) -> str:
    _, result = _resolve_job(job_id)
    if not result:
        return "No completed results are available yet."
    return _json_block(result["summary_json"].get("cleaning", {}))


def _search_text(query: str, job_id: str | None) -> str:
    hits = search_local_knowledge(query)
    sections = []
    if hits:
        sections.append(
            "\n\n".join(f"{item['path']}\n{item['snippet']}" for item in hits)
        )
    else:
        sections.append("No local doc/report matches found.")
    if any(term in query.lower() for term in ("dataset", "model", "accuracy", "clean", "feature")):
        sections.append(_job_summary(job_id))
    return "\n\n".join(section for section in sections if section)


def onboard_text(settings: dict[str, Any], provider_id: str | None = None) -> str:
    policy = resolve_agent_policy(settings)
    provider = _provider_lookup(provider_id or policy["provider"])
    if not provider:
        return "Unknown provider. Use /onboard ollama, /onboard openai, /onboard anthropic, /onboard google, or /onboard kimi."
    suggested = "\n".join(f"  {model_ref}" for model_ref in provider.get("suggested_models", [])) or "  none"
    return "\n".join(
        [
            f"provider: {provider['label']} ({provider['id']})",
            f"type: {provider['provider_type']}",
            f"auth modes: {', '.join(provider.get('auth_modes', [])) or 'none'}",
            f"capabilities: {', '.join(provider.get('capabilities', [])) or 'none'}",
            f"base URL: {provider.get('default_base_url') or 'n/a'}",
            f"hint: {provider.get('onboarding_hint') or 'n/a'}",
            "suggested models:",
            suggested,
            "next step: configure the provider in the app, save a profile, then run /model scan or ask the agent a question.",
        ]
    )


async def list_remote_models(settings: dict[str, Any]) -> list[str]:
    import httpx

    normalized = resolve_agent_policy(settings)
    provider = normalized.get("provider", "")
    profiles = auth_profile_chain(normalized, provider)

    async with httpx.AsyncClient(timeout=12.0) as client:
        for profile in profiles:
            auth_mode = profile.get("auth_mode", "")
            if auth_mode in {"browser_login", "oauth"} and not profile.get("api_key"):
                continue
            base_url = profile.get("base_url", "")
            try:
                if provider == "ollama":
                    response = await client.get(f"{base_url.rstrip('/')}/api/tags")
                    response.raise_for_status()
                    return [item["name"] for item in response.json().get("models", [])]

                if provider in OPENAI_COMPATIBLE_PROVIDERS:
                    headers = {"Authorization": f"Bearer {profile['api_key']}"} if profile.get("api_key") else {}
                    url = f"{base_url.rstrip('/')}/models" if not base_url.endswith("/v1") else f"{base_url}/models"
                    response = await client.get(url, headers=headers)
                    response.raise_for_status()
                    return [item["id"] for item in response.json().get("data", [])]

                if provider == "anthropic":
                    headers = {
                        "x-api-key": profile.get("api_key", ""),
                        "anthropic-version": "2023-06-01",
                    }
                    response = await client.get(f"{base_url.rstrip('/')}/models", headers=headers)
                    response.raise_for_status()
                    payload = response.json()
                    return [item["id"] for item in payload.get("data", []) or payload.get("models", [])]

                if provider == "google":
                    api_key = profile.get("api_key", "")
                    if not api_key:
                        continue
                    response = await client.get(f"{base_url.rstrip('/')}/v1beta/models", params={"key": api_key})
                    response.raise_for_status()
                    return [item["name"].split("/")[-1] for item in response.json().get("models", [])]
            except Exception:
                continue

    return CURATED_REMOTE_MODELS.get(provider, [])


async def resolve_console_command(command: str, settings: dict[str, Any], job_id: str | None = None) -> str | None:
    normalized = command.strip()
    lower = normalized.lower()

    if lower in {"help", "/help"}:
        return HELP_TEXT
    if lower in {"list jobs", "/jobs"}:
        jobs = database.list_jobs(limit=20)
        return "\n".join(f"{job['id']} | {job['status']} | {job['filename']}" for job in jobs) or "No jobs yet."
    if lower.startswith("/job show "):
        return _job_summary(normalized.split(maxsplit=2)[2])
    if lower in {"latest result", "recommended model"}:
        return _job_summary(job_id)
    if lower in {"/dataset", "/dataset summary", "dataset summary"}:
        return _dataset_summary(job_id)
    if lower in {"/dataset columns", "dataset columns"}:
        return _dataset_columns(job_id)
    if lower in {"/dataset head", "dataset head"}:
        return _dataset_head(job_id)
    if lower in {"/dataset stats", "dataset stats"}:
        return _dataset_stats(job_id)
    if lower in {"/analysis", "analysis"}:
        return _analysis_text(job_id)
    if lower in {"/cleaning", "cleaning summary"}:
        return _cleaning_text(job_id)
    if lower in {"provider status", "/providers"}:
        return _json_block(provider_catalog())
    if lower in {"/model", "/model status"}:
        return format_policy(settings)
    if lower == "/model list":
        return "\n\n".join([format_policy(settings), _format_catalog(settings)])
    if lower in {"/model scan", "/model list --remote"}:
        models = await list_remote_models(settings)
        provider = resolve_agent_policy(settings)["provider"]
        return "\n".join(f"{provider}/{model_name}" for model_name in models) or f"No remote models found for {provider}."
    if lower == "/model status --probe":
        models = await list_remote_models(settings)
        probe = f"probe: {len(models)} models visible" if models else "probe: no remote models detected"
        return f"{format_policy(settings)}\n{probe}"
    if lower.startswith("/search "):
        return _search_text(normalized.split(maxsplit=1)[1], job_id)
    if lower == "/onboard" or lower.startswith("/onboard "):
        provider_id = normalized.split(maxsplit=1)[1] if " " in normalized else None
        return onboard_text(settings, provider_id)
    return None


def _job_context_block(job_id: str | None) -> str:
    job, result = _resolve_job(job_id)
    if not job:
        return "No prior jobs are available."
    context = {
        "job_id": job["id"],
        "filename": job["filename"],
        "status": job["status"],
        "dataset_type": job.get("dataset_type"),
        "problem_type": job.get("problem_type"),
        "target_column": job.get("target_column"),
    }
    if result:
        comparison = result.get("comparison_json", [])
        context["winner"] = comparison[0]["name"] if comparison else None
        context["metrics"] = result.get("metrics_json", {})
        context["cleaning"] = result.get("summary_json", {}).get("cleaning", {})
        context["analysis"] = result.get("summary_json", {}).get("analysis", {})
    return _json_block(context)


def _knowledge_block(command: str) -> str:
    hits = search_local_knowledge(command, limit=2)
    if not hits:
        return "No related local docs matched."
    return "\n\n".join(f"{item['path']}\n{item['snippet']}" for item in hits)


def _build_chat_payload(command: str, job_id: str | None) -> dict[str, Any]:
    system_prompt = (
        "You are ML Pipeline Agent running fully on the user's local machine. "
        "Answer with practical, dataset-aware guidance. Use the provided local job context and knowledge snippets. "
        "If context is missing, say so. Do not claim you trained or inspected something unless the context shows it. "
        "If the user asks to delete datasets or artifacts, tell them to use the explicit delete confirmation flow."
    )
    user_prompt = (
        f"User request:\n{command}\n\n"
        f"Latest local job context:\n{_job_context_block(job_id)}\n\n"
        f"Relevant local knowledge:\n{_knowledge_block(command)}"
    )
    return {
        "system": system_prompt,
        "user": user_prompt,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }


async def dispatch_provider_command(command: str, settings: dict[str, Any], job_id: str | None = None) -> str:
    import httpx

    normalized = resolve_agent_policy(settings)
    payload = _build_chat_payload(command, job_id)

    async with httpx.AsyncClient(timeout=30.0) as client:
        for route in model_chain(normalized, capability="chat"):
            provider = route["provider"]
            model = route["model"]
            if not model:
                continue
            profiles = auth_profile_chain(normalized, provider)
            for profile in profiles:
                auth_mode = profile.get("auth_mode")
                if auth_mode in {"browser_login", "oauth"} and not profile.get("api_key"):
                    continue
                base_url = profile.get("base_url") or route["base_url"]
                try:
                    if provider == "ollama":
                        response = await client.post(
                            f"{base_url.rstrip('/')}/api/chat",
                            json={
                                "model": model,
                                "messages": payload["messages"],
                                "stream": False,
                            },
                        )
                        response.raise_for_status()
                        content = response.json().get("message", {}).get("content", "")
                        if content:
                            return f"[{route['ref']} via {profile['label']}]\n{content}"

                    if provider in OPENAI_COMPATIBLE_PROVIDERS:
                        headers = {"Authorization": f"Bearer {profile['api_key']}"} if profile.get("api_key") else {}
                        url = f"{base_url.rstrip('/')}/chat/completions" if not base_url.endswith("/v1") else f"{base_url}/chat/completions"
                        response = await client.post(
                            url,
                            headers=headers,
                            json={
                                "model": model,
                                "messages": payload["messages"],
                                "temperature": 0.2,
                            },
                        )
                        response.raise_for_status()
                        choices = response.json().get("choices", [])
                        if choices:
                            content = choices[0].get("message", {}).get("content", "")
                            if content:
                                return f"[{route['ref']} via {profile['label']}]\n{content}"

                    if provider == "anthropic":
                        headers = {
                            "x-api-key": profile.get("api_key", ""),
                            "anthropic-version": "2023-06-01",
                            "content-type": "application/json",
                        }
                        response = await client.post(
                            f"{base_url.rstrip('/')}/messages",
                            headers=headers,
                            json={
                                "model": model,
                                "system": payload["system"],
                                "max_tokens": 1200,
                                "messages": [{"role": "user", "content": payload["user"]}],
                            },
                        )
                        response.raise_for_status()
                        parts = response.json().get("content", [])
                        content = "\n".join(part.get("text", "") for part in parts if part.get("type") == "text").strip()
                        if content:
                            return f"[{route['ref']} via {profile['label']}]\n{content}"

                    if provider == "google":
                        api_key = profile.get("api_key", "")
                        if not api_key:
                            continue
                        response = await client.post(
                            f"{base_url.rstrip('/')}/v1beta/models/{model}:generateContent",
                            params={"key": api_key},
                            json={
                                "system_instruction": {"parts": [{"text": payload["system"]}]},
                                "contents": [{"role": "user", "parts": [{"text": payload["user"]}]}],
                            },
                        )
                        response.raise_for_status()
                        candidates = response.json().get("candidates", [])
                        for candidate in candidates:
                            parts = candidate.get("content", {}).get("parts", [])
                            content = "\n".join(part.get("text", "") for part in parts if part.get("text")).strip()
                            if content:
                                return f"[{route['ref']} via {profile['label']}]\n{content}"
                except Exception:
                    continue
    return ""
