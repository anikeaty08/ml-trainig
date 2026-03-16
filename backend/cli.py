from __future__ import annotations

import asyncio
import shlex
import shutil
import sys
import uuid
from pathlib import Path

from . import database
from .main import _dispatch_provider_command
from .orchestrator import Orchestrator
from .jobs.state import JobStateManager
from .utils.agent_routing import resolve_agent_policy
from .utils.local_search import search_local_knowledge
from .utils.storage import delete_job_artifacts, save_upload


HELP_TEXT = """
Commands
  help
  status
  config show
  config set provider <provider>
  config set primary <provider/model>
  config set image <provider/model>
  config set fallback <provider/model,provider/model>
  profile add <provider> <label> <auth_mode> <base_url> [api_key]
  profile list
  model list
  jobs list
  job show <job_id>
  delete <job_id>
  search <query>
  train <path-to-dataset>
  exit

Any other text is sent to the configured agent model chain as a chat prompt.
""".strip()


def _print(text: str) -> None:
    sys.stdout.write(f"{text}\n")
    sys.stdout.flush()


def _load_config() -> dict:
    return resolve_agent_policy(database.get_agent_config())


def _save_config(config: dict) -> dict:
    normalized = resolve_agent_policy(config)
    database.save_agent_config(normalized)
    return normalized


def _status_text() -> str:
    config = _load_config()
    jobs = database.list_jobs(limit=5)
    return "\n".join(
        [
            f"primary: {config.get('primary_model_ref') or 'unset'}",
            f"image: {config.get('image_model_ref') or 'unset'}",
            f"fallbacks: {', '.join(config.get('fallback_model_refs') or []) or 'none'}",
            f"recent jobs: {len(jobs)}",
        ]
    )


def _model_list_text() -> str:
    config = _load_config()
    catalog_lines = [
        f"{item.get('alias') or item['ref']} -> {item['ref']} [{item.get('capability', 'chat')}]"
        for item in config.get("model_catalog", [])
    ]
    chain_lines = [
        f"chat: {' -> '.join(route['ref'] for route in config.get('chat_chain', [])) or 'none'}",
        f"image: {' -> '.join(route['ref'] for route in config.get('image_chain', [])) or 'none'}",
    ]
    return "\n".join(chain_lines + ["catalog:"] + (catalog_lines or ["none"]))


def _add_profile(args: list[str]) -> str:
    if len(args) < 4:
        return "Usage: profile add <provider> <label> <auth_mode> <base_url> [api_key]"
    provider, label, auth_mode, base_url = args[:4]
    api_key = args[4] if len(args) > 4 else ""
    config = _load_config()
    profile_id = f"{provider}-{label}".lower().replace(" ", "-")
    next_profiles = [item for item in config.get("auth_profiles", []) if item.get("id") != profile_id]
    next_profiles.append(
        {
            "id": profile_id,
            "label": label,
            "provider": provider,
            "auth_mode": auth_mode,
            "base_url": base_url,
            "api_key": api_key,
            "enabled": True,
        }
    )
    config["auth_profiles"] = next_profiles
    config["auth_order"] = {
        **config.get("auth_order", {}),
        provider: [*config.get("auth_order", {}).get(provider, []), profile_id],
    }
    _save_config(config)
    return f"Saved profile {label} for {provider}."


def _profile_list_text() -> str:
    config = _load_config()
    profiles = config.get("auth_profiles", [])
    if not profiles:
        return "No saved auth profiles."
    return "\n".join(
        f"{item['label']} | {item['provider']} | {item['auth_mode']} | {item.get('base_url') or 'default'}"
        for item in profiles
    )


def _train_dataset(path_text: str) -> str:
    source_path = Path(path_text).expanduser().resolve()
    if not source_path.exists():
        return f"Dataset not found: {source_path}"

    database.init_database()
    state_manager = JobStateManager()
    orchestrator = Orchestrator(state_manager)
    job_id = uuid.uuid4().hex[:12]
    upload_path = save_upload(job_id, source_path.name, source_path.read_bytes())
    config = _load_config()
    database.create_job(job_id, source_path.name, str(upload_path), config.get("provider", ""), config.get("model", ""))
    orchestrator.process_job(job_id, str(upload_path))

    job = database.get_job(job_id)
    result = database.get_job_result(job_id)
    if not job:
        return f"Training failed for {source_path.name}"
    if job.get("status") == "failed":
        return f"Training failed: {job.get('error_message') or job.get('message')}"
    winner = result["comparison_json"][0]["name"] if result and result.get("comparison_json") else "unknown"
    metrics = result["metrics_json"] if result else {}
    return f"Training complete.\njob_id: {job_id}\nwinner: {winner}\nmetrics: {metrics}"


def _delete_job(job_id: str) -> str:
    job = database.get_job(job_id)
    if not job:
        return f"Job not found: {job_id}"
    confirmation = input(f"Delete dataset and artifacts for {job_id} ({job['filename']})? [y/N]: ").strip().lower()
    if confirmation not in {"y", "yes"}:
        return "Deletion cancelled."
    delete_job_artifacts(job)
    database.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    database.execute("DELETE FROM job_logs WHERE job_id = ?", (job_id,))
    database.execute("DELETE FROM job_results WHERE job_id = ?", (job_id,))
    return f"Deleted {job_id}."


def _job_show(job_id: str) -> str:
    job = database.get_job(job_id)
    if not job:
        return f"Job not found: {job_id}"
    result = database.get_job_result(job_id)
    lines = [f"{job['id']} | {job['status']} | {job['filename']} | {job['dataset_type']} | {job['problem_type']}"]
    if result:
        lines.append(f"winner: {result['comparison_json'][0]['name'] if result['comparison_json'] else 'unknown'}")
        lines.append(f"metrics: {result['metrics_json']}")
    return "\n".join(lines)


async def _chat(prompt: str) -> str:
    config = _load_config()
    response = await _dispatch_provider_command(prompt, config)
    return response or "No provider response. Check your configured model and auth profiles."


def _search(query: str) -> str:
    hits = search_local_knowledge(query)
    if not hits:
        return "No local matches found."
    return "\n\n".join(f"{item['path']}\n{item['snippet']}" for item in hits)


def _set_config(parts: list[str]) -> str:
    if len(parts) < 3:
        return "Usage: config set <provider|primary|image|fallback> <value>"
    field = parts[1]
    value = " ".join(parts[2:])
    config = _load_config()
    if field == "provider":
        config["provider"] = value
    elif field == "primary":
        config["primary_model_ref"] = value
    elif field == "image":
        config["image_model_ref"] = value
    elif field == "fallback":
        config["fallback_model_refs"] = [item.strip() for item in value.split(",") if item.strip()]
    else:
        return f"Unknown config field: {field}"
    saved = _save_config(config)
    return f"Updated config.\nprimary: {saved['primary_model_ref']}\nimage: {saved['image_model_ref']}"


def handle_command(line: str) -> str:
    parts = shlex.split(line)
    if not parts:
        return ""
    command = parts[0].lower()

    if command == "help":
        return HELP_TEXT
    if command == "status":
        return _status_text()
    if command == "config":
        if len(parts) == 1 or parts[1] == "show":
            return _model_list_text()
        if parts[1] == "set":
            return _set_config(parts[1:])
    if command == "profile":
        if len(parts) > 1 and parts[1] == "list":
            return _profile_list_text()
        if len(parts) > 1 and parts[1] == "add":
            return _add_profile(parts[2:])
    if command == "model" and len(parts) > 1 and parts[1] == "list":
        return _model_list_text()
    if command == "jobs" and len(parts) > 1 and parts[1] == "list":
        jobs = database.list_jobs(limit=20)
        return "\n".join(f"{job['id']} | {job['status']} | {job['filename']}" for job in jobs) or "No jobs yet."
    if command == "job" and len(parts) > 2 and parts[1] == "show":
        return _job_show(parts[2])
    if command == "delete" and len(parts) > 1:
        return _delete_job(parts[1])
    if command == "search" and len(parts) > 1:
        return _search(" ".join(parts[1:]))
    if command == "train" and len(parts) > 1:
        return _train_dataset(" ".join(parts[1:]))
    return asyncio.run(_chat(line))


def repl() -> None:
    database.init_database()
    _print("ML Pipeline Agent Terminal")
    _print("Type help for commands. Natural language prompts go to the configured agent model chain.")
    while True:
        try:
            line = input("ml-agent> ").strip()
        except (EOFError, KeyboardInterrupt):
            _print("\nBye.")
            return
        if not line:
            continue
        if line.lower() in {"exit", "quit"}:
            _print("Bye.")
            return
        output = handle_command(line)
        if output:
            _print(output)


def main() -> None:
    if len(sys.argv) > 1:
        output = handle_command(" ".join(shlex.quote(arg) for arg in sys.argv[1:]))
        if output:
            _print(output)
        return
    repl()


if __name__ == "__main__":
    main()
