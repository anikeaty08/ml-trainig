from __future__ import annotations

import asyncio
import shlex
import sys
import uuid
from pathlib import Path

from . import database
from .modules.recommender import provider_catalog
from .utils.agent_console import HELP_TEXT as AGENT_HELP_TEXT, dispatch_provider_command, format_policy, list_remote_models, onboard_text, resolve_console_command
from .utils.agent_routing import resolve_agent_policy
from .utils.local_search import search_local_knowledge
from .utils.setup_manager import initialize_setup, install_packs, list_packs, remove_pack, setup_status


CLI_HELP_TEXT = (
    f"{AGENT_HELP_TEXT}\n\n"
    "Terminal-only commands\n"
    "  status\n"
    "  config show\n"
    "  config set provider <provider>\n"
    "  config set primary <provider/model>\n"
    "  config set image <provider/model>\n"
    "  config set fallback <provider/model,provider/model>\n"
    "  config set allow <provider/model,provider/model>\n"
    "  config set auth <local|api_key|browser_login|oauth>\n"
    "  profile add <provider> <label> <auth_mode> <base_url> [api_key]\n"
    "  profile list\n"
    "  profile remove <profile_id>\n"
    "  setup\n"
    "  packs\n"
    "  pack install <pack_id>\n"
    "  pack remove <pack_id>\n"
    "  jobs list\n"
    "  job show <job_id>\n"
    "  delete <job_id>\n"
    "  train <path-to-dataset>\n"
    "  exit\n"
)


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
            format_policy(config),
            f"recent jobs: {len(jobs)}",
        ]
    )


def _model_list_text() -> str:
    return format_policy(_load_config())


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


def _remove_profile(profile_id: str) -> str:
    config = _load_config()
    next_profiles = [item for item in config.get("auth_profiles", []) if item.get("id") != profile_id]
    if len(next_profiles) == len(config.get("auth_profiles", [])):
        return f"Profile not found: {profile_id}"
    next_order = {
        provider: [item for item in ids if item != profile_id]
        for provider, ids in config.get("auth_order", {}).items()
    }
    config["auth_profiles"] = next_profiles
    config["auth_order"] = next_order
    _save_config(config)
    return f"Removed profile {profile_id}."


def _train_dataset(path_text: str) -> str:
    from .jobs.state import JobStateManager
    from .orchestrator import Orchestrator
    from .utils.storage import save_upload

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
    from .utils.storage import delete_job_artifacts

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
    response = await dispatch_provider_command(prompt, config)
    return response or "No provider response. Check your configured model and auth profiles."


def _search(query: str) -> str:
    hits = search_local_knowledge(query)
    if not hits:
        return "No local matches found."
    return "\n\n".join(f"{item['path']}\n{item['snippet']}" for item in hits)


def _set_config(parts: list[str]) -> str:
    if len(parts) < 3:
        return "Usage: config set <provider|primary|image|fallback|allow|auth> <value>"
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
    elif field == "allow":
        config["model_allowlist"] = [item.strip() for item in value.split(",") if item.strip()]
    elif field == "auth":
        config["auth_mode"] = value
    else:
        return f"Unknown config field: {field}"
    saved = _save_config(config)
    return f"Updated config.\nprimary: {saved['primary_model_ref']}\nimage: {saved['image_model_ref']}"


def _scan_models() -> str:
    config = _load_config()
    models = asyncio.run(list_remote_models(config))
    provider = resolve_agent_policy(config)["provider"]
    return "\n".join(f"{provider}/{model_name}" for model_name in models) or f"No remote models found for {provider}."


def _packs_text() -> str:
    packs = list_packs()
    return "\n".join(
        f"{pack['id']} | {'installed' if pack['installed'] else 'not-installed'} | {pack['size_mb']} MB | {pack['label']}"
        for pack in packs
    )


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    raw = input(f"{prompt}{suffix}: ").strip()
    return raw or default


def _pick_single(items: list[dict], *, label_key: str = "label", default_index: int = 1) -> dict:
    for index, item in enumerate(items, start=1):
        label = item.get(label_key) or item.get("id") or str(item)
        extra = item.get("id", "")
        if extra and extra != label:
            _print(f"{index}. {label} ({extra})")
        else:
            _print(f"{index}. {label}")
    raw_choice = _ask("Choose one", str(default_index))
    try:
        chosen_index = max(1, min(len(items), int(raw_choice)))
    except Exception:
        chosen_index = default_index
    return items[chosen_index - 1]


def _pick_multiple_ids(
    options: list[str],
    prompt: str,
    default_values: list[str] | None = None,
    *,
    allow_any: bool = False,
) -> list[str]:
    default_csv = ", ".join(default_values or [])
    raw = _ask(prompt, default_csv).strip()
    selected = [item.strip() for item in raw.split(",") if item.strip()] if raw else list(default_values or [])
    if allow_any:
        return selected
    return [item for item in selected if item in options]


def _build_profile_from_prompt(provider_id: str, auth_mode: str, base_url: str, api_key: str, browser_session_hint: str) -> dict:
    profile_label = _ask("Profile label", f"{provider_id}-default")
    profile_id = profile_label.lower().replace(" ", "-")
    return {
        "id": profile_id,
        "label": profile_label,
        "provider": provider_id,
        "auth_mode": auth_mode,
        "base_url": base_url,
        "api_key": api_key,
        "browser_session_hint": browser_session_hint,
        "enabled": True,
    }


def _configure_agent_provider(selected_provider_ids: list[str]) -> dict:
    providers = provider_catalog()
    provider_map = {provider["id"]: provider for provider in providers}
    available_provider_ids = [provider["id"] for provider in providers]
    primary_provider_id = selected_provider_ids[0] if selected_provider_ids else available_provider_ids[0]
    provider = provider_map[primary_provider_id]

    _print("")
    _print("Agent provider onboarding")
    _print(onboard_text({}, primary_provider_id))
    auth_mode = _pick_single(
        [{"id": mode, "label": mode} for mode in provider.get("auth_modes", []) or ["local"]],
        label_key="label",
        default_index=1,
    )["id"]
    base_url = _ask("Base URL", provider.get("default_base_url") or "")
    api_key = ""
    browser_session_hint = ""
    if auth_mode == "api_key":
        api_key = _ask("API key", "")
    elif auth_mode == "browser_login":
        browser_session_hint = _ask("Browser-login hint", "Use local browser session")

    seed_settings = resolve_agent_policy(
        {
            "provider": primary_provider_id,
            "auth_mode": auth_mode,
            "base_url": base_url,
            "api_key": api_key,
            "browser_session_hint": browser_session_hint,
        }
    )

    remote_models = asyncio.run(list_remote_models(seed_settings))
    suggested_models = provider.get("suggested_models", [])
    if remote_models:
        _print("Visible models:")
        for model_ref in remote_models[:12]:
            _print(f"- {primary_provider_id}/{model_ref}" if "/" not in model_ref else f"- {model_ref}")
    else:
        _print("No live models were discovered. Falling back to suggested refs.")
        for model_ref in suggested_models:
            _print(f"- {model_ref}")

    default_primary = ""
    if remote_models:
        default_primary = remote_models[0] if "/" in remote_models[0] else f"{primary_provider_id}/{remote_models[0]}"
    elif suggested_models:
        default_primary = suggested_models[0]
    primary_model_ref = _ask("Primary chat model ref", default_primary)

    image_suggestion = next((ref for ref in suggested_models if "vision" in ref or "llava" in ref), "")
    image_model_ref = _ask("Image model ref (optional)", image_suggestion)
    fallback_model_refs = _pick_multiple_ids(
        options=[*(suggested_models or []), *([f"{primary_provider_id}/{item}" for item in remote_models] if remote_models else [])],
        prompt="Fallback model refs (comma-separated, optional)",
        default_values=[],
        allow_any=True,
    )

    profile = _build_profile_from_prompt(primary_provider_id, auth_mode, base_url, api_key, browser_session_hint)
    return {
        "provider": primary_provider_id,
        "auth_mode": auth_mode,
        "base_url": base_url,
        "api_key": api_key,
        "browser_session_hint": browser_session_hint,
        "primary_model_ref": primary_model_ref,
        "image_model_ref": image_model_ref,
        "fallback_model_refs": fallback_model_refs,
        "model_allowlist": [],
        "model_catalog": [],
        "auth_profiles": [profile],
        "auth_order": {primary_provider_id: [profile["id"]]},
    }


def _run_setup_wizard() -> str:
    status = setup_status()
    profiles = status["profiles"]
    packs = status["packs"]
    provider_defs = provider_catalog()
    provider_ids = [provider["id"] for provider in provider_defs]

    _print("Local setup wizard")
    _print(f"System recommendation: {status['runtime'].get('recommended_profile', 'core')}")
    selected_profile = _pick_single(
        profiles,
        label_key="label",
        default_index=max(1, next((index for index, profile in enumerate(profiles, start=1) if profile["id"] == status["runtime"].get("recommended_profile")), 1)),
    )
    _print(f"Selected profile: {selected_profile['label']} ({selected_profile['id']})")

    _print("Selected packs in this profile:")
    for pack in packs:
        if pack["id"] in selected_profile["pack_ids"]:
            _print(f"- {pack['id']} ({pack['label']})")

    custom = _ask("Add or remove packs manually? [y/N]", "n").lower()
    selected_pack_ids = list(selected_profile["pack_ids"])
    if custom in {"y", "yes"}:
        _print("Available packs:")
        for pack in packs:
            _print(f"  {pack['id']} - {pack['label']}")
        raw_packs = _ask("Enter comma-separated pack ids", ", ".join(selected_pack_ids)).strip()
        if raw_packs:
            selected_pack_ids = [item.strip() for item in raw_packs.split(",") if item.strip()]

    _print("")
    _print("Primary agent provider")
    selected_provider = _pick_single(provider_defs, label_key="label", default_index=1)
    extra_provider_ids = _pick_multiple_ids(
        provider_ids,
        "Extra providers to track as installed (comma-separated, optional)",
        [],
    )
    selected_provider_ids = [selected_provider["id"], *[item for item in extra_provider_ids if item != selected_provider["id"]]]
    raw_download = _ask("Download selected packs now? [Y/n]", "y").lower()
    download_now = raw_download not in {"n", "no"}

    agent_config = _configure_agent_provider(selected_provider_ids)

    state = initialize_setup(
        profile_id=selected_profile["id"],
        pack_ids=selected_pack_ids,
        provider_ids=selected_provider_ids,
        download_now=download_now,
    )
    saved_config = _save_config(agent_config)
    return (
        "Setup saved.\n"
        f"profile: {state['selected_profile']}\n"
        f"packs: {', '.join(state['selected_pack_ids']) or 'none'}\n"
        f"providers: {', '.join(state['selected_provider_ids']) or 'none'}\n"
        f"primary agent model: {saved_config.get('primary_model_ref') or 'not set'}\n"
        f"image model: {saved_config.get('image_model_ref') or 'not set'}"
    )


def handle_command(line: str) -> str:
    parts = shlex.split(line)
    if not parts:
        return ""
    command = parts[0].lower()

    local_response = asyncio.run(resolve_console_command(line, _load_config()))
    if local_response:
        return local_response

    if command == "help":
        return CLI_HELP_TEXT
    if command == "status":
        return _status_text()
    if command == "setup":
        return _run_setup_wizard()
    if command == "onboard":
        return onboard_text(_load_config(), parts[1] if len(parts) > 1 else None)
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
        if len(parts) > 2 and parts[1] == "remove":
            return _remove_profile(parts[2])
    if command == "model" and len(parts) > 1 and parts[1] == "list":
        return _model_list_text()
    if command == "model" and len(parts) > 1 and parts[1] == "scan":
        return _scan_models()
    if command == "packs":
        return _packs_text()
    if command == "pack" and len(parts) > 2 and parts[1] == "install":
        installed = install_packs([parts[2]])
        return f"Installed packs: {', '.join(installed) or 'none'}"
    if command == "pack" and len(parts) > 2 and parts[1] == "remove":
        state = remove_pack(parts[2])
        return f"Removed {parts[2]}. Remaining installed packs: {', '.join(state.get('installed_pack_ids', [])) or 'none'}"
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
    status = setup_status()
    _print("ML Pipeline Agent Terminal")
    _print("Type help for commands. Natural language prompts go to the configured agent model chain.")
    if status["needs_setup"]:
        _print("No local setup profile has been saved yet. Starting terminal setup wizard...")
        _print(_run_setup_wizard())
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
