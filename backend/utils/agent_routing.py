from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_PROVIDER_URLS = {
    "ollama": "http://127.0.0.1:11434",
    "lmstudio": "http://127.0.0.1:1234/v1",
    "openai_compatible": "http://127.0.0.1:4000/v1",
    "openai": "https://api.openai.com/v1",
    "openai-codex": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "google": "https://generativelanguage.googleapis.com",
    "kimi": "https://api.moonshot.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}

LEGACY_DEFAULT_CONFIG = {
    "provider": "ollama",
    "auth_mode": "local",
    "base_url": "http://127.0.0.1:11434",
    "primary_model_ref": "ollama/llama3.2",
    "image_model_ref": "ollama/llava:7b",
    "fallback_model_refs": "",
    "model_allowlist": "",
    "model_catalog": [],
    "auth_profiles": [],
    "auth_order": {},
    "browser_session_hint": "",
    "api_key": "",
}


@dataclass(frozen=True)
class ModelRoute:
    provider: str
    model: str

    @property
    def ref(self) -> str:
        return f"{self.provider}/{self.model}" if self.model else self.provider


def parse_model_ref(model_ref: str, default_provider: str = "") -> ModelRoute | None:
    normalized = (model_ref or "").strip()
    if not normalized:
        return None
    if "/" in normalized:
        provider, model = normalized.split("/", 1)
        return ModelRoute(provider=provider.strip(), model=model.strip())
    if default_provider:
        return ModelRoute(provider=default_provider.strip(), model=normalized)
    return ModelRoute(provider=normalized, model="")


def parse_fallback_refs(raw_value: str | list[str] | None, default_provider: str = "") -> list[ModelRoute]:
    if isinstance(raw_value, list):
        values = raw_value
    else:
        values = [part.strip() for part in (raw_value or "").split(",") if part.strip()]
    routes: list[ModelRoute] = []
    seen: set[str] = set()
    for value in values:
        route = parse_model_ref(value, default_provider=default_provider)
        if route and route.ref not in seen:
            seen.add(route.ref)
            routes.append(route)
    return routes


def _normalize_catalog(raw_catalog: Any) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    if isinstance(raw_catalog, list):
        for item in raw_catalog:
            if not isinstance(item, dict):
                continue
            ref = str(item.get("ref") or "").strip()
            if not ref:
                continue
            catalog.append(
                {
                    "ref": ref,
                    "alias": str(item.get("alias") or "").strip(),
                    "capability": str(item.get("capability") or "chat").strip(),
                    "description": str(item.get("description") or "").strip(),
                }
            )
    return catalog


def _catalog_alias_map(catalog: list[dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in catalog:
        if item["ref"]:
            mapping[item["ref"]] = item["ref"]
        if item["alias"]:
            mapping[item["alias"]] = item["ref"]
    return mapping


def _resolve_catalog_ref(value: str, alias_map: dict[str, str]) -> str:
    normalized = (value or "").strip()
    return alias_map.get(normalized, normalized)


def _normalize_auth_profiles(
    raw_profiles: Any,
    *,
    provider_urls: dict[str, str],
    fallback_api_key: str,
    fallback_auth_mode: str,
    fallback_base_url: str,
    fallback_browser_session_hint: str,
    provider: str,
) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    if isinstance(raw_profiles, list):
        for index, item in enumerate(raw_profiles, start=1):
            if not isinstance(item, dict):
                continue
            profile_provider = str(item.get("provider") or provider).strip()
            label = str(item.get("label") or item.get("id") or f"{profile_provider}-{index}").strip()
            profile_id = str(item.get("id") or label).strip()
            profiles.append(
                {
                    "id": profile_id,
                    "label": label,
                    "provider": profile_provider,
                    "auth_mode": str(item.get("auth_mode") or fallback_auth_mode or "local").strip().replace("oauth", "browser_login"),
                    "base_url": str(item.get("base_url") or provider_urls.get(profile_provider) or "").strip(),
                    "api_key": str(item.get("api_key") or "").strip(),
                    "browser_session_hint": str(item.get("browser_session_hint") or "").strip(),
                    "enabled": bool(item.get("enabled", True)),
                }
            )

    return profiles


def normalize_agent_settings(settings: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(settings or {})
    provider = str(payload.get("provider") or "").strip()
    provider_urls = payload.get("provider_urls") or {}
    if not isinstance(provider_urls, dict):
        provider_urls = {}
    provider_urls = {**DEFAULT_PROVIDER_URLS, **provider_urls}

    raw_catalog = payload.get("model_catalog") or payload.get("models_catalog") or []
    catalog = _normalize_catalog(raw_catalog)
    alias_map = _catalog_alias_map(catalog)

    primary_ref = _resolve_catalog_ref(str(payload.get("primary_model_ref") or payload.get("model") or "").strip(), alias_map)
    primary_route = parse_model_ref(primary_ref, default_provider=provider)
    if primary_route and not provider:
        provider = primary_route.provider

    image_ref = _resolve_catalog_ref(str(payload.get("image_model_ref") or "").strip(), alias_map)
    image_route = parse_model_ref(image_ref, default_provider=provider)

    fallback_routes = parse_fallback_refs(
        [_resolve_catalog_ref(item, alias_map) for item in (payload.get("fallback_model_refs") if isinstance(payload.get("fallback_model_refs"), list) else str(payload.get("fallback_model_refs") or payload.get("fallback_models") or "").split(",")) if str(item).strip()],
        default_provider=provider,
    )

    allowlist = []
    raw_allowlist = payload.get("model_allowlist") or payload.get("allowed_model_refs") or []
    if isinstance(raw_allowlist, list):
        allowlist = [_resolve_catalog_ref(str(item).strip(), alias_map) for item in raw_allowlist if str(item).strip()]
    else:
        allowlist = [_resolve_catalog_ref(part.strip(), alias_map) for part in str(raw_allowlist).split(",") if part.strip()]

    base_url = str(payload.get("base_url") or provider_urls.get(provider) or "").strip()
    auth_mode = str(payload.get("auth_mode") or "local").strip().replace("oauth", "browser_login")
    api_key = str(payload.get("api_key") or "").strip()
    browser_session_hint = str(payload.get("browser_session_hint") or "").strip()

    auth_profiles = _normalize_auth_profiles(
        payload.get("auth_profiles"),
        provider_urls=provider_urls,
        fallback_api_key=api_key,
        fallback_auth_mode=auth_mode,
        fallback_base_url=base_url,
        fallback_browser_session_hint=browser_session_hint,
        provider=provider,
    )

    raw_auth_order = payload.get("auth_order") or {}
    auth_order = raw_auth_order if isinstance(raw_auth_order, dict) else {}

    return {
        "provider": provider,
        "model": primary_route.model if primary_route else "",
        "primary_model_ref": primary_route.ref if primary_route else "",
        "image_model_ref": image_route.ref if image_route else "",
        "fallback_model_refs": [route.ref for route in fallback_routes],
        "model_allowlist": allowlist,
        "model_catalog": catalog,
        "auth_mode": auth_mode,
        "api_key": api_key,
        "browser_session_hint": browser_session_hint,
        "base_url": base_url,
        "provider_urls": provider_urls,
        "auth_profiles": auth_profiles,
        "auth_order": auth_order,
    }


def auth_profile_chain(settings: dict[str, Any] | None, provider: str) -> list[dict[str, Any]]:
    normalized = normalize_agent_settings(settings)
    if not provider:
        return []
    profiles = [profile for profile in normalized["auth_profiles"] if profile["provider"] == provider and profile["enabled"]]
    preferred_ids = normalized["auth_order"].get(provider, []) if isinstance(normalized.get("auth_order"), dict) else []
    if preferred_ids:
        order_map = {profile_id: index for index, profile_id in enumerate(preferred_ids)}
        profiles.sort(key=lambda item: order_map.get(item["id"], len(order_map)))
    if not profiles:
        profiles = [
            {
                "id": f"default-{provider}",
                "label": "default",
                "provider": provider,
                "auth_mode": normalized.get("auth_mode", "local"),
                "base_url": normalized["provider_urls"].get(provider, normalized.get("base_url", "")),
                "api_key": normalized.get("api_key", ""),
                "browser_session_hint": normalized.get("browser_session_hint", ""),
                "enabled": True,
            }
        ]
    return profiles


def model_chain(settings: dict[str, Any] | None, capability: str = "chat") -> list[dict[str, str]]:
    normalized = normalize_agent_settings(settings)
    if not normalized["primary_model_ref"] and capability == "chat":
        return []
    if capability == "image" and not normalized["image_model_ref"]:
        return []
    alias_map = _catalog_alias_map(normalized["model_catalog"])
    primary_ref = normalized["image_model_ref"] if capability == "image" and normalized["image_model_ref"] else normalized["primary_model_ref"]
    chain = [parse_model_ref(_resolve_catalog_ref(primary_ref, alias_map), default_provider=normalized["provider"])]
    chain.extend(parse_fallback_refs(normalized["fallback_model_refs"], default_provider=normalized["provider"]))

    results: list[dict[str, str]] = []
    seen: set[str] = set()
    allowlist = set(normalized["model_allowlist"])
    for route in chain:
        if not route or route.ref in seen or not route.model:
            continue
        if allowlist and route.ref not in allowlist:
            continue
        seen.add(route.ref)
        results.append(
            {
                "provider": route.provider,
                "model": route.model,
                "ref": route.ref,
                "base_url": normalized["provider_urls"].get(route.provider, normalized["base_url"]),
            }
        )
    return results


def resolve_agent_policy(settings: dict[str, Any] | None) -> dict[str, Any]:
    normalized = normalize_agent_settings(settings)
    return {
        **normalized,
        "chat_chain": model_chain(normalized, capability="chat"),
        "image_chain": model_chain(normalized, capability="image"),
        "auth_summary": {
            provider: [profile["label"] for profile in auth_profile_chain(normalized, provider)]
            for provider in sorted({profile["provider"] for profile in normalized["auth_profiles"]})
        },
    }
