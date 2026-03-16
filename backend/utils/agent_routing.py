from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_PROVIDER_URLS = {
    "ollama": "http://127.0.0.1:11434",
    "lmstudio": "http://127.0.0.1:1234/v1",
    "openai_compatible": "http://127.0.0.1:4000/v1",
    "openai": "https://api.openai.com/v1",
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


def normalize_agent_settings(settings: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(settings or {})
    provider = str(payload.get("provider") or "ollama").strip()
    primary_ref = str(payload.get("primary_model_ref") or "").strip()
    model = str(payload.get("model") or "").strip()

    primary_route = parse_model_ref(primary_ref, default_provider=provider)
    if not primary_route and model:
        primary_route = parse_model_ref(model, default_provider=provider)
    if not primary_route:
        primary_route = ModelRoute(provider=provider, model="")

    fallback_routes = parse_fallback_refs(
        payload.get("fallback_model_refs") or payload.get("fallback_models"),
        default_provider=primary_route.provider,
    )
    provider_urls = payload.get("provider_urls") or {}
    if not isinstance(provider_urls, dict):
        provider_urls = {}

    normalized = {
        "provider": primary_route.provider,
        "model": primary_route.model,
        "primary_model_ref": primary_route.ref,
        "fallback_model_refs": [route.ref for route in fallback_routes],
        "auth_mode": payload.get("auth_mode") or "local",
        "api_key": payload.get("api_key") or "",
        "browser_session_hint": payload.get("browser_session_hint") or "",
        "base_url": payload.get("base_url") or provider_urls.get(primary_route.provider) or DEFAULT_PROVIDER_URLS.get(primary_route.provider, ""),
        "provider_urls": {**DEFAULT_PROVIDER_URLS, **provider_urls},
    }
    return normalized


def model_chain(settings: dict[str, Any] | None) -> list[dict[str, str]]:
    normalized = normalize_agent_settings(settings)
    chain = [parse_model_ref(normalized["primary_model_ref"])]
    chain.extend(parse_fallback_refs(normalized["fallback_model_refs"], default_provider=normalized["provider"]))

    results: list[dict[str, str]] = []
    seen: set[str] = set()
    for route in chain:
        if not route or route.ref in seen:
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
