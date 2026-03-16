from backend.utils.agent_routing import auth_profile_chain, model_chain, normalize_agent_settings, parse_model_ref, resolve_agent_policy


def test_parse_model_ref_and_chain():
    route = parse_model_ref("ollama/llama3.2")
    assert route is not None
    assert route.provider == "ollama"
    assert route.model == "llama3.2"

    settings = normalize_agent_settings(
        {
            "provider": "ollama",
            "primary_model_ref": "ollama/llama3.2",
            "fallback_model_refs": "lmstudio/qwen2.5,openai/gpt-4.1-mini",
        }
    )
    chain = model_chain(settings)

    assert chain[0]["ref"] == "ollama/llama3.2"
    assert chain[1]["ref"] == "lmstudio/qwen2.5"
    assert chain[2]["ref"] == "openai/gpt-4.1-mini"


def test_policy_supports_image_model_and_auth_profiles():
    settings = resolve_agent_policy(
        {
            "provider": "openai",
            "primary_model_ref": "openai/gpt-4.1-mini",
            "image_model_ref": "openai/gpt-4.1",
            "fallback_model_refs": "ollama/llama3.2",
            "model_allowlist": "openai/gpt-4.1-mini, openai/gpt-4.1, ollama/llama3.2",
            "auth_profiles": [
                {
                    "id": "team-a",
                    "label": "team-a",
                    "provider": "openai",
                    "auth_mode": "api_key",
                    "api_key": "sk-test",
                    "base_url": "https://api.openai.com/v1",
                }
            ],
        }
    )

    assert settings["chat_chain"][0]["ref"] == "openai/gpt-4.1-mini"
    assert settings["image_chain"][0]["ref"] == "openai/gpt-4.1"
    profiles = auth_profile_chain(settings, "openai")
    assert profiles[0]["label"] == "team-a"
