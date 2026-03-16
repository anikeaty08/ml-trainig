from backend.utils.agent_routing import model_chain, normalize_agent_settings, parse_model_ref


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
