from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_provider_catalog_endpoint():
    response = client.get("/api/agent/providers")
    assert response.status_code == 200
    assert response.json()["providers"]
    assert "routing_style" in response.json()


def test_agent_policy_endpoint():
    response = client.post(
        "/api/agent/policy",
        json={
            "provider": "ollama",
            "primary_model_ref": "ollama/llama3.2",
            "image_model_ref": "ollama/llava:7b",
            "fallback_model_refs": "lmstudio/qwen2.5",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["chat_chain"]
    assert payload["image_chain"]
