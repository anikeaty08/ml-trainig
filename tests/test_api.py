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
