"""Contract tests for the local FastAPI service."""

from fastapi.testclient import TestClient

from api import app


def _client(monkeypatch, tmp_path) -> TestClient:
    import utils.student_profile as profile_module

    monkeypatch.setenv("STUDY_TUTOR_STORAGE", "sqlite")
    monkeypatch.setenv("STUDY_TUTOR_MODE", "demo")
    monkeypatch.setattr(profile_module, "DATA_DIR", str(tmp_path))
    return TestClient(app)


def test_health_describes_local_runtime(monkeypatch, tmp_path):
    response = _client(monkeypatch, tmp_path).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "ai_mode": "demo",
        "storage": "sqlite",
        "scope": "local-only",
    }


def test_profile_contract(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    payload = {
        "name": "API Student",
        "learning_style": "visual",
        "courses": [{"name": "Operating Systems", "difficulty": 4}],
    }

    created = client.post("/api/v1/profiles", json=payload)
    duplicate = client.post("/api/v1/profiles", json=payload)
    loaded = client.get("/api/v1/profiles/API Student")
    summary = client.get("/api/v1/profiles/API Student/summary")

    assert created.status_code == 201
    assert created.json()["learning_style"] == "visual"
    assert duplicate.status_code == 409
    assert loaded.status_code == 200
    assert "Operating Systems" in loaded.json()["courses"]
    assert summary.status_code == 200
    assert summary.json()["total_quizzes"] == 0

    removed = client.delete("/api/v1/profiles/API Student")
    assert removed.status_code == 204
    assert client.get("/api/v1/profiles/API Student").status_code == 404


def test_profile_validation_rejects_invalid_difficulty(monkeypatch, tmp_path):
    response = _client(monkeypatch, tmp_path).post(
        "/api/v1/profiles",
        json={"name": "Bad Input", "courses": [{"name": "Math", "difficulty": 9}]},
    )

    assert response.status_code == 422
