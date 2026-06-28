"""Smoke test for FastAPI server."""

from fastapi.testclient import TestClient
from src.api.server import app

client = TestClient(app)


def test_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
