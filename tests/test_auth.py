"""Tests for API key authentication."""

import os

os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
os.environ.setdefault("SERVICE_API_KEY", "test-service-key")

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.auth import verify_api_key
from app.config import settings


@pytest.fixture
def auth_client():
    """Create a test client with a protected route using verify_api_key."""
    app = FastAPI()

    @app.get("/protected")
    async def protected_route(_: str = Depends(verify_api_key)):
        return {"authenticated": True}

    return TestClient(app)


def test_missing_api_key_returns_401(auth_client):
    """Missing x-api-key header should return 401."""
    response = auth_client.get("/protected")
    assert response.status_code == 401
    assert response.json()["detail"]["error"] == "missing_api_key"
    assert response.json()["detail"]["message"] == "x-api-key header is required"


def test_wrong_api_key_returns_401(auth_client):
    """Invalid x-api-key value should return 401."""
    response = auth_client.get("/protected", headers={"x-api-key": "wrong-key"})
    assert response.status_code == 401
    assert response.json()["detail"]["error"] == "invalid_api_key"
    assert response.json()["detail"]["message"] == "The provided API key is not valid"


def test_correct_api_key_passes_auth(auth_client):
    """Valid x-api-key should pass authentication."""
    response = auth_client.get(
        "/protected",
        headers={"x-api-key": settings.SERVICE_API_KEY},
    )
    assert response.status_code == 200
    assert response.json() == {"authenticated": True}
