"""Tests for FastAPI endpoints."""

import os

os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
os.environ.setdefault("SERVICE_API_KEY", "test-service-key")

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.models import ActionType, HarvestResponse, OptimizationAction


@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def valid_payload():
    """Return a valid harvest request payload."""
    return {
        "campaign_id": "camp-123",
        "campaign_name": "Summer Sale Campaign",
        "business_context": "Premium kitchen gadgets brand targeting home cooks.",
        "user_constraints": {
            "target_acos": 30.0,
            "max_bid": 2.50,
        },
        "search_terms": [
            {
                "search_term": "silicone spatula",
                "ad_group_id": "ag-001",
                "metrics": {
                    "impressions": 1000,
                    "clicks": 50,
                    "spend": 75.0,
                    "sales": 300.0,
                    "orders": 10,
                    "acos": 25.0,
                    "ctr": 5.0,
                    "cpc": 1.5,
                    "conversion_rate": 20.0,
                    "current_bid": 1.25,
                },
            }
        ],
    }


@pytest.fixture
def mock_harvest_response():
    """Return a mocked HarvestResponse."""
    return HarvestResponse(
        campaign_id="camp-123",
        campaign_name="Summer Sale Campaign",
        total_terms_analyzed=1,
        actions=[
            OptimizationAction(
                search_term="silicone spatula",
                ad_group_id="ag-001",
                action=ActionType.INCREASE_BID,
                recommended_bid=1.45,
                reasoning="ACoS is well below target with strong conversion rate.",
                confidence=0.85,
            )
        ],
        summary="One term recommended for bid increase due to strong performance.",
        model_used="gemini-2.5-flash",
    )


def test_health_endpoint(client):
    """GET /health should return 200 with healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == settings.APP_NAME
    assert data["version"] == settings.APP_VERSION
    assert data["model"] == settings.GEMINI_MODEL


def test_harvest_without_auth(client, valid_payload):
    """POST /api/agent/harvest without auth should return 401."""
    response = client.post("/api/agent/harvest", json=valid_payload)
    assert response.status_code == 401
    assert response.json()["detail"]["error"] == "missing_api_key"


def test_harvest_with_wrong_auth(client, valid_payload):
    """POST /api/agent/harvest with invalid key should return 401."""
    response = client.post(
        "/api/agent/harvest",
        json=valid_payload,
        headers={"x-api-key": "wrong-key"},
    )
    assert response.status_code == 401
    assert response.json()["detail"]["error"] == "invalid_api_key"


@patch("app.main.analyze_search_terms", new_callable=AsyncMock)
def test_harvest_with_valid_auth_and_mock(
    mock_analyze, client, valid_payload, mock_harvest_response
):
    """POST /api/agent/harvest with valid auth and mocked AI should return 200."""
    mock_analyze.return_value = mock_harvest_response

    response = client.post(
        "/api/agent/harvest",
        json=valid_payload,
        headers={"x-api-key": settings.SERVICE_API_KEY},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["campaign_id"] == "camp-123"
    assert data["total_terms_analyzed"] == 1
    assert len(data["actions"]) == 1
    assert data["actions"][0]["action"] == "increase_bid"
    mock_analyze.assert_called_once()


def test_harvest_empty_search_terms(client, valid_payload):
    """POST with empty search_terms should return 422."""
    payload = {**valid_payload, "search_terms": []}
    response = client.post(
        "/api/agent/harvest",
        json=payload,
        headers={"x-api-key": settings.SERVICE_API_KEY},
    )
    assert response.status_code == 422


def test_harvest_invalid_target_acos(client, valid_payload):
    """POST with invalid target_acos should return 422."""
    payload = {
        **valid_payload,
        "user_constraints": {"target_acos": -5, "max_bid": 2.50},
    }
    response = client.post(
        "/api/agent/harvest",
        json=payload,
        headers={"x-api-key": settings.SERVICE_API_KEY},
    )
    assert response.status_code == 422
