"""Tests for Pydantic request and response models."""

import os

os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
os.environ.setdefault("SERVICE_API_KEY", "test-service-key")

import pytest
from pydantic import ValidationError

from app.models import (
    HarvestRequest,
    MetricPayload,
    OptimizationAction,
    SearchTermData,
    UserConstraints,
)


def _sample_metric(**overrides) -> MetricPayload:
    """Build a valid MetricPayload with optional overrides."""
    defaults = {
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
    }
    defaults.update(overrides)
    return MetricPayload(**defaults)


def _sample_harvest_request(**overrides) -> HarvestRequest:
    """Build a valid HarvestRequest with optional overrides."""
    defaults = {
        "campaign_id": "camp-123",
        "campaign_name": "Summer Sale Campaign",
        "business_context": "Premium kitchen gadgets brand targeting home cooks.",
        "user_constraints": UserConstraints(target_acos=30.0, max_bid=2.50),
        "search_terms": [
            SearchTermData(
                search_term="silicone spatula",
                ad_group_id="ag-001",
                metrics=_sample_metric(),
            )
        ],
    }
    defaults.update(overrides)
    return HarvestRequest(**defaults)


def test_harvest_request_valid_data():
    """Valid HarvestRequest data should parse successfully."""
    request = _sample_harvest_request()
    assert request.campaign_id == "camp-123"
    assert len(request.search_terms) == 1


def test_user_constraints_target_acos_negative():
    """target_acos below 0 should raise ValidationError."""
    with pytest.raises(ValidationError):
        UserConstraints(target_acos=-5, max_bid=2.0)


def test_user_constraints_target_acos_above_100():
    """target_acos at or above 100 should raise ValidationError."""
    with pytest.raises(ValidationError):
        UserConstraints(target_acos=110, max_bid=2.0)


def test_harvest_request_empty_search_terms():
    """Empty search_terms list should raise ValidationError."""
    with pytest.raises(ValidationError):
        _sample_harvest_request(search_terms=[])


def test_optimization_action_confidence_out_of_range():
    """confidence above 1.0 should raise ValidationError."""
    with pytest.raises(ValidationError):
        OptimizationAction(
            search_term="test term",
            ad_group_id="ag-001",
            action="no_action",
            reasoning="Insufficient data",
            confidence=1.5,
        )


def test_metric_payload_acos_none():
    """acos=None should be valid for zero-sales search terms."""
    metric = _sample_metric(acos=None, sales=0.0, orders=0, conversion_rate=None)
    assert metric.acos is None
