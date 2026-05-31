"""
Integration tests for the full ETA Prediction Agent pipeline.
Uses MOCK_AI_CORE=true to avoid real AI Core calls.
"""

import os
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

# Force mock mode for all tests
os.environ["MOCK_AI_CORE"] = "true"

from agent.models import FreightOrderInput, DelayRisk
from agent.agent import ETAPredictionAgent, PredictionOutput


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def agent():
    return ETAPredictionAgent(top_k_context=8)


@pytest.fixture
def hamburg_munich_order():
    return FreightOrderInput(
        source_location="Hamburg",
        destination_location="Munich",
        gross_weight_kg=5200.0,
        planned_departure=datetime(2024, 3, 15, 8, 0, 0),
        planned_arrival=datetime(2024, 3, 16, 18, 0, 0),
    )


@pytest.fixture
def heavy_freight_order():
    return FreightOrderInput(
        source_location="Frankfurt",
        destination_location="Berlin",
        gross_weight_kg=22000.0,
        planned_departure=datetime(2024, 4, 10, 6, 0, 0),
        planned_arrival=datetime(2024, 4, 11, 15, 0, 0),
    )


@pytest.fixture
def short_haul_order():
    return FreightOrderInput(
        source_location="Vienna",
        destination_location="Budapest",
        gross_weight_kg=1500.0,
        planned_departure=datetime(2024, 5, 20, 10, 0, 0),
        planned_arrival=datetime(2024, 5, 20, 16, 0, 0),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Model validation tests
# ──────────────────────────────────────────────────────────────────────────────

class TestFreightOrderInput:
    def test_valid_order_creates_successfully(self):
        order = FreightOrderInput(
            source_location="Hamburg",
            destination_location="Munich",
            gross_weight_kg=5000.0,
            planned_departure=datetime(2024, 3, 15, 8, 0, 0),
            planned_arrival=datetime(2024, 3, 16, 18, 0, 0),
        )
        assert order.source_location == "Hamburg"
        assert order.gross_weight_kg == 5000.0

    def test_arrival_before_departure_raises(self):
        with pytest.raises(ValueError, match="planned_arrival must be after planned_departure"):
            FreightOrderInput(
                source_location="Hamburg",
                destination_location="Munich",
                gross_weight_kg=5000.0,
                planned_departure=datetime(2024, 3, 16, 8, 0, 0),
                planned_arrival=datetime(2024, 3, 15, 8, 0, 0),
            )

    def test_zero_weight_raises(self):
        with pytest.raises(ValueError):
            FreightOrderInput(
                source_location="Hamburg",
                destination_location="Munich",
                gross_weight_kg=0.0,
                planned_departure=datetime(2024, 3, 15, 8, 0, 0),
                planned_arrival=datetime(2024, 3, 16, 18, 0, 0),
            )

    def test_planned_duration_hours_computed(self, hamburg_munich_order):
        # 34 hours difference
        assert hamburg_munich_order.planned_duration_hours == pytest.approx(34.0, abs=0.1)

    def test_route_key_normalized(self, hamburg_munich_order):
        assert hamburg_munich_order.route_key == "hamburg→munich"


# ──────────────────────────────────────────────────────────────────────────────
# Agent prediction tests
# ──────────────────────────────────────────────────────────────────────────────

class TestETAPredictionAgent:
    def test_predict_returns_output(self, agent, hamburg_munich_order):
        output = agent.predict(hamburg_munich_order)
        assert isinstance(output, PredictionOutput)
        assert output.result is not None

    def test_predict_eta_is_after_departure(self, agent, hamburg_munich_order):
        output = agent.predict(hamburg_munich_order)
        assert output.result.predicted_eta > hamburg_munich_order.planned_departure

    def test_predict_confidence_in_range(self, agent, hamburg_munich_order):
        output = agent.predict(hamburg_munich_order)
        assert 0.0 <= output.result.confidence_score <= 1.0

    def test_predict_delay_risk_is_valid(self, agent, hamburg_munich_order):
        output = agent.predict(hamburg_munich_order)
        assert output.result.delay_risk in [DelayRisk.LOW, DelayRisk.MEDIUM, DelayRisk.HIGH]

    def test_context_orders_within_top_k(self, agent, hamburg_munich_order):
        output = agent.predict(hamburg_munich_order)
        assert output.result.context_orders_count <= agent.top_k

    def test_predict_heavy_freight(self, agent, heavy_freight_order):
        output = agent.predict(heavy_freight_order)
        assert output.result is not None
        assert output.result.predicted_duration_hours > 0

    def test_predict_short_haul(self, agent, short_haul_order):
        output = agent.predict(short_haul_order)
        assert output.result is not None
        # For short hauls predicted duration should be > 0
        assert output.result.predicted_duration_hours > 0

    def test_to_dict_has_required_keys(self, agent, hamburg_munich_order):
        output = agent.predict(hamburg_munich_order)
        d = output.to_dict()
        assert "input" in d
        assert "prediction" in d
        assert "context" in d
        assert "recommendation" in d

    def test_to_dict_prediction_fields(self, agent, hamburg_munich_order):
        output = agent.predict(hamburg_munich_order)
        pred = output.to_dict()["prediction"]
        for field in ["predictedEta", "predictedDurationHours", "confidenceScore", "delayRisk"]:
            assert field in pred, f"Missing field: {field}"

    def test_recommendation_is_non_empty(self, agent, hamburg_munich_order):
        output = agent.predict(hamburg_munich_order)
        assert output.result.recommendation
        assert len(output.result.recommendation) > 10


# ──────────────────────────────────────────────────────────────────────────────
# Edge case tests
# ──────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_predict_very_heavy_load(self, agent):
        """Test with maximum truck load."""
        order = FreightOrderInput(
            source_location="Hamburg",
            destination_location="Frankfurt",
            gross_weight_kg=24000.0,
            planned_departure=datetime(2024, 6, 1, 6, 0, 0),
            planned_arrival=datetime(2024, 6, 1, 18, 0, 0),
        )
        output = agent.predict(order)
        assert output.result is not None

    def test_predict_unfamiliar_route(self, agent):
        """Agent should still work for routes not in historical data."""
        order = FreightOrderInput(
            source_location="NewCity",
            destination_location="OtherTown",
            gross_weight_kg=3000.0,
            planned_departure=datetime(2024, 7, 15, 9, 0, 0),
            planned_arrival=datetime(2024, 7, 16, 9, 0, 0),
        )
        output = agent.predict(order)
        assert output.result is not None

    def test_predict_overnight_departure(self, agent):
        """Test with overnight departure (edge case for time-of-day scoring)."""
        order = FreightOrderInput(
            source_location="Berlin",
            destination_location="Warsaw",
            gross_weight_kg=8000.0,
            planned_departure=datetime(2024, 8, 20, 23, 30, 0),
            planned_arrival=datetime(2024, 8, 21, 14, 0, 0),
        )
        output = agent.predict(order)
        assert output.result.predicted_eta > order.planned_departure
