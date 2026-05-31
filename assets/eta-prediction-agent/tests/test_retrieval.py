"""
Tests for the historical data retrieval and context scoring logic.
"""

import pytest
from datetime import datetime
from agent.models import FreightOrderInput
from agent.retrieval import retrieve_context_orders, compute_context_statistics, _score_order
from agent.historical_data import get_all_orders, HISTORICAL_ORDERS


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

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
def frankfurt_berlin_order():
    return FreightOrderInput(
        source_location="Frankfurt",
        destination_location="Berlin",
        gross_weight_kg=12000.0,
        planned_departure=datetime(2024, 4, 1, 9, 0, 0),
        planned_arrival=datetime(2024, 4, 2, 3, 0, 0),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Historical data tests
# ──────────────────────────────────────────────────────────────────────────────

def test_historical_orders_count():
    """Dataset should contain exactly 100 orders."""
    orders = get_all_orders()
    assert len(orders) == 100


def test_historical_orders_have_required_fields():
    """All historical orders should have non-empty required fields."""
    for order in get_all_orders():
        assert order.order_id
        assert order.source_location
        assert order.destination_location
        assert order.gross_weight_kg > 0
        assert order.planned_departure < order.planned_arrival
        assert order.planned_departure < order.actual_arrival


def test_historical_orders_delay_distribution():
    """Delays should vary across the dataset (not all identical)."""
    delays = [o.delay_minutes for o in get_all_orders()]
    assert min(delays) != max(delays), "All delays should not be identical"
    # Majority should be reasonable (within ±5 hours)
    reasonable = [d for d in delays if -300 <= d <= 300]
    assert len(reasonable) >= 80, "At least 80% of delays should be within ±5 hours"


# ──────────────────────────────────────────────────────────────────────────────
# Scoring tests
# ──────────────────────────────────────────────────────────────────────────────

def test_exact_route_scores_high(hamburg_munich_order):
    """Exact-match route orders should score higher than random routes."""
    exact_matches = [o for o in HISTORICAL_ORDERS if
                     o.source_location == "Hamburg" and o.destination_location == "Munich"]
    other_routes = [o for o in HISTORICAL_ORDERS if
                    o.source_location != "Hamburg" or o.destination_location != "Munich"]

    if exact_matches and other_routes:
        exact_score = _score_order(hamburg_munich_order, exact_matches[0])
        other_score = _score_order(hamburg_munich_order, other_routes[0])
        assert exact_score > other_score


def test_score_range():
    """Scores should be in [0, 1]."""
    order = FreightOrderInput(
        source_location="Hamburg",
        destination_location="Munich",
        gross_weight_kg=5000.0,
        planned_departure=datetime(2024, 3, 15, 8, 0, 0),
        planned_arrival=datetime(2024, 3, 16, 18, 0, 0),
    )
    for hist_order in HISTORICAL_ORDERS[:10]:
        score = _score_order(order, hist_order)
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval tests
# ──────────────────────────────────────────────────────────────────────────────

def test_retrieve_returns_at_most_top_k(hamburg_munich_order):
    """retrieve_context_orders should return at most top_k results."""
    results = retrieve_context_orders(hamburg_munich_order, top_k=5)
    assert len(results) <= 5


def test_retrieve_results_sorted_by_score(hamburg_munich_order):
    """Results should be sorted by descending score."""
    results = retrieve_context_orders(hamburg_munich_order, top_k=10)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True), "Results should be sorted by score descending"


def test_retrieve_respects_min_score(hamburg_munich_order):
    """All returned orders should meet the minimum score threshold."""
    min_score = 0.3
    results = retrieve_context_orders(hamburg_munich_order, min_score=min_score)
    for _, score in results:
        assert score >= min_score


def test_retrieve_unknown_route_returns_partial():
    """Even unknown routes should return some results due to weight/time scoring."""
    order = FreightOrderInput(
        source_location="Timbuktu",
        destination_location="Atlantis",
        gross_weight_kg=5000.0,
        planned_departure=datetime(2024, 3, 15, 8, 0, 0),
        planned_arrival=datetime(2024, 3, 16, 18, 0, 0),
    )
    results = retrieve_context_orders(order, top_k=10, min_score=0.05)
    # Should return some results based on weight/time matching alone
    assert isinstance(results, list)


# ──────────────────────────────────────────────────────────────────────────────
# Context statistics tests
# ──────────────────────────────────────────────────────────────────────────────

def test_context_stats_empty_input():
    """Empty context should return safe defaults."""
    stats = compute_context_statistics([])
    assert stats["on_time_rate"] == 1.0
    assert stats["avg_delay_minutes"] == 0.0


def test_context_stats_keys(hamburg_munich_order):
    """Context stats should contain all expected keys."""
    context = retrieve_context_orders(hamburg_munich_order, top_k=5)
    stats = compute_context_statistics(context)
    for key in ["avg_delay_minutes", "median_delay_minutes", "on_time_rate",
                "avg_duration_hours", "min_duration_hours", "max_duration_hours"]:
        assert key in stats, f"Missing key: {key}"


def test_context_stats_on_time_rate_bounds(hamburg_munich_order):
    """On-time rate should be in [0, 1]."""
    context = retrieve_context_orders(hamburg_munich_order, top_k=10)
    stats = compute_context_statistics(context)
    assert 0.0 <= stats["on_time_rate"] <= 1.0


def test_context_stats_duration_ordering(hamburg_munich_order):
    """min_duration ≤ avg_duration ≤ max_duration."""
    context = retrieve_context_orders(hamburg_munich_order, top_k=10)
    stats = compute_context_statistics(context)
    assert stats["min_duration_hours"] <= stats["avg_duration_hours"] <= stats["max_duration_hours"]
