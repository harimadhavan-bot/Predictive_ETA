"""
Context retrieval module.

Retrieves the most relevant historical freight orders as context rows
for the ETA prediction. Uses a combination of:
  - Exact route matching (source → destination)
  - Fuzzy route matching (partial city name overlap)
  - Weight similarity scoring
  - Departure time-of-day proximity
"""

import logging
import math
from typing import List, Tuple
from datetime import datetime
from models import FreightOrderInput, HistoricalFreightOrder
from historical_data import get_all_orders

logger = logging.getLogger(__name__)


def _location_similarity(a: str, b: str) -> float:
    """
    Simple location similarity: 1.0 for exact match, 0.6 for prefix match,
    0.3 for substring, 0.0 otherwise.
    """
    a_norm = a.strip().lower()
    b_norm = b.strip().lower()
    if a_norm == b_norm:
        return 1.0
    if a_norm.startswith(b_norm) or b_norm.startswith(a_norm):
        return 0.6
    if b_norm in a_norm or a_norm in b_norm:
        return 0.3
    return 0.0


def _weight_similarity(input_weight: float, hist_weight: float) -> float:
    """
    Weight similarity using Gaussian kernel.
    Returns value in [0, 1]; peaks at 1.0 when weights are identical.
    Sigma = 5000 kg.
    """
    sigma = 5000.0
    return math.exp(-((input_weight - hist_weight) ** 2) / (2 * sigma**2))


def _time_of_day_similarity(input_dt: datetime, hist_dt: datetime) -> float:
    """
    Departure hour similarity. Returns 1.0 if same hour, decays with distance.
    """
    diff_hours = abs(input_dt.hour - hist_dt.hour)
    diff_hours = min(diff_hours, 24 - diff_hours)  # wrap around midnight
    return max(0.0, 1.0 - diff_hours / 12.0)


def _score_order(
    query: FreightOrderInput,
    order: HistoricalFreightOrder,
) -> float:
    """
    Compute a relevance score for a historical order against the query.

    Score breakdown (0–1):
      - Route source match:       30%
      - Route destination match:  30%
      - Weight similarity:        25%
      - Departure time of day:    15%
    """
    src_score = _location_similarity(query.source_location, order.source_location)
    dst_score = _location_similarity(query.destination_location, order.destination_location)
    weight_score = _weight_similarity(query.gross_weight_kg, order.gross_weight_kg)
    time_score = _time_of_day_similarity(query.planned_departure, order.planned_departure)

    total = 0.30 * src_score + 0.30 * dst_score + 0.25 * weight_score + 0.15 * time_score
    return round(total, 4)


def retrieve_context_orders(
    query: FreightOrderInput,
    top_k: int = 10,
    min_score: float = 0.1,
) -> List[Tuple[HistoricalFreightOrder, float]]:
    """
    Retrieve the top-k most relevant historical freight orders for the query.

    Args:
        query: The freight order input to find context for.
        top_k: Maximum number of context rows to return.
        min_score: Minimum relevance score threshold (0–1).

    Returns:
        List of (HistoricalFreightOrder, score) tuples, sorted by score descending.
    """
    all_orders = get_all_orders()
    scored: List[Tuple[HistoricalFreightOrder, float]] = []

    for order in all_orders:
        score = _score_order(query, order)
        if score >= min_score:
            scored.append((order, score))

    # Sort by score descending, then by most recent departure
    scored.sort(key=lambda x: (x[1], x[0].planned_departure), reverse=True)

    top = scored[:top_k]
    logger.info(f"Retrieved {len(top)} context orders (from {len(all_orders)} total, min_score={min_score})")

    for order, score in top:
        logger.debug(f"  [{order.order_id}] {order.source_location}→{order.destination_location} "
                     f"wt={order.gross_weight_kg:.0f}kg score={score:.3f} delay={order.delay_minutes}min")

    return top


def compute_context_statistics(
    context: List[Tuple[HistoricalFreightOrder, float]],
) -> dict:
    """
    Compute aggregate statistics from context orders.

    Returns a dict with:
      - avg_delay_minutes
      - median_delay_minutes
      - on_time_rate
      - avg_duration_hours
      - min_duration_hours
      - max_duration_hours
    """
    if not context:
        return {
            "avg_delay_minutes": 0.0,
            "median_delay_minutes": 0.0,
            "on_time_rate": 1.0,
            "avg_duration_hours": 0.0,
            "min_duration_hours": 0.0,
            "max_duration_hours": 0.0,
        }

    orders = [o for o, _ in context]
    delays = sorted(o.delay_minutes for o in orders)
    durations = [o.actual_duration_hours for o in orders]
    on_time_count = sum(1 for o in orders if o.was_on_time)

    n = len(delays)
    avg_delay = sum(delays) / n
    median_delay = delays[n // 2] if n % 2 == 1 else (delays[n // 2 - 1] + delays[n // 2]) / 2
    on_time_rate = on_time_count / n

    return {
        "avg_delay_minutes": round(avg_delay, 1),
        "median_delay_minutes": round(median_delay, 1),
        "on_time_rate": round(on_time_rate, 3),
        "avg_duration_hours": round(sum(durations) / n, 2),
        "min_duration_hours": round(min(durations), 2),
        "max_duration_hours": round(max(durations), 2),
    }
