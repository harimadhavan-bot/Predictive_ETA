"""
Core agent orchestration module.

The ETAPredictionAgent ties together:
  1. Input validation (FreightOrderInput)
  2. Historical context retrieval (retrieval.py)
  3. SAP AI Core RPT1.5 inference (ai_core_client.py)
  4. Result assembly and presentation
"""

import logging
from datetime import datetime
from typing import Optional, List, Tuple

from models import FreightOrderInput, HistoricalFreightOrder, ETAPredictionResult
from retrieval import retrieve_context_orders, compute_context_statistics
from ai_core_client import AICoreRPTClient

logger = logging.getLogger(__name__)


class ETAPredictionAgent:
    """
    Shipment ETA Prediction Agent.

    Orchestrates the full prediction pipeline:
      1. Validate freight order input
      2. Retrieve relevant historical orders (context)
      3. Compute context statistics
      4. Invoke RPT1.5 model on SAP AI Core
      5. Return structured prediction result
    """

    def __init__(self, top_k_context: int = 10):
        """
        Initialize the agent.

        Args:
            top_k_context: Max number of historical orders to use as context.
        """
        self.top_k = top_k_context
        self._client: Optional[AICoreRPTClient] = None
        logger.info(f"ETAPredictionAgent initialized (top_k={self.top_k})")

    @property
    def client(self) -> AICoreRPTClient:
        """Lazy-init the AI Core client."""
        if self._client is None:
            self._client = AICoreRPTClient()
        return self._client

    def predict(self, order: FreightOrderInput) -> "PredictionOutput":
        """
        Run the full ETA prediction pipeline for a freight order.

        Args:
            order: Validated FreightOrderInput with shipment details.

        Returns:
            PredictionOutput containing the result + context used.

        Raises:
            ValueError: If input validation fails.
            RuntimeError: If the AI Core call fails.
        """
        logger.info(f"Starting ETA prediction for: {order.source_location} → {order.destination_location}")

        # Step 1: Retrieve relevant historical context
        logger.info("Step 1/3 — Retrieving historical context...")
        context_with_scores = retrieve_context_orders(order, top_k=self.top_k)
        context_orders = [o for o, _ in context_with_scores]

        # Step 2: Compute context statistics
        context_stats = compute_context_statistics(context_with_scores)
        context_stats["count"] = len(context_orders)
        logger.info(
            f"Context: {len(context_orders)} orders, "
            f"avg_delay={context_stats['avg_delay_minutes']:.0f}min, "
            f"on_time_rate={context_stats['on_time_rate']:.0%}"
        )

        # Step 3: Invoke SAP AI Core RPT1.5
        logger.info("Step 2/3 — Invoking SAP AI Core RPT1.5...")
        result = self.client.predict_eta(order, context_orders, context_stats)

        logger.info(
            f"Step 3/3 — Prediction complete: ETA={result.predicted_eta}, "
            f"confidence={result.confidence_pct:.1f}%, risk={result.delay_risk}"
        )

        return PredictionOutput(
            input_order=order,
            result=result,
            context_orders=context_with_scores,
            context_stats=context_stats,
        )


class PredictionOutput:
    """Container for the full prediction output including context."""

    def __init__(
        self,
        input_order: FreightOrderInput,
        result: ETAPredictionResult,
        context_orders: List[Tuple[HistoricalFreightOrder, float]],
        context_stats: dict,
    ):
        self.input_order = input_order
        self.result = result
        self.context_orders = context_orders
        self.context_stats = context_stats

    def to_dict(self) -> dict:
        """Serialize the full output to a plain dict (for JSON responses)."""
        return {
            "input": {
                "sourceLocation": self.input_order.source_location,
                "destinationLocation": self.input_order.destination_location,
                "grossWeightKg": self.input_order.gross_weight_kg,
                "plannedDeparture": self.input_order.planned_departure.isoformat(),
                "plannedArrival": self.input_order.planned_arrival.isoformat(),
                "plannedDurationHours": round(self.input_order.planned_duration_hours, 2),
            },
            "prediction": {
                "predictedEta": self.result.predicted_eta.isoformat(),
                "predictedDurationHours": self.result.predicted_duration_hours,
                "confidenceScore": self.result.confidence_score,
                "confidencePct": round(self.result.confidence_pct, 1),
                "varianceHours": self.result.variance_hours,
                "varianceDisplay": self.result.variance_display,
                "delayRisk": self.result.delay_risk.value,
                "modelName": self.result.model_name,
                "modelVersion": self.result.model_version,
            },
            "context": {
                "ordersCount": self.result.context_orders_count,
                "avgDelayMinutes": self.context_stats.get("avg_delay_minutes"),
                "onTimeRate": self.context_stats.get("on_time_rate"),
                "onTimeRatePct": round(self.context_stats.get("on_time_rate", 0) * 100, 1),
                "avgDurationHours": self.context_stats.get("avg_duration_hours"),
            },
            "recommendation": self.result.recommendation,
        }
