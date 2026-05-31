"""
SAP AI Core client for the RPT1.5 (Route Prediction & Transit) model.

Handles:
  - OAuth2 token acquisition (client_credentials flow)
  - Token caching with expiry
  - Inference request construction per RPT1.5 API contract
  - Response parsing and error handling
  - Mock mode for local development without real AI Core credentials
"""

import os
import json
import time
import logging
import random
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import requests
from dotenv import load_dotenv

from models import FreightOrderInput, HistoricalFreightOrder, ETAPredictionResult, DelayRisk

load_dotenv()

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
class AICoreConfig:
    """Reads AI Core configuration from environment variables."""

    def __init__(self):
        self.client_id: str = os.getenv("AI_CORE_CLIENT_ID", "")
        self.client_secret: str = os.getenv("AI_CORE_CLIENT_SECRET", "")
        self.auth_url: str = os.getenv("AI_CORE_AUTH_URL", "").rstrip("/")
        self.base_url: str = os.getenv("AI_CORE_BASE_URL", "").rstrip("/")
        self.resource_group: str = os.getenv("AI_CORE_RESOURCE_GROUP", "default")
        self.deployment_id: str = os.getenv("AI_CORE_DEPLOYMENT_ID", "")
        self.mock_mode: bool = os.getenv("MOCK_AI_CORE", "false").lower() == "true"

    @property
    def is_configured(self) -> bool:
        return bool(
            self.client_id
            and self.client_secret
            and self.auth_url
            and self.base_url
            and self.deployment_id
        )

    def validate(self):
        """Raise ValueError if not fully configured and not in mock mode."""
        if not self.mock_mode and not self.is_configured:
            missing = []
            if not self.client_id:
                missing.append("AI_CORE_CLIENT_ID")
            if not self.client_secret:
                missing.append("AI_CORE_CLIENT_SECRET")
            if not self.auth_url:
                missing.append("AI_CORE_AUTH_URL")
            if not self.base_url:
                missing.append("AI_CORE_BASE_URL")
            if not self.deployment_id:
                missing.append("AI_CORE_DEPLOYMENT_ID")
            raise ValueError(
                f"Missing required AI Core configuration: {', '.join(missing)}. "
                f"Set these in your .env file or use MOCK_AI_CORE=true for testing."
            )


# ──────────────────────────────────────────────────────────────────────────────
# OAuth2 Token Manager
# ──────────────────────────────────────────────────────────────────────────────
class TokenManager:
    """Manages OAuth2 access tokens with automatic refresh."""

    def __init__(self, config: AICoreConfig):
        self._config = config
        self._token: Optional[str] = None
        self._expires_at: float = 0.0

    def get_token(self) -> str:
        """Return a valid access token, fetching a new one if needed."""
        if self._token and time.time() < self._expires_at - 60:
            return self._token

        logger.debug("Fetching new OAuth2 token from AI Core...")
        url = f"{self._config.auth_url}/oauth/token"
        response = requests.post(
            url,
            data={
                "grant_type": "client_credentials",
                "client_id": self._config.client_id,
                "client_secret": self._config.client_secret,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        self._token = data["access_token"]
        self._expires_at = time.time() + data.get("expires_in", 3600)
        logger.debug(f"Token acquired, expires in {data.get('expires_in', 3600)}s")
        return self._token


# ──────────────────────────────────────────────────────────────────────────────
# Payload Builder
# ──────────────────────────────────────────────────────────────────────────────
def _build_rpt_payload(
    query: FreightOrderInput,
    context_orders: List[HistoricalFreightOrder],
    context_stats: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the inference request payload for SAP RPT1.5.

    RPT1.5 accepts:
      - A query object with freight order details
      - A list of context rows (historical orders)
      - Statistical summaries for guided prediction

    Ref: SAP AI Core RPT1.5 API contract
    """
    # Serialize context orders as rows for the model
    context_rows = []
    for order in context_orders:
        context_rows.append({
            "orderId": order.order_id,
            "sourceLocation": order.source_location,
            "destinationLocation": order.destination_location,
            "grossWeightKg": order.gross_weight_kg,
            "plannedDepartureUtc": order.planned_departure.isoformat(),
            "plannedArrivalUtc": order.planned_arrival.isoformat(),
            "actualArrivalUtc": order.actual_arrival.isoformat(),
            "actualDurationHours": order.actual_duration_hours,
            "delayMinutes": order.delay_minutes,
            "carrier": order.carrier,
            "transportMode": order.transport_mode,
            "weatherCondition": order.weather_condition,
            "trafficCondition": order.traffic_condition,
        })

    payload = {
        "model": "sap-rpt",
        "model_version": "1.5",
        "inputs": [
            {
                "role": "system",
                "content": (
                    "You are SAP RPT1.5, a route prediction and transit time model specialized in "
                    "freight logistics. Analyze the query freight order and the provided historical "
                    "context rows to generate a precise ETA prediction with confidence score. "
                    "Return a JSON object with: predicted_arrival_utc, predicted_duration_hours, "
                    "confidence_score (0-1), delay_risk (LOW/MEDIUM/HIGH), reasoning."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({
                    "task": "predict_eta",
                    "query": {
                        "sourceLocation": query.source_location,
                        "destinationLocation": query.destination_location,
                        "grossWeightKg": query.gross_weight_kg,
                        "plannedDepartureUtc": query.planned_departure.isoformat(),
                        "plannedArrivalUtc": query.planned_arrival.isoformat(),
                        "plannedDurationHours": query.planned_duration_hours,
                    },
                    "contextOrders": context_rows,
                    "contextStatistics": context_stats,
                }),
            },
        ],
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 512,
            "response_format": {"type": "json_object"},
        },
    }
    return payload


# ──────────────────────────────────────────────────────────────────────────────
# Response Parser
# ──────────────────────────────────────────────────────────────────────────────
def _parse_rpt_response(
    response_data: Dict[str, Any],
    query: FreightOrderInput,
    context_stats: Dict[str, Any],
) -> ETAPredictionResult:
    """
    Parse the RPT1.5 model response into an ETAPredictionResult.

    RPT1.5 returns a JSON content string within the chat completions format.
    """
    # Extract content from chat completion response
    content_str = (
        response_data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "{}")
    )

    try:
        prediction = json.loads(content_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse RPT1.5 response JSON: {e}\nContent: {content_str}")
        raise ValueError(f"RPT1.5 returned invalid JSON: {e}")

    # Parse predicted arrival
    predicted_arrival_str = prediction.get("predicted_arrival_utc") or prediction.get("predicted_eta")
    if not predicted_arrival_str:
        raise ValueError("RPT1.5 response missing 'predicted_arrival_utc' field")

    predicted_arrival = datetime.fromisoformat(predicted_arrival_str.replace("Z", "+00:00"))
    # Normalize to naive datetime if needed
    if predicted_arrival.tzinfo is not None:
        predicted_arrival = predicted_arrival.replace(tzinfo=None)

    predicted_duration_hours = prediction.get("predicted_duration_hours") or (
        (predicted_arrival - query.planned_departure).total_seconds() / 3600
    )
    confidence_score = float(prediction.get("confidence_score", 0.75))
    delay_risk_str = prediction.get("delay_risk", "MEDIUM").upper()

    try:
        delay_risk = DelayRisk(delay_risk_str)
    except ValueError:
        delay_risk = DelayRisk.MEDIUM

    variance_hours = predicted_duration_hours - query.planned_duration_hours

    # Build recommendation
    recommendation = _build_recommendation(
        variance_hours=variance_hours,
        delay_risk=delay_risk,
        confidence=confidence_score,
        on_time_rate=context_stats.get("on_time_rate", 1.0),
        reasoning=prediction.get("reasoning", ""),
    )

    return ETAPredictionResult(
        predicted_eta=predicted_arrival,
        predicted_duration_hours=round(predicted_duration_hours, 2),
        confidence_score=round(confidence_score, 3),
        planned_duration_hours=query.planned_duration_hours,
        variance_hours=round(variance_hours, 2),
        delay_risk=delay_risk,
        context_orders_count=0,  # filled by caller
        avg_historical_delay_minutes=context_stats.get("avg_delay_minutes", 0.0),
        historical_on_time_rate=context_stats.get("on_time_rate", 1.0),
        model_name="SAP-RPT-1.5",
        model_version="1.5",
        recommendation=recommendation,
    )


def _build_recommendation(
    variance_hours: float,
    delay_risk: DelayRisk,
    confidence: float,
    on_time_rate: float,
    reasoning: str,
) -> str:
    """Generate a human-readable recommendation for the transportation planner."""
    if delay_risk == DelayRisk.LOW and variance_hours <= 0.5:
        action = "Shipment is on track. No action required."
    elif delay_risk == DelayRisk.LOW:
        h = int(variance_hours)
        m = int((variance_hours - h) * 60)
        action = f"Minor delay of ~{h}h {m:02d}m expected. Consider notifying the consignee."
    elif delay_risk == DelayRisk.MEDIUM:
        h = int(abs(variance_hours))
        action = (
            f"Moderate delay of ~{h}h expected. "
            f"Recommend proactive consignee notification and reviewing alternative routes."
        )
    else:  # HIGH
        h = int(abs(variance_hours))
        action = (
            f"Significant delay of ~{h}h expected. "
            f"Immediate consignee notification required. Consider expedited alternatives."
        )

    conf_note = f"(Confidence: {confidence * 100:.0f}%, Historical on-time rate: {on_time_rate * 100:.0f}%)"
    if reasoning:
        return f"{action} {conf_note} — Model insight: {reasoning[:120]}"
    return f"{action} {conf_note}"


# ──────────────────────────────────────────────────────────────────────────────
# Mock Predictor (for local testing without AI Core)
# ──────────────────────────────────────────────────────────────────────────────
def _mock_predict(
    query: FreightOrderInput,
    context_stats: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Simulate RPT1.5 model response for local testing.
    Uses historical context statistics to generate a realistic mock prediction.
    """
    random.seed(abs(hash(query.route_key)) % 1000)

    avg_delay = context_stats.get("avg_delay_minutes", 15.0)
    # Add some randomness around the historical average
    predicted_delay_minutes = avg_delay + random.gauss(0, 20)
    predicted_duration_hours = query.planned_duration_hours + predicted_delay_minutes / 60

    predicted_arrival = query.planned_departure + timedelta(hours=predicted_duration_hours)

    confidence = min(0.95, max(0.55, 0.80 + random.gauss(0, 0.08)))

    on_time_rate = context_stats.get("on_time_rate", 0.75)
    if predicted_delay_minutes > 120:
        delay_risk = "HIGH"
    elif predicted_delay_minutes > 45:
        delay_risk = "MEDIUM"
    else:
        delay_risk = "LOW"

    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "predicted_arrival_utc": predicted_arrival.isoformat(),
                        "predicted_duration_hours": round(predicted_duration_hours, 2),
                        "confidence_score": round(confidence, 3),
                        "delay_risk": delay_risk,
                        "reasoning": (
                            f"Based on {context_stats.get('count', 0)} similar historical orders, "
                            f"average delay is {avg_delay:.0f} minutes with "
                            f"{on_time_rate * 100:.0f}% on-time rate."
                        ),
                    })
                }
            }
        ]
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main AI Core Client
# ──────────────────────────────────────────────────────────────────────────────
class AICoreRPTClient:
    """
    Client for SAP AI Core RPT1.5 model inference.

    Usage:
        client = AICoreRPTClient()
        result = client.predict_eta(query, context_orders, context_stats)
    """

    def __init__(self):
        self.config = AICoreConfig()
        self.config.validate()
        if not self.config.mock_mode:
            self._token_manager = TokenManager(self.config)
        logger.info(
            f"AICoreRPTClient initialized — mock_mode={self.config.mock_mode}, "
            f"resource_group={self.config.resource_group}"
        )

    def predict_eta(
        self,
        query: FreightOrderInput,
        context_orders: List[HistoricalFreightOrder],
        context_stats: Dict[str, Any],
    ) -> ETAPredictionResult:
        """
        Invoke SAP AI Core RPT1.5 model to predict ETA.

        Args:
            query: The freight order to predict ETA for.
            context_orders: Relevant historical orders as context.
            context_stats: Aggregate statistics from context orders.

        Returns:
            ETAPredictionResult with predicted ETA and metadata.
        """
        logger.info(f"Predicting ETA for route: {query.route_key}, weight={query.gross_weight_kg:.0f}kg")

        payload = _build_rpt_payload(query, context_orders, context_stats)

        if self.config.mock_mode:
            logger.info("MOCK MODE: Simulating RPT1.5 response")
            response_data = _mock_predict(query, context_stats)
        else:
            response_data = self._call_ai_core(payload)

        result = _parse_rpt_response(response_data, query, context_stats)
        result.context_orders_count = len(context_orders)
        return result

    def _call_ai_core(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make the actual HTTP call to SAP AI Core inference endpoint."""
        token = self._token_manager.get_token()

        # SAP AI Core inference URL pattern
        url = (
            f"{self.config.base_url}/v2/inference/deployments/"
            f"{self.config.deployment_id}/chat/completions"
        )

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "AI-Resource-Group": self.config.resource_group,
        }

        logger.info(f"POST {url} (resource-group: {self.config.resource_group})")

        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=60,
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error(f"AI Core HTTP error: {e.response.status_code} — {e.response.text}")
            raise RuntimeError(
                f"SAP AI Core returned {e.response.status_code}: {e.response.text[:300]}"
            ) from e
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"Could not connect to SAP AI Core: {e}") from e
        except requests.exceptions.Timeout:
            raise RuntimeError("SAP AI Core request timed out after 60 seconds")

        return response.json()
