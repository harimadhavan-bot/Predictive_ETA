"""
Pydantic data models for the Shipment ETA Prediction Agent.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class DelayRisk(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class FreightOrderInput(BaseModel):
    """Input model for a freight order to predict ETA for."""

    source_location: str = Field(..., description="Origin city or location of the shipment")
    destination_location: str = Field(..., description="Destination city or location of the shipment")
    gross_weight_kg: float = Field(..., gt=0, description="Total gross weight of the shipment in kg")
    planned_departure: datetime = Field(..., description="Planned departure date and time (ISO 8601)")
    planned_arrival: datetime = Field(..., description="Planned arrival date and time (ISO 8601)")

    @field_validator("planned_arrival")
    @classmethod
    def arrival_after_departure(cls, v: datetime, info) -> datetime:
        if "planned_departure" in info.data and v <= info.data["planned_departure"]:
            raise ValueError("planned_arrival must be after planned_departure")
        return v

    @property
    def planned_duration_hours(self) -> float:
        """Planned transit duration in hours."""
        delta = self.planned_arrival - self.planned_departure
        return delta.total_seconds() / 3600

    @property
    def route_key(self) -> str:
        """Normalized route key for matching historical orders."""
        return f"{self.source_location.strip().lower()}→{self.destination_location.strip().lower()}"


class HistoricalFreightOrder(BaseModel):
    """A historical freight order record used as context."""

    order_id: str = Field(..., description="Unique freight order ID")
    source_location: str
    destination_location: str
    gross_weight_kg: float
    planned_departure: datetime
    planned_arrival: datetime
    actual_arrival: datetime
    actual_duration_hours: float
    delay_minutes: int  # negative = early, positive = late
    carrier: str
    transport_mode: str  # ROAD, RAIL, SEA, AIR
    weather_condition: str  # CLEAR, RAIN, SNOW, FOG
    traffic_condition: str  # LOW, MODERATE, HIGH

    @property
    def planned_duration_hours(self) -> float:
        delta = self.planned_arrival - self.planned_departure
        return delta.total_seconds() / 3600

    @property
    def was_on_time(self) -> bool:
        return self.delay_minutes <= 30  # within 30 min tolerance


class ETAPredictionResult(BaseModel):
    """The result of an ETA prediction from SAP AI Core RPT1.5."""

    # Core prediction
    predicted_eta: datetime = Field(..., description="Predicted arrival date and time")
    predicted_duration_hours: float = Field(..., description="Predicted transit duration in hours")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Model confidence (0–1)")

    # Analysis
    planned_duration_hours: float = Field(..., description="Planned transit duration in hours")
    variance_hours: float = Field(..., description="Predicted variance vs planned (positive = delay)")
    delay_risk: DelayRisk = Field(..., description="Delay risk classification")

    # Context used
    context_orders_count: int = Field(..., description="Number of historical orders used as context")
    avg_historical_delay_minutes: float = Field(..., description="Average historical delay for similar routes")
    historical_on_time_rate: float = Field(..., ge=0.0, le=1.0, description="Historical on-time rate (0–1)")

    # Model metadata
    model_name: str = Field(default="SAP-RPT-1.5", description="Model used for prediction")
    model_version: str = Field(default="1.5", description="Model version")

    # Recommendation
    recommendation: str = Field(..., description="Human-readable recommendation for the planner")

    @property
    def confidence_pct(self) -> float:
        return self.confidence_score * 100

    @property
    def on_time_rate_pct(self) -> float:
        return self.historical_on_time_rate * 100

    @property
    def variance_display(self) -> str:
        if self.variance_hours >= 0:
            h = int(self.variance_hours)
            m = int((self.variance_hours - h) * 60)
            return f"+{h}h {m:02d}m (delayed)"
        else:
            h = int(abs(self.variance_hours))
            m = int((abs(self.variance_hours) - h) * 60)
            return f"-{h}h {m:02d}m (early)"
