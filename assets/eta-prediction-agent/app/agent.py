import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncGenerator, Literal

from sap_cloud_sdk.agent_decorators import agent_model, prompt_section
from langchain_core.messages import HumanMessage
from langchain_litellm import ChatLiteLLM

from eta_agent import ETAPredictionAgent
from models import FreightOrderInput

logger = logging.getLogger(__name__)


# ── Model & prompt configuration ──────────────────────────────────────────────

@agent_model(
    key="config.model",
    label="LLM Model",
    description="The language model powering this agent",
)
def get_model_name() -> str:
    return "sap/anthropic--claude-4.5-sonnet"


@prompt_section(
    key="prompts.system",
    label="System Prompt",
    description="System prompt for the ETA Prediction Agent",
    validation={"format": "markdown", "max_length": 5000},
)
def get_system_prompt() -> str:
    return """You are a Shipment ETA Prediction Agent for transportation planners.

Your job is to extract freight order details from the user's message and predict the Estimated Time of Arrival (ETA).

When a user provides shipment details, extract:
- source_location: origin city/location
- destination_location: destination city/location  
- gross_weight_kg: weight in kg (numeric)
- planned_departure: ISO datetime (YYYY-MM-DDTHH:MM:SS)
- planned_arrival: ISO datetime (YYYY-MM-DDTHH:MM:SS)

Return ONLY a valid JSON object with these exact keys. Do not include any other text.
Example:
{
  "source_location": "Hamburg",
  "destination_location": "Munich",
  "gross_weight_kg": 5200,
  "planned_departure": "2024-03-15T08:00:00",
  "planned_arrival": "2024-03-16T18:00:00"
}

If any required field is missing, return:
{"error": "missing_fields", "message": "Please provide: <list of missing fields>"}
"""


@dataclass
class AgentResponse:
    status: Literal["input_required", "completed", "error"]
    message: str


class SampleAgent:
    """ETA Prediction Agent using SAP AI Core RPT1.5."""

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self):
        self.llm = ChatLiteLLM(model=get_model_name(), temperature=0.0)
        self._eta_agent = ETAPredictionAgent(top_k_context=10)

    async def stream(
        self,
        query: str,
        session_id: str,
        tools=None,
    ) -> AsyncGenerator[dict, None]:
        """Process an ETA prediction request and yield streaming status updates."""

        yield {"is_task_complete": False, "require_user_input": False,
               "content": "🔍 Extracting freight order details from your request..."}

        # Step 1: Use LLM to extract structured freight order from natural language
        try:
            extraction = await self._extract_freight_order(query)
        except Exception as e:
            logger.exception("LLM extraction failed")
            yield {"is_task_complete": True, "require_user_input": False,
                   "content": f"❌ Failed to parse your request: {e}"}
            return

        # Handle extraction errors / missing fields
        if "error" in extraction:
            yield {
                "is_task_complete": False,
                "require_user_input": True,
                "content": extraction.get("message", "Please provide all required shipment details."),
            }
            return

        # Step 2: Validate and build FreightOrderInput
        try:
            order = FreightOrderInput(
                source_location=extraction["source_location"],
                destination_location=extraction["destination_location"],
                gross_weight_kg=float(extraction["gross_weight_kg"]),
                planned_departure=datetime.fromisoformat(extraction["planned_departure"]),
                planned_arrival=datetime.fromisoformat(extraction["planned_arrival"]),
            )
        except (KeyError, ValueError) as e:
            yield {
                "is_task_complete": False,
                "require_user_input": True,
                "content": (
                    f"⚠️ Could not build a valid freight order: {e}. "
                    "Please double-check the details and try again."
                ),
            }
            return

        yield {"is_task_complete": False, "require_user_input": False,
               "content": f"📦 Freight order parsed: {order.source_location} → {order.destination_location}, "
                          f"{order.gross_weight_kg:,.0f} kg. Retrieving historical context..."}

        # Step 3: Run ETA prediction pipeline
        try:
            output = self._eta_agent.predict(order)
        except Exception as e:
            logger.exception("ETA prediction failed")
            yield {"is_task_complete": True, "require_user_input": False,
                   "content": f"❌ ETA prediction failed: {e}"}
            return

        # Step 4: Format and return the result
        result_text = self._format_result(output)
        yield {"is_task_complete": True, "require_user_input": False, "content": result_text}

    async def _extract_freight_order(self, query: str) -> dict:
        """Use the LLM to extract structured freight order details from free-form text."""
        messages = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": query},
        ]
        response = await self.llm.ainvoke([HumanMessage(content=query)] if not messages else
                                          [m if isinstance(m, HumanMessage) else
                                           HumanMessage(content=m["content"]) for m in messages])

        # Try to parse the LLM response as JSON
        content = response.content if hasattr(response, "content") else str(response)
        # Extract JSON from the response (handles markdown code blocks)
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"error": "parse_failed", "message": "Could not extract freight details from your message."}

    def _format_result(self, output) -> str:
        """Format the prediction output as readable text for the transportation planner."""
        r = output.result
        o = output.input_order
        ctx = output.context_stats

        risk_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(r.delay_risk.value, "⚪")
        conf_emoji = "✅" if r.confidence_pct >= 80 else "⚠️" if r.confidence_pct >= 60 else "❌"

        lines = [
            "═" * 60,
            "  📊 SHIPMENT ETA PREDICTION RESULT",
            "═" * 60,
            "",
            f"  Route         : {o.source_location}  →  {o.destination_location}",
            f"  Gross Weight  : {o.gross_weight_kg:,.0f} kg",
            f"  Planned Dep.  : {o.planned_departure.strftime('%Y-%m-%d %H:%M')}",
            f"  Planned Arr.  : {o.planned_arrival.strftime('%Y-%m-%d %H:%M')}",
            "",
            "─" * 60,
            "",
            f"  🕐 Predicted ETA    : {r.predicted_eta.strftime('%Y-%m-%d %H:%M')}",
            f"  ⏱  Duration         : {r.predicted_duration_hours:.1f}h  (planned: {o.planned_duration_hours:.1f}h)",
            f"  📈 Variance         : {r.variance_display}",
            f"  {conf_emoji} Confidence       : {r.confidence_pct:.1f}%",
            f"  {risk_emoji} Delay Risk       : {r.delay_risk.value}",
            f"  🤖 Model            : {r.model_name} v{r.model_version}",
            "",
            "─" * 60,
            "  📋 Historical Context",
            "─" * 60,
            "",
            f"  Orders Analyzed    : {r.context_orders_count}",
            f"  Avg Hist. Delay    : {'+' if r.avg_historical_delay_minutes > 0 else ''}{r.avg_historical_delay_minutes:.0f} min",
            f"  On-Time Rate       : {r.on_time_rate_pct:.0f}%",
            "",
            "─" * 60,
            "  💡 Recommendation",
            "─" * 60,
            "",
            f"  {r.recommendation}",
            "",
            "═" * 60,
        ]
        return "\n".join(lines)
