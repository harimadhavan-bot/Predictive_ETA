# CRITICAL: Initialize telemetry BEFORE importing AI frameworks
from sap_cloud_sdk.aicore import set_aicore_config
from sap_cloud_sdk.core.telemetry import auto_instrument

set_aicore_config()
auto_instrument()

import logging
import os

import click
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from agent_executor import AgentExecutor
from opentelemetry.instrumentation.starlette import StarletteInstrumentor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "5000"))


@click.command()
@click.option("--host", default=HOST)
@click.option("--port", default=PORT)
def main(host: str, port: int):
    skill = AgentSkill(
        id="eta-prediction-agent",
        name="eta-prediction-agent",
        description=(
            "Predicts the Estimated Time of Arrival (ETA) for a freight shipment. "
            "Provide source location, destination, gross weight (kg), planned departure "
            "and planned arrival dates. The agent retrieves similar historical shipments "
            "and invokes SAP AI Core RPT1.5 to deliver a data-driven ETA with confidence "
            "score and delay risk assessment."
        ),
        tags=["eta", "prediction", "agent"],
        examples=[
            "Predict ETA for a shipment from Hamburg to Munich, 5200 kg, departing 2024-03-15T08:00:00, planned arrival 2024-03-16T18:00:00",
            "What is the predicted arrival time for a freight order from Frankfurt to Berlin, 12000 kg, leaving 2024-04-01T09:00:00, expected by 2024-04-02T03:00:00?",
        ],
    )
    agent_card = AgentCard(
        name="Shipment ETA Prediction Agent",
        description=(
            "AI agent for transportation planners. Predicts freight shipment ETA using "
            "SAP AI Core RPT1.5 model, historical context retrieval, and delay risk analysis."
        ),
        url=os.environ.get("AGENT_PUBLIC_URL", f"http://{host}:{port}/"),
        version="1.0.0",
        default_input_modes=["text", "text/plain"],
        default_output_modes=["text", "text/plain"],
        capabilities=AgentCapabilities(streaming=True, push_notifications=False),
        skills=[skill],
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=DefaultRequestHandler(
            agent_executor=AgentExecutor(),
            task_store=InMemoryTaskStore(),
        ),
    )
    app = server.build()
    StarletteInstrumentor().instrument_app(app)

    logger.info(f"Starting ETA Prediction Agent A2A server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
