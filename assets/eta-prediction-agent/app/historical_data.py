"""
Mock historical freight order dataset with 100 realistic records.
Simulates the kind of historical data a transportation planner would
have access to in an SAP TM / S/4HANA system.
"""

import random
from datetime import datetime, timedelta
from typing import List
from models import HistoricalFreightOrder

# ──────────────────────────────────────────────────────────────────────────────
# Seed for reproducible mock data
# ──────────────────────────────────────────────────────────────────────────────
random.seed(42)

# Common European logistics routes
ROUTES = [
    ("Hamburg", "Munich"),
    ("Hamburg", "Frankfurt"),
    ("Frankfurt", "Berlin"),
    ("Berlin", "Warsaw"),
    ("Munich", "Vienna"),
    ("Vienna", "Budapest"),
    ("Amsterdam", "Brussels"),
    ("Brussels", "Paris"),
    ("Paris", "Lyon"),
    ("Lyon", "Barcelona"),
    ("Frankfurt", "Zurich"),
    ("Zurich", "Milan"),
    ("Milan", "Rome"),
    ("Rome", "Naples"),
    ("Hamburg", "Copenhagen"),
    ("Copenhagen", "Stockholm"),
    ("Stockholm", "Oslo"),
    ("Oslo", "Helsinki"),
    ("Lisbon", "Madrid"),
    ("Madrid", "Valencia"),
]

# Route distances in km (approximate)
ROUTE_DISTANCES_KM = {
    ("Hamburg", "Munich"): 778,
    ("Hamburg", "Frankfurt"): 493,
    ("Frankfurt", "Berlin"): 545,
    ("Berlin", "Warsaw"): 575,
    ("Munich", "Vienna"): 451,
    ("Vienna", "Budapest"): 243,
    ("Amsterdam", "Brussels"): 210,
    ("Brussels", "Paris"): 306,
    ("Paris", "Lyon"): 465,
    ("Lyon", "Barcelona"): 645,
    ("Frankfurt", "Zurich"): 362,
    ("Zurich", "Milan"): 301,
    ("Milan", "Rome"): 572,
    ("Rome", "Naples"): 225,
    ("Hamburg", "Copenhagen"): 482,
    ("Copenhagen", "Stockholm"): 522,
    ("Stockholm", "Oslo"): 529,
    ("Oslo", "Helsinki"): 1164,
    ("Lisbon", "Madrid"): 624,
    ("Madrid", "Valencia"): 358,
}

CARRIERS = ["DHL Freight", "DB Schenker", "Kuehne+Nagel", "DSV", "Geodis", "Rhenus Logistics"]
TRANSPORT_MODES = ["ROAD", "ROAD", "ROAD", "ROAD", "RAIL", "RAIL"]  # weighted toward road
WEATHER_CONDITIONS = ["CLEAR", "CLEAR", "CLEAR", "RAIN", "RAIN", "SNOW", "FOG"]
TRAFFIC_CONDITIONS = ["LOW", "MODERATE", "MODERATE", "HIGH"]


def _base_duration_hours(source: str, destination: str) -> float:
    """Estimate base transit duration based on route distance."""
    key = (source, destination)
    rev_key = (destination, source)
    dist = ROUTE_DISTANCES_KM.get(key) or ROUTE_DISTANCES_KM.get(rev_key)
    if dist:
        # Average truck speed ~80 km/h including rest stops
        return dist / 80
    # Default: 12 hours for unknown routes
    return 12.0


def _delay_minutes(weather: str, traffic: str, weight_kg: float, transport_mode: str) -> int:
    """Calculate realistic delay in minutes based on conditions."""
    base_delay = random.gauss(15, 45)  # baseline: avg 15 min late, std 45 min

    # Weather impact
    weather_impact = {"CLEAR": 0, "RAIN": random.uniform(20, 60), "SNOW": random.uniform(60, 180), "FOG": random.uniform(30, 90)}
    base_delay += weather_impact.get(weather, 0)

    # Traffic impact
    traffic_impact = {"LOW": -10, "MODERATE": random.uniform(10, 30), "HIGH": random.uniform(45, 120)}
    base_delay += traffic_impact.get(traffic, 0)

    # Heavy load impact
    if weight_kg > 15000:
        base_delay += random.uniform(15, 45)

    # Rail is more reliable
    if transport_mode == "RAIL":
        base_delay *= 0.4

    return int(base_delay)


def generate_historical_orders(n: int = 100) -> List[HistoricalFreightOrder]:
    """
    Generate n mock historical freight orders with realistic attributes.

    Returns a list of HistoricalFreightOrder records representing past shipments
    across European logistics routes.
    """
    orders: List[HistoricalFreightOrder] = []

    # Base date: orders from the past 18 months
    base_date = datetime(2023, 1, 1, 6, 0, 0)

    for i in range(n):
        route = random.choice(ROUTES)
        source, destination = route

        # Random departure within the past 18 months
        days_offset = random.randint(0, 540)
        hour_offset = random.choice([6, 7, 8, 9, 10])
        planned_departure = base_date + timedelta(days=days_offset, hours=random.randint(0, 3))
        planned_departure = planned_departure.replace(hour=hour_offset, minute=0, second=0, microsecond=0)

        # Planned duration based on route
        base_duration = _base_duration_hours(source, destination)
        planned_duration_variation = random.uniform(0.9, 1.2)
        planned_duration = base_duration * planned_duration_variation

        planned_arrival = planned_departure + timedelta(hours=planned_duration)

        # Conditions
        weather = random.choice(WEATHER_CONDITIONS)
        traffic = random.choice(TRAFFIC_CONDITIONS)
        carrier = random.choice(CARRIERS)
        transport_mode = random.choice(TRANSPORT_MODES)
        gross_weight = round(random.uniform(500, 24000), 0)

        # Actual arrival: apply delay
        delay_mins = _delay_minutes(weather, traffic, gross_weight, transport_mode)
        actual_arrival = planned_arrival + timedelta(minutes=delay_mins)
        actual_duration = (actual_arrival - planned_departure).total_seconds() / 3600

        order = HistoricalFreightOrder(
            order_id=f"FO-{2023000 + i + 1:07d}",
            source_location=source,
            destination_location=destination,
            gross_weight_kg=gross_weight,
            planned_departure=planned_departure,
            planned_arrival=planned_arrival,
            actual_arrival=actual_arrival,
            actual_duration_hours=round(actual_duration, 2),
            delay_minutes=delay_mins,
            carrier=carrier,
            transport_mode=transport_mode,
            weather_condition=weather,
            traffic_condition=traffic,
        )
        orders.append(order)

    return orders


# ──────────────────────────────────────────────────────────────────────────────
# Singleton dataset loaded once at import
# ──────────────────────────────────────────────────────────────────────────────
HISTORICAL_ORDERS: List[HistoricalFreightOrder] = generate_historical_orders(100)


def get_all_orders() -> List[HistoricalFreightOrder]:
    """Return the full mock historical dataset."""
    return HISTORICAL_ORDERS
