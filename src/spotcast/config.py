"""Project-wide settings: API URLs, paths, date ranges and the list of spots."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
TIMEZONE = "Europe/Istanbul"
REQUEST_TIMEOUT_SECONDS = 10

# Project root is two folders above this file: src/spotcast/config.py
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
HISTORY_START = date(2024, 1, 1)
HISTORY_END = date(2025, 12, 31)
HISTORY_TABLE = "conditions_history"


@dataclass(frozen=True)
class Spot:
    name: str
    latitude: float
    longitude: float


SPOTS = [
    Spot("Kuşadası Pamucak", 37.94, 27.27),
    Spot("Alaçatı", 38.255, 26.375),
    Spot("Akyaka", 37.04, 28.31),
]
