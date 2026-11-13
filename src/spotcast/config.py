"""Project-wide settings: API URLs, paths and date ranges.

The locations (spots and places) are data: see locations.toml and locations.py.
"""

from datetime import date
from pathlib import Path

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
TIMEZONE = "Europe/Istanbul"
REQUEST_TIMEOUT_SECONDS = 20  # one request for 29 places takes longer than for one

# Project root is two folders above this file: src/spotcast/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
HISTORY_START = date(2022, 1, 1)  # marine (wave) data starts in 2022
HISTORY_END = date(2026, 10, 24)  # ~1 week back: the newest days are not final yet
HISTORY_TABLE = "conditions_history"
