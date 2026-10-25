import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine
from sqlalchemy.exc import SQLAlchemyError

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
TIMEZONE = "Europe/Istanbul"
REQUEST_TIMEOUT_SECONDS = 10

# Project root is two folders above this file: src/spotcast/__init__.py
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


def _fetch_hourly(
    url: str,
    latitude: float,
    longitude: float,
    variables: list[str],
    **extra_params: object,
) -> pd.DataFrame:
    """Fetch hourly data from an Open-Meteo API and return it as a DataFrame."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": variables,
        "timezone": TIMEZONE,
    } | extra_params

    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()

    df = pd.DataFrame(response.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def _date_params(start: date | None, end: date | None) -> dict[str, str]:
    """Turn optional start/end dates into Open-Meteo query params.

    Both missing -> {} (forecast). Both given -> start_date/end_date (history).
    """
    if (start is None) != (end is None):
        raise ValueError("Provide both start and end, or neither")
    if start is None or end is None:
        return {}
    return {"start_date": start.isoformat(), "end_date": end.isoformat()}


def fetch_wind(
    latitude: float,
    longitude: float,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Hourly wind speed (knots) and direction (degrees).

    Without dates: 7-day forecast. With dates: historical data.
    """
    dates = _date_params(start, end)
    url = ARCHIVE_URL if dates else FORECAST_URL
    return _fetch_hourly(
        url,
        latitude,
        longitude,
        ["wind_speed_10m", "wind_direction_10m"],
        wind_speed_unit="kn",
        **dates,
    )


def fetch_marine(
    latitude: float,
    longitude: float,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Hourly wave height (m), direction (degrees) and period (s).

    Without dates: 7-day forecast. With dates: historical data.
    """
    return _fetch_hourly(
        MARINE_URL,
        latitude,
        longitude,
        ["wave_height", "wave_direction", "wave_period"],
        **_date_params(start, end),
    )


def fetch_conditions(
    latitude: float,
    longitude: float,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Wind and wave data combined into one table, one row per hour."""
    wind_df = fetch_wind(latitude, longitude, start, end)
    marine_df = fetch_marine(latitude, longitude, start, end)
    return pd.merge(wind_df, marine_df, on="time", how="inner", validate="one_to_one")


def fetch_all_spots(
    spots: list[Spot],
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Conditions for every spot stacked into one table, with a "spot" column."""
    frames = []
    for spot in spots:
        df = fetch_conditions(spot.latitude, spot.longitude, start, end)
        df.insert(0, "spot", spot.name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_history(start: date, end: date, refresh: bool = False) -> pd.DataFrame:
    """Historical conditions for all spots, cached in a local Parquet file.

    The first call fetches from the API and saves the file; later calls read the file.
    Pass refresh=True to ignore the file and fetch again.
    """
    path = DATA_DIR / f"history_{start}_{end}.parquet"

    if path.exists() and not refresh:
        print(f"Loading cached history from {path.name}")
        return pd.read_parquet(path)

    print("Fetching history from the API...")
    df = fetch_all_spots(SPOTS, start, end)
    DATA_DIR.mkdir(exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Saved {len(df)} rows to {path.name}")
    return df


def get_engine() -> Engine:
    """Database connection pool, configured by DATABASE_URL in the .env file."""
    load_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Copy .env.example to .env and fill it in."
        )
    return create_engine(url)


def save_to_db(df: pd.DataFrame, table: str, engine: Engine) -> None:
    """Write the DataFrame to a table, replacing it if it already exists."""
    df.to_sql(table, engine, if_exists="replace", index=False, chunksize=5_000)
    print(f"Saved {len(df)} rows to table {table!r}")


def main() -> None:
    pd.set_option("display.width", None)  # use the full terminal width when printing

    try:
        history = load_history(HISTORY_START, HISTORY_END)
    except requests.RequestException as error:
        print(f"Could not fetch data: {error}")
        return

    try:
        engine = get_engine()
        save_to_db(history, HISTORY_TABLE, engine)
        summary = pd.read_sql(
            f"""
            SELECT spot, COUNT(*) AS hours, ROUND(AVG(wind_speed_10m)::numeric, 2) AS avg_wind_kn
            FROM {HISTORY_TABLE}
            GROUP BY spot
            ORDER BY avg_wind_kn DESC
            """,
            engine,
        )
    except (RuntimeError, SQLAlchemyError) as error:
        print(f"Database error: {error}")
        return

    print(summary)


if __name__ == "__main__":
    main()
