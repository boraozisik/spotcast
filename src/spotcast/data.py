"""Fetching wind and wave data from Open-Meteo, and caching history as Parquet."""

import time
from datetime import date

import pandas as pd
import requests

from spotcast.config import (
    ARCHIVE_URL,
    DATA_DIR,
    FORECAST_URL,
    MARINE_URL,
    REQUEST_TIMEOUT_SECONDS,
    SPOTS,
    TIMEZONE,
    Spot,
)


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

    response = _get_with_retries(url, params)

    df = pd.DataFrame(response.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


RETRIES = 3


def _get_with_retries(
    url: str, params: dict, retries: int = RETRIES
) -> requests.Response:
    """A slow or dropped connection is often a one-off: wait a little and try again
    (1 s, then 2 s). Errors the server returns on purpose (HTTP 400) are not retried."""
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response
        except (requests.ConnectionError, requests.Timeout):
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


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
    """Hourly wind speed (knots), direction (degrees) and daylight (is_day).

    Without dates: 7-day forecast. With dates: historical data.
    """
    dates = _date_params(start, end)
    url = ARCHIVE_URL if dates else FORECAST_URL
    df = _fetch_hourly(
        url,
        latitude,
        longitude,
        ["wind_speed_10m", "wind_direction_10m", "is_day"],
        wind_speed_unit="kn",
        **dates,
    )
    df["is_day"] = df["is_day"].astype(bool)  # API sends 1/0
    return df


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
