"""Fetching wind and wave data from Open-Meteo, and caching history as Parquet."""

import hashlib
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
    TIMEZONE,
)
from spotcast.locations import SPOTS, Location
from spotcast.shore import with_shore_facing

# ---------- Open-Meteo's free-use limits ----------
# 600 calls a minute, 5,000 an hour, 10,000 a day. A request counts as more than one call
# when it covers several locations, more than 10 variables or more than 2 weeks: 5 years of
# hourly history for one place is about 126 calls. Live forecasts are cheap (1 per place);
# a training history is not, so requests are sized and paced to stay under the limits.

CALLS_PER_MINUTE = 500  # Open-Meteo allows 600; keep a margin
MAX_CALLS_PER_REQUEST = 250  # bigger history requests are split by location
FORECAST_DAYS = 7  # what the forecast endpoints return without dates


def call_weight(locations: int, variables: int, days: int) -> float:
    """How many API calls Open-Meteo counts for one request (its published rule)."""
    return locations * max(1.0, variables / 10) * max(1.0, days / 14)


class RateLimiter:
    """Keeps the calls of the last 60 seconds under a budget, waiting when needed.

    Like a token bucket: every request "spends" its weight, and spending more than the
    budget within a minute means waiting until old spending is a minute old.
    """

    def __init__(self, budget: float, window: float = 60.0) -> None:
        self.budget, self.window = budget, window
        self.spent: list[tuple[float, float]] = []  # (time, weight)

    def wait(self, weight: float) -> None:
        while True:
            now = time.monotonic()
            self.spent = [(t, w) for t, w in self.spent if now - t < self.window]
            used = sum(w for _, w in self.spent)
            if not self.spent or used + weight <= self.budget:
                self.spent.append((now, weight))
                return
            time.sleep(self.window - (now - self.spent[0][0]) + 0.1)


LIMITER = RateLimiter(CALLS_PER_MINUTE)


def _fetch_hourly(
    url: str,
    points: list[tuple[float, float]],
    variables: list[str],
    **extra_params: object,
) -> list[pd.DataFrame]:
    """Hourly data for one or more points in ONE request; one DataFrame per point."""
    params = {
        # Several locations: comma-separated, and the answer becomes a list
        "latitude": ",".join(str(lat) for lat, _ in points),
        "longitude": ",".join(str(lon) for _, lon in points),
        "hourly": variables,
        "timezone": TIMEZONE,
    } | extra_params
    days = (
        (
            date.fromisoformat(params["end_date"])
            - date.fromisoformat(params["start_date"])
        ).days
        + 1
        if "start_date" in params
        else FORECAST_DAYS
    )
    LIMITER.wait(call_weight(len(points), len(variables), days))

    answer = _get_with_retries(url, params).json()
    frames = []
    for item in answer if isinstance(answer, list) else [answer]:
        df = pd.DataFrame(item["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        frames.append(df)
    return frames


RETRIES = 3
RATE_LIMIT_WAIT_SECONDS = 60


def _get_with_retries(
    url: str, params: dict, retries: int = RETRIES
) -> requests.Response:
    """A slow or dropped connection is often a one-off: wait a little and try again
    (1 s, then 2 s). "Too many requests" (HTTP 429): wait a minute and try again. Other
    errors the server returns on purpose (HTTP 400) are not retried."""
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            if response.status_code == 429 and attempt < retries - 1:
                time.sleep(RATE_LIMIT_WAIT_SECONDS)
                continue
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


WIND_VARIABLES = ["wind_speed_10m", "wind_direction_10m", "is_day"]
MARINE_VARIABLES = ["wave_height", "wave_direction", "wave_period"]


def _wind_many(points, start=None, end=None) -> list[pd.DataFrame]:
    dates = _date_params(start, end)
    url = ARCHIVE_URL if dates else FORECAST_URL
    frames = _fetch_hourly(url, points, WIND_VARIABLES, wind_speed_unit="kn", **dates)
    for df in frames:
        df["is_day"] = df["is_day"].astype(bool)  # API sends 1/0
    return frames


def _marine_many(points, start=None, end=None) -> list[pd.DataFrame]:
    return _fetch_hourly(
        MARINE_URL, points, MARINE_VARIABLES, **_date_params(start, end)
    )


def fetch_wind(
    latitude: float,
    longitude: float,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Hourly wind speed (knots), direction (degrees) and daylight (is_day).

    Without dates: 7-day forecast. With dates: historical data.
    """
    return _wind_many([(latitude, longitude)], start, end)[0]


def fetch_marine(
    latitude: float,
    longitude: float,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Hourly wave height (m), direction (degrees) and period (s).

    Without dates: 7-day forecast. With dates: historical data.
    """
    return _marine_many([(latitude, longitude)], start, end)[0]


def _merge(wind_df: pd.DataFrame, marine_df: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(wind_df, marine_df, on="time", how="inner", validate="one_to_one")


def fetch_conditions(
    latitude: float,
    longitude: float,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Wind and wave data combined into one table, one row per hour."""
    return _merge(
        fetch_wind(latitude, longitude, start, end),
        fetch_marine(latitude, longitude, start, end),
    )


def batches(
    locations: list[Location], start: date | None, end: date | None
) -> list[list[Location]]:
    """Groups of locations small enough for one request each (see MAX_CALLS_PER_REQUEST).
    A 7-day forecast for all 29 places is one group; 5 years of history is 1 per group."""
    days = (end - start).days + 1 if start and end else FORECAST_DAYS
    per_location = call_weight(1, len(WIND_VARIABLES), days)
    size = max(1, int(MAX_CALLS_PER_REQUEST // per_location))
    return [locations[i : i + size] for i in range(0, len(locations), size)]


def fetch_all_spots(
    spots: list[Location],
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Conditions for every location stacked into one table, with a "spot" column.

    Locations share requests: 29 places cost 2 requests (wind + waves), not 58.
    """
    frames = []
    for group in batches(spots, start, end):
        points = [(loc.latitude, loc.longitude) for loc in group]
        winds, waves = _wind_many(points, start, end), _marine_many(points, start, end)
        for loc, wind_df, marine_df in zip(group, winds, waves, strict=True):
            df = _merge(wind_df, marine_df)
            df.insert(0, "spot", loc.name)
            df["shore_facing"] = (
                loc.shore_facing
            )  # for the offshore rule (None: unknown)
            df["sheltered"] = loc.sheltered
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def spots_fingerprint(spots: list[Location]) -> str:
    """A short code that changes when a spot is added, removed or moved. It is part of the
    cache file name, so history fetched for old coordinates is never used by mistake."""
    text = ";".join(f"{s.name}:{s.latitude}:{s.longitude}" for s in spots)
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def load_history(
    start: date,
    end: date,
    refresh: bool = False,
    locations: list[Location] | None = None,
) -> pd.DataFrame:
    """Historical conditions, cached in a local Parquet file per set of locations.

    Default: the three guided spots (the Stage 3 wind models are built for them). The
    suitability model passes TRAINING_LOCATIONS. The first call fetches from the API and
    saves the file; later calls read it. refresh=True ignores the file.
    """
    locations = SPOTS if locations is None else locations
    path = DATA_DIR / f"history_{start}_{end}_{spots_fingerprint(locations)}.parquet"

    if path.exists() and not refresh:
        print(f"Loading cached history from {path.name}")
        df = pd.read_parquet(path)
    else:
        print("Fetching history from the API...")
        df = fetch_all_spots(locations, start, end)
        DATA_DIR.mkdir(exist_ok=True)
        df.to_parquet(path, index=False)
        print(f"Saved {len(df)} rows to {path.name}")
    # The shore direction always comes from today's locations.toml, never from the cache
    return with_shore_facing(
        df.drop(columns=["shore_facing", "sheltered"], errors="ignore")
    )
