"""SpotCast web API (FastAPI).

Run with: uv run spotcast-api   →   open http://127.0.0.1:8000/docs
"""

import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated, Any

import pandas as pd
import requests
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from spotcast.config import SPOTS
from spotcast.data import fetch_all_spots
from spotcast.model import load_model, predict_conditions
from spotcast.profiles import PROFILES, ActivityProfile, get_profile
from spotcast.recommend import Preferences, recommend

FORECAST_CACHE_SECONDS = (
    30 * 60
)  # Open-Meteo updates a few times a day; don't call it every request


# ---------- Response shapes (like DTOs in NestJS) ----------


class SpotOut(BaseModel):
    name: str
    latitude: float
    longitude: float


class HourOut(BaseModel):
    spot: str
    time: datetime
    wind_kn: float
    wave_m: float
    is_day: bool
    prediction: str = Field(description='"good", "marginal" or "bad"')


class WindowOut(BaseModel):
    spot: str
    start: datetime
    end: datetime
    hours: int
    good_hours: int
    avg_wind_kn: float
    max_wave_m: float
    score: float


class RecommendOut(BaseModel):
    sport: str
    level: str
    generated_at: datetime
    windows: list[WindowOut] = Field(
        description="Best windows first. Empty if nothing fits."
    )


# ---------- Forecast source with a small in-memory cache ----------


class ForecastCache:
    """Remembers the last forecast for a while, so many requests cause one API call."""

    def __init__(self, fetch: Callable[[], pd.DataFrame], max_age_seconds: float):
        self._fetch = fetch
        self._max_age = max_age_seconds
        self._data: pd.DataFrame | None = None
        self._fetched_at = 0.0

    def get(self) -> pd.DataFrame:
        if self._data is None or time.monotonic() - self._fetched_at > self._max_age:
            self._data = self._fetch()
            self._fetched_at = time.monotonic()
        return self._data


# ---------- App startup: load models once ----------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Runs once when the server starts (like onModuleInit), not on every request."""
    app.state.bundles = {key: load_model(profile) for key, profile in PROFILES.items()}
    app.state.forecast = ForecastCache(
        lambda: fetch_all_spots(SPOTS), FORECAST_CACHE_SECONDS
    )
    yield  # the server runs while we are paused here


app = FastAPI(
    title="SpotCast API",
    description="Is this spot good for my sport and level? Forecasts and recommendations.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------- Dependencies (like NestJS providers): easy to swap in tests ----------


def get_bundles(request: Request) -> dict[tuple[str, str], dict[str, Any]]:
    return request.app.state.bundles


def get_forecast(request: Request) -> pd.DataFrame:
    try:
        return request.app.state.forecast.get()
    except requests.RequestException as error:
        raise HTTPException(502, f"Weather service unavailable: {error}") from error


def get_now() -> datetime:
    return datetime.now()


def resolve_profile(sport: str, level: str) -> ActivityProfile:
    try:
        return get_profile(sport, level)
    except ValueError as error:
        raise HTTPException(404, str(error)) from error


Bundles = Annotated[dict[tuple[str, str], dict[str, Any]], Depends(get_bundles)]
Forecast = Annotated[pd.DataFrame, Depends(get_forecast)]
Now = Annotated[datetime, Depends(get_now)]
Sport = Annotated[str, Query(examples=["windsurf"])]
Level = Annotated[str, Query(examples=["intermediate"])]


def predicted_hours(
    sport: str, level: str, bundles: dict, forecast: pd.DataFrame
) -> tuple[ActivityProfile, pd.DataFrame]:
    profile = resolve_profile(sport, level)
    return profile, predict_conditions(
        bundles[(profile.sport, profile.level)], forecast
    )


# ---------- Endpoints ----------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/spots")
def list_spots() -> list[SpotOut]:
    return [
        SpotOut(name=s.name, latitude=s.latitude, longitude=s.longitude) for s in SPOTS
    ]


@app.get("/profiles")
def list_profiles() -> list[dict[str, str]]:
    return [{"sport": sport, "level": level} for sport, level in PROFILES]


@app.get("/forecast")
def forecast_hours(
    sport: Sport, level: Level, bundles: Bundles, forecast: Forecast
) -> list[HourOut]:
    """Every hour of the next 7 days, for every spot, with the model's prediction."""
    _, predicted = predicted_hours(sport, level, bundles, forecast)
    return [
        HourOut(
            spot=row.spot,
            time=row.time,
            wind_kn=row.wind_speed_10m,
            wave_m=row.wave_height,
            is_day=row.is_day,
            prediction=row.prediction,
        )
        for row in predicted.itertuples()
    ]


@app.get("/recommend")
def recommend_windows(
    sport: Sport,
    level: Level,
    bundles: Bundles,
    forecast: Forecast,
    now: Now,
    earliest_hour: Annotated[int, Query(ge=0, le=23)] = 9,
    latest_hour: Annotated[int, Query(ge=0, le=23)] = 18,
    min_hours: Annotated[int, Query(ge=1, le=24)] = 2,
    weekend_only: bool = False,
    spots: Annotated[
        list[str] | None, Query(description="Limit to these spots")
    ] = None,
    top: Annotated[int, Query(ge=1, le=20)] = 5,
) -> RecommendOut:
    """The best time windows in the next 7 days that fit your preferences."""
    if latest_hour < earliest_hour:
        raise HTTPException(
            422, "latest_hour must be the same as or after earliest_hour"
        )
    unknown = set(spots or []) - {s.name for s in SPOTS}
    if unknown:
        raise HTTPException(422, f"Unknown spots: {sorted(unknown)}")

    profile, predicted = predicted_hours(sport, level, bundles, forecast)
    prefs = Preferences(
        earliest_hour=earliest_hour,
        latest_hour=latest_hour,
        min_hours=min_hours,
        weekdays=frozenset({5, 6}) if weekend_only else None,
        spots=frozenset(spots) if spots else None,
        not_before=pd.Timestamp(now).ceil("h"),
    )
    best = recommend(predicted, profile, prefs, top=top)
    best = best.round({"avg_wind_kn": 2, "max_wave_m": 2, "score": 2})

    return RecommendOut(
        sport=profile.sport,
        level=profile.level,
        generated_at=now,
        windows=[WindowOut(**row) for row in best.to_dict(orient="records")],
    )


def run() -> None:
    """Start a development server that reloads when code changes."""
    import uvicorn

    uvicorn.run("spotcast.api:app", reload=True)
