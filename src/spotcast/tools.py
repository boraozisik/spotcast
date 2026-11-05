"""Tools the agent can call.

A tool is a normal Python function plus two things for the LLM:
- a description, which the model reads to decide WHEN to use the tool,
- an argument schema (a Pydantic model), which tells it HOW to call it.

The LLM never runs a tool itself. It only asks for one ("call check_conditions with ...");
our code checks the arguments, runs the function and hands the result back.

Run `uv run spotcast-tools` to try every tool without an LLM.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from typing import Any, Literal

import pandas as pd
import requests
from pydantic import BaseModel, Field, ValidationError, field_validator

from spotcast.config import REQUEST_TIMEOUT_SECONDS, SPOTS, Spot
from spotcast.data import fetch_all_spots, fetch_conditions
from spotcast.explain import turkish_date
from spotcast.model import load_model, predict_conditions
from spotcast.profiles import PROFILES, ActivityProfile, get_profile
from spotcast.recommend import Preferences, find_windows, recommend

# Only the sports and levels we have profiles for. In the schema these become a list of
# allowed values (an enum), so the LLM can't send "rüzgâr sörfü" or "orta".
Sport = Literal[tuple(sorted({sport for sport, _ in PROFILES}))]
Level = Literal[tuple(sorted({level for _, level in PROFILES}))]
SPORT_HELP = "windsurf = rüzgâr sörfü"
LEVEL_HELP = (
    "yeni başlayan = beginner, orta = intermediate, ileri = advanced. "
    "Only the allowed values are supported."
)

# The LLM often copies the user's Turkish words. Accept them instead of wasting a step.
SPORT_ALIASES = {
    "rüzgâr sörfü": "windsurf",
    "rüzgar sörfü": "windsurf",
    "sörf": "windsurf",
}
LEVEL_ALIASES = {
    "yeni başlayan": "beginner",
    "başlangıç": "beginner",
    "acemi": "beginner",
    "orta": "intermediate",
    "orta seviye": "intermediate",
    "ileri": "advanced",
    "ileri seviye": "advanced",
}


class ProfileArgs(BaseModel):
    """Sport and level, shared by the tools that judge conditions."""

    sport: Sport = Field(default="windsurf", description=SPORT_HELP)
    level: Level = Field(default="intermediate", description=LEVEL_HELP)

    @field_validator("sport", mode="before")
    @classmethod
    def _sport_alias(cls, value: object) -> object:
        return SPORT_ALIASES.get(str(value).strip().lower(), value)

    @field_validator("level", mode="before")
    @classmethod
    def _level_alias(cls, value: object) -> object:
        return LEVEL_ALIASES.get(str(value).strip().lower(), value)


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_DAYS = 7  # Open-Meteo's default forecast length
SPOT_GUIDE_RADIUS_KM = 15  # closer than this to a known spot: its guide applies


class ToolError(Exception):
    """A problem the LLM should be told about (unknown place, date too far), not a crash."""


# ---------- The Tool wrapper ----------


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args: type[BaseModel]
    run: Callable[[Any], dict]

    def schema(self) -> dict:
        """The tool definition in the format Ollama (and OpenAI-style APIs) expect."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args.model_json_schema(),
            },
        }

    def call(self, arguments: dict) -> dict:
        """Check the LLM's arguments, run the tool, and always return a dict.

        Mistakes become {"error": ...} so the LLM can read them and try again,
        instead of crashing the whole conversation.
        """
        try:
            parsed = self.args.model_validate(arguments)
        except ValidationError as error:
            problems = "; ".join(
                f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors()
            )
            return {"error": f"Invalid arguments for {self.name}: {problems}"}
        try:
            return self.run(parsed)
        except ToolError as error:
            return {"error": str(error)}


# ---------- Shared helpers ----------


def now() -> datetime:
    """The current local time. A function, so tests can replace it."""
    return datetime.now()


@lru_cache
def _bundle(sport: str, level: str) -> dict:
    """Load each profile's model once, not on every tool call."""
    return load_model(get_profile(sport, level))


def _profile(sport: str, level: str) -> ActivityProfile:
    try:
        return get_profile(sport, level)
    except ValueError as error:
        raise ToolError(str(error)) from None


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Straight-line distance on the Earth's surface (haversine formula)."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def nearest_spot(latitude: float, longitude: float) -> tuple[Spot, float]:
    spot = min(
        SPOTS, key=lambda s: distance_km(latitude, longitude, s.latitude, s.longitude)
    )
    return spot, distance_km(latitude, longitude, spot.latitude, spot.longitude)


def _check_day(day: date) -> None:
    today = now().date()
    if not 0 <= (day - today).days < FORECAST_DAYS:
        raise ToolError(
            f"{day} is outside the forecast. Only {today} to "
            f"{today + pd.Timedelta(days=FORECAST_DAYS - 1)} can be checked."
        )


# ---------- Tool 1: find_location ----------


class FindLocationArgs(BaseModel):
    name: str = Field(description="Place name in Turkey, e.g. 'Mersin' or 'Alaçatı'")


def find_location(args: FindLocationArgs) -> dict:
    response = requests.get(
        GEOCODING_URL,
        params={"name": args.name, "count": 3, "language": "tr", "countryCode": "TR"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    if not results:
        raise ToolError(f"No place called '{args.name}' was found in Turkey.")
    return {
        "places": [
            {
                "name": r["name"],
                "region": ", ".join(x for x in (r.get("admin2"), r.get("admin1")) if x),
                "latitude": r["latitude"],
                "longitude": r["longitude"],
            }
            for r in results
        ]
    }


# ---------- Tool 2: check_conditions (any place) ----------


class CheckConditionsArgs(ProfileArgs):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    day: date = Field(
        description="The day to check, YYYY-MM-DD, within the next 7 days"
    )


def check_conditions(args: CheckConditionsArgs) -> dict:
    profile = _profile(args.sport, args.level)
    _check_day(args.day)

    forecast = fetch_conditions(args.latitude, args.longitude)
    hours = forecast[forecast["time"].dt.date == args.day].copy()
    hours.insert(0, "spot", "requested place")  # find_windows groups hours by spot
    has_waves = hours["wave_height"].notna()

    # The model needs wave height. Without it we report the wind but give no verdict.
    hours["prediction"] = "unknown"
    if has_waves.any():
        known = predict_conditions(
            _bundle(profile.sport, profile.level), hours[has_waves]
        )
        hours.loc[has_waves, "prediction"] = known["prediction"]

    daylight = hours[hours["is_day"]]
    windows = find_windows(
        hours[hours["prediction"] != "unknown"],
        profile,
        Preferences(earliest_hour=0, latest_hour=23, min_hours=2, not_before=now()),
    )
    best = windows.sort_values("score", ascending=False).head(1)

    spot, km = nearest_spot(args.latitude, args.longitude)
    warnings = []
    if not has_waves.all():
        warnings.append(
            "No wave data here (on land or too close to the shore); no verdict."
        )
    if km > SPOT_GUIDE_RADIUS_KM:
        warnings.append(
            "SpotCast has no local guide for this place: check local dangers."
        )

    return {
        "day": turkish_date(pd.Timestamp(args.day)),
        "sport": profile.sport,
        "level": profile.level,
        "daylight_hours": [
            {
                "hour": f"{row.time:%H:%M}",
                "wind_knots": round(row.wind_speed_10m, 1),
                "wave_metres": None
                if pd.isna(row.wave_height)
                else round(row.wave_height, 2),
                "label": row.prediction,
            }
            for row in daylight.itertuples()
        ],
        "label_counts": daylight["prediction"].value_counts().to_dict(),
        "best_window": None
        if best.empty
        else {
            "hours": f"{best.iloc[0]['start']:%H:%M}-{best.iloc[0]['end']:%H:%M}",
            "good_hours": int(best.iloc[0]["good_hours"]),
            "avg_wind_knots": round(float(best.iloc[0]["avg_wind_kn"]), 1),
        },
        "local_guide_spot": spot.name if km <= SPOT_GUIDE_RADIUS_KM else None,
        "warnings": warnings,
    }


# ---------- Tool 3: recommend_spots (known spots) ----------


class RecommendSpotsArgs(ProfileArgs):
    day: date | None = Field(default=None, description="Only this day (YYYY-MM-DD)")
    weekend_only: bool = False
    earliest_hour: int = Field(default=9, ge=0, le=23)
    latest_hour: int = Field(default=18, ge=0, le=23)


def recommend_spots(args: RecommendSpotsArgs) -> dict:
    profile = _profile(args.sport, args.level)
    if args.day is not None:
        _check_day(args.day)

    predicted = predict_conditions(
        _bundle(profile.sport, profile.level), fetch_all_spots(SPOTS)
    )
    prefs = Preferences(
        earliest_hour=args.earliest_hour,
        latest_hour=args.latest_hour,
        weekdays=frozenset({5, 6}) if args.weekend_only else None,
        days=frozenset({args.day}) if args.day else None,
        not_before=now(),
    )
    best = recommend(predicted, profile, prefs, top=3)
    return {
        "known_spots": [s.name for s in SPOTS],
        "windows_best_first": [
            {
                "spot": row["spot"],
                "day": turkish_date(row["start"]),
                "hours": f"{row['start']:%H:%M}-{row['end']:%H:%M}",
                "good_hours": int(row["good_hours"]),
                "marginal_hours": int(row["hours"] - row["good_hours"]),
                "avg_wind_knots": round(float(row["avg_wind_kn"]), 1),
                "max_wave_metres": round(float(row["max_wave_m"]), 2),
            }
            for _, row in best.iterrows()
        ],
    }


# ---------- Tool 4: search_knowledge (RAG) ----------


class SearchKnowledgeArgs(BaseModel):
    question: str = Field(
        description="What to look up, with context, e.g. 'Alaçatı tehlikeler'"
    )
    spot: str | None = Field(
        default=None, description="Limit to one known spot's guide"
    )


@lru_cache
def _engine():
    from spotcast.db import get_engine

    return get_engine()


def search_knowledge(args: SearchKnowledgeArgs) -> dict:
    from sqlalchemy.exc import SQLAlchemyError

    from spotcast.rag import search

    try:
        results = search(
            _engine(), args.question, k=3, spots=[args.spot] if args.spot else None
        )
    except (SQLAlchemyError, RuntimeError) as error:
        raise ToolError(f"The knowledge base is not available: {error}") from None
    return {
        "excerpts": [
            {
                "title": r.chunk.title,
                "section": r.chunk.section,
                "text": r.chunk.content,
            }
            for r in results
            if r.distance <= 0.5  # farther is too unrelated to be useful
        ]
    }


# ---------- Registry ----------

TOOLS = [
    Tool(
        "find_location",
        "Find coordinates of a place in Turkey by its name. Use it before check_conditions "
        "when the user names a place.",
        FindLocationArgs,
        find_location,
    ),
    Tool(
        "check_conditions",
        "Check wind, waves and suitability hour by hour for ANY coordinates on one day in "
        "the next 7 days. Use it when the user asks about a specific place and day.",
        CheckConditionsArgs,
        check_conditions,
    ),
    Tool(
        "recommend_spots",
        f"Find the best time windows at SpotCast's known spots "
        f"({', '.join(s.name for s in SPOTS)}) in the next 7 days. Use it when the user "
        f"asks where or when to go.",
        RecommendSpotsArgs,
        recommend_spots,
    ),
    Tool(
        "search_knowledge",
        "Search SpotCast's guides: spot guides, the Beaufort scale and safety rules "
        "(offshore wind, rip currents, ...). Use it for advice and safety questions.",
        SearchKnowledgeArgs,
        search_knowledge,
    ),
]
TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}


def main() -> None:
    """Try each tool once with example arguments, without any LLM."""
    import json

    saturday = (now() + pd.offsets.Week(weekday=5)).date()
    examples = [
        ("find_location", {"name": "Mersin"}),
        (
            "check_conditions",
            {"latitude": 36.78, "longitude": 34.6, "day": str(saturday)},
        ),
        ("recommend_spots", {"weekend_only": True}),
        (
            "search_knowledge",
            {"question": "Dip akıntısına yakalanırsam ne yapmalıyım?"},
        ),
        (
            "check_conditions",
            {"latitude": 36.78, "longitude": 34.6, "day": "2030-01-01"},
        ),
        ("find_location", {"nam": "Mersin"}),
    ]
    for name, arguments in examples:
        print(f"\n>>> {name}({arguments})")
        result = TOOLS_BY_NAME[name].call(arguments)
        print(json.dumps(result, ensure_ascii=False, indent=2)[:1200])
