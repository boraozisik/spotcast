"""Batch inference for the website: compute everything ahead of time, write JSON files.

    uv run spotcast-publish --out web/public/data     # local preview
    (GitHub Actions runs it every 6 hours and pushes the files to the `site-data` branch)

The website has no backend: it only reads these files. Nothing is open to the internet
that could be abused, nothing sleeps, nothing costs money. The code that runs here is the
same model, guardrail, recommender and uncertainty notes the API and the agent use.

Files written:
    meta.json             when, which model and profile, the fixed safety text
    recommendations.json  best windows at our guided spots ("any day" and "weekend")
    spots.json            hour-by-hour forecast and verdict for each guided spot
    places.json           day-by-day assessment for every place (spots + PLACES)
"""

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from spotcast.config import HISTORY_END, HISTORY_START, SPOTS, TIMEZONE
from spotcast.data import fetch_all_spots, fetch_conditions
from spotcast.labels import add_labels
from spotcast.model import (
    FEATURES,
    load_model,
    predict_conditions,
    save_model,
    train_tree,
)
from spotcast.places import PLACES
from spotcast.profiles import ActivityProfile, get_profile
from spotcast.recommend import Preferences, recommend
from spotcast.tools import assess_day
from spotcast.uncertainty import UNCERTAIN_FROM_DAYS, days_ahead
from spotcast.versioning import code_version

SPORT, LEVEL = "windsurf", "intermediate"
TOP_WINDOWS = 5


def local_now() -> datetime:
    """Turkish time, without a timezone, like the forecast. GitHub's machines run on UTC:
    datetime.now() there would be 3 hours behind and judge the wrong hours."""
    return datetime.now(ZoneInfo(TIMEZONE)).replace(tzinfo=None)


def slug(text: str) -> str:
    """'Kuşadası Pamucak' → 'kusadasi-pamucak', for URLs."""
    plain = text.lower().translate(str.maketrans("çğıöşüâ", "cgiosua"))
    return "-".join("".join(c if c.isalnum() else " " for c in plain).split())


# ---------- The model ----------


def ensure_model(profile: ActivityProfile, retrain: bool = False) -> dict[str, Any]:
    """The saved model, or a freshly trained one if there is none (e.g. on GitHub's machine)."""
    if not retrain:
        try:
            return load_model(profile)
        except FileNotFoundError:
            pass
    print("Training the model from history...")
    history = fetch_all_spots(SPOTS, HISTORY_START, HISTORY_END)
    labeled = add_labels(history, profile)
    years = sorted(labeled["time"].dt.year.unique().tolist())
    save_model(train_tree(labeled[FEATURES], labeled["label"]), profile, years)
    return load_model(profile)


# ---------- Building the data (no network: easy to test) ----------


def lead_time(ahead: int) -> list[dict]:
    return (
        [{"code": "lead_time", "days": ahead}] if ahead >= UNCERTAIN_FROM_DAYS else []
    )


def _window(row: pd.Series, now: datetime) -> dict:
    ahead = days_ahead(row["start"], now)
    return {
        "spot": row["spot"],
        "date": f"{row['start']:%Y-%m-%d}",
        "start": f"{row['start']:%H:%M}",
        "end": f"{row['end']:%H:%M}",
        "hours": int(row["hours"]),
        "good_hours": int(row["good_hours"]),
        "marginal_hours": int(row["hours"] - row["good_hours"]),
        "avg_wind_knots": round(float(row["avg_wind_kn"]), 1),
        "max_wave_metres": round(float(row["max_wave_m"]), 2),
        "days_ahead": ahead,
        "notes": lead_time(ahead),
    }


def build_recommendations(
    predicted: pd.DataFrame, profile: ActivityProfile, now: datetime
) -> dict:
    not_before = pd.Timestamp(now).ceil("h")
    presets = {
        "any": Preferences(not_before=not_before),
        "weekend": Preferences.weekend(not_before=not_before),
    }
    return {
        key: {
            "earliest_hour": prefs.earliest_hour,
            "latest_hour": prefs.latest_hour,
            "windows": [
                _window(row, now)
                for _, row in recommend(
                    predicted, profile, prefs, top=TOP_WINDOWS
                ).iterrows()
            ],
        }
        for key, prefs in presets.items()
    }


def build_spots(predicted: pd.DataFrame, now: datetime) -> list[dict]:
    """Hour by hour, from the current hour on: past hours are no use to a visitor."""
    upcoming = predicted[predicted["time"] >= pd.Timestamp(now).floor("h")]
    spots = []
    for spot in SPOTS:
        rows = upcoming[upcoming["spot"] == spot.name].sort_values("time")
        spots.append(
            {
                "name": spot.name,
                "slug": slug(spot.name),
                "latitude": spot.latitude,
                "longitude": spot.longitude,
                "hours": [
                    {
                        "time": f"{r.time:%Y-%m-%dT%H:%M}",
                        "wind_knots": round(float(r.wind_speed_10m), 1),
                        "wind_direction": int(r.wind_direction_10m),
                        "wave_metres": None
                        if pd.isna(r.wave_height)
                        else round(float(r.wave_height), 2),
                        "is_day": bool(r.is_day),
                        "label": r.prediction,
                    }
                    for r in rows.itertuples()
                ],
            }
        )
    return spots


def site_notes(assessment: dict) -> list[dict]:
    """Warnings as codes, not sentences: the website words them in the visitor's language
    (English or Turkish). Built from facts, so they never depend on an LLM."""
    notes = []
    if assessment["label_counts"].get("unknown"):
        notes.append({"code": "no_waves"})
    if assessment["local_guide_spot"] is None:
        notes.append({"code": "no_guide"})
    return notes + lead_time(assessment["days_ahead"])


def build_place(
    name: str,
    region: str | None,  # None for our guided spots
    latitude: float,
    longitude: float,
    forecast: pd.DataFrame,
    profile: ActivityProfile,
    bundle: dict,
    now: datetime,
) -> dict:
    days = []
    for day in sorted(forecast["time"].dt.date.unique()):
        daylight = forecast[(forecast["time"].dt.date == day) & forecast["is_day"]]
        if daylight.empty or daylight["time"].max() < pd.Timestamp(now).floor("h"):
            continue  # today after sunset: nothing left to go for
        assessment = assess_day(
            forecast, day, latitude, longitude, profile, bundle, now
        )
        if not assessment["daylight_hours"]:
            continue
        # The tool's English warnings (for the LLM) and Turkish day name are replaced by
        # language-neutral codes; the website writes them in the visitor's language
        assessment.pop("warnings")
        assessment.pop("day")
        days.append(
            {"date": f"{day:%Y-%m-%d}", **assessment, "notes": site_notes(assessment)}
        )
    return {
        "name": name,
        "region": region,
        "slug": slug(name),
        "latitude": latitude,
        "longitude": longitude,
        "guided": name in {s.name for s in SPOTS},
        "days": days,
    }


def build_meta(profile: ActivityProfile, bundle: dict, now: datetime) -> dict:
    return {
        "generated_at": f"{now:%Y-%m-%dT%H:%M}",
        "timezone": TIMEZONE,
        "sport": profile.sport,
        "level": profile.level,
        "profile": asdict(profile),
        "model": {
            "type": type(bundle["model"]).__name__,
            "features": bundle["features"],
            "train_years": bundle["train_years"],
        },
        "uncertain_from_days": UNCERTAIN_FROM_DAYS,
        "code_version": code_version(),
    }


# ---------- Running it ----------


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":")), "utf-8"
    )


def publish(out: Path, retrain: bool = False) -> None:
    profile = get_profile(SPORT, LEVEL)
    bundle = ensure_model(profile, retrain)
    now = local_now()

    print("Fetching the forecast for the guided spots...")
    spot_forecast = fetch_all_spots(SPOTS)
    predicted = predict_conditions(bundle, spot_forecast)

    places = []
    for spot in SPOTS:  # reuse what we already fetched
        forecast = spot_forecast[spot_forecast["spot"] == spot.name].drop(
            columns="spot"
        )
        places.append(
            build_place(spot.name, None, spot.latitude, spot.longitude,
                        forecast, profile, bundle, now)
        )  # fmt: skip
    print(f"Fetching and assessing {len(PLACES)} places...")
    skipped = []
    for place in PLACES:
        try:
            forecast = fetch_conditions(place.latitude, place.longitude)
        except requests.RequestException as error:
            # One unreachable place must not take the whole site down
            print(f"  skipped {place.name}: {error}")
            skipped.append(place.name)
            continue
        places.append(
            build_place(place.name, place.region, place.latitude, place.longitude,
                        forecast, profile, bundle, now)
        )  # fmt: skip

    out.mkdir(parents=True, exist_ok=True)
    write_json(
        out / "meta.json",
        {**build_meta(profile, bundle, now), "skipped_places": skipped},
    )
    write_json(
        out / "recommendations.json", build_recommendations(predicted, profile, now)
    )
    write_json(out / "spots.json", build_spots(predicted, now))
    write_json(out / "places.json", places)
    sizes = ", ".join(
        f"{p.name} {p.stat().st_size // 1024} KB" for p in sorted(out.glob("*.json"))
    )
    print(f"Wrote {out}: {sizes}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the website's data files.")
    parser.add_argument("--out", type=Path, default=Path("site-data"))
    parser.add_argument(
        "--retrain", action="store_true", help="Train a fresh model first"
    )
    args = parser.parse_args()
    publish(args.out, args.retrain)
