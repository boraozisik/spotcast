"""Batch inference for the website: compute everything ahead of time, write JSON files.

    uv run spotcast-publish --out web/public/data     # local preview
    (GitHub Actions runs it every 3 hours and pushes the files to the `site-data` branch)

The website has no backend: it only reads these files. Nothing is open to the internet
that could be abused, nothing sleeps, nothing costs money. The code that runs here is the
same model, guardrail, recommender and uncertainty notes the API and the agent use.

Files written (one set per activity profile: 4 sports x 3 levels):
    meta.json                         when, which model, every profile and its limits
    profiles/<sport>_<level>/
        recommendations.json          best windows at our guided spots (any day, weekend)
        spots.json                    hour by hour at each guided spot
        places.json                   day by day at every place (spots + PLACES)
    recommendations.json, spots.json, places.json at the top: the default profile in the
    format of the first site version, so a site deployed before this change keeps working.

The weather is fetched once; only the verdicts differ per profile.
"""

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from spotcast.config import HISTORY_END, HISTORY_START, TIMEZONE
from spotcast.data import fetch_all_spots
from spotcast.locations import BY_NAME, PLACES, SPOTS, TRAINING_LOCATIONS, Location
from spotcast.model import (
    FEATURES,
    MONITORED_INPUTS,
    feature_ranges,
    load_model,
    predict_conditions,
    save_model,
    train_tree,
    training_table,
)
from spotcast.monitoring import (
    DataQualityError,
    notices,
    problems,
    quality_report,
    write_run_summary,
)
from spotcast.profiles import PROFILES, ActivityProfile, get_profile
from spotcast.recommend import Preferences, recommend
from spotcast.shore import shore_wind
from spotcast.tools import assess_day
from spotcast.uncertainty import UNCERTAIN_FROM_DAYS, days_ahead
from spotcast.versioning import code_version

SPORT, LEVEL = "windsurf", "intermediate"  # the default profile
TOP_WINDOWS = 5
LEGACY_NOTE_CODES = {"lead_time", "no_waves", "no_guide"}  # what the first site knows


def local_now() -> datetime:
    """Turkish time, without a timezone, like the forecast. GitHub's machines run on UTC:
    datetime.now() there would be 3 hours behind and judge the wrong hours."""
    return datetime.now(ZoneInfo(TIMEZONE)).replace(tzinfo=None)


def slug(text: str) -> str:
    """'Kuşadası Pamucak' → 'kusadasi-pamucak', for URLs."""
    plain = text.lower().translate(str.maketrans("çğıöşüâ", "cgiosua"))
    return "-".join("".join(c if c.isalnum() else " " for c in plain).split())


# ---------- The model ----------


def ensure_model(retrain: bool = False) -> dict[str, Any]:
    """The saved model, or a freshly trained one if there is none (e.g. on GitHub's machine,
    or when models/ only holds a model in an older format)."""
    if not retrain:
        try:
            bundle = load_model()
            if bundle["features"] == FEATURES:
                return bundle
            print("The saved model uses older features.")
        except FileNotFoundError:
            pass
    print("Training the model from history...")
    history = fetch_all_spots(TRAINING_LOCATIONS, HISTORY_START, HISTORY_END)
    profiles = list(PROFILES.values())
    table = training_table(history, profiles)
    save_model(
        train_tree(table[FEATURES], table["label"]),
        profiles,
        sorted(table["time"].dt.year.unique().tolist()),
        ranges=feature_ranges(history[MONITORED_INPUTS]),
        train_end=HISTORY_END,
    )
    return load_model()


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
                        "shore_wind": shore_wind(
                            r.wind_direction_10m, spot.shore_facing
                        ),
                        "label": r.prediction,
                    }
                    for r in rows.itertuples()
                ],
            }
        )
    return spots


def site_notes(
    assessment: dict, location: Location | None, profile: ActivityProfile
) -> list[dict]:
    """Warnings as codes, not sentences: the website words them in the visitor's language
    (English or Turkish). Built from facts, so they never depend on an LLM."""
    notes = []
    if assessment["label_counts"].get("unknown"):
        notes.append({"code": "no_waves"})
    if assessment["local_guide_spot"] is None:
        notes.append({"code": "no_guide"})
    if location is not None and not location.supports(profile.sport):
        notes.append({"code": "not_for_sport"})
    if not assessment["shore_known"]:
        notes.append({"code": "no_shore"})
    if hours := assessment["offshore_hours"]:
        notes.append({"code": "offshore", "hours": hours})
    if hours := assessment["sheltered_offshore_hours"]:
        notes.append({"code": "sheltered_offshore", "hours": hours})
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
        notes = site_notes(assessment, BY_NAME.get(name), profile)
        days.append({"date": f"{day:%Y-%m-%d}", **assessment, "notes": notes})
    return {
        "name": name,
        "region": region,
        "slug": slug(name),
        "latitude": latitude,
        "longitude": longitude,
        "guided": name in {s.name for s in SPOTS},
        "days": days,
    }


def profile_id(profile: ActivityProfile) -> str:
    """'windsurf_intermediate': the folder name of a profile's files."""
    return f"{profile.sport}_{profile.level}"


def build_meta(profile: ActivityProfile, bundle: dict, now: datetime) -> dict:
    return {
        "generated_at": f"{now:%Y-%m-%dT%H:%M}",
        "timezone": TIMEZONE,
        # The default profile, also in the first site version's fields
        "sport": profile.sport,
        "level": profile.level,
        "profile": asdict(profile),  # includes ideal_wind_kn
        "default_profile": profile_id(profile),
        "profiles": [{"id": profile_id(p), **asdict(p)} for p in PROFILES.values()],
        "spots": [
            {
                "name": s.name,
                "sports": list(s.sports) if s.sports else None,
                "sheltered": s.sheltered,
            }
            for s in SPOTS
        ],  # fmt: skip
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


def fetch_places(places: list[Location]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Forecasts for all places in one batched request (2 HTTP calls for 26 places).

    If that fails, fall back to one place at a time, so a single unreachable place can
    never take the whole site down. Returns the forecasts by name and the skipped names.
    """
    try:
        everything = fetch_all_spots(places)
        return {
            name: rows.drop(columns="spot")
            for name, rows in everything.groupby("spot", sort=False)
        }, []
    except requests.RequestException as error:
        print(f"  batched request failed ({error}); fetching place by place")

    found, skipped = {}, []
    for place in places:
        try:
            found[place.name] = fetch_all_spots([place]).drop(columns="spot")
        except requests.RequestException as error:
            print(f"  skipped {place.name}: {error}")
            skipped.append(place.name)
    return found, skipped


def build_profile(
    profile: ActivityProfile,
    bundle: dict,
    spot_forecast: pd.DataFrame,
    place_forecasts: dict[str, pd.DataFrame],
    now: datetime,
) -> dict[str, Any]:
    """The three files of one profile, from forecasts fetched once for all profiles."""
    predicted = predict_conditions(bundle, spot_forecast, profile)
    places = []
    for spot in SPOTS:  # reuse the spots' forecast
        forecast = spot_forecast[spot_forecast["spot"] == spot.name].drop(
            columns="spot"
        )
        places.append(
            build_place(spot.name, None, spot.latitude, spot.longitude,
                        forecast, profile, bundle, now)
        )  # fmt: skip
    for place in PLACES:
        if place.name in place_forecasts:
            places.append(
                build_place(place.name, place.region, place.latitude, place.longitude,
                            place_forecasts[place.name], profile, bundle, now)
            )  # fmt: skip
    return {
        "recommendations.json": build_recommendations(predicted, profile, now),
        "spots.json": build_spots(predicted, now),
        "places.json": places,
    }


def legacy_places(places: list[dict]) -> list[dict]:
    """The first site version shows any unknown note code as "no local guide": keep only
    the codes it knows."""
    return [
        place
        | {
            "days": [
                day
                | {"notes": [n for n in day["notes"] if n["code"] in LEGACY_NOTE_CODES]}
                for day in place["days"]
            ]
        }
        for place in places
    ]


def publish(out: Path, retrain: bool = False) -> None:
    default = get_profile(SPORT, LEVEL)
    bundle = ensure_model(retrain)
    now = local_now()

    print("Fetching the forecast for the guided spots...")
    spot_forecast = fetch_all_spots(SPOTS)
    print(f"Fetching {len(PLACES)} places...")
    place_forecasts, skipped = fetch_places(PLACES)

    # Check the data before anything is written: bad data must never replace good data
    fetched = pd.concat([spot_forecast, *place_forecasts.values()])
    report = quality_report(fetched, bundle, len(PLACES), skipped)
    write_run_summary(report)
    for notice in notices(report):
        print(f"  warning: {notice}")
    if stops := problems(report):
        raise DataQualityError("Not published: " + " ".join(stops))

    print(f"Assessing for {len(PROFILES)} profiles...")
    for profile in PROFILES.values():
        files = build_profile(profile, bundle, spot_forecast, place_forecasts, now)
        folder = out / "profiles" / profile_id(profile)
        folder.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            write_json(folder / name, content)
        if profile == default:  # the first site version's files, at the top
            write_json(out / "recommendations.json", files["recommendations.json"])
            write_json(out / "spots.json", files["spots.json"])
            write_json(out / "places.json", legacy_places(files["places.json"]))
    write_json(
        out / "meta.json", {**build_meta(default, bundle, now), "quality": report}
    )

    total = sum(f.stat().st_size for f in out.rglob("*.json")) // 1024
    one = sum(
        f.stat().st_size
        for f in (out / "profiles" / profile_id(default)).glob("*.json")
    )
    print(f"Wrote {out}: {total} KB in all, {one // 1024} KB per profile")


# The same threshold as the website's "this forecast is old" notice
# (web/src/components/StaleNotice.tsx)
STALE_AFTER_HOURS = 12


def published_age_hours(now: datetime) -> float | None:
    """How old the data on the live site is. None if it can't be read (e.g. run locally)."""
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not repo:
        return None
    url = f"https://raw.githubusercontent.com/{repo}/site-data/meta.json"
    try:
        generated_at = requests.get(url, timeout=20).json()["generated_at"]
    except (requests.RequestException, ValueError, KeyError):
        return None
    return (now - datetime.fromisoformat(generated_at)).total_seconds() / 3600


def _step_summary(text: str) -> None:
    """Add text to the GitHub Actions run page (no-op outside GitHub Actions)."""
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(path).open("a", encoding="utf-8") as file:
            file.write(text)


def _github_output(line: str) -> None:
    """Tell later workflow steps what happened (no-op outside GitHub Actions)."""
    if path := os.environ.get("GITHUB_OUTPUT"):
        with Path(path).open("a", encoding="utf-8") as file:
            file.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the website's data files.")
    parser.add_argument("--out", type=Path, default=Path("site-data"))
    parser.add_argument(
        "--retrain", action="store_true", help="Train a fresh model first"
    )
    args = parser.parse_args()
    try:
        publish(args.out, args.retrain)
    except requests.RequestException as error:
        # Alert on the symptom, not the cause: Open-Meteo is sometimes unreachable for a
        # few minutes (twice in the evening of 13-14 Nov 2026, even after 5 patient
        # attempts). The site keeps its last good data, and the next run usually works.
        # A failed run (and an e-mail) is only worth it once visitors see stale data.
        age = published_age_hours(local_now())
        if age is None or age >= STALE_AFTER_HOURS:
            raise
        message = (
            f"Open-Meteo did not answer ({type(error).__name__}). Nothing was published; "
            f"the site keeps its data from {age:.0f} hours ago. This becomes a failure "
            f"once that data is {STALE_AFTER_HOURS} hours old."
        )
        print(f"::warning::{message}")  # shown on the GitHub Actions run page
        _step_summary(f"### Site data not refreshed\n\n- {message}\n")
        _github_output("published=false")
        return
    _github_output("published=true")
