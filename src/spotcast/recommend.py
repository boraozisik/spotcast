"""Choosing the best spot and time window from an hourly forecast.

Input: one row per spot and hour, with the model's "good" / "marginal" / "bad" prediction.
Output: the best windows of consecutive hours that fit the user's preferences.

This is a small optimization problem. The search space is tiny (a few spots x 7 days x 24
hours), so we can simply look at every possible window and keep the best ones.
"""

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from spotcast.profiles import ActivityProfile

# How much one hour is worth. "bad" hours are never part of a window.
LABEL_POINTS = {"good": 2.0, "marginal": 1.0}


@dataclass(frozen=True)
class Preferences:
    """What the user wants. These are constraints: a window must satisfy all of them."""

    earliest_hour: int = 9  # first hour the session may start
    latest_hour: int = 18  # last hour that may be part of the session (inclusive)
    min_hours: int = 2  # nobody drives to the beach for a single hour
    weekdays: frozenset[int] | None = None  # 0 = Monday ... 6 = Sunday; None = any day
    days: frozenset[date] | None = None  # specific dates; None = any date
    spots: frozenset[str] | None = None  # allowed spot names; None = all spots
    not_before: datetime | None = (
        None  # e.g. now: hours that already passed are useless
    )

    @classmethod
    def weekend(cls, **kwargs) -> "Preferences":
        return cls(weekdays=frozenset({5, 6}), **kwargs)


def hour_score(prediction: str, wind_kn: float, profile: ActivityProfile) -> float:
    """Points for one hour: the label first, then how close the wind is to the ideal.

    The bonus (0..1) is always smaller than the gap between labels (1 point), so a
    "marginal" hour can never beat a "good" one just by having nicer wind.
    """
    # SUP and kayak: ideal 0 kn, so calmer hours score higher
    width = max(profile.good_wind_max_kn - profile.good_wind_min_kn, 1.0)
    closeness = max(0.0, 1 - abs(wind_kn - profile.ideal_wind) / width)
    return LABEL_POINTS[prediction] + 0.99 * closeness


def allowed_hours(predicted: pd.DataFrame, prefs: Preferences) -> pd.DataFrame:
    """Keep only hours that are worth going AND fit the user's constraints."""
    time = predicted["time"]
    keep = predicted["prediction"].isin(LABEL_POINTS.keys())
    keep &= time.dt.hour.between(prefs.earliest_hour, prefs.latest_hour)
    if prefs.weekdays is not None:
        keep &= time.dt.day_of_week.isin(prefs.weekdays)
    if prefs.days is not None:
        keep &= time.dt.date.isin(prefs.days)
    if prefs.spots is not None:
        keep &= predicted["spot"].isin(prefs.spots)
    if prefs.not_before is not None:
        keep &= time >= pd.Timestamp(prefs.not_before)
    return predicted[keep]


def find_windows(
    predicted: pd.DataFrame, profile: ActivityProfile, prefs: Preferences
) -> pd.DataFrame:
    """All runs of consecutive allowed hours (same spot, same day) of at least min_hours."""
    hours = allowed_hours(predicted, prefs).sort_values(["spot", "time"]).copy()
    if hours.empty:
        return _empty_windows()

    hours["score"] = [
        hour_score(label, wind, profile)
        for label, wind in zip(
            hours["prediction"], hours["wind_speed_10m"], strict=True
        )
    ]

    # A new window starts whenever the spot or day changes, or an hour is missing in between
    gap = hours["time"].diff() != pd.Timedelta(hours=1)
    new_spot = hours["spot"] != hours["spot"].shift()
    new_day = hours["time"].dt.date != hours["time"].dt.date.shift()
    hours["window_id"] = (gap | new_spot | new_day).cumsum()

    windows = hours.groupby("window_id").agg(
        spot=("spot", "first"),
        start=("time", "min"),
        end=("time", "max"),
        hours=("time", "size"),
        good_hours=("prediction", lambda p: int((p == "good").sum())),
        avg_wind_kn=("wind_speed_10m", "mean"),
        max_wave_m=("wave_height", "max"),
        score=("score", "sum"),
    )
    return windows[windows["hours"] >= prefs.min_hours].reset_index(drop=True)


def recommend(
    predicted: pd.DataFrame,
    profile: ActivityProfile,
    prefs: Preferences,
    top: int = 5,
) -> pd.DataFrame:
    """The best windows, highest total score first (more and better hours win)."""
    windows = find_windows(predicted, profile, prefs)
    return windows.sort_values(["score", "start"], ascending=[False, True]).head(top)


def _empty_windows() -> pd.DataFrame:
    columns = [
        "spot",
        "start",
        "end",
        "hours",
        "good_hours",
        "avg_wind_kn",
        "max_wave_m",
        "score",
    ]
    return pd.DataFrame(columns=columns)
