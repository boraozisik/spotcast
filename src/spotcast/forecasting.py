"""Turning the hourly history into a supervised table for wind forecasting.

Each row becomes one question/answer pair:
    question (features): what we know at this hour (wind now, wind in the past hours, time of
                         day, season, direction, waves)
    answer (target):     the wind speed HORIZON_HOURS later

Every shift is done per spot (groupby), so one spot's future never leaks into another spot.
"""

import numpy as np
import pandas as pd

HORIZON_HOURS = 48
LAG_HOURS = [1, 3, 6, 12, 24, 48]  # "how windy was it this many hours ago?"
TARGET = f"wind_in_{HORIZON_HOURS}h"

FORECAST_FEATURES = [
    "wind_speed_10m",
    *[f"wind_lag_{h}h" for h in LAG_HOURS],
    "wind_change_3h",
    "wind_change_24h",
    "wind_mean_24h",
    "wind_dir_sin",
    "wind_dir_cos",
    "wave_height",
    "wave_period",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]


def _check_hourly(df: pd.DataFrame) -> None:
    """shift() counts rows, not hours. Make sure one row really is one hour, per spot."""
    steps = df.groupby("spot")["time"].diff().dropna()
    if not (steps == pd.Timedelta(hours=1)).all():
        raise ValueError("Each spot must have consecutive hourly rows, sorted by time")


def _cyclical(values: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    """Map a repeating value (hour, month, angle) onto a circle so its ends meet.

    Without this, 23:00 and 00:00 look far apart (23 vs 0) though they are 1 hour apart.
    """
    angle = 2 * np.pi * values / period
    return np.sin(angle), np.cos(angle)


def make_forecast_table(
    history: pd.DataFrame, horizon_hours: int = HORIZON_HOURS
) -> pd.DataFrame:
    """Build features and the target from hourly history (all spots stacked).

    Rows without a full past (first 48 hours) or without a known future (last
    horizon_hours) are dropped, because they can't be complete question/answer pairs.
    """
    df = history.sort_values(["spot", "time"]).reset_index(drop=True)
    _check_hourly(df)

    wind = df.groupby("spot")["wind_speed_10m"]

    # The answer: the wind horizon_hours later (shift up)
    df[f"wind_in_{horizon_hours}h"] = wind.shift(-horizon_hours)

    # The recent past (shift down)
    for hours in LAG_HOURS:
        df[f"wind_lag_{hours}h"] = wind.shift(hours)

    # Trend: is the wind rising or falling?
    df["wind_change_3h"] = df["wind_speed_10m"] - df["wind_lag_3h"]
    df["wind_change_24h"] = df["wind_speed_10m"] - df["wind_lag_24h"]
    df["wind_mean_24h"] = wind.transform(lambda s: s.rolling(24).mean())

    # Direction and time are circular: 359° is next to 1°, December is next to January
    df["wind_dir_sin"], df["wind_dir_cos"] = _cyclical(df["wind_direction_10m"], 360)
    df["hour_sin"], df["hour_cos"] = _cyclical(df["time"].dt.hour, 24)
    df["month_sin"], df["month_cos"] = _cyclical(df["time"].dt.month - 1, 12)

    needed = [*FORECAST_FEATURES, f"wind_in_{horizon_hours}h"]
    return df.dropna(subset=needed).reset_index(drop=True)
