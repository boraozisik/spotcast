"""Rule-based suitability labels (our "ground truth" for training).

These thresholds are assumptions, not facts. They are kept here as constants so they
are easy to find, discuss and change.
"""

import pandas as pd

# Intermediate windsurfer thresholds
# Below this there isn't enough wind to sail. Not dangerous, but "don't go", so "bad".
MIN_WIND_KN = 10
# Above this it gets risky for an intermediate rider (strong gusts, hard to control).
MAX_WIND_KN = 28
# Waves at or above this height are dangerous for an intermediate rider.
MAX_WAVE_M = 2.0

# The "good" window: comfortable planing conditions with small waves
GOOD_WIND_MIN_KN = 15
GOOD_WIND_MAX_KN = 25
GOOD_WAVE_MAX_M = 1.0


def label_windsurf_intermediate(wind_kn: float, wave_m: float, is_day: bool) -> str:
    """Return "good", "marginal" or "bad" for an intermediate windsurfer.

    Danger is checked first, so the worst condition always wins.
    """
    # Sailing in the dark is unsafe for everyone: nobody can see you if you fall.
    if not is_day:
        return "bad"
    if wind_kn < MIN_WIND_KN or wind_kn > MAX_WIND_KN or wave_m >= MAX_WAVE_M:
        return "bad"
    if GOOD_WIND_MIN_KN <= wind_kn <= GOOD_WIND_MAX_KN and wave_m <= GOOD_WAVE_MAX_M:
        return "good"
    return "marginal"


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the conditions table with a "label" column added."""
    labeled = df.copy()
    labeled["label"] = [
        label_windsurf_intermediate(wind, wave, day)
        for wind, wave, day in zip(
            df["wind_speed_10m"], df["wave_height"], df["is_day"]
        )
    ]
    return labeled
