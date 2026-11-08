"""Rule-based suitability labels (our "ground truth" for training).

The rules are the same for every sport; the numbers come from an ActivityProfile.
"""

import pandas as pd

from spotcast.profiles import ActivityProfile


def is_unsafe(
    profile: ActivityProfile, wind_kn: float, wave_m: float, is_day: bool
) -> bool:
    """True if conditions are dangerous for this sport and level.

    These are hard safety limits. They are used both for labeling and as a guardrail
    on top of the model, so a model mistake can never call dangerous conditions "good".
    Low wind is not here on purpose: it means "nothing to do", not "dangerous".
    """
    # Being on the water in the dark is unsafe for everyone: nobody can see you.
    if not is_day:
        return True
    return wind_kn > profile.max_wind_kn or wave_m >= profile.max_wave_m


def label_conditions(
    profile: ActivityProfile, wind_kn: float, wave_m: float, is_day: bool
) -> str:
    """Return "good", "marginal" or "bad" for this sport and level.

    Danger is checked first, so the worst condition always wins.
    """
    if is_unsafe(profile, wind_kn, wave_m, is_day):
        return "bad"
    if wind_kn < profile.min_wind_kn:
        return "bad"
    good_wind = profile.good_wind_min_kn <= wind_kn <= profile.good_wind_max_kn
    if good_wind and wave_m <= profile.good_wave_max_m:
        return "good"
    return "marginal"


def add_labels(df: pd.DataFrame, profile: ActivityProfile) -> pd.DataFrame:
    """Return a copy of the conditions table with a "label" column for this profile."""
    labeled = df.copy()
    labeled["label"] = [
        label_conditions(profile, wind, wave, day)
        for wind, wave, day in zip(
            df["wind_speed_10m"], df["wave_height"], df["is_day"], strict=True
        )
    ]
    return labeled
