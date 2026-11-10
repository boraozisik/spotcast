"""Wind direction relative to the shore: onshore, side-shore or offshore.

`shore_facing` is the compass direction from the beach towards the open water (south-facing
beach: 180). Weather data gives the direction the wind blows FROM (north wind: 0).

- onshore:  the wind comes from the sea and pushes you back to the beach
- offshore: the wind comes from the land and pushes you out to sea. Close to the beach it
            feels weaker and the water looks flat, so it tempts people out; then it gets
            stronger and gustier further out. The main danger for SUP, kayak and kitesurf.
- side-shore: in between
"""

import numpy as np
import pandas as pd

from spotcast.locations import LOCATIONS

# The usual definition: within 45 degrees of straight from the sea / from the land.
# Wider sectors would turn "side-offshore" wind into offshore, and Alaçatı's famous
# northerly (side-offshore on its east-facing beach) would always look dangerous.
ONSHORE_MAX_DEG = 45
OFFSHORE_MIN_DEG = 135


def angle_between(a: float, b: float) -> float:
    """Smallest difference between two compass directions, 0..180."""
    return abs((a - b + 180) % 360 - 180)


def shore_wind(wind_from: float, facing: float | None) -> str:
    if facing is None or pd.isna(facing) or pd.isna(wind_from):
        return "unknown"
    difference = angle_between(wind_from, facing)
    if difference <= ONSHORE_MAX_DEG:
        return "onshore"
    if difference >= OFFSHORE_MIN_DEG:
        return "offshore"
    return "side-shore"


def offshore_flags(conditions: pd.DataFrame) -> pd.Series:
    """True for hours with offshore wind, sheltered or not. Unknown shore direction: False
    (no rule applies, and the tools say so in a warning)."""
    if "shore_facing" not in conditions:
        return pd.Series(False, index=conditions.index)
    wind_from = conditions["wind_direction_10m"].astype(float)
    facing = conditions["shore_facing"].astype(float)
    difference = np.abs((wind_from - facing + 180) % 360 - 180)
    return (difference >= OFFSHORE_MIN_DEG).fillna(False).astype(bool)


def unsafe_offshore_flags(conditions: pd.DataFrame) -> pd.Series:
    """Offshore hours that count for the safety rule: not in a sheltered bay, where
    drifting ends at another shore of the bay instead of the open sea."""
    offshore = offshore_flags(conditions)
    if "sheltered" not in conditions:
        return offshore
    return offshore & ~conditions["sheltered"].fillna(False).astype(bool)


def with_shore_facing(conditions: pd.DataFrame) -> pd.DataFrame:
    """Add the shore direction and shelter of each row's location (by the "spot" column)."""
    by_name = {loc.name: loc for loc in LOCATIONS}
    spot = conditions["spot"]
    return conditions.assign(
        shore_facing=spot.map(
            lambda n: by_name[n].shore_facing if n in by_name else None
        ).astype(float),
        sheltered=spot.map(lambda n: n in by_name and by_name[n].sheltered).astype(
            bool
        ),
    )
