"""Activity profiles: the thresholds that decide if conditions suit a sport and level.

Each profile is one sport at one skill level. Adding a new sport or level means adding a
new profile here, not changing the labeling code.

All numbers are assumptions, not facts. They are kept together so they are easy to find,
discuss and change.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ActivityProfile:
    sport: str
    level: str

    # Hard safety limits: at or beyond these, conditions are dangerous ("unsafe").
    max_wind_kn: float  # wind above this is unsafe
    max_wave_m: float  # waves at or above this height are unsafe

    # Below this there isn't enough wind for the sport. Not dangerous, just "don't go".
    # Sports that need no wind (SUP, kayak) can set this to 0.
    min_wind_kn: float

    # The "good" window: everything inside it is comfortable for this level.
    good_wind_min_kn: float
    good_wind_max_kn: float
    good_wave_max_m: float


WINDSURF_INTERMEDIATE = ActivityProfile(
    sport="windsurf",
    level="intermediate",
    max_wind_kn=28,  # strong gusts, hard to control for an intermediate rider
    max_wave_m=2.0,
    min_wind_kn=10,  # the sail doesn't fill below this
    good_wind_min_kn=15,  # planing starts around here
    good_wind_max_kn=25,
    good_wave_max_m=1.0,
)

# Every known profile, looked up by (sport, level)
PROFILES: dict[tuple[str, str], ActivityProfile] = {
    (p.sport, p.level): p for p in [WINDSURF_INTERMEDIATE]
}


def get_profile(sport: str, level: str) -> ActivityProfile:
    """Find the profile for a sport and level, with a clear error if it doesn't exist."""
    try:
        return PROFILES[(sport, level)]
    except KeyError:
        known = ", ".join(f"{s}/{lvl}" for s, lvl in PROFILES)
        raise ValueError(
            f"No profile for {sport}/{level}. Known profiles: {known}"
        ) from None
