"""Activity profiles: the thresholds that decide if conditions suit a sport and level.

Each profile is one sport at one skill level. The numbers live in profiles.toml (data, not
code): adding a sport or level is a new [[profile]] block there.

ActivityProfile stays a dataclass on purpose: saved model bundles (joblib/pickle) contain a
profile object, and changing the class type would make old model files unreadable.
"""

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

PROFILES_FILE = Path(__file__).with_name("profiles.toml")
LEVELS = ("beginner", "intermediate", "advanced")


class ProfileError(ValueError):
    """The profiles file is invalid."""


@dataclass(frozen=True)
class ActivityProfile:
    sport: str
    level: str

    # Hard safety limits: at or beyond these, conditions are dangerous ("unsafe").
    max_wind_kn: float  # wind above this is unsafe
    max_wave_m: float  # waves at or above this height are unsafe

    # Below this there isn't enough wind for the sport. Not dangerous, just "don't go".
    min_wind_kn: float

    # The "good" window: everything inside it is comfortable for this level.
    good_wind_min_kn: float
    good_wind_max_kn: float
    good_wave_max_m: float

    # Best wind inside the good window (None: its middle). A default value also keeps old
    # saved bundles working: their profiles were pickled before this field existed.
    ideal_wind_kn: float | None = None

    # Offshore wind (from land to sea) above this is unsafe. None: no offshore rule (old
    # saved profiles). 0: never in offshore wind.
    max_offshore_wind_kn: float | None = None

    def __post_init__(self) -> None:
        """Reject impossible profiles as soon as one is created."""
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in ("sport", "level"):
                if not isinstance(value, str) or not value:
                    raise ProfileError(f"{f.name} must be a non-empty text")
            elif value is not None and (
                isinstance(value, bool) or not isinstance(value, int | float)
            ):
                raise ProfileError(f"{f.name} must be a number, got {value!r}")

        if self.level not in LEVELS:
            raise ProfileError(f"level must be one of {', '.join(LEVELS)}")
        winds = [
            self.min_wind_kn,
            self.good_wind_min_kn,
            self.good_wind_max_kn,
            self.max_wind_kn,
        ]
        if not 0 <= winds[0] <= winds[1] <= winds[2] <= winds[3]:
            raise ProfileError(
                "winds must satisfy 0 <= min_wind <= good_wind_min <= good_wind_max "
                f"<= max_wind, got {winds}"
            )
        if not 0 < self.good_wave_max_m < self.max_wave_m:
            raise ProfileError("waves must satisfy 0 < good_wave_max < max_wave")
        if self.ideal_wind_kn is not None and not (
            self.good_wind_min_kn <= self.ideal_wind_kn <= self.good_wind_max_kn
        ):
            raise ProfileError("ideal_wind must be inside the good wind range")
        if self.max_offshore_wind_kn is not None and not (
            0 <= self.max_offshore_wind_kn <= self.max_wind_kn
        ):
            raise ProfileError("max_offshore_wind must be between 0 and max_wind")

    @property
    def name(self) -> str:
        return f"{self.sport}/{self.level}"

    @property
    def ideal_wind(self) -> float:
        if self.ideal_wind_kn is not None:
            return self.ideal_wind_kn
        return (self.good_wind_min_kn + self.good_wind_max_kn) / 2


def parse_profiles(text: str) -> list[ActivityProfile]:
    """Parse and check the TOML text. Raises ProfileError with a readable message."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ProfileError(f"profiles file is not valid TOML: {error}") from error

    profiles = []
    for number, entry in enumerate(data.get("profile", []), start=1):
        label = f"{entry.get('sport', '?')}/{entry.get('level', '?')} (entry {number})"
        try:
            profiles.append(ActivityProfile(**entry))
        except TypeError as error:  # unknown or missing field
            raise ProfileError(f"profile {label}: {error}") from error
        except ProfileError as error:
            raise ProfileError(f"profile {label}: {error}") from error

    keys = [(p.sport, p.level) for p in profiles]
    duplicates = sorted({f"{s}/{lvl}" for s, lvl in keys if keys.count((s, lvl)) > 1})
    if duplicates:
        raise ProfileError(f"duplicate profiles: {', '.join(duplicates)}")
    if not profiles:
        raise ProfileError("the profiles file has no [[profile]] entries")
    return profiles


def load_profiles(path: Path = PROFILES_FILE) -> list[ActivityProfile]:
    return parse_profiles(path.read_text(encoding="utf-8"))


# Every known profile, looked up by (sport, level)
PROFILES: dict[tuple[str, str], ActivityProfile] = {
    (p.sport, p.level): p for p in load_profiles()
}
SPORTS = tuple(dict.fromkeys(sport for sport, _ in PROFILES))  # in file order


def get_profile(sport: str, level: str) -> ActivityProfile:
    """Find the profile for a sport and level, with a clear error if it doesn't exist."""
    try:
        return PROFILES[(sport, level)]
    except KeyError:
        known = ", ".join(f"{s}/{lvl}" for s, lvl in PROFILES)
        raise ValueError(
            f"No profile for {sport}/{level}. Known profiles: {known}"
        ) from None


WINDSURF_INTERMEDIATE = get_profile("windsurf", "intermediate")
