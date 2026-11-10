"""The catalogue of locations SpotCast knows: data, not code.

locations.toml is the single source of truth. Every entry is checked when it is loaded, so
a typo in the file stops the program at start-up instead of producing wrong forecasts later.

A "spot" is a location with a local guide in knowledge/ (and a training history). Every
other entry is a coastal "place" that we assess from the forecast alone.

The database table `locations` is a copy for SQL queries: `uv run spotcast-sync-locations`.
The website job and the tests read the file, so they never need a database.
"""

import math
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    Engine,
    Float,
    MetaData,
    String,
    Table,
    func,
    insert,
    select,
)

from spotcast.profiles import SPORTS

LOCATIONS_FILE = Path(__file__).with_name("locations.toml")


class Location(BaseModel):
    # frozen: read-only like the old dataclass; extra="forbid": an unknown or misspelled
    # field is an error; strict: "38.2" (text) is not accepted as a number
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: str = Field(min_length=1)
    region: str = Field(min_length=1)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    guide: str | None = None  # file in knowledge/, e.g. "spots/alacati.md"
    # Compass direction from the beach to the open water; decides when wind is offshore
    shore_facing: float | None = Field(default=None, ge=0, lt=360)
    # A closed bay: offshore wind pushes you across the bay to another shore, not out to
    # the open sea, so offshore hours are not unsafe by themselves here (always explained
    # in a warning). Max wind, max waves and darkness still apply.
    sheltered: bool = False
    # Include this place's history in model training (guided spots always are). One per
    # sea, so the model sees Black Sea swell and open Mediterranean waves, not only the
    # Aegean. Each adds ~250 API calls to a training run: keep the list short.
    train: bool = False
    # Sports done here. None: we don't know (no local guide), so none is excluded
    sports: tuple[str, ...] | None = Field(default=None, strict=False)

    @field_validator("sports")
    @classmethod
    def _known_sports(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        unknown = sorted(set(value or ()) - set(SPORTS))
        if unknown:
            raise ValueError(f"unknown sports {unknown}; known: {', '.join(SPORTS)}")
        return value

    @property
    def is_spot(self) -> bool:
        return self.guide is not None

    @property
    def in_training(self) -> bool:
        return self.is_spot or self.train

    def supports(self, sport: str) -> bool:
        return self.sports is None or sport in self.sports


class CatalogError(ValueError):
    """The locations file is invalid."""


def parse_locations(text: str) -> list[Location]:
    """Parse and validate the TOML text. Raises CatalogError with a readable message."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise CatalogError(f"locations file is not valid TOML: {error}") from error

    locations = []
    for number, entry in enumerate(data.get("location", []), start=1):
        try:
            locations.append(Location(**entry))
        except ValidationError as error:
            label = entry.get("name", f"entry {number}")
            raise CatalogError(f"location {label!r}: {error}") from error

    names = [loc.name for loc in locations]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise CatalogError(f"duplicate location names: {', '.join(duplicates)}")
    if not locations:
        raise CatalogError("the locations file has no [[location]] entries")
    return locations


def load_locations(path: Path = LOCATIONS_FILE) -> list[Location]:
    return parse_locations(path.read_text(encoding="utf-8"))


LOCATIONS = load_locations()
SPOTS = [loc for loc in LOCATIONS if loc.is_spot]
PLACES = [loc for loc in LOCATIONS if not loc.is_spot]
# Whose history trains the suitability model: the spots first, then one place per sea
TRAINING_LOCATIONS = SPOTS + [loc for loc in PLACES if loc.train]
BY_NAME = {loc.name: loc for loc in LOCATIONS}


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Straight-line distance on the Earth's surface (haversine formula)."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def locations_within(
    latitude: float, longitude: float, radius_km: float, sport: str | None = None
) -> list[tuple[Location, float]]:
    """Known locations within this straight-line distance, nearest first. With a sport:
    only those where it is done (or where we don't know, i.e. no `sports` list)."""
    found = [
        (loc, distance_km(latitude, longitude, loc.latitude, loc.longitude))
        for loc in LOCATIONS
        if sport is None or loc.supports(sport)
    ]
    return sorted(
        ((loc, km) for loc, km in found if km <= radius_km), key=lambda pair: pair[1]
    )


def spots_for(sport: str) -> list[Location]:
    """Guided spots where this sport is done."""
    return [spot for spot in SPOTS if spot.supports(sport)]


def location_near(
    latitude: float, longitude: float, max_km: float = 2.0
) -> Location | None:
    """A known location this close, else None. Its shore direction and shelter only hold
    for that stretch of coast, so the radius is small."""
    lat_km, lon_km = 111.32, 111.32 * math.cos(math.radians(latitude))
    for loc in LOCATIONS:
        km = math.hypot(
            (loc.latitude - latitude) * lat_km, (loc.longitude - longitude) * lon_km
        )
        if km <= max_km:
            return loc
    return None


# --- Database copy ---

metadata = MetaData()
locations_table = Table(
    "locations",
    metadata,
    Column("name", String, primary_key=True),
    Column("region", String, nullable=False),
    Column(
        "latitude",
        Float,
        CheckConstraint("latitude BETWEEN -90 AND 90"),
        nullable=False,
    ),
    Column(
        "longitude",
        Float,
        CheckConstraint("longitude BETWEEN -180 AND 180"),
        nullable=False,
    ),
    Column("guide", String),  # NULL: a place without a local guide
    Column("shore_facing", Float),
    Column("sheltered", Boolean, nullable=False),
    Column("train", Boolean, nullable=False),
    Column("sports", String),  # comma-separated; NULL: unknown
)


def sync_to_db(engine: Engine, locations: list[Location]) -> int:
    """Replace the database copy with the file's contents, all or nothing.

    The table is dropped and recreated, so its columns always match the file even after a
    new field is added. PostgreSQL runs this inside one transaction: if anything fails,
    the old table stays as it was.
    """
    with engine.begin() as conn:
        locations_table.drop(conn, checkfirst=True)
        locations_table.create(conn)
        rows = [
            loc.model_dump() | {"sports": ",".join(loc.sports) if loc.sports else None}
            for loc in locations
        ]
        conn.execute(insert(locations_table), rows)
    return len(locations)


def main() -> None:
    from spotcast.db import get_engine

    engine = get_engine()
    count = sync_to_db(engine, LOCATIONS)
    print(f"Synced {count} locations ({len(SPOTS)} spots, {len(PLACES)} places)\n")

    # The kind of question SQL answers well: how many locations, and spots, per region?
    t = locations_table.c
    query = (
        select(
            t.region, func.count().label("total"), func.count(t.guide).label("spots")
        )
        .group_by(t.region)
        .order_by(func.count().desc(), t.region)
    )
    with engine.connect() as conn:
        rows = conn.execute(query).all()
    print(f"{'region':<12}{'total':>6}{'spots':>6}")
    for region, total, spots in rows:
        print(f"{region:<12}{total:>6}{spots:>6}")
