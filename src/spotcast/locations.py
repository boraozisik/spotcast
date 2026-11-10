"""The catalogue of locations SpotCast knows: data, not code.

locations.toml is the single source of truth. Every entry is checked when it is loaded, so
a typo in the file stops the program at start-up instead of producing wrong forecasts later.

A "spot" is a location with a local guide in knowledge/ (and a training history). Every
other entry is a coastal "place" that we assess from the forecast alone.

The database table `locations` is a copy for SQL queries: `uv run spotcast-sync-locations`.
The website job and the tests read the file, so they never need a database.
"""

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import (
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

    @property
    def is_spot(self) -> bool:
        return self.guide is not None


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
        conn.execute(insert(locations_table), [loc.model_dump() for loc in locations])
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
