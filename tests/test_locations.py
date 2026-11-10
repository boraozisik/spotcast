"""Tests for the locations catalogue: the real file, the validation rules and the DB copy."""

import pytest
from sqlalchemy import create_engine, func, select

from spotcast.locations import (
    LOCATIONS,
    PLACES,
    SPOTS,
    CatalogError,
    locations_table,
    parse_locations,
    sync_to_db,
)
from spotcast.rag import KNOWLEDGE_DIR, parse_front_matter

ONE = """
[[location]]
name = "Örnek Koyu"
region = "İzmir"
latitude = 38.3
longitude = 26.4
"""


# --- The real file ---


def test_real_file_spots_and_places():
    # Order matters: the trained wind models see the spot as a category in this order
    assert [s.name for s in SPOTS] == ["Kuşadası Pamucak", "Alaçatı", "Akyaka"]
    assert len(PLACES) == 26
    assert len(LOCATIONS) == len(SPOTS) + len(PLACES)


@pytest.mark.parametrize("spot", SPOTS, ids=lambda s: s.name)
def test_every_spot_guide_exists_and_names_the_spot(spot):
    """The data file and the knowledge documents must agree."""
    path = KNOWLEDGE_DIR / spot.guide
    assert path.exists(), f"{spot.name}: guide file {spot.guide} is missing"
    meta, _ = parse_front_matter(path.read_text(encoding="utf-8"))
    assert meta.get("spot") == spot.name


# --- Validation ---


def test_valid_entry():
    [loc] = parse_locations(ONE)
    assert loc.name == "Örnek Koyu"
    assert loc.guide is None and not loc.is_spot


def test_whole_number_coordinate_is_accepted():
    [loc] = parse_locations(ONE.replace("26.4", "26"))
    assert loc.longitude == 26.0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (("latitude = 38.3", "latitude = 98.3"), "less than or equal to 90"),
        (
            ("latitude = 38.3", 'latitude = "38.3"'),
            "valid number",
        ),  # text, not a number
        (("region = ", "regoin = "), "regoin"),  # typo in a field name
        (('region = "İzmir"\n', ""), "region"),  # required field missing
    ],
)
def test_invalid_entry_names_the_location(change, message):
    with pytest.raises(CatalogError) as error:
        parse_locations(ONE.replace(*change))
    assert "Örnek Koyu" in str(error.value)
    assert message in str(error.value)


def test_duplicate_names_are_rejected():
    with pytest.raises(CatalogError, match="duplicate location names: Örnek Koyu"):
        parse_locations(ONE + ONE)


def test_empty_or_broken_file_is_rejected():
    with pytest.raises(CatalogError, match="no \\[\\[location\\]\\]"):
        parse_locations("")
    with pytest.raises(CatalogError, match="not valid TOML"):
        parse_locations("[[location]\nname = ")


# --- Database copy (SQLite in memory: same SQLAlchemy code, no server needed) ---


def test_sync_replaces_the_table():
    engine = create_engine("sqlite://")
    sync_to_db(engine, LOCATIONS)
    sync_to_db(engine, LOCATIONS)  # running it again must not duplicate rows

    t = locations_table.c
    with engine.connect() as conn:
        total = conn.execute(select(func.count()).select_from(locations_table)).scalar()
        spots = conn.execute(select(func.count(t.guide))).scalar()
    assert total == len(LOCATIONS)
    assert spots == len(SPOTS)
