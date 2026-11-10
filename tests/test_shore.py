"""Tests for the offshore rule: wind direction against the direction a beach faces."""

import importlib

import numpy as np
import pandas as pd
import pytest

from spotcast.labels import add_labels, is_unsafe, label_conditions
from spotcast.locations import BY_NAME, CatalogError, parse_locations
from spotcast.model import FEATURES, make_features
from spotcast.profiles import get_profile
from spotcast.recommend import Preferences, recommend
from spotcast.shore import (
    angle_between,
    offshore_flags,
    shore_wind,
    unsafe_offshore_flags,
)

SOUTH_BEACH = 180  # the sea is to the south, like Alaçatı bay


@pytest.mark.parametrize(
    ("wind_from", "expected"),
    [
        (180, "onshore"),  # southerly wind comes from the sea
        (225, "onshore"),
        (90, "side-shore"),
        (270, "side-shore"),
        (0, "offshore"),  # northerly wind comes from the land
        (315, "offshore"),
        (45, "offshore"),
    ],
)
def test_shore_wind_on_a_south_facing_beach(wind_from, expected):
    assert shore_wind(wind_from, SOUTH_BEACH) == expected


def test_angles_wrap_around_north():
    assert angle_between(350, 10) == 20
    assert shore_wind(10, 350) == "onshore"  # a north-facing beach, wind from the north


def test_unknown_shore_direction_is_never_called_offshore():
    assert shore_wind(0, None) == "unknown"
    conditions = pd.DataFrame({"wind_direction_10m": [0.0], "shore_facing": [None]})
    assert not offshore_flags(conditions).any()
    assert not offshore_flags(pd.DataFrame({"wind_direction_10m": [0.0]})).any()


def test_offshore_wind_is_unsafe_above_the_profile_limit():
    sup = get_profile("sup", "intermediate")  # max offshore 5 kn
    assert not is_unsafe(sup, 4, 0.2, True, offshore=True)
    assert is_unsafe(sup, 6, 0.2, True, offshore=True)
    assert not is_unsafe(sup, 6, 0.2, True, offshore=False)  # same wind from the sea


def test_kitesurf_never_in_offshore_wind():
    kite = get_profile("kitesurf", "advanced")
    assert label_conditions(kite, 20, 0.5, True, offshore=False) == "good"
    assert label_conditions(kite, 20, 0.5, True, offshore=True) == "bad"


def test_offshore_feature_is_positive_only_when_unsafe():
    hours = pd.DataFrame(
        {
            "wind_speed_10m": [8.0, 8.0],
            "wind_direction_10m": [0.0, 180.0],  # from the land, from the sea
            "wave_height": [0.2, 0.2],
            "is_day": [True, True],
            "shore_facing": [SOUTH_BEACH, SOUTH_BEACH],
        }
    )
    features = make_features(hours, get_profile("sup", "intermediate"))
    assert list(features["offshore_over_max"]) == [3.0, -5.0]  # 8 - 5, 0 - 5


def test_location_sports_must_be_known():
    entry = """
[[location]]
name = "Örnek Koyu"
region = "İzmir"
latitude = 38.3
longitude = 26.4
shore_facing = 180
sports = ["windsurf", "skiing"]
"""
    with pytest.raises(CatalogError, match="unknown sports"):
        parse_locations(entry)
    [loc] = parse_locations(entry.replace(', "skiing"', ""))
    assert loc.sports == ("windsurf",) and loc.supports("windsurf")
    assert not loc.supports("sup")


def test_recommend_skips_spots_where_the_sport_is_not_done(monkeypatch):
    # `spotcast.recommend` is also the name of a function in the package: get the module
    module = importlib.import_module("spotcast.recommend")

    [place] = parse_locations(
        """
[[location]]
name = "Rüzgâr Koyu"
region = "İzmir"
latitude = 38.3
longitude = 26.4
sports = ["windsurf"]
"""
    )
    monkeypatch.setattr(module, "LOCATIONS", [place])
    hours = pd.date_range("2026-11-07 10:00", periods=4, freq="h")
    predicted = pd.DataFrame(
        {
            "spot": "Rüzgâr Koyu",
            "time": hours,
            "wind_speed_10m": 3.0,
            "wave_height": 0.1,
            "prediction": "good",
        }
    )
    assert recommend(predicted, get_profile("sup", "beginner"), Preferences()).empty
    windsurf = get_profile("windsurf", "beginner")
    assert not recommend(predicted, windsurf, Preferences()).empty


def test_sheltered_bay_offshore_is_not_a_safety_rule():
    """Alaçatı: drifting ends at the other shore of the bay, not the open sea."""
    hours = pd.DataFrame(
        {
            "wind_speed_10m": [8.0, 8.0],
            "wind_direction_10m": [0.0, 0.0],  # from the land at both
            "wave_height": [0.2, 0.2],
            "is_day": [True, True],
            "shore_facing": [SOUTH_BEACH, SOUTH_BEACH],
            "sheltered": [False, True],
        }
    )
    assert list(offshore_flags(hours)) == [True, True]  # still offshore, geometrically
    assert list(unsafe_offshore_flags(hours)) == [True, False]
    sup = get_profile("sup", "intermediate")  # max offshore 5 kn
    assert list(add_labels(hours, sup)["label"]) == ["bad", "good"]


def test_alacati_is_the_sheltered_spot():
    assert BY_NAME["Alaçatı"].sheltered
    assert not BY_NAME["Kuşadası Pamucak"].sheltered


def test_sheltered_offshore_hours_come_with_an_explanation(monkeypatch):
    from spotcast import tools

    hours = pd.date_range("2026-11-07 00:00", periods=24, freq="h")
    forecast = pd.DataFrame(
        {
            "time": hours,
            "wind_speed_10m": 16.0,
            "wind_direction_10m": 300.0,  # north-west: from the land at Alaçatı
            "is_day": hours.hour.isin(range(7, 18)),
            "wave_height": 0.4,
            "wave_period": 3.0,
        }
    )
    bundle = {"model": AlwaysGood(), "features": FEATURES}
    alacati = BY_NAME["Alaçatı"]
    result = tools.assess_day(
        forecast, hours[0].date(), alacati.latitude, alacati.longitude,
        get_profile("windsurf", "intermediate"), bundle, hours[0],
    )  # fmt: skip
    assert any("sheltered bay" in w for w in result["warnings"])
    assert result["label_counts"] == {"good": 11}  # 16 kn offshore, but in the bay
    assert all(h["shore_wind"] == "offshore" for h in result["daylight_hours"])


class AlwaysGood:
    def predict(self, X):
        return np.array(["good"] * len(X))
