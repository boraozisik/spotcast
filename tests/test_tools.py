"""Tests for the agent tools. No network, model file or database: those parts are faked."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from spotcast import tools
from spotcast.model import FEATURES
from spotcast.profiles import WINDSURF_INTERMEDIATE

TODAY = datetime(2026, 11, 5, 8, 0)  # a Thursday morning


class AlwaysGoodModel:
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array(["good"] * len(X))


@pytest.fixture(autouse=True)
def fixed_clock_and_model(monkeypatch):
    """Every test runs at the same time, with a fake model instead of a file on disk."""
    monkeypatch.setattr(tools, "now", lambda: TODAY)
    bundle = {
        "model": AlwaysGoodModel(),
        "features": FEATURES,
        "profile": WINDSURF_INTERMEDIATE,
    }
    monkeypatch.setattr(tools, "_bundle", lambda sport, level: bundle)


def fake_forecast(latitude, longitude, waves=True):
    hours = pd.date_range("2026-11-05 00:00", periods=7 * 24, freq="h")
    return pd.DataFrame(
        {
            "time": hours,
            "wind_speed_10m": 18.0,
            "wind_direction_10m": 0,
            "is_day": hours.hour.isin(range(7, 18)),
            "wave_height": 0.5 if waves else np.nan,
            "wave_direction": 0,
            "wave_period": 4.0,
        }
    )


def test_schema_is_what_the_llm_reads():
    schema = tools.TOOLS_BY_NAME["check_conditions"].schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "check_conditions"
    assert {"latitude", "longitude", "day"} <= set(
        schema["function"]["parameters"]["required"]
    )


def test_bad_arguments_become_an_error_the_llm_can_read():
    result = tools.TOOLS_BY_NAME["find_location"].call({"nam": "Mersin"})
    assert "error" in result and "name" in result["error"]


def test_day_outside_the_forecast_is_refused():
    result = tools.TOOLS_BY_NAME["check_conditions"].call(
        {"latitude": 36.8, "longitude": 34.6, "day": "2030-01-01"}
    )
    assert "outside the forecast" in result["error"]


def test_unsupported_level_is_refused_by_the_schema():
    result = tools.TOOLS_BY_NAME["check_conditions"].call(
        {"latitude": 36.8, "longitude": 34.6, "day": "2026-11-07", "level": "beginner"}
    )
    # The schema only allows levels we have a profile for, and says which one
    assert "Invalid arguments" in result["error"]
    assert "intermediate" in result["error"]


def test_check_conditions_far_from_known_spots(monkeypatch):
    monkeypatch.setattr(tools, "fetch_conditions", fake_forecast)

    result = tools.check_conditions(
        tools.CheckConditionsArgs(latitude=36.78, longitude=34.6, day="2026-11-07")
    )

    assert result["day"] == "7 Kasım Cumartesi"
    assert len(result["daylight_hours"]) == 11  # 07:00-17:00
    assert (
        result["best_window"]["hours"] == "07:00-17:00"
    )  # the guardrail ends it at dark
    assert result["local_guide_spot"] is None  # Mersin is far from our spots
    assert any("no local guide" in w for w in result["warnings"])


def test_no_wave_data_means_no_verdict(monkeypatch):
    monkeypatch.setattr(
        tools, "fetch_conditions", lambda lat, lon: fake_forecast(lat, lon, waves=False)
    )

    result = tools.check_conditions(
        tools.CheckConditionsArgs(latitude=38.3, longitude=26.4, day="2026-11-07")
    )

    assert set(result["label_counts"]) == {"unknown"}
    assert result["best_window"] is None
    assert any("No wave data" in w for w in result["warnings"])
    assert result["local_guide_spot"] == "Alaçatı"  # within 15 km of a known spot


def test_distance_km():
    # Alaçatı to Akyaka is roughly 200 km in a straight line
    assert 180 < tools.distance_km(38.255, 26.375, 37.04, 28.31) < 230
    assert tools.distance_km(38.0, 27.0, 38.0, 27.0) == 0


def test_find_location_with_no_results(monkeypatch):
    class EmptyResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {}

    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: EmptyResponse())
    result = tools.TOOLS_BY_NAME["find_location"].call({"name": "Atlantis"})
    assert result == {"error": "No place called 'Atlantis' was found in Turkey."}


def test_turkish_words_from_the_llm_are_accepted():
    args = tools.CheckConditionsArgs(
        latitude=36.8,
        longitude=34.6,
        day="2026-11-07",
        sport="Rüzgâr sörfü",
        level="orta",
    )
    assert (args.sport, args.level) == ("windsurf", "intermediate")


def test_beginner_is_understood_but_not_supported_yet():
    result = tools.TOOLS_BY_NAME["recommend_spots"].call({"level": "yeni başlayan"})
    assert (
        "Invalid arguments" in result["error"]
    )  # mapped to "beginner", which has no profile
