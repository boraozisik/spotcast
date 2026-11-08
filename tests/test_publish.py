"""Tests for the website's batch job. A fake model and a fixed forecast; no network."""

import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from spotcast import publish
from spotcast.model import FEATURES
from spotcast.profiles import WINDSURF_INTERMEDIATE

NOW = datetime(2026, 11, 5, 8, 0)  # Thursday morning


class AlwaysGoodModel:
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array(["good"] * len(X))


BUNDLE = {
    "model": AlwaysGoodModel(),
    "features": FEATURES,
    "profile": WINDSURF_INTERMEDIATE,
}


def forecast(days=7, waves=True):
    hours = pd.date_range("2026-11-05 00:00", periods=days * 24, freq="h")
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


def test_slug_is_safe_for_urls():
    assert publish.slug("Kuşadası Pamucak") == "kusadasi-pamucak"
    assert publish.slug("Gökçeada Aydıncık") == "gokceada-aydincik"


def test_local_now_has_no_timezone_like_the_forecast():
    assert publish.local_now().tzinfo is None


def test_place_has_seven_days_with_turkish_notes():
    place = publish.build_place(
        "Mezitli sahili",
        "Mersin",
        36.735,
        34.53,
        forecast(),
        WINDSURF_INTERMEDIATE,
        BUNDLE,
        NOW,
    )

    assert place["slug"] == "mezitli-sahili" and not place["guided"]
    assert len(place["days"]) == 7
    first, last = place["days"][0], place["days"][-1]
    assert first["date"] == "2026-11-05"
    assert "day" not in first  # day names are written by the site, in its language
    assert (
        first["best_window"]["hours"] == "08:00-17:00"
    )  # the past hours of today are skipped
    assert "warnings" not in first  # the English warnings are for the LLM, not the site
    assert {"code": "no_guide"} in first["notes"]  # Mersin is far from our guided spots
    assert {"code": "lead_time", "days": 6} in last["notes"]  # far-ahead warning


def test_place_without_waves_says_so():
    place = publish.build_place(
        "Göl", "Isparta", 37.87, 30.85, forecast(days=1, waves=False),
        WINDSURF_INTERMEDIATE, BUNDLE, NOW,
    )  # fmt: skip
    day = place["days"][0]
    assert day["best_window"] is None
    assert {"code": "no_waves"} in day["notes"]


def test_guided_spot_has_no_missing_guide_note():
    place = publish.build_place(
        "Alaçatı", None, 38.255, 26.375, forecast(days=1),
        WINDSURF_INTERMEDIATE, BUNDLE, NOW,
    )  # fmt: skip
    assert place["guided"]
    assert {"code": "no_guide"} not in place["days"][0]["notes"]


def test_recommendations_have_both_presets():
    predicted = forecast().assign(spot="Alaçatı", prediction="good")

    recs = publish.build_recommendations(predicted, WINDSURF_INTERMEDIATE, NOW)

    assert set(recs) == {"any", "weekend"}
    weekend_days = {
        datetime.fromisoformat(w["date"]).weekday() for w in recs["weekend"]["windows"]
    }
    assert weekend_days <= {5, 6}  # Saturday, Sunday
    best = recs["any"]["windows"][0]
    # 09:00-18:00 inclusive = 10 hours (labels set directly here, no guardrail)
    assert (best["start"], best["end"], best["good_hours"]) == ("09:00", "18:00", 10)


def test_written_json_keeps_turkish_letters(tmp_path):
    path = tmp_path / "x.json"
    publish.write_json(path, {"spot": "Alaçatı"})
    assert "Alaçatı" in path.read_text(encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8")) == {"spot": "Alaçatı"}


@pytest.mark.parametrize("name", [p.name for p in publish.PLACES])
def test_places_have_unique_slugs(name):
    slugs = [publish.slug(p.name) for p in publish.PLACES]
    assert slugs.count(publish.slug(name)) == 1


def test_fetch_retries_a_dropped_connection(monkeypatch):
    from spotcast import data

    calls = []

    class Ok:
        def raise_for_status(self):
            pass

    def flaky_get(url, params, timeout):
        calls.append(url)
        if len(calls) < 3:
            raise data.requests.Timeout("slow")
        return Ok()

    monkeypatch.setattr(data.requests, "get", flaky_get)
    monkeypatch.setattr(data.time, "sleep", lambda seconds: None)  # don't really wait

    assert isinstance(data._get_with_retries("https://example", {}), Ok)
    assert len(calls) == 3


def test_fetch_gives_up_after_the_last_retry(monkeypatch):
    from spotcast import data

    def always_down(url, params, timeout):
        raise data.requests.ConnectionError("down")

    monkeypatch.setattr(data.requests, "get", always_down)
    monkeypatch.setattr(data.time, "sleep", lambda seconds: None)

    with pytest.raises(data.requests.ConnectionError):
        data._get_with_retries("https://example", {})


def test_today_after_sunset_is_left_out():
    evening = datetime(2026, 11, 5, 19, 0)  # daylight ended at 17:00
    place = publish.build_place(
        "Kaş",
        "Antalya",
        36.195,
        29.64,
        forecast(days=2),
        WINDSURF_INTERMEDIATE,
        BUNDLE,
        evening,
    )
    assert [d["date"] for d in place["days"]] == ["2026-11-06"]


def test_spot_hours_start_at_the_current_hour():
    predicted = forecast(days=1).assign(spot="Alaçatı", prediction="good")
    spots = publish.build_spots(predicted, datetime(2026, 11, 5, 13, 30))
    alacati = next(s for s in spots if s["name"] == "Alaçatı")
    assert alacati["hours"][0]["time"] == "2026-11-05T13:00"
