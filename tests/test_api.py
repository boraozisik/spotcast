"""API tests. Real models and the weather service are replaced with small fakes."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from spotcast.api import app, get_bundles, get_forecast, get_llm, get_now
from spotcast.explain import Explanation
from spotcast.llm import LLMUnavailableError
from spotcast.model import FEATURES
from spotcast.profiles import WINDSURF_INTERMEDIATE


class AlwaysGoodModel:
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array(["good"] * len(X))


def fake_forecast() -> pd.DataFrame:
    """Saturday 7 Nov 2026, 06:00-21:00 at spot Alaçatı, 18 kn and small waves."""
    hours = pd.date_range("2026-11-07 06:00", "2026-11-07 21:00", freq="h")
    return pd.DataFrame(
        {
            "spot": "Alaçatı",
            "time": hours,
            "wind_speed_10m": 18.0,
            "wind_direction_10m": 0,
            "wave_height": 0.5,
            "wave_period": 4.0,
            # dark before 07:00 and after 17:00
            "is_day": hours.hour.isin(range(7, 18)),
        }
    )


@pytest.fixture
def client():
    """A test client whose dependencies are fakes, like overriding providers in NestJS."""
    bundle = {
        "model": AlwaysGoodModel(),
        "features": FEATURES,
        "profile": WINDSURF_INTERMEDIATE,
    }
    app.dependency_overrides[get_bundles] = lambda: {
        ("windsurf", "intermediate"): bundle
    }
    app.dependency_overrides[get_forecast] = fake_forecast
    app.dependency_overrides[get_now] = lambda: datetime(2026, 11, 6, 12, 0)
    yield TestClient(app)
    app.dependency_overrides.clear()


WINDSURF = {"sport": "windsurf", "level": "intermediate"}


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_recommend_returns_best_window_within_daylight_and_hours(client):
    response = client.get("/recommend", params=WINDSURF | {"latest_hour": 18})

    assert response.status_code == 200
    windows = response.json()["windows"]
    assert len(windows) == 1
    # Model says "good" everywhere, but the guardrail turns night into "bad": 07:00-17:00
    assert windows[0]["start"] == "2026-11-07T09:00:00"  # earliest_hour default is 9
    assert windows[0]["end"] == "2026-11-07T17:00:00"
    assert windows[0]["good_hours"] == 9


def test_forecast_lists_every_hour_with_guardrail(client):
    hours = client.get("/forecast", params=WINDSURF).json()

    assert len(hours) == 16
    night = next(h for h in hours if h["time"] == "2026-11-07T21:00:00")
    assert night["prediction"] == "bad"


def test_unknown_profile_is_404(client):
    response = client.get("/recommend", params={"sport": "sup", "level": "beginner"})
    assert response.status_code == 404


@pytest.mark.parametrize(
    "params",
    [
        {"earliest_hour": 25},  # outside 0..23
        {"earliest_hour": 15, "latest_hour": 10},  # ends before it starts
        {"spots": ["Atlantis"]},  # no such spot
    ],
)
def test_bad_input_is_422(client, params):
    response = client.get("/recommend", params=WINDSURF | params)
    assert response.status_code == 422


def test_missing_sport_is_422(client):
    assert client.get("/recommend", params={"level": "intermediate"}).status_code == 422


def test_explain_adds_llm_text_and_fixed_caution(client):
    def fake_llm(messages, output_model):
        return Explanation(
            headline="Cumartesi Alaçatı", reason="18 knot", alternatives=[]
        )

    app.dependency_overrides[get_llm] = lambda: fake_llm
    body = client.get("/explain", params=WINDSURF).json()

    assert body["explanation"]["headline"] == "Cumartesi Alaçatı"
    assert len(body["windows"]) == 1
    assert "kendin kontrol et" in body["caution"]


def test_explain_returns_503_when_ollama_is_down(client):
    def offline_llm(messages, output_model):
        raise LLMUnavailableError("Ollama is not running.")

    app.dependency_overrides[get_llm] = lambda: offline_llm
    response = client.get("/explain", params=WINDSURF)

    assert response.status_code == 503
