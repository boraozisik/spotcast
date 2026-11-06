"""API tests. Real models and the weather service are replaced with small fakes."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from spotcast.api import (
    app,
    get_bundles,
    get_forecast,
    get_llm,
    get_now,
    get_retriever,
)
from spotcast.explain import Explanation, Tip
from spotcast.llm import LLMOutputError, LLMUnavailableError
from spotcast.model import FEATURES
from spotcast.profiles import WINDSURF_INTERMEDIATE
from spotcast.rag import Chunk, SearchResult


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


def fake_retrieve(question, spots):
    chunk = Chunk(
        file="spots/alacati.md",
        title="Alaçatı spot rehberi",
        section="Alaçatı: tehlikeler",
        spot="Alaçatı",
        source="test notları",
        content="Yazın çok kalabalık olur.",
    )
    return [SearchResult(chunk=chunk, distance=0.3)]


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
    app.dependency_overrides[get_retriever] = lambda: fake_retrieve
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
            headline="Cumartesi Alaçatı",
            reason="18 knot",
            alternatives=[],
            tips=[Tip(text="Kalabalığa dikkat et.", source_ids=[1])],
        )

    app.dependency_overrides[get_llm] = lambda: fake_llm
    body = client.get("/explain", params=WINDSURF).json()

    assert body["explanation"]["headline"] == "Cumartesi Alaçatı"
    assert body["explanation"]["tips"][0]["source_ids"] == [1]
    assert body["sources"] == [
        {
            "id": 1,
            "title": "Alaçatı spot rehberi",
            "section": "Alaçatı: tehlikeler",
            "source": "test notları",
        }
    ]
    assert len(body["windows"]) == 1
    assert "kendin kontrol et" in body["caution"]


def test_window_far_ahead_has_a_warning(client):
    near = client.get("/recommend", params=WINDSURF).json()["windows"][0]
    app.dependency_overrides[get_now] = lambda: datetime(2026, 11, 2, 8, 0)
    far = client.get("/recommend", params=WINDSURF).json()["windows"][0]

    assert near["warnings"] == []  # Saturday seen from Friday: 1 day ahead
    assert "5 gün sonrası" in far["warnings"][0]  # seen from Monday: 5 days ahead


def explanation():
    return Explanation(headline="h", reason="r", alternatives=[], tips=[])


def test_explain_retries_once_after_a_broken_llm_answer(client):
    calls = []

    def flaky_llm(messages, output_model):
        calls.append(1)
        if len(calls) == 1:
            raise LLMOutputError("not JSON")
        return explanation()

    app.dependency_overrides[get_llm] = lambda: flaky_llm
    response = client.get("/explain", params=WINDSURF)

    assert response.status_code == 200
    assert len(calls) == 2


def test_explain_returns_502_when_the_llm_answer_stays_broken(client):
    def broken_llm(messages, output_model):
        raise LLMOutputError("not JSON")

    app.dependency_overrides[get_llm] = lambda: broken_llm
    response = client.get("/explain", params=WINDSURF)

    assert response.status_code == 502
    assert "/recommend" in response.json()["detail"]  # tells the user what still works


def test_explain_returns_503_when_ollama_is_down(client):
    def offline_llm(messages, output_model):
        raise LLMUnavailableError("Ollama is not running.")

    app.dependency_overrides[get_llm] = lambda: offline_llm
    response = client.get("/explain", params=WINDSURF)

    assert response.status_code == 503
