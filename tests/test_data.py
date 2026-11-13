"""Tests for fetching: batched requests, Open-Meteo's call weights and the rate limiter."""

from datetime import date

import pytest

from spotcast import data
from spotcast.locations import PLACES, SPOTS, TRAINING_LOCATIONS


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise data.requests.HTTPError(f"HTTP {self.status_code}")


def hourly(variables, hours=3):
    times = [f"2026-11-07T{h:02d}:00" for h in range(hours)]
    values = {"is_day": 1}
    return {"time": times} | {v: [values.get(v, 1.0)] * hours for v in variables}


@pytest.fixture
def api(monkeypatch):
    """A fake Open-Meteo that answers every location in one list, and records requests."""
    requests_seen = []

    def fake_get(url, params, timeout):
        requests_seen.append(params)
        count = len(str(params["latitude"]).split(","))
        variables = params["hourly"]
        item = {"hourly": hourly(variables)}
        return FakeResponse([item] * count if count > 1 else item)

    monkeypatch.setattr(data.requests, "get", fake_get)
    monkeypatch.setattr(data, "LIMITER", data.RateLimiter(budget=10**9))
    return requests_seen


def test_call_weight_follows_open_meteos_rule():
    assert data.call_weight(1, 3, 7) == 1  # a 7-day forecast for one place
    assert data.call_weight(29, 3, 7) == 29  # each location counts
    assert data.call_weight(1, 15, 14) == 1.5  # their example: 15 variables
    assert data.call_weight(1, 3, 28) == 2  # their example: 4 weeks


def test_all_places_share_two_requests(api):
    table = data.fetch_all_spots(PLACES)
    assert len(api) == 2  # wind + waves, for all 26 places
    assert api[0]["latitude"].count(",") == len(PLACES) - 1
    assert set(table["spot"]) == {p.name for p in PLACES}
    assert {"shore_facing", "sheltered"} <= set(table.columns)


def test_one_location_still_works(api):
    df = data.fetch_conditions(SPOTS[0].latitude, SPOTS[0].longitude)
    assert len(api) == 2 and len(df) == 3
    assert df["is_day"].dtype == bool


def test_long_history_is_split_into_requests_the_limits_allow():
    five_years = data.batches(TRAINING_LOCATIONS, date(2022, 1, 1), date(2026, 10, 24))
    assert all(len(group) == 1 for group in five_years)  # ~126 calls per place already
    week = data.batches(TRAINING_LOCATIONS, None, None)
    assert len(week) == 1  # a forecast for all of them is one request


def test_rate_limiter_waits_when_the_minute_budget_is_spent(monkeypatch):
    clock = [0.0]
    waits = []

    def fake_sleep(seconds):
        waits.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(data.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(data.time, "sleep", fake_sleep)
    limiter = data.RateLimiter(budget=300)

    limiter.wait(126)
    limiter.wait(126)  # 252: still inside the budget, no waiting
    assert waits == []
    limiter.wait(
        126
    )  # 378 would be too much in one minute: wait for the first to expire
    assert len(waits) == 1 and clock[0] >= 60


def test_too_many_requests_waits_and_retries(monkeypatch):
    answers = [FakeResponse({}, status=429), FakeResponse({"hourly": {}})]
    waits = []
    monkeypatch.setattr(
        data.requests, "get", lambda url, params, timeout: answers.pop(0)
    )
    monkeypatch.setattr(data.time, "sleep", waits.append)

    response = data._get_with_retries("https://example", {})

    assert response.status_code == 200
    assert waits == [data.RATE_LIMIT_WAIT_SECONDS]


def test_training_locations_cover_several_seas():
    names = [loc.name for loc in TRAINING_LOCATIONS]
    assert names[:3] == [s.name for s in SPOTS]  # spots first, in their fixed order
    assert {"Kilyos", "Sinop", "Trabzon", "Kaş"} <= set(names)
    assert len(names) <= 10  # each one costs ~250 API calls per training run


def test_slow_server_is_waited_out_with_growing_pauses(monkeypatch):
    """13 Nov 2026: Open-Meteo timed out for about half a minute and the job gave up."""
    answers = [
        data.requests.Timeout("handshake timed out"),
        data.requests.Timeout("handshake timed out"),
        FakeResponse({}, status=503),  # overloaded: also worth another try
        FakeResponse({"hourly": {}}),
    ]
    waits = []

    def fake_get(url, params, timeout):
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(data.requests, "get", fake_get)
    monkeypatch.setattr(data.time, "sleep", waits.append)

    assert data._get_with_retries("https://example", {}).status_code == 200
    assert waits == [2, 5, 15]  # longer each time


def test_a_bad_request_is_not_retried(monkeypatch):
    calls = []
    monkeypatch.setattr(
        data.requests,
        "get",
        lambda *a, **k: calls.append(1) or FakeResponse({}, status=400),
    )
    monkeypatch.setattr(data.time, "sleep", lambda s: None)
    with pytest.raises(data.requests.HTTPError):
        data._get_with_retries("https://example", {})
    assert len(calls) == 1
