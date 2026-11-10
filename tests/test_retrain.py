"""Tests for champion/challenger retraining. Synthetic history, temporary model folders."""

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from spotcast import retrain
from spotcast.model import load_model

TODAY = date(2026, 11, 16)  # data until 9 Nov is settled; holdout 3-9 Nov


def synthetic_history(spots, start, end):
    """Hourly wind and waves with a daily rhythm, so every label appears."""
    hours = pd.date_range(f"{start} 00:00", f"{end} 23:00", freq="h")
    frames = []
    for i, spot in enumerate(spots):
        phase = np.arange(len(hours)) / 24 * 2 * np.pi
        frames.append(
            pd.DataFrame(
                {
                    "spot": spot.name,
                    "time": hours,
                    "wind_speed_10m": 14 + 9 * np.sin(phase + i),
                    "wind_direction_10m": 0,
                    "is_day": hours.hour.isin(range(7, 18)),
                    "wave_height": 0.6 + 0.5 * np.cos(phase / 3 + i),
                    "wave_direction": 0,
                    "wave_period": 4.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def short_history(monkeypatch):
    """Start the history in September so tests train in a blink."""
    monkeypatch.setattr(retrain, "HISTORY_START", date(2026, 9, 1))


def scores(macro_f1, unsafe=0):
    return {"macro_f1": macro_f1, "accuracy": macro_f1, "unsafe_called_good": unsafe}


def test_decide_rejects_an_unsafe_challenger_first():
    decision = retrain.decide(scores(0.5), scores(0.99, unsafe=1))
    assert not decision.promote and "dangerous" in decision.reason


def test_decide_rejects_a_worse_challenger():
    assert not retrain.decide(scores(0.95), scores(0.90)).promote


def test_decide_promotes_a_tie_because_its_data_is_newer():
    decision = retrain.decide(scores(0.95), scores(0.945))  # within the tolerance
    assert decision.promote and "newer data" in decision.reason


def test_decide_promotes_the_first_model():
    assert retrain.decide(None, scores(0.8)).promote


def test_holdout_is_the_latest_settled_week():
    assert retrain.holdout_window(date(2026, 10, 24), TODAY) == (
        date(2026, 11, 3),
        date(2026, 11, 9),
    )


def test_no_holdout_when_the_champion_has_seen_those_days():
    assert retrain.holdout_window(date(2026, 11, 5), TODAY) is None


def test_first_run_promotes_and_writes_model_and_card(tmp_path, short_history):
    out = tmp_path / "candidate"
    report = retrain.retrain(
        out, TODAY, models_dir=tmp_path / "none", fetch=synthetic_history
    )

    assert report["promote"] and report["new_version"] == 1
    bundle = load_model(out)
    assert (
        bundle["train_end"] == "2026-11-02"
    )  # the day before the holdout: never saw it
    card = json.loads((out / retrain.CARD).read_text())
    assert card["holdout"] == ["2026-11-03", "2026-11-09"]
    assert card["holdout_scores"]["unsafe_called_good"] == 0
    assert len(card["profiles"]) == 12  # one model judged on every profile
    assert set(card["holdout_scores"]["per_profile_macro_f1"]) == set(card["profiles"])


def test_second_round_compares_with_the_champion(tmp_path, short_history):
    first = tmp_path / "v1"
    retrain.retrain(first, TODAY, models_dir=tmp_path / "none", fetch=synthetic_history)

    one_week_later = date(2026, 11, 23)
    out = tmp_path / "v2"
    report = retrain.retrain(
        out, one_week_later, models_dir=first, fetch=synthetic_history
    )

    assert report["champion_version"] == 1
    assert report["champion"] is not None and report["challenger"] is not None
    assert report["promote"] and report["new_version"] == 2


def test_nothing_happens_without_new_data(tmp_path, short_history):
    first = tmp_path / "v1"
    retrain.retrain(first, TODAY, models_dir=tmp_path / "none", fetch=synthetic_history)

    def must_not_fetch(*args):
        raise AssertionError("no new data: nothing should be downloaded")

    out = tmp_path / "again"
    report = retrain.retrain(out, TODAY, models_dir=first, fetch=must_not_fetch)

    assert not report["promote"] and "Not enough new data" in report["reason"]
    assert not (out / "suitability.joblib").exists()


def test_old_format_champion_is_replaced_and_its_version_continues(
    tmp_path, short_history
):
    """The `models` branch held one model per profile before Stage 11."""
    old = tmp_path / "old"
    old.mkdir()
    (old / "windsurf_intermediate.joblib").write_bytes(b"old format")
    (old / retrain.CARD).write_text(json.dumps({"version": 2}))

    out = tmp_path / "new"
    report = retrain.retrain(out, TODAY, models_dir=old, fetch=synthetic_history)

    assert report["champion"] is None  # nothing comparable to judge against
    assert report["promote"] and report["new_version"] == 3


def test_summary_mentions_the_decision():
    report = {"promote": False, "reason": "Challenger is worse."}
    assert "No change" in retrain.summary_markdown(report)
