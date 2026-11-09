"""Tests for the batch job's data quality checks."""

import numpy as np
import pandas as pd
import pytest

from spotcast import monitoring, publish
from spotcast.model import feature_ranges

BUNDLE = {"feature_ranges": {"wind_speed_10m": [0.0, 27.7], "wave_height": [0.0, 3.2]}}


def forecasts(wind=(10.0, 12.0), wave=(0.5, 0.6)):
    return pd.DataFrame(
        {
            "wind_speed_10m": list(wind),
            "wave_height": list(wave),
            "wave_period": [4.0, 4.0],
        }
    )


def test_clean_data_passes():
    report = monitoring.quality_report(forecasts(), BUNDLE, places_total=26, skipped=[])
    assert monitoring.problems(report) == []
    assert monitoring.notices(report) == []


def test_impossible_values_stop_publishing():
    report = monitoring.quality_report(forecasts(wind=(10.0, 250.0)), BUNDLE, 26, [])
    assert report["implausible_values"]["wind_speed_10m"] == 1
    assert any("implausible wind_speed_10m" in p for p in monitoring.problems(report))


def test_too_many_unreachable_places_stop_publishing():
    skipped = [f"place {i}" for i in range(6)]  # 6 of 26 > 20%
    report = monitoring.quality_report(forecasts(), BUNDLE, 26, skipped)
    assert monitoring.problems(report) == ["6 of 26 places could not be fetched."]


def test_a_few_unreachable_places_are_only_a_notice():
    report = monitoring.quality_report(forecasts(), BUNDLE, 26, ["Sinop"])
    assert monitoring.problems(report) == []
    assert monitoring.notices(report) == ["Skipped places: Sinop."]


def test_values_beyond_training_range_are_drift_not_a_stop():
    # 40 knots: a real storm, but the model never saw one. The guardrail handles the
    # verdict; monitoring records that the model is outside what it knows.
    report = monitoring.quality_report(forecasts(wind=(12.0, 40.0)), BUNDLE, 26, [])
    assert report["outside_training_range"]["wind_speed_10m"] == 1
    assert monitoring.problems(report) == []
    assert "outside the model's training range" in monitoring.notices(report)[0]


def test_missing_waves_are_counted_not_treated_as_implausible():
    report = monitoring.quality_report(forecasts(wave=(np.nan, 0.5)), BUNDLE, 26, [])
    assert report["missing_wave_share"] == pytest.approx(0.5)
    assert report["implausible_values"]["wave_height"] == 0


def test_old_models_without_ranges_still_work():
    report = monitoring.quality_report(forecasts(), {}, 26, [])
    assert report["outside_training_range"] == {}


def test_summary_is_written_on_github_actions(tmp_path, monkeypatch):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    report = monitoring.quality_report(forecasts(wind=(10.0, 250.0)), BUNDLE, 26, [])

    monitoring.write_run_summary(report)

    text = summary.read_text(encoding="utf-8")
    assert "Not published" in text and "implausible" in text


def test_feature_ranges_cover_numbers_only():
    X = pd.DataFrame({"wind_speed_10m": [3.0, 27.7], "is_day": [True, False]})
    assert feature_ranges(X) == {"wind_speed_10m": [3.0, 27.7]}


def test_bad_data_never_overwrites_published_files(tmp_path, monkeypatch):
    """The whole point: if the check fails, the previous good files stay untouched."""
    out = tmp_path / "site"
    out.mkdir()
    (out / "meta.json").write_text(
        '{"generated_at": "earlier, good"}', encoding="utf-8"
    )

    hours = pd.date_range("2026-11-05 00:00", periods=24, freq="h")
    broken = pd.DataFrame(
        {
            "time": hours,
            "wind_speed_10m": 999.0,  # broken upstream data
            "wind_direction_10m": 0,
            "is_day": hours.hour.isin(range(7, 18)),
            "wave_height": 0.5,
            "wave_direction": 0,
            "wave_period": 4.0,
        }
    )

    class GoodModel:
        def predict(self, X):
            return np.array(["good"] * len(X))

    bundle = {
        "model": GoodModel(),
        "features": publish.FEATURES,
        "profile": publish.get_profile("windsurf", "intermediate"),
        "train_years": [2024],
    }
    monkeypatch.setattr(publish, "ensure_model", lambda profile, retrain=False: bundle)
    monkeypatch.setattr(
        publish,
        "fetch_all_spots",
        lambda spots: pd.concat(broken.assign(spot=s.name) for s in spots),
    )
    monkeypatch.setattr(publish, "fetch_conditions", lambda lat, lon: broken)

    with pytest.raises(monitoring.DataQualityError):
        publish.publish(out)

    assert (out / "meta.json").read_text(
        encoding="utf-8"
    ) == '{"generated_at": "earlier, good"}'
    assert not (out / "places.json").exists()
