"""Tests for splitting data and the safety guardrail on top of the model."""

import numpy as np
import pandas as pd
import pytest

from spotcast.model import (
    FEATURES,
    TARGET,
    load_model,
    predict_conditions,
    predict_with_guardrail,
    save_model,
    split_by_year,
    train_tree,
)
from spotcast.profiles import WINDSURF_INTERMEDIATE


class AlwaysGoodModel:
    """A fake model that says "good" to everything: the worst possible model for safety."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array(["good"] * len(X))


def make_conditions(rows: list[tuple[float, float, bool]]) -> pd.DataFrame:
    """Build a features table from (wind_kn, wave_m, is_day) tuples."""
    return pd.DataFrame(
        [
            {
                "wind_speed_10m": wind,
                "wind_direction_10m": 0,
                "wave_height": wave,
                "wave_period": 4.0,
                "is_day": day,
            }
            for wind, wave, day in rows
        ],
        columns=FEATURES,
    )


def test_guardrail_overrides_model_on_unsafe_conditions():
    X = make_conditions(
        [
            (40, 0.5, True),  # storm
            (18, 2.5, True),  # big waves
            (18, 0.5, False),  # night
        ]
    )

    assert predict_with_guardrail(AlwaysGoodModel(), X, WINDSURF_INTERMEDIATE) == [
        "bad",
        "bad",
        "bad",
    ]


def test_guardrail_keeps_model_prediction_when_safe():
    X = make_conditions([(18, 0.5, True), (5, 0.1, True)])

    # Low wind is not "unsafe", so the guardrail does not touch it
    assert predict_with_guardrail(AlwaysGoodModel(), X, WINDSURF_INTERMEDIATE) == [
        "good",
        "good",
    ]


def test_split_by_year_puts_each_year_on_the_right_side():
    df = make_conditions([(18, 0.5, True)] * 4)
    df["time"] = pd.to_datetime(
        ["2024-06-01 12:00", "2024-12-31 23:00", "2025-01-01 00:00", "2025-06-01 12:00"]
    )
    df[TARGET] = ["good", "good", "bad", "bad"]

    X_train, X_test, y_train, y_test = split_by_year(df, test_year=2025)

    assert len(X_train) == 2
    assert len(X_test) == 2
    assert list(y_train) == ["good", "good"]
    assert list(y_test) == ["bad", "bad"]
    assert list(X_train.columns) == FEATURES  # no "time" or "label" leaks into X


def test_save_and_load_model_round_trip(tmp_path):
    X = make_conditions([(18, 0.5, True), (5, 0.1, True), (18, 0.5, False)])
    y = ["good", "bad", "bad"]
    tree = train_tree(X, pd.Series(y))

    path = save_model(tree, WINDSURF_INTERMEDIATE, [2024], models_dir=tmp_path)
    bundle = load_model(WINDSURF_INTERMEDIATE, models_dir=tmp_path)

    assert path.name == "windsurf_intermediate.joblib"
    assert bundle["features"] == FEATURES
    assert bundle["profile"] == WINDSURF_INTERMEDIATE
    assert bundle["train_years"] == [2024]
    assert list(bundle["model"].predict(X)) == y  # same answers after loading


def test_load_model_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run `uv run spotcast` first"):
        load_model(WINDSURF_INTERMEDIATE, models_dir=tmp_path)


def test_predict_conditions_applies_guardrail():
    bundle = {
        "model": AlwaysGoodModel(),
        "features": FEATURES,
        "profile": WINDSURF_INTERMEDIATE,
    }
    forecast = make_conditions([(18, 0.5, True), (40, 0.5, True)])
    forecast["spot"] = "Alaçatı"

    result = predict_conditions(bundle, forecast)

    assert list(result["prediction"]) == ["good", "bad"]
    assert "prediction" not in forecast.columns  # input is untouched
