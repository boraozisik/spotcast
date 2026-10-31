"""Tests for the wind forecasting baselines, metrics and data split."""

import pandas as pd
import pytest

from spotcast.forecasting import TARGET
from spotcast.wind_model import (
    climatology_forecast,
    regression_metrics,
    split_forecast_by_year,
)


def test_regression_metrics():
    metrics = regression_metrics(pd.Series([10, 10]), pd.Series([12, 6]))

    assert metrics["MAE"] == pytest.approx(3.0)  # misses of 2 and 4
    assert metrics["RMSE"] == pytest.approx(10**0.5)  # sqrt((4 + 16) / 2)
    assert metrics["bias"] == pytest.approx(-1.0)  # +2 and -4 average to -1: too low


def test_split_uses_the_year_of_the_predicted_time():
    table = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2024-06-01 12:00", "2024-12-31 12:00", "2025-06-01 12:00"]
            )
        }
    )

    train, test = split_forecast_by_year(table, test_year=2025)

    # 31 Dec 2024 + 48 h = 2 Jan 2025, so that row predicts the test year
    assert list(train["time"].dt.month) == [6]
    assert len(test) == 2


def test_climatology_only_learns_from_training_data():
    when = pd.to_datetime(["2024-07-01 12:00", "2024-07-02 12:00"])
    train = pd.DataFrame({"spot": "A", "time": when, TARGET: [10.0, 20.0]})
    test = pd.DataFrame(
        {"spot": "A", "time": when + pd.DateOffset(years=1), TARGET: [99.0, 99.0]}
    )

    forecast = climatology_forecast(train, test)

    # Same spot, month and hour -> the training average (15), never the test answers (99)
    assert list(forecast) == [15.0, 15.0]
