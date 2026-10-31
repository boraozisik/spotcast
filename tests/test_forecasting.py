"""Tests for building the wind forecasting table."""

import numpy as np
import pandas as pd
import pytest

from spotcast.forecasting import FORECAST_FEATURES, make_forecast_table


def make_history(spot_winds: dict[str, list[float]]) -> pd.DataFrame:
    """Hourly history for several spots. Each spot's wind list is one value per hour."""
    frames = []
    for spot, winds in spot_winds.items():
        hours = len(winds)
        frames.append(
            pd.DataFrame(
                {
                    "spot": spot,
                    "time": pd.date_range("2024-07-01", periods=hours, freq="h"),
                    "wind_speed_10m": winds,
                    "wind_direction_10m": 0,
                    "wave_height": 0.5,
                    "wave_period": 4.0,
                    "is_day": True,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_target_is_the_wind_horizon_hours_later():
    winds = list(range(100))  # wind equals the hour number, easy to check
    table = make_forecast_table(make_history({"A": winds}), horizon_hours=2)

    # Each row's answer is the wind 2 hours later, and the 1 hour lag is 1 hour earlier
    assert (table["wind_in_2h"] == table["wind_speed_10m"] + 2).all()
    assert (table["wind_lag_1h"] == table["wind_speed_10m"] - 1).all()


def test_shifts_never_cross_from_one_spot_to_another():
    history = make_history({"A": [5.0] * 100, "B": [20.0] * 100})
    table = make_forecast_table(history, horizon_hours=2)

    # If B's values leaked into A, some of A's targets or lags would be 20
    a = table[table["spot"] == "A"]
    assert (a["wind_in_2h"] == 5.0).all()
    assert (a["wind_lag_48h"] == 5.0).all()


def test_rows_without_full_past_or_future_are_dropped():
    table = make_forecast_table(make_history({"A": [5.0] * 100}), horizon_hours=2)

    # 48 hours are needed for the longest lag, 2 hours for the future answer
    assert len(table) == 100 - 48 - 2
    assert not table[FORECAST_FEATURES].isna().any().any()


def test_cyclical_hours_put_midnight_next_to_23():
    table = make_forecast_table(make_history({"A": [5.0] * 100}), horizon_hours=2)
    at = table.set_index(table["time"].dt.hour)
    h23 = np.array([at.loc[23, "hour_sin"].iloc[0], at.loc[23, "hour_cos"].iloc[0]])
    h0 = np.array([at.loc[0, "hour_sin"].iloc[0], at.loc[0, "hour_cos"].iloc[0]])
    h12 = np.array([at.loc[12, "hour_sin"].iloc[0], at.loc[12, "hour_cos"].iloc[0]])

    assert np.linalg.norm(h23 - h0) < np.linalg.norm(h12 - h0)


def test_gaps_in_hours_are_rejected():
    history = make_history({"A": [5.0] * 100}).drop(index=10)  # one missing hour
    with pytest.raises(ValueError, match="consecutive hourly rows"):
        make_forecast_table(history)
