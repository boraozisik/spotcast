"""Tests for turning history into LSTM sequences, scaling, and the network's shape."""

import numpy as np
import pandas as pd
import torch

from spotcast.forecasting import make_forecast_table
from spotcast.lstm_model import STEP_FEATURES, Scaler, WindLSTM, make_sequences
from spotcast.wind_model import SPOT_NAMES


def make_history(spot_winds: dict[str, list[float]]) -> pd.DataFrame:
    frames = []
    for spot, winds in spot_winds.items():
        frames.append(
            pd.DataFrame(
                {
                    "spot": spot,
                    "time": pd.date_range("2024-07-01", periods=len(winds), freq="h"),
                    "wind_speed_10m": winds,
                    "wind_direction_10m": 0,
                    "wave_height": 0.5,
                    "wave_period": 4.0,
                    "is_day": True,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_window_ends_now_and_target_is_horizon_later():
    winds = [float(h) for h in range(100)]  # wind equals the hour number
    seq = make_sequences(
        make_history({SPOT_NAMES[0]: winds}),
        window_hours=4,
        horizon_hours=2,
        first_end=3,
    )

    wind_col = STEP_FEATURES.index("wind_speed_10m")
    first_window = seq.windows[0, :, wind_col]
    assert list(first_window) == [0.0, 1.0, 2.0, 3.0]  # hours 0..3, oldest first
    assert seq.targets[0] == 5.0  # 2 hours after the window's last hour
    assert len(seq.targets) == 100 - 3 - 2


def test_windows_never_mix_spots():
    history = make_history({SPOT_NAMES[0]: [5.0] * 100, SPOT_NAMES[1]: [20.0] * 100})
    seq = make_sequences(history, window_hours=4, horizon_hours=2, first_end=3)

    wind_col = STEP_FEATURES.index("wind_speed_10m")
    for spot_index, expected in [(0, 5.0), (1, 20.0)]:
        rows = seq.spots[:, spot_index] == 1.0
        assert (seq.windows[rows, :, wind_col] == expected).all()
        assert (seq.targets[rows] == expected).all()


def test_same_examples_as_the_xgboost_table():
    history = make_history({SPOT_NAMES[0]: list(np.random.rand(200) * 20)})

    seq = make_sequences(history)  # defaults: 48 h window, 48 h horizon
    table = make_forecast_table(history)

    assert len(seq.targets) == len(table)
    assert np.allclose(seq.targets, table["wind_in_48h"].to_numpy())


def test_scaler_uses_training_statistics_only():
    train = make_sequences(make_history({SPOT_NAMES[0]: [10.0, 20.0] * 60}))
    other = make_sequences(make_history({SPOT_NAMES[0]: [100.0] * 120}))

    scaler = Scaler.fit(train)

    assert scaler.target_mean == 15.0  # from training data, not affected by `other`
    assert np.allclose(scaler.to_knots(scaler.targets(other)), other.targets)


def test_network_returns_one_number_per_example():
    model = WindLSTM(n_features=len(STEP_FEATURES), n_spots=len(SPOT_NAMES))
    windows = torch.zeros(7, 48, len(STEP_FEATURES))
    spots = torch.zeros(7, len(SPOT_NAMES))

    assert model(windows, spots).shape == (7,)
