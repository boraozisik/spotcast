"""Forecasting the wind 48 hours ahead with an LSTM (PyTorch), compared with XGBoost.

XGBoost reads a table of hand-made features (lags, changes, averages). The LSTM instead
reads the last WINDOW_HOURS hours one by one, in order, and learns by itself what to remember.

Run with: uv run spotcast-lstm
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from spotcast.config import HISTORY_END, HISTORY_START
from spotcast.data import load_history
from spotcast.forecasting import HORIZON_HOURS, TARGET, cyclical, make_forecast_table
from spotcast.wind_model import (
    SPOT_NAMES,
    climatology_forecast,
    compare,
    model_inputs,
    persistence_forecast,
    target_time,
    train_xgboost,
)

WINDOW_HOURS = 48  # how many past hours the LSTM reads for each prediction
STEP_FEATURES = [  # what the LSTM sees at every hour of the window
    "wind_speed_10m",
    "wind_dir_sin",
    "wind_dir_cos",
    "wave_height",
    "wave_period",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]
VALIDATION_YEAR = 2024  # used to decide when to stop training
TEST_YEAR = 2025
SEED = 42


@dataclass
class Sequences:
    """LSTM inputs and answers. Row i of every array belongs to the same example."""

    windows: np.ndarray  # (examples, WINDOW_HOURS, len(STEP_FEATURES))
    spots: np.ndarray  # (examples, number of spots), one-hot: which spot this is
    targets: np.ndarray  # (examples,), wind HORIZON_HOURS after the window ends
    keys: pd.DataFrame  # spot + time of the window's last hour, to line up with XGBoost

    def select(self, mask: np.ndarray) -> "Sequences":
        return Sequences(
            self.windows[mask],
            self.spots[mask],
            self.targets[mask],
            self.keys[mask].reset_index(drop=True),
        )


def make_sequences(
    history: pd.DataFrame,
    window_hours: int = WINDOW_HOURS,
    horizon_hours: int = HORIZON_HOURS,
    first_end: int = 48,
) -> Sequences:
    """Cut each spot's hourly history into overlapping windows, one per hour.

    The window ending at hour e covers hours e-window_hours+1 .. e, and its answer is the
    wind at hour e+horizon_hours. Windows never cross from one spot into another.
    first_end=48 makes the examples match the rows of the XGBoost table exactly.
    """
    df = history.sort_values(["spot", "time"]).reset_index(drop=True)
    df["wind_dir_sin"], df["wind_dir_cos"] = cyclical(df["wind_direction_10m"], 360)
    df["hour_sin"], df["hour_cos"] = cyclical(df["time"].dt.hour, 24)
    df["month_sin"], df["month_cos"] = cyclical(df["time"].dt.month - 1, 12)

    windows, spots, targets, keys = [], [], [], []
    for spot, group in df.groupby("spot", sort=False):
        values = group[STEP_FEATURES].to_numpy(dtype=np.float32)
        wind = group["wind_speed_10m"].to_numpy(dtype=np.float32)
        n = len(group)

        ends = np.arange(max(first_end, window_hours - 1), n - horizon_hours)
        all_windows = sliding_window_view(values, (window_hours, len(STEP_FEATURES)))
        # all_windows[i, 0] is the window that starts at row i, so it ends at row i+window-1
        windows.append(all_windows[ends - window_hours + 1, 0])
        targets.append(wind[ends + horizon_hours])

        one_hot = np.zeros(len(SPOT_NAMES), dtype=np.float32)
        one_hot[SPOT_NAMES.index(spot)] = 1.0
        spots.append(np.tile(one_hot, (len(ends), 1)))
        keys.append(
            pd.DataFrame({"spot": spot, "time": group["time"].to_numpy()[ends]})
        )

    return Sequences(
        np.concatenate(windows),
        np.concatenate(spots),
        np.concatenate(targets),
        pd.concat(keys, ignore_index=True),
    )


@dataclass
class Scaler:
    """Standardize numbers to mean 0, std 1. Fit on training data ONLY (no leakage)."""

    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: float
    target_std: float

    @classmethod
    def fit(cls, train: Sequences) -> "Scaler":
        flat = train.windows.reshape(-1, train.windows.shape[-1])
        return cls(
            flat.mean(axis=0),
            flat.std(axis=0) + 1e-6,  # avoid dividing by zero
            float(train.targets.mean()),
            float(train.targets.std()),
        )

    def windows(self, seq: Sequences) -> np.ndarray:
        return (seq.windows - self.feature_mean) / self.feature_std

    def targets(self, seq: Sequences) -> np.ndarray:
        return (seq.targets - self.target_mean) / self.target_std

    def to_knots(self, scaled: np.ndarray) -> np.ndarray:
        return scaled * self.target_std + self.target_mean


class WindLSTM(nn.Module):
    """LSTM reads the window hour by hour; a small head turns its memory into knots."""

    def __init__(self, n_features: int, n_spots: int, hidden_size: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + n_spots, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, windows: torch.Tensor, spots: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(windows)  # hidden: the memory after the last hour
        summary = torch.cat([hidden[-1], spots], dim=1)
        return self.head(summary).squeeze(1)


def pick_device() -> torch.device:
    """Use the Apple GPU (MPS) when available, otherwise the CPU."""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _tensors(
    seq: Sequences, scaler: Scaler, device: torch.device
) -> list[torch.Tensor]:
    return [
        torch.tensor(scaler.windows(seq), dtype=torch.float32, device=device),
        torch.tensor(seq.spots, device=device),
        torch.tensor(scaler.targets(seq), dtype=torch.float32, device=device),
    ]


def predict_knots(
    model: WindLSTM, seq: Sequences, scaler: Scaler, device: torch.device
) -> np.ndarray:
    windows, spots, _ = _tensors(seq, scaler, device)
    model.eval()  # switch off training-only behaviour
    with torch.no_grad():  # no learning here, so skip the gradient bookkeeping
        scaled = model(windows, spots).cpu().numpy()
    return scaler.to_knots(scaled)


def train_lstm(
    train: Sequences,
    validation: Sequences,
    scaler: Scaler,
    max_epochs: int = 20,
    patience: int = 3,
    batch_size: int = 256,
) -> WindLSTM:
    """Train with early stopping: stop when validation error hasn't improved for `patience` epochs."""
    torch.manual_seed(SEED)
    device = pick_device()
    model = WindLSTM(len(STEP_FEATURES), len(SPOT_NAMES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(*_tensors(train, scaler, device)),
        batch_size=batch_size,
        shuffle=True,  # fine here: each window is a complete, independent example
        generator=torch.Generator().manual_seed(SEED),
    )

    best_mae, best_state, epochs_without_improvement = float("inf"), None, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for windows, spots, targets in loader:
            optimizer.zero_grad()  # forget the previous batch's corrections
            loss = loss_fn(model(windows, spots), targets)  # how wrong are we?
            loss.backward()  # work out which way to nudge every weight
            optimizer.step()  # nudge them

        val_pred = predict_knots(model, validation, scaler, device)
        val_mae = float(np.mean(np.abs(val_pred - validation.targets)))
        print(f"  epoch {epoch:2d}  validation MAE {val_mae:.3f}")

        if val_mae < best_mae:
            best_mae, epochs_without_improvement = val_mae, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  stopping: no improvement for {patience} epochs")
                break

    model.load_state_dict(best_state)
    return model


def main() -> None:
    pd.set_option("display.width", None)
    history = load_history(HISTORY_START, HISTORY_END)

    # Same examples for both models: one row per (spot, hour), split by the predicted time
    table = make_forecast_table(history)
    seq = make_sequences(history)
    assert len(seq.targets) == len(table), "LSTM and XGBoost examples must match"
    table = table.sort_values(["spot", "time"]).reset_index(drop=True)
    seq_order = seq.keys.merge(
        table.reset_index()[["spot", "time", "index"]], on=["spot", "time"]
    )
    table = table.loc[seq_order["index"]].reset_index(drop=True)

    year = target_time(table).dt.year.to_numpy()
    train_mask, val_mask, test_mask = (
        year < VALIDATION_YEAR,
        year == VALIDATION_YEAR,
        year == TEST_YEAR,
    )
    print(
        f"Train (until {VALIDATION_YEAR - 1}): {train_mask.sum()}, "
        f"validation ({VALIDATION_YEAR}): {val_mask.sum()}, test ({TEST_YEAR}): {test_mask.sum()}"
    )

    train_seq, val_seq, test_seq = (
        seq.select(train_mask),
        seq.select(val_mask),
        seq.select(test_mask),
    )
    scaler = Scaler.fit(train_seq)

    print(f"\nTraining LSTM on {pick_device()}...")
    lstm = train_lstm(train_seq, val_seq, scaler)

    train_table, test_table = table[train_mask], table[test_mask]
    # n_jobs=1: XGBoost and PyTorch crash together on macOS if XGBoost uses many threads
    print("Training XGBoost for comparison (single thread, takes a little longer)...")
    xgb_same_years = train_xgboost(train_table, n_jobs=1)  # same years as the LSTM
    xgb_more_years = train_xgboost(table[train_mask | val_mask], n_jobs=1)

    forecasts = {
        "persistence": persistence_forecast(test_table),
        "climatology": climatology_forecast(train_table, test_table),
        "xgboost (2022-2023)": pd.Series(
            xgb_same_years.predict(model_inputs(test_table)), index=test_table.index
        ),
        "xgboost (2022-2024)": pd.Series(
            xgb_more_years.predict(model_inputs(test_table)), index=test_table.index
        ),
        "lstm (2022-2023)": pd.Series(
            predict_knots(lstm, test_seq, scaler, pick_device()), index=test_table.index
        ),
    }
    print(f"\n=== Wind in {HORIZON_HOURS} h, test year {TEST_YEAR} (knots) ===")
    print(compare(test_table, forecasts))
    assert np.allclose(test_table[TARGET].to_numpy(), test_seq.targets), (
        "targets must line up"
    )
