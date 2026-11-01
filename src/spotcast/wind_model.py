"""Forecasting the wind 48 hours ahead: baselines, an XGBoost model and their comparison.

Run with: uv run spotcast-forecast
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from spotcast.config import HISTORY_END, HISTORY_START, SPOTS
from spotcast.data import load_history
from spotcast.forecasting import (
    FORECAST_FEATURES,
    HORIZON_HOURS,
    TARGET,
    make_forecast_table,
)

TEST_YEAR = 2025
SPOT_NAMES = [spot.name for spot in SPOTS]
MODEL_FEATURES = [*FORECAST_FEATURES, "spot"]


def split_forecast_by_year(
    table: pd.DataFrame, test_year: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on years before test_year, test on test_year. Later years are left out.

    The year is taken from the time we want to predict (the target time), so a row from
    31 Dec 2024 that predicts 2 Jan 2025 belongs to the test year, not to training.
    """
    target_year = target_time(table).dt.year
    return table[target_year < test_year], table[target_year == test_year]


def target_time(table: pd.DataFrame) -> pd.Series:
    """The moment each row's answer belongs to."""
    return table["time"] + pd.Timedelta(hours=HORIZON_HOURS)


def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """MAE: average miss in knots. RMSE: punishes big misses more. Bias: + means too high."""
    error = np.asarray(predicted) - np.asarray(actual)
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
    }


def persistence_forecast(test: pd.DataFrame) -> pd.Series:
    """'In 48 hours it will be like now.' Same hour of day, because 48 h is 2 full days."""
    return test["wind_speed_10m"]


def climatology_forecast(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """The average wind for this spot, month and hour of day, learned from TRAINING data only.

    Using test data here would let the baseline peek at the answers (data leakage).
    """

    def keys(df: pd.DataFrame) -> pd.DataFrame:
        when = target_time(df)
        return pd.DataFrame(
            {"spot": df["spot"], "month": when.dt.month, "hour": when.dt.hour}
        )

    averages = (
        keys(train)
        .assign(wind=train[TARGET])
        .groupby(["spot", "month", "hour"])["wind"]
        .mean()
    )
    test_keys = pd.MultiIndex.from_frame(keys(test))
    return pd.Series(averages.reindex(test_keys).to_numpy(), index=test.index)


def model_inputs(table: pd.DataFrame) -> pd.DataFrame:
    """Feature columns for the model. 'spot' becomes a category XGBoost can split on."""
    X = table[MODEL_FEATURES].copy()
    X["spot"] = pd.Categorical(X["spot"], categories=SPOT_NAMES)
    return X


def train_xgboost(train: pd.DataFrame, n_jobs: int | None = None) -> XGBRegressor:
    """Gradient boosting: hundreds of small trees, each one fixing the errors of the ones before.

    n_jobs=None uses all CPU cores. Pass n_jobs=1 when PyTorch is running in the same
    program: on macOS their two OpenMP libraries crash together (segmentation fault).
    """
    model = XGBRegressor(
        n_estimators=400,  # number of trees
        learning_rate=0.05,  # how much each new tree is allowed to correct
        max_depth=6,  # how many questions each tree may ask
        subsample=0.8,  # each tree sees a random 80% of rows...
        colsample_bytree=0.8,  # ...and 80% of features, so trees don't all learn the same thing
        tree_method="hist",
        enable_categorical=True,
        random_state=42,
        n_jobs=n_jobs,
    )
    return model.fit(model_inputs(train), train[TARGET])


def compare(test: pd.DataFrame, forecasts: dict[str, pd.Series]) -> pd.DataFrame:
    """One row per method with MAE, RMSE and bias, best first."""
    rows = {
        name: regression_metrics(test[TARGET], pred) for name, pred in forecasts.items()
    }
    return pd.DataFrame(rows).T.sort_values("MAE").round(2)


def main() -> None:
    pd.set_option("display.width", None)

    table = make_forecast_table(load_history(HISTORY_START, HISTORY_END))
    train, test = split_forecast_by_year(table, TEST_YEAR)
    print(f"Train rows: {len(train)}, test rows ({TEST_YEAR}): {len(test)}")

    model = train_xgboost(train)
    forecasts = {
        "persistence": persistence_forecast(test),
        "climatology": climatology_forecast(train, test),
        "xgboost": pd.Series(model.predict(model_inputs(test)), index=test.index),
    }

    print(f"\n=== Wind in {HORIZON_HOURS} h, test year {TEST_YEAR} (knots) ===")
    print(compare(test, forecasts))

    # Overfitting check: is the model much better on data it has already seen?
    train_mae = regression_metrics(train[TARGET], model.predict(model_inputs(train)))[
        "MAE"
    ]
    print(f"\nXGBoost MAE on training data: {train_mae:.2f} (test: see table above)")

    print("\n=== MAE per spot ===")
    per_spot = {
        spot: {
            name: regression_metrics(group[TARGET], pred.loc[group.index])["MAE"]
            for name, pred in forecasts.items()
        }
        for spot, group in test.groupby("spot")
    }
    print(pd.DataFrame(per_spot).T.round(2))

    print("\n=== Most useful features (XGBoost) ===")
    importance = pd.Series(model.feature_importances_, index=MODEL_FEATURES)
    print(importance.sort_values(ascending=False).head(10).round(3))
