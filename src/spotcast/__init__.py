"""SpotCast command-line entry point (`uv run spotcast`)."""

import pandas as pd
import requests
from sqlalchemy.exc import SQLAlchemyError

from spotcast.config import HISTORY_END, HISTORY_START, HISTORY_TABLE, SPOTS
from spotcast.data import fetch_all_spots, load_history
from spotcast.db import get_engine, save_to_db
from spotcast.labels import add_labels
from spotcast.model import (
    FEATURES,
    evaluate,
    load_model,
    predict_conditions,
    predict_with_guardrail,
    save_model,
    show_tree,
    split_by_year,
    train_baseline,
    train_tree,
)
from spotcast.profiles import get_profile

TEST_YEAR = 2025
SPORT, LEVEL = "windsurf", "intermediate"


def main() -> None:
    pd.set_option("display.width", None)  # use the full terminal width when printing

    try:
        history = load_history(HISTORY_START, HISTORY_END)
    except requests.RequestException as error:
        print(f"Could not fetch data: {error}")
        return

    try:
        engine = get_engine()
        save_to_db(history, HISTORY_TABLE, engine)
        summary = pd.read_sql(
            f"""
            SELECT spot, COUNT(*) AS hours, ROUND(AVG(wind_speed_10m)::numeric, 2) AS avg_wind_kn
            FROM {HISTORY_TABLE}
            GROUP BY spot
            ORDER BY avg_wind_kn DESC
            """,
            engine,
        )
        print(summary)
    except (RuntimeError, SQLAlchemyError) as error:
        # The model only needs the Parquet data, so a database problem shouldn't stop it
        print(f"Database error (skipping save, continuing): {error}")
    print("*********")

    profile = get_profile(SPORT, LEVEL)
    labeled = add_labels(history, profile)
    print(f"Label counts per spot ({profile.sport}, {profile.level}):")
    print(pd.crosstab(labeled["spot"], labeled["label"], margins=True))
    print("*********")

    X_train, X_test, y_train, y_test = split_by_year(labeled, TEST_YEAR)
    print(f"Train rows: {len(X_train)}, test rows ({TEST_YEAR}): {len(X_test)}\n")

    evaluate(
        "Baseline (always most common)",
        train_baseline(X_train, y_train),
        X_test,
        y_test,
    )

    tree = train_tree(X_train, y_train)
    evaluate("Decision tree", tree, X_test, y_test)
    show_tree(tree)
    print("*********")

    # Conditions the model never saw in training (2024 max wind was 27.7 kn)
    storm = pd.DataFrame(
        [[40.0, 0, 0.5, 4.0, True], [18.0, 0, 0.5, 4.0, True]], columns=FEATURES
    )
    print("Extreme check: 40 kn and 18 kn, small waves, daytime")
    print(f"  model only:     {list(tree.predict(storm))}")
    print(f"  with guardrail: {predict_with_guardrail(tree, storm, profile)}")
    print("*********")

    # Evaluation is done, so the final model learns from all years we have
    all_years = sorted(labeled["time"].dt.year.unique().tolist())
    final_tree = train_tree(labeled[FEATURES], labeled["label"])
    path = save_model(final_tree, profile, train_years=all_years)
    print(f"Saved model trained on {all_years} to {path.name}")

    # Use the saved model the way a real app would: load it, then predict the forecast
    bundle = load_model(profile)
    try:
        forecast = fetch_all_spots(SPOTS)
    except requests.RequestException as error:
        print(f"Could not fetch forecast: {error}")
        return
    predicted = predict_conditions(bundle, forecast)
    show_forecast(predicted)


def show_forecast(predicted: pd.DataFrame) -> None:
    """Print a per-spot summary of the next 7 days and the hours worth going."""
    print("\n=== Next 7 days ===")
    print(pd.crosstab(predicted["spot"], predicted["prediction"]))

    worth_going = predicted[predicted["prediction"].isin(["good", "marginal"])]
    if worth_going.empty:
        print("\nNo good or marginal hours this week.")
        return

    # Sorting the text would put "marginal" before "good", so sort by a rank instead
    rank = worth_going["prediction"].map({"good": 0, "marginal": 1})
    ordered = worth_going.assign(rank=rank).sort_values(
        ["rank", "wind_speed_10m"], ascending=[True, False]
    )

    columns = ["spot", "time", "wind_speed_10m", "wave_height", "prediction"]
    print("\nHours worth going (good first, then windiest):")
    print(ordered[columns].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
