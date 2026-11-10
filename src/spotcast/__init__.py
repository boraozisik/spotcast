"""SpotCast command-line entry point (`uv run spotcast`)."""

import pandas as pd
import requests
from sqlalchemy.exc import SQLAlchemyError

from spotcast.config import HISTORY_END, HISTORY_START, HISTORY_TABLE
from spotcast.data import fetch_all_spots, load_history
from spotcast.db import get_engine, save_to_db
from spotcast.locations import SPOTS
from spotcast.model import (
    FEATURES,
    MONITORED_INPUTS,
    classification_scores,
    evaluate,
    feature_ranges,
    load_model,
    make_features,
    predict_conditions,
    predict_with_guardrail,
    save_model,
    show_tree,
    split_by_year,
    train_baseline,
    train_tree,
    training_table,
)
from spotcast.profiles import PROFILES, PROFILES_FILE, get_profile
from spotcast.recommend import Preferences, recommend

TEST_YEAR = 2025
SPORT, LEVEL = "windsurf", "intermediate"  # for the forecast demo at the end
REGISTERED_NAME = "suitability"  # one registered model for all profiles
EXPERIMENT = "suitability-classifier"  # MLflow groups comparable runs under one name


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

    # One model for every profile: the history once per profile, with relative features
    profiles = list(PROFILES.values())
    table = training_table(history, profiles)
    print(
        f"Label share per profile ({len(profiles)} profiles, {len(history)} hours each):"
    )
    print(
        pd.crosstab(table["profile"], table["label"], normalize="index")
        .round(3)
        .to_string()
    )
    print("*********")

    # Imported here, not at the top: every `import spotcast...` runs this file, and the
    # API and tests shouldn't have to load MLflow
    from spotcast.tracking import metric_key, mlflow, run

    X_train, X_test, y_train, y_test = split_by_year(table, TEST_YEAR)
    print(f"Train rows: {len(X_train)}, test rows ({TEST_YEAR}): {len(X_test)}\n")
    data = {
        "history": f"{HISTORY_START}..{HISTORY_END}",
        "test_year": TEST_YEAR,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "profiles": len(profiles),
    }

    with run(EXPERIMENT, "baseline", profiles="all"):
        mlflow.log_params({"model": "most_frequent", **data})
        baseline = train_baseline(X_train, y_train)
        mlflow.log_metrics(
            evaluate("Baseline (always most common)", baseline, X_test, y_test)
        )

    with run(EXPERIMENT, "decision-tree", profiles="all"):
        tree = train_tree(X_train, y_train)
        mlflow.log_params(
            {
                "model": "decision_tree",
                "max_depth": tree.max_depth,
                "features": ",".join(FEATURES),
                **data,
            }
        )
        mlflow.log_metrics(evaluate("Decision tree", tree, X_test, y_test))
        # One number per profile: a model can be fine overall and bad for one sport
        test_part = table[table["time"].dt.year == TEST_YEAR]
        per_profile = {
            name: classification_scores(rows["label"], tree.predict(rows[FEATURES]))[
                "macro_f1"
            ]
            for name, rows in test_part.groupby("profile")
        }
        print("Macro F1 per profile (test year):")
        for name, f1 in per_profile.items():
            print(f"  {name:<22}{f1:.3f}")
        mlflow.log_metrics(
            {metric_key(f"macro_f1_{n}"): v for n, v in per_profile.items()}
        )
        mlflow.log_text(show_tree(tree), "tree_rules.txt")
        mlflow.log_artifact(str(PROFILES_FILE))  # the limits this run was trained with
    print("*********")

    # Conditions the model never saw in training (the data's max wind is ~31 kn)
    profile = get_profile(SPORT, LEVEL)
    storm = pd.DataFrame(
        {"wind_speed_10m": [40.0, 18.0], "wave_height": [0.5, 0.5], "is_day": True}
    )
    print(f"Extreme check ({profile.name}): 40 kn and 18 kn, small waves, daytime")
    print(f"  model only:     {list(tree.predict(make_features(storm, profile)))}")
    print(f"  with guardrail: {predict_with_guardrail(tree, storm, profile)}")
    print("*********")

    # Evaluation is done, so the final model learns from all years we have
    all_years = sorted(table["time"].dt.year.unique().tolist())
    with run(EXPERIMENT, "final-model", profiles="all"):
        final_tree = train_tree(table[FEATURES], table["label"])
        path = save_model(
            final_tree,
            profiles,
            train_years=all_years,
            ranges=feature_ranges(history[MONITORED_INPUTS]),
            train_end=HISTORY_END,
        )
        mlflow.log_params(
            {
                "model": "decision_tree",
                "max_depth": final_tree.max_depth,
                "train_years": ",".join(map(str, all_years)),
                "train_rows": len(table),
                "profiles": len(profiles),
            }
        )
        # The model registry keeps every version of the model we serve
        version = mlflow.sklearn.log_model(
            final_tree,
            name="model",
            registered_model_name=REGISTERED_NAME,
            input_example=table[FEATURES].head(3),
            # MLflow saves with skops, which only loads types it is told to trust (unlike
            # pickle, which can run any code). This one is scikit-learn's own tree class.
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        ).registered_model_version
        mlflow.log_artifact(str(path), artifact_path="bundle")  # what the API loads
    print(
        f"Saved model trained on {all_years} to {path.name} "
        f"(registry: {REGISTERED_NAME} v{version})"
    )

    # Use the saved model the way a real app would: load it, then predict the forecast
    bundle = load_model()
    try:
        forecast = fetch_all_spots(SPOTS)
    except requests.RequestException as error:
        print(f"Could not fetch forecast: {error}")
        return
    predicted = predict_conditions(bundle, forecast, profile)
    show_forecast(predicted)

    # Forecast times are local (Europe/Istanbul), like this computer's clock
    next_hour = pd.Timestamp.now().ceil("h")
    for title, prefs in [
        ("any day, 09:00-18:00, at least 2 hours", Preferences(not_before=next_hour)),
        ("weekend only", Preferences.weekend(not_before=next_hour)),
    ]:
        print(f"\n=== Best windows: {title} ===")
        best = recommend(predicted, profile, prefs)
        print(
            "No window fits."
            if best.empty
            else best.round({"avg_wind_kn": 2, "max_wave_m": 2, "score": 2}).to_string(
                index=False
            )
        )


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
