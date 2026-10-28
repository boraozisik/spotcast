"""SpotCast command-line entry point (`uv run spotcast`)."""

import pandas as pd
import requests
from sqlalchemy.exc import SQLAlchemyError

from spotcast.config import HISTORY_END, HISTORY_START, HISTORY_TABLE
from spotcast.data import load_history
from spotcast.db import get_engine, save_to_db
from spotcast.labels import add_labels
from spotcast.model import (
    FEATURES,
    evaluate,
    predict_with_guardrail,
    show_tree,
    split_by_year,
    train_baseline,
    train_tree,
)

TEST_YEAR = 2025


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
    except (RuntimeError, SQLAlchemyError) as error:
        print(f"Database error: {error}")
        return

    print(summary)
    print("*********")

    labeled = add_labels(history)
    print("Label counts per spot (intermediate windsurf):")
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
    print(f"  with guardrail: {predict_with_guardrail(tree, storm)}")


if __name__ == "__main__":
    main()
