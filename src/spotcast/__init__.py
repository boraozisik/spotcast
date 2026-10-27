"""SpotCast command-line entry point (`uv run spotcast`)."""

import pandas as pd
import requests
from sqlalchemy.exc import SQLAlchemyError

from spotcast.config import HISTORY_END, HISTORY_START, HISTORY_TABLE
from spotcast.data import load_history
from spotcast.db import get_engine, save_to_db
from spotcast.labels import add_labels


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


if __name__ == "__main__":
    main()
