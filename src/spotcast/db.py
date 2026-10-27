"""PostgreSQL connection and helpers for reading/writing DataFrames."""

import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine


def get_engine() -> Engine:
    """Database connection pool, configured by DATABASE_URL in the .env file."""
    load_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Copy .env.example to .env and fill it in."
        )
    return create_engine(url)


def save_to_db(df: pd.DataFrame, table: str, engine: Engine) -> None:
    """Write the DataFrame to a table, replacing it if it already exists."""
    df.to_sql(table, engine, if_exists="replace", index=False, chunksize=5_000)
    print(f"Saved {len(df)} rows to table {table!r}")
