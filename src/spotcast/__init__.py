from dataclasses import dataclass

import pandas as pd
import requests

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
TIMEZONE = "Europe/Istanbul"
REQUEST_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class Spot:
    name: str
    latitude: float
    longitude: float


SPOTS = [
    Spot("Kuşadası Pamucak", 37.94, 27.27),
    Spot("Alaçatı", 38.255, 26.375),
    Spot("Akyaka", 37.04, 28.31),
]


def _fetch_hourly(
    url: str,
    latitude: float,
    longitude: float,
    variables: list[str],
    **extra_params: object,
) -> pd.DataFrame:
    """Fetch hourly data from an Open-Meteo API and return it as a DataFrame."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": variables,
        "timezone": TIMEZONE,
    } | extra_params

    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()

    df = pd.DataFrame(response.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def fetch_forecast(latitude: float, longitude: float) -> pd.DataFrame:
    """Hourly wind speed (knots) and direction (degrees)."""
    return _fetch_hourly(
        FORECAST_URL,
        latitude,
        longitude,
        ["wind_speed_10m", "wind_direction_10m"],
        wind_speed_unit="kn",
    )


def fetch_marine_forecast(latitude: float, longitude: float) -> pd.DataFrame:
    """Hourly wave height (m), direction (degrees) and period (s)."""
    return _fetch_hourly(
        MARINE_URL,
        latitude,
        longitude,
        ["wave_height", "wave_direction", "wave_period"],
    )


def fetch_conditions(latitude: float, longitude: float) -> pd.DataFrame:
    """Wind and wave forecast combined into one table, one row per hour."""
    wind_df = fetch_forecast(latitude, longitude)
    marine_df = fetch_marine_forecast(latitude, longitude)
    return pd.merge(wind_df, marine_df, on="time", how="inner", validate="one_to_one")


def fetch_all_spots(spots: list[Spot]) -> pd.DataFrame:
    """Conditions for every spot stacked into one table, with a "spot" column."""
    frames = []
    for spot in spots:
        df = fetch_conditions(spot.latitude, spot.longitude)
        df.insert(0, "spot", spot.name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    pd.set_option("display.width", None)  # use the full terminal width when printing

    try:
        conditions = fetch_all_spots(SPOTS)
    except requests.RequestException as error:
        print(f"Could not fetch forecast: {error}")
        return

    print(conditions.head())
    print("*********")
    print(f"Rows: {len(conditions)}")
    print("*********")
    print(conditions.groupby("spot")[["wind_speed_10m", "wave_height"]].max())


if __name__ == "__main__":
    main()
