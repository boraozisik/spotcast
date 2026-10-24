import pandas as pd
import requests


def fetch_forecast(latitude: float, longitude: float) -> pd.DataFrame:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ["wind_speed_10m", "wind_direction_10m"],
        "wind_speed_unit": "kn",
        "timezone": "Europe/Istanbul",
    }
    response = requests.get(url, params=params, timeout=10)
    # 1) raise an error if the request failed
    response.raise_for_status()

    # 2) parse the JSON
    data = response.json()

    df = pd.DataFrame(data["hourly"])

    df["time"] = pd.to_datetime(df["time"])

    return df


def main() -> None:
    try:
        df = fetch_forecast(37.86, 27.26)
    except requests.RequestException as error:
        print(f"Could not fetch forecast: {error}")
        return

    print(df.head())
    print("*********")
    print(df.dtypes)


if __name__ == "__main__":
    main()
