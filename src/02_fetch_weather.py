"""
Step 2 — Fetch ERA5 meteorology from the Open-Meteo historical archive.

Extracted at the coordinates of the Embassy monitor. Hourly series are kept
so that step 3 can aggregate them over the same exposure window as the
pollution data.

Wind direction is a circular variable, so it is decomposed into orthogonal
components before averaging: the arithmetic mean of 350 and 10 degrees is
180, the opposite of the true resultant direction.

    u = -w sin(theta)    positive u = westerly flow
    v = -w cos(theta)    positive v = southerly flow

Writes data/almaty_weather_hourly.csv

Note: boundary layer height is missing from the archive for 2 Jan - 30 Jun
2024. That gap affects that field alone and is documented in the paper.
"""

import os

import numpy as np
import pandas as pd
import requests

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

LAT, LON = 43.2529, 76.9312
START, END = "2020-01-01", "2025-12-31"

FIELDS = [
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "boundary_layer_height",
    "surface_pressure",
    "cloud_cover",
    "shortwave_radiation",
]


def main():
    os.makedirs(DATA, exist_ok=True)

    print("Fetching ERA5 hourly ...")
    resp = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": LAT,
            "longitude": LON,
            "start_date": START,
            "end_date": END,
            "hourly": ",".join(FIELDS),
            "timezone": "Asia/Almaty",
        },
        timeout=300,
    )
    resp.raise_for_status()

    w = pd.DataFrame(resp.json()["hourly"])
    w["time"] = pd.to_datetime(w["time"])
    w["hour"] = w["time"].dt.hour
    w["date"] = w["time"].dt.normalize()

    rad = np.deg2rad(w["wind_direction_10m"])
    w["u"] = -w["wind_speed_10m"] * np.sin(rad)
    w["v"] = -w["wind_speed_10m"] * np.cos(rad)

    w.to_csv(os.path.join(DATA, "almaty_weather_hourly.csv"), index=False)
    print(f"  {len(w):,} hourly observations")

    blh_missing = w.loc[w.boundary_layer_height.isna(), "date"]
    if len(blh_missing):
        print(
            f"  boundary layer height missing on {blh_missing.nunique()} days "
            f"({blh_missing.min().date()} to {blh_missing.max().date()})"
        )


if __name__ == "__main__":
    main()
