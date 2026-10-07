"""
Step 1 — Fetch PM2.5 from OpenAQ.

Source: US Embassy Almaty reference-grade monitor (AirNow network).
        OpenAQ location 8876, sensor 25903, 43.2529N 76.9312E.
Window: 2020-04-01 to 2025-11-14 (the monitor ceased reporting after this).

Writes data/almaty_pm25_daily.csv  and  data/almaty_pm25_hourly.csv

Requires a free OpenAQ API key:
    export OPENAQ_API_KEY=...
"""

import os
import time
import sys

import pandas as pd
import requests

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

SENSOR = 25903
DATE_FROM = "2020-04-01"
DATE_TO = "2025-11-14"

API_KEY = os.environ.get("OPENAQ_API_KEY")
if not API_KEY:
    sys.exit("Set OPENAQ_API_KEY (free key from https://explore.openaq.org).")
HEADERS = {"X-API-Key": API_KEY}


def paged(endpoint, date_from, date_to):
    """Page through an OpenAQ sensor endpoint, retrying on rate limits."""
    out, page = [], 1
    while True:
        r = requests.get(
            f"https://api.openaq.org/v3/sensors/{SENSOR}/{endpoint}",
            headers=HEADERS,
            timeout=120,
            params={
                "datetime_from": date_from,
                "datetime_to": date_to,
                "limit": 1000,
                "page": page,
            },
        )
        if r.status_code in (408, 429):
            time.sleep(15)
            continue
        if r.status_code != 200:
            print(f"  HTTP {r.status_code}: {r.text[:120]}")
            break
        res = r.json().get("results", [])
        if not res:
            break
        out.extend(res)
        if len(res) < 1000:
            break
        page += 1
        time.sleep(0.3)
    return out


def main():
    os.makedirs(DATA, exist_ok=True)

    # ---- daily means (with observed-hour counts) ----
    print("Fetching daily PM2.5 ...")
    rows = []
    for rec in paged("days", DATE_FROM, DATE_TO):
        period = (rec.get("period") or {}).get("datetimeFrom") or {}
        rows.append(
            {
                "date": period.get("local"),
                "pm25": rec.get("value"),
                "n_obs": (rec.get("coverage") or {}).get("observedCount"),
            }
        )

    daily = pd.DataFrame(rows)
    daily["date"] = pd.to_datetime(daily["date"].str.slice(0, 10))
    daily["sensor_id"] = SENSOR
    daily = (
        daily.dropna(subset=["pm25"])
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily.to_csv(os.path.join(DATA, "almaty_pm25_daily.csv"), index=False)
    print(
        f"  {len(daily)} days, "
        f"{daily.date.min().date()} to {daily.date.max().date()}"
    )

    # ---- hourly, year by year (the full-range query times out) ----
    print("Fetching hourly PM2.5 ...")
    hourly_rows = []
    years = [
        ("2020-04-01", "2020-12-31"),
        ("2021-01-01", "2021-12-31"),
        ("2022-01-01", "2022-12-31"),
        ("2023-01-01", "2023-12-31"),
        ("2024-01-01", "2024-12-31"),
        ("2025-01-01", "2025-11-14"),
    ]
    for a, b in years:
        for rec in paged("hours", a, b):
            period = (rec.get("period") or {}).get("datetimeFrom") or {}
            hourly_rows.append(
                {"local": period.get("local"), "pm25": rec.get("value")}
            )
        print(f"  {a[:4]}: {len(hourly_rows):,} cumulative")

    hourly = (
        pd.DataFrame(hourly_rows)
        .dropna(subset=["local", "pm25"])
        .drop_duplicates("local")
    )
    # Local time is read off the timestamp string rather than converted from
    # UTC, which preserves Kazakhstan's move from UTC+6 to UTC+5 on 2024-03-01.
    hourly["hour"] = hourly["local"].str.slice(11, 13).astype(int)
    hourly["date"] = pd.to_datetime(hourly["local"].str.slice(0, 10))
    hourly = hourly.sort_values("local").reset_index(drop=True)
    hourly.to_csv(os.path.join(DATA, "almaty_pm25_hourly.csv"), index=False)
    print(f"  {len(hourly):,} hourly observations")


if __name__ == "__main__":
    main()
