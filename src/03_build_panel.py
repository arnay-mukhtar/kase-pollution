"""
Step 3 — Build the estimation panel.

Aggregates pollution and weather over the exposure window, cleans the KASE
index series, merges on calendar date, and writes the analysis panel.

Exposure window
---------------
Hours 07:00-17:59 local time, requiring at least 8 hourly observations.
KASE equity trading runs from an opening auction at 11:20 to a close at
17:30, so the window includes pre-session exposure on the grounds that mood
formation precedes trading. The benchmark study's own intra-day estimates
concentrate in morning blocks.

Returns
-------
Computed from closing prices as 100 * (ln P_t - ln P_{t-1}) rather than
taken from the source's rounded percentage-change column, which would
introduce classical measurement error in the dependent variable.

Inputs   data/almaty_pm25_hourly.csv
         data/almaty_weather_hourly.csv
         data/kase_index.csv          (see data/README.md)

Writes   data/panel_final.csv         merged panel, one row per trading day
         data/panel_analysis.csv      the same plus fixed-effect dummies
"""

import os

import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

HOUR_START, HOUR_END = 7, 17      # inclusive, local time
MIN_HOURS = 8
SAMPLE_START, SAMPLE_END = "2020-04-08", "2025-11-13"

# National quarantine periods and the January 2022 civil unrest. Both
# simultaneously suppressed traffic emissions and disrupted trading.
QUARANTINE = [("2020-03-16", "2020-05-11"), ("2020-07-05", "2020-08-17")]
QANTAR = ("2022-01-04", "2022-01-19")


def window_mean(df, value_cols, label_counts=True):
    """Daily mean over the exposure window, keeping days with >= MIN_HOURS."""
    w = df[(df.hour >= HOUR_START) & (df.hour <= HOUR_END)].copy()
    agg = {c: "mean" for c in value_cols}
    out = w.groupby("date").agg(agg)
    if label_counts:
        out["n_obs"] = w.groupby("date").size()
        out = out[out.n_obs >= MIN_HOURS]
    return out.reset_index()


def load_kase():
    """Load the KASE index series and compute log returns from prices."""
    path = os.path.join(DATA, "kase_index.csv")
    ks = pd.read_csv(path)
    ks.columns = [
        c.strip().lower().replace(".", "").replace(" ", "_") for c in ks.columns
    ]

    price_col = next(
        (c for c in ("close", "price", "value") if c in ks.columns), None
    )
    if price_col is None:
        raise ValueError(f"No price column found in {path}: {list(ks.columns)}")

    ks["date"] = pd.to_datetime(ks["date"], errors="coerce", dayfirst=False)
    if ks["date"].isna().mean() > 0.5:
        ks["date"] = pd.to_datetime(ks["date"], errors="coerce", dayfirst=True)

    ks["close"] = pd.to_numeric(
        ks[price_col].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    ks = (
        ks.dropna(subset=["date", "close"])
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )
    ks["logret"] = np.log(ks["close"]).diff() * 100
    keep = ["date", "close", "logret"]
    if "weekend_session" in ks.columns:
        keep.append("weekend_session")
    return ks[keep]


def main():
    # ---- pollution over the exposure window ----
    pm_h = pd.read_csv(os.path.join(DATA, "almaty_pm25_hourly.csv"))
    pm_h["date"] = pd.to_datetime(pm_h["date"])
    pm = window_mean(pm_h, ["pm25"]).rename(columns={"pm25": "pm25_raw"})
    print(f"Pollution: {len(pm)} days pass the {MIN_HOURS}h coverage filter")

    # ---- weather over the same window ----
    wx_h = pd.read_csv(os.path.join(DATA, "almaty_weather_hourly.csv"))
    wx_h["date"] = pd.to_datetime(wx_h["date"])
    wx = window_mean(
        wx_h,
        [
            "temperature_2m",
            "wind_speed_10m",
            "boundary_layer_height",
            "surface_pressure",
            "cloud_cover",
            "shortwave_radiation",
            "u",
            "v",
        ],
        label_counts=False,
    )
    # Precipitation is a flow, so it sums over the window rather than averaging.
    precip = (
        wx_h[(wx_h.hour >= HOUR_START) & (wx_h.hour <= HOUR_END)]
        .groupby("date")["precipitation"]
        .sum()
        .reset_index()
    )
    wx = wx.merge(precip, on="date").rename(
        columns={
            "temperature_2m": "temp",
            "precipitation": "precip",
            "wind_speed_10m": "wind_speed",
            "boundary_layer_height": "blh",
            "surface_pressure": "pressure",
            "cloud_cover": "cloud",
            "shortwave_radiation": "radiation",
        }
    )
    print(f"Weather:   {len(wx)} days")

    # ---- index ----
    ks = load_kase()
    print(f"KASE:      {len(ks)} trading days")

    # ---- merge ----
    D = ks.merge(pm, on="date").merge(wx, on="date")
    D = D[(D.date >= SAMPLE_START) & (D.date <= SAMPLE_END)].copy()

    D["log_pm25"] = np.log(D.pm25_raw)
    D["log_blh"] = np.log(D.blh)
    D["dow"] = D.date.dt.dayofweek
    D["month"] = D.date.dt.month
    D["year"] = D.date.dt.year

    D["lockdown"] = 0
    for a, b in QUARANTINE:
        D.loc[(D.date >= a) & (D.date <= b), "lockdown"] = 1
    D["qantar"] = ((D.date >= QANTAR[0]) & (D.date <= QANTAR[1])).astype(int)

    D = (
        D.dropna(subset=["logret", "log_pm25", "log_blh", "u", "v", "pressure"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    D.to_csv(os.path.join(DATA, "panel_final.csv"), index=False)

    # ---- fixed-effect dummies, first level omitted ----
    for prefix, col in [("mo", "month"), ("dw", "dow"), ("yr", "year")]:
        for level in sorted(D[col].unique())[1:]:
            D[f"{prefix}_{level}"] = (D[col] == level).astype(int)
    D["const"] = 1
    D.to_csv(os.path.join(DATA, "panel_analysis.csv"), index=False)

    print(
        f"\nPANEL: {len(D)} obs, "
        f"{D.date.min().date()} to {D.date.max().date()}"
    )
    print(f"  mean PM2.5  {D.pm25_raw.mean():.2f}")
    print(f"  mean BLH    {D.blh.mean():.2f}")
    print(f"  return SD   {D.logret.std():.3f}")


if __name__ == "__main__":
    main()
