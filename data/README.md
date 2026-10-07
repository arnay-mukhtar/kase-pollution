# Data

Scripts `01` and `02` fetch pollution and meteorology from public APIs and
write their output here. The index series has to be supplied manually.

## What you need to provide

**`kase_index.csv`** — daily closing values of the KASE Index.

Minimum columns:

| Column | Notes |
|---|---|
| `date` | any parseable format |
| `close` | closing index value; `price` or `value` also accepted |
| `weekend_session` | optional flag, see below |

The scripts compute log returns from `close` themselves rather than reading a
percentage-change column, so a price column is all that is required.

Source the series from the exchange directly (kase.kz) rather than from a data
aggregator. Aggregator exports round percentage changes to two decimals and
omit compensating weekend sessions — the exchange trades on a small number of
Saturdays to make up public holidays, and those sessions produce multi-day
returns that an aggregator export silently drops. Nine such sessions fall
inside this sample. Omitting one inflates the adjacent return: the sample
maximum of 5.02 per cent computes to roughly 4.05 on the official calendar.

If your source includes those sessions, flag them in `weekend_session` so they
can be identified in robustness checks.

## What the scripts generate

| File | Written by | Contents |
|---|---|---|
| `almaty_pm25_daily.csv` | `01` | daily means with observed-hour counts |
| `almaty_pm25_hourly.csv` | `01` | hourly observations, local timestamps |
| `almaty_weather_hourly.csv` | `02` | ERA5 hourly, with wind decomposed into `u`/`v` |
| `panel_final.csv` | `03` | merged panel, one row per trading day |
| `panel_analysis.csv` | `03` | the same plus fixed-effect dummies |

These are gitignored — the repository holds code, not data.

## Known gaps

- **PM2.5, May–August 2022.** The monitor reports nothing for four consecutive
  months. Removes 82 trading days.
- **Boundary layer height, 2 Jan – 30 Jun 2024.** Missing from the ERA5 archive
  for this field alone; temperature, precipitation, wind and pressure are
  complete throughout. Because BLH is the primary instrument, these dates drop.
  Removes a further 120 trading days. Recovery through alternative model
  specifications in the same archive was attempted without success.
- **Coverage filter.** The remaining 190 lost days fall below eight hourly
  observations inside the exposure window.

Of 1,369 trading days in the sample window, 977 are retained. All three losses
arise from external data availability rather than any selection. Retained and
excluded days do not differ in mean returns (t = 0.42, p = 0.678), though
retained days show modestly lower return variance (Levene p = 0.016).

The 2024 gap falls entirely in the high-pollution half of the year, so the loss
falls disproportionately on the winter inversion episodes the design draws its
identifying variation from.
