# Replication data and code

**Air Pollution and Equity Returns in an Emerging Market: Evidence from Kazakhstan**

Arnay Mukhtar (Independent Researcher, Almaty, Kazakhstan)
Paper: https://ssrn.com/abstract=7310078

**Version 2.** Corrects the description of the HAC bandwidth used for the
instrumental variables standard errors, which version 1 stated incorrectly
as five lags. See "Inference" below. No data file changed between versions.

---

## What this deposit contains

| File | Description |
|---|---|
| `panel_cds.csv` | The estimation panel. 1,042 trading days, 10 April 2020 to 13 November 2025. Every table and figure in the paper is produced from this file. |
| `almaty_weather_daily_cds.csv` | Daily meteorological inputs, 2,192 calendar days. Boundary layer height from the Copernicus Climate Data Store; remaining variables from Open-Meteo. |
| `almaty_pm25_hourly.csv` | Hourly PM2.5 from the United States Embassy monitor, as retrieved from OpenAQ. |
| `almaty_pm25_daily_window.csv` | Daily PM2.5 aggregated over the 07:00–17:59 local exposure window, with hourly observation counts. |
| `build_and_estimate.ipynb` | Builds the panel and produces every result in the paper. |
| `README.md` | This file. |

**`panel_cds.csv` is the file to use.** Earlier versions of this work used a
smaller panel; they are superseded. The estimation sample is 1,042
observations and the preferred coefficient is −0.0826 (SE 0.0832). If a
replication does not reproduce those two numbers, it is not using this panel.

---

## Quick verification

Reproduces the paper's headline estimate. Tested end to end; runs as written.

```python
# pip install linearmodels
import numpy as np, pandas as pd
from linearmodels.iv import IV2SLS

D = pd.read_csv("panel_cds.csv")
INST = ["log_blh", "u", "v", "pressure"]

cand = [c for c in D.columns
        if c.startswith(("mo_", "dw_", "yr_")) and D[c].nunique() > 1]
cand += ["temp", "precip", "wind_speed", "cloud", "radiation",
         "lockdown", "qantar"]

# The full dummy set is collinear. Controls are selected by rank test
# rather than by dropping a fixed reference category, so the retained set
# depends on the subsample. Omitting this step raises a rank error.
ctrl = []
for c in cand:
    M = D[["const"] + ctrl + [c]].astype(float).values
    if np.linalg.matrix_rank(M) == M.shape[1]:
        ctrl.append(c)

r = IV2SLS(D["logret"], D[["const"] + ctrl], D["log_pm25"], D[INST]
           ).fit(cov_type="kernel")
print(round(r.params["log_pm25"], 4), round(r.std_errors["log_pm25"], 4))
# -0.0826 0.0832
```

The exclusion bound reported in the abstract follows as
`(abs(b) + 1.645*se) * np.log(2)` = 0.152 percentage points per doubling.

---

## Inference

Standard errors are Newey–West throughout. **For the instrumental variables
specifications the Bartlett bandwidth is selected automatically by
`linearmodels`, which chooses twelve lags on this panel** — not a fixed lag
length. Confirm with `r.cov_config`, which returns
`{'debiased': False, 'bandwidth': 12, 'kernel': 'bartlett', 'kappa': 1}`.
Ordinary least squares specifications in the paper use a fixed five-lag
Bartlett window. Section 4.7 of the paper reports sensitivity to the choice.

This matters for anything computed from the first-stage covariance matrix.
The Montiel Olea–Pflueger effective F reported in the paper (70.41) is
computed at bandwidth 12; at five lags the same formula returns 70.10.

---

## Panel variables

**Outcome and treatment**

- `logret` — daily log return on the KASE Index, in per cent
- `pm25_raw` — mean PM2.5 over 07:00–17:59 local, µg/m³ (the treatment)
- `log_pm25` — natural log of `pm25_raw`
- `n_daily` — hourly PM2.5 observations falling in the window (minimum 8)
- `pm_session`, `pm_presession`, `pm25_calendar` — alternative treatment
  windows used in the robustness table

**Instruments**

- `blh` — planetary boundary layer height, calendar-day mean, metres
- `log_blh` — natural log of `blh`
- `u`, `v` — 10 m wind components, eastward and northward, m/s
- `pressure` — surface pressure, hPa

**Controls**

- `temp`, `temp_min` — 2 m temperature, mean and minimum, °C
- `precip` — total precipitation, mm
- `wind_speed` — 10 m wind speed, m/s
- `cloud` — total cloud cover, per cent
- `radiation` — downwelling shortwave radiation, W/m²
- `lockdown` — COVID-19 restriction periods. Active on 36 trading days in
  two blocks, 10 April to 11 May and 8 July to 14 August 2020
- `qantar` — January 2022 unrest, 11 to 19 January 2022 (6 trading days)
- `mo_1`–`mo_12`, `dw_0`–`dw_4`, `yr_2020`–`yr_2025` — month, day-of-week
  and year fixed effects
- `const` — constant

---

## Two construction choices worth knowing

**Meteorological variables are calendar-day means, not window means.** The
treatment is a window mean because exposure accumulates only while the
trader is awake; the instruments are calendar-day means because dispersion
capacity is set largely overnight. Section 3.2 of the paper gives the
argument and the first-stage evidence. This is a deliberate asymmetry, not
an oversight.

**Boundary layer height comes from a different source than the other
meteorological variables.** The Open-Meteo archive has no boundary layer
height for 2 January to 30 June 2024. The Copernicus Climate Data Store,
which is the underlying source, is complete. The two series correlate at
0.973 in logs but differ systematically in the upper tail, so the paper uses
the Climate Data Store throughout for that variable rather than splicing two
series within one column.

---

## Data sources, and what is not deposited here

| Source | Licence | Included? |
|---|---|---|
| OpenAQ / US Embassy monitor (PM2.5) | Open data | Yes |
| Copernicus Climate Data Store, ERA5 (boundary layer height) | Copernicus licence, redistribution permitted with attribution | Derived daily series only |
| Open-Meteo (other meteorology) | Open data, ERA5-derived | Derived daily series only |
| KASE Index levels | Kazakhstan Stock Exchange | Derived log returns only |
| Brent futures, Global X Uranium ETF (URA), index cross-validation series | Investing.com | **Not included** — see below |

The CC BY 4.0 licence on this deposit applies to the code and to the derived
panel constructed by the author. Upstream inputs retain their original
terms, as set out in the table above.

**Raw ERA5 files** are not deposited because they are freely and
authoritatively available from the Climate Data Store. To reproduce them,
request `reanalysis-era5-single-levels`, variable `boundary_layer_height`,
2020–2025, all months, days and hours, sub-region 43.75 / 76.50 / 42.75 /
77.50 (N/W/S/E), NetCDF. Submit one year per request; the cost limit rejects
larger spans. Extract at the nearest grid point, 43.25°N 77.00°E.

**Series obtained from Investing.com** are not deposited because that
source's terms do not permit redistribution. They are used only in the
Section 4.5 channel tests and the Section 3.3 cross-validation, and are not
required to reproduce any table or figure. To obtain them, download the
daily historical series for Brent Crude futures, the Global X Uranium ETF
and the KASE Index for 2020–2025.

**Local time.** Almaty observed UTC+6 until 1 March 2024 and UTC+5
thereafter. The build code applies the break explicitly. That local day has
25 hours and is handled as such; a naive 24-hour completeness filter
silently drops it.

---

## Citation

Mukhtar, A. (2026). *Air Pollution and Equity Returns in an Emerging Market:
Evidence from Kazakhstan*. Available at SSRN: https://ssrn.com/abstract=7310078

Please cite the paper. This deposit may be cited by its own DOI where the
data or code specifically are being referenced.
