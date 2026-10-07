"""
Step 5 — Figures.

Fig 1  Daily PM2.5 time series, with data gaps shown as breaks
Fig 2  Seasonal pollution and mixing-depth cycle
Fig 3  First stage, residualised on controls and the other instruments
Fig 4  Binned scatter of returns on residualised pollution (OLS, not IV)
Fig 5  Specification chart

Reads   data/panel_analysis.csv
Writes  output/fig1..fig5 as PDF and PNG
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

warnings.filterwarnings("ignore")

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")

MONTHS = "JFMAMJJASOND"

D = pd.read_csv(os.path.join(DATA, "panel_analysis.csv"))
D["date"] = pd.to_datetime(D["date"])
D["const"] = 1

INST = ["log_blh", "u", "v", "pressure"]
FE = [c for c in D.columns if c.startswith(("mo_", "dw_", "yr_"))]
CTRL = ["temp", "precip", "wind_speed", "cloud", "radiation"] + FE + [
    c for c in ("lockdown", "qantar") if c in D.columns and D[c].nunique() > 1
]
CT = [c for c in CTRL if c in D.columns and D[c].nunique() > 1]

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(
            os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", dpi=300
        )
    plt.close(fig)


def resid(y, Xm):
    return sm.OLS(y, sm.add_constant(Xm)).fit().resid


def iv_fit(data=None, inst=None):
    d = (data if data is not None else D).copy()
    d["const"] = 1
    keep = []
    for c in CT:
        if d[c].nunique() <= 1:
            continue
        M = d[["const"] + keep + [c]].astype(float).values
        if np.linalg.matrix_rank(M) == M.shape[1]:
            keep.append(c)
    r = IV2SLS(
        d["logret"], d[["const"] + keep], d["log_pm25"], d[inst or INST]
    ).fit(cov_type="kernel")
    return r.params["log_pm25"], r.std_errors["log_pm25"]


MAX_BRIDGE_DAYS = 7


def break_long_gaps(dates, values, max_bridge=MAX_BRIDGE_DAYS):
    """Reindex to a continuous daily calendar, bridging short absences but
    leaving long ones as NaN so matplotlib breaks the line there.

    Without this, a multi-month outage is drawn as a straight segment that a
    reader takes for real measurements. Weekends and holidays are bridged;
    the 2022 monitor gap and the 2024 BLH gap are not.
    """
    full = pd.DataFrame({"date": pd.date_range(dates.min(), dates.max(), freq="D")})
    s = full.merge(
        pd.DataFrame({"date": dates.values, "value": values.values}),
        on="date", how="left",
    )
    missing = s.value.isna()
    run_id = (missing != missing.shift()).cumsum()
    run_len = missing.groupby(run_id).transform("size")
    bridgeable = missing & (run_len <= max_bridge)

    s["plot"] = s.value.interpolate(limit_area="inside")
    s.loc[missing & ~bridgeable, "plot"] = np.nan
    return s


def fig1():
    s = break_long_gaps(D.date, D.pm25_raw)

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(s.date, s["plot"], lw=0.5, c="#444")
    ax.axhline(5, ls="--", c="green", lw=1, label="WHO guideline (5)")
    ax.axhline(
        D.pm25_raw.mean(), ls="--", c="red", lw=1,
        label=f"Sample mean ({D.pm25_raw.mean():.1f})",
    )
    ax.set(
        xlabel="Date",
        ylabel="PM2.5 ($\\mu$g/m$^3$)",
        title="Figure 1. Daily PM2.5, Almaty US Embassy monitor",
    )
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "fig1_timeseries")


def fig2():
    mp = D.groupby("month").agg(pm=("pm25_raw", "mean"), blh=("blh", "mean"))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].bar(mp.index, mp.pm, color="#8B4513")
    axes[0].set(
        xlabel="Month", ylabel="Mean PM2.5 ($\\mu$g/m$^3$)",
        title="Pollution by month",
    )
    axes[1].bar(mp.index, mp.blh, color="#4682B4")
    axes[1].set(
        xlabel="Month", ylabel="Mean BLH (m)", title="Mixing depth by month"
    )
    for ax in axes:
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(list(MONTHS))
        ax.set_xlim(0.3, 12.7)
    fig.suptitle("Figure 2. Seasonal inversion pattern, Almaty basin", y=1.03)
    fig.tight_layout()
    save(fig, "fig2_seasonality")


def fig3():
    """Residualise on controls AND the other instruments, so that by
    Frisch-Waugh-Lovell the bivariate slope equals the reported first-stage
    coefficient on log BLH exactly."""
    others = [i for i in INST if i != "log_blh"]
    C = D[CT + others]
    rx, ry = resid(D.log_blh, C), resid(D.log_pm25, C)
    slope = np.polyfit(rx, ry, 1)

    fs = sm.OLS(
        D.log_pm25, sm.add_constant(D[INST + CT])
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    F = fs.f_test(", ".join(f"{i} = 0" for i in INST)).fvalue

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.scatter(rx, ry, s=4, alpha=0.25, c="#333")
    xs = np.linspace(rx.min(), rx.max(), 50)
    ax.plot(xs, np.polyval(slope, xs), c="crimson", lw=2)
    ax.set(
        xlabel="Log BLH (residualised)",
        ylabel="Log PM2.5 (residualised)",
        title="Figure 3. First stage",
    )
    ax.annotate(
        f"slope {slope[0]:.3f}\nfirst-stage F {float(F):.1f}",
        xy=(0.04, 0.06), xycoords="axes fraction", fontsize=8,
    )
    fig.tight_layout()
    save(fig, "fig3_firststage")


def fig4():
    C = D[CT]
    rx, ry = resid(D.log_pm25, C), resid(D.logret, C)
    bins = pd.qcut(rx, 20, labels=False)
    bx = pd.Series(rx).groupby(bins).mean()
    by = pd.Series(ry).groupby(bins).mean()
    bse = pd.Series(ry).groupby(bins).sem()
    slope = np.polyfit(rx, ry, 1)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.errorbar(
        bx, by, yerr=1.96 * bse, fmt="o", ms=5, c="#1f3b73",
        ecolor="#999", elinewidth=1, capsize=2,
    )
    ax.plot(bx, np.polyval(slope, bx), c="crimson", lw=1.5,
            label=f"OLS slope {slope[0]:.3f}")
    ax.axhline(0, c="k", lw=0.6, ls=":")
    ax.set(
        xlabel="Log PM2.5 (residualised)",
        ylabel="Log return, % (residualised)",
        title="Figure 4. Pollution and returns, OLS",
    )
    ax.legend(frameon=False, fontsize=8)
    fig.text(
        0.5, -0.06,
        "Ordinary least squares relationship between residualised pollution "
        "and returns.\nThis is not the IV estimate reported in the abstract.",
        ha="center", fontsize=7.5, style="italic",
    )
    fig.tight_layout()
    save(fig, "fig4_binscatter")


def fig5():
    specs = [
        ("Preferred", iv_fit()),
        ("From 2022", iv_fit(D[D.date >= "2022-01-01"])),
        ("Winter only (Nov-Mar)", iv_fit(D[D.month.isin([11, 12, 1, 2, 3])])),
        ("Excluding 2025", iv_fit(D[D.date <= "2024-12-31"])),
        ("Excl. surface pressure", iv_fit(inst=["log_blh", "u", "v"])),
        ("Excl. meridional wind (v)",
         iv_fit(inst=["log_blh", "u", "pressure"])),
        ("Excl. zonal wind (u)", iv_fit(inst=["log_blh", "v", "pressure"])),
        ("BLH alone", iv_fit(inst=["log_blh"])),
    ]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ys = np.arange(len(specs))[::-1]
    for i, (lab, (bb, ss)) in enumerate(specs):
        c = "crimson" if lab == "Preferred" else "#333"
        ax.errorbar(
            bb, ys[i], xerr=1.96 * ss, fmt="o", ms=5,
            color=c, ecolor=c, elinewidth=1.2, capsize=2,
        )
    ax.axvline(0, c="k", lw=0.8, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels([s[0] for s in specs], fontsize=8)
    ax.set(
        xlabel="Coefficient on log PM2.5 (95% CI)",
        title="Figure 5. Estimate stability across specifications",
    )
    fig.tight_layout()
    save(fig, "fig5_specification")


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in (fig1, fig2, fig3, fig4, fig5):
        f()
        print(f"  {f.__name__} done")
    print(f"\nFigures written to {os.path.relpath(OUT, ROOT)}/")


if __name__ == "__main__":
    main()
