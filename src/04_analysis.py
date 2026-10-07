"""
Step 4 — Estimation.

Regenerates every table in the paper: descriptives, first stage, reduced
form, main IV results, equivalence bounds, transport tests, robustness,
placebos, and randomisation inference.

Estimator
---------
Two-stage least squares, log PM2.5 instrumented with boundary layer height,
orthogonal wind components and surface pressure. HAC (kernel) standard
errors throughout.

Equivalence
-----------
The estimand of interest is the exclusion bound, not the point estimate or
its p-value, so robustness rows report (|b| + 1.645 * SE) * ln2 in
percentage points per doubling of PM2.5.

Transport tests
---------------
Both the benchmark estimate and ours carry uncertainty, so equality of the
two parameters is tested directly rather than asking whether one falls
inside the other's interval. Dividing by our own standard error alone would
overstate the rejection.

Reads   data/panel_analysis.csv
Writes  output/results.txt  (and prints the same to stdout)
"""

import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS

warnings.filterwarnings("ignore")

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")

LN2 = np.log(2)

# Heyes, Neidell and Saberian (2016), NBER WP 22753, Table 2 col 1:
# -0.0168 per ug/m3 (SE 0.0055), sample mean 11.53, return SD 1.28.
# Converted to percentage points per doubling at each study's own mean.
BENCHMARK = {
    "proportional": (0.194, 0.063),   # constant per doubling
    "linear": (0.500, 0.164),         # constant per ug/m3, at our mean
}

D = pd.read_csv(os.path.join(DATA, "panel_analysis.csv"))
D["date"] = pd.to_datetime(D["date"])
D["const"] = 1

INST = ["log_blh", "u", "v", "pressure"]
FE = [c for c in D.columns if c.startswith(("mo_", "dw_", "yr_"))]
CTRL = ["temp", "precip", "wind_speed", "cloud", "radiation"] + FE + [
    c for c in ("lockdown", "qantar") if c in D.columns and D[c].nunique() > 1
]

_lines = []


def say(s=""):
    print(s)
    _lines.append(s)


def head(t):
    say()
    say("=" * 78)
    say(t)
    say("=" * 78)


def full_rank(d, cols):
    """Drop collinear controls, which subsamples can introduce."""
    keep = []
    for c in cols:
        if c not in d.columns or d[c].nunique() <= 1:
            continue
        M = d[["const"] + keep + [c]].astype(float).values
        if np.linalg.matrix_rank(M) == M.shape[1]:
            keep.append(c)
    return keep


def iv(data=None, y="logret", x="log_pm25", inst=None, ctrl=None):
    d = (data if data is not None else D).copy()
    d["const"] = 1
    inst = inst or INST
    d = d.dropna(subset=[y, x] + inst)
    keep = full_rank(d, ctrl or CTRL)
    r = IV2SLS(d[y], d[["const"] + keep], d[x], d[inst]).fit(cov_type="kernel")
    b, se = r.params[x], r.std_errors[x]
    try:
        F = r.first_stage.diagnostics["f.stat"].iloc[0]
    except Exception:
        F = np.nan
    return dict(
        b=b, se=se, p=r.pvalues[x], n=len(d), F=F,
        bound=(abs(b) + 1.645 * se) * LN2, r=r, ctrl=keep,
    )


def main():
    os.makedirs(OUT, exist_ok=True)
    main_res = iv()
    b, se = main_res["b"], main_res["se"]
    CT = main_res["ctrl"]

    # ---------------------------------------------------- TABLE 1
    head("TABLE 1 - DESCRIPTIVE STATISTICS")
    lbl = {
        "logret": "Log return (%)",
        "pm25_raw": "PM2.5 (ug/m3)",
        "blh": "Boundary layer height (m)",
        "temp": "Temperature (C)",
        "precip": "Precipitation (mm)",
        "wind_speed": "Wind speed (m/s)",
        "pressure": "Surface pressure (hPa)",
        "cloud": "Cloud cover (%)",
        "radiation": "Shortwave radiation (W/m2)",
    }
    cols = [c for c in lbl if c in D.columns]
    t1 = D[cols].describe().T[["count", "mean", "std", "min", "50%", "max"]]
    t1.index = [lbl[i] for i in t1.index]
    say(t1.round(2).to_string())
    say()
    say(f"N = {len(D)}   {D.date.min().date()} to {D.date.max().date()}")
    say(f"Return AR(1)     : {D.logret.autocorr(1):.3f}")
    say(f"SD of log PM2.5  : {np.log(D.pm25_raw).std():.3f}")
    say(f"Days >55 ug/m3   : {(D.pm25_raw > 55).sum()}   "
        f">75: {(D.pm25_raw > 75).sum()}")

    # ---------------------------------------------------- TABLE 2
    head("TABLE 2 - FIRST STAGE (dep. var: log PM2.5)")
    fs = smf.ols(
        "log_pm25 ~ " + " + ".join(INST + CT), data=D
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    for v in INST:
        say(f"  {v:<12}{fs.params[v]:9.4f}  ({fs.bse[v]:.4f})   "
            f"t={fs.tvalues[v]:6.2f}   p={fs.pvalues[v]:.3f}")
    say()
    say(f"  R-squared {fs.rsquared:.3f}    First-stage F {main_res['F']:.1f}")
    try:
        pr2 = main_res["r"].first_stage.diagnostics["partial.rsquared"].iloc[0]
        say(f"  Partial R-squared {pr2:.4f}")
    except Exception:
        pass

    # ---------------------------------------------------- TABLE 3
    head("TABLE 3 - REDUCED FORM (dep. var: daily log return)")
    rf = smf.ols(
        "logret ~ " + " + ".join(INST + CT), data=D
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    for v in INST:
        say(f"  {v:<12}{rf.params[v]:9.4f}  ({rf.bse[v]:.4f})   "
            f"p={rf.pvalues[v]:.3f}")
    base = smf.ols("logret ~ " + " + ".join(CT), data=D).fit()
    full = smf.ols("logret ~ " + " + ".join(INST + CT), data=D).fit()
    partial_rf = (full.rsquared - base.rsquared) / (1 - base.rsquared)
    joint_p = rf.f_test(", ".join(f"{i} = 0" for i in INST)).pvalue
    say()
    say(f"  Joint F-test of exclusion, p = {float(joint_p):.4f}")
    say(f"  Partial R-squared            {partial_rf:.4f}")
    say(f"  TOST bound on BLH coef       "
        f"{(abs(rf.params['log_blh']) + 1.645 * rf.bse['log_blh']) * LN2:.4f} pp")

    # ---------------------------------------------------- TABLE 4
    head("TABLE 4 - MAIN RESULTS")
    ols = smf.ols(
        "logret ~ log_pm25 + " + " + ".join(CT), data=D
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    blh_only = iv(inst=["log_blh"])
    say(f"  (1) OLS          {ols.params['log_pm25']:8.4f} "
        f"({ols.bse['log_pm25']:.4f})  p={ols.pvalues['log_pm25']:.3f}")
    say(f"  (2) IV, BLH only {blh_only['b']:8.4f} ({blh_only['se']:.4f})  "
        f"p={blh_only['p']:.3f}  F={blh_only['F']:.1f}")
    say(f"  (3) IV, full set {b:8.4f} ({se:.4f})  "
        f"p={main_res['p']:.3f}  F={main_res['F']:.1f}")
    say()
    say(f"  95% CI      [{b - 1.96 * se:.4f}, {b + 1.96 * se:.4f}]")
    say(f"  Sargan      {main_res['r'].sargan.stat:.4f}  "
        f"p={main_res['r'].sargan.pval:.4f}")
    try:
        wh = main_res["r"].wu_hausman()
        say(f"  Wu-Hausman  {wh.stat:.4f}  p={wh.pval:.4f}")
    except Exception:
        pass

    # ---------------------------------------------------- EQUIVALENCE
    head("EQUIVALENCE TESTS (TOST)")
    say(f"  Point estimate {b:.4f}  (SE {se:.4f})")
    say(f"  90% CI [{b - 1.645 * se:.4f}, {b + 1.645 * se:.4f}]")
    say()
    say(f"  Reject |effect| above {abs(b) + 1.645 * se:.4f} log points")
    say(f"    = {main_res['bound']:.3f} pp per doubling of PM2.5")
    say(f"    = {main_res['bound'] / D.logret.std():.3f} return SD per doubling")
    say(f"  Per doubling: {b * LN2:.4f} (SE {se * LN2:.4f})")

    say()
    say("  Transport tests vs Heyes et al. (2016), both estimates uncertain:")
    est_pp, se_pp = b * LN2, se * LN2
    for label, (d_, sd_) in BENCHMARK.items():
        t = (d_ + est_pp) / np.sqrt(se_pp ** 2 + sd_ ** 2)
        verdict = "REJECTED" if abs(t) > 1.96 else "not rejected"
        say(f"    {label:<14} benchmark -{d_:.3f} pp (SE {sd_:.3f})   "
            f"t = {t:5.2f}   {verdict}")

    # ---------------------------------------------------- TABLE 5
    head("TABLE 5 - ROBUSTNESS")
    say(f"  {'Specification':<34}{'coef':>9}{'se':>9}{'p':>8}"
        f"{'N':>6}{'F':>8}{'bound':>8}")
    say("  " + "-" * 74)

    def row(lab, res):
        say(f"  {lab:<34}{res['b']:9.4f}{res['se']:9.4f}{res['p']:8.3f}"
            f"{res['n']:6d}{res['F']:8.1f}{res['bound']:8.3f}")

    row("Preferred", main_res)
    if "n_obs" in D.columns:
        for hrs in (11, 10, 9):
            sub = D[D.n_obs >= hrs]
            if len(sub) < 100 or len(sub) == len(D):
                continue
            row(f"Coverage >={hrs}h", iv(sub))
    for start, lab in [("2022-01-01", "Sample from 2022"),
                       ("2023-01-01", "Sample from 2023")]:
        row(lab, iv(D[D.date >= start]))
    row("Excluding 2025", iv(D[D.date <= "2024-12-31"]))
    row("Winter only (Nov-Mar)", iv(D[D.month.isin([11, 12, 1, 2, 3])]))

    dw = D.copy()
    lo, hi = dw.logret.quantile([0.01, 0.99])
    dw["logret"] = dw.logret.clip(lo, hi)
    row("Winsorised at 1%", iv(dw))

    for drop in ("pressure", "v", "u"):
        row(f"Instruments excl. {drop}",
            iv(inst=[i for i in INST if i != drop]))
    row("BLH + pressure only", iv(inst=["log_blh", "pressure"]))
    row("BLH alone", blh_only)
    row("Controls excl. cloud, radiation",
        iv(ctrl=[c for c in CTRL if c not in ("cloud", "radiation")]))
    if "qantar" in D.columns:
        row("Excluding Jan 2022 unrest", iv(D[D.qantar == 0]))

    say()
    for L in (1, 2, 3):
        d = D.copy()
        d[f"pm_l{L}"] = d.log_pm25.shift(L)
        for i in INST:
            d[f"{i}_l{L}"] = d[i].shift(L)
        row(f"Lag {L} day",
            iv(d.dropna(subset=[f"pm_l{L}"]), x=f"pm_l{L}",
               inst=[f"{i}_l{L}" for i in INST]))
    da = D.copy()
    da["absret"] = da.logret.abs()
    row("Outcome: absolute return", iv(da, y="absret"))

    say()
    say("  Threshold indicators (reduced-form OLS):")
    for thr in (35, 55, 75):
        d = D.copy()
        d["hi"] = (d.pm25_raw > thr).astype(int)
        if d.hi.sum() < 20:
            continue
        rr = smf.ols(
            "logret ~ hi + " + " + ".join(CT), data=d
        ).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        say(f"    >{thr:3d} ug/m3 ({int(d.hi.sum()):3d} days): "
            f"{rr.params['hi']:8.4f} ({rr.bse['hi']:.4f})  "
            f"p={rr.pvalues['hi']:.3f}")

    # ---------------------------------------------------- PLACEBO / RI
    head("TABLE 6 - PLACEBOS AND RANDOMISATION INFERENCE")
    say("  Pollution on day t cannot affect returns realised earlier.")
    say()
    for L in (1, 2, 3, 5):
        d = D.copy()
        d["lead"] = d.logret.shift(L)
        r = iv(d.dropna(subset=["lead"]), y="lead")
        say(f"  {L}-day lead   {r['b']:8.4f} ({r['se']:.4f})  p={r['p']:.3f}")

    X, Z, n = D[["const"] + CT], D[INST], len(D)

    def beta(y):
        return IV2SLS(
            pd.Series(y, index=D.index), X, D["log_pm25"], Z
        ).fit().params["log_pm25"]

    # Circular shifts in multiples of 5 preserve weekday alignment.
    rng = np.random.default_rng(42)
    null = []
    for _ in range(1000):
        k = int(rng.integers(6, (n - 30) // 5)) * 5
        try:
            null.append(beta(np.roll(D.logret.values, k)))
        except Exception:
            pass
    null = np.array(null)
    say()
    say(f"  Randomisation SD {null.std():.4f}   HAC SE {se:.4f}   "
        f"ratio {null.std() / se:.2f}")
    say(f"  RI p-value against zero  {(np.abs(null) >= abs(b)).mean():.3f}")
    say(f"  Null 2.5-97.5%  [{np.percentile(null, 2.5):.3f}, "
        f"{np.percentile(null, 97.5):.3f}]")
    say(f"  Estimate sits at percentile {(null < b).mean() * 100:.1f}")

    with open(os.path.join(OUT, "results.txt"), "w") as f:
        f.write("\n".join(_lines) + "\n")
    say()
    say(f"Written to {os.path.relpath(os.path.join(OUT, 'results.txt'), ROOT)}")


if __name__ == "__main__":
    main()
