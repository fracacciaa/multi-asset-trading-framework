"""
ADF stationarity test on NS classic residuals for each country/tenor pair.

For each market: load OIS → bootstrap zeros → fit NS → run ADF on each tenor residual.
Lambda selected on training data (≤ 2015-12-31), consistent with ns_classic_run.py.

Saves results to rates/signals/ns_adf_results.parquet and prints a summary table.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from rates.production.ois_data import fetch_ois_curves
from rates.production.ns_classic.bootstrap import bootstrap_curves
from rates.production.ns_classic.ns_fit import select_lambda, fit_ns_panel

MARKETS       = ["US", "EUR", "UK", "Japan", "Canada", "Australia",
                 "New Zealand", "Sweden", "Switzerland", "India", "China"]
LAM_TRAIN_END = "2015-12-31"
ADF_MAXLAG    = 10   # max lags passed to adfuller (AIC selection within)
OUT_PATH      = "rates/signals/ns_adf_results.parquet"


def run_adf(series: pd.Series) -> dict:
    """Run ADF on a single residual series. Returns stat, p-value, lags, nobs."""
    s = series.dropna()
    if len(s) < 30:
        return dict(adf_stat=np.nan, p_value=np.nan, lags_used=np.nan,
                    n_obs=len(s), reject_10=False, reject_5=False, reject_1=False)
    result = adfuller(s, maxlag=ADF_MAXLAG, autolag="AIC", regression="c")
    adf_stat, p_value, lags_used, nobs = result[0], result[1], result[2], result[3]
    return dict(
        adf_stat=round(adf_stat, 4),
        p_value=round(p_value, 4),
        lags_used=int(lags_used),
        n_obs=int(nobs),
        reject_10=bool(p_value < 0.10),
        reject_5=bool(p_value < 0.05),
        reject_1=bool(p_value < 0.01),
    )


rows = []

for market in MARKETS:
    print(f"\n{'='*60}")
    print(f"  {market}")
    print(f"{'='*60}")

    try:
        par = fetch_ois_curves(market)
    except Exception as e:
        print(f"  DATA ERROR: {e}")
        continue

    try:
        zeros = bootstrap_curves(par)
    except Exception as e:
        print(f"  BOOTSTRAP ERROR: {e}")
        continue

    try:
        lr  = select_lambda(zeros, end=LAM_TRAIN_END)
        lam = lr["best_lambda"]
        print(f"  λ = {lam}  (train ≤ {LAM_TRAIN_END})")
    except Exception as e:
        lam = 2.5
        print(f"  λ selection failed ({e}), fallback λ=2.5")

    try:
        _, resids = fit_ns_panel(zeros, lam)
    except Exception as e:
        print(f"  FIT ERROR: {e}")
        continue

    print(f"  {resids.index[0].date()} → {resids.index[-1].date()} | "
          f"{resids.shape[0]} dates | {resids.shape[1]} tenors")
    print(f"\n  {'Tenor':>6}  {'N':>5}  {'ADF stat':>9}  {'p-value':>8}  "
          f"{'10%':>5}  {'5%':>5}  {'1%':>5}")
    print(f"  {'-'*60}")

    for tenor in sorted(resids.columns):
        adf = run_adf(resids[tenor])
        star = ("***" if adf["reject_1"] else
                "**"  if adf["reject_5"] else
                "*"   if adf["reject_10"] else "")
        print(f"  {tenor:>6.2f}Y  {adf['n_obs']:>5}  "
              f"{adf['adf_stat']:>9.4f}  {adf['p_value']:>8.4f}  "
              f"{'Y' if adf['reject_10'] else 'N':>5}  "
              f"{'Y' if adf['reject_5'] else 'N':>5}  "
              f"{'Y' if adf['reject_1'] else 'N':>5}  {star}")
        rows.append(dict(
            country=market,
            tenor_y=float(tenor),
            lambda_used=lam,
            **adf,
        ))

# Save results
results = pd.DataFrame(rows)
results = results.sort_values(["country", "tenor_y"]).reset_index(drop=True)
results.to_parquet(OUT_PATH, index=False)
print(f"\n\nResults saved → {OUT_PATH}")
print(f"Total rows: {len(results)}")

# Summary: rejection rates by market
print("\n── Rejection rates (% of tenors rejecting H0: unit root) ──────────────")
summary = (
    results.groupby("country")[["reject_10", "reject_5", "reject_1"]]
    .mean()
    .mul(100)
    .round(1)
)
summary.columns = ["10% (%)", "5% (%)", "1% (%)"]
print(summary.to_string())
