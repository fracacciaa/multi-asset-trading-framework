"""
DV01-neutral spread signal diagnostics.

For each market, runs:
  1. Full NS signal (reference)
  2. 2Y-10Y DV01-neutral spread
  3. Reports coverage ratio, factor loadings, Sharpe for both.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from rates.production.ois_data import fetch_ois_curves
from rates.production.ns_classic.bootstrap import bootstrap_curves
from rates.production.ns_classic.ns_fit import select_lambda
from rates.production.ns_classic.pnl import classic_ns_pnl_repriced
from rates.production.ns_classic.signal import classic_ns_signal
from rates.production.ns_classic.spread_signal import spread_pnl_repriced

TARGET_VOL  = 100.0
BURN_IN     = 252
LAM_TRAIN_END = "2015-12-31"
T1, T2      = 2.0, 10.0

MARKETS = ["US", "EUR", "UK", "Japan", "Canada", "Australia",
           "New Zealand", "Sweden"]


def _sr(pnl):
    v = pnl.dropna()
    return float(np.sqrt(252) * v.mean() / v.std()) if len(v) > 30 else float("nan")


def _factor_loadings(pnl, betas):
    db  = betas.diff() * 100.0
    idx = pnl.dropna().index.intersection(db.dropna().index)
    if len(idx) < 100:
        return None, float("nan")
    y   = pnl.loc[idx].to_numpy(float)
    Xr  = db.loc[idx].to_numpy(float)
    Xc  = np.column_stack([np.ones(len(y)), Xr])
    b, _, _, _ = np.linalg.lstsq(Xc, y, rcond=None)
    res = y - Xc @ b
    n, k = Xc.shape
    s2  = np.dot(res, res) / (n - k)
    se  = np.sqrt(np.maximum(np.diag(s2 * np.linalg.pinv(Xc.T @ Xc)), 0))
    t   = b / np.where(se > 0, se, np.nan)
    ss_tot = np.dot(y - y.mean(), y - y.mean())
    r2  = 1 - np.dot(res, res) / ss_tot if ss_tot > 0 else float("nan")
    return pd.DataFrame(
        {"coef": b[1:], "t-stat": t[1:]},
        index=["Δβ0", "Δβ1", "Δβ2"]
    ), r2


def _cov_stats(cov_ratio):
    v = cov_ratio.dropna()
    return float(v.mean()), float(v.median()), float((v < 0.5).mean())


def run(market):
    sep = "=" * 68
    print(f"\n{sep}\n  {market}\n{sep}")

    try:
        par = fetch_ois_curves(market)
    except Exception as e:
        print(f"  DATA ERROR: {e}"); return

    # λ selection
    try:
        zeros = bootstrap_curves(par)
        lr    = select_lambda(zeros, end=LAM_TRAIN_END)
        lam   = lr["best_lambda"]
    except Exception as e:
        lam = 2.5
        print(f"  λ fallback 2.5 ({e})")

    print(f"  λ = {lam}  ({T1:.0f}Y-{T2:.0f}Y spread)")

    # ── Full NS signal (reference) ────────────────────────────────────────────
    try:
        pnl_full, z_scores, resids = classic_ns_pnl_repriced(
            par, lam, target_ann_vol=TARGET_VOL, burn_in=BURN_IN
        )
        _, betas, _ = classic_ns_signal(zeros, lam, burn_in=BURN_IN)
    except Exception as e:
        print(f"  FULL SIGNAL ERROR: {e}"); return

    # ── Spread signal ─────────────────────────────────────────────────────────
    # Check both tenors exist in this market
    tenor_cols = par.columns.to_numpy(float)
    if not (any(abs(tenor_cols - T1) < 0.1) and any(abs(tenor_cols - T2) < 0.1)):
        print(f"  Tenors {T1}Y / {T2}Y not available, skipping spread."); return

    try:
        pnl_spread, sp_z, cov, k_arr = spread_pnl_repriced(
            par, lam, T1, T2, target_ann_vol=TARGET_VOL, burn_in=BURN_IN
        )
    except Exception as e:
        print(f"  SPREAD ERROR: {e}"); return

    # ── Coverage ratio ────────────────────────────────────────────────────────
    print("\n  ── COVERAGE RATIO ─────────────────────────────────────────────")
    cov_mean, cov_med, frac_lt50 = _cov_stats(cov)
    print(f"  Mean: {cov_mean:.3f}  Median: {cov_med:.3f}  "
          f"Fraction of days <50%: {frac_lt50*100:.1f}%")
    print(f"  (1.0 = z_T1 = -z_T2 pure spread; 0.0 = z_T1 = z_T2 no spread info)")

    # ── Factor loadings ───────────────────────────────────────────────────────
    print("\n  ── FACTOR LOADINGS ────────────────────────────────────────────")
    for label, pnl_s in [("Full signal", pnl_full), ("Spread only", pnl_spread)]:
        fl, r2 = _factor_loadings(pnl_s, betas)
        print(f"\n  {label}  (R²={r2:.4f}):")
        if fl is not None:
            for name, row in fl.iterrows():
                flag = "  ← LEAK" if abs(row["t-stat"]) > 2.0 else ""
                print(f"    {name:<6}  coef={row['coef']:+.4f}  t={row['t-stat']:+.2f}{flag}")
        else:
            print("    insufficient data")

    # ── Sharpe ────────────────────────────────────────────────────────────────
    print("\n  ── SHARPE ─────────────────────────────────────────────────────")
    for label, pnl_s in [("Full signal", pnl_full), ("Spread only", pnl_spread)]:
        sr  = _sr(pnl_s)
        v   = pnl_s.dropna()
        ret = v.mean() * 252
        vol = v.std() * np.sqrt(252)
        n   = len(v) / 252
        print(f"  {label:<14}  SR={sr:+.2f}  "
              f"ret={ret:+.0f}  vol={vol:.0f} bp/yr  ({n:.1f}yr)")


if __name__ == "__main__":
    for m in MARKETS:
        run(m)
    print("\nAll markets done.")
