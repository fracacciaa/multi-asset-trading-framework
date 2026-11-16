"""
Compare per-tenor Sharpe: z-score with mean subtraction vs without.

Original:  z = (resid - rolling_mean) / rolling_std
Variant:   z = resid / rolling_std   (direction from raw NS residual)

Everything else identical: same lambda, burn-in, DZ cap, k=1.
Saves: rates/signals/ns_sharpe_demean_compare.parquet
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from rates.production.ois_data import fetch_ois_curves
from rates.production.ns_classic.bootstrap import bootstrap_curves
from rates.production.ns_classic.ns_fit import select_lambda, fit_ns_panel
from rates.production.ns_classic.signal import (
    DEFAULT_BURN_IN, DEFAULT_VOL_FLOOR, build_dv01_positions,
)

MARKETS       = ["US", "EUR", "UK", "Japan", "Canada", "Australia",
                 "New Zealand", "Sweden", "Switzerland", "India", "China"]
LAM_TRAIN_END = "2015-12-31"
BURN_IN       = DEFAULT_BURN_IN
VOL_FLOOR     = DEFAULT_VOL_FLOOR
DZ_CAP        = 0.25
OUT_PATH      = "rates/signals/ns_sharpe_demean_compare.parquet"


def zscores_demeaned(resids: pd.DataFrame) -> pd.DataFrame:
    mu = resids.expanding(min_periods=BURN_IN).mean().shift(1)
    sd = resids.expanding(min_periods=BURN_IN).std(ddof=1).shift(1)
    z  = (resids - mu) / sd.clip(lower=VOL_FLOOR)
    z.iloc[:BURN_IN] = np.nan
    return z


def zscores_raw(resids: pd.DataFrame) -> pd.DataFrame:
    sd = resids.expanding(min_periods=BURN_IN).std(ddof=1).shift(1)
    z  = resids / sd.clip(lower=VOL_FLOOR)
    z.iloc[:BURN_IN] = np.nan
    return z


def per_tenor_pnl(z: pd.DataFrame, zeros: pd.DataFrame) -> pd.DataFrame:
    dv01, _ = build_dv01_positions(z, zeros, k=1.0)
    tenors   = dv01.columns
    dz  = zeros.reindex(columns=tenors).diff().clip(-DZ_CAP, DZ_CAP)
    return (dv01.shift(1) * dz * 100.0).rename(columns=float)


def sharpe(s: pd.Series) -> float:
    v = s.dropna()
    if len(v) < 30 or v.std() == 0:
        return float("nan")
    return float(np.sqrt(252) * v.mean() / v.std())


rows = []

for market in MARKETS:
    print(f"\n{'='*70}\n  {market}\n{'='*70}")

    try:
        par   = fetch_ois_curves(market)
        zeros = bootstrap_curves(par)
    except Exception as e:
        print(f"  ERROR: {e}"); continue

    try:
        lam = select_lambda(zeros, end=LAM_TRAIN_END)["best_lambda"]
    except Exception:
        lam = 2.5

    try:
        _, resids = fit_ns_panel(zeros, lam)
    except Exception as e:
        print(f"  FIT ERROR: {e}"); continue

    pnl_dm  = per_tenor_pnl(zscores_demeaned(resids), zeros)
    pnl_raw = per_tenor_pnl(zscores_raw(resids), zeros)

    print(f"\n  {'Tenor':>6}  {'SR demeaned':>12}  {'SR raw resid':>13}  {'Δ Sharpe':>10}  {'Flip?':>6}")
    print(f"  {'-'*58}")

    for tenor in sorted(pnl_dm.columns):
        sr_dm  = sharpe(pnl_dm[tenor])
        sr_raw = sharpe(pnl_raw[tenor])
        n      = int(pnl_dm[tenor].dropna().shape[0])
        if n == 0:
            continue

        delta = (sr_raw - sr_dm) if not (np.isnan(sr_raw) or np.isnan(sr_dm)) else float("nan")
        flip  = ("FLIP" if not np.isnan(sr_dm) and not np.isnan(sr_raw)
                        and np.sign(sr_dm) != np.sign(sr_raw) else "")

        print(f"  {tenor:>6.0f}Y  {sr_dm:>12.3f}  {sr_raw:>13.3f}  {delta:>+10.3f}  {flip:>6}")

        rows.append(dict(
            country=market, tenor_y=float(tenor), lambda_used=lam, n_obs=n,
            sharpe_demeaned=round(sr_dm, 4) if not np.isnan(sr_dm) else float("nan"),
            sharpe_raw=round(sr_raw, 4) if not np.isnan(sr_raw) else float("nan"),
            delta_sharpe=round(delta, 4) if not np.isnan(delta) else float("nan"),
            flipped=bool(flip == "FLIP"),
        ))

results = pd.DataFrame(rows)
results.to_parquet(OUT_PATH, index=False)
print(f"\n\nSaved → {OUT_PATH}  ({len(results)} rows)")

# ── Summary ────────────────────────────────────────────────────────────────────
tradable = results[results["n_obs"] >= 500].dropna(subset=["sharpe_demeaned","sharpe_raw"])

print("\n── Mean Sharpe by country (n≥500): demeaned vs raw ─────────────────────")
print(f"\n  {'Country':<16}  {'SR demeaned':>12}  {'SR raw':>8}  {'Δ':>7}  {'Flips':>6}")
print(f"  {'-'*56}")
for c in MARKETS:
    sub = tradable[tradable["country"] == c]
    if sub.empty: continue
    n_flip = int(sub["flipped"].sum())
    print(f"  {c:<16}  {sub['sharpe_demeaned'].mean():>12.3f}  "
          f"{sub['sharpe_raw'].mean():>8.3f}  "
          f"{sub['sharpe_raw'].mean() - sub['sharpe_demeaned'].mean():>+7.3f}  "
          f"{n_flip:>6}")

print("\n── Pairs that flip sign (demeaned → raw) ────────────────────────────────")
flips = (tradable[tradable["flipped"]]
         .sort_values("delta_sharpe", ascending=False)
         [["country","tenor_y","sharpe_demeaned","sharpe_raw","delta_sharpe"]])
print(flips.to_string(index=False))

print("\n── Biggest improvements (raw − demeaned, top 15) ────────────────────────")
top = (tradable.sort_values("delta_sharpe", ascending=False)
               .head(15)
               [["country","tenor_y","sharpe_demeaned","sharpe_raw","delta_sharpe"]])
print(top.to_string(index=False))
