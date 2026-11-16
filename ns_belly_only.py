"""
Belly-only NS signal: restrict positions to belly tenors, compare vs full curve.

Two belly definitions tested side by side:
  Fixed  : 5Y – 15Y across all markets
  λ-rel  : tenors where NS curvature loading > 50% of its peak value
           (i.e. within the main hump of the β₂ factor, per-market)

Portfolio P&L = sum of per-tenor DV01 P&L across selected tenors.
Vol-scaled to 100bp/yr at portfolio level for fair Sharpe comparison.
Saves: rates/signals/ns_belly_sharpe.parquet
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from rates.production.ois_data import fetch_ois_curves
from rates.production.ns_classic.bootstrap import bootstrap_curves
from rates.production.ns_classic.ns_fit import (
    select_lambda, fit_ns_panel, ns_basis, DEFAULT_MIN_TENOR,
)
from rates.production.ns_classic.signal import (
    DEFAULT_BURN_IN, DEFAULT_VOL_FLOOR, build_dv01_positions,
    compute_zscores,
)

MARKETS       = ["US", "EUR", "UK", "Japan", "Canada", "Australia",
                 "New Zealand", "Sweden", "Switzerland", "India", "China"]
LAM_TRAIN_END = "2015-12-31"
BURN_IN       = DEFAULT_BURN_IN
DZ_CAP        = 0.25
TARGET_VOL    = 100.0   # bp/yr for vol-scaled Sharpe
OUT_PATH      = "rates/signals/ns_belly_sharpe.parquet"

FIXED_LO, FIXED_HI = 5.0, 15.0


def _belly_lam(tenors: np.ndarray, lam: float, threshold: float = 0.50) -> np.ndarray:
    """Tenors where curvature loading > threshold × peak curvature loading."""
    B = ns_basis(tenors, lam)
    curv = B[:, 2]
    return tenors[curv >= threshold * curv.max()]


def _vol_scale(port_pnl: pd.Series, target: float = TARGET_VOL) -> pd.Series:
    """Scale daily PnL series to target annualised vol."""
    ann_vol = port_pnl.dropna().std() * np.sqrt(252)
    if ann_vol > 0:
        return port_pnl * (target / ann_vol)
    return port_pnl


def _sr(pnl: pd.Series) -> float:
    v = pnl.dropna()
    return float(np.sqrt(252) * v.mean() / v.std()) if len(v) > 30 else float("nan")


def _portfolio_pnl(z: pd.DataFrame, zeros: pd.DataFrame,
                   tenor_mask: np.ndarray) -> pd.Series:
    """Sum DV01 PnL across selected tenors."""
    z_sel = z[[c for c in z.columns if float(c) in set(tenor_mask)]]
    if z_sel.empty:
        return pd.Series(dtype=float)
    dv01, _ = build_dv01_positions(z_sel, zeros, k=1.0)
    tenors   = dv01.columns
    dz       = zeros.reindex(columns=tenors).diff().clip(-DZ_CAP, DZ_CAP)
    return (dv01.shift(1) * dz * 100.0).sum(axis=1)


rows = []

print(f"\n  {'Country':<16}  {'λ':>4}  {'Belly (λ-rel)':>18}  "
      f"{'SR full':>8}  {'SR fixed 5-15Y':>14}  {'SR λ-rel belly':>14}")
print("  " + "-" * 82)

for market in MARKETS:
    try:
        par   = fetch_ois_curves(market)
        zeros = bootstrap_curves(par)
    except Exception as e:
        print(f"  {market:<16}  ERROR: {e}"); continue

    try:
        lam = select_lambda(zeros, end=LAM_TRAIN_END)["best_lambda"]
    except Exception:
        lam = 2.5

    try:
        _, resids = fit_ns_panel(zeros, lam)
        z_scores  = compute_zscores(resids, burn_in=BURN_IN)
    except Exception as e:
        print(f"  {market:<16}  FIT ERROR: {e}"); continue

    all_tenors  = z_scores.columns.to_numpy(dtype=float)
    tradable    = all_tenors[all_tenors >= DEFAULT_MIN_TENOR]

    belly_fixed = tradable[(tradable >= FIXED_LO) & (tradable <= FIXED_HI)]
    belly_lam   = _belly_lam(tradable, lam)

    pnl_full  = _portfolio_pnl(z_scores, zeros, tradable)
    pnl_fixed = _portfolio_pnl(z_scores, zeros, belly_fixed)
    pnl_lam   = _portfolio_pnl(z_scores, zeros, belly_lam)

    sr_full  = _sr(_vol_scale(pnl_full))
    sr_fixed = _sr(_vol_scale(pnl_fixed))
    sr_lam   = _sr(_vol_scale(pnl_lam))

    belly_lam_str = " ".join(f"{t:.0f}Y" for t in belly_lam)

    print(f"  {market:<16}  {lam:>4.1f}  {belly_lam_str:>18}  "
          f"{sr_full:>8.3f}  {sr_fixed:>14.3f}  {sr_lam:>14.3f}")

    rows.append(dict(
        country=market, lambda_used=lam,
        belly_fixed_tenors=str([float(t) for t in belly_fixed]),
        belly_lam_tenors=str([float(t) for t in belly_lam]),
        sharpe_full=round(sr_full, 4),
        sharpe_fixed_belly=round(sr_fixed, 4),
        sharpe_lam_belly=round(sr_lam, 4),
        delta_fixed=round(sr_fixed - sr_full, 4),
        delta_lam=round(sr_lam - sr_full, 4),
    ))

results = pd.DataFrame(rows)
results.to_parquet(OUT_PATH, index=False)

print(f"\n\nSaved → {OUT_PATH}")

print("\n── Delta vs full curve ──────────────────────────────────────────────────")
print(f"\n  {'Country':<16}  {'Δ fixed 5-15Y':>14}  {'Δ λ-rel belly':>14}")
print("  " + "-" * 48)
for _, r in results.iterrows():
    df = r["delta_fixed"]; dl = r["delta_lam"]
    ff = ("▲" if df > 0 else "▼"); fl = ("▲" if dl > 0 else "▼")
    print(f"  {r['country']:<16}  {ff} {df:>+12.3f}  {fl} {dl:>+12.3f}")

print(f"\n  Mean Δ fixed  : {results['delta_fixed'].mean():>+.3f}")
print(f"  Mean Δ λ-rel  : {results['delta_lam'].mean():>+.3f}")
print(f"\n  Markets improved (fixed) : "
      f"{(results['delta_fixed'] > 0).sum()} / {len(results)}")
print(f"  Markets improved (λ-rel) : "
      f"{(results['delta_lam'] > 0).sum()} / {len(results)}")
