"""
Cross-market blend comparison for the NS classic MR signal.

Computes per-market PnL (same IS/OOS split as is_oos_run.py),
then blends the 8 markets together using every available static
and rolling method.  Vol-normalises using IS-window sigma before
fitting so the blend isn't dominated by the highest-vol market.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from rates.production.ois_data import fetch_ois_curves
from rates.production.ns_classic.bootstrap import bootstrap_curves
from rates.production.ns_classic.ns_fit import select_lambda
from rates.production.ns_classic.pnl import classic_ns_pnl_repriced
from equities.blend import (
    equal_weight, inv_vol_blend, erc_blend, hrp_blend,
    sharpe_heuristic, msrr_ridge, msrr_lasso, RollingMSRR,
    inv_msfe_blend, pcr_blend, pls_blend,
)

IS_END        = pd.Timestamp("2015-10-30")
OOS_START     = pd.Timestamp("2015-10-31")
TARGET_VOL    = 100.0
BURN_IN       = 252
LAM_TRAIN_END = "2015-12-31"

MARKETS = ["US", "EUR", "UK", "Japan", "Canada", "Australia", "New Zealand", "Sweden"]


def _metrics(r: pd.Series) -> dict:
    v = r.dropna()
    if len(v) < 30:
        return dict(sr=float("nan"), ret=float("nan"), vol=float("nan"), dd=float("nan"), n_yr=0.0)
    cum = v.cumsum()
    dd  = float((cum - cum.cummax()).min())
    return dict(
        sr   = float(np.sqrt(252) * v.mean() / v.std()),
        ret  = float(v.mean() * 252),
        vol  = float(v.std() * np.sqrt(252)),
        dd   = dd,
        n_yr = len(v) / 252,
    )


def _fmt(m: dict) -> str:
    if np.isnan(m["sr"]):
        return "insufficient data"
    return (f"SR={m['sr']:+.2f}  ret={m['ret']:+.0f}  vol={m['vol']:.0f} bp/yr  "
            f"dd={m['dd']:+.0f} bp")


def _apply_static(bw, F_norm: pd.DataFrame, F_raw: pd.DataFrame) -> pd.Series:
    """Fit on F_norm, report on F_raw (bp units)."""
    w = bw.weights.reindex(F_norm.columns).fillna(0.0)
    norm = w.abs().sum()
    if norm < 1e-12:
        return pd.Series(np.nan, index=F_raw.index)
    # weights calibrated for unit-vol series → scale by per-market IS vol
    return (F_raw * (w / norm)).sum(axis=1)


def _apply_rolling(w_df: pd.DataFrame, F_norm: pd.DataFrame, F_raw: pd.DataFrame) -> pd.Series:
    """Fit on F_norm, report on F_raw (bp units)."""
    idx = F_raw.index.intersection(w_df.dropna(how="any").index)
    if len(idx) == 0:
        return pd.Series(np.nan, index=F_raw.index)
    F_r, w_ = F_raw.loc[idx], w_df.loc[idx]
    norm = w_.abs().sum(axis=1).replace(0.0, np.nan)
    return (F_r * w_.div(norm, axis=0)).sum(axis=1)


# ── Step 1: per-market PnL ────────────────────────────────────────────────────

print("Computing per-market PnL...")
pnl_dict: dict[str, pd.Series] = {}

for market in MARKETS:
    try:
        par   = fetch_ois_curves(market)
        zeros = bootstrap_curves(par)
        lr    = select_lambda(zeros, end=LAM_TRAIN_END)
        lam   = lr["best_lambda"]
    except Exception:
        lam = 2.5
    try:
        pnl, _, _ = classic_ns_pnl_repriced(
            par, lam, target_ann_vol=TARGET_VOL, burn_in=BURN_IN
        )
        pnl_dict[market] = pnl
        print(f"  {market}: OK")
    except Exception as e:
        print(f"  {market}: ERROR — {e}")

# ── Step 2: managed-return matrix ─────────────────────────────────────────────
# Restrict to dates where ALL markets have valid PnL (avoids NaN in blend fits)

F_raw = pd.DataFrame(pnl_dict).sort_index().dropna(how="any")

F_is_raw  = F_raw.loc[F_raw.index <= IS_END]
F_oos_raw = F_raw.loc[F_raw.index >= OOS_START]

# Vol-normalise using IS window so blend weights are in unit-vol space
sigma_is = F_is_raw.std() * np.sqrt(252)
sigma_is = sigma_is.replace(0.0, np.nan)

F_is   = F_is_raw.div(sigma_is)
F_oos  = F_oos_raw.div(sigma_is)
F_full = pd.concat([F_is, F_oos]).sort_index()

print(f"\n  IS  : {F_is.index[0].date()} → {F_is.index[-1].date()}  "
      f"({len(F_is)/252:.1f}yr, {F_is.shape[1]} markets)")
print(f"  OOS : {F_oos.index[0].date()} → {F_oos.index[-1].date()}  "
      f"({len(F_oos)/252:.1f}yr, {F_oos.shape[1]} markets)")

# ── Step 3: blend methods ─────────────────────────────────────────────────────

results: dict[str, tuple[pd.Series, pd.Series]] = {}

# Static methods — IS fit, constant weights applied to both periods
static_methods = [
    ("Equal weight",      lambda: equal_weight(list(F_is.columns))),
    ("Inv-vol",           lambda: inv_vol_blend(F_is)),
    ("ERC",               lambda: erc_blend(F_is)),
    ("HRP",               lambda: hrp_blend(F_is)),
    ("Sharpe heuristic",  lambda: sharpe_heuristic(F_is)),
    ("Static Ridge",      lambda: msrr_ridge(F_is, lam=0.01)),
    ("Static Lasso",      lambda: msrr_lasso(F_is, lam=0.01)),
    ("Inv-MSFE",          lambda: inv_msfe_blend(F_is)),
    ("PCR (2)",           lambda: pcr_blend(F_is, n_components=2)),
    ("PLS (2)",           lambda: pls_blend(F_is, n_components=2)),
]

for name, fn in static_methods:
    try:
        bw = fn()
        results[name] = (
            _apply_static(bw, F_is, F_is_raw),
            _apply_static(bw, F_oos, F_oos_raw),
        )
    except Exception as e:
        print(f"  {name}: SKIP — {e}")

# Rolling methods — walk-forward, no lookahead
rolling_methods = [
    ("Rolling OLS",   RollingMSRR(penalty="ols",   lookback=252, min_obs=63)),
    ("Rolling Ridge", RollingMSRR(penalty="ridge",  lookback=252, min_obs=63, lam=0.01)),
    ("Rolling Lasso", RollingMSRR(penalty="lasso",  lookback=252, min_obs=63, alpha=0.01)),
]

for name, model in rolling_methods:
    try:
        w_raw = model._roll(F_full)
        norm  = w_raw.abs().sum(axis=1).replace(0.0, np.nan)
        w_all = w_raw.div(norm, axis=0)
        results[name] = (
            _apply_rolling(w_all.reindex(F_is.index),  F_is, F_is_raw),
            _apply_rolling(w_all.reindex(F_oos.index), F_oos, F_oos_raw),
        )
    except Exception as e:
        print(f"  {name}: SKIP — {e}")

# ── Step 4: print results ──────────────────────────────────────────────────────

W = 80
print(f"\n{'='*W}")
print(f"  NS CLASSIC MR — CROSS-MARKET BLEND COMPARISON")
print(f"  IS ≤ {IS_END.date()}   |   OOS ≥ {OOS_START.date()}")
print(f"  vol-normalised per-market before blend; {len(MARKETS)} markets")
print(f"{'='*W}")
print(f"  {'Method':<20}  {'Period':<6}  {'SR':>7}  {'ret bp/yr':>10}  "
      f"{'vol bp/yr':>10}  {'MDD bp':>8}")
print("  " + "─" * (W - 2))

for name, (r_is, r_oos) in results.items():
    for period, r in [("IS", r_is), ("OOS", r_oos)]:
        m = _metrics(r)
        sr  = f"{m['sr']:+.2f}"   if not np.isnan(m["sr"])  else "   n/a"
        ret = f"{m['ret']:+.0f}"  if not np.isnan(m["ret"]) else "   n/a"
        vol = f"{m['vol']:.0f}"   if not np.isnan(m["vol"]) else "   n/a"
        dd  = f"{m['dd']:+.0f}"   if not np.isnan(m["dd"])  else "   n/a"
        print(f"  {name:<20}  {period:<6}  {sr:>7}  {ret:>10}  {vol:>10}  {dd:>8}")
    print("  " + "─" * (W - 2))

print(f"{'='*W}")
print("\nDone.")
