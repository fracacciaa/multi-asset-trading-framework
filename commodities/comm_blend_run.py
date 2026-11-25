"""
Commodities signal blend sweep.

Mirrors mom_carry_blend_run.py: vol-normalise the 3 pre-computed signal PnL
parquets on IS σ, then apply every static and rolling blend method.

Signals (from commodities/signals/):
    tsmom_252    — 252-day time-series momentum
    xs_carry_enh — enhanced cross-sectional carry (tenor-norm + seasonal + sector)
    curvature    — NS butterfly (curve curvature)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from commodities.blend import load_managed_returns, IS_END, OOS_START
from equities.blend import (
    equal_weight, inv_vol_blend, erc_blend,
    sharpe_heuristic, msrr_ridge, msrr_lasso, RollingMSRR,
    inv_msfe_blend, pcr_blend, pls_blend,
)

IS_END_TS    = pd.Timestamp(IS_END)
OOS_START_TS = pd.Timestamp(OOS_START)


# ── helpers ───────────────────────────────────────────────────────────────────

def _metrics(r: pd.Series) -> dict:
    v = r.dropna()
    if len(v) < 30:
        return dict(sr=float("nan"), ret=float("nan"), vol=float("nan"), mdd=float("nan"))
    cum = (1 + v * 0.10).cumprod()
    mdd = float((cum / cum.cummax() - 1).min() * 100)
    return dict(
        sr  = float(np.sqrt(252) * v.mean() / v.std()),
        ret = float(v.mean() * 252 * 100),    # annualised %, vol-normalised units
        vol = float(v.std() * np.sqrt(252) * 100),
        mdd = mdd,
    )


def _apply_static(bw, F_norm: pd.DataFrame, F_raw: pd.DataFrame) -> pd.Series:
    w    = bw.weights.reindex(F_norm.columns).fillna(0.0)
    norm = w.abs().sum()
    if norm < 1e-12:
        return pd.Series(np.nan, index=F_raw.index)
    return (F_raw * (w / norm)).sum(axis=1)


def _apply_rolling(w_df: pd.DataFrame, F_norm: pd.DataFrame, F_raw: pd.DataFrame) -> pd.Series:
    idx = F_raw.index.intersection(w_df.dropna(how="any").index)
    if len(idx) == 0:
        return pd.Series(np.nan, index=F_raw.index)
    F_r, w_ = F_raw.loc[idx], w_df.loc[idx]
    norm = w_.abs().sum(axis=1).replace(0.0, np.nan)
    return (F_r * w_.div(norm, axis=0)).sum(axis=1)


# ── main blend sweep ──────────────────────────────────────────────────────────

from commodities.blend import build_signal_parquets, SIGNAL_FILES, _SIGNALS_DIR

_missing = [_SIGNALS_DIR / f for f in SIGNAL_FILES.values() if not (_SIGNALS_DIR / f).exists()]
if _missing:
    print(f"Signal parquets missing ({len(_missing)}) — building now...")
    build_signal_parquets()

print("Loading commodities managed-return parquets...")
F_raw = load_managed_returns()
print(f"  Signals : {list(F_raw.columns)}")
print(f"  Date range: {F_raw.index[0].date()} → {F_raw.index[-1].date()}  ({len(F_raw)} rows)")

F_is_raw  = F_raw.loc[F_raw.index <= IS_END_TS]
F_oos_raw = F_raw.loc[F_raw.index >= OOS_START_TS]

if len(F_is_raw) < 63 or len(F_oos_raw) < 30:
    raise RuntimeError(f"Insufficient data: IS={len(F_is_raw)}, OOS={len(F_oos_raw)}")

sigma_is = F_is_raw.std() * np.sqrt(252)
sigma_is = sigma_is.replace(0.0, np.nan)

F_is   = F_is_raw.div(sigma_is)
F_oos  = F_oos_raw.div(sigma_is)
F_full = pd.concat([F_is, F_oos]).sort_index()

print(f"\n  IS : {F_is.index[0].date()} → {F_is.index[-1].date()}  "
      f"({len(F_is)/252:.1f}yr, {F_is.shape[1]} signals)")
print(f"  OOS: {F_oos.index[0].date()} → {F_oos.index[-1].date()}  "
      f"({len(F_oos)/252:.1f}yr)")

# ── individual signal stats ───────────────────────────────────────────────────
W = 86
print(f"\n{'='*W}")
print(f"  INDIVIDUAL SIGNALS")
print(f"{'='*W}")
print(f"  {'Signal':<20}  {'Period':<6}  {'SR':>7}  {'ret %':>8}  {'vol %':>8}  {'MDD %':>8}")
print("  " + "─" * (W - 2))
for sig in F_is.columns:
    for period, F_p, F_raw_p in [("IS", F_is, F_is_raw), ("OOS", F_oos, F_oos_raw)]:
        m = _metrics(F_raw_p[sig])
        sr  = f"{m['sr']:+.2f}"   if not np.isnan(m["sr"])  else "   n/a"
        ret = f"{m['ret']:+.1f}"  if not np.isnan(m["ret"]) else "   n/a"
        vol = f"{m['vol']:.1f}"   if not np.isnan(m["vol"]) else "   n/a"
        mdd = f"{m['mdd']:+.1f}"  if not np.isnan(m["mdd"]) else "   n/a"
        print(f"  {sig:<20}  {period:<6}  {sr:>7}  {ret:>8}  {vol:>8}  {mdd:>8}")
    print("  " + "─" * (W - 2))
print(f"{'='*W}")

# ── blend sweep ───────────────────────────────────────────────────────────────

results: dict[str, tuple[pd.Series, pd.Series]] = {}

static_methods = [
    ("Equal weight",     lambda: equal_weight(list(F_is.columns))),
    ("Inv-vol",          lambda: inv_vol_blend(F_is)),
    ("ERC",              lambda: erc_blend(F_is)),
    ("Sharpe heuristic", lambda: sharpe_heuristic(F_is)),
    ("Static Ridge",     lambda: msrr_ridge(F_is, lam=0.01)),
    ("Static Lasso",     lambda: msrr_lasso(F_is, lam=0.01)),
    ("Inv-MSFE",         lambda: inv_msfe_blend(F_is)),
    ("PCR (2)",          lambda: pcr_blend(F_is, n_components=2)),
    ("PLS (2)",          lambda: pls_blend(F_is, n_components=2)),
]

for name, fn in static_methods:
    try:
        bw = fn()
        results[name] = (
            _apply_static(bw, F_is, F_is_raw),
            _apply_static(bw, F_oos, F_oos_raw),
        )
    except Exception as e:
        print(f"  [WARN] {name}: {e}")

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
        print(f"  [WARN] {name}: {e}")

# ── print table ───────────────────────────────────────────────────────────────

print(f"\n{'='*W}")
print(f"  COMMODITIES — SIGNAL BLEND COMPARISON  (3 signals, vol-normalised on IS σ)")
print(f"  IS ≤ {IS_END_TS.date()}   |   OOS ≥ {OOS_START_TS.date()}")
print(f"{'='*W}")
print(f"  {'Method':<20}  {'Period':<6}  {'SR':>7}  {'ret %':>8}  {'vol %':>8}  {'MDD %':>8}")
print("  " + "─" * (W - 2))

for name, (r_is, r_oos) in results.items():
    for period, r in [("IS", r_is), ("OOS", r_oos)]:
        m   = _metrics(r)
        sr  = f"{m['sr']:+.2f}"   if not np.isnan(m["sr"])  else "   n/a"
        ret = f"{m['ret']:+.1f}"  if not np.isnan(m["ret"]) else "   n/a"
        vol = f"{m['vol']:.1f}"   if not np.isnan(m["vol"]) else "   n/a"
        mdd = f"{m['mdd']:+.1f}"  if not np.isnan(m["mdd"]) else "   n/a"
        print(f"  {name:<20}  {period:<6}  {sr:>7}  {ret:>8}  {vol:>8}  {mdd:>8}")
    print("  " + "─" * (W - 2))

print(f"{'='*W}")

# ── print IS blend weights for each static method ────────────────────────────

print(f"\n  IS blend weights")
print(f"  {'Method':<20}" + "".join(f"  {s:>14}" for s in F_is.columns))
print("  " + "─" * (20 + 16 * len(F_is.columns)))

static_weight_methods = [
    ("Equal weight",     lambda: equal_weight(list(F_is.columns))),
    ("Inv-vol",          lambda: inv_vol_blend(F_is)),
    ("ERC",              lambda: erc_blend(F_is)),
    ("Sharpe heuristic", lambda: sharpe_heuristic(F_is)),
    ("Static Ridge",     lambda: msrr_ridge(F_is, lam=0.01)),
    ("Static Lasso",     lambda: msrr_lasso(F_is, lam=0.01)),
]
for name, fn in static_weight_methods:
    try:
        bw = fn()
        w  = bw.weights.reindex(F_is.columns).fillna(0.0)
        norm = w.abs().sum()
        w_n = (w / norm) if norm > 1e-12 else w
        print(f"  {name:<20}" + "".join(f"  {w_n.get(s, 0.0):>+13.1%}" for s in F_is.columns))
    except Exception:
        pass

print()
print("Done.")
