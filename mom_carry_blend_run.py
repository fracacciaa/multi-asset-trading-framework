"""
Cross-market blend comparison for momentum and carry signals.

Mirrors ns_blend_run.py: vol-normalise per-market IS PnL, then apply every
static and rolling blend method.  Uses fast approximations (no swap repricing)
since the blend-method ranking is insensitive to that level of precision.

Momentum  : pos(t) × ret_d(t) summed across tenors, per market.
Carry     : §6.2 DV01-approximation via ois_par_carry_pnl, per market.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from rates.production.momentum.returns import build_return_panels
from rates.production.momentum.signal_a import build_signal_a_daily, DM_MARKETS
from rates.production.carry.carry_ois_par import ois_par_carry_pnl, OIS_PAR_MARKETS
from equities.blend import (
    equal_weight, inv_vol_blend, erc_blend, hrp_blend,
    sharpe_heuristic, msrr_ridge, msrr_lasso, RollingMSRR,
    inv_msfe_blend, pcr_blend, pls_blend,
)

IS_END    = pd.Timestamp("2015-10-30")
OOS_START = pd.Timestamp("2015-10-31")

MOM_MARKETS   = ["US", "EUR", "UK", "Japan", "Canada", "Australia", "New Zealand", "Sweden"]
CARRY_MARKETS = [m for m in OIS_PAR_MARKETS if m in MOM_MARKETS]  # intersection


# ── helpers (identical to ns_blend_run.py) ────────────────────────────────────

def _metrics(r: pd.Series) -> dict:
    v = r.dropna()
    if len(v) < 30:
        return dict(sr=float("nan"), ret=float("nan"), vol=float("nan"), dd=float("nan"))
    cum = v.cumsum()
    dd  = float((cum - cum.cummax()).min())
    return dict(
        sr  = float(np.sqrt(252) * v.mean() / v.std()),
        ret = float(v.mean() * 252),
        vol = float(v.std() * np.sqrt(252)),
        dd  = dd,
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


def run_blend_suite(signal_name: str, pnl_dict: dict[str, pd.Series]) -> None:
    """Apply every static and rolling blend method and print IS / OOS table."""

    F_raw = pd.DataFrame(pnl_dict).sort_index().dropna(how="any")
    if F_raw.empty:
        print(f"  No data for {signal_name}"); return

    F_is_raw  = F_raw.loc[F_raw.index <= IS_END]
    F_oos_raw = F_raw.loc[F_raw.index >= OOS_START]

    if len(F_is_raw) < 63 or len(F_oos_raw) < 30:
        print(f"  Insufficient data for {signal_name}"); return

    sigma_is = F_is_raw.std() * np.sqrt(252)
    sigma_is = sigma_is.replace(0.0, np.nan)

    F_is   = F_is_raw.div(sigma_is)
    F_oos  = F_oos_raw.div(sigma_is)
    F_full = pd.concat([F_is, F_oos]).sort_index()

    print(f"\n  IS : {F_is.index[0].date()} → {F_is.index[-1].date()}  "
          f"({len(F_is)/252:.1f}yr, {F_is.shape[1]} markets)")
    print(f"  OOS: {F_oos.index[0].date()} → {F_oos.index[-1].date()}  "
          f"({len(F_oos)/252:.1f}yr)")

    results: dict[str, tuple[pd.Series, pd.Series]] = {}

    static_methods = [
        ("Equal weight",     lambda: equal_weight(list(F_is.columns))),
        ("Inv-vol",          lambda: inv_vol_blend(F_is)),
        ("ERC",              lambda: erc_blend(F_is)),
        ("HRP",              lambda: hrp_blend(F_is)),
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
        except Exception:
            pass

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
        except Exception:
            pass

    W = 80
    print(f"\n{'='*W}")
    print(f"  {signal_name.upper()} — CROSS-MARKET BLEND COMPARISON")
    print(f"  IS ≤ {IS_END.date()}   |   OOS ≥ {OOS_START.date()}")
    print(f"  vol-normalised per-market before blend; {F_raw.shape[1]} markets")
    print(f"{'='*W}")
    print(f"  {'Method':<20}  {'Period':<6}  {'SR':>7}  {'ret bp/yr':>10}  "
          f"{'vol bp/yr':>10}  {'MDD bp':>8}")
    print("  " + "─" * (W - 2))

    for name, (r_is, r_oos) in results.items():
        for period, r in [("IS", r_is), ("OOS", r_oos)]:
            m  = _metrics(r)
            sr  = f"{m['sr']:+.2f}"   if not np.isnan(m["sr"])  else "   n/a"
            ret = f"{m['ret']:+.0f}"  if not np.isnan(m["ret"]) else "   n/a"
            vol = f"{m['vol']:.0f}"   if not np.isnan(m["vol"]) else "   n/a"
            dd  = f"{m['dd']:+.0f}"   if not np.isnan(m["dd"])  else "   n/a"
            print(f"  {name:<20}  {period:<6}  {sr:>7}  {ret:>10}  {vol:>10}  {dd:>8}")
        print("  " + "─" * (W - 2))

    print(f"{'='*W}")


# ── Step 1: Momentum per-market PnL ──────────────────────────────────────────

print("Computing momentum per-market PnL (pos × ret_d approximation)...")
mom_pnl: dict[str, pd.Series] = {}

for market in MOM_MARKETS:
    try:
        panels = build_return_panels(market)
        mom_df = build_signal_a_daily({market: panels}, k=252)
        if mom_df.empty:
            continue
        pos_df = mom_df.xs(market, level="market", axis=1)
        ret_d  = panels["ret_d"]
        common_cols = pos_df.columns.intersection(ret_d.columns)
        common_idx  = pos_df.index.intersection(ret_d.index)
        pnl = (
            pos_df.loc[common_idx, common_cols]
            * ret_d.loc[common_idx, common_cols]
        ).sum(axis=1, min_count=1)
        mom_pnl[market] = pnl
        print(f"  {market}: {len(pnl.dropna())} days")
    except Exception as e:
        print(f"  {market}: ERROR — {e}")


# ── Step 2: Carry per-market PnL ─────────────────────────────────────────────

print("\nComputing carry per-market PnL (§6.2 approximation)...")
carry_pnl_frames = ois_par_carry_pnl(markets=CARRY_MARKETS)

carry_pnl: dict[str, pd.Series] = {}
for mkt in CARRY_MARKETS:
    if mkt in carry_pnl_frames.columns:
        s = carry_pnl_frames[mkt].dropna()
        # skip markets with no data in the IS window
        is_days = int((s.index <= IS_END).sum())
        if is_days < 63:
            print(f"  {mkt}: {len(s)} days  SKIPPED (only {is_days} IS days)")
            continue
        carry_pnl[mkt] = carry_pnl_frames[mkt]
        print(f"  {mkt}: {len(s)} days")
    else:
        print(f"  {mkt}: missing")


# ── Step 3: Run blend suites ──────────────────────────────────────────────────

run_blend_suite("MOMENTUM", mom_pnl)
run_blend_suite("CARRY",    carry_pnl)

print("\nDone.")
