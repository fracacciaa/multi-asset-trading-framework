"""
Per-instrument blend sweep for commodity signals.

For each signal (tsmom_252, xs_carry_enh, curvature), computes per-instrument
vol-scaled managed returns (T × N instruments) and applies the full suite of
blend methods — finding the optimal way to combine ~35 commodities within each
signal.

Pipeline per signal:
    raw_signal[i,t]  →  vol_scale()  →  w[i,t]
    F[i,t] = w[i,t-1] × r[i,t]        (per-instrument managed return, NOT summed)
    F (T × N)  →  blend suite  →  combined return
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from commodities.data import fetch_prices, fetch_term_structure, fetch_tickers
from commodities.backtest import (
    specific_contract_returns, vol_scale, rebalance, _build_signals, PARAMS,
)
from commodities.blend import IS_END, OOS_START, BLEND_PARAMS
from equities.blend import (
    equal_weight, inv_vol_blend, erc_blend, hrp_blend,
    sharpe_heuristic, msrr_ridge, msrr_lasso, RollingMSRR,
    inv_msfe_blend, pcr_blend, pls_blend,
)

IS_END_TS    = pd.Timestamp(IS_END)
OOS_START_TS = pd.Timestamp(OOS_START)

MIN_IS_OBS  = 252   # drop instruments with fewer IS observations
PCR_COMPS   = 5     # PCA components (more sensible for ~35 instruments)
PLS_COMPS   = 5


# ── helpers ───────────────────────────────────────────────────────────────────

def _metrics(r: pd.Series, vol_target: float = 0.10) -> dict:
    v = r.dropna()
    if len(v) < 30:
        return dict(sr=float("nan"), mdd=float("nan"))
    cum = (1 + v * vol_target).cumprod()
    mdd = float((cum / cum.cummax() - 1).min() * 100)
    return dict(
        sr  = float(np.sqrt(252) * v.mean() / v.std()),
        mdd = mdd,
    )


def _apply_static(bw, F_norm: pd.DataFrame, F_raw: pd.DataFrame) -> pd.Series:
    w    = bw.weights.reindex(F_norm.columns).fillna(0.0)
    norm = w.abs().sum()
    if norm < 1e-12:
        return pd.Series(np.nan, index=F_raw.index)
    # fillna(0): NaN on a roll-buffer day = no position = zero PnL
    return (F_raw.fillna(0.0) * (w / norm)).sum(axis=1)


def _apply_rolling(w_df: pd.DataFrame, F_raw: pd.DataFrame) -> pd.Series:
    idx = F_raw.index.intersection(w_df.dropna(how="any").index)
    if len(idx) == 0:
        return pd.Series(np.nan, index=F_raw.index)
    w_  = w_df.loc[idx]
    norm = w_.abs().sum(axis=1).replace(0.0, np.nan)
    return (F_raw.loc[idx].fillna(0.0) * w_.div(norm, axis=0)).sum(axis=1)


def run_blend_suite(signal_name: str, F_per_instr: pd.DataFrame) -> None:
    """Apply every static and rolling blend method and print IS / OOS table."""

    # Drop instruments with insufficient IS history
    F_is_raw  = F_per_instr.loc[F_per_instr.index <= IS_END_TS]
    F_oos_raw = F_per_instr.loc[F_per_instr.index >= OOS_START_TS]

    is_counts = F_is_raw.count()
    keep = is_counts[is_counts >= MIN_IS_OBS].index
    F_is_raw  = F_is_raw[keep]
    F_oos_raw = F_oos_raw[keep]

    # Drop rows where ALL instruments are NaN
    F_is_raw  = F_is_raw.dropna(how="all").fillna(0.0)
    F_oos_raw = F_oos_raw.dropna(how="all").fillna(0.0)

    N = len(keep)
    if len(F_is_raw) < 63 or N < 2:
        print(f"\n  [{signal_name}] insufficient data (IS={len(F_is_raw)}, N={N}) — skipped")
        return

    # Vol-normalise on IS σ (each instrument to unit annual vol before blending)
    sigma_is = F_is_raw.std() * np.sqrt(252)
    sigma_is = sigma_is.replace(0.0, np.nan)
    F_is  = F_is_raw.div(sigma_is)
    F_oos = F_oos_raw.div(sigma_is)

    # For methods that need a clean NaN-free matrix: drop rows with any NaN
    F_is_clean  = F_is.dropna(how="any")
    F_full_clean = pd.concat([F_is_clean, F_oos.dropna(how="any")]).sort_index()

    print(f"\n  IS : {F_is_raw.index[0].date()} → {F_is_raw.index[-1].date()}  "
          f"({len(F_is_raw)/252:.1f}yr,  {N} instruments,  "
          f"{len(F_is_clean)} clean rows / {len(F_is_raw)} total)")
    print(f"  OOS: {F_oos_raw.index[0].date()} → {F_oos_raw.index[-1].date()}  "
          f"({len(F_oos_raw)/252:.1f}yr)")

    results: dict[str, tuple[pd.Series, pd.Series]] = {}

    # ── static methods ────────────────────────────────────────────────────────
    # equal_weight / inv_vol / erc / hrp use full F_is (std() ignores NaN)
    # Ridge / Lasso / PCR / PLS need clean rows → F_is_clean
    static_methods = [
        ("Equal weight",     lambda: equal_weight(list(F_is.columns)),       F_is),
        ("Inv-vol",          lambda: inv_vol_blend(F_is),                    F_is),
        ("ERC",              lambda: erc_blend(F_is_clean),                  F_is),
        ("HRP",              lambda: hrp_blend(F_is_clean),                  F_is),
        ("Sharpe heuristic", lambda: sharpe_heuristic(F_is_clean),           F_is),
        ("Static Ridge",     lambda: msrr_ridge(F_is_clean, lam=0.01),       F_is),
        ("Static Lasso",     lambda: msrr_lasso(F_is_clean, lam=0.01),       F_is),
        ("Inv-MSFE",         lambda: inv_msfe_blend(F_is_clean),             F_is),
        (f"PCR ({PCR_COMPS})", lambda: pcr_blend(F_is_clean, n_components=PCR_COMPS), F_is),
        (f"PLS ({PLS_COMPS})", lambda: pls_blend(F_is_clean, n_components=PLS_COMPS), F_is),
    ]

    for name, fn, F_fit in static_methods:
        try:
            bw = fn()
            results[name] = (
                _apply_static(bw, F_fit, F_is_raw),
                _apply_static(bw, F_fit, F_oos_raw),
            )
        except Exception as e:
            print(f"    [WARN] {name}: {e}")

    # ── rolling methods ───────────────────────────────────────────────────────
    rolling_methods = [
        ("Rolling OLS",   RollingMSRR(penalty="ols",   lookback=252, min_obs=63)),
        ("Rolling Ridge", RollingMSRR(penalty="ridge",  lookback=252, min_obs=63, lam=0.01)),
        ("Rolling Lasso", RollingMSRR(penalty="lasso",  lookback=252, min_obs=63, alpha=0.01)),
    ]

    for name, model in rolling_methods:
        try:
            w_raw = model._roll(F_full_clean)
            norm  = w_raw.abs().sum(axis=1).replace(0.0, np.nan)
            w_all = w_raw.div(norm, axis=0)
            results[name] = (
                _apply_rolling(w_all.reindex(F_is_raw.index),  F_is_raw),
                _apply_rolling(w_all.reindex(F_oos_raw.index), F_oos_raw),
            )
        except Exception as e:
            print(f"    [WARN] {name}: {e}")

    # ── print table ───────────────────────────────────────────────────────────
    W = 72
    print(f"\n{'='*W}")
    print(f"  {signal_name.upper()} — INSTRUMENT BLEND COMPARISON  ({N} commodities)")
    print(f"  IS ≤ {IS_END_TS.date()}   |   OOS ≥ {OOS_START_TS.date()}")
    print(f"  vol-normalised per instrument on IS σ before blend")
    print(f"{'='*W}")
    print(f"  {'Method':<20}  {'Period':<6}  {'SR':>7}  {'MDD %':>8}")
    print("  " + "─" * (W - 2))

    for name, (r_is, r_oos) in results.items():
        for period, r in [("IS", r_is), ("OOS", r_oos)]:
            m  = _metrics(r)
            sr  = f"{m['sr']:+.2f}"  if not np.isnan(m["sr"])  else "   n/a"
            mdd = f"{m['mdd']:+.1f}%" if not np.isnan(m["mdd"]) else "    n/a"
            print(f"  {name:<20}  {period:<6}  {sr:>7}  {mdd:>8}")
        print("  " + "─" * (W - 2))
    print(f"{'='*W}")
    return results

    # ── top/bottom instruments by IS Sharpe ───────────────────────────────────
    inst_sr = {}
    for col in F_is_raw.columns:
        v = F_is_raw[col].dropna()
        if len(v) >= 60:
            inst_sr[col] = float(v.mean() / v.std() * np.sqrt(252))
    if inst_sr:
        sr_s = pd.Series(inst_sr).sort_values(ascending=False)
        print(f"\n  IS Sharpe per instrument (top 5 / bottom 5):")
        for label, subset in [("Top 5", sr_s.head(5)), ("Bot 5", sr_s.tail(5))]:
            row = "  " + label + ":  " + "  ".join(f"{c}={v:+.2f}" for c, v in subset.items())
            print(row)
        print()


# ── load data ─────────────────────────────────────────────────────────────────

print("Loading Bloomberg data...")
prices  = fetch_prices()
ts      = fetch_term_structure()
tk      = fetch_tickers()
returns = specific_contract_returns(ts, tk).reindex(columns=prices.columns)
print(f"  {len(prices.columns)} commodities,  {len(returns)} daily rows")

# Use BLEND_PARAMS for vol-scaling (matches the managed-return parquets)
p = BLEND_PARAMS

print("Building raw signals...")
raw_signals = _build_signals(prices, ts, tk, {**PARAMS, **p})
print(f"  Signals: {list(raw_signals.keys())}")

# Map _build_signals keys → display names (same as in blend.py)
_key_map = {
    "tsmom_252":         "tsmom_252",
    "xs_carry_enh_f1f2": "xs_carry_enh",
    "curvature":         "curvature",
}

# ── compute per-instrument managed returns and run blend suite ────────────────

all_results: dict[str, dict] = {}

for raw_key, signal_name in _key_map.items():
    if raw_key not in raw_signals:
        print(f"\n  [{signal_name}] not available — skipped")
        continue

    raw_sig = raw_signals[raw_key]
    sig = raw_sig.reindex(index=returns.index, columns=returns.columns)

    # Vol-scale: w[i,t] = signal[i,t] × target_vol / sigma_hat[i,t]
    w = vol_scale(sig, returns, target_vol=p["asset_vol"], com=p["vol_com"])
    w = rebalance(w, rule=p["rebal"])

    # Per-instrument managed return: w[i,t-1] × r[i,t]  (NOT summed)
    F_per_instr = w.shift(p["lag"]) * returns

    # Drop columns that are entirely NaN
    F_per_instr = F_per_instr.dropna(how="all", axis=1)

    print(f"\n{'─'*72}")
    print(f"  Signal: {signal_name}  ({F_per_instr.shape[1]} instruments before IS filter)")
    all_results[signal_name] = run_blend_suite(signal_name, F_per_instr)

PICKS = {"tsmom_252": "Equal weight", "xs_carry_enh": "Static Ridge", "curvature": "Rolling Ridge"}
print(f"\n{'='*72}\n  SELECTED INSTRUMENT BLEND PER SIGNAL\n{'='*72}")
print(f"  {'Signal':<14} {'Method':<16} {'IS SR':>7} {'OOS SR':>7}")
for sig, method in PICKS.items():
    if sig in all_results and method in all_results[sig]:
        r_is, r_oos = all_results[sig][method]
        print(f"  {sig:<14} {method:<16} {_metrics(r_is)['sr']:>+7.2f} {_metrics(r_oos)['sr']:>+7.2f}")

print("\nDone.")
