"""
Rates signal blend layer.

Loads the three production PnL parquets, aggregates across markets, and
exposes the same blend API as equities/blend.py (all algorithms are imported
directly — no duplication).

Production blend: rolling OLS MSRR (RollingMSRR, penalty="ols", lookback=252d).
  Refit OLS weights at each step on the last 252 trading days.
  Walk-forward on the full timeline (IS + OOS) so the OOS window includes
  the IS tail — no lookahead.

Pipeline position:
    rates/signals/*_pnl.parquet  →  load_managed_returns()  →  F (T × 3)
    F  →  vol_normalise()  →  F_norm
    F_norm  →  RollingMSRR(penalty="ols")  →  time-varying blend weights
    blend weights  →  combined portfolio return

Signals (production, from rates/signals/):
    momentum        momentum_a_daily_pnl.parquet   daily TSMOM (9 markets)
    carry           carry_ois_par_pnl.parquet       within-curve OIS carry (9 markets)
    mean_reversion  mean_reversion_pnl.parquet      factor-neutral NS residuals (11 markets)

Each PnL parquet is date × market (bp, daily); load_managed_returns sums
each signal across the common market intersection to produce a single
time-series per signal (the cross-market portfolio managed return).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Re-export every blend algorithm from the equities layer ──────────────────
from equities.blend import (                                        # noqa: F401
    BlendWeights,
    equal_weight,
    sharpe_heuristic,
    msrr_ridge,
    msrr_lasso,
    RollingMSRR,
    combine_signals,
    build_managed_returns,
    build_state_features,
    enet_blend,
    c_enet_blend,
    e_enet_blend,
    inv_msfe_blend,
    pcr_blend,
    pls_blend,
    winsorize_signals,
    pool_blend,
    fit_static,
    MLMSRR,
    LinearDynamicBlend,
    GBRTBlend,
    RandomForestBlend,
    DMABlend,
    STMOM,
)

# ── Rates-specific constants ─────────────────────────────────────────────────

_SIGNALS_DIR = Path(__file__).parent / "signals"

SIGNAL_FILES: dict[str, str] = {
    "momentum":       "momentum_a_daily_pnl.parquet",
    "carry":          "carry_ois_par_pnl.parquet",
    "mean_reversion": "mean_reversion_pnl.parquet",
}

_POS_FILES: dict[str, str] = {
    "momentum":       "momentum_a_daily.parquet",
    "carry":          "carry_ois_par.parquet",
    "mean_reversion": "mean_reversion.parquet",
}

IS_END    = "2018-12-31"
OOS_START = "2019-01-01"


# ── Data loading ─────────────────────────────────────────────────────────────

def load_managed_returns(
    signals: dict[str, str] | None = None,
    signals_dir: str | Path | None = None,
    markets: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load rates PnL parquets and build the managed-return matrix F (T × K).

    Each signal's cross-market portfolio return at time t is the sum of
    per-market PnL values on that date (after restricting to the common
    market intersection).  This is the rates analog of
    equities.blend.build_managed_returns.

    Parameters
    ----------
    signals     Map of signal_name → parquet filename.  Default: SIGNAL_FILES.
    signals_dir Directory containing the parquet files.  Default: rates/signals/.
    markets     Restrict to these markets before summing.  Default: all common
                markets present in every signal file.

    Returns
    -------
    pd.DataFrame, shape (T, K), index=date, columns=signal names.
    Only dates where all signals are available are returned (inner join).
    """
    if signals is None:
        signals = SIGNAL_FILES
    sig_dir = Path(signals_dir) if signals_dir else _SIGNALS_DIR

    frames: dict[str, pd.DataFrame] = {}
    for name, fname in signals.items():
        path = sig_dir / fname
        frames[name] = pd.read_parquet(path)

    # Common market intersection across all signals
    if markets is None:
        common_mkts = sorted(
            set.intersection(*(set(df.columns) for df in frames.values()))
        )
    else:
        common_mkts = markets

    # Sum each signal across the common markets → scalar per date
    series: dict[str, pd.Series] = {
        name: df[common_mkts].sum(axis=1) for name, df in frames.items()
    }

    F = pd.DataFrame(series)
    F.index.name = "date"
    F = F.sort_index()

    # Inner join: only dates where every signal has data
    F = F.dropna(how="any")
    return F


def combined_return(
    F: pd.DataFrame,
    weights: "BlendWeights",
) -> pd.Series:
    """
    Apply static blend weights to the managed-return matrix.

    For rates, the combined portfolio return at time t is:
        R_t = Σ_k  w_k · F_k[t]

    Returns a pd.Series of daily portfolio returns.
    """
    w = weights.weights
    r = pd.Series(0.0, index=F.index)
    for k in w.index:
        if k in F.columns:
            r = r + w[k] * F[k]
    r.name = "combined"
    return r


def rolling_combined_return(
    F: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.Series:
    """
    Apply time-varying blend weights to the managed-return matrix.

    weights is a T × K DataFrame (output of RollingMSRR._roll).
    Rows with any NaN in weights are excluded.  Weights are L1-normalised
    at each step so the portfolio is always fully invested.

    Returns a pd.Series of daily portfolio returns.
    """
    idx = F.index.intersection(weights.dropna(how="any").index)
    F_, w_ = F.loc[idx], weights.loc[idx]
    norm = w_.abs().sum(axis=1).replace(0.0, np.nan)
    return (F_ * w_.div(norm, axis=0)).sum(axis=1).rename("combined")


def fit_rolling_ols(
    lookback: int = 252,
    min_obs: int = 63,
) -> "RollingMSRR":
    """
    Build the production rolling OLS blend model.

    Refit OLS MSRR weights at each step using the last `lookback` trading days.
    Call model._roll(F_norm) on vol-normalised managed returns to get the
    time-varying weight DataFrame (T × K), then pass to rolling_combined_return().
    """
    return RollingMSRR(penalty="ols", lookback=lookback, min_obs=min_obs)


def vol_normalise(
    F_train: pd.DataFrame,
    F_apply: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scale each signal column to unit annualised vol using training-window σ only.

    Prevents the blend estimator from being dominated by the highest-variance
    signal (momentum operates on a ~1000× larger PnL scale than MR).
    Returns (F_train_normalised, F_apply_normalised).
    """
    sigma = F_train.std() * np.sqrt(252)
    sigma = sigma.replace(0.0, np.nan)
    return F_train.div(sigma), F_apply.div(sigma)


def run_blend(
    signals_dir: Optional[str | Path] = None,
    lookback: int = 252,
    min_obs: int = 63,
) -> dict:
    """
    Canonical rates blend pipeline (rolling OLS MSRR).

    Loads managed returns, vol-normalises using IS σ, runs rolling OLS on the
    full timeline (IS + OOS) so the OOS window always has access to the IS tail.

    Returns
    -------
    dict with keys:
        weights     pd.DataFrame of time-varying blend weights (T_full × K)
        F_is        vol-normalised IS managed returns (T_is × K)
        F_oos       vol-normalised OOS managed returns (T_oos × K)
        r_is        combined IS return series
        r_oos       combined OOS return series
    """
    F = load_managed_returns(signals_dir=signals_dir)
    F_is_raw  = F.loc[F.index <= IS_END]
    F_oos_raw = F.loc[F.index >= OOS_START]

    F_is, F_oos = vol_normalise(F_is_raw, F_oos_raw)
    F_full = pd.concat([F_is, F_oos]).sort_index()

    model   = fit_rolling_ols(lookback=lookback, min_obs=min_obs)
    weights_raw = model._roll(F_full)
    # L1-normalise rows so weights reflect actual portfolio allocations
    _norm   = weights_raw.abs().sum(axis=1).replace(0.0, np.nan)
    weights = weights_raw.div(_norm, axis=0)

    w_is  = weights.reindex(F_is.index)
    w_oos = weights.reindex(F_oos.index)

    return {
        "weights": weights,
        "F_is":    F_is,
        "F_oos":   F_oos,
        "r_is":    rolling_combined_return(F_is, w_is),
        "r_oos":   rolling_combined_return(F_oos, w_oos),
    }


def run_rates_blend(
    markets: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    lookback: int = 252,
    min_obs: int = 63,
    rebuild_signals: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Full rates pipeline: build signals → rolling OLS blend.

    Step 1 — Signal pipeline (rates/production/run.py):
        Fetches OIS curves, computes momentum, carry and mean_reversion PnL
        parquets and writes them to rates/signals/.  Skipped when
        rebuild_signals=False and all three PnL parquets already exist.

    Step 2 — Blend (rates/blend.run_blend):
        Loads the three PnL parquets, vol-normalises to unit annual vol using
        IS σ, runs rolling OLS MSRR on the full timeline.

    Parameters
    ----------
    markets         OIS markets to include (default: all 11).
    start / end     Date range for the signal pipeline (default: full history).
    lookback        Rolling window in trading days (default 252 = 1 year).
    min_obs         Minimum observations before first weight estimate (default 63).
    rebuild_signals Force re-run of the signal pipeline even if parquets exist.
    verbose         Print progress from the signal pipeline.

    Returns
    -------
    dict with keys:
        weights     pd.DataFrame of time-varying blend weights (T_full × K)
        F_is        vol-normalised IS managed returns (T_is × 3)
        F_oos       vol-normalised OOS managed returns (T_oos × 3)
        r_is        combined IS return series
        r_oos       combined OOS return series
    """
    from rates.production.run import run as _run_signals, OUTPUT_DIR

    pnl_paths = [OUTPUT_DIR / f for f in SIGNAL_FILES.values()]
    needs_build = rebuild_signals or not all(p.exists() for p in pnl_paths)

    if needs_build:
        if verbose:
            missing = [p.name for p in pnl_paths if not p.exists()]
            if missing:
                print(f"Missing parquets: {missing} — running signal pipeline...")
            else:
                print("rebuild_signals=True — re-running signal pipeline...")
        _run_signals(markets=markets, start=start, end=end, verbose=verbose)
    elif verbose:
        print("Signal parquets found — skipping rebuild. Pass rebuild_signals=True to refresh.")

    return run_blend(lookback=lookback, min_obs=min_obs)


def run_rates_prod(
    signals_dir: Optional[str | Path] = None,
    lookback: int = 252,
    min_obs: int = 63,
    print_results: bool = True,
    vol_target: float = 0.10,
) -> pd.Series:
    """
    Production rates pipeline: blend signals, report metrics, return returns.

    Runs the rolling OLS MSRR blend on all three production signals (momentum,
    carry, mean_reversion), prints an IS / OOS summary table (Sharpe, MDD,
    position-level turnover), and returns the full combined daily return series.

    Parameters
    ----------
    signals_dir     Override the default rates/signals/ directory.
    lookback        Rolling OLS window in trading days (default 252).
    min_obs         Minimum observations before first blend weight (default 63).
    print_results   Print the summary table (default True).

    Returns
    -------
    pd.Series of daily combined portfolio returns (IS + OOS concatenated).
    """
    result = run_blend(signals_dir=signals_dir, lookback=lookback, min_obs=min_obs)

    r_is   = result["r_is"]
    r_oos  = result["r_oos"]
    w_full = result["weights"]
    F_is   = result["F_is"]
    F_oos  = result["F_oos"]

    r_full = pd.concat([r_is, r_oos]).sort_index()
    r_full.name = "rates"

    if not print_results:
        return r_full

    sig_dir = Path(signals_dir) if signals_dir else _SIGNALS_DIR

    # ── metric helpers ────────────────────────────────────────────────────────

    def _sharpe(r: pd.Series) -> float:
        r = r.dropna()
        if len(r) < 30 or r.std() < 1e-12:
            return float("nan")
        return float(r.mean() / r.std() * np.sqrt(252))

    def _mdd(r: pd.Series) -> float:
        r = r.dropna()
        if len(r) < 2:
            return float("nan")
        r_scaled = r * vol_target
        cum = (1 + r_scaled).cumprod()
        return float((cum / cum.cummax() - 1).min() * 100)

    def _sig_to(pos_df: pd.DataFrame, start, end) -> float:
        p = pos_df.loc[start:end].dropna(how="all")
        if len(p) < 2:
            return float("nan")
        gross = p.abs().sum(axis=1).replace(0.0, np.nan).mean()
        if gross < 1e-12 or np.isnan(gross):
            return float("nan")
        return float(p.diff().dropna(how="all").abs().sum(axis=1).mean() * 252 / gross)

    # Load position parquets once
    sig_pos: dict[str, pd.DataFrame] = {}
    for name, fname in _POS_FILES.items():
        path = sig_dir / fname
        if path.exists():
            sig_pos[name] = pd.read_parquet(path)

    def _blend_to(w_df: pd.DataFrame, start, end) -> float:
        """Time-averaged position-level TO weighted by rolling blend allocations."""
        idx = w_df.dropna(how="any").index
        idx = idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
        if len(idx) == 0 or not sig_pos:
            return float("nan")
        w_period = w_df.loc[idx]
        norms    = w_period.abs().sum(axis=1).replace(0.0, np.nan)
        w_norm   = w_period.div(norms, axis=0)
        sig_tos  = {k: _sig_to(sig_pos[k], start, end) for k in sig_pos}
        weighted = w_norm.abs().mul(
            pd.Series(sig_tos).reindex(w_norm.columns).fillna(0.0)
        )
        return float(weighted.sum(axis=1).mean())

    # ── compute metrics ───────────────────────────────────────────────────────

    w_is  = w_full.reindex(F_is.index)
    w_oos = w_full.reindex(F_oos.index)

    periods = {
        ("IS",  f"IS  ({F_is.index[0].date()} → {F_is.index[-1].date()})"): (
            r_is, w_is, F_is.index[0], F_is.index[-1]
        ),
        ("OOS", f"OOS ({F_oos.index[0].date()} → {F_oos.index[-1].date()})"): (
            r_oos, w_oos, F_oos.index[0], F_oos.index[-1]
        ),
    }

    rows = {}
    for (short, long_label), (r, w, t0, t1) in periods.items():
        rows[short] = {
            "label":    long_label,
            "sharpe":   _sharpe(r),
            "mdd":      _mdd(r),
            "turnover": _blend_to(w, t0, t1),
            "w_avg":    w.dropna().mean(),
        }

    # ── print table ───────────────────────────────────────────────────────────

    W = 68
    print("=" * W)
    print(f"  RATES PRODUCTION  (rolling OLS MSRR, lookback=252d, vol target={vol_target*100:.0f}%)")
    print("=" * W)
    print(f"  {'Period':<42}  {'Sharpe':>7}  {'MDD':>8}  {'TO':>7}")
    print("  " + "─" * (W - 2))
    for m in rows.values():
        sr  = f"{m['sharpe']:+.2f}"   if not np.isnan(m["sharpe"])   else "   n/a"
        mdd = f"{m['mdd']:.1f}%"      if not np.isnan(m["mdd"])      else "   n/a"
        to  = f"{m['turnover']:.1f}x" if not np.isnan(m["turnover"]) else "   n/a"
        print(f"  {m['label']:<42}  {sr:>7}  {mdd:>8}  {to:>7}")
    print("=" * W)

    # ── avg blend weights ─────────────────────────────────────────────────────

    names = list(F_is.columns)
    col_w = 16
    print(f"\n  Avg blend weights  (rolling OLS)")
    print("  " + "─" * (22 + col_w * len(names)))
    print(f"  {'Period':<20}" + "".join(f"  {s:>{col_w}}" for s in names))
    print("  " + "─" * (22 + col_w * len(names)))
    for period_short, m in rows.items():
        w_str = "".join(f"  {m['w_avg'].get(s, 0.0):>+{col_w}.1%}" for s in names)
        print(f"  {period_short:<20}{w_str}")
    print("  " + "─" * (22 + col_w * len(names)))

    return r_full
