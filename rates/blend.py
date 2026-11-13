"""
Rates signal blend layer.

Loads the three production PnL parquets, aggregates across markets, and
exposes the same blend API as equities/blend.py (all algorithms are imported
directly — no duplication).

Production blend: rolling Ridge MSRR (RollingMSRR, penalty="ridge", lookback=252d).
  Refit Ridge weights at each step on the last 252 trading days.
  Walk-forward on the full timeline (IS + OOS) so the OOS window includes
  the IS tail — no lookahead.

Pipeline position:
    rates/signals/*_pnl.parquet  →  load_managed_returns()  →  F (T × 3)
    F  →  vol_normalise()  →  F_norm
    F_norm  →  RollingMSRR(penalty="ridge")  →  time-varying blend weights
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
    inv_vol_blend,
    erc_blend,
    hrp_blend,
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
    "mean_reversion": "mean_reversion_dv01_pos.parquet",
}

IS_END    = "2018-12-31"
OOS_START = "2019-01-01"

# ── Market-specific TC scenarios ─────────────────────────────────────────────
# Half-spread estimates in bp of yield, one-way, per OIS market.
# best  = tight normal-market conditions
# base  = typical institutional conditions
# worst = stress / illiquidity tail events
#
# Rationale for key differences:
#   Japan  — BoJ YCC 2022-23 froze market; >50% float absorbed; spreads 5-10×
#   UK     — Sept 2022 Gilt/LDI crisis; bid-ask blew out 10-20× intraday
#   EUR    — broadly liquid but ECB fragmentation episodes widen spreads
#   AUS    — structurally smaller market; RBA YCC 2020-21 distortions
TC_SCENARIOS: dict[str, dict[str, float]] = {
    "best": {
        "US": 0.25, "EUR": 0.50, "UK": 0.50,
        "Japan": 0.50, "Canada": 0.50, "Australia": 1.00,
    },
    "base": {
        "US": 0.75, "EUR": 1.00, "UK": 1.50,
        "Japan": 2.00, "Canada": 1.25, "Australia": 2.00,
    },
    "worst": {
        "US": 2.00, "EUR": 2.50, "UK": 5.00,
        "Japan": 8.00, "Canada": 3.00, "Australia": 5.00,
    },
}

_TC_DEFAULT_BP = 2.0    # fallback for markets not in TC_SCENARIOS
_MOM_EFF_DUR   = 5.0    # effective duration (yr) for momentum TC: avg of 2Y≈1.9, 5Y≈4.5, 10Y≈8.5


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
    """Build the rolling OLS blend model (kept for research use)."""
    return RollingMSRR(penalty="ols", lookback=lookback, min_obs=min_obs)


def fit_rolling_ridge(
    lookback: int = 252,
    min_obs: int = 63,
    lam: float = 0.01,
) -> "RollingMSRR":
    """
    Build the production rolling Ridge blend model.

    Refit Ridge MSRR weights at each step using the last `lookback` trading days.
    Call model._roll(F_norm) on vol-normalised managed returns to get the
    time-varying weight DataFrame (T × K), then pass to rolling_combined_return().
    """
    return RollingMSRR(penalty="ridge", lam=lam, lookback=lookback, min_obs=min_obs)


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

    model   = fit_rolling_ridge(lookback=lookback, min_obs=min_obs)
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


def tc_sweep_rates(
    scenarios: Optional[dict] = None,
    signals_dir: Optional[str | Path] = None,
    is_end: str = IS_END,
    oos_start: str = OOS_START,
    lookback: int = 252,
    min_obs: int = 63,
    vol_target: float = 0.10,
    _collect_returns: bool = False,
) -> "pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]":
    """
    TC sensitivity sweep for the rates blend, with market-specific half-spreads.

    Runs two parallel sweeps and prints them side by side:

      Fixed weights  — Rolling Ridge fit ONCE on TC-free signals; only the TC
                       deduction changes across scenarios.  Sharpe decreases
                       monotonically with higher TC — this is the pure cost impact.

      Adaptive Ridge — Rolling Ridge RE-FIT under each TC scenario; blend weights
                       shift away from high-TC signals as costs rise.  Sharpe can
                       increase with TC because the optimizer selects a cleaner
                       signal mix.  Useful for understanding optimal strategy-mix
                       under each cost regime, not for measuring cost drag.

    TC is deducted in native signal units per signal type:
      carry          : |Δpos_dv01_m| × h_bp_m          (DV01-normalized, PnL in bp)
      momentum       : |Δpos_m| × h_bp_m × eff_dur      (vol-scaled pos, PnL in bp)
      mean_reversion : |Δ(z×DV01)_m| × h_bp_m / 100    (DV01-scaled pos, PnL in year·%)

    Parameters
    ----------
    scenarios   Dict mapping scenario label → {market: half_spread_bp}.
                Default: TC_SCENARIOS (best / base / worst).
    signals_dir Directory containing the parquet files.
    vol_target  For MDD computation (scales NAV to target annual vol).

    Returns
    -------
    pd.DataFrame with columns [Method, Scenario, Period, Sharpe, MDD_pct, TO].
    Method is "Fixed weights" or "Adaptive Ridge".
    When _collect_returns=True (used internally by run_rates_tc), returns a
    tuple (df, returns_df) where returns_df is a DataFrame with MultiIndex
    columns (Method, Scenario) containing the IS+OOS combined daily return series.
    """
    if scenarios is None:
        scenarios = TC_SCENARIOS

    sig_dir = Path(signals_dir) if signals_dir else _SIGNALS_DIR

    # ── Load PnL parquets (date × market) ────────────────────────────────────
    pnl_frames: dict[str, pd.DataFrame] = {}
    for name, fname in SIGNAL_FILES.items():
        pnl_frames[name] = pd.read_parquet(sig_dir / fname)

    # ── Load position parquets ────────────────────────────────────────────────
    pos_frames: dict[str, pd.DataFrame] = {}
    for name, fname in _POS_FILES.items():
        p = sig_dir / fname
        if p.exists():
            pos_frames[name] = pd.read_parquet(p)

    # Common market intersection (same logic as load_managed_returns)
    common_mkts = sorted(
        set.intersection(*(set(df.columns) for df in pnl_frames.values()))
    )

    # ── TC adjustment per signal ──────────────────────────────────────────────

    def _tc_adjust(
        pnl_df: pd.DataFrame,
        pos_df: pd.DataFrame,
        tc_per_market: dict,
        signal: str,
    ) -> pd.DataFrame:
        """Deduct daily TC cost from per-market PnL, in-place copy."""
        pnl_adj = pnl_df.copy()
        is_mi = isinstance(pos_df.columns, pd.MultiIndex)
        pos_markets = set(
            pos_df.columns.get_level_values("market") if is_mi else pos_df.columns
        )
        for mkt in pnl_df.columns:
            if mkt not in pos_markets:
                continue
            h = tc_per_market.get(mkt, _TC_DEFAULT_BP)
            if is_mi:
                pos_m = pos_df.xs(mkt, level="market", axis=1)
            else:
                pos_m = pos_df[[mkt]]
            pos_m = pos_m.reindex(pnl_df.index, method="ffill")
            daily_abs_chg = pos_m.diff().abs().sum(axis=1)
            if signal == "carry":
                tc_cost = daily_abs_chg * h
            elif signal == "momentum":
                tc_cost = daily_abs_chg * h * _MOM_EFF_DUR
            else:   # mean_reversion: w×DV01 positions, PnL in bp — same formula as carry
                tc_cost = daily_abs_chg * h
            pnl_adj[mkt] = pnl_adj[mkt] - tc_cost.reindex(pnl_adj.index).fillna(0.0)
        return pnl_adj

    # ── Metric helpers ────────────────────────────────────────────────────────

    def _sharpe(r: pd.Series) -> float:
        r = r.dropna()
        if len(r) < 30 or r.std() < 1e-12:
            return float("nan")
        return float(r.mean() / r.std() * np.sqrt(252))

    def _mdd(r: pd.Series) -> float:
        r = r.dropna()
        if len(r) < 2:
            return float("nan")
        cum = (1 + r * vol_target).cumprod()
        return float((cum / cum.cummax() - 1).min() * 100)

    def _sig_to_local(pos_df: pd.DataFrame, t0, t1) -> float:
        p = pos_df.loc[t0:t1].dropna(how="all")
        if len(p) < 2:
            return float("nan")
        gross = p.abs().sum(axis=1).replace(0.0, np.nan).mean()
        if gross < 1e-12 or np.isnan(gross):
            return float("nan")
        return float(p.diff().dropna(how="all").abs().sum(axis=1).mean() * 252 / gross)

    def _blend_to(w_df: pd.DataFrame, t0, t1) -> float:
        idx = w_df.dropna(how="any").index
        idx = idx[(idx >= pd.Timestamp(t0)) & (idx <= pd.Timestamp(t1))]
        if len(idx) == 0 or not pos_frames:
            return float("nan")
        w = w_df.loc[idx]
        norms = w.abs().sum(axis=1).replace(0.0, np.nan)
        w = w.div(norms, axis=0)
        total_delta = np.zeros(len(idx))
        total_gross = np.zeros(len(idx))
        for k in w.columns:
            if k not in pos_frames or k not in sigma_free.index or np.isnan(sigma_free[k]):
                continue
            pk = pos_frames[k].reindex(index=idx).fillna(0.0).values
            wk = w[k].values[:, None]
            scaled = pk * wk / float(sigma_free[k])
            total_gross += np.abs(scaled).sum(axis=1)
            total_delta[1:] += np.abs(np.diff(scaled, axis=0)).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            to_arr = np.where(total_gross > 1e-12, total_delta / total_gross, np.nan)
        valid = to_arr[~np.isnan(to_arr)]
        return float(valid.mean() * 252) if len(valid) > 0 else float("nan")

    # ── TC-free baseline: fit Rolling Ridge ONCE for fixed-weight method ──────
    F_free = pd.DataFrame(
        {sig: pnl_frames[sig][common_mkts].sum(axis=1) for sig in pnl_frames}
    ).dropna(how="any").sort_index()
    F_free_is  = F_free.loc[F_free.index <= is_end]
    F_free_oos = F_free.loc[F_free.index >= oos_start]
    F_free_is_n, F_free_oos_n = vol_normalise(F_free_is, F_free_oos)
    # Save IS sigma for consistent normalisation across all TC scenarios
    sigma_free = F_free_is.std() * np.sqrt(252)
    sigma_free = sigma_free.replace(0.0, np.nan)
    F_free_full = pd.concat([F_free_is_n, F_free_oos_n]).sort_index()
    w_raw_free  = RollingMSRR(penalty="ridge", lookback=lookback, min_obs=min_obs)._roll(F_free_full)
    norm_free   = w_raw_free.abs().sum(axis=1).replace(0.0, np.nan)
    w_fixed     = w_raw_free.div(norm_free, axis=0)   # time-varying but TC-free

    # ── Sweep ─────────────────────────────────────────────────────────────────
    records = []
    _returns_dict: dict[tuple, pd.Series] = {}   # populated when _collect_returns=True

    for scenario_name, tc_per_mkt in scenarios.items():
        # TC-adjusted managed returns (shared by both methods)
        series: dict[str, pd.Series] = {}
        for sig, pnl_df in pnl_frames.items():
            if sig in pos_frames:
                pnl_adj = _tc_adjust(pnl_df, pos_frames[sig], tc_per_mkt, sig)
            else:
                pnl_adj = pnl_df
            series[sig] = pnl_adj[common_mkts].sum(axis=1)

        F_tc     = pd.DataFrame(series).dropna(how="any").sort_index()
        F_tc_is  = F_tc.loc[F_tc.index <= is_end]
        F_tc_oos = F_tc.loc[F_tc.index >= oos_start]

        if len(F_tc_is) < min_obs or len(F_tc_oos) < 30:
            continue

        # ── Method 1: Fixed weights — TC-free Ridge, only deduct cost ─────────
        # Normalise using the TC-free IS sigma so the fixed weights stay on the
        # same scale they were calibrated on.
        F_tc_is_n  = F_tc_is.div(sigma_free)
        F_tc_oos_n = F_tc_oos.div(sigma_free)
        w_is_f  = w_fixed.reindex(F_tc_is_n.index)
        w_oos_f = w_fixed.reindex(F_tc_oos_n.index)
        r_is_f  = rolling_combined_return(F_tc_is_n, w_is_f)
        r_oos_f = rolling_combined_return(F_tc_oos_n, w_oos_f)

        for period, r, w, t0, t1 in [
            ("IS",  r_is_f,  w_is_f,  F_tc_is_n.index[0],  F_tc_is_n.index[-1]),
            ("OOS", r_oos_f, w_oos_f, F_tc_oos_n.index[0], F_tc_oos_n.index[-1]),
        ]:
            records.append({
                "Method":   "Fixed weights",
                "Scenario": scenario_name,
                "Period":   period,
                "Sharpe":   _sharpe(r),
                "MDD_pct":  _mdd(r),
                "TO":       _blend_to(w, t0, t1),
            })

        if _collect_returns:
            _returns_dict[("Fixed weights", scenario_name)] = pd.concat(
                [r_is_f, r_oos_f]
            ).sort_index()

        # ── Method 2: Adaptive Ridge — re-fit under each TC scenario ──────────
        F_is_n, F_oos_n = vol_normalise(F_tc_is, F_tc_oos)
        F_full_n = pd.concat([F_is_n, F_oos_n]).sort_index()

        w_raw = RollingMSRR(penalty="ridge", lookback=lookback, min_obs=min_obs)._roll(F_full_n)
        norm  = w_raw.abs().sum(axis=1).replace(0.0, np.nan)
        w_all = w_raw.div(norm, axis=0)
        w_is_a  = w_all.reindex(F_is_n.index)
        w_oos_a = w_all.reindex(F_oos_n.index)
        r_is_a  = rolling_combined_return(F_is_n, w_is_a)
        r_oos_a = rolling_combined_return(F_oos_n, w_oos_a)

        for period, r, w, t0, t1 in [
            ("IS",  r_is_a,  w_is_a,  F_is_n.index[0],  F_is_n.index[-1]),
            ("OOS", r_oos_a, w_oos_a, F_oos_n.index[0], F_oos_n.index[-1]),
        ]:
            records.append({
                "Method":   "Adaptive Ridge",
                "Scenario": scenario_name,
                "Period":   period,
                "Sharpe":   _sharpe(r),
                "MDD_pct":  _mdd(r),
                "TO":       _blend_to(w, t0, t1),
            })

        if _collect_returns:
            _returns_dict[("Adaptive Ridge", scenario_name)] = pd.concat(
                [r_is_a, r_oos_a]
            ).sort_index()

    df = pd.DataFrame(records)

    # ── Print results ─────────────────────────────────────────────────────────
    W = 80
    for method in ["Fixed weights", "Adaptive Ridge"]:
        sub = df[df["Method"] == method]
        label = (
            "FIXED WEIGHTS — TC-free Ridge fit; only cost deducted"
            if method == "Fixed weights"
            else "ADAPTIVE RIDGE — Ridge re-fit per TC scenario"
        )
        print("\n" + "=" * W)
        print(f"  {label}")
        print(f"  IS ≤ {is_end}   |   OOS ≥ {oos_start}   |   vol target={vol_target*100:.0f}%")
        print("=" * W)
        print(f"  {'Scenario':<12}  {'Period':<6}  {'Sharpe':>8}  {'MDD':>8}  {'TO':>8}")
        print("  " + "─" * (W - 2))
        prev_scen = None
        for _, row in sub.iterrows():
            if prev_scen is not None and row["Scenario"] != prev_scen:
                print("  " + "─" * (W - 2))
            prev_scen = row["Scenario"]
            sr  = f"{row['Sharpe']:+.2f}"   if not np.isnan(row["Sharpe"])   else "   n/a"
            mdd = f"{row['MDD_pct']:.1f}%"  if not np.isnan(row["MDD_pct"]) else "    n/a"
            to  = f"{row['TO']:.1f}x"       if not np.isnan(row["TO"])       else "   n/a"
            print(f"  {row['Scenario']:<12}  {row['Period']:<6}  {sr:>8}  {mdd:>8}  {to:>8}")
        print("=" * W)

    # ── TC table ──────────────────────────────────────────────────────────────
    all_mkts   = sorted({m for sc in scenarios.values() for m in sc})
    scen_names = list(scenarios.keys())
    col_w = 10
    print(f"\n  TC assumptions (yield half-spread, one-way, bp)")
    print("  " + "─" * (20 + col_w * len(scen_names)))
    print(f"  {'Market':<18}" + "".join(f"  {s:>{col_w}}" for s in scen_names))
    print("  " + "─" * (20 + col_w * len(scen_names)))
    for mkt in all_mkts:
        vals = "".join(f"  {scenarios[sc].get(mkt, _TC_DEFAULT_BP):>{col_w}.2f}"
                       for sc in scen_names)
        print(f"  {mkt:<18}{vals}")
    print("  " + "─" * (20 + col_w * len(scen_names)))

    if _collect_returns:
        returns_df = pd.DataFrame(_returns_dict)
        returns_df.columns = pd.MultiIndex.from_tuples(
            returns_df.columns, names=["Method", "Scenario"]
        )
        return df, returns_df
    return df


def run_rates_tc(
    scenarios: Optional[dict] = None,
    signals_dir: Optional[str | Path] = None,
    is_end: str = IS_END,
    oos_start: str = OOS_START,
    lookback: int = 252,
    min_obs: int = 63,
    vol_target: float = 0.10,
) -> dict:
    """
    Full rates TC sweep: metrics table + daily return timeseries for all scenarios.

    Identical to tc_sweep_rates() in computation and printed output, but also
    captures the per-scenario combined daily return series (IS + OOS) for both
    blend methods so they can be plotted or further analysed.

    Returns
    -------
    dict with keys:
        'metrics'   pd.DataFrame — [Method, Scenario, Period, Sharpe, MDD_pct, TO]
        'returns'   pd.DataFrame — daily returns, MultiIndex columns (Method, Scenario)
                    Each column is the IS + OOS combined series for that blend config.
    """
    df, returns_df = tc_sweep_rates(
        scenarios=scenarios,
        signals_dir=signals_dir,
        is_end=is_end,
        oos_start=oos_start,
        lookback=lookback,
        min_obs=min_obs,
        vol_target=vol_target,
        _collect_returns=True,
    )
    return {"metrics": df, "returns": returns_df}


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

    # IS sigma for blended-position TO
    _F_raw_to = load_managed_returns(signals_dir=sig_dir)
    sigma_blend = _F_raw_to.loc[_F_raw_to.index <= IS_END].std() * np.sqrt(252)
    sigma_blend = sigma_blend.replace(0.0, np.nan)

    def _blend_to(w_df: pd.DataFrame, start, end) -> float:
        """Blended instrument-level annualised one-way turnover."""
        idx = w_df.dropna(how="any").index
        idx = idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
        if len(idx) == 0 or not sig_pos:
            return float("nan")
        w = w_df.loc[idx]
        norms = w.abs().sum(axis=1).replace(0.0, np.nan)
        w = w.div(norms, axis=0)
        total_delta = np.zeros(len(idx))
        total_gross = np.zeros(len(idx))
        for k in w.columns:
            if k not in sig_pos or k not in sigma_blend.index or np.isnan(sigma_blend[k]):
                continue
            pk = sig_pos[k].reindex(index=idx).fillna(0.0).values
            wk = w[k].values[:, None]
            scaled = pk * wk / float(sigma_blend[k])
            total_gross += np.abs(scaled).sum(axis=1)
            total_delta[1:] += np.abs(np.diff(scaled, axis=0)).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            to_arr = np.where(total_gross > 1e-12, total_delta / total_gross, np.nan)
        valid = to_arr[~np.isnan(to_arr)]
        return float(valid.mean() * 252) if len(valid) > 0 else float("nan")

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
    print(f"  RATES PRODUCTION  (rolling Ridge MSRR, lookback=252d, vol target={vol_target*100:.0f}%)")
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
    print(f"\n  Avg blend weights  (rolling Ridge)")
    print("  " + "─" * (22 + col_w * len(names)))
    print(f"  {'Period':<20}" + "".join(f"  {s:>{col_w}}" for s in names))
    print("  " + "─" * (22 + col_w * len(names)))
    for period_short, m in rows.items():
        w_str = "".join(f"  {m['w_avg'].get(s, 0.0):>+{col_w}.1%}" for s in names)
        print(f"  {period_short:<20}{w_str}")
    print("  " + "─" * (22 + col_w * len(names)))

    return r_full


def compare_blend_methods(
    signals_dir: Optional[str | Path] = None,
    lookback: int = 252,
    min_obs: int = 63,
    ridge_lam: float = 0.01,
    vol_target: float = 0.10,
) -> pd.DataFrame:
    """
    Compare three rates blend methods side-by-side: Sharpe, MDD, TO.

    Methods
    -------
    Rolling OLS MSRR   production baseline, refit every step on last `lookback` days
    Rolling Ridge MSRR same rolling window with L2 penalty (lam=ridge_lam)
    Static Ridge MSRR  IS-fit once with ridge, constant weights applied to full period

    Returns
    -------
    pd.DataFrame with columns [Method, Period, Sharpe, MDD, TO].
    """
    F = load_managed_returns(signals_dir=signals_dir)
    F_is_raw  = F.loc[F.index <= IS_END]
    F_oos_raw = F.loc[F.index >= OOS_START]

    F_is, F_oos = vol_normalise(F_is_raw, F_oos_raw)
    F_full = pd.concat([F_is, F_oos]).sort_index()

    def _sharpe(r: pd.Series) -> float:
        r = r.dropna()
        if len(r) < 30 or r.std() < 1e-12:
            return float("nan")
        return float(r.mean() / r.std() * np.sqrt(252))

    def _mdd(r: pd.Series) -> float:
        r = r.dropna()
        if len(r) < 2:
            return float("nan")
        cum = (1 + r * vol_target).cumprod()
        return float((cum / cum.cummax() - 1).min() * 100)

    def _normalise_w(w_raw: pd.DataFrame) -> pd.DataFrame:
        norm = w_raw.abs().sum(axis=1).replace(0.0, np.nan)
        return w_raw.div(norm, axis=0)

    # Load signal position parquets once for TO computation
    sig_dir = Path(signals_dir) if signals_dir else _SIGNALS_DIR
    sig_pos: dict[str, pd.DataFrame] = {}
    for name, fname in _POS_FILES.items():
        path = sig_dir / fname
        if path.exists():
            sig_pos[name] = pd.read_parquet(path)

    sigma_cmp = F_is_raw.std() * np.sqrt(252)
    sigma_cmp = sigma_cmp.replace(0.0, np.nan)

    def _blend_to(w_df: pd.DataFrame, start, end) -> float:
        idx = w_df.dropna(how="any").index
        idx = idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
        if len(idx) == 0 or not sig_pos:
            return float("nan")
        w = w_df.loc[idx]
        norms = w.abs().sum(axis=1).replace(0.0, np.nan)
        w = w.div(norms, axis=0)
        total_delta = np.zeros(len(idx))
        total_gross = np.zeros(len(idx))
        for k in w.columns:
            if k not in sig_pos or k not in sigma_cmp.index or np.isnan(sigma_cmp[k]):
                continue
            pk = sig_pos[k].reindex(index=idx).fillna(0.0).values
            wk = w[k].values[:, None]
            scaled = pk * wk / float(sigma_cmp[k])
            total_gross += np.abs(scaled).sum(axis=1)
            total_delta[1:] += np.abs(np.diff(scaled, axis=0)).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            to_arr = np.where(total_gross > 1e-12, total_delta / total_gross, np.nan)
        valid = to_arr[~np.isnan(to_arr)]
        return float(valid.mean() * 252) if len(valid) > 0 else float("nan")

    # ── Method 1: Rolling OLS ────────────────────────────────────────────────
    w_ols = _normalise_w(
        RollingMSRR(penalty="ols", lookback=lookback, min_obs=min_obs)._roll(F_full)
    )

    # ── Method 2: Rolling Ridge ──────────────────────────────────────────────
    w_roll_ridge = _normalise_w(
        RollingMSRR(penalty="ridge", lam=ridge_lam, lookback=lookback, min_obs=min_obs)
        ._roll(F_full)
    )

    # ── Method 3: Static Ridge (IS-fit, constant weights) ───────────────────
    static_bw = msrr_ridge(F_is, lam=ridge_lam)
    w_static_s = static_bw.weights / static_bw.weights.abs().sum()
    w_static = pd.DataFrame(
        np.tile(w_static_s.reindex(F_full.columns).values, (len(F_full), 1)),
        index=F_full.index,
        columns=F_full.columns,
    )

    methods = {
        "Rolling OLS":   w_ols,
        "Rolling Ridge": w_roll_ridge,
        "Static Ridge":  w_static,
    }

    # ── Collect metrics ──────────────────────────────────────────────────────
    records = []
    for method_name, w_full in methods.items():
        w_is  = w_full.reindex(F_is.index)
        w_oos = w_full.reindex(F_oos.index)
        r_is  = rolling_combined_return(F_is,  w_is)
        r_oos = rolling_combined_return(F_oos, w_oos)
        for period, r, w, t0, t1 in [
            ("IS",  r_is,  w_is,  F_is.index[0],  F_is.index[-1]),
            ("OOS", r_oos, w_oos, F_oos.index[0], F_oos.index[-1]),
        ]:
            records.append({
                "Method": method_name,
                "Period": period,
                "Sharpe": _sharpe(r),
                "MDD":    _mdd(r),
                "TO":     _blend_to(w, t0, t1),
            })

    df = pd.DataFrame(records)

    # ── Print table ──────────────────────────────────────────────────────────
    W = 72
    print("=" * W)
    print(f"  RATES BLEND COMPARISON  (vol target={vol_target*100:.0f}%, λ_ridge={ridge_lam})")
    print("=" * W)
    print(f"  {'Method':<20}  {'Period':<6}  {'Sharpe':>7}  {'MDD':>8}  {'TO':>7}")
    print("  " + "─" * (W - 2))
    for _, row in df.iterrows():
        sr  = f"{row['Sharpe']:+.2f}" if not np.isnan(row["Sharpe"]) else "   n/a"
        mdd = f"{row['MDD']:.1f}%"    if not np.isnan(row["MDD"])    else "    n/a"
        to  = f"{row['TO']:.1f}x"     if not np.isnan(row["TO"])     else "   n/a"
        print(f"  {row['Method']:<20}  {row['Period']:<6}  {sr:>7}  {mdd:>8}  {to:>7}")
        if row["Period"] == "OOS":
            print("  " + "─" * (W - 2))
    print("=" * W)

    # ── Avg static weights for Static Ridge ──────────────────────────────────
    names  = list(F_is.columns)
    col_w  = 16
    print(f"\n  Avg blend weights  (Static Ridge IS-fit, λ={ridge_lam})")
    print("  " + "─" * (22 + col_w * len(names)))
    print(f"  {'Signal':<20}" + "".join(f"  {s:>{col_w}}" for s in names))
    print("  " + "─" * (22 + col_w * len(names)))
    print(f"  {'IS+OOS':<20}" + "".join(f"  {w_static_s.get(s, 0.0):>+{col_w}.1%}" for s in names))
    print("  " + "─" * (22 + col_w * len(names)))

    return df
