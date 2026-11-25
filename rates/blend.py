"""
Rates signal blend layer.

Loads the three production PnL parquets, applies per-signal cross-market blending,
then blends the three signals using rolling Ridge MSRR.

Cross-market blend (within each signal):
    All signals     Inverse-vol weights (IS σ, frozen)

Signal-level blend (across the three signals):
    Rolling Ridge MSRR (walk-forward, lookback=252d)

Pipeline:
    rates/signals/*_pnl.parquet  →  load_managed_returns()  →  F (T × 3)
    F  →  vol_normalise()  →  F_norm
    F_norm  →  RollingMSRR(penalty="ridge")  →  time-varying blend weights
    blend weights  →  combined portfolio return

Signals (production, from rates/signals/):
    momentum        momentum_a_daily_pnl.parquet   daily TSMOM (8 markets)
    carry           carry_ois_par_pnl.parquet       within-curve OIS carry (7 markets)
    mean_reversion  mean_reversion_pnl.parquet      NS classic MR, full swap repricing

Each PnL parquet is date × market (bp, daily).  load_managed_returns aggregates
each signal to a scalar series using the signal-specific cross-market blend method.
"""

from __future__ import annotations

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

IS_END    = "2020-12-31"
OOS_START = "2021-01-01"

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
#   NZ/SE  — small, thin OIS markets; wider than AUS across scenarios
#   CH     — SARON OIS thin and short-history; low yield level keeps best-case tight
#   India/China — EM, limited tenors, onshore access frictions; widest spreads
TC_SCENARIOS: dict[str, dict[str, float]] = {
    "best": {
        "US": 0.25, "EUR": 0.50, "UK": 0.50,
        "Japan": 0.50, "Canada": 0.50, "Australia": 1.00,
        "New Zealand": 1.50, "Sweden": 1.00, "Switzerland": 1.00,
        "India": 2.00, "China": 1.50,
    },
    "base": {
        "US": 0.75, "EUR": 1.00, "UK": 1.50,
        "Japan": 2.00, "Canada": 1.25, "Australia": 2.00,
        "New Zealand": 3.00, "Sweden": 2.50, "Switzerland": 2.00,
        "India": 4.00, "China": 3.00,
    },
    "worst": {
        "US": 2.00, "EUR": 2.50, "UK": 5.00,
        "Japan": 8.00, "Canada": 3.00, "Australia": 5.00,
        "New Zealand": 7.00, "Sweden": 6.00, "Switzerland": 5.00,
        "India": 10.00, "China": 8.00,
    },
}

_TC_DEFAULT_BP = 2.0    # fallback for markets not in TC_SCENARIOS
_MOM_EFF_DUR   = 5.0    # effective duration (yr) for momentum TC: avg of 2Y≈1.9, 5Y≈4.5, 10Y≈8.5

# ── Cross-market blend (within each signal) ───────────────────────────────────
CROSS_MARKET_BLEND: dict[str, str] = {
    "momentum":       "inv_vol",
    "carry":          "inv_vol",
    "mean_reversion": "inv_vol",
}
_CROSS_MARKET_MIN_MKT_DAYS: int   = 756   # min IS days a market needs to be included (≈3yr)


def _inv_vol_weights(
    pnl_df: pd.DataFrame,
    is_end: str,
    min_mkt_days: int = _CROSS_MARKET_MIN_MKT_DAYS,
) -> pd.Series | None:
    """w_i ∝ 1/σ_i (IS σ) over markets with ≥ min_mkt_days IS obs; None if none qualify."""
    pnl = pnl_df.sort_index()
    is_raw = pnl.loc[pnl.index <= is_end]
    is_counts = is_raw.notna().sum()
    eligible  = is_counts[is_counts >= min_mkt_days].index.tolist()
    if not eligible:
        return None
    sigma_is = is_raw[eligible].std() * np.sqrt(252)
    sigma_is = sigma_is.replace(0.0, np.nan).dropna()
    if sigma_is.empty:
        return None
    w = 1.0 / sigma_is
    return w / w.sum()


def _mkt_blend_inv_vol(
    pnl_df: pd.DataFrame,
    is_end: str,
    min_mkt_days: int = _CROSS_MARKET_MIN_MKT_DAYS,
) -> pd.Series:
    """
    Inverse-vol cross-market blend: w_i ∝ 1/σ_i (IS σ), frozen and applied to
    the full period.  More liquid markets tend to have lower vol and therefore
    receive higher weight.  Falls back to equal-weight sum on insufficient data.

    Weights are re-normalised at each date to the subset of markets with data,
    so the blend extends back to the first date any eligible market has a return
    rather than being truncated to the latest-starting market.
    """
    pnl = pnl_df.sort_index()
    w = _inv_vol_weights(pnl, is_end, min_mkt_days)
    if w is None:
        return pnl.sum(axis=1, min_count=1)
    avail        = pnl[w.index].notna()
    w_avail_sum  = avail.multiply(w).sum(axis=1)
    weighted_sum = (pnl[w.index] * w).sum(axis=1, min_count=1)
    return (weighted_sum / w_avail_sum.replace(0.0, np.nan)).rename("blend")


# ── Data loading ─────────────────────────────────────────────────────────────

def load_managed_returns(
    signals: dict[str, str] | None = None,
    signals_dir: str | Path | None = None,
    markets: list[str] | None = None,
    cross_market_blend: dict[str, str] | None = None,
    cross_market_is_end: str = IS_END,
) -> pd.DataFrame:
    """
    Load rates PnL parquets and build the managed-return matrix F (T × K).

    Aggregates each signal's per-market PnL into a single time-series using
    the method specified in `cross_market_blend`:
      "sum"     plain sum across the common market intersection
      "inv_vol" inverse-vol weights (IS σ, frozen) — default for all signals

    Parameters
    ----------
    signals             Map of signal_name → parquet filename.  Default: SIGNAL_FILES.
    signals_dir         Directory containing the parquet files.  Default: rates/signals/.
    markets             Restrict each signal to these markets before blending.
    cross_market_blend  Per-signal aggregation method.  Default: CROSS_MARKET_BLEND.
    cross_market_is_end IS end date for vol-normalisation and inv-vol weight fit.

    Returns
    -------
    pd.DataFrame, shape (T, K), index=date, columns=signal names.
    Only dates where all signals are available are returned (inner join).
    """
    if signals is None:
        signals = SIGNAL_FILES
    if cross_market_blend is None:
        cross_market_blend = CROSS_MARKET_BLEND
    sig_dir = Path(signals_dir) if signals_dir else _SIGNALS_DIR

    frames: dict[str, pd.DataFrame] = {}
    for name, fname in signals.items():
        path = sig_dir / fname
        frames[name] = pd.read_parquet(path)

    # Restrict columns if a market list was given
    if markets is not None:
        frames = {
            n: df[[m for m in markets if m in df.columns]]
            for n, df in frames.items()
        }

    # Common market intersection only used for "sum" signals
    sum_names = [n for n in signals if cross_market_blend.get(n, "sum") == "sum"]
    if sum_names and markets is None:
        common_mkts: list[str] = sorted(
            set.intersection(*(set(frames[n].columns) for n in sum_names))
        )
    elif sum_names:
        common_mkts = list(markets)  # type: ignore[arg-type]
    else:
        common_mkts = []

    series: dict[str, pd.Series] = {}
    for name, df in frames.items():
        method = cross_market_blend.get(name, "inv_vol")
        if method == "inv_vol":
            series[name] = _mkt_blend_inv_vol(df, cross_market_is_end)
        else:
            series[name] = df[common_mkts].sum(axis=1)

    F = pd.DataFrame(series).sort_index().dropna(how="any")
    F.index.name = "date"
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

    Fixed weights: the rolling Ridge is fit ONCE on TC-free signals; only the TC
    deduction changes across scenarios, so Sharpe decreases monotonically with
    higher TC (pure cost impact).

    TC is deducted in native signal units per signal type, AFTER blend weighting
    (cost scales with |w|, so a negative weight never earns a rebate):
      carry          : |Δpos_dv01_m| × h_bp_m          (DV01-normalized, PnL in bp)
      momentum       : |Δpos_m| × h_bp_m × eff_dur      (vol-scaled pos, PnL in bp)
      mean_reversion : |Δ(z×DV01)_m| × h_bp_m            (DV01-scaled pos, PnL in bp)

    Parameters
    ----------
    scenarios   Dict mapping scenario label → {market: half_spread_bp}.
                Default: TC_SCENARIOS (best / base / worst).
    signals_dir Directory containing the parquet files.
    vol_target  For MDD computation (scales NAV to target annual vol).

    Returns
    -------
    pd.DataFrame with columns [Method, Scenario, Period, Sharpe, MDD_pct, TO].
    Method is always "Fixed weights".
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

    def _tc_cost(
        pnl_df: pd.DataFrame,
        pos_df: pd.DataFrame,
        tc_per_market: dict,
        signal: str,
    ) -> pd.DataFrame:
        """Daily TC cost per market (>= 0), same shape as pnl_df."""
        cost_df = pd.DataFrame(0.0, index=pnl_df.index, columns=pnl_df.columns)
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
            # Net position per market (sum across tenors): TC is on the change in
            # total market allocation, not the sum of within-market tenor moves.
            daily_abs_chg = pos_m.sum(axis=1).diff().abs()
            if signal == "momentum":
                tc_cost = daily_abs_chg * h * _MOM_EFF_DUR
            else:
                tc_cost = daily_abs_chg * h
            cost_df[mkt] = tc_cost.reindex(pnl_df.index).fillna(0.0)
        return cost_df

    def _net_return(G_n: pd.DataFrame, C_n: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
        """Σ w_k·gross_k − Σ |w_k|·cost_k with L1-normalised weights (cost never earns a rebate)."""
        idx = G_n.index.intersection(weights.dropna(how="any").index)
        w = weights.loc[idx]
        w = w.div(w.abs().sum(axis=1).replace(0.0, np.nan), axis=0)
        return ((G_n.loc[idx] * w).sum(axis=1)
                - (C_n.loc[idx] * w.abs()).sum(axis=1)).rename("combined")

    # ── Metric helpers ────────────────────────────────────────────────────────

    def _sharpe(r: pd.Series) -> float:
        r = r.dropna()
        if len(r) < 30 or r.std() < 1e-12:
            return float("nan")
        return float(r.mean() / r.std() * np.sqrt(252))

    def _mdd(r: pd.Series) -> float:
        r = r.dropna()
        if len(r) < 2 or r.std() < 1e-12:
            return float("nan")
        ann_vol = r.std() * np.sqrt(252)
        cum = (1 + r * (vol_target / ann_vol)).cumprod()
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

    # ── Pre-compute inv_vol market weights per signal (TC-free IS PnL) ──────────
    _mkt_weights: dict[str, pd.Series] = {}
    for sig, pnl_df in pnl_frames.items():
        is_raw = pnl_df.loc[pnl_df.index <= is_end]
        is_counts = is_raw.notna().sum()
        eligible = is_counts[is_counts >= _CROSS_MARKET_MIN_MKT_DAYS].index.tolist()
        if eligible:
            sigma_m = is_raw[eligible].std() * np.sqrt(252)
            sigma_m = sigma_m.replace(0.0, np.nan).dropna()
            if not sigma_m.empty:
                w = 1.0 / sigma_m
                _mkt_weights[sig] = w / w.sum()

    def _apply_mkt_weights(pnl_df: pd.DataFrame, sig: str) -> pd.Series:
        if sig in _mkt_weights:
            w = _mkt_weights[sig]
            avail = pnl_df[w.index].notna()
            w_avail_sum = avail.multiply(w).sum(axis=1)
            weighted_sum = (pnl_df[w.index] * w).sum(axis=1, min_count=1)
            return weighted_sum / w_avail_sum.replace(0.0, np.nan)
        return pnl_df[common_mkts].sum(axis=1, min_count=1)

    # ── TC-free baseline: fit Rolling Ridge ONCE for fixed-weight method ──────
    F_free = pd.DataFrame(
        {sig: _apply_mkt_weights(pnl_frames[sig], sig) for sig in pnl_frames}
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
        # Blended per-signal cost series; net = gross − cost (used only to FIT weights)
        cost_blend: dict[str, pd.Series] = {}
        for sig, pnl_df in pnl_frames.items():
            if sig in pos_frames:
                cdf = _tc_cost(pnl_df, pos_frames[sig], tc_per_mkt, sig)
            else:
                cdf = pd.DataFrame(0.0, index=pnl_df.index, columns=pnl_df.columns)
            cost_blend[sig] = _apply_mkt_weights(cdf, sig)
        C_tc = pd.DataFrame(cost_blend).reindex(F_free.index).fillna(0.0)
        F_tc = F_free - C_tc
        F_tc_is  = F_tc.loc[F_tc.index <= is_end]
        F_tc_oos = F_tc.loc[F_tc.index >= oos_start]

        if len(F_tc_is) < min_obs or len(F_tc_oos) < 30:
            continue

        # ── Method 1: Fixed weights — TC-free Ridge, only deduct cost ─────────
        # Normalise using the TC-free IS sigma so the fixed weights stay on the
        # same scale they were calibrated on.
        G_is_f, G_oos_f = F_free_is.div(sigma_free), F_free_oos.div(sigma_free)
        C_is_f  = C_tc.loc[F_tc_is.index].div(sigma_free)
        C_oos_f = C_tc.loc[F_tc_oos.index].div(sigma_free)
        w_is_f  = w_fixed.reindex(F_tc_is.index)
        w_oos_f = w_fixed.reindex(F_tc_oos.index)
        r_is_f  = _net_return(G_is_f, C_is_f, w_is_f)
        r_oos_f = _net_return(G_oos_f, C_oos_f, w_oos_f)

        for period, r, w, t0, t1 in [
            ("IS",  r_is_f,  w_is_f,  F_tc_is.index[0],  F_tc_is.index[-1]),
            ("OOS", r_oos_f, w_oos_f, F_tc_oos.index[0], F_tc_oos.index[-1]),
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

    df = pd.DataFrame(records)

    # ── Print results ─────────────────────────────────────────────────────────
    W = 80
    label = "FIXED WEIGHTS — TC-free Ridge fit; only cost deducted"
    print("\n" + "=" * W)
    print(f"  {label}")
    print(f"  IS ≤ {is_end}   |   OOS ≥ {oos_start}   |   vol target={vol_target*100:.0f}%")
    print("=" * W)
    print(f"  {'Scenario':<12}  {'Period':<6}  {'Sharpe':>8}  {'MDD':>8}  {'TO':>8}")
    print("  " + "─" * (W - 2))
    prev_scen = None
    for _, row in df.iterrows():
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


# ── Market-level blend evaluation ─────────────────────────────────────────────

def _try_market_blend(
    pnl_file: str,
    signal_label: str,
    signals_dir: Optional[str | Path] = None,
    is_end: str = IS_END,
    lookback: int = 252,
    min_obs: int = 63,
    vol_target: float = 0.10,
    min_market_obs: int = 252,
) -> dict:
    """
    Evaluate all blend methods applied to per-market PnL for one signal.

    Loads a date × market PnL parquet (in bp/day), vol-normalises on IS
    realized σ, then runs the full static + dynamic blend suite treating
    markets as "signals".  Analogous to try_asset_blend() in core/asset_blend.py
    but at the within-signal, cross-market level.

    Parameters
    ----------
    pnl_file        Parquet filename under signals_dir; shape date × market.
    signal_label    Display label for the header ("MEAN REVERSION", etc.).
    signals_dir     Directory containing the parquet files.  Default: rates/signals/.
    is_end          Last IS date.  Default: IS_END.
    lookback        Rolling-MSRR / DMA window in trading days.  Default: 252.
    min_obs         Minimum observations before rolling weights are non-NaN.  Default: 63.
    vol_target      Annual vol for MDD scaling.  Default: 0.10.
    min_market_obs  Drop markets with fewer than this many non-NaN IS rows.  Default: 252.

    Returns
    -------
    dict  keyed by method name, each value:
        {"is": {"sharpe", "mdd", "turnover"}, "oos": {...}, "type": "static"|"dynamic"}
    """
    from core.asset_blend import _drift_to

    sig_dir = Path(signals_dir) if signals_dir else _SIGNALS_DIR

    pnl = pd.read_parquet(sig_dir / pnl_file)
    pnl.index = pd.to_datetime(pnl.index)
    pnl = pnl.sort_index()

    # Drop markets with insufficient IS history
    is_counts = pnl.loc[pnl.index <= is_end].notna().sum()
    pnl = pnl[is_counts[is_counts >= min_market_obs].index]
    pnl = pnl.dropna(how="any")

    F_is_raw  = pnl.loc[pnl.index <= is_end]
    F_oos_raw = pnl.loc[pnl.index >  is_end]

    if F_is_raw.empty:
        raise ValueError(f"No IS data at or before {is_end} in {pnl_file}.")

    # Vol-normalise on IS realized σ × √252
    sigma = F_is_raw.std() * np.sqrt(252)
    sigma = sigma.replace(0.0, np.nan)
    F_is   = F_is_raw.div(sigma)
    F_oos  = F_oos_raw.div(sigma)
    F_full = pd.concat([F_is, F_oos]).sort_index()
    names  = list(F_is.columns)

    # ── Metric helpers ────────────────────────────────────────────────────────

    def _sharpe(r: pd.Series) -> float:
        r = r.dropna()
        if len(r) < 30 or r.std() < 1e-12:
            return float("nan")
        return float(r.mean() / r.std() * np.sqrt(252))

    def _mdd_m(r: pd.Series) -> float:
        r = r.dropna()
        if r.empty or r.std() < 1e-12:
            return float("nan")
        ann_vol = r.std() * np.sqrt(252)
        cum = (1 + r * (vol_target / ann_vol)).cumprod()
        return float((cum / cum.cummax() - 1).min() * 100)

    def _to_m(w_df: pd.DataFrame, F: pd.DataFrame) -> float:
        # Drift-corrected: returns at the vol target pull weights off target each day.
        return _drift_to(w_df, F * vol_target, freq=252, l1_norm=True)

    def _static_w_df(bw, F: pd.DataFrame) -> pd.DataFrame:
        w = bw.weights.reindex(names).fillna(0.0)
        total = w.abs().sum()
        if total > 1e-12:
            w = w / total
        return pd.DataFrame(np.tile(w.values, (len(F), 1)), index=F.index, columns=names)

    def _apply_static_bw(bw) -> tuple[pd.Series, pd.Series]:
        w = bw.weights.reindex(names).fillna(0.0)
        total = w.abs().sum()
        if total > 1e-12:
            w = w / total
        return (F_is * w).sum(axis=1), (F_oos * w).sum(axis=1)

    def _apply_tvw_df(w_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        def _apply(F):
            idx = F.index.intersection(w_df.dropna(how="any").index)
            if idx.empty:
                return pd.Series(dtype=float)
            w = w_df.loc[idx]
            norm = w.abs().sum(axis=1).replace(0.0, np.nan)
            return (F.loc[idx] * w.div(norm, axis=0)).sum(axis=1)
        return _apply(F_is), _apply(F_oos)

    def _normalise_w(w_raw: pd.DataFrame) -> pd.DataFrame:
        norm = w_raw.abs().sum(axis=1).replace(0.0, np.nan)
        return w_raw.div(norm, axis=0)

    # ── Static methods ────────────────────────────────────────────────────────

    static_blends: dict = {
        "equal_weight": equal_weight(names),
        "inv_vol":      inv_vol_blend(F_is),
        "erc":          erc_blend(F_is),
        "sh_p1":        sharpe_heuristic(F_is, power=1),
        "msrr_ols":     msrr_ridge(F_is, lam=0.0),
        "msrr_ridge":   msrr_ridge(F_is, lam=0.01),
    }

    # ── Dynamic methods ────────────────────────────────────────────────────────

    dma = DMABlend(alpha=0.99)
    dma.fit(F_is)
    w_dma = pd.concat([dma._weights_history, dma.predict_weights(F_oos)]).sort_index()

    dynamic_blends: dict = {
        "rolling_msrr_ols":   _normalise_w(RollingMSRR(penalty="ols",   lookback=lookback, min_obs=min_obs)._roll(F_full)),
        "rolling_msrr_ridge": _normalise_w(RollingMSRR(penalty="ridge", lam=0.01, lookback=lookback, min_obs=min_obs)._roll(F_full)),
        "rolling_msrr_lasso": _normalise_w(RollingMSRR(penalty="lasso", lookback=lookback, min_obs=min_obs)._roll(F_full)),
        "dma":                w_dma,
    }

    # ── Collect results ────────────────────────────────────────────────────────

    result: dict = {}

    for mname, bw in static_blends.items():
        r_is, r_oos = _apply_static_bw(bw)
        result[mname] = {
            "is":   {"sharpe": _sharpe(r_is),  "mdd": _mdd_m(r_is),  "turnover": _to_m(_static_w_df(bw, F_is),  F_is)},
            "oos":  {"sharpe": _sharpe(r_oos), "mdd": _mdd_m(r_oos), "turnover": _to_m(_static_w_df(bw, F_oos), F_oos)},
            "type": "static",
        }

    for mname, w_df in dynamic_blends.items():
        r_is, r_oos = _apply_tvw_df(w_df)
        result[mname] = {
            "is":   {"sharpe": _sharpe(r_is),  "mdd": _mdd_m(r_is),  "turnover": _to_m(w_df.reindex(F_is.index),  F_is)},
            "oos":  {"sharpe": _sharpe(r_oos), "mdd": _mdd_m(r_oos), "turnover": _to_m(w_df.reindex(F_oos.index), F_oos)},
            "type": "dynamic",
        }

    # ── Print table ────────────────────────────────────────────────────────────

    oos_end   = F_oos.index[-1]  if not F_oos.empty  else None
    is_start  = F_is.index[0]
    W = 84

    print()
    print("=" * W)
    print(f"  RATES {signal_label} — MARKET-LEVEL BLEND")
    print(f"  IS : {is_start.strftime('%Y-%m')} → {is_end[:7]}   |   "
          f"OOS: {is_end[:7]} → {oos_end.strftime('%Y-%m') if oos_end else 'n/a'}")
    print(f"  Markets : {names}")
    print(f"  MDD at {vol_target*100:.0f}% vol target  |  rolling lookback = {lookback}d")
    print("=" * W)
    print(
        f"  {'Method':<22}  {'Type':<8}"
        f"  {'IS SR':>7}  {'IS MDD':>8}  {'IS TO':>7}"
        f"  {'OOS SR':>7}  {'OOS MDD':>8}  {'OOS TO':>8}"
    )
    print("  " + "─" * (W - 2))

    prev_type = None
    for mname, stats in result.items():
        t = stats["type"]
        if prev_type is not None and t != prev_type:
            print("  " + "─" * (W - 2))
        prev_type = t
        i, o = stats["is"], stats["oos"]

        def _sr(v):  return f"{v:+.2f}"  if np.isfinite(v) else "   n/a"
        def _mdd(v): return f"{v:+.1f}%" if np.isfinite(v) else "    n/a"
        def _to(v):  return f"{v:.1f}x"  if np.isfinite(v) else "   n/a"

        print(
            f"  {mname:<22}  {t:<8}"
            f"  {_sr(i['sharpe']):>7}  {_mdd(i['mdd']):>8}  {_to(i['turnover']):>7}"
            f"  {_sr(o['sharpe']):>7}  {_mdd(o['mdd']):>8}  {_to(o['turnover']):>8}"
        )

    print("=" * W)

    # Print IS weights for all static methods
    col_w = max(10, max(len(n) for n in names) + 2)
    print(f"\n  IS weights (static methods, L1-normalised)")
    print("  " + "─" * (22 + col_w * len(names)))
    print(f"  {'Method':<20}" + "".join(f"  {n:>{col_w}}" for n in names))
    print("  " + "─" * (22 + col_w * len(names)))
    for mname, bw in static_blends.items():
        w = bw.weights.reindex(names).fillna(0.0)
        total = w.abs().sum()
        if total > 1e-12:
            w = w / total
        print(f"  {mname:<20}" + "".join(f"  {w.get(n, 0.0):>+{col_w}.1%}" for n in names))
    print("  " + "─" * (22 + col_w * len(names)))

    return result


def try_mr_market_blend(
    signals_dir: Optional[str | Path] = None,
    is_end: str = IS_END,
    lookback: int = 252,
    min_obs: int = 63,
    vol_target: float = 0.10,
) -> dict:
    """
    Evaluate all blend methods applied to per-market NS classic MR PnL.

    Treats each market's MR return series as a "signal" and applies the same
    blend suite used for signal-level combination, in contrast to the default
    equal-weight sum in load_managed_returns().

    Uses mean_reversion_pnl.parquet (date × market, bp/day).

    Returns
    -------
    dict keyed by method name with IS/OOS {"sharpe", "mdd", "turnover"}.
    """
    return _try_market_blend(
        pnl_file="mean_reversion_pnl.parquet",
        signal_label="MEAN REVERSION",
        signals_dir=signals_dir,
        is_end=is_end, lookback=lookback,
        min_obs=min_obs, vol_target=vol_target,
    )


def try_carry_market_blend(
    signals_dir: Optional[str | Path] = None,
    is_end: str = IS_END,
    lookback: int = 252,
    min_obs: int = 63,
    vol_target: float = 0.10,
) -> dict:
    """
    Evaluate all blend methods applied to per-market within-curve carry OIS PnL.

    Uses carry_ois_par_pnl.parquet (date × market, bp/day).

    Returns
    -------
    dict keyed by method name with IS/OOS {"sharpe", "mdd", "turnover"}.
    """
    return _try_market_blend(
        pnl_file="carry_ois_par_pnl.parquet",
        signal_label="CARRY",
        signals_dir=signals_dir,
        is_end=is_end, lookback=lookback,
        min_obs=min_obs, vol_target=vol_target,
    )


def try_momentum_market_blend(
    signals_dir: Optional[str | Path] = None,
    is_end: str = IS_END,
    lookback: int = 252,
    min_obs: int = 63,
    vol_target: float = 0.10,
) -> dict:
    """
    Evaluate all blend methods applied to per-market TSMOM PnL.

    Uses momentum_a_daily_pnl.parquet (date × market, bp/day).

    Returns
    -------
    dict keyed by method name with IS/OOS {"sharpe", "mdd", "turnover"}.
    """
    return _try_market_blend(
        pnl_file="momentum_a_daily_pnl.parquet",
        signal_label="MOMENTUM",
        signals_dir=signals_dir,
        is_end=is_end, lookback=lookback,
        min_obs=min_obs, vol_target=vol_target,
    )


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
        if len(r) < 2 or r.std() < 1e-12:
            return float("nan")
        ann_vol = r.std() * np.sqrt(252)
        cum = (1 + r * (vol_target / ann_vol)).cumprod()
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
        if len(r) < 2 or r.std() < 1e-12:
            return float("nan")
        ann_vol = r.std() * np.sqrt(252)
        cum = (1 + r * (vol_target / ann_vol)).cumprod()
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
