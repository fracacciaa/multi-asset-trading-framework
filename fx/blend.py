"""
FX signal blend layer.

Loads production position parquets, computes managed returns, and exposes
the same blend API as equities/blend.py (all algorithms are re-exported).

Pipeline position:
    fx/signals/*_positions*.parquet  →  load_managed_returns()  →  F (T × 6)
    F  →  fit_static / MLMSRR / LinearDynamicBlend / …  →  blend weights
    blend weights  →  combined portfolio return

Signals (production, from fx/signals/):
    momentum        signal_a_f9_h1.parquet              XS monthly f=9m h=1 (ffilled)
    carry_E_only    signal_d_E_only_weights_daily.parquet  Dupuy MV E-only (daily)
    carry_bnp_base  signal_e_carry_daily.parquet        BNP base XS weights (daily)
    carry_bnp_e3    signal_e_carry_daily.parquet        BNP E3 crash-risk adjusted (daily)
    mr_ns_loo       mr_ns_loo_positions_daily.parquet   NS LOO XS positions (daily)
    mr_fly          mr_fly_positions.parquet             NS fly XS positions (daily)

Managed return convention: F_k[t] = (pos_k[t-1] × rx_d[t]).sum(pairs)
Monthly momentum is forward-filled to business days before applying this rule.
signal_e_carry_daily uses a two-level MultiIndex (base_weights / e3_weights).

Default pair universe: 19-pair intersection that excludes USDTRY (which is absent
from both MR position files). Data starts 2006-08-25 (first date where all six
signals, including the option-data-dependent carry_bnp_e3, are available).
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
    RollingMSRR,
)

# ── FX-specific constants ─────────────────────────────────────────────────────

_SIGNALS_DIR = Path(__file__).parent / "signals"

# Map signal name → parquet filename.
# carry_bnp_base and carry_bnp_e3 share a file; _SIGNAL_LEVEL selects the level.
SIGNAL_FILES: dict[str, str] = {
    "momentum":       "signal_a_f9_h1.parquet",
    "carry_E_only":   "signal_d_E_only_weights_daily.parquet",
    "carry_bnp_base": "signal_e_carry_daily.parquet",
    "carry_bnp_e3":   "signal_e_carry_daily.parquet",
    "mr_ns_loo":      "mr_ns_loo_positions_daily.parquet",
    "mr_fly":         "mr_fly_positions.parquet",
}

# Top-level key to select from a MultiIndex parquet (two-level columns).
_SIGNAL_LEVEL: dict[str, str] = {
    "carry_bnp_base": "base_weights",
    "carry_bnp_e3":   "e3_weights",
}

# Monthly signals that need forward-filling to daily (using business-day ffill).
_MONTHLY_SIGNALS: frozenset[str] = frozenset({"momentum"})

# Returns parquet (daily excess returns, date × pair).
_RX_FILE = "rx_d.parquet"

# ── Cross-pair blend (within each monthly signal) ────────────────────────────
CROSS_PAIR_BLEND: dict[str, str] = {
    "momentum":  "inv_vol",
    "mr_ns_loo": "inv_vol",
    "carry_xs":  "inv_vol",
}
_CROSS_PAIR_MIN_PAIR_MONTHS = 36    # min IS months a pair needs to be included


def _pair_inv_vol_weights(
    pnl_df: pd.DataFrame,
    is_end: str,
    min_pair_months: int = _CROSS_PAIR_MIN_PAIR_MONTHS,
) -> pd.Series | None:
    """w_i ∝ 1/σ_i (IS σ × √12) over pairs with ≥ min_pair_months IS obs; None if none qualify."""
    pnl = pnl_df.sort_index()
    is_data = pnl.loc[pnl.index <= is_end]
    is_counts = is_data.notna().sum()
    eligible = is_counts[is_counts >= min_pair_months].index.tolist()
    if not eligible:
        return None
    sigma_is = is_data[eligible].std() * np.sqrt(12)
    sigma_is = sigma_is.replace(0.0, np.nan).dropna()
    if sigma_is.empty:
        return None
    w = 1.0 / sigma_is
    return w / w.sum()


def _pair_blend_inv_vol(
    pnl_df: pd.DataFrame,
    is_end: str,
    min_pair_months: int = _CROSS_PAIR_MIN_PAIR_MONTHS,
) -> pd.Series:
    """
    Inverse-vol cross-pair blend: w_i ∝ 1/σ_i (IS σ × √12), frozen and applied
    to the full period.  Falls back to plain sum on insufficient data.
    """
    pnl = pnl_df.sort_index()
    w = _pair_inv_vol_weights(pnl, is_end, min_pair_months)
    if w is None:
        return pnl.sum(axis=1)
    return (pnl[w.index] * w).sum(axis=1, min_count=len(w))


# ── Data loading ─────────────────────────────────────────────────────────────

def load_managed_returns(
    signals_dir: str | Path | None = None,
    pairs: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load FX production signals and build the managed-return matrix F (T × K).

    For each signal k, the managed return at time t is:
        F_k[t] = Σ_{pair}  pos_k[t-1, pair] × rx_d[t, pair]

    Monthly signals (momentum) are forward-filled to business days before
    applying the convention.  MultiIndex signals (carry_bnp) are sliced by
    their sub-key before processing.

    Parameters
    ----------
    signals_dir
        Directory containing the parquet files.  Default: fx/signals/.
    pairs
        Currency pairs to include.  Default: the 19-pair intersection of all
        signal files (excludes USDTRY which is absent from MR position files).

    Returns
    -------
    pd.DataFrame, shape (T, 6), index=date, columns=signal names.
    Only dates where all six signals are non-NaN are returned (inner join).
    Data starts 2006-08-25 (first date carry_bnp_e3 is available).
    """
    sig_dir = Path(signals_dir) if signals_dir else _SIGNALS_DIR

    rx_d = pd.read_parquet(sig_dir / _RX_FILE)

    # Cache MultiIndex parquets that are referenced by more than one signal.
    _file_cache: dict[str, pd.DataFrame] = {}

    def _load_raw(fname: str) -> pd.DataFrame:
        if fname not in _file_cache:
            _file_cache[fname] = pd.read_parquet(sig_dir / fname)
        return _file_cache[fname]

    # Determine pair universe.
    if pairs is None:
        # Build default 19-pair intersection: rx_d ∩ all signal files.
        # MR files have 19 pairs (no USDTRY); that is the binding constraint.
        mr_ns = _load_raw(SIGNAL_FILES["mr_ns_loo"])
        pairs = sorted(set(rx_d.columns) & set(mr_ns.columns))

    rx = rx_d[pairs]

    def _managed_ret(name: str) -> pd.Series:
        raw = _load_raw(SIGNAL_FILES[name])
        # Slice MultiIndex level if needed.
        if name in _SIGNAL_LEVEL:
            raw = raw[_SIGNAL_LEVEL[name]]
        # Restrict to target pairs.
        available = [p for p in pairs if p in raw.columns]
        pos = raw[available].reindex(rx.index)
        # Monthly → forward-fill to daily business days.
        if name in _MONTHLY_SIGNALS:
            pos = pos.ffill()
        # Managed return: pos[t-1] × rx[t]; min_count=1 avoids silently treating
        # NaN positions as zero (important for carry_bnp before option data starts).
        return (pos.shift(1) * rx[available]).sum(axis=1, min_count=1)

    series = {name: _managed_ret(name) for name in SIGNAL_FILES}
    F = pd.DataFrame(series)
    F.index.name = "date"
    F = F.sort_index()
    F = F.dropna(how="any")
    return F


# ── Portfolio-level combination ───────────────────────────────────────────────

def combined_return(
    F: pd.DataFrame,
    weights: "BlendWeights",
) -> pd.Series:
    """
    Apply static blend weights to the managed-return matrix.

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


# ── Monthly 3-signal blend pipeline ──────────────────────────────────────────

IS_END    = "2020-12-31"
OOS_START = "2021-01-01"


def load_monthly_managed_returns(
    signals_dir: str | Path | None = None,
    cross_pair_blend: dict[str, str] | None = None,
    cross_pair_is_end: str = IS_END,
) -> pd.DataFrame:
    """
    Build the monthly managed-return matrix F (T × 3) for the three
    production FX signals used in the blend pipeline.

    Signals:
        momentum   Momentum XS f=9m h=1       (monthly weights, shift(1) → rx_m)
        mr_ns_loo  MR NS-LOO monthly VF-TS    (daily positions  → rx_d → monthly sum)
        carry_xs   Carry XS monthly            (monthly weights, shift(1) → rx_m)

    Within each signal, per-pair PnL series are aggregated using inv_vol weights
    (IS σ × √12, frozen at cross_pair_is_end).

    carry_dupuy is excluded: it has 0.82 IS / 0.79 OOS correlation with carry_xs.

    Returns
    -------
    pd.DataFrame, shape (T, 3), monthly frequency (BME index).
    """
    if cross_pair_blend is None:
        cross_pair_blend = CROSS_PAIR_BLEND
    sdir = Path(signals_dir) if signals_dir else _SIGNALS_DIR
    rx_m = pd.read_parquet(sdir / "returns.parquet")["rx_m"]
    rx_d = pd.read_parquet(sdir / _RX_FILE)

    def _blend(pnl_df: pd.DataFrame, signal: str) -> pd.Series:
        method = cross_pair_blend.get(signal, "inv_vol")
        if method == "inv_vol":
            return _pair_blend_inv_vol(pnl_df, is_end=cross_pair_is_end)
        return pnl_df.sum(axis=1)

    w_mom   = pd.read_parquet(sdir / "signal_a_f9_h1.parquet")
    pnl_mom = _blend(w_mom.shift(1) * rx_m.reindex(columns=w_mom.columns), "momentum")

    pos_mr     = pd.read_parquet(sdir / "mr_ns_loo_positions.parquet")
    pnl_mr_d   = pos_mr * rx_d.reindex(columns=pos_mr.columns)
    pnl_mr     = _blend(pnl_mr_d.resample("BME").sum(), "mr_ns_loo")

    w_cxs   = pd.read_parquet(sdir / "view_xs_carry_m.parquet")
    pnl_cxs = _blend(w_cxs.shift(1) * rx_m.reindex(columns=w_cxs.columns), "carry_xs")

    F = pd.DataFrame({
        "momentum":  pnl_mom,
        "mr_ns_loo": pnl_mr,
        "carry_xs":  pnl_cxs,
    }).dropna()
    F.index.name = "date"
    return F


def roll_weights_full(
    model: "RollingMSRR", F_is: pd.DataFrame, F_oos: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rolling weights over the concatenated IS+OOS history, split back at the IS/OOS
    boundary. model.predict_weights(F_oos) would restart the lookback at the first
    OOS month, leaving the first min_obs OOS months without weights.
    """
    w_all = model._roll(pd.concat([F_is, F_oos]).sort_index())
    return w_all.reindex(F_is.index).dropna(how="all"), w_all.reindex(F_oos.index).dropna(how="all")


def combined_return_rolling(
    F: pd.DataFrame,
    weights_history: pd.DataFrame,
) -> pd.Series:
    """
    Apply time-varying blend weights to the managed-return matrix.

    At each t, weights are normalised to sum to 1 in absolute value.
    Dates where weights are NaN (warm-up period) are dropped.
    """
    idx = F.index.intersection(weights_history.dropna().index)
    w = weights_history.loc[idx]
    row_norm = w.abs().sum(axis=1).replace(0.0, np.nan)
    w = w.div(row_norm, axis=0)
    return (F.loc[idx] * w).sum(axis=1).rename("combined")


def run_fx_blend(
    signals_dir: str | Path | None = None,
    is_end: str = IS_END,
    lookback: int = 36,
    min_obs: int = 36,
) -> dict:
    """
    Canonical FX blend pipeline: monthly managed returns → rolling Ridge MSRR.

    Loads the three production signal managed returns, vol-normalises to unit
    annual vol using IS σ only (no look-ahead), fits a rolling Ridge MSRR on IS
    data (lookback=36m), and predicts weights on OOS.

    Parameters
    ----------
    signals_dir   Directory containing the parquet files.  Default: fx/signals/.
    is_end        Last date of the in-sample period.  Default: "2020-12-31".
    lookback      Rolling window length in months.  Default: 36.
    min_obs       Minimum observations before weights are produced.  Default: 24.

    Returns
    -------
    dict with keys:
        weights_history   DataFrame of time-varying IS blend weights  (T_is × 3)
        weights_oos       DataFrame of time-varying OOS blend weights (T_oos × 3)
        F_is              vol-normalised IS managed returns  (T_is × 3, monthly)
        F_oos             vol-normalised OOS managed returns (T_oos × 3, monthly)
        r_is              combined IS return series  (monthly)
        r_oos             combined OOS return series (monthly)
    """
    F = load_monthly_managed_returns(signals_dir=signals_dir)
    F_is_raw  = F.loc[F.index <= is_end]
    F_oos_raw = F.loc[F.index >  is_end]

    sigma = F_is_raw.std() * np.sqrt(12)
    sigma = sigma.replace(0.0, np.nan)
    F_is  = F_is_raw.div(sigma)
    F_oos = F_oos_raw.div(sigma)

    model = RollingMSRR(penalty="ridge", lam=0.01, lookback=lookback, min_obs=min_obs)
    w_is, w_oos = roll_weights_full(model, F_is, F_oos)

    return {
        "weights_history": w_is,
        "weights_oos":     w_oos,
        "F_is":            F_is,
        "F_oos":           F_oos,
        "r_is":            combined_return_rolling(F_is,  w_is),
        "r_oos":           combined_return_rolling(F_oos, w_oos),
        "sigma":           sigma,
    }


def run_fx_prod(
    signals_dir: str | Path | None = None,
    is_end: str = IS_END,
    lookback: int = 36,
    min_obs: int = 36,
    print_table: bool = True,
    vol_target: float = 0.10,
) -> pd.Series:
    """
    FX production pipeline: blend 3 signals with rolling OLS MSRR, print
    IS / OOS Sharpe, MDD and turnover, and return the full combined return series.

    Parameters
    ----------
    signals_dir   Directory containing the parquet files.  Default: fx/signals/.
    is_end        Last date of the in-sample period.  Default: "2020-12-31".
    lookback      Rolling window length in months.  Default: 36.
    min_obs       Minimum observations before weights are produced.  Default: 24.
    print_table   Whether to print the summary table.  Default: True.

    Returns
    -------
    pd.Series of monthly combined returns (IS + OOS, vol-normalised).
    """
    sdir = Path(signals_dir) if signals_dir else _SIGNALS_DIR

    result  = run_fx_blend(signals_dir=sdir, is_end=is_end,
                           lookback=lookback, min_obs=min_obs)
    r_is    = result["r_is"]
    r_oos   = result["r_oos"]
    w_is    = result["weights_history"]
    w_oos   = result["weights_oos"]
    sigma   = result["sigma"]
    names   = list(result["F_is"].columns)

    # ── monthly positions for position-level turnover ─────────────────────────
    monthly_pos = {
        "momentum":  pd.read_parquet(sdir / "signal_a_f9_h1.parquet"),
        "mr_ns_loo": (pd.read_parquet(sdir / "mr_ns_loo_positions.parquet")
                      .resample("BME").last()),
        "carry_xs":  pd.read_parquet(sdir / "view_xs_carry_m.parquet"),
    }
    common_pairs = sorted(set.intersection(*[set(p.columns) for p in monthly_pos.values()]))

    def _blended_pos(w_df: pd.DataFrame) -> pd.DataFrame:
        idx = w_df.dropna().index
        w   = w_df.loc[idx]
        w   = w.div(w.abs().sum(axis=1).replace(0.0, np.nan), axis=0)
        pos = pd.DataFrame(0.0, index=idx, columns=common_pairs)
        for k in names:
            if k not in monthly_pos or k not in sigma.index:
                continue
            pk = monthly_pos[k].reindex(index=idx, columns=common_pairs).fillna(0.0)
            pos += pk.mul(w[k] / sigma[k], axis=0)
        return pos

    pos_is  = _blended_pos(w_is)
    pos_oos = _blended_pos(w_oos)

    # ── metrics ───────────────────────────────────────────────────────────────
    def _sharpe(r: pd.Series) -> float:
        r = r.dropna()
        if len(r) < 12 or r.std() < 1e-12:
            return float("nan")
        return float(r.mean() / r.std() * np.sqrt(12))

    def _mdd(r: pd.Series) -> float:
        r = r.dropna()
        if r.empty:
            return float("nan")
        r_scaled = r * vol_target
        cum = (1 + r_scaled).cumprod()
        return float((cum / cum.cummax() - 1).min())

    def _turnover(pos: pd.DataFrame) -> float:
        gross = pos.abs().sum(axis=1).replace(0.0, np.nan)
        delta = pos.diff().abs().sum(axis=1)
        return float((delta / gross).dropna().mean() * 12)

    metrics = {
        "IS":  (_sharpe(r_is),  _mdd(r_is),  _turnover(pos_is)),
        "OOS": (_sharpe(r_oos), _mdd(r_oos), _turnover(pos_oos)),
    }

    if print_table:
        W = 72
        is_start  = result["F_is"].index[0].strftime("%Y-%m")
        oos_end   = result["F_oos"].index[-1].strftime("%Y-%m")
        print()
        print("=" * W)
        print("  FX PRODUCTION  —  Rolling Ridge MSRR  (lookback={:d}m, 3 signals)".format(lookback))
        print(f"  IS  : {is_start} → {is_end}   |   OOS : {OOS_START[:7]} → {oos_end}")
        print("=" * W)
        print(f"  {'':20}  {'Sharpe':>8}  {'MDD':>9}  {'Turnover':>10}")
        print("  " + "─" * (W - 2))
        for split, (sr, dd, to) in metrics.items():
            sr_s = f"{sr:+.2f}" if sr == sr else "   n/a"
            dd_s = f"{dd*100:+.1f}%" if dd == dd else "    n/a"
            to_s = f"{to:.1f}x"  if to == to else "   n/a"
            print(f"  {split:<20}  {sr_s:>8}  {dd_s:>9}  {to_s:>10}")
        print("=" * W)
        print(f"  Signals: {names}")
        print(f"  MDD = wealth drawdown at {vol_target*100:.0f}% vol target  |  TO = annual one-way (blended positions)\n")

    return pd.concat([r_is, r_oos]).sort_index().rename("fx_combined")


# FX forward half-spreads, one-way, bps.
# Forward markets are less liquid than exchange-traded rate futures;
# spreads reflect the spot + forward-points bid-ask combined.
# EM pairs (USDTRY, USDZAR, USDMXN, etc.) are substantially wider.
_TC_DEFAULT_FX: float = 5.0   # fallback for pairs not explicitly listed

TC_SCENARIOS_FX: dict[str, dict[str, float]] = {
    "best": {
        # G10 / developed
        "EURUSD": 0.5,  "USDJPY": 0.5,  "GBPUSD": 1.0,  "AUDUSD": 1.0,
        "USDCAD": 1.0,  "USDCHF": 1.0,  "NZDUSD": 1.5,  "USDDKK": 1.0,
        "USDNOK": 2.0,  "USDSEK": 2.0,  "USDSGD": 2.0,
        # EM
        "USDMXN": 3.0,  "USDPLN": 3.0,  "USDCZK": 3.0,  "USDHUF": 3.0,
        "USDKRW": 4.0,  "USDCNY": 3.0,  "USDTHB": 4.0,  "USDTRY": 8.0,
        "USDZAR": 4.0,
    },
    "base": {
        # G10 / developed
        "EURUSD": 1.5,  "USDJPY": 1.5,  "GBPUSD": 2.5,  "AUDUSD": 2.5,
        "USDCAD": 2.5,  "USDCHF": 2.5,  "NZDUSD": 3.5,  "USDDKK": 2.0,
        "USDNOK": 4.0,  "USDSEK": 4.0,  "USDSGD": 5.0,
        # EM
        "USDMXN": 7.0,  "USDPLN": 7.0,  "USDCZK": 7.0,  "USDHUF": 7.0,
        "USDKRW": 10.0, "USDCNY": 7.0,  "USDTHB": 10.0, "USDTRY": 20.0,
        "USDZAR": 8.0,
    },
    "worst": {
        # G10 / developed
        "EURUSD": 5.0,  "USDJPY": 5.0,  "GBPUSD": 8.0,  "AUDUSD": 8.0,
        "USDCAD": 8.0,  "USDCHF": 8.0,  "NZDUSD": 10.0, "USDDKK": 6.0,
        "USDNOK": 12.0, "USDSEK": 12.0, "USDSGD": 12.0,
        # EM
        "USDMXN": 20.0, "USDPLN": 20.0, "USDCZK": 20.0, "USDHUF": 20.0,
        "USDKRW": 25.0, "USDCNY": 20.0, "USDTHB": 25.0, "USDTRY": 50.0,
        "USDZAR": 20.0,
    },
}


def tc_sweep_fx(
    scenarios: Optional[dict] = None,
    signals_dir: str | Path | None = None,
    is_end: str = IS_END,
    oos_start: str = OOS_START,
    lookback: int = 36,
    min_obs: int = 36,
    vol_target: float = 0.10,
    _collect_returns: bool = False,
) -> "pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]":
    """
    TC sensitivity sweep for the FX blend with market-specific forward spreads.

    Fixed weights: the rolling Ridge is fit ONCE on TC-free signals; only the TC
    deduction changes across scenarios (same logic as tc_sweep_rates).  Pure cost
    impact, Sharpe decreases monotonically with higher TC.

    TC is deducted AFTER blend weighting (cost scales with |w|, no rebate from
    negative weights), per signal:
      momentum / carry_xs  : Σ_pair |Δpos_pair| × (h_pair_bp / 10_000)
      mr_ns_loo            : Σ_pair daily |Δpos_pair| × (h_pair_bp / 10_000), summed to month
                             (mr_ns_loo rebalances daily, so TC accumulates intra-month)

    Parameters
    ----------
    scenarios   Dict mapping scenario label → {pair: half_spread_bp}.
                Default: TC_SCENARIOS_FX (pair-specific G10/EM spreads).
                Pairs not in the dict use _TC_DEFAULT_FX (5 bp).
    signals_dir Directory containing the parquet files.  Default: fx/signals/.
    is_end      Last IS date.  Default: "2020-12-31".
    oos_start   First OOS date.  Default: "2021-01-01".
    lookback    Rolling Ridge window in months.  Default: 36.
    min_obs     Minimum observations before first weight estimate.  Default: 24.
    vol_target  For MDD computation (scales NAV to target annual vol).

    Returns
    -------
    pd.DataFrame with columns [Method, Scenario, Period, Sharpe, MDD_pct, TO].
    Method is always "Fixed weights".
    When _collect_returns=True (used internally by run_fx_tc), returns a tuple
    (df, returns_df) where returns_df is a DataFrame with MultiIndex columns
    (Method, Scenario) containing the IS+OOS combined monthly return series.
    """
    if scenarios is None:
        scenarios = TC_SCENARIOS_FX

    sdir = Path(signals_dir) if signals_dir else _SIGNALS_DIR

    # ── Load data ─────────────────────────────────────────────────────────────
    rx_m = pd.read_parquet(sdir / "returns.parquet")["rx_m"]
    rx_d = pd.read_parquet(sdir / _RX_FILE)

    pos_mom = pd.read_parquet(sdir / "signal_a_f9_h1.parquet")
    pos_cxs = pd.read_parquet(sdir / "view_xs_carry_m.parquet")
    pos_mr  = pd.read_parquet(sdir / "mr_ns_loo_positions.parquet")  # daily

    # ── Gross managed returns: identical to the production F (inv-vol pairs) ──
    F_gross = load_monthly_managed_returns(signals_dir=sdir, cross_pair_is_end=is_end)

    # Frozen cross-pair weights per signal (same as load_monthly_managed_returns)
    pair_pnl = {
        "momentum":  pos_mom.shift(1) * rx_m.reindex(columns=pos_mom.columns),
        "mr_ns_loo": (pos_mr * rx_d.reindex(columns=pos_mr.columns)).resample("BME").sum(),
        "carry_xs":  pos_cxs.shift(1) * rx_m.reindex(columns=pos_cxs.columns),
    }
    pair_w: dict[str, pd.Series] = {}
    for sig, df in pair_pnl.items():
        w = (_pair_inv_vol_weights(df, is_end)
             if CROSS_PAIR_BLEND.get(sig, "inv_vol") == "inv_vol" else None)
        pair_w[sig] = w if w is not None else pd.Series(1.0, index=df.columns)

    def _wcols(df: pd.DataFrame, sig: str) -> pd.DataFrame:
        return df.mul(pair_w[sig].reindex(df.columns).fillna(0.0), axis=1)

    # ── Precompute weighted per-pair |Δpos| (factor out of scenario loop) ─────
    # momentum / carry_xs: monthly rebalance → one TC event per month per pair
    dpos_mom = _wcols(pos_mom.diff().abs(), "momentum")
    dpos_cxs = _wcols(pos_cxs.diff().abs(), "carry_xs")
    # mr_ns_loo: daily rebalance → accumulate daily TC within each month, per pair
    dpos_mr_m = _wcols(pos_mr.diff().abs().resample("BME").sum(), "mr_ns_loo")

    # Monthly pair-weighted positions for blended-pair TO computation
    monthly_pos = {
        "momentum":  _wcols(pos_mom, "momentum"),
        "mr_ns_loo": _wcols(pos_mr.resample("BME").last(), "mr_ns_loo"),
        "carry_xs":  _wcols(pos_cxs, "carry_xs"),
    }
    common_pairs = sorted(
        set.intersection(*[set(p.columns) for p in monthly_pos.values()])
    )

    # ── Metric helpers (monthly frequency: annualise with √12) ───────────────

    def _sharpe(r: pd.Series) -> float:
        r = r.dropna()
        if len(r) < 12 or r.std() < 1e-12:
            return float("nan")
        return float(r.mean() / r.std() * np.sqrt(12))

    def _mdd(r: pd.Series) -> float:
        r = r.dropna()
        if r.empty:
            return float("nan")
        cum = (1 + r * vol_target).cumprod()
        return float((cum / cum.cummax() - 1).min() * 100)

    def _to(w_df: pd.DataFrame, sigma: pd.Series, t0, t1) -> float:
        """Blended pair-level annualised one-way turnover, same as run_fx_prod."""
        idx = w_df.dropna().index
        idx = idx[(idx >= pd.Timestamp(t0)) & (idx <= pd.Timestamp(t1))]
        if len(idx) == 0:
            return float("nan")
        w = w_df.loc[idx]
        w = w.div(w.abs().sum(axis=1).replace(0.0, np.nan), axis=0)
        pos = pd.DataFrame(0.0, index=idx, columns=common_pairs)
        for k in w.columns:
            if k not in monthly_pos or k not in sigma.index:
                continue
            pk = monthly_pos[k].reindex(index=idx, columns=common_pairs).fillna(0.0)
            pos += pk.mul(w[k] / sigma[k], axis=0)
        gross = pos.abs().sum(axis=1).replace(0.0, np.nan)
        delta = pos.diff().abs().sum(axis=1)
        return float((delta / gross).dropna().mean() * 12)

    def _net_return(G_n: pd.DataFrame, C_n: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
        """Σ w_k·gross_k − Σ |w_k|·cost_k with L1-normalised weights (cost never earns a rebate)."""
        idx = G_n.index.intersection(weights.dropna().index)
        w = weights.loc[idx]
        w = w.div(w.abs().sum(axis=1).replace(0.0, np.nan), axis=0)
        return ((G_n.loc[idx] * w).sum(axis=1)
                - (C_n.loc[idx] * w.abs()).sum(axis=1)).rename("combined")

    # ── TC-free baseline: fit Rolling Ridge ONCE for fixed-weight method ──────
    F_free_is  = F_gross.loc[F_gross.index <= is_end]
    F_free_oos = F_gross.loc[F_gross.index >= oos_start]

    sigma_free = F_free_is.std() * np.sqrt(12)
    sigma_free = sigma_free.replace(0.0, np.nan)
    F_free_is_n  = F_free_is.div(sigma_free)
    F_free_oos_n = F_free_oos.div(sigma_free)

    model_free = RollingMSRR(penalty="ridge", lam=0.01, lookback=lookback, min_obs=min_obs)
    w_is_fixed, w_oos_fixed = roll_weights_full(model_free, F_free_is_n, F_free_oos_n)

    # ── Sweep ─────────────────────────────────────────────────────────────────
    records = []
    _returns_dict: dict[tuple, pd.Series] = {}   # populated when _collect_returns=True

    for scenario_name, spreads in scenarios.items():
        def _tc_series(dpos_df: pd.DataFrame) -> pd.Series:
            """Monthly TC series: Σ_pair |Δpos_pair| × h_pair (decimal)."""
            h_vec = pd.Series(
                {p: spreads.get(p, _TC_DEFAULT_FX) / 10_000 for p in dpos_df.columns}
            )
            return (dpos_df * h_vec).sum(axis=1)

        # Per-signal cost series; net = gross − cost (used only to FIT weights)
        C_tc = pd.DataFrame({
            "momentum":  _tc_series(dpos_mom).reindex(F_gross.index).fillna(0.0),
            "mr_ns_loo": _tc_series(dpos_mr_m).reindex(F_gross.index).fillna(0.0),
            "carry_xs":  _tc_series(dpos_cxs).reindex(F_gross.index).fillna(0.0),
        })
        F_tc = (F_gross - C_tc).dropna().sort_index()
        F_tc_is  = F_tc.loc[F_tc.index <= is_end]
        F_tc_oos = F_tc.loc[F_tc.index >= oos_start]

        if len(F_tc_is) < min_obs or len(F_tc_oos) < 12:
            continue

        # ── Method 1: Fixed weights — TC-free Ridge, only deduct cost ─────────
        F_tc_is_n  = F_tc_is.div(sigma_free)
        F_tc_oos_n = F_tc_oos.div(sigma_free)
        C_is_f  = C_tc.loc[F_tc_is_n.index].div(sigma_free)
        C_oos_f = C_tc.loc[F_tc_oos_n.index].div(sigma_free)
        r_is_f  = _net_return(F_free_is_n.loc[F_tc_is_n.index], C_is_f,
                              w_is_fixed.reindex(F_tc_is_n.index))
        r_oos_f = _net_return(F_free_oos_n.loc[F_tc_oos_n.index], C_oos_f,
                              w_oos_fixed.reindex(F_tc_oos_n.index))

        for period, r, w, t0, t1 in [
            ("IS",  r_is_f,  w_is_fixed,  F_tc_is_n.index[0],  F_tc_is_n.index[-1]),
            ("OOS", r_oos_f, w_oos_fixed, F_tc_oos_n.index[0], F_tc_oos_n.index[-1]),
        ]:
            records.append({
                "Method":   "Fixed weights",
                "Scenario": scenario_name,
                "Period":   period,
                "Sharpe":   _sharpe(r),
                "MDD_pct":  _mdd(r),
                "TO":       _to(w, sigma_free, t0, t1),
            })

        if _collect_returns:
            _returns_dict[("Fixed weights", scenario_name)] = pd.concat(
                [r_is_f, r_oos_f]
            ).sort_index()

    df = pd.DataFrame(records)

    # ── Print results ─────────────────────────────────────────────────────────
    W = 80
    for method in ["Fixed weights"]:
        sub = df[df["Method"] == method]
        label = "FIXED WEIGHTS — TC-free Ridge fit; only cost deducted"
        print("\n" + "=" * W)
        print(f"  FX  |  {label}")
        print(f"  IS ≤ {is_end}  |  OOS ≥ {oos_start}  |  vol target={vol_target*100:.0f}%  |  monthly")
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
            to  = f"{row['TO']:.2f}x"       if not np.isnan(row["TO"])       else "   n/a"
            print(f"  {row['Scenario']:<12}  {row['Period']:<6}  {sr:>8}  {mdd:>8}  {to:>8}")
        print("=" * W)

    # ── TC table ──────────────────────────────────────────────────────────────
    all_pairs = sorted({p for s in scenarios.values() for p in s})
    scen_names = list(scenarios.keys())
    col_w = 8
    hdr_w = 10
    total_w = hdr_w + col_w * len(scen_names) + 2
    print(f"\n  TC assumptions (FX forward half-spread, one-way, bp)")
    print("  " + "─" * total_w)
    print(f"  {'Pair':<{hdr_w}}" + "".join(f"  {s:>{col_w}}" for s in scen_names))
    print("  " + "─" * total_w)
    for pair in all_pairs:
        row = f"  {pair:<{hdr_w}}"
        for sn in scen_names:
            h = scenarios[sn].get(pair, _TC_DEFAULT_FX)
            row += f"  {h:>{col_w}.1f}"
        print(row)
    print("  " + "─" * total_w)

    if _collect_returns:
        returns_df = pd.DataFrame(_returns_dict)
        returns_df.columns = pd.MultiIndex.from_tuples(
            returns_df.columns, names=["Method", "Scenario"]
        )
        return df, returns_df
    return df


def run_fx_tc(
    scenarios: Optional[dict] = None,
    signals_dir: str | Path | None = None,
    is_end: str = IS_END,
    oos_start: str = OOS_START,
    lookback: int = 36,
    min_obs: int = 36,
    vol_target: float = 0.10,
) -> dict:
    """
    Full FX TC sweep: metrics table + monthly return timeseries for all scenarios.

    Identical to tc_sweep_fx() in computation and printed output, but also
    captures the per-scenario combined monthly return series (IS + OOS) for both
    blend methods so they can be plotted or further analysed.

    TC uses pair-specific FX forward spreads (see TC_SCENARIOS_FX).  Pairs not
    listed in a scenario fall back to _TC_DEFAULT_FX (5 bp).

    Returns
    -------
    dict with keys:
        'metrics'   pd.DataFrame — [Method, Scenario, Period, Sharpe, MDD_pct, TO]
        'returns'   pd.DataFrame — monthly returns, MultiIndex columns (Method, Scenario)
                    Each column is the IS + OOS combined series for that blend config.
    """
    df, returns_df = tc_sweep_fx(
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


def compare_blend_methods(
    signals_dir: str | Path | None = None,
    lookback: int = 36,
    min_obs: int = 36,
    ridge_lam: float = 0.01,
    vol_target: float = 0.10,
) -> pd.DataFrame:
    """
    Compare three FX blend methods side-by-side: Sharpe, MDD, TO.

    Methods
    -------
    Rolling OLS MSRR   production baseline, fit on IS rolling then predict OOS
    Rolling Ridge MSRR same rolling window with L2 penalty (lam=ridge_lam)
    Static Ridge MSRR  IS-fit once with ridge, constant weights applied to both periods

    Monthly frequency; Sharpe annualised with √12.
    MDD at vol_target wealth scaling.
    TO = annualised one-way turnover from blended currency-pair positions.

    Returns
    -------
    pd.DataFrame with columns [Method, Period, Sharpe, MDD, TO].
    """
    sdir = Path(signals_dir) if signals_dir else _SIGNALS_DIR

    F = load_monthly_managed_returns(signals_dir=sdir)
    F_is_raw  = F.loc[F.index <= IS_END]
    F_oos_raw = F.loc[F.index >  IS_END]

    sigma = F_is_raw.std() * np.sqrt(12)
    sigma = sigma.replace(0.0, np.nan)
    F_is  = F_is_raw.div(sigma)
    F_oos = F_oos_raw.div(sigma)
    names = list(F_is.columns)

    # Monthly signal position parquets for TO computation
    monthly_pos = {
        "momentum":  pd.read_parquet(sdir / "signal_a_f9_h1.parquet"),
        "mr_ns_loo": (pd.read_parquet(sdir / "mr_ns_loo_positions.parquet")
                      .resample("BME").last()),
        "carry_xs":  pd.read_parquet(sdir / "view_xs_carry_m.parquet"),
    }
    common_pairs = sorted(set.intersection(*[set(p.columns) for p in monthly_pos.values()]))

    def _blended_pos(w_df: pd.DataFrame) -> pd.DataFrame:
        idx = w_df.dropna().index
        w   = w_df.loc[idx]
        w   = w.div(w.abs().sum(axis=1).replace(0.0, np.nan), axis=0)
        pos = pd.DataFrame(0.0, index=idx, columns=common_pairs)
        for k in names:
            if k not in monthly_pos or k not in sigma.index:
                continue
            pk = monthly_pos[k].reindex(index=idx, columns=common_pairs).fillna(0.0)
            pos += pk.mul(w[k] / sigma[k], axis=0)
        return pos

    def _sharpe(r: pd.Series) -> float:
        r = r.dropna()
        if len(r) < 12 or r.std() < 1e-12:
            return float("nan")
        return float(r.mean() / r.std() * np.sqrt(12))

    def _mdd(r: pd.Series) -> float:
        r = r.dropna()
        if r.empty:
            return float("nan")
        cum = (1 + r * vol_target).cumprod()
        return float((cum / cum.cummax() - 1).min() * 100)

    def _turnover(pos: pd.DataFrame) -> float:
        gross = pos.abs().sum(axis=1).replace(0.0, np.nan)
        delta = pos.diff().abs().sum(axis=1)
        return float((delta / gross).dropna().mean() * 12)

    def _normalise_w(w_raw: pd.DataFrame) -> pd.DataFrame:
        norm = w_raw.abs().sum(axis=1).replace(0.0, np.nan)
        return w_raw.div(norm, axis=0)

    # ── Method 1: Rolling OLS ────────────────────────────────────────────────
    model_ols = RollingMSRR(penalty="ols", lookback=lookback, min_obs=min_obs)
    w_ols_is, w_ols_oos = roll_weights_full(model_ols, F_is, F_oos)

    # ── Method 2: Rolling Ridge ──────────────────────────────────────────────
    model_ridge = RollingMSRR(penalty="ridge", lam=ridge_lam, lookback=lookback, min_obs=min_obs)
    w_ridge_is, w_ridge_oos = roll_weights_full(model_ridge, F_is, F_oos)

    # ── Method 3: Static Ridge (IS-fit, constant weights) ───────────────────
    static_bw  = msrr_ridge(F_is, lam=ridge_lam)
    w_static_s = static_bw.weights / static_bw.weights.abs().sum()
    w_static_is  = pd.DataFrame(
        np.tile(w_static_s.reindex(F_is.columns).values,  (len(F_is),  1)),
        index=F_is.index,  columns=F_is.columns,
    )
    w_static_oos = pd.DataFrame(
        np.tile(w_static_s.reindex(F_oos.columns).values, (len(F_oos), 1)),
        index=F_oos.index, columns=F_oos.columns,
    )

    methods = {
        "Rolling OLS":   (w_ols_is,    w_ols_oos),
        "Rolling Ridge": (w_ridge_is,  w_ridge_oos),
        "Static Ridge":  (w_static_is, w_static_oos),
    }

    # ── Collect metrics ──────────────────────────────────────────────────────
    records = []
    for method_name, (w_is_, w_oos_) in methods.items():
        r_is_  = combined_return_rolling(F_is,  w_is_)
        r_oos_ = combined_return_rolling(F_oos, w_oos_)
        pos_is_  = _blended_pos(w_is_)
        pos_oos_ = _blended_pos(w_oos_)
        for period, r, pos in [
            ("IS",  r_is_,  pos_is_),
            ("OOS", r_oos_, pos_oos_),
        ]:
            records.append({
                "Method": method_name,
                "Period": period,
                "Sharpe": _sharpe(r),
                "MDD":    _mdd(r),
                "TO":     _turnover(pos),
            })

    df = pd.DataFrame(records)

    # ── Print table ──────────────────────────────────────────────────────────
    W = 72
    print("=" * W)
    print(f"  FX BLEND COMPARISON  (vol target={vol_target*100:.0f}%, λ_ridge={ridge_lam})")
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

    # ── Static Ridge IS weights ───────────────────────────────────────────────
    col_w = 16
    print(f"\n  Avg blend weights  (Static Ridge IS-fit, λ={ridge_lam})")
    print("  " + "─" * (22 + col_w * len(names)))
    print(f"  {'Signal':<20}" + "".join(f"  {s:>{col_w}}" for s in names))
    print("  " + "─" * (22 + col_w * len(names)))
    print(f"  {'IS+OOS':<20}" + "".join(f"  {w_static_s.get(s, 0.0):>+{col_w}.1%}" for s in names))
    print("  " + "─" * (22 + col_w * len(names)))

    return df


# ── Per-pair managed-return matrix ────────────────────────────────────────────

def load_pair_managed_returns(
    signal: str,
    signals_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Build the monthly per-pair managed-return matrix for one production signal.

    For each currency pair p the managed return at month t is:
        F[t, p] = Σ_{d in month t}  pos[d-1, p] × rx_d[d, p]   (MR — daily positions)
        F[t, p] = pos[t-1, p] × rx_m[t, p]                      (momentum / carry_xs)

    Parameters
    ----------
    signal      One of "momentum", "mr_ns_loo", "carry_xs".
    signals_dir Directory containing the parquet files.  Default: fx/signals/.

    Returns
    -------
    pd.DataFrame, shape (T, N_pairs), monthly BME index.
    Only pairs present in both the signal file and rx are included.
    Rows with any NaN are dropped (inner join across pairs).
    """
    sdir = Path(signals_dir) if signals_dir else _SIGNALS_DIR
    rx_d = pd.read_parquet(sdir / _RX_FILE)
    rx_m = pd.read_parquet(sdir / "returns.parquet")["rx_m"]

    if signal == "momentum":
        pos = pd.read_parquet(sdir / "signal_a_f9_h1.parquet")
        pairs = sorted(set(pos.columns) & set(rx_m.columns))
        F = (pos[pairs].shift(1) * rx_m.reindex(columns=pairs)).rename_axis("date")

    elif signal == "mr_ns_loo":
        pos = pd.read_parquet(sdir / "mr_ns_loo_positions.parquet")
        pairs = sorted(set(pos.columns) & set(rx_d.columns))
        daily = (pos[pairs] * rx_d.reindex(columns=pairs)).rename_axis("date")
        F = daily.resample("BME").sum()

    elif signal == "carry_xs":
        pos = pd.read_parquet(sdir / "view_xs_carry_m.parquet")
        pairs = sorted(set(pos.columns) & set(rx_m.columns))
        F = (pos[pairs].shift(1) * rx_m.reindex(columns=pairs)).rename_axis("date")

    else:
        raise ValueError(f"Unknown signal '{signal}'. Choose: momentum, mr_ns_loo, carry_xs.")

    return F.dropna(how="any")


# ── Pair-level blend evaluation ────────────────────────────────────────────────

def _try_fx_pair_blend(
    signal: str,
    signal_label: str,
    signals_dir: str | Path | None = None,
    is_end: str = IS_END,
    lookback: int = 36,
    min_obs: int = 36,
    vol_target: float = 0.10,
    min_pair_obs: int = 36,
) -> dict:
    """
    Evaluate all blend methods applied to per-pair managed returns for one signal.

    Loads per-pair monthly managed returns, vol-normalises on IS σ × √12, then
    runs the full static + dynamic blend suite treating currency pairs as "signals".
    Analogous to _try_market_blend() in rates/blend.py but at monthly frequency.

    Parameters
    ----------
    signal          "momentum", "mr_ns_loo", or "carry_xs".
    signal_label    Display label for the header.
    signals_dir     Directory containing the parquet files.  Default: fx/signals/.
    is_end          Last IS date.  Default: IS_END ("2020-12-31").
    lookback        Rolling-MSRR window in months.  Default: 36.
    min_obs         Minimum observations before rolling weights are non-NaN.  Default: 24.
    vol_target      Annual vol for MDD scaling.  Default: 0.10.
    min_pair_obs    Drop pairs with fewer than this many non-NaN IS rows.  Default: 36.

    Returns
    -------
    dict  keyed by method name, each value:
        {"is": {"sharpe", "mdd", "turnover"}, "oos": {...}, "type": "static"|"dynamic"}
    """
    from core.asset_blend import _drift_to

    F_raw = load_pair_managed_returns(signal=signal, signals_dir=signals_dir)

    # Drop pairs with insufficient IS history or near-zero IS variance
    is_data = F_raw.loc[F_raw.index <= is_end]
    is_counts = is_data.notna().sum()
    is_std    = is_data.std()
    keep = is_counts[(is_counts >= min_pair_obs) & (is_std > 1e-8)].index
    F_raw = F_raw[keep].dropna(how="any")

    F_is_raw  = F_raw.loc[F_raw.index <= is_end]
    F_oos_raw = F_raw.loc[F_raw.index >  is_end]

    if F_is_raw.empty:
        raise ValueError(f"No IS data at or before {is_end} for signal '{signal}'.")

    sigma = F_is_raw.std() * np.sqrt(12)
    sigma = sigma.replace(0.0, np.nan)
    F_is   = F_is_raw.div(sigma)
    F_oos  = F_oos_raw.div(sigma)
    F_full = pd.concat([F_is, F_oos]).sort_index()
    pairs  = list(F_is.columns)

    # ── Metric helpers (monthly: annualise with √12) ──────────────────────────

    def _sharpe(r: pd.Series) -> float:
        r = r.dropna()
        if len(r) < 12 or r.std() < 1e-12:
            return float("nan")
        return float(r.mean() / r.std() * np.sqrt(12))

    def _mdd_m(r: pd.Series) -> float:
        r = r.dropna()
        if r.empty:
            return float("nan")
        cum = (1 + r * vol_target).cumprod()
        return float((cum / cum.cummax() - 1).min() * 100)

    def _to_m(w_df: pd.DataFrame, F: pd.DataFrame) -> float:
        # Drift-corrected: returns at the vol target pull weights off target each month.
        return _drift_to(w_df, F * vol_target, freq=12, l1_norm=True)

    def _static_w_df(bw, F: pd.DataFrame) -> pd.DataFrame:
        w = bw.weights.reindex(pairs).fillna(0.0)
        total = w.abs().sum()
        if total > 1e-12:
            w = w / total
        return pd.DataFrame(np.tile(w.values, (len(F), 1)), index=F.index, columns=pairs)

    def _apply_static_bw(bw) -> tuple[pd.Series, pd.Series]:
        w = bw.weights.reindex(pairs).fillna(0.0)
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
        "equal_weight": equal_weight(pairs),
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

    oos_end  = F_oos.index[-1] if not F_oos.empty else None
    is_start = F_is.index[0]
    W = 84

    print()
    print("=" * W)
    print(f"  FX {signal_label} — PAIR-LEVEL BLEND")
    print(f"  IS : {is_start.strftime('%Y-%m')} → {is_end[:7]}   |   "
          f"OOS: {is_end[:7]} → {oos_end.strftime('%Y-%m') if oos_end else 'n/a'}")
    print(f"  Pairs   : {pairs}")
    print(f"  MDD at {vol_target*100:.0f}% vol target  |  rolling lookback = {lookback}m  |  monthly")
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
    col_w = max(8, max(len(p) for p in pairs) + 1)
    print(f"\n  IS weights (static methods, L1-normalised)")
    print("  " + "─" * (22 + col_w * len(pairs)))
    print(f"  {'Method':<20}" + "".join(f"  {p:>{col_w}}" for p in pairs))
    print("  " + "─" * (22 + col_w * len(pairs)))
    for mname, bw in static_blends.items():
        w = bw.weights.reindex(pairs).fillna(0.0)
        total = w.abs().sum()
        if total > 1e-12:
            w = w / total
        print(f"  {mname:<20}" + "".join(f"  {w.get(p, 0.0):>+{col_w}.1%}" for p in pairs))
    print("  " + "─" * (22 + col_w * len(pairs)))

    return result


def try_momentum_pair_blend(
    signals_dir: str | Path | None = None,
    is_end: str = IS_END,
    lookback: int = 36,
    min_obs: int = 36,
    vol_target: float = 0.10,
) -> dict:
    """
    Evaluate all blend methods applied to per-pair FX momentum managed returns.

    Treats each currency pair's monthly momentum return (pos × rx_m) as a
    "signal" and applies the full static + dynamic blend suite to find optimal
    pair weights, in contrast to the default XS cross-sectional weighting.

    Returns
    -------
    dict keyed by method name with IS/OOS {"sharpe", "mdd", "turnover"}.
    """
    return _try_fx_pair_blend(
        signal="momentum", signal_label="MOMENTUM",
        signals_dir=signals_dir, is_end=is_end,
        lookback=lookback, min_obs=min_obs, vol_target=vol_target,
    )


def try_mr_pair_blend(
    signals_dir: str | Path | None = None,
    is_end: str = IS_END,
    lookback: int = 36,
    min_obs: int = 36,
    vol_target: float = 0.10,
) -> dict:
    """
    Evaluate all blend methods applied to per-pair FX mean-reversion managed returns.

    Treats each currency pair's monthly NS-LOO MR return (daily pos × rx_d,
    summed to month) as a "signal" and applies the full blend suite.

    Returns
    -------
    dict keyed by method name with IS/OOS {"sharpe", "mdd", "turnover"}.
    """
    return _try_fx_pair_blend(
        signal="mr_ns_loo", signal_label="MEAN REVERSION",
        signals_dir=signals_dir, is_end=is_end,
        lookback=lookback, min_obs=min_obs, vol_target=vol_target,
    )


def try_carry_pair_blend(
    signals_dir: str | Path | None = None,
    is_end: str = IS_END,
    lookback: int = 36,
    min_obs: int = 36,
    vol_target: float = 0.10,
) -> dict:
    """
    Evaluate all blend methods applied to per-pair FX carry managed returns.

    Treats each currency pair's monthly carry XS return (pos × rx_m) as a
    "signal" and applies the full blend suite.

    Returns
    -------
    dict keyed by method name with IS/OOS {"sharpe", "mdd", "turnover"}.
    """
    return _try_fx_pair_blend(
        signal="carry_xs", signal_label="CARRY",
        signals_dir=signals_dir, is_end=is_end,
        lookback=lookback, min_obs=min_obs, vol_target=vol_target,
    )
