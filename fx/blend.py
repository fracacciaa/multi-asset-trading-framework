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
    hrp_blend,
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
) -> pd.DataFrame:
    """
    Build the monthly managed-return matrix F (T × 3) for the three
    production FX signals used in the blend pipeline.

    Signals:
        momentum   Momentum XS f=9m h=1       (monthly weights, shift(1) → rx_m)
        mr_ns_loo  MR NS-LOO monthly VF-TS    (daily positions  → rx_d → monthly sum)
        carry_xs   Carry XS monthly            (monthly weights, shift(1) → rx_m)

    carry_dupuy is excluded: it has 0.82 IS / 0.79 OOS correlation with carry_xs.

    Returns
    -------
    pd.DataFrame, shape (T, 3), monthly frequency (BME index).
    """
    sdir = Path(signals_dir) if signals_dir else _SIGNALS_DIR
    rx_m = pd.read_parquet(sdir / "returns.parquet")["rx_m"]
    rx_d = pd.read_parquet(sdir / _RX_FILE)

    w_mom = pd.read_parquet(sdir / "signal_a_f9_h1.parquet")
    pnl_mom = (w_mom.shift(1) * rx_m.reindex(columns=w_mom.columns)).sum(axis=1)

    pos_mr = pd.read_parquet(sdir / "mr_ns_loo_positions.parquet")
    pnl_mr = (
        (pos_mr * rx_d.reindex(columns=pos_mr.columns)).sum(axis=1)
        .resample("BME").sum()
    )

    w_cxs = pd.read_parquet(sdir / "view_xs_carry_m.parquet")
    pnl_cxs = (w_cxs.shift(1) * rx_m.reindex(columns=w_cxs.columns)).sum(axis=1)

    F = pd.DataFrame({
        "momentum":  pnl_mom,
        "mr_ns_loo": pnl_mr,
        "carry_xs":  pnl_cxs,
    }).dropna()
    F.index.name = "date"
    return F


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
    min_obs: int = 24,
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
    model.fit(F_is)
    w_is  = model._weights_history
    w_oos = model.predict_weights(F_oos)

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
    min_obs: int = 24,
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
        return float(pos.diff().abs().sum(axis=1).mean() * 12)

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


TC_SCENARIOS_FX: dict[str, float] = {
    "best":  0.3,   # major G10 in tight conditions (EURUSD/USDJPY ~0.3 bp)
    "base":  1.0,   # typical institutional across full 20-pair universe
    "worst": 3.0,   # stress / flash crash / thin liquidity
}


def tc_sweep_fx(
    scenarios: Optional[dict] = None,
    signals_dir: str | Path | None = None,
    is_end: str = IS_END,
    oos_start: str = OOS_START,
    lookback: int = 36,
    min_obs: int = 24,
    vol_target: float = 0.10,
    _collect_returns: bool = False,
) -> "pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]":
    """
    TC sensitivity sweep for the FX blend, with a single average spread per scenario.

    Runs two parallel sweeps (same logic as tc_sweep_rates in rates/blend.py):
      Fixed weights  — Rolling Ridge fit ONCE on TC-free signals; only the TC
                       deduction changes across scenarios.  Pure cost impact,
                       Sharpe decreases monotonically with higher TC.
      Adaptive Ridge — Rolling Ridge RE-FIT on TC-adjusted IS per scenario;
                       blend weights shift away from high-TC signals as costs rise.

    TC deducted from each signal's monthly managed return:
      momentum / carry_xs  : |Δmonthly_pos|.sum(pairs) × (h_bp / 10_000)
      mr_ns_loo            : daily |Δpos|.sum(pairs) × (h_bp / 10_000), summed to month
                             (mr_ns_loo rebalances daily, so TC accumulates intra-month)

    Parameters
    ----------
    scenarios   Dict mapping scenario label → avg G10 half_spread_bp (single spread
                applied uniformly across all pairs).
                Default: TC_SCENARIOS_FX (best=0.3 bp, base=1.0 bp, worst=3.0 bp).
    signals_dir Directory containing the parquet files.  Default: fx/signals/.
    is_end      Last IS date.  Default: "2020-12-31".
    oos_start   First OOS date.  Default: "2021-01-01".
    lookback    Rolling Ridge window in months.  Default: 36.
    min_obs     Minimum observations before first weight estimate.  Default: 24.
    vol_target  For MDD computation (scales NAV to target annual vol).

    Returns
    -------
    pd.DataFrame with columns [Method, Scenario, Period, Sharpe, MDD_pct, TO].
    Method is "Fixed weights" or "Adaptive Ridge".
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

    # ── Gross managed returns (TC-free, matches load_monthly_managed_returns) ─
    pnl_mom = (pos_mom.shift(1) * rx_m.reindex(columns=pos_mom.columns)).sum(axis=1)
    pnl_mr  = ((pos_mr * rx_d.reindex(columns=pos_mr.columns)).sum(axis=1)
               .resample("BME").sum())
    pnl_cxs = (pos_cxs.shift(1) * rx_m.reindex(columns=pos_cxs.columns)).sum(axis=1)

    F_gross = pd.DataFrame({
        "momentum":  pnl_mom,
        "mr_ns_loo": pnl_mr,
        "carry_xs":  pnl_cxs,
    }).dropna().sort_index()

    # ── Precompute monthly |Δpos| per signal (factor out of scenario loop) ────
    # momentum / carry_xs: monthly rebalance → one TC event per month
    chg_mom = pos_mom.diff().abs().sum(axis=1)
    chg_cxs = pos_cxs.diff().abs().sum(axis=1)
    # mr_ns_loo: daily rebalance → accumulate daily TC within each month
    chg_mr  = pos_mr.diff().abs().sum(axis=1).resample("BME").sum()

    pos_chg = {
        "momentum":  chg_mom,
        "mr_ns_loo": chg_mr,
        "carry_xs":  chg_cxs,
    }

    # Monthly positions for blended-pair TO computation (mirrors run_fx_prod)
    monthly_pos = {
        "momentum":  pos_mom,
        "mr_ns_loo": pos_mr.resample("BME").last(),
        "carry_xs":  pos_cxs,
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
        return float(pos.diff().abs().sum(axis=1).mean() * 12)

    # ── TC-free baseline: fit Rolling Ridge ONCE for fixed-weight method ──────
    F_free_is  = F_gross.loc[F_gross.index <= is_end]
    F_free_oos = F_gross.loc[F_gross.index >= oos_start]

    sigma_free = F_free_is.std() * np.sqrt(12)
    sigma_free = sigma_free.replace(0.0, np.nan)
    F_free_is_n  = F_free_is.div(sigma_free)
    F_free_oos_n = F_free_oos.div(sigma_free)

    model_free = RollingMSRR(penalty="ridge", lam=0.01, lookback=lookback, min_obs=min_obs)
    model_free.fit(F_free_is_n)
    w_is_fixed  = model_free._weights_history
    w_oos_fixed = model_free.predict_weights(F_free_oos_n)

    # ── Sweep ─────────────────────────────────────────────────────────────────
    records = []
    _returns_dict: dict[tuple, pd.Series] = {}   # populated when _collect_returns=True

    for scenario_name, h_bp in scenarios.items():
        h = h_bp / 10_000  # bp → decimal (same units as rx_m)

        # TC-adjusted managed returns
        series = {}
        for sig in F_gross.columns:
            tc_m = (pos_chg[sig] * h).reindex(F_gross.index).fillna(0.0)
            series[sig] = F_gross[sig] - tc_m

        F_tc = pd.DataFrame(series).dropna().sort_index()
        F_tc_is  = F_tc.loc[F_tc.index <= is_end]
        F_tc_oos = F_tc.loc[F_tc.index >= oos_start]

        if len(F_tc_is) < min_obs or len(F_tc_oos) < 12:
            continue

        # ── Method 1: Fixed weights — TC-free Ridge, only deduct cost ─────────
        F_tc_is_n  = F_tc_is.div(sigma_free)
        F_tc_oos_n = F_tc_oos.div(sigma_free)
        r_is_f  = combined_return_rolling(F_tc_is_n,  w_is_fixed.reindex(F_tc_is_n.index))
        r_oos_f = combined_return_rolling(F_tc_oos_n, w_oos_fixed.reindex(F_tc_oos_n.index))

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

        # ── Method 2: Adaptive Ridge — re-fit per TC scenario ─────────────────
        sigma_tc = F_tc_is.std() * np.sqrt(12)
        sigma_tc = sigma_tc.replace(0.0, np.nan)
        F_tc_is_n2  = F_tc_is.div(sigma_tc)
        F_tc_oos_n2 = F_tc_oos.div(sigma_tc)

        model_tc = RollingMSRR(penalty="ridge", lam=0.01, lookback=lookback, min_obs=min_obs)
        model_tc.fit(F_tc_is_n2)
        w_is_a  = model_tc._weights_history
        w_oos_a = model_tc.predict_weights(F_tc_oos_n2)
        r_is_a  = combined_return_rolling(F_tc_is_n2, w_is_a)
        r_oos_a = combined_return_rolling(F_tc_oos_n2, w_oos_a)

        for period, r, w, t0, t1 in [
            ("IS",  r_is_a,  w_is_a,  F_tc_is_n2.index[0],  F_tc_is_n2.index[-1]),
            ("OOS", r_oos_a, w_oos_a, F_tc_oos_n2.index[0], F_tc_oos_n2.index[-1]),
        ]:
            records.append({
                "Method":   "Adaptive Ridge",
                "Scenario": scenario_name,
                "Period":   period,
                "Sharpe":   _sharpe(r),
                "MDD_pct":  _mdd(r),
                "TO":       _to(w, sigma_tc, t0, t1),
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
    print(f"\n  TC assumptions (avg G10 half-spread, one-way, bp)")
    print("  " + "─" * 42)
    print(f"  {'Scenario':<12}  {'half-spread (bp)':>18}")
    print("  " + "─" * 42)
    for name, h in scenarios.items():
        print(f"  {name:<12}  {h:>18.1f}")
    print("  " + "─" * 42)

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
    min_obs: int = 24,
    vol_target: float = 0.10,
) -> dict:
    """
    Full FX TC sweep: metrics table + monthly return timeseries for all scenarios.

    Identical to tc_sweep_fx() in computation and printed output, but also
    captures the per-scenario combined monthly return series (IS + OOS) for both
    blend methods so they can be plotted or further analysed.

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
    min_obs: int = 24,
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
        return float(pos.diff().abs().sum(axis=1).mean() * 12)

    def _normalise_w(w_raw: pd.DataFrame) -> pd.DataFrame:
        norm = w_raw.abs().sum(axis=1).replace(0.0, np.nan)
        return w_raw.div(norm, axis=0)

    # ── Method 1: Rolling OLS ────────────────────────────────────────────────
    model_ols = RollingMSRR(penalty="ols", lookback=lookback, min_obs=min_obs)
    model_ols.fit(F_is)
    w_ols_is  = model_ols._weights_history
    w_ols_oos = model_ols.predict_weights(F_oos)

    # ── Method 2: Rolling Ridge ──────────────────────────────────────────────
    model_ridge = RollingMSRR(penalty="ridge", lam=ridge_lam, lookback=lookback, min_obs=min_obs)
    model_ridge.fit(F_is)
    w_ridge_is  = model_ridge._weights_history
    w_ridge_oos = model_ridge.predict_weights(F_oos)

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
