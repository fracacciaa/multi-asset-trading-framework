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
    Canonical FX blend pipeline: monthly managed returns → rolling OLS MSRR.

    Loads the three production signal managed returns, vol-normalises to unit
    annual vol using IS σ only (no look-ahead), fits a rolling OLS MSRR on IS
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

    model = RollingMSRR(penalty="ols", lookback=lookback, min_obs=min_obs)
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
        cum = (1 + r).cumprod()
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
        print("  FX PRODUCTION  —  Rolling OLS MSRR  (lookback={:d}m, 3 signals)".format(lookback))
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
        print(f"  MDD = peak-to-trough NAV drawdown  |  TO = annual one-way (blended positions)\n")

    return pd.concat([r_is, r_oos]).sort_index().rename("fx_combined")
