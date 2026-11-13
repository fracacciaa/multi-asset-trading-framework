"""
Cross-asset-class blend layer.

Loads the production composite return series for each asset class (equities,
rates, FX, commodities) and weights them against each other using a menu of
allocation methods.

Each asset class's internal signal blend is treated as a single "sleeve" here;
this layer only decides how much capital to allocate across those four sleeves.

Data flow
---------
equities   run_eq_prod(verbose=False)   → daily  pd.Series
rates      rates.blend.run_blend()      → daily  pd.Series (r_is + r_oos)
fx         fx.blend.run_fx_blend()      → monthly pd.Series (r_is + r_oos)
commodities commodities.blend.run_comm_blend() → daily pd.Series (r_is + r_oos)

All series are resampled to monthly (BME) before blending and
vol-normalised to unit annual vol using IS σ so that blend weights
are directly comparable across asset classes.

IS/OOS split: IS ≤ IS_END = "2020-12-31", OOS > IS_END.

Blend methods
-------------
Static (IS-fitted once, constant thereafter):
    equal_weight   — 1/4 to each asset class  (DeMiguel et al. 2009)
    inv_vol        — 1/σ_k naïve risk parity  (Moskowitz et al. 2012)
    erc            — Equal Risk Contribution  (Maillard et al. 2010)
    sh_p1          — Sharpe heuristic  weight ∝ max(0, SR_k)
    msrr_ols       — MSRR OLS  b = M⁻¹ μ
    msrr_ridge     — MSRR Ridge  b = (M + λI)⁻¹ μ
    hrp            — Hierarchical Risk Parity (López de Prado 2016)

Dynamic (time-varying weights, monthly refit):
    rolling_msrr   — Rolling OLS MSRR  (lookback=36m default)
    dma            — Dynamic Model Averaging  (Raftery et al. 2010; α=0.99)

Entry point
-----------
try_asset_blend(equities_r, rates_r, fx_r, comm_r, ...)
    Runs all 9 methods and prints IS / OOS Sharpe, MDD, turnover.
    All four return series are optional; missing ones are built from the
    production blend functions (requires parquet caches and Bloomberg for
    equities).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from equities.blend import (
    BlendWeights,
    equal_weight,
    inv_vol_blend,
    erc_blend,
    hrp_blend,
    sharpe_heuristic,
    msrr_ridge,
    RollingMSRR,
    DMABlend,
)

IS_END    = "2020-12-31"
OOS_START = "2021-01-01"

ASSET_CLASSES = ["equities", "rates", "fx", "commodities"]

# Maps each cross-asset scenario to per-asset-class TC scenario labels / bp levels.
# Rates / FX / commodities use their own "best"/"base"/"worst" labels.
# Equities uses numeric one-way TC bps (no named scenarios in tc_sweep_eq).
TC_SCENARIOS_ASSETS: dict[str, dict] = {
    "best":  {"rates": "best",  "fx": "best",  "commodities": "best",  "equities": 0.25},
    "base":  {"rates": "base",  "fx": "base",  "commodities": "base",  "equities": 0.50},
    "worst": {"rates": "worst", "fx": "worst", "commodities": "worst", "equities": 1.00},
}


# ── Data loading ───────────────────────────────────────────────────────────────

def _to_monthly(s: pd.Series, name: str) -> pd.Series:
    """Resample a daily or monthly series to business-month-end frequency."""
    return s.resample("BME").sum().rename(name)


def load_asset_returns(
    equities_r: Optional[pd.Series] = None,
    rates_r:    Optional[pd.Series] = None,
    fx_r:       Optional[pd.Series] = None,
    comm_r:     Optional[pd.Series] = None,
    verbose:    bool = True,
) -> pd.DataFrame:
    """
    Build the T × 4 monthly composite-return matrix for the four asset classes.

    Any series not provided is built by calling the corresponding production
    blend function (rates/fx/commodities load from parquet caches; equities
    additionally requires Bloomberg price data or a pre-built vf_positions cache).

    Parameters
    ----------
    equities_r   Pre-computed equities composite return (daily or monthly).
    rates_r      Pre-computed rates composite return (daily or monthly).
    fx_r         Pre-computed FX composite return (daily or monthly).
    comm_r       Pre-computed commodities composite return (daily or monthly).
    verbose      Print build progress.

    Returns
    -------
    pd.DataFrame, shape (T, 4), monthly BME index, columns = ASSET_CLASSES.
    Only months where all four series have data are returned (inner join).
    """
    if equities_r is None:
        from equities.pipeline import run_eq_prod
        if verbose:
            print("  building equities composite return …")
        equities_r = run_eq_prod(verbose=False)

    if rates_r is None:
        from rates.blend import run_blend
        if verbose:
            print("  building rates composite return …")
        res = run_blend()
        rates_r = pd.concat([res["r_is"], res["r_oos"]]).sort_index().rename("rates")

    if fx_r is None:
        from fx.blend import run_fx_blend
        if verbose:
            print("  building FX composite return …")
        res = run_fx_blend()
        fx_r = pd.concat([res["r_is"], res["r_oos"]]).sort_index().rename("fx")

    if comm_r is None:
        from commodities.blend import run_comm_blend
        if verbose:
            print("  building commodities composite return …")
        res = run_comm_blend()
        comm_r = pd.concat([res["r_is"], res["r_oos"]]).sort_index().rename("commodities")

    F = pd.DataFrame({
        "equities":    _to_monthly(equities_r,  "equities"),
        "rates":       _to_monthly(rates_r,     "rates"),
        "fx":          _to_monthly(fx_r,        "fx"),
        "commodities": _to_monthly(comm_r,      "commodities"),
    })
    F.index.name = "date"
    return F.sort_index().dropna(how="any")


# ── Return / metric helpers ────────────────────────────────────────────────────

def _apply_static(F: pd.DataFrame, bw: BlendWeights) -> pd.Series:
    """Apply constant blend weights to a managed-return matrix."""
    w = bw.weights.reindex(F.columns).fillna(0.0)
    total = w.abs().sum()
    if total > 1e-12:
        w = w / total
    return (F * w).sum(axis=1)


def _apply_tvw(F: pd.DataFrame, w_df: pd.DataFrame) -> pd.Series:
    """Apply time-varying blend weights (L1-normalised row-wise) to F."""
    idx = F.index.intersection(w_df.dropna(how="any").index)
    if idx.empty:
        return pd.Series(dtype=float)
    w = w_df.loc[idx]
    norm = w.abs().sum(axis=1).replace(0.0, np.nan)
    w = w.div(norm, axis=0)
    return (F.loc[idx] * w).sum(axis=1)


def _sharpe(r: pd.Series, freq: int = 12) -> float:
    r = r.dropna()
    if len(r) < 12 or r.std() < 1e-12:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(freq))


def _mdd(r: pd.Series, vol_target: float = 0.10, freq: int = 12) -> float:
    r = r.dropna()
    if r.empty:
        return float("nan")
    ann_vol = r.std() * np.sqrt(freq)
    if ann_vol < 1e-12:
        return float("nan")
    r_scaled = r * (vol_target / ann_vol)
    cum = (1 + r_scaled).cumprod()
    return float((cum / cum.cummax() - 1).min())


def _tvw_turnover(w_df: pd.DataFrame, freq: int = 12) -> float:
    """Annualised asset-class weight turnover (sum of absolute monthly changes)."""
    w = w_df.dropna(how="any")
    if len(w) < 2:
        return float("nan")
    return float(w.diff().dropna(how="all").abs().sum(axis=1).mean() * freq)


def _portfolio_to(
    w_df: pd.DataFrame,
    to_df: pd.DataFrame,
    idx: pd.Index,
) -> float:
    """
    Total portfolio instrument-level turnover: blend-weight-averaged per-asset TOs.

    to_df is annualised (output of _build_position_turnover_series).
    Returns NaN if there is insufficient overlap.
    """
    w_period = w_df.reindex(idx).dropna(how="any")
    if len(w_period) < 2:
        return float("nan")
    to_period = to_df.reindex(w_period.index)
    w_avg = w_period.mean()
    total = w_avg.abs().sum()
    if total < 1e-12:
        return float("nan")
    w_norm = w_avg.abs() / total
    to_avg = to_period.mean()          # skipna=True — ignores early NaN rows
    cols = w_norm.index.intersection(to_avg.index)
    return float((w_norm[cols] * to_avg[cols]).sum())


def _drift_to(w_df: pd.DataFrame, F: pd.DataFrame, freq: int = 12) -> float:
    """
    Drift-corrected annualised one-way turnover of blend weights.

    At each step t, the portfolio drifts from last period's target weights
    under that month's returns.  TO = L1 distance from drifted weights to
    the new target, annualised.  Unlike _tvw_turnover, this is non-zero for
    static allocations because returns pull the portfolio away from its fixed
    target each month and a rebalance is needed to restore it.
    """
    idx = w_df.dropna(how="any").index.intersection(F.dropna(how="any").index)
    if len(idx) < 2:
        return float("nan")
    w = w_df.loc[idx].values.astype(float)
    r = F.loc[idx].values.astype(float)
    to_vals = []
    for i in range(1, len(idx)):
        w_prev = w[i - 1]
        gross  = (w_prev * (1.0 + r[i])).sum()
        if abs(gross) < 1e-12:
            continue
        w_drift = w_prev * (1.0 + r[i]) / gross
        to_vals.append(float(np.abs(w[i] - w_drift).sum()))
    return float(np.mean(to_vals) * freq) if to_vals else float("nan")


def _metrics(
    r: pd.Series,
    w_df: Optional[pd.DataFrame],
    is_tvw: bool,
    vol_target: float = 0.10,
    freq: int = 12,
) -> dict:
    return {
        "sharpe":   _sharpe(r, freq),
        "mdd":      _mdd(r, vol_target, freq),
        "turnover": _tvw_turnover(w_df, freq) if (is_tvw and w_df is not None) else 0.0,
        "n_months": int(r.dropna().shape[0]),
    }


# ── Main evaluation function ───────────────────────────────────────────────────

def try_asset_blend(
    equities_r: Optional[pd.Series] = None,
    rates_r:    Optional[pd.Series] = None,
    fx_r:       Optional[pd.Series] = None,
    comm_r:     Optional[pd.Series] = None,
    is_end:     str   = IS_END,
    lookback:   int   = 36,
    min_obs:    int   = 24,
    vol_target: float = 0.10,
    verbose:    bool  = True,
) -> dict:
    """
    Evaluate all nine cross-asset-class blend methods and report IS / OOS metrics.

    Fits every method strictly on IS data (no look-ahead). Dynamic methods
    (rolling_msrr, dma) are rolled forward on the full IS+OOS timeline.
    Reports annualised Sharpe, MDD at vol_target, and asset-class weight
    turnover for each method in IS and OOS.

    Parameters
    ----------
    equities_r   Equities composite return series (daily or monthly).
                 If None, calls equities.pipeline.run_eq_prod() — requires
                 Bloomberg data or the equities/cache/vf_positions.pkl cache.
    rates_r      Rates composite return series.  If None, built from parquets.
    fx_r         FX composite return series.    If None, built from parquets.
    comm_r       Commodities composite return.  If None, built from parquets.
    is_end       Last IS date (inclusive).  Default: "2020-12-31".
    lookback     Rolling-MSRR window in months.  Default: 36.
    min_obs      Minimum months before rolling weights are non-NaN.  Default: 24.
    vol_target   Annual vol target for MDD scaling.  Default: 0.10.
    verbose      Print the summary table.  Default: True.

    Returns
    -------
    dict  keyed by method name, each value:
        {"is": {"sharpe", "mdd", "turnover", "n_months"},
         "oos": {...},
         "type": "static" | "dynamic",
         "weights_is": pd.Series of avg IS weights}
    """
    # ── 1. Load monthly return matrix ─────────────────────────────────────────

    if verbose:
        print("\nLoading asset-class composite returns …")
    F_all = load_asset_returns(equities_r, rates_r, fx_r, comm_r, verbose=verbose)

    F_is_raw  = F_all.loc[F_all.index <= is_end]
    F_oos_raw = F_all.loc[F_all.index >  is_end]

    if F_is_raw.empty:
        raise ValueError(f"No IS data at or before {is_end}.")

    # ── 2. Vol-normalise to unit annual vol using IS σ only ────────────────────

    sigma     = F_is_raw.std() * np.sqrt(12)
    sigma     = sigma.replace(0.0, np.nan)
    F_is      = F_is_raw.div(sigma)
    F_oos     = F_oos_raw.div(sigma)
    F_full    = pd.concat([F_is, F_oos]).sort_index()
    names     = list(F_is.columns)

    # ── 3. Static methods (IS-fitted, constant weights) ────────────────────────

    static_blends: dict[str, BlendWeights] = {
        "equal_weight": equal_weight(names),
        "inv_vol":      inv_vol_blend(F_is),
        "erc":          erc_blend(F_is),
        "sh_p1":        sharpe_heuristic(F_is, power=1),
        "msrr_ols":     msrr_ridge(F_is, lam=0.0),
        "msrr_ridge":   msrr_ridge(F_is, lam=0.01),
        "hrp":          hrp_blend(F_is),
    }

    # ── 4. Dynamic methods (time-varying weights) ──────────────────────────────

    def _roll_normalised(penalty: str, lam: float = 0.0) -> pd.DataFrame:
        m = RollingMSRR(penalty=penalty, lam=lam, lookback=lookback, min_obs=min_obs)
        w_raw = m._roll(F_full)
        norm  = w_raw.abs().sum(axis=1).replace(0.0, np.nan)
        return w_raw.div(norm, axis=0)

    # DMA — fit on IS, predict on OOS
    dma = DMABlend(alpha=0.99)
    dma.fit(F_is)
    w_dma_oos = dma.predict_weights(F_oos)
    w_dma = pd.concat([dma._weights_history, w_dma_oos]).sort_index()

    dynamic_blends: dict[str, pd.DataFrame] = {
        "rolling_msrr_ols":   _roll_normalised("ols"),
        "rolling_msrr_ridge": _roll_normalised("ridge", lam=0.01),
        "rolling_msrr_lasso": _roll_normalised("lasso"),
        "dma":                w_dma,
    }

    # ── 5. Compute returns and metrics ─────────────────────────────────────────

    result: dict = {}

    for mname, bw in static_blends.items():
        r_is  = _apply_static(F_is,  bw)
        r_oos = _apply_static(F_oos, bw)
        w_norm = bw.weights.reindex(names).fillna(0.0)
        total  = w_norm.abs().sum()
        w_norm = w_norm / total if total > 1e-12 else w_norm
        result[mname] = {
            "is":         _metrics(r_is,  None, False, vol_target),
            "oos":        _metrics(r_oos, None, False, vol_target),
            "type":       "static",
            "weights_is": w_norm,
        }

    for mname, w_df in dynamic_blends.items():
        r_is  = _apply_tvw(F_is,  w_df)
        r_oos = _apply_tvw(F_oos, w_df)
        w_is_slice  = w_df.reindex(F_is.index).dropna()
        w_oos_slice = w_df.reindex(F_oos.index).dropna()
        result[mname] = {
            "is":         _metrics(r_is,  w_is_slice,  True, vol_target),
            "oos":        _metrics(r_oos, w_oos_slice, True, vol_target),
            "type":       "dynamic",
            "weights_is": w_is_slice.mean() if not w_is_slice.empty else pd.Series(dtype=float),
        }

    # ── 6. Print ───────────────────────────────────────────────────────────────

    if verbose:
        _print_table(result, is_end, F_is.index[0], F_oos.index[-1] if not F_oos.empty else None,
                     vol_target, lookback)

    return result


# ── Results → DataFrame ────────────────────────────────────────────────────────

def results_to_df(result: dict) -> pd.DataFrame:
    """
    Convert try_asset_blend() result dict to a tidy DataFrame.

    Columns: MultiIndex (period, metric) where
        period ∈ {"IS", "OOS"}
        metric ∈ {"Sharpe", "MDD", "Turnover"}
    Index: method names.
    MDD is expressed as a percentage (e.g. -7.76).
    """
    rows = {}
    for mname, stats in result.items():
        i, o = stats["is"], stats["oos"]
        rows[mname] = {
            ("IS",  "Sharpe"):   round(i["sharpe"],   2),
            ("IS",  "MDD %"):    round(i["mdd"] * 100, 2),
            ("IS",  "Turnover"): round(i["turnover"],  1),
            ("OOS", "Sharpe"):   round(o["sharpe"],   2),
            ("OOS", "MDD %"):    round(o["mdd"] * 100, 2),
            ("OOS", "Turnover"): round(o["turnover"],  1),
        }
    df = pd.DataFrame(rows).T
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df.index.name = "method"
    return df


# ── Print helpers ──────────────────────────────────────────────────────────────

def _print_table(
    result: dict,
    is_end: str,
    is_start: pd.Timestamp,
    oos_end: Optional[pd.Timestamp],
    vol_target: float,
    lookback: int,
) -> None:
    df = results_to_df(result)

    oos_end_str = oos_end.strftime("%Y-%m") if oos_end is not None else "n/a"
    W = 72

    print()
    print("=" * W)
    print("  CROSS-ASSET BLEND EVALUATION")
    print(f"  IS : {is_start.strftime('%Y-%m')} → {is_end[:7]}   |   OOS: {is_end[:7]} → {oos_end_str}")
    print(f"  MDD at {vol_target*100:.0f}% vol target  |  rolling lookback = {lookback}m")
    print("=" * W)
    print(
        f"  {'Method':<18}  {'Type':<8}"
        f"  {'IS SR':>7}  {'IS MDD':>8}  {'IS TO':>6}"
        f"  {'OOS SR':>7}  {'OOS MDD':>8}  {'OOS TO':>7}"
    )
    print("  " + "─" * (W - 2))

    prev_type = None
    for mname, stats in result.items():
        t = stats["type"]
        if prev_type is not None and t != prev_type:
            print("  " + "─" * (W - 2))
        prev_type = t
        r = df.loc[mname]

        def _sr(v):  return f"{v:+.2f}"  if np.isfinite(v) else "  n/a"
        def _mdd(v): return f"{v:+.1f}%" if np.isfinite(v) else "   n/a"
        def _to(v):  return f"{v:.1f}x"  if np.isfinite(v) else "  n/a"

        print(
            f"  {mname:<18}  {t:<8}"
            f"  {_sr(r[('IS','Sharpe')]):>7}"
            f"  {_mdd(r[('IS','MDD %')]):>8}"
            f"  {_to(r[('IS','Turnover')]):>6}"
            f"  {_sr(r[('OOS','Sharpe')]):>7}"
            f"  {_mdd(r[('OOS','MDD %')]):>8}"
            f"  {_to(r[('OOS','Turnover')]):>7}"
        )

    print("=" * W)

    # ── Avg IS weights ────────────────────────────────────────────────────────
    names = list(result[list(result.keys())[0]]["weights_is"].index)
    cw, hw = 13, 18
    print()
    print("  Avg IS blend weights")
    print("  " + "─" * (hw + cw * len(names) + 2))
    print(f"  {'Method':<{hw}}" + "".join(f"  {n:>{cw}}" for n in names))
    print("  " + "─" * (hw + cw * len(names) + 2))
    for mname, stats in result.items():
        w   = stats["weights_is"]
        row = "".join(f"  {w.get(n, 0.0):>{cw}.1%}" for n in names)
        print(f"  {mname:<{hw}}{row}")
    print("  " + "─" * (hw + cw * len(names) + 2))
    print()


# ── Position-level turnover ────────────────────────────────────────────────────

def _build_position_turnover_series() -> pd.DataFrame:
    """
    Compute monthly rolling-annualised instrument-level one-way turnover for each
    asset class.

    Equities   : blended VF ticker positions → daily |Δpos|/gross → 12m rolling × 252
    Rates      : position parquets, blend-weight averaged → same
    FX         : monthly momentum/carry positions (× 12) + daily MR positions (× 252),
                 blend-weight averaged → monthly
    Commodities: vol-scaled weekly-rebalanced positions rebuilt from raw signals
                 → daily |Δpos|/gross → 12m rolling × 252

    Returns pd.DataFrame, monthly BME index, columns = ASSET_CLASSES.
    """
    import contextlib, io as _io
    from pathlib import Path as _Path

    def _daily_ann_to(pos: pd.DataFrame, roll: int = 252, min_p: int = 63) -> pd.Series:
        gross = pos.abs().sum(axis=1).replace(0.0, np.nan)
        delta = pos.diff().abs().sum(axis=1)
        return (delta / gross).rolling(roll, min_periods=min_p).mean() * 252

    def _monthly_ann_to(pos: pd.DataFrame) -> pd.Series:
        gross = pos.abs().sum(axis=1).replace(0.0, np.nan)
        delta = pos.diff().abs().sum(axis=1)
        return (delta / gross) * 12  # already monthly, × 12 annualises

    # ── Equities ──────────────────────────────────────────────────────────────
    with contextlib.redirect_stdout(_io.StringIO()):
        from equities.pipeline import (
            _load_or_build_positions, PRODUCTION_SIGNALS, IS_END as _EQ_IS
        )
        from equities.data import fetch_prices as _fetch_eq
        from equities.blend import build_managed_returns, msrr_ridge, combine_signals

        _eq_pos = _load_or_build_positions(rebuild=False, verbose=False)
        _eq_pos = {k: v for k, v in _eq_pos.items() if k in PRODUCTION_SIGNALS}
        _eq_px  = _fetch_eq(tickers=None)
        _eq_ret = _eq_px.pct_change(fill_method=None).dropna(how="all")
        _ret_is_eq = _eq_ret.loc[_eq_ret.index <= _EQ_IS]
        _pos_is_eq = {k: v.loc[v.index <= _EQ_IS] for k, v in _eq_pos.items()}
        _F_is_eq   = build_managed_returns(_pos_is_eq, _ret_is_eq)
        _bw_eq     = msrr_ridge(_F_is_eq, lam=0.01)
        _comb_eq   = combine_signals(_eq_pos, _bw_eq)

    to_eq = _daily_ann_to(_comb_eq).resample("BME").last().rename("equities")

    # ── Rates ─────────────────────────────────────────────────────────────────
    from rates.blend import (
        run_blend as _run_rates_blend,
        _POS_FILES as _R_POS_FILES,
        _SIGNALS_DIR as _R_SDIR,
    )
    with contextlib.redirect_stdout(_io.StringIO()):
        _rates_res = _run_rates_blend()

    _w_rates = _rates_res["weights"]
    _wn_rates = _w_rates.abs().div(
        _w_rates.abs().sum(axis=1).replace(0.0, np.nan), axis=0
    )
    _to_rates = pd.Series(0.0, index=_wn_rates.index)
    for _rname, _rfname in _R_POS_FILES.items():
        _rpath = _R_SDIR / _rfname
        if not _rpath.exists() or _rname not in _wn_rates.columns:
            continue
        _rpos  = pd.read_parquet(_rpath)
        _gross = _rpos.abs().sum(axis=1).replace(0.0, np.nan)
        _delta = _rpos.diff().abs().sum(axis=1)
        _to_sig = (_delta / _gross).reindex(_wn_rates.index).fillna(0.0)
        _to_rates += _wn_rates[_rname].fillna(0.0) * _to_sig

    to_rates = (
        _to_rates.rolling(252, min_periods=63).mean() * 252
    ).resample("BME").last().rename("rates")

    # ── FX ────────────────────────────────────────────────────────────────────
    _FX_SDIR = _Path(__file__).parent / "fx" / "signals"
    from fx.blend import run_fx_blend as _run_fx_blend
    with contextlib.redirect_stdout(_io.StringIO()):
        _fx_res = _run_fx_blend()

    _w_fx = pd.concat(
        [_fx_res["weights_history"], _fx_res["weights_oos"]]
    ).sort_index().dropna(how="all")
    _sigma_fx = _fx_res["sigma"]
    _wn_fx = _w_fx.abs().div(
        _w_fx.abs().sum(axis=1).replace(0.0, np.nan), axis=0
    )

    # Build blended pair-level positions, then compute TO on the combined
    # portfolio (mirrors tc_sweep_fx._to / run_fx_prod._blended_pos).
    _fx_monthly_pos = {
        "momentum":  pd.read_parquet(_FX_SDIR / "signal_a_f9_h1.parquet"),
        "mr_ns_loo": (pd.read_parquet(_FX_SDIR / "mr_ns_loo_positions.parquet")
                      .resample("BME").last()),
        "carry_xs":  pd.read_parquet(_FX_SDIR / "view_xs_carry_m.parquet"),
    }
    _fx_common_pairs = sorted(
        set.intersection(*[set(p.columns) for p in _fx_monthly_pos.values()])
    )

    _fx_idx = _wn_fx.dropna(how="any").index
    _fx_pos = pd.DataFrame(0.0, index=_fx_idx, columns=_fx_common_pairs)
    for _fk in _wn_fx.columns:
        if _fk not in _fx_monthly_pos or _fk not in _sigma_fx.index:
            continue
        _pk = _fx_monthly_pos[_fk].reindex(
            index=_fx_idx, columns=_fx_common_pairs
        ).fillna(0.0)
        _fx_pos += _pk.mul(_wn_fx[_fk] / _sigma_fx[_fk], axis=0)

    _fx_gross = _fx_pos.abs().sum(axis=1).replace(0.0, np.nan)
    _fx_delta = _fx_pos.diff().abs().sum(axis=1)
    to_fx = (
        (_fx_delta / _fx_gross).rolling(12, min_periods=6).mean() * 12
    ).rename("fx")

    # ── Commodities ───────────────────────────────────────────────────────────
    from commodities.backtest import (
        PARAMS as _CP,
        specific_contract_returns as _spec_rets,
        _build_signals as _cbuild,
        vol_scale as _cvs,
        rebalance as _creb,
        book_scale as _cbook,
    )
    from commodities.data import (
        fetch_prices as _fetch_comm,
        fetch_term_structure as _fetch_ts,
        fetch_tickers as _fetch_tk,
    )
    from commodities.blend import run_comm_blend as _run_comm_blend

    with contextlib.redirect_stdout(_io.StringIO()):
        _comm_res = _run_comm_blend()

    _px_c   = _fetch_comm()
    _ts_c   = _fetch_ts()
    _tk_c   = _fetch_tk()
    _ret_c  = _spec_rets(_ts_c, _tk_c).reindex(columns=_px_c.columns)
    _rsigs  = _cbuild(_px_c, _ts_c, _tk_c, _CP)

    _cname_map = {
        "tsmom_252":         "tsmom_252",
        "xs_carry_enh_f1f2": "xs_carry_enh",
        "curvature":         "curvature",
    }
    _w_k_c: dict[str, pd.DataFrame] = {}
    for _rk, _bk in _cname_map.items():
        if _rk not in _rsigs:
            continue
        _sig_c = _rsigs[_rk].reindex(index=_ret_c.index, columns=_ret_c.columns)
        _w_c   = _cvs(_sig_c, _ret_c, target_vol=_CP["asset_vol"], com=_CP["vol_com"])
        _w_c   = _creb(_w_c, rule=_CP["rebal"])
        _w_c   = _cbook(_w_c, _ret_c, target_vol=_CP["book_vol"],
                        cov_hl=_CP["cov_hl"], k_max=_CP["k_max"])
        _w_k_c[_bk] = _w_c

    _w_is_c   = _comm_res["weights_is"]
    _w_oos_c  = _comm_res["weights_oos"]
    _alpha_c  = pd.concat([_w_is_c, _w_oos_c]).sort_index()
    _all_idx  = _w_k_c[list(_w_k_c)[0]].index
    _all_cols = _w_k_c[list(_w_k_c)[0]].columns

    _a_df   = _alpha_c.reindex(_all_idx).ffill().fillna(0.0)
    _norms  = _a_df.abs().sum(axis=1).replace(0.0, np.nan)
    _a_norm = _a_df.div(_norms, axis=0).fillna(0.0)

    _W_comm = pd.DataFrame(0.0, index=_all_idx, columns=_all_cols)
    for _ck in _a_norm.columns:
        if _ck in _w_k_c:
            _W_comm += (
                _w_k_c[_ck]
                .reindex(index=_all_idx, columns=_all_cols).fillna(0.0)
                .mul(_a_norm[_ck], axis=0)
            )

    to_comm = _daily_ann_to(_W_comm).resample("BME").last().rename("commodities")

    to_df = pd.DataFrame({
        "equities":    to_eq,
        "rates":       to_rates,
        "fx":          to_fx,
        "commodities": to_comm,
    }).sort_index()

    _cache_dir = _Path(__file__).parent / "cache"
    _cache_dir.mkdir(exist_ok=True)
    to_df.to_parquet(_cache_dir / "position_turnover.parquet")

    return to_df


# ── Production entry point ─────────────────────────────────────────────────────

def run_asset_blend(
    equities_r: Optional[pd.Series] = None,
    rates_r:    Optional[pd.Series] = None,
    fx_r:       Optional[pd.Series] = None,
    comm_r:     Optional[pd.Series] = None,
    is_end:     str   = IS_END,
    lookback:   int   = 36,
    min_obs:    int   = 24,
    lam:        float = 0.01,
    vol_target: float = 0.10,
    plot:       bool  = True,
    verbose:    bool  = True,
) -> pd.Series:
    """
    Production cross-asset blend: rolling MSRR Ridge.

    Combines the four asset-class composite return series into a single
    multi-asset portfolio using a rolling-window Ridge MSRR blend, refitted
    every month on the last `lookback` months of vol-normalised managed returns.

    Prints IS / OOS Sharpe, MDD, and turnover, avg blend weights for each
    period, and optionally plots the cumulative return and time-varying weights.

    Parameters
    ----------
    equities_r   Pre-computed equities composite (daily or monthly).
    rates_r      Pre-computed rates composite.
    fx_r         Pre-computed FX composite.
    comm_r       Pre-computed commodities composite.
    is_end       Last IS date.  Default: "2020-12-31".
    lookback     Rolling window in months.  Default: 36.
    min_obs      Minimum observations before first weight.  Default: 24.
    lam          Ridge regularisation.  Default: 0.01.
    vol_target   Annual vol target for MDD scaling and plot.  Default: 0.10.
    plot         Show and save cumulative-return + weights chart.  Default: True.
    verbose      Print the summary table.  Default: True.

    Returns
    -------
    pd.Series of monthly combined portfolio returns (IS + OOS),
    name = "asset_blend", vol-normalised to unit annual vol (IS σ).
    """
    # ── 1. Load ───────────────────────────────────────────────────────────────

    if verbose:
        print("\nLoading asset-class composite returns …")
    F_all = load_asset_returns(equities_r, rates_r, fx_r, comm_r, verbose=verbose)

    F_is_raw  = F_all.loc[F_all.index <= is_end]
    F_oos_raw = F_all.loc[F_all.index >  is_end]

    # ── 2. Vol-normalise (IS σ only) ──────────────────────────────────────────

    sigma  = F_is_raw.std() * np.sqrt(12)
    sigma  = sigma.replace(0.0, np.nan)
    F_is   = F_is_raw.div(sigma)
    F_oos  = F_oos_raw.div(sigma)
    F_full = pd.concat([F_is, F_oos]).sort_index()
    names  = list(F_is.columns)

    # ── 3. Rolling MSRR Ridge — walk-forward on full timeline ────────────────

    model = RollingMSRR(penalty="ridge", lam=lam, lookback=lookback, min_obs=min_obs)
    w_raw = model._roll(F_full)
    norm  = w_raw.abs().sum(axis=1).replace(0.0, np.nan)
    w_all = w_raw.div(norm, axis=0)

    w_is  = w_all.reindex(F_is.index)
    w_oos = w_all.reindex(F_oos.index)

    r_is  = _apply_tvw(F_is,  w_is)
    r_oos = _apply_tvw(F_oos, w_oos)
    r_full = pd.concat([r_is, r_oos]).sort_index().rename("asset_blend")

    # ── 4. Metrics ────────────────────────────────────────────────────────────

    is_sr  = _sharpe(r_is)
    is_mdd = _mdd(r_is, vol_target)
    is_to  = _tvw_turnover(w_is.dropna())

    oos_sr  = _sharpe(r_oos)
    oos_mdd = _mdd(r_oos, vol_target)
    oos_to  = _tvw_turnover(w_oos.dropna())

    w_is_avg  = w_is.dropna().mean()
    w_oos_avg = w_oos.dropna().mean()

    if verbose:
        W = 60
        print()
        print("=" * W)
        print("  MULTI-ASSET PRODUCTION  —  Rolling MSRR Ridge")
        print(f"  lookback={lookback}m  λ={lam}  vol target={vol_target*100:.0f}%")
        print(f"  IS : {F_is.index[0].strftime('%Y-%m')} → {F_is.index[-1].strftime('%Y-%m')}"
              f"  ({len(r_is)} months)")
        print(f"  OOS: {F_oos.index[0].strftime('%Y-%m')} → {F_oos.index[-1].strftime('%Y-%m')}"
              f"  ({len(r_oos)} months)")
        print("=" * W)
        print(f"  {'':22}  {'IS':>12}  {'OOS':>12}")
        print("  " + "─" * (W - 2))
        print(f"  {'Sharpe (ann)':<22}  {is_sr:>+12.2f}  {oos_sr:>+12.2f}")
        print(f"  {'MDD':<22}  {is_mdd*100:>+11.2f}%  {oos_mdd*100:>+11.2f}%")
        print(f"  {'Turnover (ann)':<22}  {is_to:>11.1f}x  {oos_to:>11.1f}x")
        print("=" * W)

        # Avg blend weights
        cw = 10
        print()
        print("  Avg blend weights")
        print(f"  {'':14}  {'IS avg':>{cw}}  {'OOS avg':>{cw}}")
        print("  " + "─" * (14 + (cw + 2) * 2))
        for n in names:
            print(f"  {n:<14}  {w_is_avg.get(n, 0.0):>{cw}.1%}  {w_oos_avg.get(n, 0.0):>{cw}.1%}")
        print("  " + "─" * (14 + (cw + 2) * 2))

    # ── 5. Plot ───────────────────────────────────────────────────────────────

    if plot:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            if verbose:
                print("\n  Computing position-level turnover …")
            to_df = _build_position_turnover_series()

            # Scale full series to vol_target annual vol (using IS vol)
            ann_vol = r_is.std() * np.sqrt(12)
            scale   = vol_target / ann_vol if ann_vol > 1e-12 else 1.0
            r_plot  = r_full * scale
            cum     = (1 + r_plot).cumprod()

            is_split = pd.Timestamp(is_end)
            colors   = {"equities": "#2196F3", "rates": "#4CAF50",
                        "fx": "#FF9800", "commodities": "#9C27B0"}

            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(12, 8),
                gridspec_kw={"height_ratios": [3, 2]},
                sharex=True,
            )

            # ── Top panel: cumulative return ──────────────────────────────────
            ax1.fill_betweenx(
                [cum.min() * 0.95, cum.max() * 1.05],
                cum.index[0], is_split,
                color="#f5f5f5", zorder=0, label="_nolegend_"
            )
            ax1.plot(cum.index, cum.values, color="#1a237e", lw=1.8, zorder=2)
            ax1.axvline(is_split, color="gray", lw=1.0, ls="--", zorder=1)
            ax1.set_ylabel("Wealth index", fontsize=10)
            ax1.set_title(
                f"Multi-Asset Blend  —  Rolling MSRR Ridge  "
                f"(lookback={lookback}m, λ={lam})\n"
                f"IS Sharpe {is_sr:+.2f}  MDD {is_mdd*100:+.1f}%    "
                f"OOS Sharpe {oos_sr:+.2f}  MDD {oos_mdd*100:+.1f}%",
                fontsize=10,
            )
            ax1.text(
                0.02, 0.97, "IS", transform=ax1.transAxes,
                fontsize=9, color="gray", va="top",
            )
            ax1.text(
                0.72, 0.97, "OOS", transform=ax1.transAxes,
                fontsize=9, color="gray", va="top",
            )
            ax1.yaxis.grid(True, ls=":", lw=0.5, color="#cccccc")
            ax1.set_axisbelow(True)

            # ── Bottom panel: instrument-level position turnover ──────────────
            to_plot = to_df.dropna(how="all")
            _to_ymin = max(0.0, to_plot.stack().quantile(0.02) * 0.9)
            _to_ymax = to_plot.stack().quantile(0.98) * 1.1

            ax2.fill_betweenx(
                [_to_ymin, _to_ymax],
                to_plot.index[0], is_split,
                color="#f5f5f5", zorder=0,
            )
            for n in ASSET_CLASSES:
                if n not in to_plot.columns:
                    continue
                _s = to_plot[n].dropna()
                ax2.plot(
                    _s.index, _s.values,
                    color=colors.get(n, "black"), lw=1.4, label=n,
                )
            ax2.axvline(is_split, color="gray", lw=1.0, ls="--")
            ax2.set_ylabel("Ann. position turnover (×)", fontsize=10)
            ax2.set_ylim(bottom=_to_ymin, top=_to_ymax)
            ax2.legend(loc="upper right", fontsize=8, ncol=2)
            ax2.yaxis.grid(True, ls=":", lw=0.5, color="#cccccc")
            ax2.set_axisbelow(True)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax2.xaxis.set_major_locator(mdates.YearLocator(2))

            fig.tight_layout()
            out_path = Path(__file__).parent / "asset_blend_composite.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            if verbose:
                print(f"  Chart saved → {out_path}")
            plt.show()

        except Exception as e:
            if verbose:
                print(f"\n  [plot skipped: {e}]")

    return r_full


def compare_blend_methods(
    equities_r: Optional[pd.Series] = None,
    rates_r:    Optional[pd.Series] = None,
    fx_r:       Optional[pd.Series] = None,
    comm_r:     Optional[pd.Series] = None,
    is_end:     str   = IS_END,
    lookback:   int   = 36,
    min_obs:    int   = 24,
    ridge_lam:  float = 0.01,
    vol_target: float = 0.10,
    verbose:    bool  = True,
) -> pd.DataFrame:
    """
    Compare three cross-asset blend methods side-by-side: Sharpe, MDD, TO.

    Methods
    -------
    Rolling OLS MSRR   rolling lookback=36m refit on full IS+OOS timeline
    Rolling Ridge MSRR same rolling window with L2 penalty — PRODUCTION method
    Static Ridge MSRR  IS-fit once with ridge, constant weights thereafter

    Monthly frequency; Sharpe annualised with √12.
    MDD at exactly vol_target (r scaled to vol_target annual vol).
    TO = annualised asset-class weight turnover (static methods = 0.0x).

    Returns
    -------
    pd.DataFrame with columns [Method, Period, Sharpe, MDD, TO].
    """
    if verbose:
        print("\nLoading asset-class composite returns …")
    F_all = load_asset_returns(equities_r, rates_r, fx_r, comm_r, verbose=verbose)

    F_is_raw  = F_all.loc[F_all.index <= is_end]
    F_oos_raw = F_all.loc[F_all.index >  is_end]

    sigma  = F_is_raw.std() * np.sqrt(12)
    sigma  = sigma.replace(0.0, np.nan)
    F_is   = F_is_raw.div(sigma)
    F_oos  = F_oos_raw.div(sigma)
    F_full = pd.concat([F_is, F_oos]).sort_index()

    def _normalise_w(w_raw: pd.DataFrame) -> pd.DataFrame:
        norm = w_raw.abs().sum(axis=1).replace(0.0, np.nan)
        return w_raw.div(norm, axis=0)

    # ── Method 1: Rolling OLS ────────────────────────────────────────────────
    w_ols = _normalise_w(
        RollingMSRR(penalty="ols", lookback=lookback, min_obs=min_obs)._roll(F_full)
    )

    # ── Method 2: Rolling Ridge (PRODUCTION) ─────────────────────────────────
    w_ridge = _normalise_w(
        RollingMSRR(penalty="ridge", lam=ridge_lam,
                    lookback=lookback, min_obs=min_obs)._roll(F_full)
    )

    # ── Method 3: Static Ridge (IS-fit) ──────────────────────────────────────
    bw_static  = msrr_ridge(F_is, lam=ridge_lam)
    w_static_s = bw_static.weights / bw_static.weights.abs().sum()
    w_static = pd.DataFrame(
        np.tile(w_static_s.reindex(F_full.columns).values, (len(F_full), 1)),
        index=F_full.index,
        columns=F_full.columns,
    )

    methods = {
        "Rolling OLS":          w_ols,
        "Rolling Ridge [prod]": w_ridge,
        "Static Ridge":         w_static,
    }

    records = []
    for method_name, w_full in methods.items():
        for period, F, w in [
            ("IS",  F_is,  w_full.reindex(F_is.index)),
            ("OOS", F_oos, w_full.reindex(F_oos.index)),
        ]:
            r  = _apply_tvw(F, w)
            to = _drift_to(w, F)
            records.append({
                "Method": method_name,
                "Period": period,
                "Sharpe": _sharpe(r),
                "MDD":    _mdd(r, vol_target) * 100,
                "TO":     to,
            })

    df = pd.DataFrame(records)

    # ── Print table ──────────────────────────────────────────────────────────
    W = 72
    print()
    print("=" * W)
    print(f"  CROSS-ASSET BLEND COMPARISON  (vol target={vol_target*100:.0f}%, λ_ridge={ridge_lam})")
    print(f"  IS : {F_is.index[0].strftime('%Y-%m')} → {is_end[:7]}   |"
          f"   OOS: {is_end[:7]} → {F_oos.index[-1].strftime('%Y-%m')}")
    print("=" * W)
    print(f"  {'Method':<24}  {'Period':<6}  {'Sharpe':>7}  {'MDD':>8}  {'TO':>7}")
    print("  " + "─" * (W - 2))
    for _, row in df.iterrows():
        sr  = f"{row['Sharpe']:+.2f}" if np.isfinite(row["Sharpe"]) else "   n/a"
        mdd = f"{row['MDD']:.1f}%"    if np.isfinite(row["MDD"])    else "    n/a"
        to  = f"{row['TO']:.2f}x"     if np.isfinite(row["TO"])     else "   n/a"
        print(f"  {row['Method']:<24}  {row['Period']:<6}  {sr:>7}  {mdd:>8}  {to:>7}")
        if row["Period"] == "OOS":
            print("  " + "─" * (W - 2))
    print("=" * W)

    # ── Avg IS weights ────────────────────────────────────────────────────────
    names = list(F_is.columns)
    cw    = 13
    print(f"\n  Avg IS weights")
    print("  " + "─" * (26 + cw * len(names)))
    print(f"  {'Method':<24}" + "".join(f"  {n:>{cw}}" for n in names))
    print("  " + "─" * (26 + cw * len(names)))
    for method_name, w_full in methods.items():
        w_avg = w_full.reindex(F_is.index).dropna(how="any").mean()
        row   = "".join(f"  {w_avg.get(n, 0.0):>{cw}.1%}" for n in names)
        print(f"  {method_name:<24}{row}")
    print("  " + "─" * (26 + cw * len(names)))

    return df


def run_assets_tc(
    scenarios: Optional[dict] = None,
    is_end: str = IS_END,
    oos_start: str = OOS_START,
    lookback: int = 36,
    min_obs: int = 24,
    vol_target: float = 0.10,
    lam: float = 0.01,
) -> dict:
    """
    Full cross-asset TC sweep: metrics table + monthly return timeseries.

    For each TC scenario (best / base / worst), feeds the TC-adjusted "Fixed
    weights" composite return from each asset class into the cross-asset blend
    layer and evaluates two blend methods:

      Fixed weights  — Rolling Ridge fit ONCE on TC-free cross-asset returns;
                       only the per-asset TC deduction changes across scenarios.
                       Shows pure TC impact propagated from asset-class level.
      Adaptive Ridge — Rolling Ridge RE-FIT on TC-adjusted cross-asset returns
                       per scenario; allocator can shift away from high-TC assets.

    Per-asset TC uses the "Fixed weights" method from each asset class's sweep:
      rates       : TC_SCENARIOS  (market-specific OIS spreads)
      fx          : TC_SCENARIOS_FX  (pair-specific forward spreads)
      commodities : TC_SCENARIOS_COMM  (estimated spread × TO drag)
      equities    : tc_bps mapped to 0.25 / 0.50 / 1.00 for best/base/worst

    Parameters
    ----------
    scenarios   Dict mapping label → {asset_class: scenario_key}.
                Default: TC_SCENARIOS_ASSETS.
    is_end      Last IS date (inclusive).  Default: "2020-12-31".
    oos_start   First OOS date (inclusive).  Default: "2021-01-01".
    lookback    Rolling Ridge window in months.  Default: 36.
    min_obs     Minimum months before first weight estimate.  Default: 24.
    vol_target  For MDD computation.  Default: 0.10.
    lam         Ridge penalty.  Default: 0.01.

    Returns
    -------
    dict:
        'metrics'   pd.DataFrame — [Method, Scenario, Period, Sharpe, MDD_pct, TO]
        'returns'   pd.DataFrame — monthly returns, MultiIndex columns (Method, Scenario)
    """
    import contextlib
    import io as _io

    if scenarios is None:
        scenarios = TC_SCENARIOS_ASSETS

    from rates.blend       import tc_sweep_rates, TC_SCENARIOS       as _TC_RATES
    from fx.blend          import tc_sweep_fx,    TC_SCENARIOS_FX    as _TC_FX
    from commodities.blend import tc_sweep_comm,  TC_SCENARIOS_COMM  as _TC_COMM
    from equities.pipeline import tc_sweep_eq

    # ── 1. Run per-asset TC sweeps, suppress their print output ──────────────
    print("Running per-asset TC sweeps …")
    with contextlib.redirect_stdout(_io.StringIO()):
        _, rates_ret = tc_sweep_rates(_collect_returns=True)
        _, fx_ret    = tc_sweep_fx(_collect_returns=True)
        _, comm_ret  = tc_sweep_comm(_collect_returns=True)
        _, eq_ret    = tc_sweep_eq(_collect_returns=True)

    # ── 1b. Instrument-level position turnover (computed once, reused) ────────
    print("Computing instrument-level turnover …")
    with contextlib.redirect_stdout(_io.StringIO()):
        to_df = _build_position_turnover_series()

    # ── 2. TC-free cross-asset returns for Fixed weights baseline ─────────────
    print("Building TC-free cross-asset baseline …")
    F_free_all     = load_asset_returns(verbose=False)
    F_free_is_raw  = F_free_all.loc[F_free_all.index <= is_end]
    F_free_oos_raw = F_free_all.loc[F_free_all.index >= oos_start]

    sigma_free = F_free_is_raw.std() * np.sqrt(12)
    sigma_free = sigma_free.replace(0.0, np.nan)
    F_free_is  = F_free_is_raw.div(sigma_free)
    F_free_oos = F_free_oos_raw.div(sigma_free)

    # Walk-forward Rolling Ridge on full gross IS+OOS timeline.
    # predict_weights(_oos_only) would restart the lookback window from scratch,
    # losing IS history and producing NaN for the first min_obs OOS months.
    # Using _roll(full) carries IS data into OOS, matching run_asset_blend().
    model_free  = RollingMSRR(penalty="ridge", lam=lam, lookback=lookback, min_obs=min_obs)
    F_free_full = pd.concat([F_free_is, F_free_oos]).sort_index()
    w_free_all  = model_free._roll(F_free_full)
    w_is_fixed  = w_free_all.reindex(F_free_is.index)
    w_oos_fixed = w_free_all.reindex(F_free_oos.index)

    def _norm_w(w_df: pd.DataFrame) -> pd.DataFrame:
        """L1-normalise weights row-wise (mirrors _apply_tvw logic)."""
        n = w_df.abs().sum(axis=1).replace(0.0, np.nan)
        return w_df.div(n, axis=0)

    # ── 3. TC scenario sweep ──────────────────────────────────────────────────
    records: list = []
    _returns_dict: dict[tuple, pd.Series] = {}

    for scenario_name, asset_scens in scenarios.items():
        # Pull "Fixed weights" TC-adjusted return series from each asset class
        r_rates = rates_ret[("Fixed weights", asset_scens["rates"])]
        r_fx    = fx_ret   [("Fixed weights", asset_scens["fx"])]
        r_comm  = comm_ret [("Fixed weights", asset_scens["commodities"])]
        r_eq    = eq_ret   [("Static Ridge [prod]", asset_scens["equities"])]

        # Build monthly TC-adjusted F matrix (same structure as load_asset_returns)
        F_tc_all = pd.DataFrame({
            "equities":    _to_monthly(r_eq,    "equities"),
            "rates":       _to_monthly(r_rates, "rates"),
            "fx":          _to_monthly(r_fx,    "fx"),
            "commodities": _to_monthly(r_comm,  "commodities"),
        }).sort_index().dropna(how="any")

        F_tc_is_raw  = F_tc_all.loc[F_tc_all.index <= is_end]
        F_tc_oos_raw = F_tc_all.loc[F_tc_all.index >= oos_start]

        if len(F_tc_is_raw) < min_obs or F_tc_oos_raw.empty:
            continue

        # Average one-way spread per asset class for this scenario (bps).
        # Rates and FX: mean of instrument/pair-specific spreads.
        # Commodities: bps/yr drag ÷ est. 4× annual TO ≈ per-trade spread.
        # Equities: numeric tc_bps level (already one-way bp).
        h_bp = {
            "rates":       float(np.mean(list(_TC_RATES[asset_scens["rates"]].values()))),
            "fx":          float(np.mean(list(_TC_FX[asset_scens["fx"]].values()))),
            "commodities": _TC_COMM[asset_scens["commodities"]] / 4.0,
            "equities":    float(asset_scens["equities"]),
        }
        h_vec = pd.Series({k: v / 10_000.0 for k, v in h_bp.items()})

        # ── Method 1: Fixed weights ───────────────────────────────────────────
        # Normalise using TC-free IS sigma (no look-ahead, no re-fitting)
        F_tc_is_n  = F_tc_is_raw.div(sigma_free)
        F_tc_oos_n = F_tc_oos_raw.div(sigma_free)

        r_is_f  = _apply_tvw(F_tc_is_n,  w_is_fixed.reindex(F_tc_is_n.index))
        r_oos_f = _apply_tvw(F_tc_oos_n, w_oos_fixed.reindex(F_tc_oos_n.index))

        # Deduct cross-asset rebalancing TC: cost of scaling sleeve k by |Δw_k|
        w_f_comb = pd.concat([
            _norm_w(w_is_fixed.reindex(r_is_f.index).dropna(how="any")),
            _norm_w(w_oos_fixed.reindex(r_oos_f.index).dropna(how="any")),
        ]).sort_index()
        tc_cross_f = (w_f_comb.diff().abs().fillna(0.0) * h_vec).sum(axis=1)
        r_net_f    = pd.concat([r_is_f, r_oos_f]).sort_index() - tc_cross_f.reindex(
            pd.concat([r_is_f, r_oos_f]).index).fillna(0.0)
        r_is_f  = r_net_f.loc[r_net_f.index <= is_end]
        r_oos_f = r_net_f.loc[r_net_f.index >= oos_start]

        for period, r, w in [
            ("IS",  r_is_f,  w_is_fixed),
            ("OOS", r_oos_f, w_oos_fixed),
        ]:
            records.append({
                "Method":   "Fixed weights",
                "Scenario": scenario_name,
                "Period":   period,
                "Sharpe":   _sharpe(r),
                "MDD_pct":  _mdd(r, vol_target) * 100,
                "TO":       _portfolio_to(w, to_df, r.index),
            })

        _returns_dict[("Fixed weights", scenario_name)] = (
            pd.concat([r_is_f, r_oos_f]).sort_index()
        )

        # ── Method 2: Adaptive Ridge ──────────────────────────────────────────
        # Re-normalise and re-fit per scenario
        sigma_tc = F_tc_is_raw.std() * np.sqrt(12)
        sigma_tc = sigma_tc.replace(0.0, np.nan)
        F_tc_is_n2  = F_tc_is_raw.div(sigma_tc)
        F_tc_oos_n2 = F_tc_oos_raw.div(sigma_tc)

        model_tc     = RollingMSRR(penalty="ridge", lam=lam, lookback=lookback, min_obs=min_obs)
        F_tc_full_n2 = pd.concat([F_tc_is_n2, F_tc_oos_n2]).sort_index()
        w_all_a      = model_tc._roll(F_tc_full_n2)
        w_is_a       = w_all_a.reindex(F_tc_is_n2.index)
        w_oos_a      = w_all_a.reindex(F_tc_oos_n2.index)

        r_is_a  = _apply_tvw(F_tc_is_n2,  w_is_a)
        r_oos_a = _apply_tvw(F_tc_oos_n2, w_oos_a)

        # Deduct cross-asset rebalancing TC for adaptive weights
        w_a_comb = pd.concat([
            _norm_w(w_is_a.dropna(how="any")),
            _norm_w(w_oos_a.dropna(how="any")),
        ]).sort_index()
        tc_cross_a = (w_a_comb.diff().abs().fillna(0.0) * h_vec).sum(axis=1)
        r_net_a    = pd.concat([r_is_a, r_oos_a]).sort_index() - tc_cross_a.reindex(
            pd.concat([r_is_a, r_oos_a]).index).fillna(0.0)
        r_is_a  = r_net_a.loc[r_net_a.index <= is_end]
        r_oos_a = r_net_a.loc[r_net_a.index >= oos_start]

        for period, r, w in [
            ("IS",  r_is_a,  w_is_a),
            ("OOS", r_oos_a, w_oos_a),
        ]:
            records.append({
                "Method":   "Adaptive Ridge",
                "Scenario": scenario_name,
                "Period":   period,
                "Sharpe":   _sharpe(r),
                "MDD_pct":  _mdd(r, vol_target) * 100,
                "TO":       _portfolio_to(w, to_df, r.index),
            })

        _returns_dict[("Adaptive Ridge", scenario_name)] = (
            pd.concat([r_is_a, r_oos_a]).sort_index()
        )

    df = pd.DataFrame(records)

    # ── 4. Print results ──────────────────────────────────────────────────────
    W = 80
    for method in ["Fixed weights", "Adaptive Ridge"]:
        sub = df[df["Method"] == method]
        label = (
            "FIXED WEIGHTS — TC-free Ridge fit; only cost deducted"
            if method == "Fixed weights"
            else "ADAPTIVE RIDGE — Ridge re-fit per TC scenario"
        )
        print("\n" + "=" * W)
        print(f"  MULTI-ASSET TC SWEEP — {label}")
        print(f"  IS ≤ {is_end}  |  OOS ≥ {oos_start}  |  lookback={lookback}m  |  λ={lam}")
        print("=" * W)
        print(f"  {'Scenario':<12}  {'Period':<6}  {'Sharpe':>8}  {'MDD':>8}  {'TO':>7}")
        print("  " + "─" * (W - 2))
        prev = None
        for _, row in sub.iterrows():
            if prev is not None and row["Scenario"] != prev:
                print("  " + "─" * (W - 2))
            prev = row["Scenario"]
            sr  = f"{row['Sharpe']:+.2f}"   if not np.isnan(row["Sharpe"])  else "   n/a"
            mdd = f"{row['MDD_pct']:.1f}%"  if not np.isnan(row["MDD_pct"]) else "   n/a"
            to  = f"{row['TO']:.2f}x"       if not np.isnan(row["TO"])      else "   n/a"
            print(f"  {row['Scenario']:<12}  {row['Period']:<6}  {sr:>8}  {mdd:>8}  {to:>7}")
        print("=" * W)

    # ── Scenario mapping table ────────────────────────────────────────────────
    scen_names = list(scenarios.keys())
    col_w = 12
    print(f"\n  Per-asset TC scenario mapping")
    print("  " + "─" * (22 + col_w * len(scen_names)))
    print(f"  {'Asset class':<20}" + "".join(f"  {s:>{col_w}}" for s in scen_names))
    print("  " + "─" * (22 + col_w * len(scen_names)))
    for ac in ["rates", "fx", "commodities", "equities"]:
        row = f"  {ac:<20}"
        for sn in scen_names:
            v = scenarios[sn][ac]
            label = f"{v:.2f} bp" if isinstance(v, float) else str(v)
            row += f"  {label:>{col_w}}"
        print(row)
    print("  " + "─" * (22 + col_w * len(scen_names)))

    # ── Build and return ──────────────────────────────────────────────────────
    returns_df = pd.DataFrame(_returns_dict)
    returns_df.columns = pd.MultiIndex.from_tuples(
        returns_df.columns, names=["Method", "Scenario"]
    )
    return {
        "metrics":     df,
        "returns":     returns_df,
        "to_df":       to_df,
        "w_is_fixed":  w_is_fixed,
        "w_oos_fixed": w_oos_fixed,
    }


# ── Production TC entry point ──────────────────────────────────────────────────

def run_asset_tc(
    scenarios: Optional[dict] = None,
    is_end:     str   = IS_END,
    oos_start:  str   = OOS_START,
    lookback:   int   = 36,
    min_obs:    int   = 24,
    vol_target: float = 0.10,
    lam:        float = 0.01,
    plot:       bool  = True,
    verbose:    bool  = True,
) -> pd.Series:
    """
    Production cross-asset TC analysis: gross vs net-of-TC metrics + chart.

    Builds the TC-free gross portfolio (Rolling Ridge MSRR), runs run_assets_tc()
    for all TC scenarios, then prints a combined IS / OOS summary table and
    optionally plots the cumulative return under gross, best, base, and worst
    TC scenarios.

    Two methods are compared, mirroring run_assets_tc():
      Fixed weights  — Rolling Ridge fit once on gross returns; only the
                       per-asset cost propagates into the return.
      Adaptive Ridge — Rolling Ridge refit on TC-adjusted returns per scenario;
                       the allocator shifts weight away from costly assets.

    The printed table adds a "gross" row to both methods as the TC = 0 baseline.
    The chart shows the Adaptive Ridge lines (gross + 3 scenarios) since that is
    the production method.

    Parameters
    ----------
    scenarios   Dict mapping label → {asset_class: scenario_key}.
                Default: TC_SCENARIOS_ASSETS (best / base / worst).
    is_end      Last IS date.  Default: "2020-12-31".
    oos_start   First OOS date.  Default: "2021-01-01".
    lookback    Rolling Ridge window in months.  Default: 36.
    min_obs     Minimum months before first weight estimate.  Default: 24.
    vol_target  For MDD scaling and plot.  Default: 0.10.
    lam         Ridge regularisation.  Default: 0.01.
    plot        Save and show the cumulative-return chart.  Default: True.
    verbose     Print the summary table.  Default: True.

    Returns
    -------
    pd.Series — base-scenario Adaptive Ridge net-of-TC monthly return series,
    name = "asset_blend_net".
    """
    import contextlib, io as _io

    if scenarios is None:
        scenarios = TC_SCENARIOS_ASSETS

    # ── 1. Gross baseline (TC = 0, Rolling Ridge) ─────────────────────────────
    if verbose:
        print("Building gross (TC-free) baseline …")
    with contextlib.redirect_stdout(_io.StringIO()):
        r_gross = run_asset_blend(
            is_end=is_end, lookback=lookback, min_obs=min_obs,
            lam=lam, vol_target=vol_target, plot=False, verbose=False,
        )

    r_gross_is  = r_gross.loc[r_gross.index <= is_end]
    r_gross_oos = r_gross.loc[r_gross.index >= oos_start]

    # ── 2. TC sweep ───────────────────────────────────────────────────────────
    if verbose:
        print("Running TC sweep …")
    with contextlib.redirect_stdout(_io.StringIO()):
        tc_res = run_assets_tc(
            scenarios=scenarios, is_end=is_end, oos_start=oos_start,
            lookback=lookback, min_obs=min_obs, vol_target=vol_target, lam=lam,
        )

    metrics_df  = tc_res["metrics"]
    returns_df  = tc_res["returns"]
    to_df       = tc_res["to_df"]
    w_is_fixed  = tc_res["w_is_fixed"]
    w_oos_fixed = tc_res["w_oos_fixed"]

    # ── 3. Build gross metrics row ────────────────────────────────────────────
    # Gross uses the same Rolling Ridge blend weights as the Fixed-weights baseline.
    gross_rows = []
    for period, r, w in [
        ("IS",  r_gross_is,  w_is_fixed),
        ("OOS", r_gross_oos, w_oos_fixed),
    ]:
        gross_to = _portfolio_to(w, to_df, r.index)
        for method in ["Fixed weights", "Adaptive Ridge"]:
            gross_rows.append({
                "Method":   method,
                "Scenario": "gross",
                "Period":   period,
                "Sharpe":   _sharpe(r),
                "MDD_pct":  _mdd(r, vol_target) * 100,
                "TO":       gross_to,
            })

    gross_df  = pd.DataFrame(gross_rows)
    all_metrics = pd.concat([gross_df, metrics_df], ignore_index=True)

    # ── 4. Print combined table ───────────────────────────────────────────────
    if verbose:
        scen_order = ["gross"] + list(scenarios.keys())
        W = 80
        for method in ["Fixed weights", "Adaptive Ridge"]:
            label = (
                "FIXED WEIGHTS — TC-free Ridge fit; only cost propagated"
                if method == "Fixed weights"
                else "ADAPTIVE RIDGE — Ridge refit per TC scenario  [production]"
            )
            sub = all_metrics[all_metrics["Method"] == method]
            print("\n" + "=" * W)
            print(f"  CROSS-ASSET TC ANALYSIS — {label}")
            print(f"  IS ≤ {is_end}  |  OOS ≥ {oos_start}"
                  f"  |  lookback={lookback}m  |  λ={lam}")
            print("=" * W)
            print(f"  {'Scenario':<16}  {'Period':<6}"
                  f"  {'Sharpe':>8}  {'MDD':>8}  {'TO':>7}")
            print("  " + "─" * (W - 2))
            prev = None
            for scen in scen_order:
                rows = sub[sub["Scenario"] == scen]
                if rows.empty:
                    continue
                if prev is not None:
                    print("  " + "─" * (W - 2))
                prev = scen
                for _, row in rows.iterrows():
                    tag = " [prod]" if scen == "base" else ""
                    label_scen = f"{scen}{tag}"
                    sr  = f"{row['Sharpe']:+.2f}"  if np.isfinite(row["Sharpe"])  else "   n/a"
                    mdd = f"{row['MDD_pct']:.1f}%"  if np.isfinite(row["MDD_pct"]) else "   n/a"
                    to  = f"{row['TO']:.2f}x"       if np.isfinite(row["TO"])      else "    —"
                    print(f"  {label_scen:<16}  {row['Period']:<6}"
                          f"  {sr:>8}  {mdd:>8}  {to:>7}")
            print("=" * W)

        # Sharpe degradation summary
        print()
        print("  Sharpe degradation  (Adaptive Ridge, OOS)")
        print("  " + "─" * 44)
        gross_oos_sr = all_metrics.loc[
            (all_metrics["Method"] == "Adaptive Ridge") &
            (all_metrics["Scenario"] == "gross") &
            (all_metrics["Period"] == "OOS"), "Sharpe"
        ].values[0]
        for scen in list(scenarios.keys()):
            net_sr = all_metrics.loc[
                (all_metrics["Method"] == "Adaptive Ridge") &
                (all_metrics["Scenario"] == scen) &
                (all_metrics["Period"] == "OOS"), "Sharpe"
            ]
            if net_sr.empty:
                continue
            tag = " ← production" if scen == "base" else ""
            diff = net_sr.values[0] - gross_oos_sr
            print(f"  {scen:<8}  {net_sr.values[0]:+.2f}  "
                  f"(Δ {diff:+.2f} vs gross){tag}")
        print("  " + "─" * 44)

    # ── 5. Plot ───────────────────────────────────────────────────────────────
    if plot:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            is_split = pd.Timestamp(is_end)
            scen_styles = {
                "gross": ("#1a237e", 2.0, "-",  "gross (no TC)"),
                "best":  ("#43a047", 1.4, "--", "best TC"),
                "base":  ("#e65100", 1.8, "-",  "base TC  [production]"),
                "worst": ("#b71c1c", 1.4, ":",  "worst TC"),
            }

            fig, ax = plt.subplots(figsize=(12, 6))

            all_cums = []
            for scen, (color, lw, ls, label_s) in scen_styles.items():
                if scen == "gross":
                    r_s = r_gross
                else:
                    key = ("Adaptive Ridge", scen)
                    if key not in returns_df.columns:
                        continue
                    r_s = returns_df[key].dropna()

                # Scale each series by its own IS vol so all lines target the
                # same annual vol.  Using a single gross IS scale distorts TC
                # scenario lines (their IS vol differs from gross IS vol).
                r_s_is   = r_s.loc[r_s.index <= is_end]
                ann_vol_s = r_s_is.std() * np.sqrt(12)
                scale_s   = vol_target / ann_vol_s if ann_vol_s > 1e-12 else 1.0

                r_scaled = r_s * scale_s
                cum = (1 + r_scaled).cumprod()
                all_cums.append(cum)
                ax.plot(cum.index, cum.values,
                        color=color, lw=lw, ls=ls, label=label_s, zorder=2)

            # IS shading
            if all_cums:
                y_min = min(c.min() for c in all_cums) * 0.95
                y_max = max(c.max() for c in all_cums) * 1.05
                ax.fill_betweenx([y_min, y_max], all_cums[0].index[0], is_split,
                                 color="#f5f5f5", zorder=0)
                ax.set_ylim(y_min, y_max)

            ax.axvline(is_split, color="gray", lw=1.0, ls="--", zorder=1)
            ax.text(0.02, 0.97, "IS",  transform=ax.transAxes,
                    fontsize=9, color="gray", va="top")
            ax.text(0.72, 0.97, "OOS", transform=ax.transAxes,
                    fontsize=9, color="gray", va="top")

            # Annotate OOS Sharpe for each scenario on the right margin
            for scen, (color, _, _, _) in scen_styles.items():
                row = all_metrics.loc[
                    (all_metrics["Method"] == "Adaptive Ridge") &
                    (all_metrics["Scenario"] == scen) &
                    (all_metrics["Period"] == "OOS")
                ]
                if not row.empty and np.isfinite(row["Sharpe"].values[0]):
                    ax.annotate(
                        f"SR {row['Sharpe'].values[0]:+.2f}",
                        xy=(1.0, 0.0), xycoords="axes fraction",
                        xytext=(5, list(scen_styles).index(scen) * 14),
                        textcoords="offset points",
                        fontsize=7.5, color=color, ha="left",
                    )

            ax.set_ylabel("Wealth index", fontsize=10)
            ax.set_title(
                f"Cross-Asset Blend — TC Sensitivity  "
                f"(Adaptive Ridge, lookback={lookback}m, λ={lam}, "
                f"vol target={vol_target*100:.0f}%)",
                fontsize=10,
            )
            ax.legend(loc="upper left", fontsize=8)
            ax.yaxis.grid(True, ls=":", lw=0.5, color="#cccccc")
            ax.set_axisbelow(True)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.xaxis.set_major_locator(mdates.YearLocator(2))

            fig.tight_layout()
            out_path = Path(__file__).parent / "asset_tc_composite.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            if verbose:
                print(f"\n  Chart saved → {out_path}")
            plt.show()

        except Exception as e:
            if verbose:
                print(f"\n  [plot skipped: {e}]")

    # ── 6. Return base-scenario Adaptive Ridge series ─────────────────────────
    base_key = ("Adaptive Ridge", "base")
    r_base = returns_df[base_key].dropna() if base_key in returns_df.columns else r_gross
    r_base.name = "asset_blend_net"
    return r_base
