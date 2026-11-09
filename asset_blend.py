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
                f"IS Sharpe {is_sr:+.2f}  MDD {is_mdd*100:+.1f}%  TO {is_to:.1f}x    "
                f"OOS Sharpe {oos_sr:+.2f}  MDD {oos_mdd*100:+.1f}%  TO {oos_to:.1f}x",
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

            # ── Bottom panel: time-varying weights ────────────────────────────
            w_plot = w_all.dropna()
            for n in names:
                ax2.plot(
                    w_plot.index, w_plot[n].values,
                    color=colors.get(n, "black"), lw=1.4, label=n,
                )
            ax2.axhline(0, color="black", lw=0.6, ls="-")
            ax2.axvline(is_split, color="gray", lw=1.0, ls="--")
            ax2.fill_betweenx(
                [w_plot.values.min() - 0.05, w_plot.values.max() + 0.05],
                w_plot.index[0], is_split,
                color="#f5f5f5", zorder=0,
            )
            ax2.set_ylabel("Blend weight", fontsize=10)
            ax2.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{x:.0%}")
            )
            ax2.legend(loc="upper right", fontsize=8, ncol=2)
            ax2.yaxis.grid(True, ls=":", lw=0.5, color="#cccccc")
            ax2.set_axisbelow(True)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax2.xaxis.set_major_locator(mdates.YearLocator(2))

            fig.tight_layout()
            out_path = Path(__file__).parent / "asset_blend_composite.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            if verbose:
                print(f"\n  Chart saved → {out_path}")
            plt.show()

        except Exception as e:
            if verbose:
                print(f"\n  [plot skipped: {e}]")

    return r_full
