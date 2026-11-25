"""
Commodities signal blend layer.

Loads pre-computed vol-scaled managed-return parquets, and exposes the same
blend API as equities/blend.py (all algorithms are re-exported).

Pipeline position:
    commodities/signals/*_pnl.parquet  →  load_managed_returns()  →  F (T × 4)
    F  →  fit_static / MLMSRR / LinearDynamicBlend / …  →  blend weights
    blend weights  →  combined portfolio return

Signals (production, from commodities/signals/):
    tsmom_252        tsmom_252_pnl.parquet        TS momentum 252-day (sign)
    xs_carry         xs_carry_pnl.parquet          XS tenor-normalised F1/F2 carry
    xs_carry_enh     xs_carry_enh_pnl.parquet      XS carry + seasonal/sector/roll
    curvature        curvature_pnl.parquet         NS butterfly (F1+F4-2*F2)

Managed-return convention: for each signal k and commodity i,
    w_k[i,t] = signal_k[i,t] × target_vol / sigma_hat[i,t]   (vol-targeted)
    F_k[t]   = Σ_i  w_k[i,t-1] × r[i,t]                     (weekly W-FRI rebalance)

The NS null-space butterfly (Signal B) is excluded: its daily vol is ~10× lower
than directional signals; the vol-scaling step would apply ~10× leverage, making
its managed return non-comparable and its blend weight trivially dominant.

Parquets are pre-computed by build_signal_parquets(); loading them takes ~1 ms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Re-export every blend algorithm from the equities layer ──────────────────
from core.asset_blend import _drift_to
from equities.blend import (                                        # noqa: F401
    BlendWeights,
    equal_weight,
    inv_vol_blend,
    erc_blend,
    sharpe_heuristic,
    msrr_ridge,
    msrr_lasso,
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

# ── Commodities-specific constants ────────────────────────────────────────────

IS_END    = "2020-12-31"
OOS_START = "2021-01-01"

_TC_ANNUAL_DRAG_COMM: dict[str, float] = {
    "best":   2.0,   # bps/yr — est. 1 bp spread × ~2x annual instrument TO
    "base":  12.0,   # bps/yr — est. 3 bp spread × ~4x annual instrument TO
    "worst": 56.0,   # bps/yr — est. 7 bp spread × ~8x annual instrument TO
}

# Per-instrument half-spread scenarios for tc_sweep_comm_actual.
# TC per signal is computed as Σ_i (spread_i × |Δw_eff_k[i,t]|) from committed parquets.
# Structure mirrors TC_SCENARIOS (rates) and TC_SCENARIOS_FX (FX).
_TC_DEFAULT_COMM: float = 5.0   # fallback half-spread (bps) for unlisted instruments

TC_SCENARIOS_COMM: dict[str, dict[str, float]] = {
    "best": {
        # Energy
        "CL":  0.5,   # WTI Crude — most liquid commodity futures
        "XB":  1.0,   # RBOB Gasoline
        "HO":  1.0,   # Heating Oil
        # Metals
        "GC":  0.25,  # Gold — deepest metals market
        "HG":  0.5,   # Copper
        "SI":  1.0,   # Silver
        # Grains
        "C":   0.5,   # Corn
        "S":   0.5,   # Soybeans
        "W":   1.0,   # CBOT Wheat
        "SM":  1.0,   # Soybean Meal
        "BO":  1.0,   # Soybean Oil
        "O":   3.0,   # Oats — thin CBOT market
        "RR":  5.0,   # Rough Rice — very thin
        # Softs
        "KC":  2.0,   # Coffee
        "CC":  2.0,   # Cocoa
        "CT":  1.5,   # Cotton
        "JO":  3.0,   # OJ Futures — thin ICE market
        "LB":  6.0,   # Lumber — structurally illiquid
        # Livestock
        "LC":  1.5,   # Live Cattle
        "LHD": 1.5,   # Lean Hogs
        "FC":  2.5,   # Feeder Cattle — less liquid than LC
    },
    "base": {
        "CL":  1.5,  "XB":  3.0,  "HO":  3.0,
        "GC":  1.0,  "HG":  2.0,  "SI":  3.5,
        "C":   2.0,  "S":   2.0,  "W":   3.5,  "SM":  3.5,  "BO":  3.5,
        "O":   8.0,  "RR": 12.0,
        "KC":  5.0,  "CC":  6.0,  "CT":  4.0,  "JO":  8.0,  "LB": 15.0,
        "LC":  4.0,  "LHD": 4.0,  "FC":  7.0,
    },
    "worst": {
        "CL":  4.0,  "XB":  8.0,  "HO":  8.0,
        "GC":  3.0,  "HG":  5.0,  "SI": 10.0,
        "C":   5.0,  "S":   5.0,  "W":  10.0,  "SM": 10.0,  "BO": 10.0,
        "O":  20.0,  "RR": 28.0,
        "KC": 15.0,  "CC": 18.0,  "CT": 12.0,  "JO": 20.0,  "LB": 35.0,
        "LC": 10.0,  "LHD": 10.0, "FC": 18.0,
    },
}

_SIGNALS_DIR = Path(__file__).parent / "signals"

# Cross-instrument aggregation applied within each signal when building the
# managed-return parquets.  "equal" sums the per-instrument managed returns;
# "static_ridge" fits ridge MSRR once on IS instrument returns and freezes the weights.
CROSS_INSTRUMENT_BLEND: dict[str, str] = {
    "tsmom_252":    "inv_vol",
    "xs_carry_enh": "inv_vol",
    "curvature":    "inv_vol",
}
_CROSS_INSTR_MIN_OBS: int = 252   # min IS observations for an instrument to be included

SIGNAL_FILES: dict[str, str] = {
    "tsmom_252":    "tsmom_252_pnl.parquet",
    "xs_carry_enh": "xs_carry_enh_pnl.parquet",
    "curvature":    "curvature_pnl.parquet",
}

# Production pipeline parameters (must match those used in build_signal_parquets).
# Subset of backtest.PARAMS relevant to the managed-return computation.
BLEND_PARAMS = dict(
    asset_vol = 0.10,
    vol_com   = 60,
    rebal     = "W-FRI",
    lag       = 1,
    mode      = "sign",
    clip      = 2.0,
)


# ── Cross-instrument aggregation ──────────────────────────────────────────────

def _instrument_multipliers(
    F_i: pd.DataFrame,
    method: str,
    is_end: str,
    min_obs: int = _CROSS_INSTR_MIN_OBS,
) -> pd.Series:
    """
    Per-instrument multipliers v_i applied to the vol-scaled instrument weights,
    so that the signal's managed return is Σ_i v_i · F_i.

    "equal"   plain sum across instruments.
    "inv_vol" w_i ∝ 1/σ_i (IS σ), frozen and applied to the full period.
    """
    if method == "equal":
        return pd.Series(1.0, index=F_i.columns)
    if method != "inv_vol":
        raise ValueError(f"unknown cross-instrument method: {method!r}")

    F_is = F_i.loc[:is_end]
    keep = F_is.count()[lambda c: c >= min_obs].index
    sigma = F_is[keep].std()
    keep = sigma[sigma > 1e-12].index
    sigma = sigma[keep]
    if sigma.empty:
        return pd.Series(1.0, index=F_i.columns)
    w = 1.0 / sigma
    w = w / w.sum()
    # Rescale to the same gross as equal-weight sum (len(keep)) for TC comparability
    v = w * len(keep)
    return v.reindex(F_i.columns).fillna(0.0)


def _aggregate_instruments(
    F_i: pd.DataFrame,
    method: str,
    is_end: str,
    min_obs: int = _CROSS_INSTR_MIN_OBS,
) -> pd.Series:
    """Collapse per-instrument managed returns (T × N) into one series Σ_i v_i · F_i."""
    v = _instrument_multipliers(F_i, method, is_end, min_obs)
    return (F_i.fillna(0.0) * v).sum(axis=1)


# ── One-time parquet builder ──────────────────────────────────────────────────

def _signal_panels(
    params: dict | None = None,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Per-signal instrument panels: {signal: (w, F_i)}.

    w   vol-scaled, rebalanced instrument weights (T × N, before lag)
    F_i per-instrument managed returns w[t-1] × r[t] (T × N, all-NaN columns dropped)
    """
    from commodities.production.data import fetch_prices, fetch_term_structure, fetch_tickers
    from commodities.production.backtest import (
        specific_contract_returns, vol_scale, rebalance, _build_signals,
    )

    p = params or BLEND_PARAMS

    print("Loading Bloomberg data …")
    prices  = fetch_prices()
    ts      = fetch_term_structure()
    tk      = fetch_tickers()

    returns = specific_contract_returns(ts, tk).reindex(columns=prices.columns)
    raw_signals = _build_signals(prices, ts, tk, p)

    # Map from _build_signals keys → SIGNAL_FILES keys
    _key_map = {
        "tsmom_252":         "tsmom_252",
        "xs_carry_enh_f1f2": "xs_carry_enh",
        "curvature":         "curvature",
    }

    panels: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for raw_key, blend_key in _key_map.items():
        if raw_key not in raw_signals:
            print(f"  Skipping {raw_key} (not available)")
            continue
        sig = raw_signals[raw_key].reindex(index=returns.index, columns=returns.columns)
        w = vol_scale(sig, returns, target_vol=p["asset_vol"], com=p["vol_com"])
        w = rebalance(w, rule=p["rebal"])
        # Managed return: w[t-1] × r[t], zero rows dropped
        F_i = (w.shift(p["lag"]) * returns).dropna(how="all", axis=1)
        panels[blend_key] = (w, F_i)
    return panels


def build_signal_parquets(
    output_dir: str | Path | None = None,
    params: dict | None = None,
    is_end: str = IS_END,
    panels: dict | None = None,
) -> None:
    """
    Compute vol-scaled managed returns for all production signals and save
    them to parquet.  Instruments are aggregated per CROSS_INSTRUMENT_BLEND
    (ridge weights are fit on data up to `is_end`).  Call once; subsequent load_managed_returns() calls read
    from disk in ~1 ms.

    Parameters
    ----------
    output_dir  Directory for parquet output.  Default: commodities/signals/.
    params      Pipeline parameters.  Default: BLEND_PARAMS.
    panels      Pre-computed _signal_panels() output; computed here if None.
    """
    out_dir = Path(output_dir) if output_dir else _SIGNALS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if panels is None:
        panels = _signal_panels(params)

    print("Computing managed returns and instrument positions …")
    w_eff_all: dict[str, pd.DataFrame] = {}
    for blend_key, (w, F_i) in panels.items():
        F_k = _aggregate_instruments(F_i, CROSS_INSTRUMENT_BLEND[blend_key], is_end)
        F_k = F_k.replace(0.0, np.nan).dropna()
        F_k.index.name = "date"
        F_k.name = blend_key
        path = out_dir / SIGNAL_FILES[blend_key]
        F_k.to_frame().to_parquet(path)
        sr = F_k.mean() / F_k.std() * np.sqrt(252)
        print(f"  {blend_key:15s}  SR={sr:+.3f}  rows={len(F_k)}  → {path.name}")

        v = _instrument_multipliers(F_i, CROSS_INSTRUMENT_BLEND[blend_key], is_end)
        w_eff = w.mul(v.reindex(w.columns).fillna(0.0), axis=1)
        w_eff.index.name = "date"
        w_eff.to_parquet(out_dir / f"{blend_key}_positions.parquet")
        w_eff_all[blend_key] = w_eff
        print(f"  {blend_key:15s}  positions → {blend_key}_positions.parquet")

    # Combined instrument-level position (blend-weighted)
    res = run_comm_blend(signals_dir=out_dir, is_end=is_end)
    w_is, w_oos = res["weights_is"], res["weights_oos"]
    alpha_full = pd.concat([w_is, w_oos]).sort_index()
    F_raw   = load_managed_returns(signals_dir=out_dir)
    sigma_k = F_raw.loc[F_raw.index <= is_end].std() * np.sqrt(252)
    names   = list(w_is.columns)
    all_idx  = w_eff_all[names[0]].index
    all_cols = w_eff_all[names[0]].columns
    a_df    = alpha_full.reindex(all_idx).ffill().fillna(0.0)
    a_norm  = a_df.div(sigma_k.reindex(a_df.columns), axis=1).fillna(0.0)
    W_blend = pd.DataFrame(0.0, index=all_idx, columns=all_cols)
    for k in names:
        if k in w_eff_all:
            W_blend += (w_eff_all[k]
                        .reindex(index=all_idx, columns=all_cols).fillna(0.0)
                        .mul(a_norm[k], axis=0))
    W_blend.index.name = "date"
    W_blend.to_parquet(out_dir / "combined_positions.parquet")
    print(f"  combined         positions → combined_positions.parquet")

    print("Done.")


# ── Data loading ─────────────────────────────────────────────────────────────

def load_managed_returns(
    signals: dict[str, str] | None = None,
    signals_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load commodities managed-return parquets and return F (T × K).

    Parameters
    ----------
    signals     Map of signal_name → parquet filename.  Default: SIGNAL_FILES.
    signals_dir Directory containing the parquet files.  Default: commodities/signals/.

    Returns
    -------
    pd.DataFrame, shape (T, K), index=date, columns=signal names.
    Only dates where all signals are available are returned (inner join).

    Raises
    ------
    FileNotFoundError if parquets are missing.  Run build_signal_parquets() first.
    """
    if signals is None:
        signals = SIGNAL_FILES
    sig_dir = Path(signals_dir) if signals_dir else _SIGNALS_DIR

    series: dict[str, pd.Series] = {}
    for name, fname in signals.items():
        path = sig_dir / fname
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found.  Run commodities.blend.build_signal_parquets() first."
            )
        df = pd.read_parquet(path)
        series[name] = df.iloc[:, 0]

    F = pd.DataFrame(series)
    F.index.name = "date"
    F = F.sort_index().dropna(how="any")
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


# ── Rolling OLS blend ─────────────────────────────────────────────────────────

def _expanding_ols_weights(F: pd.DataFrame, min_obs: int = 252) -> pd.DataFrame:
    """
    Expanding-window MSRR OLS weights (causal).

    At row t, the weight is fitted on F.iloc[:t] (all rows strictly before t),
    so applying it directly to F.iloc[t] is look-ahead free.
    Rows 0..min_obs-1 are NaN (insufficient history).
    """
    vals = F.values.astype(float)
    T, K = vals.shape
    out = np.full((T, K), np.nan)
    for t in range(min_obs, T):
        Ft = vals[:t]
        ones_t = np.ones(t)
        b, _, _, _ = np.linalg.lstsq(Ft, ones_t, rcond=None)
        out[t] = b
    return pd.DataFrame(out, index=F.index, columns=F.columns)


def run_comm_blend(
    signals_dir: str | Path | None = None,
    is_end: str = IS_END,
    lam: float = 0.01,
) -> dict:
    """
    Canonical commodities blend pipeline: daily managed returns → static Ridge MSRR.

    Vol-normalises to unit annual vol using IS σ only (no look-ahead), then fits
    a static Ridge MSRR blend on IS data and applies the fixed weights to both IS
    and OOS.

    Parameters
    ----------
    signals_dir   Directory with parquet files.  Default: commodities/signals/.
    is_end        Last IS date.  Default: IS_END ("2020-12-31").
    lam           Ridge regularisation.  Default: 0.01.

    Returns
    -------
    dict:
        F_is        vol-normalised IS managed returns  (T_is × 3)
        F_oos       vol-normalised OOS managed returns (T_oos × 3)
        weights_is  pd.DataFrame (T_is × 3) constant Ridge weights, IS slice
        weights_oos pd.DataFrame (T_oos × 3) constant Ridge weights, OOS slice
        r_is        combined IS return series
        r_oos       combined OOS return series
    """
    F = load_managed_returns(signals_dir=signals_dir)
    F_is_raw  = F.loc[F.index <= is_end]
    F_oos_raw = F.loc[F.index >  is_end]

    sigma = F_is_raw.std() * np.sqrt(252)
    sigma = sigma.replace(0.0, np.nan)
    F_is  = F_is_raw.div(sigma)
    F_oos = F_oos_raw.div(sigma)

    # Static Ridge MSRR: fit on IS, apply constant weights to both periods
    bw        = msrr_ridge(F_is, lam=lam)
    w_static  = bw.weights / bw.weights.abs().sum()
    w_is  = pd.DataFrame(
        np.tile(w_static.reindex(F_is.columns).values,  (len(F_is),  1)),
        index=F_is.index,  columns=F_is.columns,
    )
    w_oos = pd.DataFrame(
        np.tile(w_static.reindex(F_oos.columns).values, (len(F_oos), 1)),
        index=F_oos.index, columns=F_oos.columns,
    )

    r_is  = (F_is  * w_is).sum(axis=1).rename("comm_static_ridge")
    r_oos = (F_oos * w_oos).sum(axis=1).rename("comm_static_ridge")

    return {
        "F_is":        F_is,
        "F_oos":       F_oos,
        "weights_is":  w_is,
        "weights_oos": w_oos,
        "r_is":        r_is,
        "r_oos":       r_oos,
    }


def run_commodities_blend(
    signals_dir: str | Path | None = None,
    is_end: str = IS_END,
    min_obs: int = 252,
    rebuild_signals: bool = False,
) -> dict:
    """
    Full commodities pipeline: build signals → static Ridge MSRR blend.

    Step 1 — Signal pipeline (commodities/blend.build_signal_parquets):
        Computes vol-scaled managed returns for tsmom_252, xs_carry_enh, and
        curvature and writes them to commodities/signals/.  Skipped when
        rebuild_signals=False and all three parquets already exist.

    Step 2 — Blend (run_comm_blend):
        Loads the three parquets, vol-normalises on IS σ, fits a static Ridge
        MSRR blend on IS data and applies fixed weights to both IS and OOS.

    Parameters
    ----------
    signals_dir       Parquet directory.  Default: commodities/signals/.
    is_end            Last IS date.  Default: IS_END ("2020-12-31").
    min_obs           Unused (kept for interface compatibility).
    rebuild_signals   Force re-run of the signal pipeline.  Default: False.

    Returns
    -------
    Same dict as run_comm_blend().
    """
    sig_dir   = Path(signals_dir) if signals_dir else _SIGNALS_DIR
    pnl_paths = [sig_dir / f for f in SIGNAL_FILES.values()]
    needs_build = rebuild_signals or not all(p.exists() for p in pnl_paths)

    if needs_build:
        missing = [p.name for p in pnl_paths if not p.exists()]
        if missing:
            print(f"Missing parquets: {missing} — running signal pipeline...")
        else:
            print("rebuild_signals=True — re-running signal pipeline...")
        build_signal_parquets(output_dir=sig_dir)
    else:
        print("Signal parquets found — skipping rebuild. Pass rebuild_signals=True to refresh.")

    return run_comm_blend(signals_dir=sig_dir, is_end=is_end)


# ── Production entry point ────────────────────────────────────────────────────

def run_comm_prod(
    is_end: str = IS_END,
    min_obs: int = 252,
    rebuild_signals: bool = False,
    vol_target: float = 0.10,
) -> pd.Series:
    """
    Production commodities pipeline: 3 signals → static ridge MSRR blend.

    Loads (or rebuilds) signal parquets, fits a static ridge MSRR blend on IS,
    prints IS / OOS Sharpe, MDD, and annual instrument-level turnover, and
    returns the full daily return series (IS concatenated with OOS).

    Parameters
    ----------
    is_end            Last IS date.  Default: IS_END ("2020-12-31").
    params            Pipeline parameters.  Default: commodities/backtest.PARAMS.
    min_obs           Warmup before blend weights are non-NaN.  Default: 252.
    rebuild_signals   Force rebuild of signal parquets.  Default: False.

    Returns
    -------
    pd.Series — daily return series named "commodities", IS then OOS.
    """
    TD = 252

    # ── 1. Blend ──────────────────────────────────────────────────────────────
    res   = run_commodities_blend(is_end=is_end, min_obs=min_obs,
                                  rebuild_signals=rebuild_signals)
    r_is  = res["r_is"]
    r_oos = res["r_oos"]
    w_is  = res["weights_is"]   # T_is × 3  (L1-normalised blend weights)
    w_oos = res["weights_oos"]  # T_oos × 3
    names = list(w_is.columns)

    # ── 2. Load instrument-level combined positions from committed parquet ─────
    _combined_path = _SIGNALS_DIR / "combined_positions.parquet"
    if not _combined_path.exists():
        raise FileNotFoundError(
            f"{_combined_path} not found. "
            "Run commodities.blend.build_signal_parquets() to generate it."
        )
    W_blend  = pd.read_parquet(_combined_path)
    all_idx  = W_blend.index
    ts_cut   = pd.Timestamp(is_end)

    def _inst_to(w: pd.DataFrame, freq: int) -> float:
        gross = w.abs().sum(axis=1).replace(0.0, np.nan)
        delta = w.diff().abs().sum(axis=1)
        return float((delta / gross).dropna().mean() * freq)

    to_is  = _inst_to(W_blend.loc[W_blend.index <= ts_cut], TD)
    to_oos = _inst_to(W_blend.loc[W_blend.index >  ts_cut], TD)

    # ── 4. Performance metrics ────────────────────────────────────────────────
    def _sr(r: pd.Series) -> float:
        r = r.dropna()
        return float(r.mean() / r.std() * np.sqrt(TD)) if len(r) > 60 else float("nan")

    def _mdd(r: pd.Series) -> float:
        r = r.dropna()
        if r.empty or r.std() < 1e-12:
            return float("nan")
        ann_vol = r.std() * np.sqrt(TD)
        cum = (1 + r * (vol_target / ann_vol)).cumprod()
        return float((cum / cum.cummax() - 1).min())

    # ── 5. Print ──────────────────────────────────────────────────────────────
    W = 72
    print("=" * W)
    print(f"  COMMODITIES PRODUCTION  —  Static Ridge MSRR  (vol target={vol_target*100:.0f}%)")
    print(f"  IS ≤ {is_end}   |   OOS > {is_end}")
    print("=" * W)
    print(f"  {'':20}  {'SR':>7}  {'MDD':>8}  {'TO (ann)':>9}")
    print(f"  {'─'*20}  {'─'*7}  {'─'*8}  {'─'*9}")
    for label, r, to in [("IS", r_is, to_is), ("OOS", r_oos, to_oos)]:
        sr  = _sr(r)
        mdd = _mdd(r)
        print(f"  {label:<20}  {sr:>+7.2f}  {mdd*100:>+7.1f}%  {to:>8.1f}x")
    print("=" * W)
    print()
    print("  Avg blend weights (IS):")
    for col in names:
        print(f"    {col:<20}  {w_is[col].mean():>+.4f}")
    print()

    # ── 6. Return series ──────────────────────────────────────────────────────
    r_full = pd.concat([r_is, r_oos]).sort_index()
    r_full.name = "commodities"
    return r_full


def export_signals(
    output_dir: "str | Path | None" = None,
    params: "dict | None" = None,
    is_end: str = IS_END,
    oos_start: str = OOS_START,
) -> None:
    """
    Build commodities signal parquets and write manifest.json to commodities/signals/.

    Thin wrapper around build_signal_parquets() that also records provenance.
    Call once with Bloomberg data present; downstream layers read parquets only.
    """
    import datetime
    import json
    import subprocess

    out_dir = Path(output_dir) if output_dir else _SIGNALS_DIR
    build_signal_parquets(output_dir=out_dir, params=params, is_end=is_end)

    # Derive universe and date range from the written managed-return parquets
    F = load_managed_returns(signals_dir=out_dir)
    start_str = str(F.index[0].date()) if not F.empty else None
    end_str   = str(F.index[-1].date()) if not F.empty else None

    _sha_r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True)

    # Read one per-signal positions file to get instrument universe
    pos_path = out_dir / f"{list(SIGNAL_FILES)[0]}_positions.parquet"
    universe: list = []
    if pos_path.exists():
        import pandas as _pd
        universe = sorted(_pd.read_parquet(pos_path).columns.tolist())

    manifest = {
        "universe":  universe,
        "start":     start_str,
        "end":       end_str,
        "is_end":    is_end,
        "oos_start": oos_start,
        "params":    params or BLEND_PARAMS,
        "files": {
            k: v for k, v in {
                "tsmom_252_pnl":        "date × 1; aggregated managed return (daily)",
                "xs_carry_enh_pnl":     "date × 1; aggregated managed return (daily)",
                "curvature_pnl":        "date × 1; aggregated managed return (daily)",
                "tsmom_252_positions":        "date × instrument; vol-scaled eff. weight panel",
                "xs_carry_enh_positions":     "date × instrument; vol-scaled eff. weight panel",
                "curvature_positions":        "date × instrument; vol-scaled eff. weight panel",
                "combined_positions":   "date × instrument; blend-weighted combined position",
            }.items()
        },
        "git_sha":   _sha_r.stdout.strip() if _sha_r.returncode == 0 else "unknown",
        "built_at":  datetime.datetime.utcnow().isoformat() + "Z",
    }
    with open(out_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(f"  Wrote manifest.json")


from equities.blend import RollingMSRR as _RollingMSRR   # noqa: E402


def tc_sweep_comm(
    scenarios: Optional[dict] = None,
    signals_dir: str | Path | None = None,
    is_end: str = IS_END,
    oos_start: str = OOS_START,
    vol_target: float = 0.10,
    lam: float = 0.01,
    _collect_returns: bool = False,
) -> "pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]":
    """
    TC sensitivity sweep for the commodities blend (fixed static-Ridge weights).

    TC approximation: instrument-level position parquets are not available here
    (only pre-computed signal PnL parquets), so TC is modelled as a fixed annual
    drag applied equally to all signals.  Scenarios express estimated cost as
    spread × annual instrument-level TO (see _TC_ANNUAL_DRAG_COMM for defaults).

    Fixed weights (Static Ridge): IS-fitted static Ridge weights held constant
    across all scenarios; only the TC cost changes.  Shows pure TC impact.

    Parameters
    ----------
    scenarios       dict mapping name → annual TC drag in bps.  Default: _TC_ANNUAL_DRAG_COMM.
    signals_dir     Path to signal parquets.  Default: commodities/signals/.
    is_end          Last IS date (inclusive).  Default: IS_END.
    oos_start       First OOS date (inclusive).  Default: OOS_START.
    vol_target      Used only for MDD computation.  Default: 0.10.
    lam             Ridge penalty.  Default: 0.01.
    _collect_returns  Internal flag: when True returns (df, returns_df) tuple.

    Returns
    -------
    pd.DataFrame with columns [Method, Scenario, Period, Sharpe, MDD_pct, TO]
    or (df, returns_df) when _collect_returns=True.
    """
    if scenarios is None:
        scenarios = _TC_ANNUAL_DRAG_COMM

    F = load_managed_returns(signals_dir=signals_dir)

    F_is_raw  = F.loc[F.index <= is_end]
    F_oos_raw = F.loc[F.index >= oos_start]

    # IS annualized vol — computed on TC-free raw returns, used for all scenarios
    sigma = F_is_raw.std() * np.sqrt(252)
    sigma = sigma.replace(0.0, np.nan)

    # TC-free vol-normalised managed returns
    F_is_free  = F_is_raw.div(sigma)
    F_oos_free = F_oos_raw.div(sigma)

    # Fit Static Ridge once on TC-free IS (fixed weights baseline)
    bw_free  = msrr_ridge(F_is_free, lam=lam)
    w_static = bw_free.weights / bw_free.weights.abs().sum()
    w_is_fixed = pd.DataFrame(
        np.tile(w_static.reindex(F_is_free.columns).values,  (len(F_is_free),  1)),
        index=F_is_free.index,  columns=F_is_free.columns,
    )
    w_oos_fixed = pd.DataFrame(
        np.tile(w_static.reindex(F_oos_free.columns).values, (len(F_oos_free), 1)),
        index=F_oos_free.index, columns=F_oos_free.columns,
    )

    def _sharpe(r: pd.Series) -> float:
        r = r.dropna()
        if len(r) < 60 or r.std() < 1e-12:
            return float("nan")
        return float(r.mean() / r.std() * np.sqrt(252))

    def _mdd(r: pd.Series) -> float:
        r = r.dropna()
        if r.empty or r.std() < 1e-12:
            return float("nan")
        ann_vol = r.std() * np.sqrt(252)
        cum = (1 + r * (vol_target / ann_vol)).cumprod()
        return float((cum / cum.cummax() - 1).min() * 100)

    def _blend_to(w_df: pd.DataFrame, F_n: pd.DataFrame) -> float:
        return _drift_to(w_df, F_n * vol_target, freq=252, l1_norm=True)

    records = []
    _returns_dict: dict[tuple, pd.Series] = {}

    for scenario_name, h_bps in scenarios.items():
        # Daily TC drag in raw managed-return units; annual cumulative = h_bps / 10_000
        tc_daily = h_bps / (252.0 * 10_000.0)

        F_is_tc  = F_is_raw  - tc_daily
        F_oos_tc = F_oos_raw - tc_daily

        # Normalise using the same TC-free IS sigma across all scenarios (no look-ahead)
        F_is_n  = F_is_tc.div(sigma)
        F_oos_n = F_oos_tc.div(sigma)
        # Cost is charged on |w| (a negative weight must not earn a rebate)
        c_n = tc_daily / sigma

        # ── Method 1: Fixed weights (Static Ridge) ────────────────────────────
        r_is_f  = (F_is_free  * w_is_fixed).sum(axis=1)  - (w_is_fixed.abs()  * c_n).sum(axis=1)
        r_oos_f = (F_oos_free * w_oos_fixed).sum(axis=1) - (w_oos_fixed.abs() * c_n).sum(axis=1)

        for period, r, w_f, F_f in [("IS", r_is_f, w_is_fixed, F_is_n), ("OOS", r_oos_f, w_oos_fixed, F_oos_n)]:
            records.append({
                "Method":   "Fixed weights",
                "Scenario": scenario_name,
                "Period":   period,
                "Sharpe":   _sharpe(r),
                "MDD_pct":  _mdd(r),
                "TO":       _blend_to(w_f, F_f),
            })

        if _collect_returns:
            _returns_dict[("Fixed weights", scenario_name)] = (
                pd.concat([r_is_f, r_oos_f]).sort_index()
            )

    df = pd.DataFrame(records)

    # ── Print table ───────────────────────────────────────────────────────────
    W = 82
    label = "FIXED WEIGHTS (Static Ridge, IS-fitted) — only TC cost deducted"
    print("\n" + "=" * W)
    print(f"  COMMODITIES TC SWEEP — {label}")
    print(f"  IS ≤ {is_end}   |   OOS ≥ {oos_start}   |   λ={lam}")
    print(f"  TC = estimated annual drag (bps/yr) = approx. instrument spread × annual TO")
    print("=" * W)
    print(
        f"  {'Scenario':<10}  {'TC (bps/yr)':>12}  {'Period':<6}"
        f"  {'Sharpe':>8}  {'MDD':>8}  {'Blend TO':>9}"
    )
    print("  " + "─" * (W - 2))
    prev = None
    for _, row in df.iterrows():
        if prev is not None and row["Scenario"] != prev:
            print("  " + "─" * (W - 2))
        prev = row["Scenario"]
        tc_v = scenarios[row["Scenario"]]
        sr   = f"{row['Sharpe']:+.2f}"   if not np.isnan(row["Sharpe"])  else "   n/a"
        mdd  = f"{row['MDD_pct']:.1f}%"  if not np.isnan(row["MDD_pct"]) else "   n/a"
        to_s = f"{row['TO']:.2f}x"       if not np.isnan(row["TO"])      else "   n/a"
        print(
            f"  {row['Scenario']:<10}  {tc_v:>12.1f}  {row['Period']:<6}"
            f"  {sr:>8}  {mdd:>8}  {to_s:>9}"
        )
    print("=" * W)

    if _collect_returns:
        returns_df = pd.DataFrame(_returns_dict)
        returns_df.columns = pd.MultiIndex.from_tuples(
            returns_df.columns, names=["Method", "Scenario"]
        )
        return df, returns_df
    return df


def run_comm_tc(
    scenarios: Optional[dict] = None,
    signals_dir: str | Path | None = None,
    is_end: str = IS_END,
    oos_start: str = OOS_START,
    vol_target: float = 0.10,
    lam: float = 0.01,
) -> dict:
    """
    Full commodities TC sweep: metrics table + daily return timeseries.

    Returns
    -------
    dict:
        'metrics'   pd.DataFrame — [Method, Scenario, Period, Sharpe, MDD_pct, TO]
        'returns'   pd.DataFrame — daily returns, MultiIndex columns (Method, Scenario)
    """
    df, returns_df = tc_sweep_comm(
        scenarios=scenarios, signals_dir=signals_dir, is_end=is_end,
        oos_start=oos_start,
        vol_target=vol_target, lam=lam, _collect_returns=True,
    )
    return {"metrics": df, "returns": returns_df}


def tc_sweep_comm_actual(
    scenarios: Optional[dict] = None,
    signals_dir: str | Path | None = None,
    is_end: str = IS_END,
    oos_start: str = OOS_START,
    vol_target: float = 0.10,
    lam: float = 0.01,
    _collect_returns: bool = False,
) -> "pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]":
    """
    TC sensitivity sweep using actual per-signal, per-instrument turnover.

    Unlike tc_sweep_comm(), which applies a uniform annual-drag estimate, this
    function loads the committed per-signal position parquets and computes TC as

        TC_k[t] = Σ_i ( spread_i × |w_eff_k[i,t] - w_eff_k[i,t-1]| )

    for each signal k and each day t, using per-instrument half-spreads.
    The TC is deducted from each signal's raw managed return:

        F_k_net[t] = F_k[t] - Σ_i ( spread_i × |Δw_eff_k[i,t]| )

    Vol-normalisation uses TC-gross IS σ (same denominator across scenarios, so
    blend weights are comparable across spread levels).  Ridge MSRR blend weights
    are re-fitted on TC-net IS returns for each scenario.

    Parameters
    ----------
    scenarios     dict mapping name → {instrument: half_spread_bps}.
                  Default: TC_SCENARIOS_COMM.  Instruments not in the dict
                  use _TC_DEFAULT_COMM (5 bps).
    signals_dir   Path to signal and position parquets.    Default: commodities/signals/.
    is_end        Last IS date (inclusive).                Default: IS_END.
    oos_start     First OOS date (inclusive).              Default: OOS_START.
    vol_target    Used only for MDD computation.           Default: 0.10.
    lam           Ridge penalty.                           Default: 0.01.

    Returns
    -------
    pd.DataFrame with columns [Scenario, Period, Sharpe, MDD_pct, Blend_TO,
                                Blended_TC_bps, TC_tsmom_bps, TC_carry_bps,
                                TC_curv_bps, Blend_w_tsmom, Blend_w_carry,
                                Blend_w_curv]
    When _collect_returns=True returns (df, returns_df) where returns_df has
    MultiIndex columns ("Fixed weights", scenario_name).
    """
    if scenarios is None:
        scenarios = TC_SCENARIOS_COMM

    sig_dir = Path(signals_dir) if signals_dir else _SIGNALS_DIR
    F = load_managed_returns(signals_dir=sig_dir)

    # Per-signal per-instrument absolute daily position changes: T × I
    dw: dict[str, pd.DataFrame] = {}
    for k in F.columns:
        pos_path = sig_dir / f"{k}_positions.parquet"
        if not pos_path.exists():
            raise FileNotFoundError(
                f"{pos_path} not found.  Run build_signal_parquets() first."
            )
        w_eff = pd.read_parquet(pos_path)
        dw[k] = w_eff.diff().abs().reindex(F.index).fillna(0.0)   # T × I

    F_is_raw  = F.loc[F.index <= is_end]
    F_oos_raw = F.loc[F.index >= oos_start]

    # TC-gross IS σ — fixed denominator across all scenarios (no look-ahead)
    sigma = F_is_raw.std() * np.sqrt(252)
    sigma = sigma.replace(0.0, np.nan)

    # Print header
    names = list(F.columns)
    _abbrev = {"tsmom_252": "tsmom", "xs_carry_enh": "carry", "curvature": "curv"}
    abbrevs = [_abbrev.get(n, n[:6]) for n in names]

    W = 90
    print("=" * W)
    print("  STAGE 4 — TC-AWARE SCENARIOS — Actual TC: Σ_i(spread_i × |Δw_eff_k[i,t]|)")
    print(f"  IS ≤ {is_end}   |   OOS ≥ {oos_start}   |   λ={lam}")
    print(f"  Signals: {names}")
    print("=" * W)

    print(f"\n  Per-signal annual TC (IS, bps/yr) by scenario:")
    hdr = f"  {'Scenario':<10}"
    for ab in abbrevs:
        hdr += f"  {ab+' TC':>12}"
    hdr += f"  {'Blended TC':>12}"
    print(hdr)
    print("  " + "─" * (W - 2))

    def _sharpe(r: pd.Series) -> float:
        r = r.dropna()
        if len(r) < 60 or r.std() < 1e-12:
            return float("nan")
        return float(r.mean() / r.std() * np.sqrt(252))

    def _mdd(r: pd.Series) -> float:
        r = r.dropna()
        if r.empty or r.std() < 1e-12:
            return float("nan")
        ann_vol = r.std() * np.sqrt(252)
        cum = (1 + r * (vol_target / ann_vol)).cumprod()
        return float((cum / cum.cummax() - 1).min() * 100)

    records = []
    _returns_dict: dict[tuple, pd.Series] = {}

    for scenario_name, spread_map in scenarios.items():

        # Build per-instrument spread vectors for each signal's position columns
        tc_is:  dict[str, pd.Series] = {}
        tc_oos: dict[str, pd.Series] = {}
        ann_tc: dict[str, float]     = {}
        for k in names:
            sv = pd.Series(
                {col: spread_map.get(col, _TC_DEFAULT_COMM) / 10_000.0
                 for col in dw[k].columns}
            )
            tc_is[k]  = dw[k].loc[dw[k].index <= is_end].multiply(sv).sum(axis=1)
            tc_oos[k] = dw[k].loc[dw[k].index >= oos_start].multiply(sv).sum(axis=1)
            ann_tc[k] = float(tc_is[k].mean() * 252 * 10_000)

        # TC-net managed returns
        F_is_net  = F_is_raw.copy()
        F_oos_net = F_oos_raw.copy()
        for k in names:
            F_is_net[k]  = F_is_raw[k]  - tc_is[k]
            F_oos_net[k] = F_oos_raw[k] - tc_oos[k]

        # Vol-normalise using TC-gross IS σ (fixed across scenarios)
        F_is_n  = F_is_net.div(sigma)
        F_oos_n = F_oos_net.div(sigma)

        # Fit Ridge MSRR on TC-net IS normalised returns
        bw = msrr_ridge(F_is_n, lam=lam)
        w_static = bw.weights / bw.weights.abs().sum()

        w_is = pd.DataFrame(
            np.tile(w_static.reindex(F_is_n.columns).values, (len(F_is_n), 1)),
            index=F_is_n.index, columns=F_is_n.columns,
        )
        w_oos = pd.DataFrame(
            np.tile(w_static.reindex(F_oos_n.columns).values, (len(F_oos_n), 1)),
            index=F_oos_n.index, columns=F_oos_n.columns,
        )

        r_is  = (F_is_n  * w_is).sum(axis=1)
        r_oos = (F_oos_n * w_oos).sum(axis=1)

        if _collect_returns:
            _returns_dict[("Fixed weights", scenario_name)] = (
                pd.concat([r_is, r_oos]).sort_index()
            )

        blended_tc = sum(abs(float(w_static[k])) * ann_tc[k] for k in names)

        # Print per-signal TC row
        line = f"  {scenario_name:<10}"
        for k in names:
            line += f"  {ann_tc[k]:>11.1f}b"
        line += f"  {blended_tc:>11.1f}b"
        print(line)

        blend_w = {f"Blend_w_{_abbrev.get(k, k[:6])}": float(w_static[k]) for k in names}
        tc_cols  = {f"TC_{_abbrev.get(k, k[:6])}_bps": ann_tc[k] for k in names}

        for period, r, w_df, F_n in [("IS", r_is, w_is, F_is_n), ("OOS", r_oos, w_oos, F_oos_n)]:
            rec = {
                "Scenario":       scenario_name,
                "Period":         period,
                "Sharpe":         _sharpe(r),
                "MDD_pct":        _mdd(r),
                "Blend_TO":       _drift_to(w_df, F_n * vol_target, freq=252, l1_norm=True),
                "Blended_TC_bps": blended_tc,
            }
            rec.update(tc_cols)
            rec.update(blend_w)
            records.append(rec)

    # Print metrics table
    print("\n" + "=" * W)
    print(f"  {'Scenario':<10}  {'Period':<6}  {'Sharpe':>8}  {'MDD':>8}  {'Blend TO':>9}  {'Blended TC':>11}")
    print("  " + "─" * (W - 2))
    prev = None
    for r in records:
        if prev is not None and r["Scenario"] != prev:
            print("  " + "─" * (W - 2))
        prev = r["Scenario"]
        sr  = f"{r['Sharpe']:+.2f}"   if not np.isnan(r["Sharpe"])  else "   n/a"
        mdd = f"{r['MDD_pct']:.1f}%"  if not np.isnan(r["MDD_pct"]) else "    n/a"
        to_ = f"{r['Blend_TO']:.2f}x" if not np.isnan(r["Blend_TO"]) else "   n/a"
        tc_ = f"{r['Blended_TC_bps']:.1f}b"
        print(
            f"  {r['Scenario']:<10}  {r['Period']:<6}"
            f"  {sr:>8}  {mdd:>8}  {to_:>9}  {tc_:>11}"
        )
    print("=" * W)

    # Print blend weights per scenario
    print(f"\n  Blend weights (IS-fitted Ridge, λ={lam}) per scenario:")
    print(f"  {'Scenario':<10}" + "".join(f"  {ab:>14}" for ab in abbrevs))
    print("  " + "─" * (10 + 16 * len(names)))
    seen = set()
    for r in records:
        if r["Scenario"] not in seen:
            seen.add(r["Scenario"])
            line = f"  {r['Scenario']:<10}"
            for k in names:
                line += f"  {r[f'Blend_w_{_abbrev.get(k, k[:6])}']: >+13.1%}"
            print(line)
    print("=" * W)

    df = pd.DataFrame(records)
    if _collect_returns:
        return df, pd.DataFrame(_returns_dict)
    return df


def compare_blend_methods(
    signals_dir: str | Path | None = None,
    is_end: str = IS_END,
    lookback: int = 252,
    min_obs: int = 252,
    ridge_lam: float = 0.01,
    vol_target: float = 0.10,
) -> pd.DataFrame:
    """
    Compare three commodities blend methods side-by-side: Sharpe, MDD, TO.

    Methods
    -------
    Rolling OLS MSRR   fixed lookback=252d rolling window
    Rolling Ridge MSRR same rolling window with L2 penalty (lam=ridge_lam)
    Static Ridge MSRR  IS-fit once with ridge, constant weights applied to full period

    Daily frequency; Sharpe annualised with √252.
    MDD at vol_target wealth scaling.
    TO = annualised one-way drift-corrected turnover of blend weights (static weights
         still trade back to target after returns move them).  Blend-level only;
         instrument-level trading is not included.

    Returns
    -------
    pd.DataFrame with columns [Method, Period, Sharpe, MDD, TO].
    """
    F = load_managed_returns(signals_dir=signals_dir)
    F_is_raw  = F.loc[F.index <= is_end]
    F_oos_raw = F.loc[F.index >  is_end]

    sigma = F_is_raw.std() * np.sqrt(252)
    sigma = sigma.replace(0.0, np.nan)
    F_is  = F_is_raw.div(sigma)
    F_oos = F_oos_raw.div(sigma)
    F_full = pd.concat([F_is, F_oos]).sort_index()

    def _sharpe(r: pd.Series) -> float:
        r = r.dropna()
        if len(r) < 60 or r.std() < 1e-12:
            return float("nan")
        return float(r.mean() / r.std() * np.sqrt(252))

    def _mdd(r: pd.Series) -> float:
        r = r.dropna()
        if r.empty or r.std() < 1e-12:
            return float("nan")
        ann_vol = r.std() * np.sqrt(252)
        cum = (1 + r * (vol_target / ann_vol)).cumprod()
        return float((cum / cum.cummax() - 1).min() * 100)

    def _blend_weight_to(w_df: pd.DataFrame, F_n: pd.DataFrame) -> float:
        """Drift-corrected annual one-way turnover of L1-normalised blend weights."""
        return _drift_to(w_df, F_n * vol_target, freq=252, l1_norm=True)

    def _normalise_w(w_raw: pd.DataFrame) -> pd.DataFrame:
        norm = w_raw.abs().sum(axis=1).replace(0.0, np.nan)
        return w_raw.div(norm, axis=0)

    # ── Method 1: Rolling OLS ────────────────────────────────────────────────
    w_ols = _normalise_w(
        _RollingMSRR(penalty="ols", lookback=lookback, min_obs=min_obs)._roll(F_full)
    )

    # ── Method 2: Rolling Ridge ──────────────────────────────────────────────
    w_roll_ridge = _normalise_w(
        _RollingMSRR(penalty="ridge", lam=ridge_lam, lookback=lookback, min_obs=min_obs)
        ._roll(F_full)
    )

    # ── Method 3: Static Ridge (IS-fit, constant weights) ───────────────────
    static_bw  = msrr_ridge(F_is, lam=ridge_lam)
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
        w_is_  = w_full.reindex(F_is.index)
        w_oos_ = w_full.reindex(F_oos.index)
        r_is_  = (F_is.loc[w_is_.dropna().index]   * w_is_.dropna()).sum(axis=1)
        r_oos_ = (F_oos.loc[w_oos_.dropna().index] * w_oos_.dropna()).sum(axis=1)
        for period, r, w in [
            ("IS",  r_is_,  w_is_),
            ("OOS", r_oos_, w_oos_),
        ]:
            records.append({
                "Method": method_name,
                "Period": period,
                "Sharpe": _sharpe(r),
                "MDD":    _mdd(r),
                "TO":     _blend_weight_to(w, F_is if period == "IS" else F_oos),
            })

    df = pd.DataFrame(records)

    # ── Print table ──────────────────────────────────────────────────────────
    W = 72
    print("=" * W)
    print(f"  COMMODITIES BLEND COMPARISON  (vol target={vol_target*100:.0f}%, λ_ridge={ridge_lam})")
    print(f"  note: TO = drift-corrected blend-weight turnover (ann.); excludes instrument-level trading")
    print("=" * W)
    print(f"  {'Method':<20}  {'Period':<6}  {'Sharpe':>7}  {'MDD':>8}  {'TO':>7}")
    print("  " + "─" * (W - 2))
    for _, row in df.iterrows():
        sr  = f"{row['Sharpe']:+.2f}" if not np.isnan(row["Sharpe"]) else "   n/a"
        mdd = f"{row['MDD']:.1f}%"    if not np.isnan(row["MDD"])    else "    n/a"
        to  = f"{row['TO']:.2f}x"     if not np.isnan(row["TO"])     else "   n/a"
        print(f"  {row['Method']:<20}  {row['Period']:<6}  {sr:>7}  {mdd:>8}  {to:>7}")
        if row["Period"] == "OOS":
            print("  " + "─" * (W - 2))
    print("=" * W)

    # ── Static Ridge IS weights ───────────────────────────────────────────────
    names  = list(F_is.columns)
    col_w  = 16
    print(f"\n  Avg blend weights  (Static Ridge IS-fit, λ={ridge_lam})")
    print("  " + "─" * (22 + col_w * len(names)))
    print(f"  {'Signal':<20}" + "".join(f"  {s:>{col_w}}" for s in names))
    print("  " + "─" * (22 + col_w * len(names)))
    print(f"  {'IS+OOS':<20}" + "".join(f"  {w_static_s.get(s, 0.0):>+{col_w}.1%}" for s in names))
    print("  " + "─" * (22 + col_w * len(names)))

    return df
