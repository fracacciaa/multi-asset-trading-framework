"""
FX NS mean-reversion relative-value signals.
Direct parallel to rates/production/mean_reversion/rv.py.

For each currency pair, fits Nelson-Siegel to the annualised forward discount
term structure (1M/3M/6M/12M/2Y) and computes per-tenor residuals.  Five
structure types (FX analogues of the DV01-neutral rates structures):

  agg          mean of all per-tenor residuals         [factor-neutral aggregate]
  slope_1m2y   resid(1M) − resid(2Y)                  [← slope_2s10s]
  fly_1m3m6m   resid(3M) − 0.5×resid(1M) − 0.5×resid(6M)    [← fly_2s5s10s]
  fly_3m6m12m  resid(6M) − 0.5×resid(3M) − 0.5×resid(12M)
  fly_6m12m2y  resid(12M) − 0.5×resid(6M) − 0.5×resid(2Y)   [← fly_5s10s20s]

No DV01 weighting: FX forward discounts are dimensionless rates, so equal
weights are used (unlike yield-curve structures which require duration hedging).

Two models (rate analogues in brackets):
  ns3_fixed   λ calibrated per-pair on training data  [ns_calibrated]
  ns3_grid    λ daily grid-search per-pair             [ns_free]

Smoothing option (mirrors rv.py smooth=True):
  AR(1) halflife estimated per-pair per-tenor on training data.
  Residuals are smoothed with EWM(halflife) before z-scoring, which filters
  noise above the mean-reversion frequency.

All SR measured OOS (≥ 2019-01-01 by default).
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from fx.ns_fwd import (
    build_fwd_panel,
    compute_xs_positions,
    _hat_matrices_k,
    _fit_residuals,
    _calibrate_lam,
    LAM_GRID,
    TAU_YRS,
    FWD_TENORS,
    _TRAIN_END,
)
from fx.production.data import build_panels, CARRY_PAIRS
from rates.production.ns_core import ns_loadings

STRUCTURES = ["agg", "slope_1m2y", "fly_1m3m6m", "fly_3m6m12m", "fly_6m12m2y"]
MODELS     = ["ns3_fixed", "ns3_grid"]


# ── AR(1) half-life ────────────────────────────────────────────────────────────

def ar1_halflife(s: pd.Series) -> float:
    """AR(1) half-life in business days.  Returns nan if rho ∉ (0, 1)."""
    s = s.dropna()
    if len(s) < 100:
        return np.nan
    rho = np.polyfit(s.iloc[:-1].values, s.iloc[1:].values, 1)[0]
    if rho <= 0.0 or rho >= 1.0:
        return np.nan
    return float(np.log(0.5) / np.log(rho))


def compute_halflife_table(
    fwd_panels: dict,
    lam_fixed: dict,
    train_end: str = _TRAIN_END,
) -> pd.DataFrame:
    """
    AR(1) half-life (business days) of NS3 residuals per pair × tenor.

    Estimated on the training window (≤ train_end).  Used to set per-tenor
    EWM smoothing halflife in build_rv_signals when smooth=True.

    Returns
    -------
    DataFrame(pair × tenor label str)
    """
    rows: dict = {}
    H3 = _hat_matrices_k(TAU_YRS, LAM_GRID, k=3)

    for pair, df in fwd_panels.items():
        tenors = [t for t in FWD_TENORS if t in df.columns]
        if len(tenors) < 3:
            continue
        train = df[df.index <= pd.Timestamp(train_end)]
        if len(train) < 100:
            continue

        tau    = np.array([TAU_YRS[FWD_TENORS.index(t)] for t in tenors])
        H      = H3 if tenors == FWD_TENORS else _hat_matrices_k(tau, LAM_GRID, k=3)
        fd_arr = train[tenors].values
        resid  = _fit_residuals(fd_arr, H)
        resid_df = pd.DataFrame(resid, index=train.index, columns=tenors)
        rows[pair] = {t: ar1_halflife(resid_df[t]) for t in tenors}

    return pd.DataFrame(rows).T  # pair × tenor


# ── Structure functions ────────────────────────────────────────────────────────

def agg(resid_df: pd.DataFrame) -> pd.Series:
    """Mean of all per-tenor residuals.  Positive → curve globally above NS fit."""
    return resid_df.mean(axis=1).rename("agg")


def slope_1m2y(resid_df: pd.DataFrame) -> pd.Series:
    """
    1M vs 2Y spread.  Analogue of slope_2s10s (long short-end, short long-end).

    Positive → 1M residual above 2Y residual → short end cheap relative to long end.
    """
    if "1M" not in resid_df or "2Y" not in resid_df:
        return pd.Series(np.nan, index=resid_df.index, name="slope_1m2y")
    return (resid_df["1M"] - resid_df["2Y"]).rename("slope_1m2y")


def fly_1m3m6m(resid_df: pd.DataFrame) -> pd.Series:
    """
    Belly 3M vs wings 1M + 6M.  Analogue of fly_2s5s10s.

    Positive → belly (3M) cheap relative to wings → expect belly to cheapen further? No:
    positive belly residual = belly ABOVE NS fit = belly RICH in mean-reversion terms
    = expect 3M to revert downward vs 1M + 6M.
    Sign note: positive signal here means belly rich (same convention as rates).
    """
    cols = {"1M", "3M", "6M"}
    if not cols <= set(resid_df.columns):
        return pd.Series(np.nan, index=resid_df.index, name="fly_1m3m6m")
    return (resid_df["3M"] - 0.5 * resid_df["1M"] - 0.5 * resid_df["6M"]).rename("fly_1m3m6m")


def fly_3m6m12m(resid_df: pd.DataFrame) -> pd.Series:
    """Belly 6M vs wings 3M + 12M."""
    cols = {"3M", "6M", "12M"}
    if not cols <= set(resid_df.columns):
        return pd.Series(np.nan, index=resid_df.index, name="fly_3m6m12m")
    return (resid_df["6M"] - 0.5 * resid_df["3M"] - 0.5 * resid_df["12M"]).rename("fly_3m6m12m")


def fly_6m12m2y(resid_df: pd.DataFrame) -> pd.Series:
    """
    Belly 12M vs wings 6M + 2Y.  Analogue of fly_5s10s20s.
    """
    cols = {"6M", "12M", "2Y"}
    if not cols <= set(resid_df.columns):
        return pd.Series(np.nan, index=resid_df.index, name="fly_6m12m2y")
    return (resid_df["12M"] - 0.5 * resid_df["6M"] - 0.5 * resid_df["2Y"]).rename("fly_6m12m2y")


_STRUCT_FN = {
    "agg":          agg,
    "slope_1m2y":   slope_1m2y,
    "fly_1m3m6m":   fly_1m3m6m,
    "fly_3m6m12m":  fly_3m6m12m,
    "fly_6m12m2y":  fly_6m12m2y,
}


# ── Per-pair residuals ─────────────────────────────────────────────────────────

def _pair_residuals(
    pair: str,
    df: pd.DataFrame,
    model: str,
    lam_fixed: dict,
    smooth: bool = False,
    halflife_map: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Compute NS residuals for one pair.

    Parameters
    ----------
    model        : 'ns3_fixed' or 'ns3_grid'
    lam_fixed    : per-pair calibrated λ (used when model='ns3_fixed')
    smooth       : if True, smooth each tenor with EWM(halflife) from halflife_map
    halflife_map : {tenor_label: halflife_days}; ignored when smooth=False

    Returns
    -------
    DataFrame(date × tenor) of NS residuals (annualised fd units)
    """
    tenors = [t for t in FWD_TENORS if t in df.columns]
    if len(tenors) < 3:
        return pd.DataFrame()

    tau    = np.array([TAU_YRS[FWD_TENORS.index(t)] for t in tenors])
    fd_arr = df[tenors].values

    if model == "ns3_fixed":
        lam = lam_fixed.get(pair, float(np.median(LAM_GRID)))
        X   = ns_loadings(tau, lam)
        try:
            G = np.linalg.solve(X.T @ X, X.T)
        except np.linalg.LinAlgError:
            G = np.linalg.lstsq(X, np.eye(len(tau)), rcond=None)[0].T
        H1    = X @ G
        resid = fd_arr - (H1 @ fd_arr.T).T
    else:  # ns3_grid
        H = _hat_matrices_k(tau, LAM_GRID, k=3)
        resid = _fit_residuals(fd_arr, H)

    resid_df = pd.DataFrame(resid, index=df.index, columns=tenors)

    if smooth and halflife_map:
        for tenor, hl in halflife_map.items():
            if tenor in resid_df.columns and np.isfinite(hl) and hl > 1:
                resid_df[tenor] = resid_df[tenor].ewm(
                    halflife=hl, min_periods=int(hl)
                ).mean()

    return resid_df


# ── Portfolio-level builders ───────────────────────────────────────────────────

def build_rv_signals(
    model: str = "ns3_grid",
    pairs: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    smooth: bool = False,
    halflife_table: Optional[pd.DataFrame] = None,
    path: Optional[str] = None,
    lam_fixed: Optional[dict] = None,
    fwd_panels: Optional[dict] = None,
) -> dict[str, pd.DataFrame]:
    """
    Build all five RV structure signals for all pairs under one model.

    Mirrors rates/production/mean_reversion/rv.py:build_rv_signals().

    Parameters
    ----------
    model          : 'ns3_fixed' | 'ns3_grid'
    smooth         : smooth residuals with per-tenor AR(1) halflife EWM
    halflife_table : output of compute_halflife_table(); auto-computed if needed
    lam_fixed      : per-pair calibrated λ; auto-computed if model='ns3_fixed'
    fwd_panels     : pre-built panel from build_fwd_panel(); reloads if None

    Returns
    -------
    dict[structure_name → DataFrame(date × pair)]
      Raw (un-z-scored) structure signals.
    """
    if pairs is None:
        pairs = CARRY_PAIRS

    if fwd_panels is None:
        fwd_panels = build_fwd_panel(pairs=pairs, start=start, end=end, path=path)

    if model == "ns3_fixed" and lam_fixed is None:
        lam_fixed = _calibrate_lam(fwd_panels, LAM_GRID, k=3)
    lam_fixed = lam_fixed or {}

    if smooth and halflife_table is None:
        halflife_table = compute_halflife_table(fwd_panels, lam_fixed)

    struct_data: dict[str, dict] = {s: {} for s in STRUCTURES}

    for pair, df in fwd_panels.items():
        hl_map = halflife_table.loc[pair].to_dict() if (smooth and halflife_table is not None
                                                        and pair in halflife_table.index) else None
        resid_df = _pair_residuals(pair, df, model, lam_fixed,
                                    smooth=smooth, halflife_map=hl_map)
        if resid_df.empty:
            continue
        for sname, fn in _STRUCT_FN.items():
            struct_data[sname][pair] = fn(resid_df)

    def _panel(d: dict) -> pd.DataFrame:
        if not d:
            return pd.DataFrame()
        df = pd.concat(d.values(), axis=1)
        df.columns = list(d.keys())
        df.index.name = "date"
        return df.sort_index()

    return {s: _panel(struct_data[s]) for s in STRUCTURES}


# ── Comparison harness ─────────────────────────────────────────────────────────

def compare_rv(
    pairs: Optional[list[str]] = None,
    start: Optional[str] = "2005-01-01",
    end: Optional[str] = None,
    oos_start: Optional[str] = "2019-01-01",
    n: int = 4,
    path: Optional[str] = None,
    ewma_hl: int = 63,
    min_periods: int = 30,
) -> pd.DataFrame:
    """
    Run all model × structure × smooth combinations and print OOS comparison table.

    Grid: 2 models × 5 structures × 2 smooth = 20 variants.
    Carry (XS top-n/bottom-n on fd_1m) is shown as benchmark.

    Signal is built from `start` for EWMA warmup; SR measured on [oos_start, end].

    Returns
    -------
    DataFrame(variant → SR / skew / maxDD / corr_carry), sorted by OOS SR.
    """
    if pairs is None:
        pairs = CARRY_PAIRS

    # ── Shared data ────────────────────────────────────────────────────────────
    print("Loading forward panel and calibrating λ...")
    fwd_panels = build_fwd_panel(pairs=pairs, start=start, end=end, path=path)
    lam_fixed  = _calibrate_lam(fwd_panels, LAM_GRID, k=3)

    print("Computing AR(1) halflives for smoothing...")
    hl_table = compute_halflife_table(fwd_panels, lam_fixed)

    load_kw = {"path": path} if path else {}
    panels  = build_panels(pairs=pairs, start=start, end=end, **load_kw)
    rx_d    = panels["rx_d"]
    fd_d    = panels["fd_1m"].resample("B").ffill()

    # Carry benchmark
    carry_pos = compute_xs_positions(fd_d, n=n)
    common_c  = carry_pos.columns.intersection(rx_d.columns)
    c_pnl_full = (carry_pos[common_c].shift(1) * rx_d[common_c]).sum(axis=1)
    c_pnl_full = c_pnl_full.loc[c_pnl_full != 0]
    c_oos  = c_pnl_full.loc[oos_start:] if oos_start else c_pnl_full
    c_sr   = float(c_oos.mean() / c_oos.std() * np.sqrt(252))

    def _metrics(pnl: pd.Series) -> dict:
        oos = pnl.loc[oos_start:] if oos_start else pnl
        if len(oos) < 63:
            return dict(SR=np.nan, skew=np.nan, maxDD=np.nan, corr=np.nan)
        sr  = float(oos.mean() / oos.std() * np.sqrt(252))
        m, v = oos.mean(), oos.std()
        skw  = float(((oos - m) ** 3).mean() / v ** 3) if v > 0 else 0.0
        cum  = oos.cumsum()
        mdd  = float((cum - cum.cummax()).min())
        c_idx = oos.index.intersection(c_oos.index)
        corr  = float(np.corrcoef(oos.reindex(c_idx), c_oos.reindex(c_idx))[0, 1]) \
                if len(c_idx) > 10 else np.nan
        return dict(SR=sr, skew=skw, maxDD=mdd, corr=corr)

    def _run_model_smooth(model: str, smooth: bool) -> dict[str, pd.Series]:
        """Returns {structure_name: pnl_series} for one (model, smooth) combo."""
        sigs = build_rv_signals(
            model=model, pairs=pairs, start=start, end=end,
            smooth=smooth,
            halflife_table=hl_table if smooth else None,
            lam_fixed=lam_fixed if model == "ns3_fixed" else None,
            path=path,
        )
        pnls: dict[str, pd.Series] = {}
        for sname, sig_df in sigs.items():
            if sig_df.empty:
                continue
            # Z-score each pair's signal with lagged EWMA vol
            z_dict: dict[str, pd.Series] = {}
            for pair_col in sig_df.columns:
                s = sig_df[pair_col].dropna()
                if len(s) < min_periods:
                    continue
                vol = s.shift(1).ewm(halflife=ewma_hl, min_periods=min_periods).std()
                z   = (s / vol.clip(lower=1e-8)).clip(-3.0, 3.0)
                # Negate: positive structure signal = above NS fit = excess carry → short
                z_dict[pair_col] = -z
            if not z_dict:
                continue
            z_df = pd.DataFrame(z_dict).sort_index()
            pos  = compute_xs_positions(z_df, n=n)
            common = pos.columns.intersection(rx_d.columns)
            pos_a  = pos[common].reindex(rx_d.index).fillna(0.0)
            pnl    = (pos_a.shift(1) * rx_d[common]).sum(axis=1)
            pnls[sname] = pnl.loc[pnl != 0]
        return pnls

    # ── Run all combinations ───────────────────────────────────────────────────
    rows = []
    for model in MODELS:
        for smooth in [False, True]:
            smooth_tag = "smooth" if smooth else "raw"
            print(f"  {model} · {smooth_tag}...")
            pnls = _run_model_smooth(model, smooth)
            for sname in STRUCTURES:
                pnl = pnls.get(sname, pd.Series(dtype=float))
                m   = _metrics(pnl)
                label = f"{model}·{smooth_tag}·{sname}"
                rows.append({"variant": label, **m})

    df_out = pd.DataFrame(rows).set_index("variant")

    # ── Print table ────────────────────────────────────────────────────────────
    oos_label = f"OOS {oos_start}→end" if oos_start else "full sample"
    print(f"\n{oos_label}")
    print(f"{'Variant':<48} {'SR':>6} {'Skew':>6} {'MaxDD':>9} {'Corr/carry':>11}")
    print("─" * 84)
    print(f"{'CARRY (benchmark)':<48} {c_sr:>+6.3f}")
    print("─" * 84)
    for _, row in df_out.sort_values("SR", ascending=False).iterrows():
        sr_str = f"{row['SR']:>+6.3f}" if not np.isnan(row["SR"]) else "   nan"
        print(f"{row.name:<48} {sr_str} {row['skew']:>+6.3f}"
              f" {row['maxDD']:>9.4f} {row['corr']:>+11.3f}")

    return df_out
