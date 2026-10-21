"""
NS mean-reversion signal comparison for commodity futures curves.

Tries all combinations of:
  Models   : ns_calibrated | ns_free | bc_calibrated | bc_free
  Residuals: in_sample (is) | LOO (loo)
  Projection: raw | factor_neutral (fn)
  Structures: scalar | slope_F1FN | fly_F1FmidFN | near_fly_F1F4F2
  Positioning: continuous clip(±3) | hysteresis (entry=1, exit=0)

Plus two factor-level mean-reversion signals:
  level_mr : -(β₁ − 252d mean)  →  front-contract return
  slope_mr : -(β₂ − 252d mean)  →  slope return (F1 − FN)

Signal z-scored with EWMA vol (hl=63d, lagged 1 day, no lookahead).
Position formed at close of t, earns t→t+1 log return of the structure.
OOS window: OOS_START onwards; λ calibrated on data ≤ TRAIN_END.

Usage:
    python -m commodities.mean_reversion.research
or import and call run_comparison() / summary_table().
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import Optional

from commodities.data import fetch_term_structure, fetch_term_structure_far
from commodities.mean_reversion.ns_core import (
    calibrate_lambda, extract_factors, extract_residuals,
    factor_neutral_residuals, tenors_for, TRAIN_END, LOADINGS,
)

# ── Config ─────────────────────────────────────────────────────────────────────

OOS_START    = "2019-01-01"
N_CONTRACTS  = 8        # use full 8-contract curve where available
Z_EWMA_HL    = 63
Z_MIN_PER    = 30
Z_VOL_FLOOR  = 1e-6
Z_CAP        = 3.0
HYST_ENTRY   = 1.0

MODELS = ["ns_calibrated", "ns_free", "bc_calibrated", "bc_free"]


# ── Data loading ───────────────────────────────────────────────────────────────

def build_log_curves(n_contracts: int = N_CONTRACTS) -> dict[str, pd.DataFrame]:
    """
    Build per-commodity log-price term structure DataFrames.
    Columns are integer contract numbers 1..N.
    When n_contracts > 4, contracts 5-8 are fetched from the far file.
    """
    ts   = fetch_term_structure()
    keys14 = ["front", "second", "third", "fourth"]
    commodities = list(ts["front"].columns)

    tsf = fetch_term_structure_far() if n_contracts > 4 else None
    keys58 = ["fifth", "sixth", "seventh", "eighth"]

    curves = {}
    for com in commodities:
        parts = []
        for i, key in enumerate(keys14[:min(n_contracts, 4)]):
            s = np.log(ts[key][com].replace(0.0, np.nan))
            parts.append(s.rename(i + 1))
        if tsf is not None:
            for i, key in enumerate(keys58[:max(0, n_contracts - 4)]):
                s = np.log(tsf[key][com].replace(0.0, np.nan))
                parts.append(s.rename(i + 5))
        df = pd.concat(parts, axis=1).sort_index().dropna(how="all")
        curves[com] = df

    return curves


# ── Z-score and positioning ────────────────────────────────────────────────────

def _zscore(series: pd.Series) -> pd.Series:
    """EWMA z-score; vol estimated on data through t-1 (no lookahead)."""
    vol = series.shift(1).ewm(halflife=Z_EWMA_HL, min_periods=Z_MIN_PER).std()
    return series / vol.clip(lower=Z_VOL_FLOOR)


def _continuous(z: pd.Series) -> pd.Series:
    return z.clip(-Z_CAP, Z_CAP)


def _hysteresis(z: pd.Series) -> pd.Series:
    """Enter when |z| > HYST_ENTRY; exit when z crosses zero."""
    pos = np.zeros(len(z))
    cur = 0.0
    for i, zv in enumerate(z.values):
        if np.isnan(zv):
            pos[i] = np.nan
            continue
        if cur == 0.0:
            if   zv >  HYST_ENTRY: cur =  1.0
            elif zv < -HYST_ENTRY: cur = -1.0
        else:
            if cur > 0 and zv <= 0: cur = 0.0
            elif cur < 0 and zv >= 0: cur = 0.0
        pos[i] = cur
    return pd.Series(pos, index=z.index)


# ── Return structures ──────────────────────────────────────────────────────────

def _build_returns(log_curve: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Daily log-return series for each signal structure:
      scalar    — Δlog(F1)  (whole-commodity direction)
      slope     — Δlog(F1) − Δlog(FN)  (duration-neutral slope)
      fly       — Δlog(F1) + Δlog(FN) − 2·Δlog(F_mid)  (butterfly)
      near_fly  — Δlog(F1) + Δlog(F4) − 2·Δlog(F2)  (short-end fly)
    """
    n = log_curve.shape[1]
    tenors = tenors_for(n)

    rets: dict[str, pd.Series] = {}
    rets["scalar"] = log_curve[1].diff()

    if n >= 2:
        # Slope: hedge ratio w = τ₁ / τ_N so both legs have equal τ-exposure
        w_slope = tenors[0] / tenors[-1]
        rets["slope"] = log_curve[1].diff() - w_slope * log_curve[n].diff()

    mid = (1 + n) // 2
    if n >= 3 and mid in log_curve.columns:
        # Butterfly: equal DV01 on each wing
        w_fly = (tenors[0] + tenors[-1]) / tenors[mid - 1]
        rets["fly"] = (log_curve[1].diff() + log_curve[n].diff()
                       - w_fly * log_curve[mid].diff())

    if 4 in log_curve.columns:
        # Near-end butterfly: F1+F4 vs F2
        w_near = (tenors[0] + tenors[3]) / tenors[1]
        rets["near_fly"] = (log_curve[1].diff() + log_curve[4].diff()
                            - w_near * log_curve[2].diff())

    return rets


# ── Per-commodity signal extraction ───────────────────────────────────────────

def _build_signals(
    log_curve: pd.DataFrame,
    model: str,
    lam: float,
) -> dict[str, pd.Series]:
    """
    All NS mean-reversion signals for one commodity × one model.
    Returns raw (pre-zscore) signal series.
    """
    basis  = "bc" if model.startswith("bc") else "ns"
    n      = log_curve.shape[1]
    tenors = tenors_for(n)

    factors              = extract_factors(log_curve, model=model, lam=lam)
    resid_is, resid_loo  = extract_residuals(log_curve, model=model, lam=lam)
    resid_is_fn          = factor_neutral_residuals(resid_is,  tenors, lam, basis)
    resid_loo_fn         = factor_neutral_residuals(resid_loo, tenors, lam, basis)

    sigs: dict[str, pd.Series] = {}

    # ── Factor-level mean reversion ───────────────────────────────────────────
    for fac_name, ret_str in [("level", "scalar"), ("slope", "slope")]:
        if fac_name in factors.columns:
            mu = factors[fac_name].rolling(252, min_periods=63).mean()
            sigs[f"{fac_name}_mr"] = -(factors[fac_name] - mu)

    # ── Aggregate cheapness (mean across contracts) ───────────────────────────
    sigs["is_raw_scalar"]  = resid_is.mean(axis=1)
    sigs["is_fn_scalar"]   = resid_is_fn.mean(axis=1)
    sigs["loo_raw_scalar"] = resid_loo.mean(axis=1)
    sigs["loo_fn_scalar"]  = resid_loo_fn.mean(axis=1)

    # ── Structural (contract-pair) signals ────────────────────────────────────
    for label, resid_df in [("is_raw", resid_is), ("loo_raw", resid_loo),
                             ("is_fn",  resid_is_fn), ("loo_fn", resid_loo_fn)]:
        if 1 not in resid_df.columns or n not in resid_df.columns:
            continue
        w_slope = tenors[0] / tenors[-1]
        sigs[f"{label}_slope"] = resid_df[1] - w_slope * resid_df[n]

        mid = (1 + n) // 2
        if mid in resid_df.columns:
            w_fly = (tenors[0] + tenors[-1]) / tenors[mid - 1]
            sigs[f"{label}_fly"] = resid_df[1] + resid_df[n] - w_fly * resid_df[mid]

        if 4 in resid_df.columns:
            w_near = (tenors[0] + tenors[3]) / tenors[1]
            sigs[f"{label}_near_fly"] = (resid_df[1] + resid_df[4]
                                          - w_near * resid_df[2])

    return sigs


# map signal name → which return structure it should use
_RETURN_MAP = {
    "level_mr":         "scalar",
    "slope_mr":         "slope",
    "is_raw_scalar":    "scalar",
    "is_fn_scalar":     "scalar",
    "loo_raw_scalar":   "scalar",
    "loo_fn_scalar":    "scalar",
    "is_raw_slope":     "slope",
    "is_fn_slope":      "slope",
    "loo_raw_slope":    "slope",
    "loo_fn_slope":     "slope",
    "is_raw_fly":       "fly",
    "is_fn_fly":        "fly",
    "loo_raw_fly":      "fly",
    "loo_fn_fly":       "fly",
    "is_raw_near_fly":  "near_fly",
    "is_fn_near_fly":   "near_fly",
    "loo_raw_near_fly": "near_fly",
    "loo_fn_near_fly":  "near_fly",
}


# ── Sharpe helper ──────────────────────────────────────────────────────────────

def _sharpe(s: pd.Series) -> float:
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 60:
        return np.nan
    std = s.std()
    return float(s.mean() / std * np.sqrt(252)) if std > 1e-12 else np.nan


# ── Main comparison ────────────────────────────────────────────────────────────

def run_comparison(
    commodities: Optional[list[str]] = None,
    n_contracts: int = N_CONTRACTS,
    models: Optional[list[str]] = None,
    oos_start: str = OOS_START,
) -> pd.DataFrame:
    """
    Run all NS variants × all signals. Returns OOS Sharpe per (commodity, model).

    Columns: one per (signal_name, positioning) pair, e.g. 'is_raw_scalar_cont'.
    """
    if models is None:
        models = MODELS

    print("Loading term structure data...", flush=True)
    curves = build_log_curves(n_contracts=n_contracts)
    if commodities is not None:
        curves = {c: v for c, v in curves.items() if c in commodities}

    print(f"Calibrating λ on data ≤ {TRAIN_END}...", flush=True)
    lambdas = calibrate_lambda(curves, train_end=TRAIN_END)

    rows = []
    for model in models:
        print(f"  {model}", flush=True)
        basis = "bc" if model.startswith("bc") else "ns"

        for com, log_curve in curves.items():
            lam = lambdas[basis].get(com)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sigs = _build_signals(log_curve, model, lam)
                    rets = _build_returns(log_curve)
            except Exception as e:
                print(f"    WARN {com}: {e}")
                continue

            row: dict = {"commodity": com, "model": model}

            for sig_name, raw_sig in sigs.items():
                ret_key = _RETURN_MAP.get(sig_name, "scalar")
                if ret_key not in rets:
                    continue

                z = _zscore(raw_sig)

                for pos_tag, pos_fn in [("cont", _continuous), ("hyst", _hysteresis)]:
                    # lag-1: position entered at t, earns return at t+1
                    pos = pos_fn(z).shift(1)
                    pnl = pos * rets[ret_key]
                    sh = _sharpe(pnl.loc[oos_start:])
                    row[f"{sig_name}_{pos_tag}"] = sh

            rows.append(row)

    return pd.DataFrame(rows).set_index(["commodity", "model"])


def summary_table(comp: pd.DataFrame) -> pd.DataFrame:
    """Mean OOS Sharpe across all commodities, indexed by model."""
    return comp.groupby("model").mean(numeric_only=True)


def best_signals(comp: pd.DataFrame, top_n: int = 10) -> pd.Series:
    """
    Cross-model, cross-commodity mean Sharpe per signal column.
    Returns the top_n signals sorted descending.
    """
    means = comp.mean(numeric_only=True)
    return means.sort_values(ascending=False).head(top_n)


def commodity_ranking(comp: pd.DataFrame) -> pd.DataFrame:
    """
    Per-commodity mean Sharpe averaged across all signals and models.
    Useful for spotting which commodities are most amenable to curve RV.
    """
    return comp.mean(axis=1).groupby("commodity").mean().sort_values(ascending=False)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    comp = run_comparison()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", "{:.3f}".format)

    print("\n=== Best signals (avg Sharpe across all commodities & models) ===")
    print(best_signals(comp, top_n=15).to_string())

    print("\n=== Summary by model (avg across all commodities & signals) ===")
    print(summary_table(comp).to_string())

    print("\n=== Commodity ranking (avg Sharpe across all signals & models) ===")
    print(commodity_ranking(comp).to_string())
