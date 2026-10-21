"""
BC mean-reversion curve fly signal for commodity futures.

Best performer from research.py comparison across 21 commodities / 8 contracts:
  bc_calibrated + LOO residual fly + hysteresis → OOS Sharpe 1.783

For each commodity: z-score the LOO residual butterfly (F1 + FN − w·F_mid),
using an EWMA vol (hl=63d, lagged 1 day). Position via hysteresis
(enter |z|>1, exit when z crosses zero).

BC λ calibrated once on data ≤ TRAIN_END and held fixed OOS.
Positive → front and back are cheap vs mid → buy wings, sell belly.

LOO residuals are essential here: IS residuals at curve endpoints (F1, FN)
are heavily self-absorbed under BC's 4-factor fit, making IS fly signals
sign-flip artefacts (IS fly OOS Sharpe ≈ −3.2 vs LOO ≈ +1.8).
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import Optional

from commodities.data import fetch_term_structure, fetch_term_structure_far
from commodities.mean_reversion.ns_core import (
    calibrate_lambda, extract_residuals, tenors_for, TRAIN_END,
)

# ── Config ────────────────────────────────────────────────────────────────────

MODEL       = "bc_calibrated"
N_CONTRACTS = 8
Z_EWMA_HL   = 63
Z_MIN_PER   = 30
Z_VOL_FLOOR = 1e-6
HYST_ENTRY  = 1.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_log_curves(n_contracts: int = N_CONTRACTS) -> dict[str, pd.DataFrame]:
    ts   = fetch_term_structure()
    keys14 = ["front", "second", "third", "fourth"]
    commodities = list(ts["front"].columns)

    tsf    = fetch_term_structure_far() if n_contracts > 4 else None
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


def _zscore(series: pd.Series) -> pd.Series:
    """EWMA z-score; vol estimated on data through t-1 (no lookahead)."""
    vol = series.shift(1).ewm(halflife=Z_EWMA_HL, min_periods=Z_MIN_PER).std()
    return series / vol.clip(lower=Z_VOL_FLOOR)


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


def _fly_signal(resid_loo: pd.DataFrame, tenors: np.ndarray) -> pd.Series:
    """
    LOO residual butterfly: resid_loo[F1] + resid_loo[FN] − w_fly·resid_loo[F_mid].
    Hedge ratio w_fly = (τ₁ + τ_N) / τ_mid ensures equal τ-exposure on each wing.
    """
    n = resid_loo.shape[1]
    mid = (1 + n) // 2
    if n < 3 or mid not in resid_loo.columns:
        return pd.Series(np.nan, index=resid_loo.index)
    w_fly = (tenors[0] + tenors[-1]) / tenors[mid - 1]
    return resid_loo[1] + resid_loo[n] - w_fly * resid_loo[mid]


# ── Public API ────────────────────────────────────────────────────────────────

def compute_signal(
    n_contracts: int = N_CONTRACTS,
    lambdas: Optional[dict] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    BC calibrated LOO fly mean-reversion signal for all commodities.

    Returns a DataFrame (date × commodity) of z-scored signals (pre-positioning).
    Positive → front and back are cheap vs mid → buy wings, sell belly.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        curves = _build_log_curves(n_contracts=n_contracts)

    if lambdas is None:
        lambdas = calibrate_lambda(curves, train_end=TRAIN_END)

    signals: dict[str, pd.Series] = {}
    for com, log_curve in curves.items():
        lam = lambdas["bc"].get(com)
        tenors = tenors_for(log_curve.shape[1])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, resid_loo = extract_residuals(log_curve, model=MODEL, lam=lam)
        except Exception:
            continue
        raw = _fly_signal(resid_loo, tenors)
        signals[com] = _zscore(raw)

    df = pd.concat(signals.values(), axis=1, keys=signals.keys())
    df.index.name = "date"
    if start is not None:
        df = df.loc[start:]
    if end is not None:
        df = df.loc[:end]
    return df


def compute_positions(
    n_contracts: int = N_CONTRACTS,
    lambdas: Optional[dict] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    BC calibrated LOO fly mean-reversion positions (hysteresis) for all commodities.

    Returns a DataFrame (date × commodity) of {−1, 0, +1} positions.
    Position entered at close of t, earns t→t+1 fly return:
      ΔlogF1 + ΔlogFN − w_fly·ΔlogF_mid
    """
    z = compute_signal(n_contracts=n_contracts, lambdas=lambdas,
                       start=start, end=end)
    return z.apply(_hysteresis)
