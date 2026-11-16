"""
DV01-neutral two-tenor spread signal from NS residuals.

The full unconstrained signal at two tenors is d_target = [-k z_T1, -k z_T2].
The DV01-neutral structure s = [+1, -1] projects that target via OLS:

    w        = s · d_target / ||s||²  =  k (z_T2 - z_T1) / 2
    d_T1     = +w  =  k (z_T2 - z_T1) / 2       [payer-positive]
    d_T2     = -w  = -k (z_T2 - z_T1) / 2

When T2 is cheap relative to T1 (z_T2 > z_T1):
    d_T1 > 0  → payer at T1  (short duration, fades T1 richness)
    d_T2 < 0  → receiver at T2 (long duration, fades T2 cheapness)

Coverage ratio:
    ||d_actual|| / ||d_target|| = |z_T2 - z_T1| / (√2 √(z_T1² + z_T2²))

This equals 1 when z_T1 = -z_T2 (pure spread signal), 0 when z_T1 = z_T2
(no spread signal — both tenors equally rich/cheap). Tracking it shows how
much of the daily residual information the structure captures.

Vol scaling:
    P&L ≈ k × (z_T2 - z_T1)/2 × (Δresid_T1 - Δresid_T2) × 100   [bp]
    daily_var = k² × ((z_T2 - z_T1)/2)² × EWMA_var(Δresid_T1 - Δresid_T2) × 100²
    k_t = target_ann_vol / sqrt(252 × daily_var_t)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, Optional

from rates.production.ns_classic.bootstrap import (
    _ACT360, _T_PLUS_DEFAULT, bootstrap_curves,
)
from rates.production.ns_classic.pnl import (
    _row_to_curve, swap_pv, par_rate, annuity_pv,
)
from rates.production.ns_classic.signal import classic_ns_signal


# ── Column lookup ─────────────────────────────────────────────────────────────

def _col(df: pd.DataFrame, T: float) -> pd.Series:
    """Return the column of df closest to tenor T (years)."""
    if T in df.columns:
        return df[T]
    cols = df.columns.to_numpy(dtype=float)
    return df.iloc[:, int(np.argmin(np.abs(cols - T)))]


# ── Spread signal construction ────────────────────────────────────────────────

def spread_zscores(z_scores: pd.DataFrame, T1: float, T2: float) -> pd.Series:
    """
    Combined spread z-score: (z_T2 - z_T1) / 2.

    This is the OLS projection weight w for the DV01-neutral structure
    s = [+1, -1].  Positive = T2 cheap relative to T1.
    """
    return (_col(z_scores, T2) - _col(z_scores, T1)) / 2.0


def coverage_ratio(z_T1: pd.Series, z_T2: pd.Series) -> pd.Series:
    """
    Fraction of the two-tenor signal energy captured by the DV01-neutral structure.

    ||d_actual|| / ||d_target|| = |z_T2 - z_T1| / (√2 × √(z_T1² + z_T2²))

    Returns NaN where both z-scores are zero.
    """
    num = (z_T2 - z_T1).abs()
    den = np.sqrt(2.0) * np.sqrt(z_T1 ** 2 + z_T2 ** 2)
    return num / den.replace(0.0, np.nan)


# ── Vol scaling ───────────────────────────────────────────────────────────────

def spread_vol_scale_k(
    z_T1: pd.Series,
    z_T2: pd.Series,
    resid_T1: pd.Series,
    resid_T2: pd.Series,
    target_ann_vol: float,
    halflife: int = 63,
    min_periods: int = 63,
) -> pd.Series:
    """
    Time-varying k targeting target_ann_vol (bp/year) for the spread position.

    P&L variance = k² × ((z_T2 - z_T1)/2)² × EWMA_var(Δresid_T1 - Δresid_T2) × 100²
    k_t = target_ann_vol / sqrt(252 × signal_sq(t-1) × ewma_var(t))

    Parameters
    ----------
    z_T1, z_T2        : z-scores at the two tenors (from classic_ns_signal)
    resid_T1, resid_T2: NS residuals in % (from classic_ns_signal)
    target_ann_vol    : annualised target portfolio vol in bp
    halflife          : EWMA halflife (trading days)
    min_periods       : minimum periods for EWMA to produce a non-NaN estimate
    """
    signal_sq = ((z_T2 - z_T1) / 2.0) ** 2

    # Daily residual spread change in bp; sign doesn't matter for variance
    dr_spread = (resid_T1 - resid_T2).diff() * 100.0

    ewma_var = (dr_spread ** 2).ewm(
        halflife=halflife, min_periods=min_periods
    ).mean().shift(1)

    port_var = signal_sq * ewma_var                       # bp²/day
    port_ann_vol = np.sqrt(252.0 * port_var.clip(lower=0.0))

    k = (target_ann_vol / port_ann_vol).replace([np.inf, -np.inf], np.nan)
    k.name = "k_spread"
    return k


# ── Full repricing backtest ───────────────────────────────────────────────────

def spread_pnl_repriced(
    par_curves: pd.DataFrame,
    lam: float,
    T1: float,
    T2: float,
    k: float = 1.0,
    target_ann_vol: Optional[float] = None,
    min_tenor: float = 1.0,
    burn_in: int = 252,
    window: Optional[int] = None,
    weight_dv01: bool = True,
    tau_fn: Callable[[float, float], float] = _ACT360,
    t_plus: float = _T_PLUS_DEFAULT,
    vol_halflife: int = 63,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Full repricing backtest for the DV01-neutral T1-T2 spread.

    P&L is computed from full swap repricing (not DV01 × Δy), so carry,
    roll-down, and convexity are all included.

    Parameters
    ----------
    par_curves    : date × tenor OIS par rates in % (from fetch_ois_curves)
    lam           : fixed NS lambda (from select_lambda)
    T1, T2        : the two tenors (years); T2 > T1 by convention
    k             : flat scale factor; overridden by target_ann_vol if set
    target_ann_vol: annualised target portfolio vol in bp
    min_tenor     : floor for NS fit and position loop
    burn_in       : dates before signal goes live
    window        : rolling z-score window; None = expanding
    weight_dv01   : DV01-weight the NS fit (recommended)
    tau_fn        : day-count function (default: act/360)
    t_plus        : T+2 spot offset in years
    vol_halflife  : EWMA halflife for vol-scaling denominator

    Returns
    -------
    pnl          : pd.Series of daily P&L in bp
    spread_z     : pd.Series of combined spread z-score (z_T2 - z_T1) / 2
    cov_ratio    : pd.Series of coverage ratio per day
    k_arr        : pd.Series of daily scale factor (for diagnostics)
    """
    # ── 1. Bootstrap + NS signal ──────────────────────────────────────────────
    zeros = bootstrap_curves(par_curves, tau_fn=tau_fn, t_plus=t_plus)
    z_scores, _, resids = classic_ns_signal(
        zeros, lam,
        min_tenor=min_tenor, burn_in=burn_in,
        window=window, weight_dv01=weight_dv01,
    )

    z_T1    = _col(z_scores, T1)
    z_T2    = _col(z_scores, T2)
    resid_T1 = _col(resids, T1)
    resid_T2 = _col(resids, T2)

    sp_z   = spread_zscores(z_scores, T1, T2)
    cov    = coverage_ratio(z_T1, z_T2)

    # ── 2. Vol scaling ────────────────────────────────────────────────────────
    if target_ann_vol is not None:
        k_arr = spread_vol_scale_k(
            z_T1, z_T2, resid_T1, resid_T2,
            target_ann_vol, halflife=vol_halflife,
        )
    else:
        k_arr = pd.Series(float(k), index=zeros.index)

    # ── 3. Daily repricing loop ───────────────────────────────────────────────
    dates     = zeros.index
    pnl_vals: dict = {}

    for i in range(1, len(dates)):
        date_t  = dates[i]
        date_t1 = dates[i - 1]

        z1_t1 = float(z_T1.iloc[i - 1])   # z-score at T1 as of t-1
        z2_t1 = float(z_T2.iloc[i - 1])   # z-score at T2 as of t-1
        k_t1  = float(k_arr.iloc[i - 1])

        if not (np.isfinite(z1_t1) and np.isfinite(z2_t1) and np.isfinite(k_t1)):
            pnl_vals[date_t] = np.nan
            continue

        t_1, df_1 = _row_to_curve(zeros.loc[date_t1])
        t_t, df_t = _row_to_curve(zeros.loc[date_t])

        if len(t_1) < 2 or len(t_t) < 2:
            pnl_vals[date_t] = np.nan
            continue

        dt_yr = (date_t - date_t1).days / 365.0

        # Spread signal (the OLS weight w):  d_T1 = +w,  d_T2 = -w
        w = k_t1 * (z2_t1 - z1_t1) / 2.0

        daily_pnl = 0.0
        valid = True
        for T, sign in [(T1, +1.0), (T2, -1.0)]:
            if T < min_tenor:
                continue

            dv01_tgt = w * sign
            K   = par_rate(T, t_1, df_1, tau_fn)
            ann = annuity_pv(T, t_1, df_1, tau_fn)

            if not (np.isfinite(K) and K > 0.0 and ann > 1e-10):
                valid = False
                break

            notional = dv01_tgt / ann
            T_rem    = T - dt_yr
            if T_rem <= 0.0:
                continue

            pv = swap_pv(K, T_rem, t_t, df_t, tau_fn, notional=notional)
            daily_pnl += pv * 10_000.0

        pnl_vals[date_t] = daily_pnl if valid else np.nan

    pnl = pd.Series(pnl_vals, name="pnl_bp")
    return pnl, sp_z, cov, k_arr
