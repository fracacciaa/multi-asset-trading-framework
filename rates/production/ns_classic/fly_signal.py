"""
Mean-reversion FLY signal: NS residual traded as an explicit DV01-neutral butterfly.

Motivation
----------
``classic_ns_pnl_repriced`` trades the whole-curve residual vector, demeaned so
that Σ dv01 = 0.  Row demeaning enforces neutrality to the NS *level* loading
only, not to slope or curvature, and the resulting book re-hedges up to 15 legs
per market every day.  This module expresses the same residual idea as a single
three-leg butterfly per market, which

  * is neutral to level, slope and curvature by construction of the leg weights,
  * trades as one quotable package rather than 15 independent outrights,
  * cuts gross tenor-level turnover roughly in half.

Leg selection rule (fixed a priori; no per-market tuning)
---------------------------------------------------------
    upper = longest tenor with >= MIN_OBS in-sample z-score observations
    body  = available tenor nearest 0.40 x upper
    lower = available tenor nearest 0.333 x upper

Weights are (-0.5, +1.0, -0.5) in DV01 space, so the package is DV01-neutral.

Sign convention matches ``signal.py``: dv01 > 0 is payer (profits when rates
rise).  A positive fly residual means the body is cheap to the wings, so the
body is received and the wings are paid.

P&L is full repricing, identical in method to ``classic_ns_pnl_repriced``:
each leg is struck at yesterday's par rate and marked on today's curve.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from rates.production.ns_classic.bootstrap import bootstrap_curves
from rates.production.ns_classic.signal import classic_ns_signal
from rates.production.ns_classic.pnl import (
    par_rate, annuity_pv, swap_pv, _row_to_curve, _ACT360, _T_PLUS_DEFAULT,
)

# ── Leg-selection constants ───────────────────────────────────────────────────

MIN_OBS:      int   = 500    # min IS z-score observations for a tenor to be usable
BODY_FRAC:    float = 0.40   # body  ≈ BODY_FRAC  × longest usable tenor
LOWER_FRAC:   float = 1 / 3  # lower ≈ LOWER_FRAC × longest usable tenor
FLY_WEIGHTS:  tuple = (-0.5, 1.0, -0.5)

# ── Sizing constants (mirror ns_classic.signal) ───────────────────────────────

DEFAULT_TARGET_VOL: float = 100.0   # bp/yr per market
DEFAULT_BURN_IN:    int   = 252
DEFAULT_EWMA_HL:    int   = 63
DEFAULT_MIN_TENOR:  float = 1.0


def select_fly_legs(
    z_scores: pd.DataFrame,
    is_end: str,
    min_obs: int = MIN_OBS,
) -> Optional[tuple[float, float, float]]:
    """
    Pick (lower, body, upper) by the fixed rule.  Returns None if no fly is
    constructible (fewer than three usable tenors).

    Only in-sample z-scores are inspected, so leg choice carries no look-ahead.
    """
    counts = z_scores.loc[z_scores.index <= is_end].notna().sum()
    usable = sorted(float(t) for t in counts.index if counts[t] >= min_obs)
    if len(usable) < 3:
        return None

    upper = max(usable)
    below = [t for t in usable if t < upper]
    if not below:
        return None
    body = min(below, key=lambda t: (abs(t - BODY_FRAC * upper), t))

    under_body = [t for t in usable if t < body]
    if not under_body:
        return None
    lower = min(under_body, key=lambda t: (abs(t - LOWER_FRAC * upper), t))

    return lower, body, upper


def fly_vol_scale_k(
    fly_resid: pd.Series,
    target_ann_vol: float = DEFAULT_TARGET_VOL,
    halflife: int = DEFAULT_EWMA_HL,
    min_periods: int = DEFAULT_EWMA_HL,
) -> pd.Series:
    """
    Time-varying scale so the package runs at target_ann_vol (bp/yr).

    Ex-ante daily variance uses the EWMA variance of the fly residual's own
    daily change, shifted one day, so the estimate is causal.
    """
    d = fly_resid.diff() * 100.0                       # % → bp
    ewma_var = (d ** 2).ewm(halflife=halflife, min_periods=min_periods).mean().shift(1)
    daily_var = (fly_resid ** 2) * ewma_var
    ann_vol = np.sqrt(252.0 * daily_var.clip(lower=0))
    return (target_ann_vol / ann_vol.replace(0.0, np.nan)).rename("k_scale")


def fly_pnl_repriced(
    par_curves: pd.DataFrame,
    lam: float,
    is_end: str,
    target_ann_vol: float = DEFAULT_TARGET_VOL,
    min_tenor: float = DEFAULT_MIN_TENOR,
    burn_in: int = DEFAULT_BURN_IN,
    weight_dv01: bool = True,
    tau_fn: Callable[[float, float], float] = _ACT360,
    t_plus: float = _T_PLUS_DEFAULT,
    vol_halflife: int = DEFAULT_EWMA_HL,
    legs: Optional[tuple[float, float, float]] = None,
) -> tuple[pd.Series, pd.DataFrame, tuple[float, float, float]]:
    """
    Full-repricing backtest of the NS-residual butterfly for one market.

    Returns
    -------
    pnl        daily P&L in bp, indexed by date
    dv01_pos   date × tenor held-DV01 frame (only the three legs are non-NaN)
    legs       the (lower, body, upper) actually used
    """
    zeros = bootstrap_curves(par_curves, tau_fn=tau_fn, t_plus=t_plus)
    z_scores, _, _ = classic_ns_signal(
        zeros, lam, min_tenor=min_tenor, burn_in=burn_in,
        window=None, weight_dv01=weight_dv01,
    )

    legs = legs or select_fly_legs(z_scores, is_end)
    if legs is None:
        raise ValueError("no constructible fly: fewer than three usable tenors")
    lower, body, upper = legs
    w_lo, w_bd, w_up = FLY_WEIGHTS

    zz = z_scores[[lower, body, upper]].dropna()
    fly_resid = w_lo * zz[lower] + w_bd * zz[body] + w_up * zz[upper]
    k_arr = fly_vol_scale_k(fly_resid, target_ann_vol, halflife=vol_halflife)

    dv01 = pd.DataFrame(
        {lower: -(k_arr * fly_resid) * w_lo,
         body:  -(k_arr * fly_resid) * w_bd,
         upper: -(k_arr * fly_resid) * w_up}
    ).dropna(how="all")

    dates = zeros.index
    pnl_vals: dict = {}
    for i in range(1, len(dates)):
        t0, t1 = dates[i - 1], dates[i]
        if t0 not in dv01.index:
            continue
        row = dv01.loc[t0]
        if row.isna().all():
            continue
        a1, d1 = _row_to_curve(zeros.loc[t0])
        at, dt_ = _row_to_curve(zeros.loc[t1])
        if len(a1) < 2 or len(at) < 2:
            continue
        dt_yr = (t1 - t0).days / 365.0
        total = 0.0
        for T in (lower, body, upper):
            dv = float(row[T])
            if not np.isfinite(dv):
                continue
            K = par_rate(T, a1, d1, tau_fn)
            ann = annuity_pv(T, a1, d1, tau_fn)
            if not (np.isfinite(K) and K > 0.0 and ann > 1e-10):
                continue
            T_rem = T - dt_yr
            if T_rem <= 0.0:
                continue
            total += swap_pv(K, T_rem, at, dt_, tau_fn, notional=dv / ann) * 10_000.0
        pnl_vals[t1] = total

    pnl = pd.Series(pnl_vals, name="pnl_bp").sort_index()
    dv01_full = dv01.reindex(columns=sorted(z_scores.columns))
    dv01_full.index.name = "date"
    return pnl, dv01_full, legs
