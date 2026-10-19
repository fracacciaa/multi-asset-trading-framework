"""
Constant-maturity total return decomposition.

For a constant-maturity position at tenor tau:

  ret_total   = ret_carry + ret_yldchg
  ret_carry   = income + rolldn + funding   ← smooth, the premium
  ret_yldchg  = price change from yield move  ← volatile, noise

All terms in bp per time step dt (typically 1/252 year).
Accumulate over N days to get bp over the horizon.

Key rule: the roll-down leg uses fit_prev.predict() at both tau and tau-dt.
Using the FITTED curve keeps carry orthogonal to the NS residual signal.
Using TODAY's duration (fit_now) would introduce lookahead; use fit_prev.

References: carry_signal_spec.md §5, Martens et al (2019) Table 1.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from rates.production.ns_core import NSFit

from .measures import approx_dv01


def cm_total_return_bp(
    y_prev: np.ndarray,
    y_now: np.ndarray,
    fit_prev: "NSFit",
    tau: np.ndarray,
    rf_prev: float,
    dur_prev: np.ndarray,
    dt: float = 1.0 / 252,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constant-maturity total-return decomposition for one time step.

    Parameters
    ----------
    y_prev   : (n,) actual yields at t-1, in bp
    y_now    : (n,) actual yields at t, in bp
    fit_prev : NS fit at t-1 — used for BOTH roll-down points (rule 1)
    tau      : (n,) tenors in years (constant-maturity grid)
    rf_prev  : funding rate at t-1, in bp
    dur_prev : (n,) modified duration at t-1 from fit_prev, in years
    dt       : time step in years (default 1/252 for daily)

    Returns
    -------
    ret_carry  : (n,) income + roll-down + funding, in bp
    ret_yldchg : (n,) price change from yield move, in bp

    ret_total  = ret_carry + ret_yldchg  (identity, tested in test_returns.py)

    All quantities are from t-1 except y_now (the only t-date input).
    """
    tau = np.asarray(tau, float)
    y_prev = np.asarray(y_prev, float)
    y_now = np.asarray(y_now, float)
    dur_prev = np.asarray(dur_prev, float)

    # Income: yield earned over dt (bp × years = bp over step)
    income = y_prev * dt

    # Roll-down: the fitted curve moves the position from tau to tau-dt
    # Use fit_prev for both evaluations (orthogonality rule)
    yhat_tau = fit_prev.predict(tau)                         # fitted yield at tau
    tau_aged = np.maximum(tau - dt, 1e-8)
    yhat_aged = fit_prev.predict(tau_aged)                   # fitted yield at tau-dt
    rolldn = dur_prev * (yhat_tau - yhat_aged)               # years × bp = bp

    # Price change from actual yield moves
    pricech = -dur_prev * (y_now - y_prev)                   # years × bp = bp

    # Funding cost
    funding = -rf_prev * dt                                  # scalar, bp over step

    ret_carry = income + rolldn + funding
    ret_yldchg = pricech

    return ret_carry, ret_yldchg


def compute_dur_panel(
    fit_prev: "NSFit",
    tau: np.ndarray,
) -> np.ndarray:
    """Modified duration at each tenor using the NS-fitted yield."""
    tau = np.asarray(tau, float)
    y_fitted = fit_prev.predict(tau)
    return np.array([approx_dv01(float(t), float(y) / 1e4) for t, y in zip(tau, y_fitted)])


def pnl_book(
    pos: np.ndarray,
    ret_carry: np.ndarray,
    ret_yldchg: np.ndarray,
) -> Tuple[float, float]:
    """
    Aggregate position-weighted P&L for one date.

    pos        : (n,) DV01 position weights (from dv01_positions)
    ret_carry  : (n,) carry component in bp (from cm_total_return_bp)
    ret_yldchg : (n,) yield-change component in bp

    Returns (pnl_carry_bp, pnl_yldchg_bp).
    """
    return float(np.dot(pos, ret_carry)), float(np.dot(pos, ret_yldchg))
