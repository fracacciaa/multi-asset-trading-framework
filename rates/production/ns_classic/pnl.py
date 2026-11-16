"""
Correct P&L backtest for the classic-NS signal via full contract repricing.

Do NOT use −D × Δy.  Reprice the actual swap contract on the new curve.

    P&L_t = swap_pv(K, T − dt, curve_t) − swap_pv(K, T, curve_{t-1})

The second term is zero by construction (position opened at par yesterday).
The first term captures carry, roll-down, and the full curve move in a single
number.  Decompose afterwards for attribution if needed; do not build the
P&L from the decomposition.

Two implementation details that matter:

  1.  Calendar aging.  T_rem = T − (date_t − date_{t-1}).days / 365.
      Using 1/252 instead drifts the roll-down term, particularly over
      weekends (3-day gap = 3×carry + 3×roll-down, not 1×).

  2.  Constant-maturity rebalancing.  Every day: mark yesterday's contract
      with today's curve, receive the PV in cash, open a new at-par contract
      for the updated target notional.  The realised P&L is entirely from the
      repricing step; the new position costs zero at inception.

Sign convention (payer-positive):
  notional > 0 → payer  (pays K, receives floating, profits when rates rise)
  notional < 0 → receiver (pays floating, receives K, profits when rates fall)

  dv01_target = −k × z_score   (receiver where tenor is cheap → notional < 0)
  notional    = dv01_target / annuity_pv
  P&L (bp)    = Σ_i swap_pv(K_i, T_i − dt, curve_t, notional_i) × 10 000

The ×10 000 converts the PV fraction (decimal × time ≈ dimensionless) to
basis-point P&L consistent with the vol-scaling target in signal.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, Optional

from rates.production.ns_classic.bootstrap import _ACT360, _T_PLUS_DEFAULT, bootstrap_curves, _logdf_interp
from rates.production.ns_classic.signal import classic_ns_signal, vol_scale_k


# ── Swap primitives ───────────────────────────────────────────────────────────

def swap_schedule(T_rem: float) -> np.ndarray:
    """
    Annual coupon payment times for a swap with T_rem years remaining.

    For T_rem ≤ 1: single cash flow at T_rem (zero-coupon structure).
    For T_rem > 1: annual payments at first_coupon, first_coupon+1, ..., T_rem
                   where first_coupon = T_rem − floor(T_rem − ε).

    Examples (original 3Y, aged 1 calendar day, dt ≈ 1/365):
      T_rem = 2.9973 → [0.9973, 1.9973, 2.9973]
      T_rem = 0.9973 → [0.9973]
    """
    if T_rem <= 1.0 + 1e-9:
        return np.array([max(T_rem, 1e-9)])
    n = int(np.ceil(T_rem - 1e-6))
    first = T_rem - (n - 1)
    return np.array([first + k for k in range(n)])


def swap_pv(
    K: float,
    T_rem: float,
    times: np.ndarray,
    dfs: np.ndarray,
    tau_fn: Callable[[float, float], float],
    notional: float = 1.0,
) -> float:
    """
    PV of a payer OIS swap with T_rem years remaining, fixed rate K (decimal).

    pv = notional × [(1 − df_{T_rem}) − K × Σ_i a_i × df_i]

    Positive for payer when K < current par rate (rates have risen above K).
    Negative notional flips to receiver convention.

    Parameters
    ----------
    K        : fixed coupon rate in decimal (0.05 = 5%)
    T_rem    : remaining maturity in years (use actual calendar aging)
    times    : bootstrapped zero-curve knot times (years from spot)
    dfs      : corresponding discount factors from the bootstrap
    tau_fn   : day-count fraction function(start_yr, end_yr)
    notional : signed notional; negative = receiver

    Returns
    -------
    float — PV in [notional × rate × time] units (multiply by 10 000 for bp)
    """
    if T_rem <= 0.0:
        return 0.0
    t_i   = swap_schedule(T_rem)
    t_prev = np.concatenate([[0.0], t_i[:-1]])
    a_i   = np.array([tau_fn(s, e) for s, e in zip(t_prev, t_i)])
    ldf   = np.log(np.maximum(dfs, 1e-10))
    df_i  = np.exp(np.array([_logdf_interp(t, times, ldf) for t in t_i]))
    ann   = float(np.dot(a_i, df_i))
    return notional * ((1.0 - df_i[-1]) - K * ann)


def par_rate(
    T: float,
    times: np.ndarray,
    dfs: np.ndarray,
    tau_fn: Callable[[float, float], float],
) -> float:
    """
    At-par fixed rate (decimal) for a T-year OIS.

    par = (1 − df_T) / Σ_i a_i × df_i
    """
    t_i   = swap_schedule(T)
    t_prev = np.concatenate([[0.0], t_i[:-1]])
    a_i   = np.array([tau_fn(s, e) for s, e in zip(t_prev, t_i)])
    ldf   = np.log(np.maximum(dfs, 1e-10))
    df_i  = np.exp(np.array([_logdf_interp(t, times, ldf) for t in t_i]))
    ann   = float(np.dot(a_i, df_i))
    return float((1.0 - df_i[-1]) / ann) if ann > 1e-10 else float("nan")


def annuity_pv(
    T: float,
    times: np.ndarray,
    dfs: np.ndarray,
    tau_fn: Callable[[float, float], float],
) -> float:
    """
    PV annuity: Σ_i a_i × df_i for the full payment schedule at tenor T.

    Used to convert a DV01 target to a contract notional:
      notional = dv01_target / annuity_pv
    For a small yield move Δy: ΔPV ≈ notional × annuity × Δy = dv01_target × Δy.
    """
    t_i   = swap_schedule(T)
    t_prev = np.concatenate([[0.0], t_i[:-1]])
    a_i   = np.array([tau_fn(s, e) for s, e in zip(t_prev, t_i)])
    ldf   = np.log(np.maximum(dfs, 1e-10))
    df_i  = np.exp(np.array([_logdf_interp(t, times, ldf) for t in t_i]))
    return float(np.dot(a_i, df_i))


# ── Curve utilities ───────────────────────────────────────────────────────────

def _row_to_curve(zero_row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert one row of a zero-rate DataFrame to (times, discount_factors).

    times : ascending tenor knots in years (NaN entries dropped)
    dfs   : corresponding discount factors  df_i = exp(−z_i/100 × t_i)
    """
    valid = zero_row.dropna().sort_index()
    t  = valid.index.to_numpy(dtype=float)
    z  = valid.to_numpy(dtype=float)
    df = np.exp(-z / 100.0 * t)
    return t, df


# ── Full repricing backtest ───────────────────────────────────────────────────

def classic_ns_pnl_repriced(
    par_curves: pd.DataFrame,
    lam: float,
    k: float = 1.0,
    target_ann_vol: Optional[float] = None,
    min_tenor: float = 1.0,
    burn_in: int = 252,
    window: Optional[int] = None,
    weight_dv01: bool = True,
    tau_fn: Callable[[float, float], float] = _ACT360,
    t_plus: float = _T_PLUS_DEFAULT,
    vol_halflife: int = 63,
    directional: bool = False,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Full repricing backtest for the classic-NS signal.

    Pipeline:
      1. Bootstrap par → zero curve (log-DF, act/360, T+2)
      2. Fit NS, compute z-scores on expanding causal window
      3. At each date t: reprice yesterday's contract on today's curve
         P&L_t = Σ_i notional_i(t-1) × swap_pv(K_i(t-1), T_i − dt, curve_t) × 10 000

    The ×10 000 converts [fraction × dimensionless] to bp, consistent with the
    vol-scaling target (which is also in bp).

    Parameters
    ----------
    par_curves     : date × tenor DataFrame of OIS par rates in percent
                     (from fetch_ois_curves)
    lam            : fixed NS lambda (from select_lambda)
    k              : flat scale factor; overridden by target_ann_vol if set
    target_ann_vol : annualised target portfolio vol in bp; sets time-varying k
    min_tenor      : exclude tenors below this from the NS fit and positions
    burn_in        : dates before the signal goes live (z-scores are NaN)
    window         : rolling window for z-score standardisation; None = expanding
    weight_dv01    : DV01-weight the NS fit (recommended; True = factor-neutral)
    tau_fn         : day-count function (default: act/360 approximation)
    t_plus         : T+2 spot offset in years (default 2/365)
    vol_halflife   : EWMA halflife (days) for vol-scaling denominator

    Returns
    -------
    pnl      : pd.Series of daily P&L in bp, indexed by date
    z_scores : date × tenor z-score DataFrame (NaN during burn-in)
    resids   : date × tenor NS residual DataFrame
    """
    # ── 1. Bootstrap ──────────────────────────────────────────────────────────
    zeros = bootstrap_curves(par_curves, tau_fn=tau_fn, t_plus=t_plus)

    # ── 2. NS signal (fit + z-scores) ─────────────────────────────────────────
    z_scores, _, resids = classic_ns_signal(
        zeros, lam,
        min_tenor=min_tenor, burn_in=burn_in,
        window=window, weight_dv01=weight_dv01,
    )

    # ── 3. Time-varying scale factor ──────────────────────────────────────────
    signal_z = np.sign(z_scores) if directional else z_scores
    if target_ann_vol is not None:
        k_arr = vol_scale_k(signal_z, resids, target_ann_vol, halflife=vol_halflife)
    else:
        k_arr = pd.Series(float(k), index=zeros.index)

    # ── 4. Daily repricing loop ───────────────────────────────────────────────
    dates     = zeros.index
    pnl_vals: dict = {}

    for i in range(1, len(dates)):
        date_t  = dates[i]
        date_t1 = dates[i - 1]

        z_t1  = signal_z.loc[date_t1]
        k_t1  = float(k_arr.iloc[i - 1])

        # Skip: no signal yet or k not estimable
        if z_t1.isna().all() or not np.isfinite(k_t1):
            pnl_vals[date_t] = np.nan
            continue

        # Curves as (times, dfs)
        t_1, df_1 = _row_to_curve(zeros.loc[date_t1])
        t_t, df_t = _row_to_curve(zeros.loc[date_t])

        if len(t_1) < 2 or len(t_t) < 2:
            pnl_vals[date_t] = np.nan
            continue

        # Actual calendar aging (handles weekends correctly)
        dt_yr = (date_t - date_t1).days / 365.0

        daily_pnl = 0.0

        for col in z_t1.dropna().index:
            T = float(col)
            if T < min_tenor:
                continue
            z_val = float(z_t1[col])
            if not np.isfinite(z_val):
                continue

            # Yesterday: at-par fixed rate and annuity
            K   = par_rate(T, t_1, df_1, tau_fn)
            ann = annuity_pv(T, t_1, df_1, tau_fn)
            if not (np.isfinite(K) and K > 0.0 and ann > 1e-10):
                continue

            # Signed notional (payer-positive: negative = receiver)
            dv01_tgt = -k_t1 * z_val
            notional = dv01_tgt / ann

            # Remaining maturity with actual calendar aging
            T_rem = T - dt_yr
            if T_rem <= 0.0:
                continue

            # Reprice at today's curve; ×10 000 → bp
            pv = swap_pv(K, T_rem, t_t, df_t, tau_fn, notional=notional)
            daily_pnl += pv * 10_000.0

        pnl_vals[date_t] = daily_pnl

    pnl = pd.Series(pnl_vals, name="pnl_bp")
    return pnl, z_scores, resids
