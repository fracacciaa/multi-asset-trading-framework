"""
Full swap-repricing P&L for the within-curve carry OIS signal.

Replaces the §6.2 approximation  pos × [carry − D_mod × Δy]  with the same
full contract repricing used by classic_ns_pnl_repriced:

    P&L_t = Σ_T  swap_pv(K_T, T − dt, curve_t, notional_T) × 10 000

Position → notional conversion (carry)
---------------------------------------
Carry positions are in DV01 units (years-equivalent), Σ|pos| = gross = 1.
Positive  = receiver (long duration).
D_mod relates the DV01 unit to actual yield sensitivity:

    dv01_tgt = −pos_T × D_mod(K_T, T)        [negative = receiver]
    notional  = dv01_tgt / annuity_pv(T, t−1)

The sign is negative because swap_pv uses payer-positive convention (positive
notional = payer).  With dv01_tgt < 0 for a receiver:

    swap_pv × 10 000 > 0  when K_T > current par rate  (rates fell)  ✓

Carry, roll-down, and convexity are all automatically captured by the full
repricing — no separate carry or roll term is needed.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional

import numpy as np
import pandas as pd

from rates.production.ois_data import fetch_ois_curves
from rates.production.ns_classic.bootstrap import (
    _ACT360, _T_PLUS_DEFAULT, bootstrap_curves,
)
from rates.production.ns_classic.pnl import (
    _row_to_curve, swap_pv, par_rate, annuity_pv,
)
from rates.production.carry.carry_ois_par import (
    ois_par_carry_panel, OIS_PAR_MARKETS, BENCHMARK_TENORS, OIS_COUPON_FREQ,
)
from rates.production.carry.carry_bonds import _dmod_par_vec
from rates.production.carry.signal import build_positions


def carry_pnl_repriced(
    market: str,
    tenors: Optional[list[float]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    tau_fn: Callable = _ACT360,
    t_plus: float = _T_PLUS_DEFAULT,
) -> pd.Series:
    """
    Full repricing P&L for the within-curve carry OIS signal, one market.

    Parameters
    ----------
    market  : OIS market name (e.g. "US", "EUR")
    tenors  : swap tenors; default BENCHMARK_TENORS = [5.0, 10.0]
    start, end : date range for OIS data fetch

    Returns
    -------
    pd.Series of daily P&L in bp, indexed by date.
    """
    if tenors is None:
        tenors = BENCHMARK_TENORS

    par = fetch_ois_curves(market, start=start, end=end)
    zeros = bootstrap_curves(par, tau_fn=tau_fn, t_plus=t_plus)

    # ── Build carry positions (same logic as ois_par_carry_pnl) ──────────────
    panel = ois_par_carry_panel(market, tenors=tenors, horizons={"3M": 0.25},
                                start=start, end=end)
    if panel.empty:
        return pd.Series(dtype=float, name="pnl_bp")

    sub        = panel.xs("3M", level="horizon")
    carry_wide = sub["total_carry_bp"].unstack("tenor")
    dmod_wide  = sub["dmod"].unstack("tenor")

    pos_df = build_positions(
        carry_wide, dmod_wide,
        rebalance="daily", lag=1,
        xs_mode="z", neutralize=("level",),
        gross=1.0, duration_adj="per_dur",
    )

    # ── Daily repricing loop ──────────────────────────────────────────────────
    dates     = zeros.index
    pnl_vals: dict = {}

    for i in range(1, len(dates)):
        date_t  = dates[i]
        date_t1 = dates[i - 1]

        if date_t1 not in pos_df.index:
            pnl_vals[date_t] = np.nan
            continue

        pos = pos_df.loc[date_t1]
        if pos.isna().all():
            pnl_vals[date_t] = np.nan
            continue

        t_1, df_1 = _row_to_curve(zeros.loc[date_t1])
        t_t, df_t = _row_to_curve(zeros.loc[date_t])

        if len(t_1) < 2 or len(t_t) < 2:
            pnl_vals[date_t] = np.nan
            continue

        dt_yr     = (date_t - date_t1).days / 365.0
        daily_pnl = 0.0

        for T in tenors:
            if T not in pos.index or not np.isfinite(pos[T]):
                continue
            pos_val = float(pos[T])
            if pos_val == 0.0:
                continue

            K   = par_rate(T, t_1, df_1, tau_fn)
            ann = annuity_pv(T, t_1, df_1, tau_fn)
            if not (np.isfinite(K) and K > 0.0 and ann > 1e-10):
                continue

            # D_mod from the bootstrapped par rate (consistent with existing carry calc)
            d = float(_dmod_par_vec(np.array([K * 100.0]), T, freq=OIS_COUPON_FREQ)[0])

            # receiver = negative notional; payer-positive convention
            dv01_tgt = -pos_val * d
            notional = dv01_tgt / ann

            T_rem = T - dt_yr
            if T_rem <= 0.0:
                continue

            pv = swap_pv(K, T_rem, t_t, df_t, tau_fn, notional=notional)
            daily_pnl += pv * 10_000.0

        pnl_vals[date_t] = daily_pnl

    return pd.Series(pnl_vals, name="pnl_bp")


def ois_par_carry_pnl_repriced(
    markets: Optional[list[str]] = None,
    tenors: Optional[list[float]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Full repricing P&L for the within-curve carry OIS signal, all markets.

    Returns
    -------
    date × market DataFrame of daily P&L in bp.
    """
    if markets is None:
        markets = OIS_PAR_MARKETS
    if tenors is None:
        tenors = BENCHMARK_TENORS

    mkt_series: dict[str, pd.Series] = {}
    for market in markets:
        try:
            s = carry_pnl_repriced(market, tenors=tenors, start=start, end=end)
            mkt_series[market] = s
        except Exception as exc:
            warnings.warn(f"carry_pnl_repriced failed for {market}: {exc}")

    if not mkt_series:
        return pd.DataFrame()

    df = pd.DataFrame(mkt_series)
    df.index.name = "date"
    return df
