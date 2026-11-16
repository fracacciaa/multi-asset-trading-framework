"""
Full swap-repricing P&L for the momentum signal.

Replaces  pos × ret_d  (CM return approximation, duration = Σ k·df_k)
with the same full contract repricing used by classic_ns_pnl_repriced:

    P&L_t = Σ_T  swap_pv(K_T, T − dt, curve_t, notional_T) × 10 000

Position → notional conversion (momentum)
------------------------------------------
The momentum position p_T has units chosen so that p_T × dur_T × Δy_bp
gives the daily P&L, where dur_T = Σ_{k=1}^{int(T)} k × df_k (the same
"duration" used in the CM return formula in build_return_panels).

This quantity serves as the effective DV01 for the full repricing:

    dv01_tgt = −p_T × dur_T         [negative = receiver for p_T > 0]
    notional  = dv01_tgt / annuity_pv(T, t−1)

Positive position = bullish momentum = receiver (long duration), so the
sign flip ensures payer-positive convention:
    notional < 0  →  receiver  →  profits when rates fall  ✓

IMPORTANT: dur_T is computed from build_zero_curve(par), not bootstrap_curves.
This matches build_return_panels exactly (positions are sized using this
duration), ensuring the DV01 target is consistent with the original vol scaling.
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
from rates.production.yc_builder import build_zero_curve
from rates.production.momentum.returns import build_return_panels
from rates.production.momentum.signal_a import build_signal_a_daily, DM_MARKETS


def momentum_pnl_repriced(
    market: str,
    formation: int = 252,
    start: Optional[str] = None,
    end: Optional[str] = None,
    min_tenor: float = 1.0,
    tau_fn: Callable = _ACT360,
    t_plus: float = _T_PLUS_DEFAULT,
) -> pd.Series:
    """
    Full repricing P&L for the momentum signal, one market.

    Parameters
    ----------
    market    : OIS market name (e.g. "US", "EUR")
    formation : lookback window in trading days for the momentum signal
    start, end: date range for OIS data fetch
    min_tenor : skip tenors below this threshold (years)

    Returns
    -------
    pd.Series of daily P&L in bp, indexed by date.
    """
    par   = fetch_ois_curves(market, start=start, end=end)
    zeros = bootstrap_curves(par, tau_fn=tau_fn, t_plus=t_plus)

    # dur_code uses build_zero_curve — same as build_return_panels (returns.py).
    # Pre-compute integer-year discount factors for all dates (vectorised).
    zero_yc    = build_zero_curve(par)
    yc_cols    = zero_yc.columns.to_numpy(float)
    yc_np      = zero_yc.to_numpy(float)              # (D, 70) cc zero rates in %
    int_yrs    = np.arange(1.0, 31.0, dtype=float)
    int_idx    = np.array([int(np.argmin(np.abs(yc_cols - u))) for u in int_yrs])
    int_df_mat = np.exp(-yc_np[:, int_idx] / 100.0 * int_yrs[None, :])  # (D, 30)
    yc_dates   = zero_yc.index

    # ── Build momentum positions ──────────────────────────────────────────────
    panels = build_return_panels(market, start=start, end=end)
    if not panels or "ret_d" not in panels:
        return pd.Series(dtype=float, name="pnl_bp")

    mom_df = build_signal_a_daily({market: panels}, k=formation)
    if mom_df.empty:
        return pd.Series(dtype=float, name="pnl_bp")

    # Extract single-market positions (date × tenor)
    try:
        pos_df = mom_df.xs(market, level="market", axis=1)
    except KeyError:
        pos_df = mom_df[market] if market in mom_df.columns else pd.DataFrame()
    if pos_df.empty:
        return pd.Series(dtype=float, name="pnl_bp")

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
        active = pos.dropna()
        if active.empty:
            pnl_vals[date_t] = np.nan
            continue

        t_1, df_1 = _row_to_curve(zeros.loc[date_t1])
        t_t, df_t = _row_to_curve(zeros.loc[date_t])

        if len(t_1) < 2 or len(t_t) < 2:
            pnl_vals[date_t] = np.nan
            continue

        # Integer-year DFs from yc_builder (consistent with returns.py)
        if date_t1 not in yc_dates:
            pnl_vals[date_t] = np.nan
            continue
        yc_i    = yc_dates.get_loc(date_t1)
        int_dfs = int_df_mat[yc_i]  # (30,)

        dt_yr     = (date_t - date_t1).days / 365.0
        daily_pnl = 0.0

        for T_col in active.index:
            T = float(T_col)
            if T < min_tenor:
                continue
            pos_val = float(active[T_col])
            if not np.isfinite(pos_val) or pos_val == 0.0:
                continue

            # dur_T = Σ_{k=1}^{n_c} k × df_k  (same formula as returns.py)
            n_c = int(round(T))
            if n_c < 1 or n_c > 30:
                continue
            weights  = np.arange(1, n_c + 1, dtype=float)
            dur_code = float(np.dot(weights, int_dfs[:n_c]))
            if dur_code < 1e-10:
                continue

            K   = par_rate(T, t_1, df_1, tau_fn)
            ann = annuity_pv(T, t_1, df_1, tau_fn)
            if not (np.isfinite(K) and K > 0.0 and ann > 1e-10):
                continue

            # receiver = negative notional; payer-positive convention
            dv01_tgt = -pos_val * dur_code
            notional = dv01_tgt / ann

            T_rem = T - dt_yr
            if T_rem <= 0.0:
                continue

            pv = swap_pv(K, T_rem, t_t, df_t, tau_fn, notional=notional)
            daily_pnl += pv * 10_000.0

        pnl_vals[date_t] = daily_pnl

    return pd.Series(pnl_vals, name="pnl_bp")


def momentum_pnl_all_repriced(
    markets: Optional[list[str]] = None,
    formation: int = 252,
    start: Optional[str] = None,
    end: Optional[str] = None,
    min_tenor: float = 1.0,
) -> pd.DataFrame:
    """
    Full repricing P&L for the momentum signal, all markets.

    Returns
    -------
    date × market DataFrame of daily P&L in bp.
    """
    if markets is None:
        markets = DM_MARKETS

    mkt_series: dict[str, pd.Series] = {}
    for market in markets:
        try:
            s = momentum_pnl_repriced(
                market, formation=formation,
                start=start, end=end, min_tenor=min_tenor,
            )
            mkt_series[market] = s
        except Exception as exc:
            warnings.warn(f"momentum_pnl_repriced failed for {market}: {exc}")

    if not mkt_series:
        return pd.DataFrame()

    df = pd.DataFrame(mkt_series)
    df.index.name = "date"
    return df
