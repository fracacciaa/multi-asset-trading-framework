"""
Signal A: Moskowitz, Ooi & Pedersen (2012) time-series momentum.

Raw signal = per-instrument position size (spec §2):

  pos^s_t = sign(r^s_{t-12,t}) × (40% / sigma^s_t)

where sigma is the annualised ex ante vol estimated on daily returns with
an EWMA decay equivalent to a 60-day centre-of-mass (delta = 60/61).

Primary instrument set: tenors {2, 5, 10, 30}Y across DM OIS markets.
Full variant: all 27 OIS tenors.

Sign convention: positive position = long = receive fixed = benefit from
rates falling.  A positive 12-month cumulative return in the cm_return_bp
series means rates fell over the past 12 months → the signal is positive →
continue receiving fixed.  The sign() therefore maps directly to the
"long duration when past duration trade was profitable" intuition.

Spec: ois_momentum_signals_spec.md §2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

# ── Constants (MOP Table 2, Panel D specification) ────────────────────────────

DELTA         = 60.0 / 61.0          # EWMA decay: centre-of-mass = 60 days
TARGET_VOL_BP = 0.40 * 1e4           # 40% annualized, in bp
MOP_K         = 12                   # 12-month formation window (months)
MOP_H         = 1                    # 1-month holding period

# Primary DM instrument set (mirrors MOP's 13 bond futures)
DM_MARKETS = ["US", "EUR", "UK", "Japan", "Canada", "Australia", "New Zealand", "Sweden"]
PRIMARY_TENORS: list[float] = [2.0, 5.0, 10.0, 30.0]


# ── Ex ante volatility ────────────────────────────────────────────────────────

def ex_ante_vol(r_daily_bp: pd.Series) -> pd.Series:
    """
    MOP eq (1): annualised ex ante vol in bp, computed from daily returns.

    Uses data through t-1 only (the .shift(1) encodes MOP's own lookahead
    rule: "we use the volatility estimate at time t-1 applied to time-t returns").
    """
    r   = r_daily_bp.shift(1)
    mu  = r.ewm(alpha=1.0 - DELTA, adjust=False).mean()
    var = ((r - mu) ** 2).ewm(alpha=1.0 - DELTA, adjust=False).mean()
    return np.sqrt(261.0 * var)


# ── Per-instrument TSMOM position ────────────────────────────────────────────

def tsmom_position(
    ret_m_bp: pd.Series,
    sigma_ann_bp: pd.Series,
    k: int = MOP_K,
    h: int = MOP_H,
) -> pd.Series:
    """
    MOP eq (5): position = sign(past_k_months) × (TARGET_VOL / sigma).

    For h > 1, the JT-style overlapping portfolio is equivalent to an
    h-period rolling mean of the h=1 position series.

    ret_m_bp     : monthly cm_return_bp series for one (market, tenor)
    sigma_ann_bp : monthly-sampled ex ante vol (from ex_ante_vol on daily)
    """
    past = ret_m_bp.rolling(k).sum()
    pos  = np.sign(past) * (TARGET_VOL_BP / sigma_ann_bp.replace(0, np.nan))
    if h > 1:
        pos = pos.rolling(h).mean()
    return pos


# ── Panel builder ─────────────────────────────────────────────────────────────

def build_signal_a(
    panels_by_market: dict[str, dict[str, pd.DataFrame]],
    tenors: Optional[list[float]] = None,
    k: int = MOP_K,
    h: int = MOP_H,
) -> pd.DataFrame:
    """
    Build Signal A for all markets and selected tenors.

    Parameters
    ----------
    panels_by_market : output of build_all_return_panels()
    tenors           : constant-maturity tenors; defaults to PRIMARY_TENORS
    k, h             : MOP formation window (months) and holding period

    Returns
    -------
    DataFrame with MultiIndex columns (market, tenor), index = month-end dates.
    Values are MOP position sizes (unitless: bp / bp = notional fraction).
    Positive = long duration (receiver swap).
    """
    if tenors is None:
        tenors = PRIMARY_TENORS

    cols: dict = {}

    for market, panels in panels_by_market.items():
        ret_d = panels["ret_d"]
        ret_m = panels["ret_m"]

        for tau in tenors:
            if tau not in ret_d.columns or tau not in ret_m.columns:
                continue

            # Ex ante vol from daily returns, resampled to month-ends
            sigma_daily = ex_ante_vol(ret_d[tau].dropna())
            sigma_m     = sigma_daily.groupby(
                [sigma_daily.index.year, sigma_daily.index.month]
            ).last()
            sigma_m.index = pd.to_datetime(
                [f"{y}-{m:02d}" for y, m in sigma_m.index]
            ) + pd.offsets.MonthEnd(0)

            ret_m_s = ret_m[tau].dropna()
            common  = ret_m_s.index.intersection(sigma_m.index)
            if len(common) < k + 5:
                continue

            pos = tsmom_position(
                ret_m_s.loc[common],
                sigma_m.loc[common],
                k=k, h=h,
            )
            cols[(market, tau)] = pos

    if not cols:
        return pd.DataFrame()

    result = pd.DataFrame(cols)
    result.columns = pd.MultiIndex.from_tuples(result.columns, names=["market", "tenor"])
    result.index.name = "date"
    return result
