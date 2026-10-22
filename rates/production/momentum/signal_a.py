"""
Signal A daily: Moskowitz, Ooi & Pedersen (2012) time-series momentum, daily.

Per-instrument position (spec §2):

  pos^s_t = sign(r^s_{t-252,t}) × (40% / sigma^s_t)

sigma is the annualised ex ante vol from daily returns with EWMA decay
equivalent to a 60-day centre-of-mass (delta = 60/61).

Instrument set: tenors {2, 5, 10}Y across DM-9 OIS markets (DM-8 + Switzerland).
Each (market, tenor) pair is a fully independent bet; no cross-tenor aggregation.

Sign convention: positive = long = receive fixed = benefit from rates falling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

# ── Constants (MOP Table 2, Panel D specification) ────────────────────────────

DELTA         = 60.0 / 61.0          # EWMA decay: centre-of-mass = 60 days
TARGET_VOL_BP = 0.40 * 1e4           # 40% annualized, in bp
DAILY_K       = 252                  # ~1 trading year

# Primary DM instrument set
DM_MARKETS = ["US", "EUR", "UK", "Japan", "Canada", "Australia", "New Zealand", "Sweden", "Switzerland"]
PRIMARY_TENORS: list[float] = [2.0, 5.0, 10.0]
DAILY_K = 252  # ~1 trading year


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


# ── Panel builder ────────────────────────────────────────────────────────────

def build_signal_a_daily(
    panels_by_market: dict[str, dict[str, pd.DataFrame]],
    tenors: Optional[list[float]] = None,
    k: int = DAILY_K,
) -> pd.DataFrame:
    """
    Daily TSMOM: formation window and vol-scaling at daily frequency.

    Formation: rolling k-day sum of ret_d (daily CM returns in bp).
    Vol:       ex_ante_vol() kept daily — no month-end resampling.
    Position:  sign(past_k_days) × (TARGET_VOL_BP / sigma_daily)

    Same sign convention and position units as build_signal_a().
    Positive = long duration (receive fixed).
    Apply shift(1) before multiplying by ret_d to get PnL.
    """
    if tenors is None:
        tenors = PRIMARY_TENORS

    cols: dict = {}

    for market, panels in panels_by_market.items():
        ret_d = panels["ret_d"]

        for tau in tenors:
            if tau not in ret_d.columns:
                continue

            sigma_daily = ex_ante_vol(ret_d[tau].dropna())
            past        = ret_d[tau].rolling(k).sum()

            common = sigma_daily.index.intersection(past.index)
            if len(common) < k + 5:
                continue

            pos = np.sign(past.loc[common]) * (
                TARGET_VOL_BP / sigma_daily.loc[common].replace(0, np.nan)
            )
            cols[(market, tau)] = pos

    if not cols:
        return pd.DataFrame()

    result = pd.DataFrame(cols)
    result.columns = pd.MultiIndex.from_tuples(result.columns, names=["market", "tenor"])
    result.index.name = "date"
    return result


