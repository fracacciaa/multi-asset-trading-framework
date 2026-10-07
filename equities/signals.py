"""
Equities signals: momentum, mean reversion, carry.
All signals return a cross-sectional z-scored DataFrame indexed date × ticker.
"""

import pandas as pd
import numpy as np
from typing import Optional


def momentum(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    half_life: Optional[int] = 126,
    winsor_pct: float = 0.01,
) -> pd.DataFrame:
    """
    Cross-sectional momentum signal.

    For each date t and ticker i, computes a formation-window return, then
    winsorizes and z-scores cross-sectionally.

    lookback:   formation window in trading days (default 252 = ~12 months)
    skip:       days excluded at the recent end to avoid short-term reversal
                (default 21 = ~1 month, giving classic 12-1m momentum)
    half_life:  if set, uses exponentially weighted log returns within the
                formation window (inspired by toraniko's factor_mom).
                Smooths the cliff-effect when old returns roll off the window.
                Default 126 = ~6-month half-life. Set to None for plain cumret.
    winsor_pct: symmetric cross-sectional winsorization percentile (default 1%)

    Returns date × ticker DataFrame of z-scored scores; NaN until sufficient history.
    """
    if half_life is not None:
        # Exponentially weighted: recent months count more, old history decays smoothly
        log_ret = np.log(prices / prices.shift(1))
        lagged  = log_ret.shift(skip)
        cumret  = lagged.ewm(halflife=half_life, min_periods=lookback).mean()
    else:
        price_end   = prices.shift(skip)
        price_start = prices.shift(lookback + skip)
        cumret = price_end / price_start - 1

    # Cross-sectional winsorize per date
    q_lo = cumret.quantile(winsor_pct, axis=1)
    q_hi = cumret.quantile(1 - winsor_pct, axis=1)
    cumret = cumret.clip(lower=q_lo, upper=q_hi, axis=0)

    # Cross-sectional z-score per date
    mean   = cumret.mean(axis=1)
    std    = cumret.std(axis=1)
    scores = cumret.sub(mean, axis=0).div(std.replace(0, np.nan), axis=0)

    return scores


def mean_reversion(prices: pd.DataFrame, lookback: int = 21) -> pd.DataFrame:
    """
    Short-term mean reversion (1-month reversal).
    lookback: lookback window in trading days
    """
    raise NotImplementedError

def carry(fundamentals: dict) -> pd.DataFrame:
    """
    Carry signal derived from earnings yield (BEST_EPS / PX_LAST).
    fundamentals: output of data.fetch_fundamentals()
    """
    raise NotImplementedError

def combined_signal(
    prices: pd.DataFrame,
    fundamentals=None,
    weights=None,
) -> pd.DataFrame:
    """
    Combine available signals into a single z-scored signal.
    weights: dict with keys 'momentum', 'mean_reversion', 'carry' summing to 1.
    Only implemented signals are included; weights are renormalized accordingly.
    """
    if weights is None:
        weights = {"momentum": 1 / 3, "mean_reversion": 1 / 3, "carry": 1 / 3}

    parts = {}
    try:
        parts["momentum"] = momentum(prices)
    except Exception:
        pass
    # mean_reversion and carry added once implemented

    if not parts:
        raise RuntimeError("No signals could be computed")

    available_weight = sum(weights[k] for k in parts)
    sig = sum(weights[k] / available_weight * v for k, v in parts.items())
    return sig
