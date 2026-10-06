"""
Equities signals: momentum, mean reversion, carry.
All signals return a cross-sectional z-scored DataFrame indexed date × ticker.
"""

import pandas as pd
import numpy as np


def momentum(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """
    Classic 12-1 month cross-sectional momentum.

    For each date t and ticker i:
        cumret[t, i] = price[t - skip] / price[t - (lookback + skip)] - 1

    Then per-date: winsorize at 1% tails, z-score cross-sectionally.

    lookback: formation window in trading days (default 252 = ~12 months)
    skip:     most-recent days excluded to avoid short-term reversal (default 21 = ~1 month)

    Returns date × ticker DataFrame of z-scored scores; NaN until sufficient history.
    """
    # Lagged prices at start and end of the formation window
    price_end   = prices.shift(skip)
    price_start = prices.shift(lookback + skip)
    cumret = price_end / price_start - 1

    # Cross-sectional winsorize at 1% tails per date (inspired by toraniko's winsor_factor=0.01)
    q_lo = cumret.quantile(0.01, axis=1)
    q_hi = cumret.quantile(0.99, axis=1)
    cumret = cumret.clip(lower=q_lo, upper=q_hi, axis=0)

    # Cross-sectional z-score per date
    mean = cumret.mean(axis=1)
    std  = cumret.std(axis=1)
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
