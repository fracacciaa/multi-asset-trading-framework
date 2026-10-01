"""
Equities signals: momentum, mean reversion, carry.
All signals return a cross-sectional z-scored Series/DataFrame indexed by date x ticker.
"""

import pandas as pd


def momentum(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """
    12-1 month cross-sectional momentum.
    lookback: total lookback in trading days (default 12 months)
    skip: skip most recent `skip` days to avoid short-term reversal
    """
    raise NotImplementedError

def mean_reversion(prices: pd.DataFrame, lookback: int = 21) -> pd.DataFrame:
    """
    Short-term mean reversion (1-month reversal).
    lookback: lookback window in trading days
    """
    raise NotImplementedError

def carry(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """
    Carry signal derived from dividend yield or earnings yield.
    fundamentals: output of data.fetch_fundamentals()
    """
    raise NotImplementedError

def combined_signal(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    weights: dict | None = None,
) -> pd.DataFrame:
    """
    Combine momentum, mean reversion and carry into a single z-scored signal.
    weights: dict with keys 'momentum', 'mean_reversion', 'carry' summing to 1.
    """
    if weights is None:
        weights = {"momentum": 1 / 3, "mean_reversion": 1 / 3, "carry": 1 / 3}

    sig = (
        weights["momentum"] * momentum(prices)
        + weights["mean_reversion"] * mean_reversion(prices)
        + weights["carry"] * carry(fundamentals)
    )
    return sig
