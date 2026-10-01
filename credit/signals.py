"""
Credit signals: momentum, mean reversion, carry.
"""

import pandas as pd


def momentum(spreads: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """
    Momentum on spread changes (tightening = positive for long credit).
    """
    raise NotImplementedError

def mean_reversion(spreads: pd.DataFrame, lookback: int = 63) -> pd.DataFrame:
    """
    Mean reversion: spreads vs. their medium-term moving average.
    """
    raise NotImplementedError

def carry(spreads: pd.DataFrame) -> pd.DataFrame:
    """
    Credit carry: excess spread over the risk-free rate.
    Higher spread = more carry for being long credit.
    """
    raise NotImplementedError

def combined_signal(
    spreads: pd.DataFrame,
    weights: dict | None = None,
) -> pd.DataFrame:
    if weights is None:
        weights = {"momentum": 1 / 3, "mean_reversion": 1 / 3, "carry": 1 / 3}

    sig = (
        weights["momentum"] * momentum(spreads)
        + weights["mean_reversion"] * mean_reversion(spreads)
        + weights["carry"] * carry(spreads)
    )
    return sig
