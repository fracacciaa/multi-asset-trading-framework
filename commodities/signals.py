"""
Commodities signals: momentum, mean reversion, carry.
"""

import pandas as pd


def momentum(futures: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """
    Price momentum on rolling futures returns.
    """
    raise NotImplementedError

def mean_reversion(futures: pd.DataFrame, lookback: int = 63) -> pd.DataFrame:
    """
    Mean reversion relative to a medium-term moving average.
    """
    raise NotImplementedError

def carry(term_structure: pd.DataFrame) -> pd.DataFrame:
    """
    Commodities carry: roll yield from the futures term structure.
    Positive = backwardation (front > back), negative = contango.
    """
    raise NotImplementedError

def combined_signal(
    futures: pd.DataFrame,
    term_structure: pd.DataFrame,
    weights: dict | None = None,
) -> pd.DataFrame:
    if weights is None:
        weights = {"momentum": 1 / 3, "mean_reversion": 1 / 3, "carry": 1 / 3}

    sig = (
        weights["momentum"] * momentum(futures)
        + weights["mean_reversion"] * mean_reversion(futures)
        + weights["carry"] * carry(term_structure)
    )
    return sig
