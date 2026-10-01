"""
Rates signals: momentum, mean reversion, carry.
Signals are expressed in terms of yield changes or futures returns.
"""

import pandas as pd


def momentum(yields: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """
    Trend/momentum on yields or futures prices.
    Positive signal = expect rates to continue in the same direction.
    """
    raise NotImplementedError

def mean_reversion(yields: pd.DataFrame, lookback: int = 63) -> pd.DataFrame:
    """
    Mean reversion on yields relative to a medium-term moving average.
    """
    raise NotImplementedError

def carry(yields: pd.DataFrame) -> pd.DataFrame:
    """
    Carry for rates: roll-down along the yield curve.
    Positive carry = receive (long duration) where the curve is steep.
    """
    raise NotImplementedError

def combined_signal(
    yields: pd.DataFrame,
    weights: dict | None = None,
) -> pd.DataFrame:
    if weights is None:
        weights = {"momentum": 1 / 3, "mean_reversion": 1 / 3, "carry": 1 / 3}

    sig = (
        weights["momentum"] * momentum(yields)
        + weights["mean_reversion"] * mean_reversion(yields)
        + weights["carry"] * carry(yields)
    )
    return sig
