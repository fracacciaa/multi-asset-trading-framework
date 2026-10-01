"""
FX signals: momentum, mean reversion, carry.
"""

import pandas as pd


def momentum(spot: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """
    Time-series or cross-sectional momentum on FX spot returns.
    """
    raise NotImplementedError

def mean_reversion(spot: pd.DataFrame, lookback: int = 63) -> pd.DataFrame:
    """
    PPP or short-term mean reversion signal.
    """
    raise NotImplementedError

def carry(forwards: pd.DataFrame, spot: pd.DataFrame) -> pd.DataFrame:
    """
    FX carry: implied yield differential from forward discount/premium.
    Positive signal = long high-yielder, short low-yielder.
    """
    raise NotImplementedError

def combined_signal(
    spot: pd.DataFrame,
    forwards: pd.DataFrame,
    weights: dict | None = None,
) -> pd.DataFrame:
    if weights is None:
        weights = {"momentum": 1 / 3, "mean_reversion": 1 / 3, "carry": 1 / 3}

    sig = (
        weights["momentum"] * momentum(spot)
        + weights["mean_reversion"] * mean_reversion(spot)
        + weights["carry"] * carry(forwards, spot)
    )
    return sig
