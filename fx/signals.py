"""
FX signals: momentum and carry.
"""

import pandas as pd


def momentum(spot: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """
    Time-series or cross-sectional momentum on FX spot returns.
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
        weights = {"momentum": 0.5, "carry": 0.5}

    sig = (
        weights["momentum"] * momentum(spot)
        + weights["carry"] * carry(forwards, spot)
    )
    return sig
