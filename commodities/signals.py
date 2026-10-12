"""
Commodities signals: momentum, carry, mean reversion.

All signals are time-series (directional) — each commodity is long or short
independently. Normalization is handled by ViewFormationTS.
"""

import pandas as pd
import numpy as np
from typing import Optional


def momentum(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    winsor_pct: float = 0.01,
) -> pd.DataFrame:
    """
    Time-series momentum (12-1m) on front-month futures.

    Mirrors equities tsmom: skip=21 removes short-term reversal noise.
    Returns raw winsorized cumulative returns; normalization by ViewFormationTS.
    """
    cumret = prices.shift(1 + skip) / prices.shift(lookback + 1 + skip) - 1
    q_lo = cumret.quantile(winsor_pct, axis=1)
    q_hi = cumret.quantile(1 - winsor_pct, axis=1)
    return cumret.clip(lower=q_lo, upper=q_hi, axis=0)


def carry(
    front: pd.DataFrame,
    second: pd.DataFrame,
    annualize: int = 12,
    winsor_pct: float = 0.01,
) -> pd.DataFrame:
    """
    Roll yield carry signal.

    Signal = log(F1 / F2) × annualize.
    Positive (backwardation) → long, negative (contango) → short.

    annualize: multiplier to convert to annualized units (12 assumes monthly
               contract spacing, which holds for all contracts in this universe).
    Returns raw winsorized carry; normalization by ViewFormationTS.
    """
    roll_yield = np.log(front / second) * annualize
    q_lo = roll_yield.quantile(winsor_pct, axis=1)
    q_hi = roll_yield.quantile(1 - winsor_pct, axis=1)
    return roll_yield.clip(lower=q_lo, upper=q_hi, axis=0)


def mean_reversion(
    prices: pd.DataFrame,
    lookback: int = 21,
    winsor_pct: float = 0.01,
) -> pd.DataFrame:
    """
    Short-term reversal on front-month futures (1-day lag).

    Signal = -(21-day return shifted 1 day). Recent losers → long.
    Returns raw winsorized signal; normalization by ViewFormationTS.
    """
    ret = prices.shift(1) / prices.shift(lookback + 1) - 1
    signal = -ret
    q_lo = signal.quantile(winsor_pct, axis=1)
    q_hi = signal.quantile(1 - winsor_pct, axis=1)
    return signal.clip(lower=q_lo, upper=q_hi, axis=0)


def combined_signal(
    prices: pd.DataFrame,
    term_structure: dict,
    weights: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Equal-weight blend of momentum, carry, and mean reversion.
    weights: dict with keys 'momentum', 'carry', 'mean_reversion' summing to 1.
    """
    if weights is None:
        weights = {"momentum": 1 / 3, "carry": 1 / 3, "mean_reversion": 1 / 3}

    parts = {}
    try:
        parts["momentum"] = momentum(prices)
    except Exception:
        pass
    try:
        parts["carry"] = carry(term_structure["front"], term_structure["second"])
    except Exception:
        pass
    try:
        parts["mean_reversion"] = mean_reversion(prices)
    except Exception:
        pass

    if not parts:
        raise RuntimeError("No commodities signals could be computed")

    available_weight = sum(weights[k] for k in parts)
    return sum(weights[k] / available_weight * v for k, v in parts.items())
