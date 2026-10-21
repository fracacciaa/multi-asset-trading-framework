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


def tsmom(
    prices: pd.DataFrame,
    lookbacks: tuple = (21, 63, 252),
    mode: str = "sign",
    clip: float = 2.0,
    vol_com: int = 60,
    winsor_pct: float = 0.01,
) -> pd.DataFrame:
    """
    Multi-horizon time-series momentum (Hurst/Ooi/Pedersen 2017).

    mode="sign": equal-weight average of sign(cumret_L) across lookbacks.
    mode="clip": cumret_L / (sigma * sqrt(L/252)) clipped to ±clip, then averaged.
    Returns raw signal; normalisation by vol_scale in backtest.py.
    """
    r0 = prices.pct_change().fillna(0.0)
    sigma = (r0.ewm(com=vol_com, min_periods=max(120, 2 * vol_com))
              .std(bias=False) * np.sqrt(252))

    parts = []
    for L in lookbacks:
        cumret = prices / prices.shift(L) - 1
        if mode == "sign":
            s = np.sign(cumret)
        else:
            denom = sigma * np.sqrt(L / 252)
            s = (cumret / denom.replace(0, np.nan)).clip(-clip, clip)
        q_lo = s.quantile(winsor_pct, axis=1)
        q_hi = s.quantile(1 - winsor_pct, axis=1)
        parts.append(s.clip(lower=q_lo, upper=q_hi, axis=0))

    return sum(parts) / len(parts)


def ma_crossover(
    prices: pd.DataFrame,
    fast: int = 20,
    slow: int = 120,
    winsor_pct: float = 0.01,
) -> pd.DataFrame:
    """
    Moving-average crossover signal (Szakmary et al. 2010).

    sign(SMA_fast − SMA_slow) on log-prices.
    """
    lp = np.log(prices)
    s = np.sign(lp.rolling(fast, min_periods=fast).mean()
                - lp.rolling(slow, min_periods=slow).mean())
    q_lo = s.quantile(winsor_pct, axis=1)
    q_hi = s.quantile(1 - winsor_pct, axis=1)
    return s.clip(lower=q_lo, upper=q_hi, axis=0)


def channel_breakout(
    prices: pd.DataFrame,
    window: int = 55,
    winsor_pct: float = 0.01,
) -> pd.DataFrame:
    """
    Channel-breakout trend signal (Szakmary et al. 2010).

    +1 when price exceeds the rolling window-day high (prior days only),
    −1 when below the rolling window-day low; otherwise persist.
    """
    hi = prices.shift(1).rolling(window, min_periods=window).max()
    lo = prices.shift(1).rolling(window, min_periods=window).min()
    up = (prices > hi).astype(float)
    dn = (prices < lo).astype(float) * -1
    change = up + dn
    # persist last non-zero direction where change == 0
    out = change.replace(0, np.nan).ffill().fillna(0.0)
    q_lo = out.quantile(winsor_pct, axis=1)
    q_hi = out.quantile(1 - winsor_pct, axis=1)
    return out.clip(lower=q_lo, upper=q_hi, axis=0)


def curve_slope(
    front: pd.DataFrame,
    back: pd.DataFrame,
    annualize: int = 12,
    winsor_pct: float = 0.01,
) -> pd.DataFrame:
    """
    Futures curve slope: log(front / back) × annualize (Gorton et al. 2013).

    Generalisation of carry() — works with any two contract legs.
    Positive = backwardation (front premium), negative = contango.
    """
    slope = np.log(front / back) * annualize
    q_lo = slope.quantile(winsor_pct, axis=1)
    q_hi = slope.quantile(1 - winsor_pct, axis=1)
    return slope.clip(lower=q_lo, upper=q_hi, axis=0)


def curve_curvature(
    front: pd.DataFrame,
    mid: pd.DataFrame,
    back: pd.DataFrame,
    winsor_pct: float = 0.01,
) -> pd.DataFrame:
    """
    Futures curve butterfly curvature (Szymanowska et al. 2014).

    log(front) + log(back) − 2 × log(mid).
    Positive = humped curve (front and back above mid).
    """
    curv = np.log(front) + np.log(back) - 2 * np.log(mid)
    q_lo = curv.quantile(winsor_pct, axis=1)
    q_hi = curv.quantile(1 - winsor_pct, axis=1)
    return curv.clip(lower=q_lo, upper=q_hi, axis=0)


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
