"""
Equities signals: momentum, mean reversion, carry.
All signals return a cross-sectional winsorized DataFrame indexed date × ticker.
Normalization (z-score, Grinold, beta-orthogonalization) is handled by ViewFormationXS.
"""

import pandas as pd
import numpy as np
from typing import Optional

from equities.sectors import GICS, FINANCIALS_TICKERS


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
                formation window. Set to None for plain cumulative return.
    winsor_pct: symmetric cross-sectional winsorization percentile (default 1%)

    Returns date × ticker DataFrame of winsorized raw returns (not z-scored).
    All normalization is delegated to ViewFormationXS.
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

    return cumret


def tsmom(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    winsor_pct: float = 0.01,
) -> pd.DataFrame:
    """
    Time-series momentum signal (absolute directional).

    skip=21 (default): mirrors CSMOM's 12-1m convention, removes short-term
    reversal noise — empirically better than skip=0 across train/val periods.
    Returns raw winsorized cumulative returns; normalization by ViewFormationTS.
    """
    cumret = prices.shift(1 + skip) / prices.shift(lookback + 1 + skip) - 1
    q_lo = cumret.quantile(winsor_pct, axis=1)
    q_hi = cumret.quantile(1 - winsor_pct, axis=1)
    return cumret.clip(lower=q_lo, upper=q_hi, axis=0)


def mean_reversion(
    prices: pd.DataFrame,
    lookback: int = 21,
    winsor_pct: float = 0.01,
) -> pd.DataFrame:
    """
    Short-term reversal signal.

    Signal = negated lookback-day return (1-day lag to avoid bid-ask bounce).
    Recent losers → positive (buy), recent winners → negative (sell).
    Returns raw winsorized signal; normalization delegated to ViewFormationXS.
    """
    ret = prices.shift(1) / prices.shift(lookback + 1) - 1
    signal = -ret
    q_lo = signal.quantile(winsor_pct, axis=1)
    q_hi = signal.quantile(1 - winsor_pct, axis=1)
    return signal.clip(lower=q_lo, upper=q_hi, axis=0)


def bollinger(
    prices: pd.DataFrame,
    halflife: int = 21,
    winsor_pct: float = 0.01,
) -> pd.DataFrame:
    """
    Bollinger-style deviation from EWMA trend.

    Signal = -(price / ewma_price - 1): below trend → positive (buy),
    above trend → negative (sell). halflife controls trend smoothing
    (not vol estimation — conceptually separate).
    Returns raw winsorized signal; normalization delegated to ViewFormationXS.
    """
    ewma_price = prices.ewm(halflife=halflife).mean()
    signal = -(prices / ewma_price - 1)
    q_lo = signal.quantile(winsor_pct, axis=1)
    q_hi = signal.quantile(1 - winsor_pct, axis=1)
    return signal.clip(lower=q_lo, upper=q_hi, axis=0)

def carry(
    prices: pd.DataFrame,
    fundamentals: dict,
    fcf_lag: int = 63,
    sector_adjust: str = "demean",
    winsor_pct: float = 0.01,
) -> pd.DataFrame:
    """
    Two-component equity carry signal.

    Component 1 — FCF yield = TTM FCF / market cap (lagged fcf_lag days).
      For Financials: earnings yield = BEST_EPS / PX_LAST (no lag).

    Component 2 — Net buyback yield = -Δlog(market_cap/price) over 252 days.
      Positive = buybacks (share count falling), negative = dilution.

    sector_adjust controls how sector bets are handled:
      "zscore"  — z-score within sector (XS signal: relative within-sector bet)
      "demean"  — subtract sector mean only (TS signal: preserves within-sector
                  magnitude; removes energy-always-cheap / tech-always-expensive bias)
      "none"    — no sector adjustment (pure TS: sector-concentrated but directional)

    After sector adjustment, each component is scaled by its cross-sectional std
    so the two are comparable before averaging. Composite is winsorized.
    """
    mkt_cap = fundamentals["market_cap"]
    fcf     = fundamentals["fcf"]
    eps_fwd = fundamentals["eps_fwd"]

    # ── Component 1: yield ──────────────────────────────────────────────────
    fcf_yield = fcf.shift(fcf_lag) / mkt_cap
    eps_yield = eps_fwd / prices

    fins     = [t for t in prices.columns if t in FINANCIALS_TICKERS]
    non_fins = [t for t in prices.columns if t not in FINANCIALS_TICKERS]

    yield_signal = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    if non_fins:
        yield_signal[non_fins] = fcf_yield.reindex(columns=non_fins)
    if fins:
        yield_signal[fins] = eps_yield.reindex(columns=fins)

    # ── Component 2: net buyback yield ──────────────────────────────────────
    adj_shares    = mkt_cap / prices
    buyback_yield = -np.log(adj_shares / adj_shares.shift(252))

    # ── Winsorize raw components before sector adjustment ───────────────────
    # Essential for TS processing: extreme FCF/price ratios and large share-count
    # swings (mergers, recapitalisations) otherwise survive sector demeaning and
    # produce catastrophic weights after vol scaling.
    def _winsorize(df: pd.DataFrame) -> pd.DataFrame:
        q_lo = df.quantile(winsor_pct, axis=1)
        q_hi = df.quantile(1 - winsor_pct, axis=1)
        return df.clip(lower=q_lo, upper=q_hi, axis=0)

    yield_signal  = _winsorize(yield_signal)
    buyback_yield = _winsorize(buyback_yield)

    # ── Sector adjustment ────────────────────────────────────────────────────
    sector_map = pd.Series(GICS).reindex(prices.columns)

    def _sector_op(df: pd.DataFrame, op: str) -> pd.DataFrame:
        out = df.copy()
        for sector in sector_map.dropna().unique():
            cols = [c for c in sector_map[sector_map == sector].index if c in df.columns]
            if not cols:
                continue
            sub = df[cols]
            mu  = sub.mean(axis=1)
            out[cols] = sub.sub(mu, axis=0)
            if op == "zscore":
                sig = sub.std(axis=1).replace(0, np.nan)
                out[cols] = out[cols].div(sig, axis=0)
        return out

    if sector_adjust == "zscore":
        adj_yield    = _sector_op(yield_signal,  "zscore")
        adj_buyback  = _sector_op(buyback_yield, "zscore")
    elif sector_adjust == "demean":
        adj_yield    = _sector_op(yield_signal,  "demean")
        adj_buyback  = _sector_op(buyback_yield, "demean")
    else:  # "none"
        adj_yield    = yield_signal.copy()
        adj_buyback  = buyback_yield.copy()

    # Scale each component by its cross-sectional std so they're comparable
    # before averaging (FCF yield and buyback yield have different natural scales)
    def _xscale(df: pd.DataFrame) -> pd.DataFrame:
        std = df.std(axis=1).replace(0, np.nan)
        return df.div(std, axis=0)

    adj_yield   = _xscale(adj_yield)
    adj_buyback = _xscale(adj_buyback)

    # ── Composite: average, fall back to available component ────────────────
    composite = (adj_yield + adj_buyback) / 2
    composite = composite.fillna(adj_yield).fillna(adj_buyback)

    # ── Winsorize ────────────────────────────────────────────────────────────
    q_lo = composite.quantile(winsor_pct, axis=1)
    q_hi = composite.quantile(1 - winsor_pct, axis=1)
    return composite.clip(lower=q_lo, upper=q_hi, axis=0)

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
