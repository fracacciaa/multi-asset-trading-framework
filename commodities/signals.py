"""
Commodities signals: momentum, carry, mean reversion.

All signals are time-series (directional) — each commodity is long or short
independently. Normalization is handled by ViewFormationTS.
"""

import pandas as pd
import numpy as np
from typing import Optional


# ── Sector map for sector-neutral carry ──────────────────────────────────────

COMMODITY_SECTORS: dict[str, str] = {
    # Energy
    "CL": "energy", "HO": "energy", "XB": "energy", "RB": "energy",
    "NG": "energy", "CO": "energy", "QS": "energy", "GO": "energy",
    # Metals
    "GC": "metals", "SI": "metals", "HG": "metals", "PL": "metals",
    "PA": "metals", "LA": "metals", "LX": "metals", "LL": "metals",
    # Grains
    "C":  "grains", "W":  "grains", "S":  "grains", "SM": "grains",
    "BO": "grains", "KW": "grains", "MW": "grains", "RR": "grains",
    # Softs
    "KC": "softs", "CC": "softs", "CT": "softs", "SB": "softs", "OJ": "softs",
    # Livestock
    "LC": "livestock", "LH": "livestock", "FC": "livestock",
}


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


def xs_carry(
    front: pd.DataFrame,
    back: pd.DataFrame,
    tenor: Optional[pd.DataFrame] = None,
    annualize: int = 12,
    winsor_pct: float = 0.01,
) -> pd.DataFrame:
    """
    Cross-sectional carry signal (Gorton/Hayashi/Rouwenhorst 2013).

    carry_i = log(F1_i / FN_i) / tenor_i          if tenor provided (years)
            = log(F1_i / FN_i) * annualize         otherwise

    Demeaned cross-sectionally each day so the signal captures relative
    attractiveness within the universe, not absolute carry level.

    tenor: date × commodity DataFrame in years from tenor_from_tickers().
    """
    if tenor is not None:
        t = tenor.reindex(index=front.index, columns=front.columns)
        raw = np.log(front / back) / t.replace(0, np.nan)
    else:
        raw = np.log(front / back) * annualize
    xs = raw.sub(raw.mean(axis=1), axis=0)
    q_lo = xs.quantile(winsor_pct, axis=1)
    q_hi = xs.quantile(1 - winsor_pct, axis=1)
    return xs.clip(lower=q_lo, upper=q_hi, axis=0)


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


def _seasonal_adjust(df: pd.DataFrame, min_obs: int = 5) -> pd.DataFrame:
    """
    Subtract OOS expanding month-of-year mean from each column (strict lookahead-free).

    For calendar month m, each row in month m has the expanding mean of all
    prior rows in month m subtracted (shift(1) within the month group).
    """
    out = df.copy()
    for m in range(1, 13):
        mask = df.index.month == m
        if mask.sum() < min_obs:
            continue
        monthly_slice = df.loc[mask]
        adj = monthly_slice.expanding(min_periods=min_obs).mean().shift(1)
        out.loc[mask] = df.loc[mask] - adj
    return out


def _near_roll_mask(tickers: pd.DataFrame, days: int = 5) -> pd.DataFrame:
    """
    True on the `days` trading days immediately before a front-contract roll.

    Roll dates are derived from public exchange schedules (tickers[t] !=
    tickers[t+1]), so this is not economic lookahead.
    """
    arr = tickers.to_numpy(dtype=object)
    n, m = arr.shape
    # is_roll[t] = True when arr[t] != arr[t+1] (last day of a contract)
    is_roll = np.zeros((n, m), dtype=bool)
    is_roll[:-1] = (arr[:-1] != arr[1:]) & (arr[:-1] != None) & (arr[1:] != None)  # noqa: E711

    # mask[t] = True if any is_roll[t+k] for k in 1..days
    mask = np.zeros((n, m), dtype=bool)
    for k in range(1, days + 1):
        mask[:n - k] |= is_roll[k:]
    return pd.DataFrame(mask, index=tickers.index, columns=tickers.columns)


def xs_carry_enhanced(
    front: pd.DataFrame,
    back: pd.DataFrame,
    tenor: Optional[pd.DataFrame] = None,
    tickers_front: Optional[pd.DataFrame] = None,
    sector_map: Optional[dict] = None,
    seasonal: bool = True,
    roll_filter_days: int = 5,
    annualize: int = 12,
    winsor_pct: float = 0.01,
) -> pd.DataFrame:
    """
    Enhanced cross-sectional carry with four optional improvements.

    1. Tenor normalisation  : log(F1/FN) / tenor_years  (pass tenor DataFrame)
    2. Seasonal adjustment  : subtract OOS expanding month-of-year mean
    3. Sector-neutral demean: demean within sector before cross-sectional demean
    4. Roll-timing filter   : set carry to NaN in last roll_filter_days before roll
    """
    if tenor is not None:
        t = tenor.reindex(index=front.index, columns=front.columns)
        raw = np.log(front / back) / t.replace(0, np.nan)
    else:
        raw = np.log(front / back) * annualize

    if seasonal:
        raw = _seasonal_adjust(raw)

    if sector_map is not None:
        sec = pd.Series({c: sector_map.get(c, "other") for c in raw.columns})
        demeaned = raw.copy()
        for sector, grp in sec.groupby(sec):
            cols = [c for c in grp.index if c in raw.columns]
            if len(cols) < 2:
                continue
            demeaned[cols] = raw[cols].sub(raw[cols].mean(axis=1), axis=0)
        xs = demeaned.sub(demeaned.mean(axis=1), axis=0)
    else:
        xs = raw.sub(raw.mean(axis=1), axis=0)

    if tickers_front is not None and roll_filter_days > 0:
        near_roll = _near_roll_mask(
            tickers_front.reindex(index=xs.index, columns=xs.columns),
            days=roll_filter_days,
        )
        xs = xs.where(~near_roll)

    q_lo = xs.quantile(winsor_pct, axis=1)
    q_hi = xs.quantile(1 - winsor_pct, axis=1)
    return xs.clip(lower=q_lo, upper=q_hi, axis=0)


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
