"""
Signal C: Moskowitz, Ooi & Pedersen (2012) time-series momentum (TSMOM).

Vol-scaled version (MOP eq 5): sign(past k-month rx) * (40% / sigma) * rx_{t,t+1}
Equal-weight version (Zhang / Sihvonen): +1/N or -1/N per currency.

Both scaled and unscaled are mandatory (spec §3.3). If vol-scaled beats
equal-weight by a wide margin, the difference is the vol-timing bet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# MOP EWMA decay: delta/(1-delta) = 60 days → delta = 60/61
DELTA = 60.0 / 61.0


# ── Ex-ante volatility ────────────────────────────────────────────────────────

def ex_ante_vol(r_daily: pd.DataFrame) -> pd.DataFrame:
    """
    MOP equation (1): EWMA variance of daily excess returns, annualised.

    Uses information up to t-1 (r shifted by 1) to avoid using today's move
    when sizing a position entered at the end of today.
    """
    r = r_daily.shift(1)
    alpha = 1.0 - DELTA
    mu  = r.ewm(alpha=alpha, adjust=False).mean()
    var = ((r - mu) ** 2).ewm(alpha=alpha, adjust=False).mean()
    return np.sqrt(261.0 * var)


def ex_ante_vol_monthly(r_daily: pd.DataFrame) -> pd.DataFrame:
    """Resample ex-ante vol to month-end (take last observation of the month)."""
    return ex_ante_vol(r_daily).resample("BME").last()


# ── TSMOM signals ─────────────────────────────────────────────────────────────

def tsmom_vol_scaled(
    rx_m: pd.DataFrame,
    sigma_m: pd.DataFrame,
    k: int = 12,
    target: float = 0.40,
) -> pd.DataFrame:
    """
    Vol-scaled TSMOM: sign(past-k-month rx) * (target / sigma).

    Position is taken at month-end t for holding month t+1.
    sigma_m is the ex-ante monthly vol aligned to month-end.
    """
    past = rx_m.rolling(k).sum()
    sig  = np.sign(past) * (target / sigma_m.replace(0, np.nan))
    return sig


def tsmom_equal_weight(rx_m: pd.DataFrame, k: int = 12) -> pd.DataFrame:
    """
    Equal-weight TSMOM: +1/N long, -1/N short, based on sign of past-k-month rx.

    Normalises by the number of non-NaN signals at each date so gross
    exposure sums to 1 long + 1 short (same as Menkhoff).
    """
    sgn = np.sign(rx_m.rolling(k).sum())
    denom = sgn.abs().sum(axis=1).replace(0, np.nan)
    return sgn.div(denom, axis=0)


# ── Grid builders ─────────────────────────────────────────────────────────────

def build_signal_c(
    rx_m: pd.DataFrame,
    rx_d: pd.DataFrame,
    k: int = 12,
    h: int = 1,
    target: float = 0.40,
) -> dict[str, pd.DataFrame]:
    """
    Both TSMOM variants for a single (k, h) combination.

    Returns dict with keys 'vol_scaled' and 'equal_weight'.
    h > 1: overlapping vintages averaged (same convention as Signal A).
    """
    sigma_m = ex_ante_vol_monthly(rx_d)

    vs = tsmom_vol_scaled(rx_m, sigma_m, k=k, target=target)
    ew = tsmom_equal_weight(rx_m, k=k)

    if h > 1:
        vs = vs.rolling(h).mean()
        ew = ew.rolling(h).mean()

    return {"vol_scaled": vs, "equal_weight": ew}


def build_signal_c_grid(
    rx_m: pd.DataFrame,
    rx_d: pd.DataFrame,
    ks: tuple = (1, 3, 6, 9, 12),
    hs: tuple = (1, 3, 6, 9, 12),
    target: float = 0.40,
) -> dict[tuple[int, int], dict[str, pd.DataFrame]]:
    """
    All (k, h) combinations. Keys are (formation, holding).
    """
    sigma_m = ex_ante_vol_monthly(rx_d)
    results: dict[tuple[int, int], dict[str, pd.DataFrame]] = {}
    for k in ks:
        for h in hs:
            vs = tsmom_vol_scaled(rx_m, sigma_m, k=k, target=target)
            ew = tsmom_equal_weight(rx_m, k=k)
            if h > 1:
                vs = vs.rolling(h).mean()
                ew = ew.rolling(h).mean()
            results[(k, h)] = {"vol_scaled": vs, "equal_weight": ew}
    return results
