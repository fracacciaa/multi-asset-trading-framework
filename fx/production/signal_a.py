"""
Signal A: Menkhoff, Sarno, Schmeling & Schrimpf (2012) cross-sectional momentum.

At month-end, rank currencies on past-f-month cumulative excess returns, form
6 portfolios, go long Portfolio 6 (High) and short Portfolio 1 (Low).
Dollar-neutral because the dollar component cancels in the difference.

Spot-only variant (§2.3): sorts on -cumulated spot change instead of rx.
Overlapping-vintage holding: w_h = w.rolling(h).mean().
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ── Portfolio assignment ──────────────────────────────────────────────────────

def _assign_portfolios(row: pd.Series, n_port: int) -> pd.Series:
    v = row.dropna()
    if len(v) < n_port:
        return pd.Series(np.nan, index=row.index)
    labels = pd.qcut(v.rank(method="first"), n_port, labels=False) + 1
    return labels.reindex(row.index)


def menkhoff_portfolios(rx_m: pd.DataFrame, f: int = 6,
                        n_port: int = 6) -> pd.DataFrame:
    """date × pair integer portfolio labels 1..n_port."""
    past = rx_m.rolling(f).sum()
    return past.apply(_assign_portfolios, axis=1, n_port=n_port)


def menkhoff_weights(port: pd.DataFrame, n_port: int = 6,
                     min_size: int = 2) -> pd.DataFrame:
    """
    Long High, short Low, equal-weighted within each leg.

    Sets row to NaN if either leg has fewer than min_size currencies.
    """
    hi = (port == n_port).astype(float)
    lo = (port == 1).astype(float)
    hi_n = hi.sum(axis=1)
    lo_n = lo.sum(axis=1)
    too_thin = (hi_n < min_size) | (lo_n < min_size)
    w = hi.div(hi_n, axis=0) - lo.div(lo_n, axis=0)
    w = w.fillna(0.0)
    w[too_thin] = np.nan
    return w


# ── Main signal builder ───────────────────────────────────────────────────────

def build_signal_a(
    rx_m: pd.DataFrame,
    f: int = 6,
    h: int = 1,
    n_port: int = 6,
    min_size: int = 2,
) -> pd.DataFrame:
    """
    Menkhoff cross-sectional momentum weights (date × pair).

    Parameters
    ----------
    rx_m     Monthly excess returns, date × pair.
    f        Formation period in months.
    h        Holding period in months (overlapping vintages averaged).
    n_port   Number of portfolios (paper uses 6).
    min_size Minimum currencies per leg; otherwise NaN.
    """
    port = menkhoff_portfolios(rx_m, f=f, n_port=n_port)
    w = menkhoff_weights(port, n_port=n_port, min_size=min_size)
    if h > 1:
        w = w.rolling(h).mean()
    return w


def build_signal_a_grid(
    rx_m: pd.DataFrame,
    fs: tuple = (1, 3, 6, 9, 12),
    hs: tuple = (1, 3, 6, 9, 12),
    n_port: int = 6,
) -> dict[tuple[int, int], pd.DataFrame]:
    """All (f, h) combinations. Keys are (formation, holding)."""
    return {(f, h): build_signal_a(rx_m, f=f, h=h, n_port=n_port)
            for f in fs for h in hs}


# ── Daily signal (rolling sum score) ─────────────────────────────────────────

def rolling_mom_score(
    rx_d: pd.DataFrame,
    window: int,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """
    Rolling N-day sum of daily excess returns — the momentum formation score.

    Equivalent to monthly rx_m.rolling(f).sum() but updated every business day
    as the trailing window rolls forward. Preferred over EWMA: daily FX returns
    mean-revert at short lags so an EWMA weights the wrong direction; multi-month
    cumulative returns show positive autocorrelation and this captures that.

    Pass the result to xs_weights() or ts_weights() in formations.py.
    """
    mp = min_periods if min_periods is not None else window // 2
    return rx_d.rolling(window, min_periods=mp).sum()


# ── Daily signal (EWMA score) — kept for reference ────────────────────────────

def build_signal_a_daily(
    rx_d: pd.DataFrame,
    halflife_days: int = 63,
    n_port: int = 6,
    min_size: int = 2,
) -> pd.DataFrame:
    """
    Daily cross-sectional momentum using an EWMA score on daily excess returns.

    Halflife mapping from the monthly formation grid:
      f=1m → 10 days, f=3m → 32, f=6m → 63, f=9m → 95, f=12m → 126.

    The EWMA mean weights recent days more heavily and avoids the cliff-edge
    of a hard rolling window. min_periods=halflife_days suppresses early NaNs
    before the score is meaningful.
    """
    score = rx_d.ewm(halflife=halflife_days, min_periods=halflife_days).mean()
    port = score.apply(_assign_portfolios, axis=1, n_port=n_port)
    return menkhoff_weights(port, n_port=n_port, min_size=min_size)


def build_signal_a_grid_daily(
    rx_d: pd.DataFrame,
    halflives: tuple = (10, 32, 63, 95, 126),
    n_port: int = 6,
) -> dict[int, pd.DataFrame]:
    """All halflife combinations. Keys are halflife in business days."""
    return {hl: build_signal_a_daily(rx_d, halflife_days=hl, n_port=n_port)
            for hl in halflives}


# ── Portfolio return diagnostics ──────────────────────────────────────────────

def portfolio_returns(
    rx_m: pd.DataFrame,
    f: int = 6,
    n_port: int = 6,
) -> pd.DataFrame:
    """
    Return the average monthly excess return for each of the n_port portfolios.

    Used for the monotonicity check (spec §8.3): P1 through P6 should increase.
    Returns a date × portfolio DataFrame.
    """
    port = menkhoff_portfolios(rx_m, f=f, n_port=n_port)
    results: dict[int, pd.Series] = {}
    for p in range(1, n_port + 1):
        mask = port == p
        cnt  = mask.sum(axis=1)
        ret  = (rx_m * mask).sum(axis=1) / cnt.clip(lower=1)
        ret[cnt == 0] = np.nan
        results[p] = ret
    return pd.DataFrame(results)
