"""
Signal C: Durham (2013) duration-neutral cross-sectional momentum.

Applied per market (per OIS curve).  The signal is cross-sectional across
six duration buckets on one curve — not cross-market.

Buckets (spec §4.1):
  1-3Y, 3-5Y, 5-7Y, 7-10Y, 10-20Y, 20-30Y
  Midpoints: 2, 4, 6, 8.5, 15, 25 years.

Momentum measure: skip-one-month average of bucket returns over t-2 .. t-n
(spec §4.2).  Primary n=12 (11-month window), variants n in {5..13}.

Two portfolio variants:
  long_only   max w'Rbar s.t. 0 ≤ w_i ≤ 1, w'D = d_index, sum(w) = 1
  long_short  max w'Rbar s.t. -1 ≤ w_i ≤ 1, w'D = 0
              (no sum constraint: zero-duration swap book, natural for swaps)

Benchmark duration d_index:
  "equal"   equal-weighted average of the six bucket durations (default)
  "fixed"   constant, e.g. 6.0 years

Expect corner solutions: the LP with box + two equality constraints optimises
at a vertex, so typically 2-3 buckets are loaded with the rest at 0 or bound.

Spec: ois_momentum_signals_spec.md §4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

try:
    from scipy.optimize import linprog
except ImportError as exc:
    raise ImportError("Signal C requires scipy.optimize.linprog") from exc

from rates.production.momentum.returns import swap_duration, _dfs_from_zero_row

# ── Bucket definitions ────────────────────────────────────────────────────────

BUCKETS: list[tuple[float, float]] = [
    (1.0, 3.0), (3.0, 5.0), (5.0, 7.0),
    (7.0, 10.0), (10.0, 20.0), (20.0, 30.0),
]
MIDPOINTS: list[float] = [0.5 * (lo + hi) for lo, hi in BUCKETS]   # 2,4,6,8.5,15,25

N_BUCKETS = len(BUCKETS)


# ── Momentum measure ──────────────────────────────────────────────────────────

def durham_momentum(ret_m_bp: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    """
    Skip-one-month rolling average of bucket returns (spec eq, §4.2).

    Window: t-2 through t-n (skips t-1 month).  With n=12 the window is
    11 months long.  Durham: "average total returns from between two through,
    alternatively, five to 13 months prior."
    """
    return ret_m_bp.shift(2).rolling(n - 1, min_periods=max(1, (n - 1) // 2)).mean()


# ── LP portfolio constructors ─────────────────────────────────────────────────

def durham_long_only(
    rbar: np.ndarray,
    dur: np.ndarray,
    d_index: float,
) -> np.ndarray:
    """
    Long-only duration-matched LP (spec eq §4.3).

    max w'rbar  s.t.  0 ≤ w_i ≤ 1,  w'D = d_index,  sum(w) = 1

    The sum(w)=1 constraint is implied by the paper ("portfolio allocation with
    a benchmark").  Flag this in the manifest as an assumption.
    """
    rbar = np.asarray(rbar, float)
    dur  = np.asarray(dur, float)

    if not np.all(np.isfinite(rbar)) or not np.all(np.isfinite(dur)):
        return np.full(N_BUCKETS, np.nan)

    res = linprog(
        c=-rbar,
        A_eq=np.vstack([dur, np.ones(N_BUCKETS)]),
        b_eq=np.array([d_index, 1.0]),
        bounds=[(0.0, 1.0)] * N_BUCKETS,
        method="highs",
    )
    return res.x if res.success else np.full(N_BUCKETS, np.nan)


def durham_long_short(
    rbar: np.ndarray,
    dur: np.ndarray,
) -> np.ndarray:
    """
    Long-short duration-neutral LP (spec eq §4.4).

    max w'rbar  s.t.  -1 ≤ w_i ≤ 1,  w'D = 0

    No sum(w) constraint: the portfolio is zero-duration, not dollar-neutral.
    For an unfunded swap book this is the natural market-neutral constraint.
    """
    rbar = np.asarray(rbar, float)
    dur  = np.asarray(dur, float)

    if not np.all(np.isfinite(rbar)) or not np.all(np.isfinite(dur)):
        return np.full(N_BUCKETS, np.nan)

    res = linprog(
        c=-rbar,
        A_eq=dur.reshape(1, -1),
        b_eq=np.array([0.0]),
        bounds=[(-1.0, 1.0)] * N_BUCKETS,
        method="highs",
    )
    return res.x if res.success else np.full(N_BUCKETS, np.nan)


# ── Panel builder ─────────────────────────────────────────────────────────────

def _bucket_returns(
    ret_m: pd.DataFrame,
    zero_m: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Interpolate bucket-midpoint returns and durations from the available grid.

    Returns (ret_buckets, dur_buckets) both date × 6 DataFrames.
    """
    ret_rows: dict = {}
    dur_rows: dict = {}

    par_tenors = ret_m.columns.to_numpy(float)
    par_dates  = ret_m.index

    for t in par_dates:
        ret_row  = ret_m.loc[t].dropna()
        zero_row = zero_m.loc[t].dropna() if t in zero_m.index else None

        if zero_row is None or ret_row.empty:
            continue

        dfs_t = _dfs_from_zero_row(zero_row)
        r_vals = ret_row.to_numpy(float)
        r_tens = ret_row.index.to_numpy(float)

        bucket_ret = {}
        bucket_dur = {}

        for tau in MIDPOINTS:
            # Return at midpoint: linear interpolation of cm returns
            if tau < r_tens[0] or tau > r_tens[-1]:
                bucket_ret[tau] = np.nan
                bucket_dur[tau] = np.nan
                continue
            bucket_ret[tau] = float(np.interp(tau, r_tens, r_vals))
            bucket_dur[tau] = swap_duration(dfs_t, tau)

        ret_rows[t] = bucket_ret
        dur_rows[t] = bucket_dur

    ret_b = pd.DataFrame(ret_rows).T
    dur_b = pd.DataFrame(dur_rows).T
    ret_b.index.name = "date"
    dur_b.index.name = "date"
    return ret_b, dur_b


def build_signal_c(
    panels_by_market: dict[str, dict[str, pd.DataFrame]],
    d_index_method: str = "equal",
    d_index_fixed: float = 6.0,
    n: int = 12,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Build Signal C (Durham LP) for every market.

    Parameters
    ----------
    panels_by_market  : output of build_all_return_panels()
    d_index_method    : "equal" (default) or "fixed"
    d_index_fixed     : used when d_index_method="fixed" (years, e.g. 6.0)
    n                 : Durham lookback window (default 12 → 11-month window,
                        t-2 through t-12)

    Returns
    -------
    {market: {"long_only": df, "long_short": df}}

    Each DataFrame is date × MIDPOINTS (6 buckets), values = portfolio weights.
    Positive weight = long receiver = long duration at that bucket tenor.
    Weights are raw LP outputs; no vol-scaling or lag applied.
    Apply lag >= 1 before using in a backtest.
    """
    result: dict = {}

    for market, panels in panels_by_market.items():
        ret_m  = panels["ret_m"]
        zero_m = panels["z_m"]

        if ret_m.empty:
            continue

        ret_b, dur_b = _bucket_returns(ret_m, zero_m)

        if ret_b.empty:
            continue

        rbar = durham_momentum(ret_b, n=n)

        lo_rows: dict = {}
        ls_rows: dict = {}

        dates = rbar.dropna(how="all").index

        for t in dates:
            rbar_t = rbar.loc[t].to_numpy(float)
            dur_t  = dur_b.loc[t].to_numpy(float) if t in dur_b.index else np.full(N_BUCKETS, np.nan)

            if not np.all(np.isfinite(rbar_t)) or not np.all(np.isfinite(dur_t)):
                continue

            if d_index_method == "equal":
                d_idx = float(np.mean(dur_t))
            else:
                d_idx = d_index_fixed

            lo_rows[t] = durham_long_only(rbar_t, dur_t, d_idx)
            ls_rows[t] = durham_long_short(rbar_t, dur_t)

        lo_df = pd.DataFrame(lo_rows, index=MIDPOINTS).T
        ls_df = pd.DataFrame(ls_rows, index=MIDPOINTS).T
        lo_df.index.name = "date"
        ls_df.index.name = "date"

        result[market] = {"long_only": lo_df, "long_short": ls_df}

    return result
