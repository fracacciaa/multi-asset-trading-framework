"""
Carry signal pipeline: raw carry → duration-adjusted score → DV01 positions.

Pipeline:
  1. carry_per_duration  : adjust for the duration-vs-tenor relationship
  2. xs_score            : cross-section across tenors (within one curve)
  3. dv01_positions      : score → DV01-neutral position weights
  4. build_positions     : apply rebalance frequency and mandatory lag

Non-negotiable rules (carry_signal_spec.md §6):
  - Duration adjustment BEFORE cross-sectional ranking.
    Skipping it produces a leveraged duration bet, not a curve carry signal.
  - DV01 neutrality enforced by projection.
    Assert: np.dot(pos, dur) == 0 to within 1e-10.
  - Lag >= 1 is required; no silent zero-lag path.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ── Duration adjustment ────────────────────────────────────────────────────────

def carry_per_duration(
    carry_bp: np.ndarray,
    dur: np.ndarray,
    mode: str = "per_dur",
    vol: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Duration-adjust carry before cross-sectional ranking.

    mode = "none"        : raw carry (control; expected to show high beta_to_duration)
    mode = "per_dur"     : carry / D
    mode = "per_dur_vol" : carry / (D × trailing_vol_yield_change)

    vol : (n,) trailing vol of yield changes in bp, required for per_dur_vol.
    """
    carry_bp = np.asarray(carry_bp, float)
    dur = np.asarray(dur, float)

    if mode == "none":
        return carry_bp.copy()
    elif mode == "per_dur":
        return carry_bp / np.maximum(dur, 1e-8)
    elif mode == "per_dur_vol":
        if vol is None:
            raise ValueError("per_dur_vol requires vol argument (trailing yield-change vol in bp)")
        vol = np.asarray(vol, float)
        return carry_bp / np.maximum(dur * vol, 1e-8)
    else:
        raise ValueError(f"Unknown duration_adj mode: {mode!r}")


# ── Cross-sectional score ──────────────────────────────────────────────────────

def xs_score(c_adj: pd.Series, mode: str = "z") -> pd.Series:
    """
    Cross-sectional score across tenors within one curve, on one date.

    mode = "z"      : z-score (mean 0, std 1)
    mode = "rank"   : rank-normalized (robust to outliers on 8-15 point cross-sections)
    mode = "demean" : demean only, no std scaling

    Demeaning is what converts a level signal into a curve signal.
    """
    if mode == "z":
        s = c_adj.std(ddof=0)
        return (c_adj - c_adj.mean()) / s if s > 1e-10 else pd.Series(0.0, index=c_adj.index)
    elif mode == "rank":
        r = c_adj.rank()
        s = r.std(ddof=0)
        return (r - r.mean()) / s if s > 1e-10 else pd.Series(0.0, index=c_adj.index)
    elif mode == "demean":
        return c_adj - c_adj.mean()
    else:
        raise ValueError(f"Unknown xs_mode: {mode!r}")


# ── DV01-neutral position constructor ─────────────────────────────────────────

def dv01_positions(
    score: pd.Series,
    dur: pd.Series,
    gross: float = 1.0,
    neutralize: tuple = ("level",),
    ns_loadings_slope: Optional[np.ndarray] = None,
) -> pd.Series:
    """
    Convert cross-sectional carry scores to DV01-neutral position weights.

    The projection step removes the component of the position along the
    duration vector, enforcing exact DV01 neutrality.  A 1bp parallel shift
    of every tenor produces zero PnL (tested in test_signal.py).

    neutralize = ("level",)          : DV01-level neutralization only
    neutralize = ("level", "slope")  : also project out the NS slope loading

    gross : target sum of absolute DV01 weights per unit notional.

    Returns a Series of position weights in DV01 units (years-equivalent).
    Positive = long (receiver / buy), negative = short (payer / sell).
    """
    w = score.to_numpy(float).copy()
    d = dur.to_numpy(float)

    # Project out the duration (level) component
    dd = np.dot(d, d)
    if dd > 1e-12:
        w = w - (np.dot(w, d) / dd) * d

    # Optional: also project out the slope loading
    if "slope" in neutralize and ns_loadings_slope is not None:
        s = np.asarray(ns_loadings_slope, float)
        ss = np.dot(s, s)
        if ss > 1e-12:
            w = w - (np.dot(w, s) / ss) * s

    # Scale to target gross DV01
    norm = np.abs(w).sum()
    if norm > 1e-12:
        w = w / norm * gross

    return pd.Series(w, index=score.index)


# ── Rebalance helper ──────────────────────────────────────────────────────────

def _rebalance_dates(dates: pd.DatetimeIndex, freq: str) -> set:
    """Return the set of dates on which positions should be updated."""
    if freq == "daily":
        return set(dates)
    elif freq == "weekly":
        result = set()
        for i, d in enumerate(dates):
            if i == 0 or d.isocalendar().week != dates[i - 1].isocalendar().week:
                result.add(d)
        return result
    elif freq == "monthly":
        result = set()
        for i, d in enumerate(dates):
            if i == 0 or d.month != dates[i - 1].month:
                result.add(d)
        return result
    else:
        raise ValueError(f"Unknown rebalance freq: {freq!r}")


# ── Full position builder ──────────────────────────────────────────────────────

def build_positions(
    carry_adj: pd.DataFrame,
    dur_df: pd.DataFrame,
    rebalance: str,
    lag: int,
    xs_mode: str = "z",
    neutralize: tuple = ("level",),
    gross: float = 1.0,
    vol_df: Optional[pd.DataFrame] = None,
    duration_adj: str = "per_dur",
    ns_slope_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build the DV01-neutral position book, respecting rebalance frequency and lag.

    Parameters
    ----------
    carry_adj  : date × tenor DataFrame of carry measures (bp over h)
    dur_df     : date × tenor DataFrame of modified duration (years)
    rebalance  : "daily" | "weekly" | "monthly"
    lag        : signal lag in trading days (must be >= 1)
    xs_mode    : "z" | "rank" | "demean"
    neutralize : ("level",) or ("level", "slope")
    gross      : target gross DV01 weight
    vol_df     : date × tenor trailing yield-change vol (required if duration_adj="per_dur_vol")
    duration_adj: "none" | "per_dur" | "per_dur_vol"
    ns_slope_df: date × tenor NS slope loadings for slope neutralization

    Returns
    -------
    pos_df: date × tenor DataFrame of DV01 position weights.
            Shifted by `lag` days (signal available at t drives position from t+lag).
    """
    if lag < 1:
        raise ValueError(f"lag must be >= 1, got {lag}")

    tenors = carry_adj.columns
    dates = carry_adj.index
    reb_dates = _rebalance_dates(dates, rebalance)

    pos_rows: dict[pd.Timestamp, pd.Series] = {}
    held: Optional[pd.Series] = None

    for date in dates:
        row_carry = carry_adj.loc[date]
        row_dur = dur_df.loc[date]

        if date in reb_dates:
            # Check we have valid data
            valid = row_carry.notna() & row_dur.notna() & (row_dur > 1e-8)
            if valid.sum() >= 2:
                c = row_carry[valid].to_numpy(float)
                d = row_dur[valid].to_numpy(float)

                # Duration adjustment
                if duration_adj == "per_dur_vol" and vol_df is not None:
                    v = vol_df.loc[date, valid].to_numpy(float) if date in vol_df.index else None
                else:
                    v = None
                c_adj = carry_per_duration(c, d, mode=duration_adj, vol=v)

                # Cross-sectional score
                c_ser = pd.Series(c_adj, index=row_carry[valid].index)
                score = xs_score(c_ser, mode=xs_mode)

                # NS slope loadings if needed
                slope_arr = None
                if "slope" in neutralize and ns_slope_df is not None and date in ns_slope_df.index:
                    slope_arr = ns_slope_df.loc[date, valid].to_numpy(float)

                dur_ser = row_dur[valid]
                held = dv01_positions(
                    score, dur_ser, gross=gross,
                    neutralize=neutralize,
                    ns_loadings_slope=slope_arr,
                )
                # Reindex to all tenors (NaN for missing tenors)
                held = held.reindex(tenors)

        pos_rows[date] = held if held is not None else pd.Series(np.nan, index=tenors)

    pos_df = pd.DataFrame(pos_rows).T
    pos_df.index.name = "date"

    # Mandatory lag: signal at t drives position from t+lag
    return pos_df.shift(lag)
