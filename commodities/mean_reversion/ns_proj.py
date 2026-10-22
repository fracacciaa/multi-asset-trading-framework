"""
NS 3-factor null-space residual mean-reversion signal (Signal B).

Signal = -N · log F_{1:4}   where N = null(X_{4×3}ᵀ), X built with fixed
λ_trade = 1.7937 / 2.5 months⁻¹ (curvature peak at midpoint of traded strip).

N is constant (X4 depends only on fixed maturities [1,2,3,4]), so the signal
reduces to a fixed linear combination of four log prices — no NS fitting needed.

Position: hysteresis on z-score (enter |z| > 1, exit when z crosses zero).
PnL:      pos_{t-1} · (N^T Δlog F_{1:4, t}).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from commodities.data import fetch_term_structure, fetch_tickers, business_days_to_month_end_panel

# ── Constants ─────────────────────────────────────────────────────────────────

TRADE_DEPTH     = 4
LAM_TRADE       = 1.7937 / np.arange(1, TRADE_DEPTH + 1).mean()  # ≈ 0.7175 /month
Z_EWMA_HL       = 63
Z_MIN_PER       = 30
Z_VOL_FLOOR     = 1e-6
HYST_ENTRY      = 1.0
ROLL_OUT_BUFFER = 2  # business days before month-end to flatten entire butterfly


# ── NS design matrix ──────────────────────────────────────────────────────────

def _design(M: np.ndarray, lam: float) -> np.ndarray:
    """NS 3-factor design matrix [level, slope, curvature] for maturities M."""
    x = np.maximum(M * lam, 1e-8)
    s = (1.0 - np.exp(-x)) / x
    return np.stack([np.ones_like(M), s, s - np.exp(-x)], axis=-1)


# ── Null-space direction (computed once) ──────────────────────────────────────

def _build_N() -> np.ndarray:
    """
    Unit-norm null-space vector of X4^T, where X4 = _design([1,2,3,4], LAM_TRADE).

    N is the unique direction orthogonal to all three NS factors on the
    [1,2,3,4]-month strip.  Sign fixed so N[0] > 0.
    """
    M4 = np.arange(1, TRADE_DEPTH + 1, dtype=float)
    X4 = _design(M4, LAM_TRADE)                   # (4, 3)
    U  = np.linalg.svd(X4, full_matrices=True)[0]  # (4, 4)
    N  = U[:, 3]                                    # last left singular vector
    return N if N[0] >= 0 else -N


_N = _build_N()   # computed once at import time


# ── Data loading ──────────────────────────────────────────────────────────────

_KEYS = ["front", "second", "third", "fourth"]


def _build_log_curves() -> dict[str, pd.DataFrame]:
    ts   = fetch_term_structure()
    curves: dict[str, pd.DataFrame] = {}
    for com in ts["front"].columns:
        parts = [np.log(ts[k][com].replace(0.0, np.nan)).rename(i + 1)
                 for i, k in enumerate(_KEYS)]
        curves[com] = pd.concat(parts, axis=1).sort_index().dropna(how="all")
    return curves


def _specific_butterfly_return(
    com: str,
    index: pd.Index,
    ts: dict,
    tickers: dict,
) -> pd.Series:
    """
    N^T Δlog F using specific-contract continuity + all-or-nothing roll buffer.

    For leg j, find the price of the contract held at t-1 by scanning all 4
    generic ranks on day t (cross-rank lookup).  Entire butterfly is NaN if ANY
    of F1–F4 is within ROLL_OUT_BUFFER business days of its delivery month end.
    """
    n_ret = pd.Series(0.0, index=index, dtype=float)
    tickers_4: dict[str, pd.Series] = {}

    for j, key_j in enumerate(_KEYS):
        if key_j not in ts or com not in ts[key_j].columns:
            return pd.Series(np.nan, index=index)
        if key_j not in tickers or com not in tickers[key_j].columns:
            return pd.Series(np.nan, index=index)

        tk_held = tickers[key_j][com].reindex(index).shift(1)   # contract held yesterday
        tickers_4[key_j] = tickers[key_j][com].reindex(index)

        # Cross-rank lookup: find today's price of yesterday's held contract
        found_price = pd.Series(np.nan, index=index, dtype=float)
        for key_r in _KEYS:
            if key_r not in ts or com not in ts[key_r].columns:
                continue
            if key_r not in tickers or com not in tickers[key_r].columns:
                continue
            match = tickers[key_r][com].reindex(index).eq(tk_held) & tk_held.notna()
            found_price = found_price.combine_first(ts[key_r][com].reindex(index).where(match))

        price_prev = ts[key_j][com].reindex(index).shift(1)
        n_ret = n_ret + _N[j] * np.log(found_price / price_prev)

    # All-or-nothing buffer: flatten entire butterfly if any leg is near roll
    bdte = business_days_to_month_end_panel(pd.DataFrame(tickers_4).reindex(index))
    near_roll = (bdte <= ROLL_OUT_BUFFER).any(axis=1)

    return n_ret.where(~near_roll)


# ── Z-score and positioning ───────────────────────────────────────────────────

def _zscore(s: pd.Series) -> pd.Series:
    vol = s.shift(1).ewm(halflife=Z_EWMA_HL, min_periods=Z_MIN_PER).std()
    return s / vol.clip(lower=Z_VOL_FLOOR)


def _hyst(z: pd.Series) -> pd.Series:
    pos, cur = np.zeros(len(z)), 0.0
    for i, v in enumerate(z.values):
        if np.isnan(v):
            pos[i] = np.nan
            continue
        if cur == 0.0:
            if   v >  HYST_ENTRY: cur =  1.0
            elif v < -HYST_ENTRY: cur = -1.0
        else:
            if cur > 0 and v <= 0: cur = 0.0
            elif cur < 0 and v >= 0: cur = 0.0
        pos[i] = cur
    return pd.Series(pos, index=z.index)


# ── Per-commodity pipeline ────────────────────────────────────────────────────

def _alpha(log_curve: pd.DataFrame) -> pd.Series:
    """
    Raw signal: alpha(t) = -N · log F_{1:4}(t).

    NaN on days where any of the four log prices is missing.
    """
    P4  = log_curve.iloc[:, :TRADE_DEPTH].values.astype(float)
    raw = -(P4 @ _N)
    raw[~np.all(np.isfinite(P4), axis=1)] = np.nan
    return pd.Series(raw, index=log_curve.index)


# ── Public API ────────────────────────────────────────────────────────────────

def compute_signal(
    start: Optional[str] = None,
    end:   Optional[str] = None,
) -> pd.DataFrame:
    """Z-scored null-space signal per commodity (pre-positioning)."""
    curves = _build_log_curves()
    out = {com: _zscore(_alpha(lc))
           for com, lc in curves.items()
           if not _alpha(lc).isna().all()}
    df = pd.concat(out.values(), axis=1, keys=out.keys())
    df.index.name = "date"
    return df.loc[start:end] if start or end else df


def compute_positions(
    start: Optional[str] = None,
    end:   Optional[str] = None,
) -> pd.DataFrame:
    """Hysteresis positions {-1, 0, +1} per commodity."""
    curves = _build_log_curves()
    out = {com: _hyst(_zscore(_alpha(lc)))
           for com, lc in curves.items()
           if not _alpha(lc).isna().all()}
    df = pd.concat(out.values(), axis=1, keys=out.keys())
    df.index.name = "date"
    return df.loc[start:end] if start or end else df


def compute_pnl(
    start: Optional[str] = None,
    end:   Optional[str] = None,
) -> pd.DataFrame:
    """
    Per-commodity daily log-return PnL for Signal B (lag-1).

    Position entered at close of t:  pos_t × N  (4D weight in F1–F4)
    Earns t→t+1 return:  N^T Δlog F_{1:4, t+1}  (specific-contract continuity)

    Roll handling: specific-contract cross-rank lookup per leg; entire butterfly
    NaN when any leg is within ROLL_OUT_BUFFER business days of delivery month end.
    """
    curves  = _build_log_curves()
    ts      = fetch_term_structure()
    tickers = fetch_tickers()

    out: dict[str, pd.Series] = {}
    for com, lc in curves.items():
        a = _alpha(lc)
        if a.isna().all():
            continue
        pos   = _hyst(_zscore(a))
        n_ret = _specific_butterfly_return(com, lc.index, ts, tickers)
        out[com] = pos.shift(1) * n_ret

    df = pd.concat(out.values(), axis=1, keys=out.keys())
    df.index.name = "date"
    return df.loc[start:end] if start or end else df
