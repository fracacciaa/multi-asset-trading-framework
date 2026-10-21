"""
NS 3-factor null-space residual mean-reversion signal (Signal B).

Architecture (from discussion):
  Fit:    NS 3-factor on up to FIT_DEPTH (8) contracts; λ grid-searched daily.
  Trade:  front TRADE_DEPTH (4) contracts only.
  Signal: z-score of N·e where N = null(X_{4×3}ᵀ), λ_trade = 1.7937/M̄_t (1/months).
  Position: hysteresis × null-space direction N.
  PnL:    pos_{t-1} · (N_{t-1}ᵀ Δlog F_{1:4, t}).

Contrast with signal.py (Signal A):
  BC 4-factor | LOO residuals | manual F1/F4/F8 fly | calibrated-once λ

Key design choices:
  - IS (not LOO) residuals: fit on 8 gives enough DOF that IS artefacts are small.
  - λ_trade per-market per-day from closed form: no tuning, exact curvature alignment.
  - Sign of N fixed: first element always positive so alpha = N·e is consistently oriented.
  - Null-space dim = 1 (4 contracts, 3 factors) → unique butterfly direction per day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from commodities.data import fetch_term_structure, fetch_term_structure_far

# ── Constants ─────────────────────────────────────────────────────────────────

FIT_DEPTH   = 8
TRADE_DEPTH = 4
LAM_GRID    = np.arange(0.05, 2.01, 0.05)   # 1/months; covers peak-at-M ∈ [0.9, 36]
MIN_FIT     = 4                               # min valid contracts to attempt fit
Z_EWMA_HL   = 63
Z_MIN_PER   = 30
Z_VOL_FLOOR = 1e-6
HYST_ENTRY  = 1.0


# ── NS design matrix ──────────────────────────────────────────────────────────

def _design(M: np.ndarray, lam) -> np.ndarray:
    """
    NS 3-factor design matrix [1, slope, curvature].
    M: (..., d) maturities in months; lam: broadcastable scalar or (..., 1).
    Returns X: (..., d, 3).
    """
    x = np.maximum(M * lam, 1e-8)
    s = (1.0 - np.exp(-x)) / x
    return np.stack([np.ones_like(M), s, s - np.exp(-x)], axis=-1)


# ── Batched OLS fit ───────────────────────────────────────────────────────────

def _residuals(M: np.ndarray, P: np.ndarray, lam) -> np.ndarray:
    """
    Batched OLS NS residuals.
    M, P: (n, d); lam: scalar or (n, 1).  Returns (n, d) residuals.
    """
    X   = _design(M, lam)                          # (n, d, 3)
    XtX = np.einsum('ndi,ndj->nij', X, X)                    # (n, 3, 3)
    XtP = np.einsum('ndi,nd->ni',   X, P)                   # (n, 3)
    b   = np.linalg.solve(XtX, XtP[:, :, None]).squeeze(-1) # (n, 3)
    return P - np.einsum('ndi,ni->nd', X, b)                 # (n, d)


# ── Data loading ──────────────────────────────────────────────────────────────

def _build_log_curves(n: int = FIT_DEPTH) -> dict[str, pd.DataFrame]:
    ts   = fetch_term_structure()
    coms = list(ts["front"].columns)
    tsf  = fetch_term_structure_far() if n > 4 else None
    k4   = ["front", "second", "third", "fourth"]
    k8   = ["fifth",  "sixth",  "seventh", "eighth"]
    curves: dict[str, pd.DataFrame] = {}
    for com in coms:
        parts = []
        for i, k in enumerate(k4[:min(n, 4)]):
            parts.append(np.log(ts[k][com].replace(0.0, np.nan)).rename(i + 1))
        if tsf is not None:
            for i, k in enumerate(k8[:max(0, n - 4)]):
                parts.append(np.log(tsf[k][com].replace(0.0, np.nan)).rename(i + 5))
        curves[com] = pd.concat(parts, axis=1).sort_index().dropna(how="all")
    return curves


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

def _process(log_curve: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """
    Returns (alpha_raw, N_df):
      alpha_raw : pd.Series (T,)   — raw -N·e (sign-flipped: positive = cheap in N dir)
      N_df      : pd.DataFrame (T, TRADE_DEPTH) — null-space direction, consistent sign
    """
    n_fit   = min(FIT_DEPTH,   log_curve.shape[1])
    n_trade = min(TRADE_DEPTH, log_curve.shape[1])
    T, idx  = len(log_curve), log_curve.index

    alpha_arr = np.full(T, np.nan)
    N_arr     = np.full((T, n_trade), np.nan)

    if n_trade < TRADE_DEPTH:
        return (pd.Series(alpha_arr, index=idx),
                pd.DataFrame(N_arr,  index=idx, columns=range(1, n_trade + 1)))

    P = log_curve.iloc[:, :n_fit].values.astype(float)
    M = np.tile(np.arange(1, n_fit + 1, dtype=float), (T, 1))   # months, fixed

    good = np.all(np.isfinite(P), axis=1)
    if good.sum() < MIN_FIT:
        return (pd.Series(alpha_arr, index=idx),
                pd.DataFrame(N_arr,  index=idx, columns=range(1, n_trade + 1)))

    Pg, Mg = P[good], M[good]

    # ── Grid search: pick λ minimising SSE per day ────────────────────────────
    sse      = np.column_stack([(_residuals(Mg, Pg, lam) ** 2).sum(1) for lam in LAM_GRID])
    best_lam = LAM_GRID[sse.argmin(1)]   # (n_good,)

    # IS residuals with per-day optimal λ
    E_good = np.full_like(Pg, np.nan)
    for lv in np.unique(best_lam):
        m = best_lam == lv
        E_good[m] = _residuals(Mg[m], Pg[m], lv)

    # ── Null-space projection on front TRADE_DEPTH contracts ──────────────────
    E4 = E_good[:, :n_trade]
    M4 = Mg[:,    :n_trade]

    # λ_trade: curvature loading peaks at the mean maturity of the traded strip
    lam_tr = (1.7937 / M4.mean(1, keepdims=True))    # (n_good, 1) 1/months

    X4 = _design(M4, lam_tr)                          # (n_good, 4, 3)
    U  = np.linalg.svd(X4, full_matrices=True)[0]     # (n_good, 4, 4)
    N  = U[:, :, 3]                                    # (n_good, 4) left null space

    # Fix sign convention: first element always positive for consistent orientation
    N *= np.where(N[:, :1] >= 0, 1.0, -1.0)

    alpha = (N * E4).sum(1)    # N · e

    alpha_arr[good] = -alpha   # sign flip: positive = curve cheap in N direction
    N_arr[good]     = N

    return (pd.Series(alpha_arr, index=idx),
            pd.DataFrame(N_arr,  index=idx, columns=range(1, n_trade + 1)))


# ── Public API ────────────────────────────────────────────────────────────────

def compute_pnl(
    n_fit: int = FIT_DEPTH,
    start: Optional[str] = None,
    end:   Optional[str] = None,
) -> pd.DataFrame:
    """
    Per-commodity daily log-return PnL for Signal B (NS null-space, lag-1).

    Position entered at close of t:  pos_t × N_t  (4D weight in F1–F4)
    Earns t→t+1 return:  sum_k N_{t,k} × Δlog F_{k, t+1}
    Returns DataFrame (date × commodity).
    """
    curves = _build_log_curves(n_fit)
    out: dict[str, pd.Series] = {}

    for com, lc in curves.items():
        alpha, N_df = _process(lc)
        if alpha.isna().all():
            continue
        pos   = _hyst(_zscore(alpha))
        ret4  = lc.iloc[:, :TRADE_DEPTH].diff()
        n_ret = (N_df * ret4).sum(axis=1)     # N_{t} · Δlog F_{1:4, t}
        out[com] = pos.shift(1) * n_ret        # lag: position from t earns at t+1

    df = pd.concat(out.values(), axis=1, keys=out.keys())
    df.index.name = "date"
    if start is not None:
        df = df.loc[start:]
    if end is not None:
        df = df.loc[:end]
    return df


def compute_signal(
    n_fit: int = FIT_DEPTH,
    start: Optional[str] = None,
    end:   Optional[str] = None,
) -> pd.DataFrame:
    """Z-scored null-space loading per commodity (pre-positioning)."""
    curves = _build_log_curves(n_fit)
    out: dict[str, pd.Series] = {}
    for com, lc in curves.items():
        alpha, _ = _process(lc)
        if not alpha.isna().all():
            out[com] = _zscore(alpha)
    df = pd.concat(out.values(), axis=1, keys=out.keys())
    df.index.name = "date"
    if start is not None:
        df = df.loc[start:]
    if end is not None:
        df = df.loc[:end]
    return df


def compute_positions(
    n_fit: int = FIT_DEPTH,
    start: Optional[str] = None,
    end:   Optional[str] = None,
) -> pd.DataFrame:
    """Scalar hysteresis positions {-1, 0, +1} per commodity (pre-return application)."""
    curves = _build_log_curves(n_fit)
    out: dict[str, pd.Series] = {}
    for com, lc in curves.items():
        alpha, _ = _process(lc)
        if not alpha.isna().all():
            out[com] = _hyst(_zscore(alpha))
    df = pd.concat(out.values(), axis=1, keys=out.keys())
    df.index.name = "date"
    if start is not None:
        df = df.loc[start:]
    if end is not None:
        df = df.loc[:end]
    return df
