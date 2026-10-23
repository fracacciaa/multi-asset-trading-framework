"""
Risk model: causal EWMA covariance, shrinkage, risk-parity sizing.

Replicates notebook cells 76 (ewma_covariance, shrink_covariance, project_psd)
and 77 (build_all_sizing_weights, solve_risk_parity, sized_portfolio).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from commodities.power_ns.config import (
    TRADING_DAYS,
    TRAIN_END,
    TEST_START,
    COV_HALFLIFE,
    COV_SHRINKAGE,
    COV_MIN_OBS,
    COV_MIN_PAIR_OBS,
    COV_EXTRA_LAG,
    COV_RIDGE,
)
from commodities.power_ns.metrics import sharpe


# ── PSD projection ────────────────────────────────────────────────────────────


def project_psd(S: np.ndarray, floor: float = COV_RIDGE) -> np.ndarray:
    """Symmetrize and clip eigenvalues to obtain a finite PSD matrix."""
    A = np.asarray(S, dtype=float)
    A = np.where(np.isfinite(A), A, 0.0)
    A = 0.5 * (A + A.T)
    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, float(floor))
    out  = (vecs * vals) @ vecs.T
    return 0.5 * (out + out.T)


# ── EWMA covariance ───────────────────────────────────────────────────────────


def ewma_covariance(
    returns: pd.DataFrame,
    halflife: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Causal normalized EWMA covariance plus pair counts.

    Entry i is built from data through i-1 (strictly causal). Pairwise
    missingness can break PSD, so each matrix is projected to the PSD cone.

    Returns
    -------
    path   : ndarray (n_dates, k, k)
    counts : ndarray (n_dates, k, k) int
    """
    lam    = 0.5 ** (1.0 / halflife)
    n, k   = returns.shape
    num    = np.zeros((k, k), dtype=float)
    den    = np.zeros((k, k), dtype=float)
    pair_counts = np.zeros((k, k), dtype=int)
    out    = np.full((n, k, k), np.nan)
    counts = np.zeros((n, k, k), dtype=int)
    X      = returns.values

    for i in range(n):
        current   = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
        out[i]    = project_psd(current)
        counts[i] = pair_counts.copy()

        x    = X[i]
        mask = np.isfinite(x)
        pair = np.outer(mask, mask)
        if pair.any():
            xf    = np.where(mask, x, 0.0)
            outer = np.outer(xf, xf)
            num   = lam * num + (1.0 - lam) * np.where(pair, outer, 0.0)
            den   = lam * den + (1.0 - lam) * pair.astype(float)
            pair_counts += pair.astype(int)

    return out, counts


# ── Shrinkage ─────────────────────────────────────────────────────────────────


def shrink_covariance(
    S: np.ndarray,
    delta: float = COV_SHRINKAGE,
    ridge: float = COV_RIDGE,
) -> np.ndarray:
    """Shrink correlations toward their cross-sectional average, then enforce PSD."""
    S    = project_psd(S)
    d    = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    corr = S / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    off    = corr[np.triu_indices(len(corr), 1)]
    rho    = np.nanmean(off) if np.isfinite(off).any() else 0.0
    target = np.full_like(corr, rho)
    np.fill_diagonal(target, 1.0)
    blended = (1 - delta) * corr + delta * target
    return project_psd(blended * np.outer(d, d), floor=ridge)


# ── Live-set selection ────────────────────────────────────────────────────────


def covariance_live_set(
    i: int,
    covariance_input: pd.DataFrame,
    covariance_path: np.ndarray,
    covariance_pair_counts: np.ndarray,
    universe: list[str],
    min_obs: int = COV_MIN_OBS,
    min_pair_obs: int = COV_MIN_PAIR_OBS,
) -> list[int]:
    """
    Markets with enough own history and enough pair overlap to be included in
    the covariance block at time step i (using t-1 info only).
    """
    live = [
        j for j, m in enumerate(universe)
        if np.isfinite(covariance_input[m].iloc[i])
        and covariance_pair_counts[i, j, j] >= min_obs
    ]
    while len(live) >= 2:
        pc      = covariance_pair_counts[i][np.ix_(live, live)]
        offmask = ~np.eye(len(live), dtype=bool)
        if not offmask.any() or np.nanmin(pc[offmask]) >= min_pair_obs:
            break
        mins = []
        for a in range(len(live)):
            others = np.delete(pc[a], a)
            mins.append(np.min(others) if len(others) else np.inf)
        live.pop(int(np.argmin(mins)))
    return live


# ── Risk-parity solver ────────────────────────────────────────────────────────

_RP_TOL = 1e-3


def solve_risk_parity(
    S: np.ndarray,
    n_iter: int = 200,
    tol: float = _RP_TOL,
) -> Optional[np.ndarray]:
    """
    Long-only equal-risk-contribution weights for one live covariance matrix.

    Returns normalized weight array or None on failure.
    """
    S = np.asarray(S, dtype=float)
    k = 1.0 / np.sqrt(np.clip(np.diag(S), 1e-16, None))

    for _ in range(n_iter):
        rc = k * (S @ k)
        if not np.isfinite(rc).all() or rc.sum() <= 0:
            return None
        share = rc / rc.sum()
        if share.max() - share.min() < tol:
            break
        k = k * (rc.mean() / np.maximum(rc, 1e-16)) ** 0.5
        if not np.isfinite(k).all() or np.abs(k).sum() <= 0:
            return None
        k = k / np.abs(k).sum()

    return _normalize_weights(k)


def _normalize_weights(k: np.ndarray, label=None, clip_events=None) -> Optional[np.ndarray]:
    """Truncate to long-only and normalize to sum 1."""
    k = np.asarray(k, dtype=float)
    if label is not None and clip_events is not None and np.isfinite(k).all() and (k < 0).any():
        clip_events[label] = clip_events.get(label, 0) + 1
    k = np.clip(k, 0.0, None)
    if not np.isfinite(k).all() or k.sum() <= 0:
        return None
    return k / k.sum()


# ── Full sizing weight builder ────────────────────────────────────────────────

SIZING_VARIANTS_DEFAULT = {
    "risk parity":           ("risk_parity", 0.0),
    "equal weight":          ("equal_weight", 0.0),
    "equal Sharpe":          ("equal_sharpe", 0.0),
    "expanding Sharpe":      ("expanding_sharpe", 0.0),
    "expanding, 50% shrunk": ("expanding_sharpe", 0.5),
}


def build_all_sizing_weights(
    strategy_market_returns: pd.DataFrame,
    price_universe: list[str],
    cov_halflife: float = COV_HALFLIFE,
    cov_min_obs: int = COV_MIN_OBS,
    cov_extra_lag: int = COV_EXTRA_LAG,
    train_end: pd.Timestamp = TRAIN_END,
    test_start: pd.Timestamp = TEST_START,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Build all sizing variants in one pass over dates.

    Returns (sizing_weights, covariance_path, covariance_pair_counts).
        sizing_weights : {label: DataFrame(index=RISK_INDEX, columns=universe)}
    """
    RISK_INDEX = strategy_market_returns.index

    covariance_path, covariance_pair_counts = ewma_covariance(
        strategy_market_returns, cov_halflife
    )

    # Cache live sets
    _cov_cache: dict = {}

    def _live_shrunk(i):
        if i not in _cov_cache:
            live = covariance_live_set(
                i, strategy_market_returns, covariance_path, covariance_pair_counts,
                price_universe, COV_MIN_OBS, COV_MIN_PAIR_OBS,
            )
            S = (
                shrink_covariance(covariance_path[i][np.ix_(live, live)])
                if len(live) >= 2 else None
            )
            _cov_cache[i] = (live, S)
        return _cov_cache[i]

    # Expanding Sharpe (used by some variants)
    expanding_sharpe = sharpe(
        strategy_market_returns,
        window="expanding",
        min_periods=cov_min_obs,
    ).shift(1)

    n_dates = len(RISK_INDEX)
    n_mkts  = len(price_universe)
    Karr    = {
        label: np.full((n_dates, n_mkts), np.nan, dtype=float)
        for label in SIZING_VARIANTS_DEFAULT
    }
    exp_s_all = expanding_sharpe.reindex(columns=price_universe).to_numpy(float)

    _clip_events: dict = {}
    _rp_nc = 0

    for i in range(n_dates):
        if i < cov_min_obs:
            continue
        live, S = _live_shrunk(i)
        if len(live) < 2 or S is None:
            continue
        live = np.asarray(live, dtype=int)
        sigma = np.sqrt(np.diag(S))
        exp_s = exp_s_all[i, live]

        candidates: dict = {"equal weight": np.ones(len(live))}

        rp = solve_risk_parity(S)
        if rp is not None:
            candidates["risk parity"] = rp
        else:
            _rp_nc += 1

        rhs_labels = ["equal Sharpe"]
        rhs        = [sigma]

        if np.isfinite(exp_s).all():
            rhs_labels += ["expanding Sharpe", "expanding, 50% shrunk"]
            rhs.append(np.clip(exp_s, 0, None) * sigma)
            shr = 0.5 * exp_s + 0.5 * np.nanmean(exp_s)
            rhs.append(np.clip(shr, 0, None) * sigma)

        solved = np.linalg.solve(S, np.column_stack(rhs))
        for j, label in enumerate(rhs_labels):
            candidates[label] = solved[:, j]

        for label, raw in candidates.items():
            kw = raw if label == "risk parity" else _normalize_weights(raw, label, _clip_events)
            if kw is not None:
                Karr[label][i, live] = kw * len(live)

    sizing_weights: dict[str, pd.DataFrame] = {}
    for label, arr in Karr.items():
        df = pd.DataFrame(arr, index=RISK_INDEX, columns=price_universe)
        sizing_weights[label] = df.shift(cov_extra_lag)

    return sizing_weights, covariance_path, covariance_pair_counts


# ── Sized portfolio runner ────────────────────────────────────────────────────


def sized_portfolio(
    K: pd.DataFrame,
    weights_by_rank: dict,
    run_fn: Callable,
    universe: list[str],
    max_rank: int,
    depth_panel: Optional[pd.DataFrame],
    min_depth: int,
    lag: int = 1,
    cost_bps: float = 0.0,
) -> dict:
    """
    Fold market scalar K into rank weights and run the execution engine.

    K is a DataFrame (dates × markets) of market-level allocation scalars.
    """
    final_weights = {
        k: weights_by_rank[k].reindex(K.index) * K
        for k in range(1, max_rank + 1)
    }
    return run_fn(
        final_weights,
        max_rank=max_rank,
        depth_panel=depth_panel,
        min_depth=min_depth,
        lag=lag,
        cost_bps=cost_bps,
        market_aggregation="equal_available",
    )
