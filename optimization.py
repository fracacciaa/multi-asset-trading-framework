"""
Portfolio optimization.
Combines cross-asset signals into target portfolio weights.
"""

import pandas as pd
import numpy as np
from typing import Optional


def estimate_covariance(
    returns: pd.DataFrame,
    vol_halflife: int = 21,
    cor_halflife: int = 126,
    shrinkage_intensity: float = 0.0,
) -> pd.DataFrame:
    """
    Dual-halflife EWM covariance: Σ = D · C · D

    vol_halflife:        short halflife for volatility (reacts quickly to regime changes)
    cor_halflife:        longer halflife for correlation (stable cross-asset structure)
    shrinkage_intensity: linear shrinkage of C toward identity; 0 = none, 1 = identity

    Fully vectorized — computes only the final-date EWM estimate (no rolling loop).
    Returns a ticker × ticker covariance DataFrame.
    """
    clean = returns.dropna(axis=1, how="any")
    data  = clean.values.astype(float)
    T, N  = data.shape

    # EWM weights: alpha * (1-alpha)^{T-1-t}, normalized
    def _ewm_weights(halflife: int) -> np.ndarray:
        alpha = 1.0 - np.exp(-np.log(2) / halflife)
        w = (1.0 - alpha) ** np.arange(T - 1, -1, -1)
        w *= alpha
        return w / w.sum()

    # Volatility: sqrt of EWM variance (assume zero mean for daily returns)
    w_v  = _ewm_weights(vol_halflife)
    vols = np.sqrt((data ** 2 * w_v[:, None]).sum(axis=0))

    # Correlation: EWM second-moment matrix with cor_halflife
    w_c     = _ewm_weights(cor_halflife)
    ewm_cov = (data * w_c[:, None]).T @ data        # N × N, vectorized
    diag    = np.diag(ewm_cov)

    # Drop tickers whose EWM second-moment is zero or NaN (stale / zero-return series)
    valid = diag > 1e-12
    if not valid.all():
        data    = data[:, valid]
        ewm_cov = ewm_cov[np.ix_(valid, valid)]
        vols    = vols[valid]
        diag    = diag[valid]
        cols    = clean.columns[valid]
    else:
        cols = clean.columns

    d   = np.sqrt(diag)
    cor = ewm_cov / np.outer(d, d)
    cor = np.clip(cor, -1.0, 1.0)
    np.fill_diagonal(cor, 1.0)

    if shrinkage_intensity > 0:
        cor = (1 - shrinkage_intensity) * cor + shrinkage_intensity * np.eye(cor.shape[0])

    cov = np.outer(vols, vols) * cor
    return pd.DataFrame(cov, index=cols, columns=cols)


def build_views(signals: dict) -> tuple:
    """
    Aggregate per-asset-class signals into a views vector and confidence matrix.
    Placeholder — will delegate to ViewFormationXS once wired up.
    """
    raise NotImplementedError


def optimize(
    mu: pd.Series,
    cov: pd.DataFrame,
    constraints: dict,
    objective: str = "mean_variance",
) -> pd.Series:
    """
    Solve the portfolio optimization problem.
    objective: 'mean_variance' | 'risk_parity' | 'max_sharpe'
    """
    raise NotImplementedError
