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
    shrinkage: bool = False,
) -> pd.DataFrame:
    """
    Dual-halflife EWM covariance: Σ = D · C · D

    vol_halflife: short halflife for volatility (reacts quickly to regime changes)
    cor_halflife: longer halflife for correlation (stable cross-asset structure)
    shrinkage:    if True, apply Ledoit-Wolf analytical shrinkage to the correlation matrix
                  before combining — reduces estimation error for large universes

    Returns a ticker × ticker covariance DataFrame.
    """
    # Drop tickers with insufficient history
    clean = returns.dropna(axis=1, how="any")

    vols = clean.ewm(halflife=vol_halflife, min_periods=vol_halflife).std().iloc[-1]

    # EWM correlation — pandas returns a MultiIndex; take the last date's slice
    ewm_cor = clean.ewm(halflife=cor_halflife, min_periods=cor_halflife).corr()
    last_date = ewm_cor.index.get_level_values(0)[-1]
    cor_mat = ewm_cor.loc[last_date]

    if shrinkage:
        cor_mat = _ledoit_wolf_corr(cor_mat)

    # Ensure positive definiteness via diagonal floor
    cor_array = np.clip(cor_mat.values, -1.0, 1.0)
    np.fill_diagonal(cor_array, 1.0)

    cov_array = np.outer(vols.values, vols.values) * cor_array
    return pd.DataFrame(cov_array, index=clean.columns, columns=clean.columns)


def _ledoit_wolf_corr(cor: pd.DataFrame) -> pd.DataFrame:
    """Apply Ledoit-Wolf shrinkage to a correlation matrix (shrink toward identity)."""
    from sklearn.covariance import LedoitWolf
    n = cor.shape[0]
    # LW requires a sample covariance as input — we pass the correlation matrix as-is
    # and re-extract the shrunk correlation after fitting
    lw = LedoitWolf(assume_centered=True)
    # Simulate a dataset whose sample correlation matches cor
    rng = np.random.default_rng(42)
    samples = rng.multivariate_normal(np.zeros(n), cor.values, size=max(500, n * 3))
    lw.fit(samples)
    shrunk = lw.covariance_
    # Convert shrunk covariance back to correlation
    d = np.sqrt(np.diag(shrunk))
    shrunk_cor = shrunk / np.outer(d, d)
    np.fill_diagonal(shrunk_cor, 1.0)
    return pd.DataFrame(shrunk_cor, index=cor.index, columns=cor.columns)


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
