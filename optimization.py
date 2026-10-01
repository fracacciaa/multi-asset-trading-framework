"""
Portfolio optimization.
Combines cross-asset signals into target portfolio weights.
Optimization approach (mean-variance, risk parity, Black-Litterman, etc.) to be defined.
"""

import pandas as pd
import numpy as np


def build_views(signals: dict[str, pd.Series]) -> tuple[pd.Series, pd.DataFrame]:
    """
    Aggregate per-asset-class signals into a single views vector and confidence matrix.

    signals: dict mapping asset_class -> z-scored signal Series (index = instrument)
    Returns:
        mu:    expected return views vector (instruments,)
        omega: view uncertainty / confidence diagonal matrix (instruments x instruments)
    """
    raise NotImplementedError

def estimate_covariance(returns: pd.DataFrame, method: str = "sample") -> pd.DataFrame:
    """
    Estimate the covariance matrix of returns.
    method: 'sample' | 'ledoit_wolf' | 'ewm'
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

    mu:          expected returns
    cov:         covariance matrix
    constraints: output of constraints.build_constraints()
    objective:   'mean_variance' | 'risk_parity' | 'max_sharpe'

    Returns a Series of optimal weights indexed by instrument.
    """
    raise NotImplementedError
