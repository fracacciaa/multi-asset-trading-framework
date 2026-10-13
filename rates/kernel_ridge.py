"""
Kernel Ridge Regression for yield curve fitting.

Minimises the penalised least squares problem:
    Σᵢ (y(τᵢ) − ŷ(τᵢ))² + μ · ‖ŷ‖²_{H²}

where the H² Sobolev norm penalises the integrated squared second derivative.
The solution is a kernel ridge regression with the cubic spline kernel:
    k(s, t) = 1 + min(s,t)·max(s,t) + ½·min(s,t)²·(max(s,t) − min(s,t)/3)

which is the reproducing kernel of H²([0,∞)).  The fitted curve is spanned by
N kernel functions k(τᵢ, ·), one per observed tenor — no fixed functional form.

Larger μ → smoother fit;  μ → 0 → exact interpolation.

Reference: Filipović, Pelger & Ye (2025) "Stripping the Discount Curve:
A Robust Machine Learning Approach." Management Science.
"""

import numpy as np
import pandas as pd
from typing import Optional


def _kernel_matrix(s: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Gram matrix k(s_i, t_j) for the cubic spline (H²-Sobolev) kernel."""
    s = s[:, None]
    t = t[None, :]
    mn = np.minimum(s, t)
    mx = np.maximum(s, t)
    return 1.0 + mn * mx + 0.5 * mn**2 * (mx - mn / 3.0)


def extract_residuals_kr(
    zero_curve: pd.DataFrame,
    mu: float,
    eval_tenors: Optional[np.ndarray] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit KRR on every row of zero_curve.  Returns (residuals, fitted).

    zero_curve  : date × tenor DataFrame (full interpolated grid from
                  build_zero_curve — assumed NaN-free and constant columns)
    mu          : ridge penalty
    eval_tenors : subset of zero_curve.columns at which to report residuals;
                  defaults to all columns

    Returns
    -------
    resid  : date × eval_tenors  (actual − fitted, positive = cheap tenor)
    fitted : date × eval_tenors  (smooth KRR curve)
    """
    obs_tenors = zero_curve.columns.values.astype(float)
    ev = eval_tenors if eval_tenors is not None else obs_tenors
    n = len(obs_tenors)

    K_obs = _kernel_matrix(obs_tenors, obs_tenors)           # (N, N)
    K_ev  = _kernel_matrix(obs_tenors, ev)                   # (N, M)
    # W[m, i] = [K_ev.T (K_obs + μI)^{-1}]_{m,i}
    # fitted(date) = y_obs(date) @ W.T
    W = np.linalg.solve(K_obs + mu * np.eye(n), K_ev).T      # (M, N)

    y_obs  = zero_curve.values.astype(float)                  # (T, N)
    y_fit  = y_obs @ W.T                                      # (T, M)

    ev_idx = np.array([
        np.where(np.isclose(obs_tenors, t))[0][0] for t in ev
    ])
    y_eval = y_obs[:, ev_idx]                                  # (T, M)

    resid = y_eval - y_fit
    return (
        pd.DataFrame(resid, index=zero_curve.index, columns=ev),
        pd.DataFrame(y_fit, index=zero_curve.index, columns=ev),
    )
