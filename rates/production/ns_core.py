"""
Nelson-Siegel curve fit with leave-one-out (LOO) residuals.

Lambda convention: x = tau * lam  (lam in 1/years).
A lam=0.3 means the slope loading peaks at ~3.3 years.

The LOO residual corrects for self-absorption: when a tenor is dislocated,
the in-sample fit is pulled toward it, understating the residual.
With n=8-15 tenors the hat-diagonal can exceed 0.3 at illiquid points.

Reference: spec rule 3 in ns_residual_signal_spec.md
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


# ── Loading functions ──────────────────────────────────────────────────────────

def ns_loadings(tau: np.ndarray, lam: float) -> np.ndarray:
    """(n, 3) design matrix [level, slope, curvature]. lam in 1/years."""
    x = np.maximum(np.asarray(tau, float) * lam, 1e-8)
    d = (1.0 - np.exp(-x)) / x
    return np.column_stack([np.ones_like(x), d, d - np.exp(-x)])


def bc_loadings(tau: np.ndarray, lam: float) -> np.ndarray:
    """Bjork-Christensen: NS + extra exp(-2*lam*tau) factor. (n, 4)."""
    X = ns_loadings(tau, lam)
    return np.column_stack([X, np.exp(-2.0 * lam * np.asarray(tau, float))])


def svensson_adj_loadings(tau: np.ndarray, lam: float) -> np.ndarray:
    """Adjusted Svensson: lam2 fixed at 2*lam, no second free decay. (n, 4)."""
    X = ns_loadings(tau, lam)
    x2 = np.maximum(np.asarray(tau, float) * 2.0 * lam, 1e-8)
    d2 = (1.0 - np.exp(-x2)) / x2
    return np.column_stack([X, d2 - np.exp(-x2)])


def ns2_loadings(tau: np.ndarray, lam: float) -> np.ndarray:
    """Two-factor control: level + slope only. (n, 2)."""
    return ns_loadings(tau, lam)[:, :2]


LOADINGS: dict = {
    "ns2":           ns2_loadings,
    "ns":            ns_loadings,
    "bc":            bc_loadings,
    "svensson_adj":  svensson_adj_loadings,
}

NFACTORS: dict[str, int] = {
    "ns2": 2, "ns": 3, "bc": 4, "svensson_adj": 4,
}


# ── Fit result ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NSFit:
    beta:      np.ndarray   # (k,)
    lam:       float
    basis:     str
    tau:       np.ndarray   # (n,) observed tenors, years
    y:         np.ndarray   # (n,) observed yields, bp
    yhat:      np.ndarray   # (n,) fitted yields, bp
    resid:     np.ndarray   # (n,) in-sample residual (for diagnostics only)
    resid_loo: np.ndarray   # (n,) LOO-corrected residual — USE THIS as signal
    leverage:  np.ndarray   # (n,) hat-matrix diagonal
    sse:       float
    n_obs:     int

    def predict(self, tau_new: np.ndarray) -> np.ndarray:
        return LOADINGS[self.basis](np.asarray(tau_new, float), self.lam) @ self.beta


# ── Main fit function ──────────────────────────────────────────────────────────

def fit_ns(
    tau:        np.ndarray,
    y:          np.ndarray,
    lam:        float,
    *,
    basis:      str                  = "ns",
    prior_mean: Optional[np.ndarray] = None,   # (k,), None = zeros
    prior_w:    Optional[np.ndarray] = None,   # (k,) equivalent observations, None = OLS
    weights:    Optional[np.ndarray] = None,   # (n,) observation weights, None = equal
) -> NSFit:
    """
    Fixed-lam NS ridge fit with LOO residuals.

    Solves:
        G    = inv(Xw' X + A)
        beta = G (Xw' y + A m)

    where A = diag(prior_w), m = prior_mean, Xw = X * sqrt(weights).

    Hat diagonal (for LOO correction):
        h_i = X_i' G X_i   (weighted version when weights are used)

    LOO residual:
        resid_loo_i = resid_i / (1 - h_i)

    prior_w units: equivalent observations. Use shrink_to_prior_w() to
    convert a legacy shrink fraction to this scale.
    """
    tau = np.asarray(tau, float)
    y   = np.asarray(y,   float)
    n   = len(tau)
    k   = NFACTORS[basis]

    X = LOADINGS[basis](tau, lam)                       # (n, k)
    W = np.ones(n) if weights is None else np.asarray(weights, float)
    m = np.zeros(k) if prior_mean is None else np.asarray(prior_mean, float)[:k]
    A = np.diag(np.zeros(k) if prior_w is None else np.asarray(prior_w, float)[:k])

    Xw  = X * W[:, None]                                # (n, k) weighted rows
    G   = np.linalg.inv(Xw.T @ X + A)                  # (k, k)
    beta = G @ (Xw.T @ y + A @ m)                      # (k,)

    yhat = X @ beta                                     # (n,)
    r    = y - yhat                                     # (n,) in-sample

    # Weighted hat diagonal: h_i = (Xw_i)' G X_i
    h = np.einsum("ij,jk,ik->i", Xw, G, X)             # (n,)
    resid_loo = r / np.clip(1.0 - h, 1e-6, None)       # (n,) LOO-corrected

    return NSFit(
        beta=beta, lam=lam, basis=basis,
        tau=tau, y=y, yhat=yhat,
        resid=r, resid_loo=resid_loo,
        leverage=h,
        sse=float(r @ r),
        n_obs=n,
    )


# ── Utility ────────────────────────────────────────────────────────────────────

def shrink_to_prior_w(shrink: float, n_obs: int, k: int) -> np.ndarray:
    """Convert a legacy isotropic shrink ∈ (0,1) to per-factor prior weights."""
    return np.full(k, shrink / max(1e-6, 1.0 - shrink) * n_obs)
