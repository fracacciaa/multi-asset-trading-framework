"""
Tests for src/nsrv/ns_core.py.
All must pass before any Sharpe is computed.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.nsrv.ns_core import fit_ns, ns_loadings, NFACTORS


RNG = np.random.default_rng(42)
TAU = np.array([1., 2., 3., 5., 7., 10., 15., 20., 30.])  # typical OIS tenors
LAM = 0.4   # 1/years (≈ 2.5Y peak)


# ── Loading shape tests ────────────────────────────────────────────────────────

def test_loadings_at_zero():
    """As tau -> 0: slope loading -> 1, curvature loading -> 0."""
    eps = 1e-9
    X = ns_loadings(np.array([eps]), LAM)
    assert abs(X[0, 1] - 1.0) < 1e-6, f"slope at 0 should be 1, got {X[0,1]}"
    assert abs(X[0, 2] - 0.0) < 1e-6, f"curvature at 0 should be 0, got {X[0,2]}"


def test_curvature_peaks():
    """Curvature loading peaks at tau* = 1.7933 / lam, to within 1%."""
    tau_dense = np.linspace(0.1, 20, 2000)
    X = ns_loadings(tau_dense, LAM)
    peak_tau = tau_dense[np.argmax(X[:, 2])]
    expected = 1.7933 / LAM
    assert abs(peak_tau - expected) / expected < 0.01, (
        f"curvature peak at {peak_tau:.3f}, expected {expected:.3f}"
    )


# ── Fit correctness ────────────────────────────────────────────────────────────

def test_exact_recovery():
    """OLS on noise-free data recovers beta exactly; residuals are zero."""
    beta_true = np.array([3.5, -1.2, 0.8])
    X = ns_loadings(TAU, LAM)
    y = X @ beta_true

    fit = fit_ns(TAU, y, LAM)

    np.testing.assert_allclose(fit.beta, beta_true, atol=1e-10)
    np.testing.assert_allclose(fit.resid, np.zeros(len(TAU)), atol=1e-10)


def test_leverage_trace():
    """Sum of hat diagonal equals number of factors k."""
    y = RNG.normal(3.0, 0.5, len(TAU))
    fit = fit_ns(TAU, y, LAM)
    k = NFACTORS["ns"]
    assert abs(fit.leverage.sum() - k) < 1e-8, (
        f"h.sum()={fit.leverage.sum():.6f}, expected k={k}"
    )


def test_leverage_trace_bc():
    """Same for bc basis (k=4)."""
    y = RNG.normal(3.0, 0.5, len(TAU))
    fit = fit_ns(TAU, y, LAM, basis="bc")
    k = NFACTORS["bc"]
    assert abs(fit.leverage.sum() - k) < 1e-8


# ── LOO brute force ────────────────────────────────────────────────────────────

def test_loo_vs_brute_force():
    """
    Leave-one-out equivalence: for each tenor i, refit on the remaining n-1
    points, predict at i, and check that
        yhat_full_i + resid_loo_i  ≈  yhat_loo_i + resid_loo_i
    i.e.  yhat_loo_i  ≈  yhat_full_i + resid_loo_i  (both equal y_i corrected)

    Equivalently: the LOO predicted value at i = y_i - resid_loo_i,
    and the brute-force leave-i-out prediction at i should equal yhat_full_i + resid_loo_i.
    Both expressions equal the LOO fitted value; they should match to 1e-8.
    """
    y = RNG.normal(0, 1, len(TAU)) + ns_loadings(TAU, LAM) @ np.array([3., -1., 0.5])
    fit = fit_ns(TAU, y, LAM)

    for i in range(len(TAU)):
        mask = np.ones(len(TAU), bool)
        mask[i] = False
        fit_loo = fit_ns(TAU[mask], y[mask], LAM)

        # Prediction at the held-out tenor from the LOO fit
        yhat_loo_i = float(fit_loo.predict(TAU[i:i+1]))

        # The formula says: yhat_loo_i ≈ fit.yhat[i] + fit.resid_loo[i] - fit.resid[i]
        # More precisely: resid_loo_i = r_i / (1 - h_i)
        # and the LOO prediction is  y_i - resid_loo_i
        loo_pred_formula = float(y[i] - fit.resid_loo[i])

        assert abs(yhat_loo_i - loo_pred_formula) < 1e-8, (
            f"Tenor {i} (tau={TAU[i]}): brute-force LOO pred={yhat_loo_i:.8f}, "
            f"formula={loo_pred_formula:.8f}, diff={abs(yhat_loo_i-loo_pred_formula):.2e}"
        )


# ── Prior behaviour ────────────────────────────────────────────────────────────

def test_prior_zero_equals_ols():
    """prior_w = zeros gives the same result as no prior (OLS)."""
    y = RNG.normal(3.0, 0.5, len(TAU))
    fit_ols   = fit_ns(TAU, y, LAM)
    fit_prior = fit_ns(TAU, y, LAM, prior_w=np.zeros(3))
    np.testing.assert_allclose(fit_ols.beta, fit_prior.beta, atol=1e-12)


def test_prior_large_drives_to_mean():
    """Very large prior_w drives beta toward prior_mean regardless of data."""
    y = RNG.normal(3.0, 0.5, len(TAU))
    pm = np.array([5.0, -2.0, 1.5])
    fit = fit_ns(TAU, y, LAM, prior_mean=pm, prior_w=np.full(3, 1e9))
    np.testing.assert_allclose(fit.beta, pm, atol=1e-4)


# ── Sign sanity ────────────────────────────────────────────────────────────────

def test_residual_sign():
    """A tenor with yield above the fitted curve has positive residual."""
    beta = np.array([3.0, -1.0, 0.5])
    X = ns_loadings(TAU, LAM)
    y = X @ beta
    # Artificially push up the 5Y tenor
    y[3] += 20.0  # 20bp above the curve

    fit = fit_ns(TAU, y, LAM)
    assert fit.resid[3] > 0, "cheap tenor must have positive residual"
    assert fit.resid_loo[3] > fit.resid[3], "LOO residual must exceed in-sample at dislocated tenor"
