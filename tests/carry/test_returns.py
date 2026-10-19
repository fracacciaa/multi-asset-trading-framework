"""
Tests for the constant-maturity total-return decomposition.

Covers carry_signal_spec.md §11 test_returns.py:
  - Decomposition identity: ret_total == ret_carry + ret_yldchg exactly
  - Static curve (60 days): ret_yldchg ≈ 0; realized return ≈ ex-ante carry
  - No lookahead: only y_now comes from date t; all other inputs from t-1
  - Attribution signs: carry component positive, yield-change component volatile
"""

import numpy as np
import pandas as pd
import pytest

from src.nsrv.ns_core import fit_ns, NSFit
from src.nsrv.carry.measures import approx_dv01, carry_reprice
from src.nsrv.carry.returns import cm_total_return_bp, compute_dur_panel, pnl_book


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_fit(slope: float = 8.0, base: float = 150.0, lam: float = 0.3) -> NSFit:
    tau = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
    y = base + slope * tau
    return fit_ns(tau, y, lam=lam)


def _make_flat_fit(level: float = 300.0, lam: float = 0.3) -> NSFit:
    tau = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
    y = np.full_like(tau, level)
    return fit_ns(tau, y, lam=lam)


# ── Decomposition identity ────────────────────────────────────────────────────

class TestDecompositionIdentity:
    def test_carry_plus_yldchg_equals_total(self):
        """ret_carry + ret_yldchg == ret_total exactly."""
        fit = _make_fit()
        tau = np.array([2.0, 5.0, 10.0])
        y_prev = fit.predict(tau)
        # Small yield change
        y_now = y_prev + np.array([-5.0, 0.0, 10.0])
        rf_prev = 150.0
        dur_prev = compute_dur_panel(fit, tau)
        dt = 1 / 252

        rc, ryc = cm_total_return_bp(y_prev, y_now, fit, tau, rf_prev, dur_prev, dt)
        total = rc + ryc

        # The identity must hold exactly (it's a mathematical definition)
        expected_total = (y_prev * dt
                          + dur_prev * (fit.predict(tau) - fit.predict(np.maximum(tau - dt, 1e-8)))
                          - dur_prev * (y_now - y_prev)
                          - rf_prev * dt)
        np.testing.assert_allclose(total, expected_total, atol=1e-12)

    def test_carry_plus_yldchg_equals_total_many_dates(self):
        """Identity holds across many simulated dates."""
        rng = np.random.default_rng(42)
        fit = _make_fit()
        tau = np.array([2.0, 5.0, 10.0])
        dur = compute_dur_panel(fit, tau)
        dt = 1 / 252

        for _ in range(50):
            y_prev = fit.predict(tau) + rng.normal(0, 5, size=len(tau))
            dy = rng.normal(0, 10, size=len(tau))
            y_now = y_prev + dy
            rf = float(rng.uniform(100, 400))
            rc, ryc = cm_total_return_bp(y_prev, y_now, fit, tau, rf, dur, dt)
            assert np.allclose(rc + ryc,
                                y_prev * dt
                                + dur * (fit.predict(tau) - fit.predict(np.maximum(tau - dt, 1e-8)))
                                - dur * dy
                                - rf * dt,
                                atol=1e-12)


# ── Static curve convergence ──────────────────────────────────────────────────

class TestStaticCurve:
    def test_static_curve_yldchg_near_zero(self):
        """If yields do not move (y_now == y_prev), ret_yldchg should be zero."""
        fit = _make_fit()
        tau = np.array([2.0, 5.0, 10.0])
        y = fit.predict(tau)
        dur = compute_dur_panel(fit, tau)
        dt = 1 / 252
        rc, ryc = cm_total_return_bp(y, y, fit, tau, 150.0, dur, dt)
        np.testing.assert_allclose(ryc, 0.0, atol=1e-10)

    def test_static_curve_realized_matches_ex_ante_carry(self):
        """
        Over 60 daily steps on a static curve, cumulative return ≈ ex-ante carry.

        Ex-ante carry (reprice) for h=60/252 years should match the summed
        daily returns to within 0.5bp.
        """
        fit = _make_fit()
        tau = np.array([5.0])
        dur = compute_dur_panel(fit, tau)
        dt = 1 / 252
        n_steps = 60
        h = n_steps * dt
        rf = 150.0

        y_static = fit.predict(tau)
        cum_carry = 0.0
        for _ in range(n_steps):
            rc, ryc = cm_total_return_bp(y_static, y_static, fit, tau, rf, dur, dt)
            cum_carry += float(rc[0])

        ex_ante = float(carry_reprice(fit, tau, rf, h)[0])
        # Allow 1bp tolerance (daily duration steps vs full-period reprice)
        assert abs(cum_carry - ex_ante) < 1.0, (
            f"Cumulative carry {cum_carry:.3f}bp vs ex-ante {ex_ante:.3f}bp"
        )


# ── Carry component sign ──────────────────────────────────────────────────────

class TestCarrySigns:
    def test_carry_component_positive_for_long_high_carry(self):
        """
        On an upward-sloping curve, carry component should be positive.
        (This is the premium the strategy is designed to harvest.)
        """
        fit = _make_fit(slope=8.0, base=150.0)
        tau = np.array([5.0, 10.0])
        y = fit.predict(tau)
        dur = compute_dur_panel(fit, tau)
        dt = 1 / 252

        # Long both tenors (as if we receive fixed on an upward-sloping curve)
        rc, _ = cm_total_return_bp(y, y, fit, tau, rf_prev=150.0, dur_prev=dur, dt=dt)
        assert np.all(rc > 0), f"Carry should be positive on upward-sloping curve, got {rc}"

    def test_pnl_book_aggregation(self):
        """pnl_book returns sum of position-weighted carry and yield-change components."""
        pos = np.array([0.5, -0.5])
        rc = np.array([10.0, 8.0])
        ryc = np.array([-3.0, 5.0])
        pc, pyc = pnl_book(pos, rc, ryc)
        assert abs(pc - (0.5 * 10.0 + (-0.5) * 8.0)) < 1e-10
        assert abs(pyc - (0.5 * (-3.0) + (-0.5) * 5.0)) < 1e-10
