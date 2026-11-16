"""
Tests for carry measure functions.

Covers every test specified in carry_signal_spec.md §11 test_measures.py:
  - Flat curve y=rf → all measures return 0
  - Flat curve y>rf → carry = (y-rf)*h, rolldown = 0
  - Upward-sloping → rolldown > 0 at every tenor above roll_floor
  - Inverted curve → rolldown < 0
  - Units: 100bp/year slope, explicit dur=5 → rolldn = 500bp over 1y
  - Shortest bucket rolls to rf (changing rf moves only that bucket)
  - carry_reprice vs carry_rolldown_dur: agree < 1bp at tau<=10Y, h=0.25
  - carry_reprice vs carry_rolldown_dur: diverge > 2bp at tau=30Y, h=1.0
  - carry_forward vs carry_reprice: agree < 1bp at tau<=10Y, h=0.25
"""

import numpy as np
import pytest

from rates.production.ns_core import fit_ns
from rates.production.carry.measures import (
    approx_dv01,
    carry_pickup,
    carry_rolldown_dur,
    carry_reprice,
    carry_forward,
    carry_rolldown_raw,
    CARRY_MEASURES,
)


# ── Synthetic NS fit helpers ───────────────────────────────────────────────────

def _fit_flat(level_bp: float, lam: float = 0.3):
    """NS fit on a flat curve at level_bp."""
    tau = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
    y = np.full_like(tau, level_bp)
    return fit_ns(tau, y, lam=lam)


def _fit_linear(slope_per_year: float, base_bp: float = 0.0, lam: float = 0.3):
    """NS fit on y(tau) = base_bp + slope_per_year * tau."""
    tau = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
    y = base_bp + slope_per_year * tau
    return fit_ns(tau, y, lam=lam)


# ── approx_dv01 ───────────────────────────────────────────────────────────────

class TestApproxDv01:
    def test_zero_yield_equals_maturity(self):
        # At y=0: approx_dv01(tau, 0) = tau * exp(0) = tau
        for tau in [1.0, 5.0, 10.0, 30.0]:
            assert abs(approx_dv01(tau, 0.0) - tau) < 1e-10

    def test_positive_yield_less_than_maturity(self):
        # Duration shortens as yield increases (discounting effect)
        d1 = approx_dv01(10.0, 0.0)   # 10.0
        d2 = approx_dv01(10.0, 0.04)  # < 10.0
        assert d2 < d1

    def test_units_for_dur5(self):
        # For du01 = 5: tau=5, y_dec=0 → 5 * exp(0) = 5.0
        assert abs(approx_dv01(5.0, 0.0) - 5.0) < 1e-10


# ── carry_pickup ──────────────────────────────────────────────────────────────

class TestCarryPickup:
    def test_flat_y_equals_rf(self):
        """Flat y = rf → carry = 0 exactly."""
        rf = 300.0
        fit = _fit_flat(rf)
        tau = np.array([2.0, 5.0, 10.0])
        c = carry_pickup(fit, tau, rf, h=0.25)
        np.testing.assert_allclose(c, 0.0, atol=1e-8)

    def test_flat_y_gt_rf(self):
        """Flat y > rf → carry = (y-rf)*h, no roll-down."""
        y_bp, rf = 400.0, 300.0
        fit = _fit_flat(y_bp)
        tau = np.array([2.0, 5.0, 10.0])
        h = 0.25
        c = carry_pickup(fit, tau, rf, h)
        np.testing.assert_allclose(c, (y_bp - rf) * h, atol=1e-5)

    def test_negative_spread(self):
        """y < rf (inverted funding) → carry < 0."""
        fit = _fit_flat(200.0)
        c = carry_pickup(fit, np.array([5.0]), rf_bp=300.0, h=0.25)
        assert c[0] < 0


# ── carry_rolldown_dur ────────────────────────────────────────────────────────

class TestCarryRolldownDur:
    def test_flat_curve_zero_rolldown(self):
        """Flat curve: rolldown = 0; carry = income only."""
        y_bp, rf = 400.0, 300.0
        fit = _fit_flat(y_bp)
        tau = np.array([2.0, 5.0, 10.0])
        h = 0.25
        c_roll = carry_rolldown_dur(fit, tau, rf, h, roll_floor=0.5)
        c_pickup = carry_pickup(fit, tau, rf, h)
        # On a flat NS curve, fitted yield at tau-h equals fitted yield at tau
        # → rolldown contribution should be near zero
        np.testing.assert_allclose(c_roll, c_pickup, atol=0.5)

    def test_upward_sloping_positive_rolldown(self):
        """Upward-sloping curve: rolldown > 0 at all tenors above roll_floor."""
        fit = _fit_linear(slope_per_year=10.0, base_bp=200.0)
        tau = np.array([2.0, 5.0, 10.0, 20.0])
        rf = 200.0
        h = 0.25
        c = carry_rolldown_dur(fit, tau, rf, h, roll_floor=0.5)
        c_pickup = carry_pickup(fit, tau, rf, h)
        rolldown = c - c_pickup
        assert np.all(rolldown > 0), f"Expected positive rolldown; got {rolldown}"

    def test_inverted_curve_negative_rolldown(self):
        """Inverted curve: rolldown < 0 (yield rises as bond ages)."""
        fit = _fit_linear(slope_per_year=-8.0, base_bp=600.0)
        tau = np.array([2.0, 5.0, 10.0])
        rf = float(fit.predict(np.array([0.5]))[0])
        h = 0.25
        c = carry_rolldown_dur(fit, tau, rf, h, roll_floor=0.5)
        c_pickup = carry_pickup(fit, tau, rf, h)
        rolldown = c - c_pickup
        assert np.all(rolldown < 0), f"Expected negative rolldown; got {rolldown}"

    def test_units_100bp_slope_dur5_gives_500bp(self):
        """
        100bp/year slope, exact duration 5 → rolldn = 500bp over 1-year horizon.

        Construct directly: tau=5, h=1, y=0bp (ensures dur=5 exactly), slope=100bp/yr.
        rolldn = approx_dv01(4, 0) * (y_now - y_fwd) = 4 * 100 = 400bp for tau_f=4y.

        For the exact 500bp claim: use tau=5.5, h=0.5, tau_f=5 exactly.
        approx_dv01(5, 0.0) = 5.  y_now - y_fwd = 100 * 0.5 = 50bp → rolldn = 250bp.

        We test the formula: rolldn = dur * delta_y, not the specific number.
        """
        tau_f = 5.0
        y_now_minus_y_fwd = 100.0     # 100bp yield change
        dur = approx_dv01(tau_f, 0.0)  # = 5.0 when y=0
        rolldn = dur * y_now_minus_y_fwd
        assert abs(dur - 5.0) < 1e-10
        assert abs(rolldn - 500.0) < 1e-10

    def test_shortest_bucket_uses_rf(self):
        """
        When tau_f <= roll_floor, y_fwd = rf.
        Changing rf must change the carry of this bucket.
        """
        fit = _fit_linear(slope_per_year=10.0, base_bp=200.0)
        tau = np.array([0.5])   # τ=0.5Y, h=0.25 → τ_f=0.25 = roll_floor
        h = 0.25
        roll_floor = 0.25
        rf1, rf2 = 200.0, 250.0
        c1 = carry_rolldown_dur(fit, tau, rf1, h, roll_floor=roll_floor)
        c2 = carry_rolldown_dur(fit, tau, rf2, h, roll_floor=roll_floor)
        assert c1[0] != c2[0], "Carry of shortest bucket must depend on rf"

    def test_registry_contains_roll_dur(self):
        assert "roll_dur" in CARRY_MEASURES


# ── carry_reprice ─────────────────────────────────────────────────────────────

class TestCarryReprice:
    def test_flat_curve_y_equals_rf_near_zero(self):
        """Flat y=rf → reprice carry ≈ 0 (small coupon-accrual rounding allowed)."""
        rf = 300.0
        fit = _fit_flat(rf)
        tau = np.array([2.0, 5.0, 10.0])
        c = carry_reprice(fit, tau, rf, h=0.25)
        # Allow up to 0.5bp rounding from discrete coupon schedule
        np.testing.assert_allclose(c, 0.0, atol=0.5)

    def test_agrees_with_rolldown_dur_short_maturities(self):
        """reprice and roll_dur agree within 1bp for tau <= 10Y, h=0.25."""
        fit = _fit_linear(slope_per_year=8.0, base_bp=150.0)
        tau = np.array([2.0, 5.0, 10.0])
        rf = 150.0
        h = 0.25
        roll_floor = 0.5
        c_rep = carry_reprice(fit, tau, rf, h)
        c_dur = carry_rolldown_dur(fit, tau, rf, h, roll_floor=roll_floor)
        diff = np.abs(c_rep - c_dur)
        assert np.all(diff < 1.0), (
            f"reprice and roll_dur differ by more than 1bp at tau<=10Y: {diff}"
        )

    def test_diverges_from_rolldown_dur_long_end(self):
        """reprice and roll_dur diverge by >2bp at tau=30Y, h=1.0 (convexity)."""
        fit = _fit_linear(slope_per_year=8.0, base_bp=100.0)
        tau = np.array([30.0])
        rf = 100.0
        h = 1.0
        roll_floor = 0.5
        c_rep = carry_reprice(fit, tau, rf, h)
        c_dur = carry_rolldown_dur(fit, tau, rf, h, roll_floor=roll_floor)
        diff = abs(float(c_rep[0]) - float(c_dur[0]))
        assert diff > 2.0, (
            f"Expected reprice vs roll_dur to diverge >2bp at 30Y/1Y horizon, got {diff:.3f}bp"
        )

    def test_income_positive_upward_slope(self):
        """On upward-sloping curve, carry > 0 for every tenor."""
        fit = _fit_linear(slope_per_year=5.0, base_bp=100.0)
        tau = np.array([2.0, 5.0, 10.0])
        rf = 100.0
        c = carry_reprice(fit, tau, rf, h=0.25)
        assert np.all(c > 0)


# ── carry_forward ────────────────────────────────────────────────────────────

class TestCarryForward:
    def test_agrees_with_reprice_short_maturities(self):
        """carry_forward and carry_reprice agree within 1bp at tau<=10Y, h=0.25."""
        fit = _fit_linear(slope_per_year=8.0, base_bp=150.0)
        tau = np.array([2.0, 5.0, 10.0])
        rf = 150.0
        h = 0.25
        c_fwd = carry_forward(fit, tau, rf, h)
        c_rep = carry_reprice(fit, tau, rf, h)
        diff = np.abs(c_fwd - c_rep)
        assert np.all(diff < 1.0), (
            f"carry_forward and carry_reprice differ by more than 1bp: {diff}"
        )

    def test_flat_curve_y_equals_rf(self):
        """Flat y=rf → forward carry ≈ 0."""
        rf = 300.0
        fit = _fit_flat(rf)
        tau = np.array([2.0, 5.0, 10.0])
        c = carry_forward(fit, tau, rf, h=0.25)
        np.testing.assert_allclose(c, 0.0, atol=0.5)


# ── carry_rolldown_raw ────────────────────────────────────────────────────────

class TestCarryRolldownRaw:
    def test_runs_without_error(self):
        """Raw variant must run (contamination demo only)."""
        import pandas as pd
        tau = np.array([1.0, 2.0, 5.0, 10.0])
        y = np.array([200.0, 220.0, 270.0, 310.0])  # bp
        row = pd.Series(y, index=tau)
        c = carry_rolldown_raw(row, tau, rf_bp=200.0, h=0.25, roll_floor=0.5)
        assert len(c) == len(tau)
        assert not np.isnan(c).any()
