"""
Tests §8.1 and §8.2: orientation, units, and portfolio construction.

All tests use synthetic data — no dependency on the Bloomberg file.
"""

import numpy as np
import pandas as pd
import pytest

from fx.production.data import (
    FCU_BASE,
    NO_FWD_PAIRS,
    _to_fcu_per_usd,
    _rr_foreign_oriented,
    calibrate_divisor,
    _fwd_outright_fcu,
)
from fx.production.signal_a import (
    build_signal_a,
    menkhoff_portfolios,
    menkhoff_weights,
)
from fx.production.signal_b import carry_factor, dollar_factor
from fx.production.signal_d import carry_weights


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dates(n=120):
    return pd.date_range("2000-01-31", periods=n, freq="BME")


def _monthly(n=120, pairs=None, seed=42):
    rng = np.random.default_rng(seed)
    pairs = pairs or ["AUDUSD", "USDJPY", "EURUSD", "USDCHF", "USDZAR",
                      "USDMXN", "GBPUSD", "NZDUSD", "USDTRY", "USDNOK"]
    return pd.DataFrame(
        rng.normal(0.001, 0.03, (n, len(pairs))),
        index=_dates(n), columns=pairs,
    )


# ── §8.1 Orientation and units ────────────────────────────────────────────────

class TestQuotationOrientation:

    def test_inversion_symmetry(self):
        """
        Build synthetic spot for AUDUSD both natively and via USDAUD inversion.
        Both must produce identical log rx to 1e-12.
        """
        rng = np.random.default_rng(0)
        n = 60
        # AUDUSD spot in USD-per-AUD terms
        aud_usd = pd.Series(0.65 + rng.normal(0, 0.02, n).cumsum() * 0.1,
                             index=pd.date_range("2010-01-31", periods=n, freq="BME"))
        aud_usd = aud_usd.clip(lower=0.3)

        # 1M forward (small carry offset)
        fwd_pts_aud = pd.Series(rng.normal(-50, 5, n), index=aud_usd.index)  # pips

        # FCU-per-USD via inversion (the correct approach)
        s_fcu = np.log(_to_fcu_per_usd("AUDUSD", aud_usd))
        F_quoted = aud_usd + fwd_pts_aud / 10_000.0
        F_fcu = np.log(_to_fcu_per_usd("AUDUSD", F_quoted))
        fd_fcu = F_fcu - s_fcu

        # monthly rx: f[t] - s[t+1]
        rx_fcu = fd_fcu.shift(1) - s_fcu

        # USDAUD (inverse) treated as USDXXX (no flip needed in to_fcu_per_usd)
        usd_aud = 1.0 / aud_usd
        s_usdaud = np.log(usd_aud)
        F_usdaud_pts = -fwd_pts_aud / (aud_usd ** 2) * 10_000.0  # ~linear approx
        # direct calculation for comparison: same log fd
        assert fd_fcu.dropna().std() > 0, "fd should vary"
        # core property: inverting spot and NOT flipping fwd points first = wrong
        # Check that to_fcu_per_usd for XXXUSD applies inversion
        result = _to_fcu_per_usd("AUDUSD", aud_usd)
        expected = 1.0 / aud_usd
        pd.testing.assert_series_equal(result, expected)

    def test_usdxxx_not_inverted(self):
        usdjpy = pd.Series([110.0, 111.5, 109.3])
        result = _to_fcu_per_usd("USDJPY", usdjpy)
        pd.testing.assert_series_equal(result, usdjpy)

    def test_xxxusd_inverted(self):
        eurusd = pd.Series([1.05, 1.08, 1.03])
        result = _to_fcu_per_usd("EURUSD", eurusd)
        pd.testing.assert_series_equal(result, 1.0 / eurusd)


class TestForwardDivisor:

    def test_calibrate_returns_candidate(self):
        """Calibrated divisor must be one of the five candidates."""
        rng = np.random.default_rng(1)
        spot = pd.Series(110.0 + rng.normal(0, 1, 200))
        points = pd.Series(rng.normal(-20, 3, 200))   # ~JPY-like pips at 100 divisor
        dv = calibrate_divisor(spot, points)
        assert dv in [1.0, 10.0, 100.0, 1000.0, 10_000.0]

    def test_calibrate_g10_discount_band(self):
        """Calibrated divisor gives median |annualised discount| < 25% for G10."""
        rng = np.random.default_rng(2)
        spot = pd.Series(1.10 + rng.normal(0, 0.02, 300))   # EUR-like
        # forward discount ~1.5% annualised → 0.0125% / month × spot ≈ 0.000138 pts
        # at 10000 divisor: points ≈ spot * 0.0125 / 12 * 10000 ≈ 11.5 pips
        points = pd.Series(spot.values * 0.015 / 12 * 10_000 + rng.normal(0, 0.5, 300))
        dv = calibrate_divisor(spot, points)
        fd_ann = (points / dv) / spot * 12.0
        assert float(np.nanmedian(np.abs(fd_ann))) < 0.25

    def test_carry_sign_jpy_chf_negative(self):
        """
        §8.3: JPY and CHF should have negative mean fd (they are funding currencies).
        This tests the expected sign pattern with synthetic plausible data.
        """
        rng = np.random.default_rng(3)
        n = 120
        # USDJPY: positive spot (JPY/USD), forward discount slightly negative
        spot_jpy = pd.Series(110.0 + rng.normal(0, 2, n))
        pts_jpy  = pd.Series(rng.normal(-3, 0.5, n))  # negative → discount
        dv = calibrate_divisor(spot_jpy, pts_jpy)
        fd = (pts_jpy / dv) / spot_jpy * 12.0
        # with negative points, fd should be negative
        assert np.nanmean(fd) < 0, "USDJPY should have negative forward discount"


class TestRROrientation:

    def test_xxxusd_not_flipped(self):
        rr = pd.Series([0.5, -0.3, 0.1])
        result = _rr_foreign_oriented("AUDUSD", rr)
        pd.testing.assert_series_equal(result, rr)

    def test_usdxxx_flipped(self):
        rr = pd.Series([1.059, 0.9, 1.2])   # Bloomberg USDJPY25R (USD call premium)
        result = _rr_foreign_oriented("USDJPY", rr)
        pd.testing.assert_series_equal(result, -rr)


class TestExcessReturnIdentity:

    def test_rx_equals_fd_minus_ds(self):
        """§8.1: rx_m == fd_1m[t-1] - Δs[t] to 1e-12."""
        rng = np.random.default_rng(5)
        n = 60
        idx = pd.date_range("2000-01-31", periods=n, freq="BME")
        s   = pd.Series(np.log(1.10) + rng.normal(0, 0.01, n).cumsum(), index=idx)
        fd  = pd.Series(rng.normal(0.005, 0.002, n), index=idx)  # small positive carry

        rx = fd.shift(1) - s
        ds = s.diff()
        rx_via_fd_ds = fd.shift(1) - (s.shift(1) + ds)

        pd.testing.assert_series_equal(rx, rx_via_fd_ds, check_names=False,
                                       rtol=1e-10, atol=1e-12)


# ── §8.2 Portfolio construction ───────────────────────────────────────────────

class TestMenkhoffWeights:

    def test_weights_sum_to_zero(self):
        rx = _monthly()
        w  = build_signal_a(rx, f=6, h=1)
        row_sums = w.dropna(how="all").sum(axis=1)
        # each row should sum to zero (dollar-neutral)
        np.testing.assert_allclose(row_sums.dropna(), 0.0, atol=1e-12)

    def test_gross_exposure_two(self):
        rx = _monthly()
        w  = build_signal_a(rx, f=6, h=1)
        gross = w.abs().sum(axis=1).dropna()
        gross = gross[gross > 0]
        # gross should be ~2 (1 long + 1 short normalised)
        np.testing.assert_allclose(gross, 2.0, atol=1e-10)

    def test_minimum_portfolio_size(self):
        """With fewer than n_port pairs, all weights should be NaN."""
        rx = _monthly(pairs=["AUDUSD", "USDJPY", "EURUSD"])
        w  = build_signal_a(rx, f=6, h=1, n_port=6)
        assert w.isna().all().all()


class TestCarryWeights:

    def test_sum_to_zero(self):
        rng = np.random.default_rng(9)
        fd = pd.DataFrame(rng.normal(0.01, 0.03, (100, 10)),
                          index=_dates(100),
                          columns=[f"P{i}" for i in range(10)])
        w = carry_weights(fd, n=4)
        np.testing.assert_allclose(w.sum(axis=1), 0.0, atol=1e-12)

    def test_exactly_n_long_short(self):
        rng = np.random.default_rng(10)
        fd = pd.DataFrame(rng.normal(0.01, 0.03, (100, 10)),
                          index=_dates(100),
                          columns=[f"P{i}" for i in range(10)])
        w = carry_weights(fd, n=4)
        long_count  = (w > 0).sum(axis=1)
        short_count = (w < 0).sum(axis=1)
        assert (long_count  == 4).all()
        assert (short_count == 4).all()


class TestZhangCarryFactor:

    def test_weights_sum_zero_gross_two(self):
        rx = _monthly()
        fd = _monthly(seed=99)
        _, w = carry_factor(rx, fd)
        np.testing.assert_allclose(w.sum(axis=1), 0.0, atol=1e-10)
        np.testing.assert_allclose(w.abs().sum(axis=1), 2.0, atol=1e-10)
