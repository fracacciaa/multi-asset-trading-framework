"""
Tests §8.4: no-lookahead property.

Truncate the panel at date T, recompute every signal, assert identity
with the full-sample run at all dates up to T-1.

Checks the four likely failure points from the spec:
  1. dupuy_signal without the extra .shift(1)
  2. dupuy_signal normalised by full-sample rather than expanding mean
  3. e3_global_state z-scored on full sample
  4. Zhang monthly betas if estimated on a window past month end
"""

import numpy as np
import pandas as pd
import pytest

from fx.production.signal_a import build_signal_a
from fx.production.signal_b import monthly_betas, build_signal_b
from fx.production.signal_c import tsmom_vol_scaled, tsmom_equal_weight, ex_ante_vol_monthly
from fx.production.signal_d import dupuy_signal, expected_profit_mid, market_variance
from fx.production.signal_e import e3_global_state


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dates_monthly(n):
    return pd.date_range("1999-01-31", periods=n, freq="BME")


def _dates_daily(n):
    return pd.date_range("1999-01-04", periods=n, freq="B")


def _rx_monthly(n=120, m=10, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0.001, 0.02, (n, m)),
                        index=_dates_monthly(n),
                        columns=[f"P{i}" for i in range(m)])


def _rx_daily(n=2400, m=10, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0, 0.005, (n, m)),
                        index=_dates_daily(n),
                        columns=[f"P{i}" for i in range(m)])


def _fd_monthly(n=120, m=10, seed=2):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0.01, 0.02, (n, m)),
                        index=_dates_monthly(n),
                        columns=[f"P{i}" for i in range(m)])


def _check_no_lookahead(full: pd.DataFrame, trunc: pd.DataFrame,
                        name: str, tol: float = 1e-10):
    """Assert full[:T-1] == trunc[:T-1] element-wise."""
    T_cut  = trunc.index[-1]
    common = full.index[full.index <= T_cut][:-1]  # up to T-1
    if len(common) == 0:
        return
    f = full.reindex(common)
    t = trunc.reindex(common)
    diff = (f - t).abs().max().max()
    assert diff < tol, f"{name}: max diff {diff:.2e} exceeds {tol:.2e}"


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def data():
    rx_m = _rx_monthly(120)
    rx_d = _rx_daily(2400)
    fd_m = _fd_monthly(120)
    return {"rx_m": rx_m, "rx_d": rx_d, "fd_m": fd_m}


def _truncation_dates(index, fracs=(0.6, 0.75, 0.9)):
    n = len(index)
    return [index[int(n * f) - 1] for f in fracs]


class TestNoLookaheadSignalA:

    def test_menkhoff_no_lookahead(self, data):
        rx_m = data["rx_m"]
        full = build_signal_a(rx_m, f=6)
        for T in _truncation_dates(rx_m.index):
            trunc = build_signal_a(rx_m.loc[:T], f=6)
            _check_no_lookahead(full, trunc, f"Signal A T={T}")


class TestNoLookaheadSignalC:

    def test_tsmom_ew_no_lookahead(self, data):
        rx_m = data["rx_m"]
        full  = tsmom_equal_weight(rx_m, k=12)
        for T in _truncation_dates(rx_m.index):
            trunc = tsmom_equal_weight(rx_m.loc[:T], k=12)
            _check_no_lookahead(full, trunc, f"TSMOM EW T={T}")

    def test_tsmom_vol_no_lookahead(self, data):
        rx_m = data["rx_m"]
        rx_d = data["rx_d"]
        sigma_m = ex_ante_vol_monthly(rx_d)
        # align sigma to rx_m dates
        sigma_me = sigma_m.reindex(rx_m.index, method="ffill")
        full  = tsmom_vol_scaled(rx_m, sigma_me, k=12)
        for T in _truncation_dates(rx_m.index):
            sigma_t  = ex_ante_vol_monthly(rx_d.loc[:T]).reindex(rx_m.loc[:T].index, method="ffill")
            trunc    = tsmom_vol_scaled(rx_m.loc[:T], sigma_t, k=12)
            _check_no_lookahead(full, trunc, f"TSMOM vol T={T}")


class TestNoLookaheadDupuy:

    def test_dupuy_signal_no_lookahead(self, data):
        rx_d = data["rx_d"]
        fd_m = data["fd_m"]
        MV_full = market_variance(rx_d)
        E_full  = expected_profit_mid(fd_m)
        sig_full = dupuy_signal(E_full, MV_full, mode="E_over_V")

        for T in _truncation_dates(fd_m.index):
            MV_t   = market_variance(rx_d.loc[:T])
            E_t    = expected_profit_mid(fd_m.loc[:T])
            sig_t  = dupuy_signal(E_t, MV_t, mode="E_over_V")
            _check_no_lookahead(
                sig_full.to_frame("s"), sig_t.to_frame("s"),
                f"Dupuy signal T={T}",
            )

    def test_dupuy_extra_lag_matters(self, data):
        """
        If the extra .shift(1) is missing, the signal at t uses data from t,
        overlapping the hold period. Verify it is present by checking that
        the signal at date t does not use the E/MV at date t (it uses t-1).
        """
        rx_d = data["rx_d"]
        fd_m = data["fd_m"]
        MV   = market_variance(rx_d)
        E    = expected_profit_mid(fd_m)
        sig  = dupuy_signal(E, MV, mode="E_over_V")
        # The signal at t should equal normalised E/MV at t-1, not t
        # Proxy test: the signal must have 1 more leading NaN than E/MV
        first_sig_valid = sig.first_valid_index()
        raw = (E / MV)
        raw_norm = raw / raw.expanding(min_periods=36).mean()
        # shifted version
        raw_shifted = raw_norm.shift(1)
        assert raw_shifted.first_valid_index() >= raw.first_valid_index()
        # signal should equal raw_shifted (up to floating point)
        common = sig.index.intersection(raw_shifted.index)
        diff = (sig.reindex(common) - raw_shifted.reindex(common)).abs()
        assert diff.max() < 1e-10, "dupuy_signal does not apply the extra shift"


class TestNoLookaheadSignalE:

    def test_e3_no_lookahead(self, data):
        rng  = np.random.default_rng(7)
        n    = 120
        idx  = _dates_monthly(n)
        atm  = pd.DataFrame(rng.normal(8, 2, (n, 5)), index=idx,
                             columns=[f"P{i}" for i in range(5)])
        rr   = pd.DataFrame(rng.normal(0, 1, (n, 5)), index=idx,
                             columns=[f"P{i}" for i in range(5)])
        full = e3_global_state(atm, rr)
        for T in _truncation_dates(idx):
            trunc = e3_global_state(atm.loc[:T], rr.loc[:T])
            _check_no_lookahead(full.to_frame("s"), trunc.to_frame("s"),
                                 f"E3 state T={T}")


class TestNoLookaheadSignalB:

    def test_monthly_betas_no_lookahead(self, data):
        """
        Zhang betas estimated within each month's own data must not bleed across
        months. Truncating after month T should give identical betas for all
        prior months.
        """
        rx_d = data["rx_d"]
        # build minimal factor series
        rng  = np.random.default_rng(11)
        n    = len(rx_d)
        carry_d  = pd.Series(rng.normal(0, 0.003, n), index=rx_d.index)
        dollar_d = pd.Series(rng.normal(0, 0.003, n), index=rx_d.index)

        b_c_full, b_d_full = monthly_betas(rx_d, carry_d, dollar_d)
        T = rx_d.index[int(len(rx_d) * 0.7)]
        b_c_t, b_d_t = monthly_betas(rx_d.loc[:T], carry_d.loc[:T], dollar_d.loc[:T])

        # Find month-end dates before T in the full result
        common = b_c_full.index.intersection(b_c_t.index)[:-1]
        if len(common) == 0:
            return
        diff = (b_c_full.reindex(common) - b_c_t.reindex(common)).abs().max().max()
        assert diff < 1e-10, f"Beta lookahead detected: max diff = {diff:.2e}"
