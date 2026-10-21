"""
Tests for Family R and Family V mean-reversion signals (spec §7).

All tests use synthetic data only — no Bloomberg file dependency.

§7.1 Signs
  - Value sign: q 20% above q_bar → hl_value_signal positive, E[Δs] negative
  - Napkin calibration: 1 - RHO_3Y**h matches paper's table
  - Reversal sign: pair that fell 3% over 10 days → positive xs_reversal weight
  - Convention bridge: q == -rer_ecb on synthetic example

§7.2 Construction
  - xs_reversal weights sum to zero, gross exposure = 2
  - apply_holding at h constant weight returns same constant after h periods
  - recursive_mean at t uses no data from t onward (no-lookahead)
  - large_move_mask pass rate 2%–15% at thresh=2.0

§7.3 Replication diagnostics
  - ρ calibration matches paper's napkin numbers (3 decimal places)

§7.4 No-lookahead
  - Truncate at T, recompute, assert identity up to T-1
  - Check: recursive_mean, vol_regime_percentile
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fx.production.signal_mrev_r import (
    past_return,
    trailing_vol,
    xs_reversal,
    ts_reversal,
    apply_holding,
    large_move_mask,
    vol_regime_percentile,
    economic_significance,
)
from fx.production.signal_mrev_v import (
    rho_from_halflife,
    RHO_3Y,
    G10_ONLY,
    recursive_mean,
    hl_value_signal,
    two_hl_value_signal,
    value_weights,
    nominal_value_signal,
    c1_independent_books,
    c2_carry_filter,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _daily_dates(n: int = 500) -> pd.DatetimeIndex:
    return pd.bdate_range("2010-01-04", periods=n)


def _monthly_dates(n: int = 120) -> pd.DatetimeIndex:
    return pd.date_range("2010-01-31", periods=n, freq="BME")


def _daily_spot(n: int = 500, pairs: list[str] | None = None, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pairs = pairs or ["AUDUSD", "USDJPY", "EURUSD", "USDCHF", "USDZAR",
                      "USDMXN", "GBPUSD", "NZDUSD", "USDTRY", "USDNOK"]
    # random walk log spot
    returns = rng.normal(0.0, 0.006, (n, len(pairs)))
    levels  = np.cumsum(returns, axis=0) + 0.5   # start around 0.5
    return pd.DataFrame(levels, index=_daily_dates(n), columns=pairs)


def _monthly_frame(n: int = 120, pairs: list[str] | None = None, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pairs = pairs or ["AUDUSD", "USDJPY", "EURUSD", "USDCHF", "USDZAR",
                      "USDMXN", "GBPUSD", "NZDUSD", "USDTRY", "USDNOK"]
    return pd.DataFrame(rng.normal(0.001, 0.03, (n, len(pairs))),
                        index=_monthly_dates(n), columns=pairs)


# ── §7.1 Signs ────────────────────────────────────────────────────────────────

class TestValueSign:

    def test_hl_signal_positive_when_q_above_qbar(self):
        """
        If q sits 20% above q_bar (foreign currency is real-cheap),
        hl_value_signal must be positive: go long foreign.
        """
        idx  = _monthly_dates(80)
        cols = ["AUDUSD", "USDJPY"]
        q    = pd.DataFrame(0.1, index=idx, columns=cols)
        q_bar = pd.DataFrame(-0.1, index=idx, columns=cols)   # q - q_bar = 0.2
        sig = hl_value_signal(q, q_bar, halflife_years=3.0, h=12)
        assert (sig > 0).all().all(), "Value signal must be positive when q > q_bar"

    def test_expected_spot_change_negative_when_foreign_cheap(self):
        """
        Long foreign gains when s falls. Value signal positive → expected s falls
        → foreign currency appreciates.

        E[Δs] = -(1-ρ^h)(q-q_bar) < 0 when q > q_bar.
        """
        rho = rho_from_halflife(3.0)
        h = 12
        q_gap = 0.20
        exp_ds = -(1.0 - rho ** h) * q_gap
        assert exp_ds < 0, "Expected spot change must be negative (foreign appreciates)"

    def test_convention_bridge(self):
        """
        q == -rer_ecb where rer_ecb is defined as log(USD per FCU) + p_f - p_us.
        In codebase convention: s = log(FCU/USD), q = s + p_us - p_f.
        With ner = -s: rer_ecb = ner + (p_f - p_us) = -s + rpi.
        Therefore q = s - rpi = s + p_us - p_f = -(rer_ecb) ✓
        """
        s   = np.log(1.20)    # 1.20 FCU per USD
        rpi = np.log(1.05)    # 5% cumulative inflation differential (p_f > p_us)
        ner = -s              # log(USD per FCU) as paper defines it
        rer_ecb = ner + rpi   # paper's rer = ner + (p_f - p_us)
        q   = s - rpi         # codebase q = s + p_us - p_f = s - rpi
        assert abs(q - (-rer_ecb)) < 1e-12, "q must equal -rer_ecb"


class TestNapkinCalibration:

    @pytest.mark.parametrize("h, expected", [
        (6,  0.109),
        (12, 0.206),
        (24, 0.370),
        (36, 0.500),
    ])
    def test_rho_3y_one_minus_rho_h(self, h, expected):
        """1 - RHO_3Y^h must match the paper's napkin numbers to 3 decimal places."""
        val = 1.0 - RHO_3Y ** h
        assert abs(val - expected) < 5e-4, (
            f"h={h}: expected 1-ρ^h ≈ {expected}, got {val:.4f}"
        )


class TestReversalSign:

    def test_xs_reversal_long_loser(self):
        """
        A pair that fell 3% over 10 days (s rose, foreign currency weakened)
        should get a POSITIVE (long foreign) xs_reversal weight.

        past_return returns positive when foreign appreciated (s fell).
        A falling foreign = negative past_return → contrarian = long = positive weight.
        """
        idx = _daily_dates(50)
        pairs = ["AUDUSD", "USDJPY", "EURUSD", "USDCHF", "USDZAR"]
        s_d = pd.DataFrame(0.5, index=idx, columns=pairs)

        # AUDUSD: s rises by 0.03 over the last 10 days → foreign weakened → loser
        s_d.iloc[-10:, 0] += np.linspace(0, 0.03, 10)
        # others: unchanged
        fd_d = pd.DataFrame(0.001, index=idx, columns=pairs)

        pr = past_return(s_d, fd_d, k=10, mode="spot")
        w  = xs_reversal(pr)

        last_w = w.iloc[-1]
        # AUDUSD fell → past_return is negative → contrarian gives positive weight
        assert last_w["AUDUSD"] > 0, (
            f"Fallen pair should have positive (long) weight; got {last_w['AUDUSD']:.4f}"
        )


# ── §7.2 Construction ─────────────────────────────────────────────────────────

class TestXsReversalProperties:

    def test_weights_sum_to_zero(self):
        s_d  = _daily_spot()
        fd_d = pd.DataFrame(0.001, index=s_d.index, columns=s_d.columns)
        pr   = past_return(s_d, fd_d, k=10, mode="spot")
        w    = xs_reversal(pr)
        row_sums = w.dropna(how="all").sum(axis=1)
        assert (row_sums.abs() < 1e-10).all(), "xs_reversal weights must sum to zero"

    def test_gross_exposure_equals_two(self):
        s_d  = _daily_spot()
        fd_d = pd.DataFrame(0.001, index=s_d.index, columns=s_d.columns)
        pr   = past_return(s_d, fd_d, k=10, mode="spot")
        w    = xs_reversal(pr).dropna(how="all")
        gross = w.abs().sum(axis=1)
        assert (abs(gross - 2.0) < 1e-10).all(), "xs_reversal gross exposure must equal 2"


class TestApplyHolding:

    def test_constant_weight_recovers_after_h(self):
        """A constant weight series should remain constant after h periods."""
        idx = pd.bdate_range("2010-01-04", periods=50)
        w   = pd.DataFrame(1.0, index=idx, columns=["A", "B"])
        wh  = apply_holding(w, h=5)
        # after 5 periods the rolling mean of a constant is the constant
        assert abs(wh.iloc[-1].mean() - 1.0) < 1e-10


class TestRecursiveMean:

    def test_no_lookahead(self):
        """recursive_mean at date t must use no data from t onward."""
        idx = _monthly_dates(100)
        q   = pd.DataFrame(np.arange(100.0).reshape(-1, 1),
                            index=idx, columns=["A"])
        qbar_full = recursive_mean(q, min_periods=10)

        # truncate at T and recompute
        T = 80
        qbar_trunc = recursive_mean(q.iloc[:T], min_periods=10)

        # values before T must be identical
        common = qbar_full.iloc[:T].dropna().index.intersection(
                 qbar_trunc.dropna().index)
        if len(common) > 0:
            diff = (qbar_full.loc[common] - qbar_trunc.loc[common]).abs().max().max()
            assert diff < 1e-10, "recursive_mean must be identical up to truncation point"


class TestLargeMoveMask:

    def test_pass_rate_between_2_and_15_pct(self):
        """At thresh=2.0, about 5% of events should pass under typical FX tails."""
        s_d  = _daily_spot(n=2000)
        fd_d = pd.DataFrame(0.001, index=s_d.index, columns=s_d.columns)
        vol  = trailing_vol(s_d, window=60)
        pr   = past_return(s_d, fd_d, k=10, mode="spot")
        mask = large_move_mask(pr, vol, k=10, thresh=2.0)
        hit_rate = float(mask.stack().mean())
        assert 0.02 <= hit_rate <= 0.15, (
            f"Large-move pass rate at thresh=2.0 should be 2–15%; got {hit_rate:.2%}"
        )


# ── §7.3 Replication diagnostics ─────────────────────────────────────────────

class TestRhoCalibrationsGrid:

    @pytest.mark.parametrize("hl_years", [2.0, 3.0, 4.0, 5.0])
    def test_half_life_at_nominal_horizon(self, hl_years):
        """At h = hl_years*12, exactly half the misalignment should remain."""
        h = int(hl_years * 12)
        rho = rho_from_halflife(hl_years)
        remaining = rho ** h
        assert abs(remaining - 0.5) < 1e-6, (
            f"hl={hl_years}y: ρ^h should be 0.5; got {remaining:.6f}"
        )


# ── §7.4 No-lookahead ─────────────────────────────────────────────────────────

class TestNoLookaheadRecursiveMean:

    @pytest.mark.parametrize("T", [60, 80, 100])
    def test_truncation_identity(self, T: int):
        idx = _monthly_dates(150)
        rng = np.random.default_rng(99)
        q   = pd.DataFrame(rng.normal(0, 0.2, (150, 3)), index=idx,
                            columns=["AUDUSD", "USDJPY", "EURUSD"])

        qbar_full  = recursive_mean(q, min_periods=10)
        qbar_trunc = recursive_mean(q.iloc[:T], min_periods=10)

        common = qbar_full.iloc[:T].dropna(how="all").index.intersection(
                 qbar_trunc.dropna(how="all").index)
        if len(common) == 0:
            pytest.skip("No overlapping non-NaN dates after min_periods warm-up")

        diff = (qbar_full.loc[common] - qbar_trunc.loc[common]).abs().max().max()
        assert diff < 1e-10, f"recursive_mean identity fails at T={T}: max diff={diff:.2e}"


class TestNoLookaheadVolRegime:

    def test_vol_regime_truncation_identity(self):
        """vol_regime_percentile should be identical up to truncation point."""
        s_d   = _daily_spot(n=1000)
        vol_d = trailing_vol(s_d, window=60)

        T = 800
        regime_full  = vol_regime_percentile(vol_d, mode="realised")
        regime_trunc = vol_regime_percentile(vol_d.iloc[:T], mode="realised")

        common = regime_full.iloc[:T].dropna().index.intersection(
                 regime_trunc.dropna().index)
        if len(common) == 0:
            pytest.skip("No overlapping non-NaN dates after warm-up")
        diff = (regime_full.loc[common] - regime_trunc.loc[common]).abs().max()
        assert diff < 1e-10, f"vol_regime identity fails: max diff={diff:.2e}"


# ── Nominal-only G10 restriction ─────────────────────────────────────────────

class TestNominalOnlyRestriction:

    def test_raises_on_em_pairs(self):
        """nominal_value_signal must raise if EM pairs present with nominal_only=True."""
        idx = _monthly_dates(80)
        # include USDTRY (EM)
        s_m = pd.DataFrame(0.5, index=idx, columns=["EURUSD", "USDJPY", "USDTRY"])
        with pytest.raises(ValueError, match="EM pairs"):
            nominal_value_signal(s_m, nominal_only=True)

    def test_passes_with_g10_only(self):
        """nominal_value_signal must succeed with G10 pairs only."""
        idx = _monthly_dates(80)
        s_m = pd.DataFrame(
            np.cumsum(np.random.default_rng(1).normal(0, 0.01, (80, 3)), axis=0) + 0.5,
            index=idx, columns=["EURUSD", "USDJPY", "USDCHF"],
        )
        sig = nominal_value_signal(s_m, halflife_years=3.0, h=12, nominal_only=True)
        assert isinstance(sig, pd.DataFrame)
        assert sig.shape == s_m.shape


# ── Economic significance gate ────────────────────────────────────────────────

class TestEconomicSignificance:

    def test_returns_required_keys(self):
        idx  = pd.bdate_range("2010-01-04", periods=200)
        cols = ["AUDUSD", "USDJPY", "EURUSD"]
        rng  = np.random.default_rng(42)
        w    = pd.DataFrame(rng.normal(0, 0.1, (200, 3)), index=idx, columns=cols)
        rx   = pd.DataFrame(rng.normal(0, 0.01, (200, 3)), index=idx, columns=cols)
        gate = economic_significance(w, rx)
        for key in ("edge_bp", "turnover", "breakeven_bp"):
            assert key in gate, f"missing key: {key}"

    def test_breakeven_equals_edge_over_turnover(self):
        idx  = pd.bdate_range("2010-01-04", periods=200)
        cols = ["A", "B"]
        rng  = np.random.default_rng(7)
        w    = pd.DataFrame(rng.normal(0, 0.1, (200, 2)), index=idx, columns=cols)
        rx   = pd.DataFrame(rng.normal(0.001, 0.01, (200, 2)), index=idx, columns=cols)
        gate = economic_significance(w, rx)
        if gate["turnover"] > 1e-12:
            expected_be = gate["edge_bp"] / gate["turnover"]
            assert abs(gate["breakeven_bp"] - expected_be) < 1e-8


# ── Value weights ──────────────────────────────────────────────────────────────

class TestValueWeights:

    def test_long_short_structure(self):
        idx  = _monthly_dates(50)
        cols = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        rng  = np.random.default_rng(3)
        sig  = pd.DataFrame(rng.normal(0, 1, (50, 10)), index=idx, columns=cols)
        w    = value_weights(sig, n=4)

        long_exposure  = (w > 0).sum(axis=1)
        short_exposure = (w < 0).sum(axis=1)
        # exactly 4 long and 4 short when all pairs available
        valid = sig.notna().all(axis=1)
        assert (long_exposure[valid]  == 4).all()
        assert (short_exposure[valid] == 4).all()
        # sum to zero
        assert (w.sum(axis=1).abs() < 1e-10).all()


# ── C2 filter preserves gross exposure ────────────────────────────────────────

class TestC2Filter:

    def test_output_gross_exposure_at_most_two(self):
        """
        c2_carry_filter normalises gross exposure to 2. Dollar neutrality is NOT
        guaranteed: if unequal numbers of longs vs shorts are dropped, the
        remaining portfolio carries a net exposure. The test checks the gross cap.
        """
        idx   = _monthly_dates(50)
        cols  = ["AUDUSD", "USDJPY", "EURUSD", "USDCHF",
                 "USDZAR", "USDMXN", "GBPUSD", "NZDUSD"]
        rng   = np.random.default_rng(5)
        # use a balanced 4-long/4-short carry portfolio (dollar-neutral input)
        from fx.production.signal_d import carry_weights
        fd = pd.DataFrame(
            rng.normal(0.005, 0.01, (50, 8)), index=idx, columns=cols,
        )
        w_c   = carry_weights(fd, n=4)
        v_sig = pd.DataFrame(rng.normal(0, 1, (50, 8)), index=idx, columns=cols)
        w_out = c2_carry_filter(w_c, v_sig, pct=1.0 / 3.0)
        gross = w_out.abs().sum(axis=1).dropna()
        assert (gross <= 2.0 + 1e-10).all(), (
            "c2_carry_filter gross exposure must not exceed 2"
        )

    def test_no_long_overvalued_pairs(self):
        """
        After filtering, no long position should belong to the most-expensive
        tercile (lowest value rank < 1/3) from a balanced carry portfolio.
        """
        idx   = _monthly_dates(50)
        cols  = ["AUDUSD", "USDJPY", "EURUSD", "USDCHF",
                 "USDZAR", "USDMXN", "GBPUSD", "NZDUSD"]
        rng   = np.random.default_rng(5)
        from fx.production.signal_d import carry_weights
        fd = pd.DataFrame(rng.normal(0.005, 0.01, (50, 8)), index=idx, columns=cols)
        w_c   = carry_weights(fd, n=4)
        v_sig = pd.DataFrame(rng.normal(0, 1, (50, 8)), index=idx, columns=cols)
        w_out = c2_carry_filter(w_c, v_sig, pct=1.0 / 3.0)

        rank_pct = v_sig.rank(axis=1, pct=True)
        expensive = rank_pct < (1.0 / 3.0)
        # no long position should coincide with the most-expensive label
        long_and_expensive = (w_out > 1e-10) & expensive
        assert not long_and_expensive.any().any(), (
            "c2_carry_filter should not hold long positions in overvalued pairs"
        )
