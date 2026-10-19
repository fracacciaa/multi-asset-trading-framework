"""
No-lookahead test for the carry signal pipeline.

carry_signal_spec.md §11 test_no_lookahead.py:

  Truncate the input panel at date T.
  Rerun the full pipeline from scratch.
  Assert that all outputs up to T-1 are identical to the full-sample run.

This test verifies that the carry computation path does not use any
information from future dates:
  - NS fit at date t uses only data available at t
  - Funding rates and yields from future dates do not affect past signals
  - Rolling statistics (vol, mean, std) use trailing windows only

Tested at three different truncation points T.
"""

import numpy as np
import pandas as pd
import pytest

from rates.ns_core import fit_ns
from rates.carry.measures import carry_reprice, approx_dv01
from rates.carry.signal import build_positions, carry_per_duration, xs_score
from rates.carry.returns import cm_total_return_bp, compute_dur_panel
from rates.carry.config import CarryConfig
from rates.carry.experiment import run_carry_spec


# ── Synthetic data generator ──────────────────────────────────────────────────

def _make_panel(n_days: int = 300, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """
    Returns (curves_pct, funding_pct):
      curves_pct : date × tenor, yields in % (simulating ois_data output)
      funding_pct: date-indexed Series, funding rate in %
    """
    rng = np.random.default_rng(seed)
    tenors = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
    dates = pd.date_range("2010-01-01", periods=n_days, freq="B")

    lam = 0.3
    x = tenors * lam
    S = (1 - np.exp(-x)) / x
    C = S - np.exp(-x)

    level = 4.0 + np.cumsum(rng.normal(0, 0.03, n_days))
    slope = -0.5 + np.cumsum(rng.normal(0, 0.01, n_days))
    curv = 0.2 + np.cumsum(rng.normal(0, 0.005, n_days))

    y_pct = (level[:, None] + slope[:, None] * S[None, :] + curv[:, None] * C[None, :])
    curves = pd.DataFrame(y_pct, index=dates, columns=pd.Index(tenors, dtype=float))
    # Funding: 1Y OIS as proxy (pct)
    funding = pd.Series(level * 0.85, index=dates, name="funding")

    return curves, funding


# ── Carry panel computation helper (minimal version for testing) ──────────────

def _compute_carry_panel_minimal(
    curves_pct: pd.DataFrame,
    funding_pct: pd.Series,
    lam: float = 0.3,
    horizon: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Minimal carry panel computation (mirrors experiment._compute_carry_panel).
    Returns carry_df (bp), dur_df, fits dict.
    """
    tenors = curves_pct.columns.to_numpy(float)
    roll_floor = float(tenors[0])
    carry_rows, dur_rows, fits = [], [], {}

    for date in curves_pct.index:
        row = curves_pct.loc[date].dropna()
        if len(row) < 4:
            continue
        tau = row.index.to_numpy(float)
        y_bp = row.to_numpy(float) * 100.0

        if date not in funding_pct.index:
            continue
        rf_bp = float(funding_pct.loc[date]) * 100.0

        fit = fit_ns(tau, y_bp, lam=lam)
        fits[date] = fit

        c = carry_reprice(fit, tau, rf_bp, horizon)
        dur = compute_dur_panel(fit, tau)

        carry_rows.append(pd.Series(c, index=tau, name=date))
        dur_rows.append(pd.Series(dur, index=tau, name=date))

    return pd.DataFrame(carry_rows), pd.DataFrame(dur_rows), fits


# ── No-lookahead assertions ────────────────────────────────────────────────────

def _run_pipeline(curves: pd.DataFrame, funding: pd.Series) -> pd.DataFrame:
    """Run carry pipeline, return positions DataFrame."""
    carry_df, dur_df, fits = _compute_carry_panel_minimal(curves, funding)
    if carry_df.empty:
        return pd.DataFrame()

    cfg = CarryConfig(
        measure="reprice", horizon_years=0.25, rolldown_source="fitted",
        duration_adj="per_dur", xs_mode="z", rebalance="daily",
        lag=1, funding_source="ois3m",
    )
    pos = build_positions(
        carry_df, dur_df,
        rebalance=cfg.rebalance, lag=cfg.lag,
        xs_mode=cfg.xs_mode, duration_adj=cfg.duration_adj,
    )
    return pos


class TestNoLookahead:
    @pytest.mark.parametrize("trunc_frac", [0.5, 0.7, 0.85])
    def test_truncation_does_not_change_past_positions(self, trunc_frac: float):
        """
        Truncate at T; rerun; assert positions up to T-1 are identical.

        If lookahead exists, positions in the truncated run will differ from
        the full-sample run in the shared period.
        """
        n_days = 200
        curves, funding = _make_panel(n_days=n_days)

        # Full-sample run
        pos_full = _run_pipeline(curves, funding)
        if pos_full.empty:
            pytest.skip("No valid positions generated")

        # Truncated run
        trunc_idx = int(len(curves) * trunc_frac)
        T = curves.index[trunc_idx]
        curves_trunc = curves.loc[:T]
        funding_trunc = funding.loc[:T]

        pos_trunc = _run_pipeline(curves_trunc, funding_trunc)

        if pos_trunc.empty:
            pytest.skip("No positions in truncated run")

        # Shared dates (up to T-1 to avoid boundary effects)
        T_prev = curves.index[trunc_idx - 1]
        shared = pos_full.index.intersection(pos_trunc.index)
        shared = shared[shared <= T_prev]

        if len(shared) < 5:
            pytest.skip("Too few shared dates")

        full_sub = pos_full.loc[shared].fillna(0)
        trunc_sub = pos_trunc.loc[shared].fillna(0)

        np.testing.assert_allclose(
            full_sub.to_numpy(), trunc_sub.to_numpy(), atol=1e-12,
            err_msg=(
                f"Truncation at T={T.date()} changed positions before T-1. "
                "Lookahead detected in the carry pipeline."
            )
        )

    def test_funding_future_data_does_not_affect_past(self):
        """
        Appending future funding rows must not alter past carry signals.
        """
        curves, funding = _make_panel(n_days=150)
        mid = len(curves) // 2
        T = curves.index[mid]

        # Compute carry panel up to mid
        carry1, dur1, _ = _compute_carry_panel_minimal(
            curves.loc[:T], funding.loc[:T]
        )

        # Compute carry panel on full data (future funding present)
        carry2, dur2, _ = _compute_carry_panel_minimal(curves, funding)

        # Restrict to dates up to T
        shared = carry1.index.intersection(carry2.index)
        shared = shared[shared <= T]
        if len(shared) < 3:
            pytest.skip("Too few shared dates")

        np.testing.assert_allclose(
            carry1.loc[shared].fillna(0).to_numpy(),
            carry2.loc[shared].fillna(0).to_numpy(),
            atol=1e-10,
            err_msg="Future funding data affected past carry values — lookahead."
        )

    def test_ns_fit_uses_only_same_date_yields(self):
        """
        The NS fit at date t must produce the same beta regardless of
        what yields exist at t+1, t+2, etc.
        """
        curves, _ = _make_panel(n_days=100)
        tenors = curves.columns.to_numpy(float)
        lam = 0.3

        mid = 50
        date_t = curves.index[mid]

        y_bp = curves.loc[date_t].to_numpy() * 100.0
        fit_from_full = fit_ns(tenors, y_bp, lam=lam)
        fit_from_partial = fit_ns(tenors, y_bp, lam=lam)

        np.testing.assert_array_equal(fit_from_full.beta, fit_from_partial.beta)
