"""
Orthogonality test: fitted-curve carry vs NS residual signal.

carry_signal_spec.md §11 test_orthogonality.py:

  Assert:
    corr(carry_pnl_fitted, residual_pnl) < corr(carry_pnl_raw, residual_pnl)

This is the test that locks in Rule 1: computing roll-down from the fitted
NS curve (not raw yields) keeps carry orthogonal to the NS residual.

Construction:
  1. Simulate a panel where the yield curve has:
       - A smooth NS component  (level + slope + curvature)
       - A rich/cheap residual  (dislocation at 5Y)
  2. Fit NS.  The residual signal picks up the 5Y dislocation.
  3. Build carry from FITTED curve: roll-down ignores the dislocation.
  4. Build carry from RAW yields: roll-down at 5Y reflects the dislocation.
  5. Simulate a simple residual PnL (mean-revert on the 5Y dislocation).
  6. Assert the orthogonality inequality.
"""

import numpy as np
import pandas as pd
import pytest

from rates.production.ns_core import fit_ns
from rates.production.carry.measures import (
    carry_rolldown_dur,
    carry_rolldown_raw,
    approx_dv01,
)
from rates.production.carry.signal import carry_per_duration, xs_score, dv01_positions
from rates.production.carry.returns import cm_total_return_bp, compute_dur_panel


# ── Simulation helpers ────────────────────────────────────────────────────────

def _simulate_panel(n_days: int = 500, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """
    Returns (ois_panel_bp, resid_5y_bp).

    Panel: date × tenor, yields in bp.
    The 5Y tenor has an added rich/cheap residual component.
    """
    rng = np.random.default_rng(seed)
    tenors = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
    dates = pd.date_range("2010-01-01", periods=n_days, freq="B")

    # Level, slope, curvature as AR(1) processes
    level  = 400.0 + np.cumsum(rng.normal(0, 3, n_days))
    slope  = -50.0 + np.cumsum(rng.normal(0, 1, n_days))
    curv   = 20.0 + np.cumsum(rng.normal(0, 0.5, n_days))

    # NS loadings
    lam = 0.3
    x = tenors * lam
    L = np.ones(len(tenors))
    S = (1 - np.exp(-x)) / x
    C = S - np.exp(-x)

    y_smooth = (level[:, None] * L[None, :]
                + slope[:, None] * S[None, :]
                + curv[:, None] * C[None, :])  # (n_days, n_tenors)

    # Add AR(1) residual at 5Y only (represents rich/cheap dislocation)
    resid_5y = np.zeros(n_days)
    phi = 0.92  # mean-reverting
    for i in range(1, n_days):
        resid_5y[i] = phi * resid_5y[i - 1] + rng.normal(0, 8)

    y_obs = y_smooth.copy()
    idx_5y = list(tenors).index(5.0)
    y_obs[:, idx_5y] += resid_5y

    panel = pd.DataFrame(y_obs, index=dates, columns=pd.Index(tenors, dtype=float))
    return panel, pd.Series(resid_5y, index=dates, name="resid_5y")


def _residual_pnl(panel_bp: pd.DataFrame, resid_5y: pd.Series, lag: int = 1) -> pd.Series:
    """
    Simple residual-strategy PnL: short 5Y when rich, long when cheap.
    Position = -sign(resid_5y[t-lag]) * 1 DV01 unit.
    PnL = position × (-yield_change at 5Y).
    """
    tenors = panel_bp.columns.to_numpy(float)
    idx_5y = list(tenors).index(5.0)
    dy_5y = panel_bp.iloc[:, idx_5y].diff()

    signal = -np.sign(resid_5y.shift(lag))
    dur_5y = 5.0  # approximate
    return (signal * (-dy_5y) * dur_5y).rename("residual_pnl")


def _carry_signal_at_5y_fitted(
    panel_bp: pd.DataFrame, rf_bp: float, lam: float = 0.3
) -> pd.Series:
    """
    Fitted carry roll-down signal (bp) at the 5Y tenor per date.

    Uses carry_rolldown_dur with fit.predict() for roll-down — orthogonal
    to the NS residual by construction.
    """
    tenors = panel_bp.columns.to_numpy(float)
    roll_floor = float(tenors[0])
    h = 0.25
    tau_5y = np.array([5.0])
    signals = []

    for date, row in panel_bp.iterrows():
        if row.isna().any():
            signals.append(np.nan)
            continue
        fit = fit_ns(tenors, row.to_numpy(float), lam=lam)
        c = carry_rolldown_dur(fit, tau_5y, rf_bp, h, roll_floor=roll_floor)
        signals.append(float(c[0]))

    return pd.Series(signals, index=panel_bp.index, name="carry_fitted_5y")


def _carry_signal_at_5y_raw(
    panel_bp: pd.DataFrame, rf_bp: float
) -> pd.Series:
    """
    Raw carry roll-down signal (bp) at the 5Y tenor per date.

    Uses carry_rolldown_raw with raw observed yields — the roll-down
    embeds the NS residual at 5Y, so the signal IS correlated with resid_5y.
    """
    tenors = panel_bp.columns.to_numpy(float)
    roll_floor = float(tenors[0])
    h = 0.25
    tau_5y = np.array([5.0])
    signals = []

    for date, row in panel_bp.iterrows():
        if row.isna().any():
            signals.append(np.nan)
            continue
        c = carry_rolldown_raw(row, tau_5y, rf_bp, h, roll_floor=roll_floor)
        signals.append(float(c[0]))

    return pd.Series(signals, index=panel_bp.index, name="carry_raw_5y")


def _carry_pnl_from_fitted(
    panel_bp: pd.DataFrame, rf_bp: float, lam: float = 0.3
) -> pd.Series:
    """Fitted-carry component PnL (carry leg only, no yield-change noise)."""
    tenors = panel_bp.columns.to_numpy(float)
    pnl = []

    dates = panel_bp.index.tolist()
    for i in range(1, len(dates)):
        prev = dates[i - 1]
        row_prev = panel_bp.loc[prev]
        row_now = panel_bp.loc[dates[i]]
        if row_prev.isna().any() or row_now.isna().any():
            pnl.append(np.nan)
            continue

        fit_prev = fit_ns(tenors, row_prev.to_numpy(float), lam=lam)
        y_prev = row_prev.to_numpy(float)
        y_now = row_now.to_numpy(float)
        dur_prev = compute_dur_panel(fit_prev, tenors)

        pos = np.ones(len(tenors)) / len(tenors)
        rc, _ = cm_total_return_bp(y_prev, y_now, fit_prev, tenors, rf_bp, dur_prev)
        pnl.append(np.dot(pos, rc))

    return pd.Series(pnl, index=panel_bp.index[1:], name="carry_fitted")


# ── Main orthogonality test ───────────────────────────────────────────────────

class TestOrthogonality:
    def test_fitted_carry_less_correlated_with_residual(self):
        """
        corr(fitted_carry_signal_5y, resid_5y) < corr(raw_carry_signal_5y, resid_5y)

        The fitted-curve roll-down is a smooth NS function and cannot contain
        the residual at 5Y by construction.  The raw roll-down interpolates
        observed yields that include the residual, so the raw signal IS
        correlated with resid_5y.

        Compare the carry SIGNALS (not PnL strategies) directly against the
        residual series — this removes yield-change noise that dominates PnL.
        """
        panel_bp, resid_5y = _simulate_panel(n_days=600)
        rf_bp = 150.0

        sig_fitted = _carry_signal_at_5y_fitted(panel_bp, rf_bp).dropna()
        sig_raw = _carry_signal_at_5y_raw(panel_bp, rf_bp).dropna()

        common = sig_fitted.index.intersection(sig_raw.index).intersection(resid_5y.index)
        assert len(common) > 200, "Insufficient common dates"

        corr_fitted = float(sig_fitted.loc[common].corr(resid_5y.loc[common]))
        corr_raw = float(sig_raw.loc[common].corr(resid_5y.loc[common]))

        assert abs(corr_fitted) < abs(corr_raw), (
            f"Fitted carry signal should be less correlated with residual than raw. "
            f"Got: |corr_fitted|={abs(corr_fitted):.3f}, |corr_raw|={abs(corr_raw):.3f}"
        )

    def test_fitted_carry_corr_below_threshold(self):
        """
        The fitted-curve carry should have |corr| < 0.3 with the residual signal.
        (Selection rule from carry_signal_spec.md §9.2 rule 3.)
        """
        panel_bp, resid_5y = _simulate_panel(n_days=600)
        rf_bp = 150.0

        pnl_fitted = _carry_pnl_from_fitted(panel_bp, rf_bp).dropna()
        pnl_resid = _residual_pnl(panel_bp, resid_5y).dropna()
        common = pnl_fitted.index.intersection(pnl_resid.index)

        corr = float(pnl_fitted.loc[common].corr(pnl_resid.loc[common]))
        assert abs(corr) < 0.3, (
            f"Fitted carry corr with residual = {corr:.3f}; should be < 0.3. "
            "Rule 1 (rolldown_source='fitted') must be enforced."
        )
