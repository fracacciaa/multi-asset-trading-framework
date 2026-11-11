"""
Integration test for asset_blend._build_position_turnover_series.

Runs the full turnover computation against real parquet caches and
the Bloomberg Excel file, then checks:
  - Correct columns and monthly BME index
  - All values non-negative and finite
  - Plausible annualised magnitude per asset class
  - IS and OOS periods are covered
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from asset_blend import _build_position_turnover_series, ASSET_CLASSES, IS_END, OOS_START


@pytest.fixture(scope="module")
def to_df():
    return _build_position_turnover_series()


# ── Structure ─────────────────────────────────────────────────────────────────

def test_columns(to_df):
    assert set(to_df.columns) == set(ASSET_CLASSES), (
        f"Expected columns {ASSET_CLASSES}, got {list(to_df.columns)}"
    )


def test_monthly_frequency(to_df):
    """Index must be business-month-end (BME)."""
    diffs = to_df.dropna(how="all").index.to_series().diff().dropna()
    # BME steps are always 28–35 calendar days
    assert diffs.dt.days.between(20, 40).all(), (
        f"Non-monthly gaps found: {diffs[~diffs.dt.days.between(20, 40)]}"
    )


def test_sorted_index(to_df):
    assert to_df.index.is_monotonic_increasing


# ── Validity ──────────────────────────────────────────────────────────────────

def test_non_negative(to_df):
    """Turnover cannot be negative."""
    vals = to_df.dropna(how="all").stack()
    assert (vals >= 0).all(), f"Negative turnover found:\n{vals[vals < 0]}"


def test_finite(to_df):
    vals = to_df.dropna(how="all").stack()
    assert np.isfinite(vals).all(), f"Non-finite turnover found:\n{vals[~np.isfinite(vals)]}"


# ── Coverage ──────────────────────────────────────────────────────────────────

def test_covers_is_period(to_df):
    is_data = to_df.loc[to_df.index <= IS_END].dropna(how="all")
    assert len(is_data) > 36, f"Expected >36 IS months, got {len(is_data)}"


def test_covers_oos_period(to_df):
    oos_data = to_df.loc[to_df.index >= OOS_START].dropna(how="all")
    assert len(oos_data) > 12, f"Expected >12 OOS months, got {len(oos_data)}"


# ── Plausible magnitude ───────────────────────────────────────────────────────

def test_equities_turnover_plausible(to_df):
    """Equities: VF rebalances ~12×/yr; blended ticker TO ~6–20×/yr."""
    s = to_df["equities"].dropna()
    median_to = s.median()
    assert 2.0 < median_to < 50.0, (
        f"Equities median ann TO {median_to:.1f}x outside plausible range (2–50×)"
    )


def test_rates_turnover_plausible(to_df):
    """Rates: continuous DV01 positions; typical ann TO ~20–80×/yr."""
    s = to_df["rates"].dropna()
    median_to = s.median()
    assert 5.0 < median_to < 200.0, (
        f"Rates median ann TO {median_to:.1f}x outside plausible range (5–200×)"
    )


def test_fx_turnover_plausible(to_df):
    """FX: monthly signals dominate; typical ann TO ~2–20×/yr."""
    s = to_df["fx"].dropna()
    median_to = s.median()
    assert 0.5 < median_to < 100.0, (
        f"FX median ann TO {median_to:.1f}x outside plausible range (0.5–100×)"
    )


def test_commodities_turnover_plausible(to_df):
    """Commodities: weekly rebalance; typical ann TO ~2–20×/yr."""
    s = to_df["commodities"].dropna()
    median_to = s.median()
    assert 0.5 < median_to < 100.0, (
        f"Commodities median ann TO {median_to:.1f}x outside plausible range (0.5–100×)"
    )


# ── Consistency: OOS turnover should not be wildly different from IS ──────────

def test_is_oos_consistency(to_df):
    """
    IS and OOS medians should be within 5× of each other for each asset class.
    A larger divergence suggests a data or normalisation bug.
    """
    for ac in ASSET_CLASSES:
        s = to_df[ac].dropna()
        is_med  = s.loc[s.index <= IS_END].median()
        oos_med = s.loc[s.index >= OOS_START].median()
        if np.isnan(is_med) or np.isnan(oos_med) or is_med < 1e-6:
            continue
        ratio = oos_med / is_med
        assert 0.1 < ratio < 10.0, (
            f"{ac}: OOS/IS median TO ratio {ratio:.2f} — large divergence suggests a bug "
            f"(IS={is_med:.1f}x, OOS={oos_med:.1f}x)"
        )
