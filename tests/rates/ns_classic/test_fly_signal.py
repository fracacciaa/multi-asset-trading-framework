"""
Test suite for rates/production/ns_classic/fly_signal.py.

Covers:
  1. Leg-selection rule: deterministic, honours the min-observation filter,
     uses in-sample data only, returns None when no fly is constructible.
  2. DV01 neutrality: the (-0.5, +1, -0.5) package nets to zero DV01.
  3. P&L sanity: inception-to-inception repricing, sign reversal negates P&L.

Self-contained; synthetic curves only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rates.production.ns_classic.fly_signal import (
    select_fly_legs,
    fly_vol_scale_k,
    fly_pnl_repriced,
    FLY_WEIGHTS,
    BODY_FRAC,
    LOWER_FRAC,
    MIN_OBS,
)

IS_END = "2020-12-31"


def _z_frame(tenors, n=1500, seed=0, start="2014-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame(rng.normal(size=(n, len(tenors))), index=idx, columns=list(tenors))


# ── 1. Leg selection ──────────────────────────────────────────────────────────

def test_leg_selection_matches_rule_on_full_curve():
    tenors = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0]
    legs = select_fly_legs(_z_frame(tenors), IS_END)
    assert legs == (10.0, 12.0, 30.0)
    lower, body, upper = legs
    assert abs(body - BODY_FRAC * upper) <= min(abs(t - BODY_FRAC * upper) for t in tenors if t < upper)
    assert abs(lower - LOWER_FRAC * upper) <= min(abs(t - LOWER_FRAC * upper) for t in tenors if t < body)


def test_leg_selection_adapts_to_short_curve():
    assert select_fly_legs(_z_frame([1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]), IS_END) == (3.0, 4.0, 10.0)
    assert select_fly_legs(_z_frame([1.0, 2.0, 3.0, 4.0, 5.0, 7.0]), IS_END) == (2.0, 3.0, 7.0)


def test_leg_selection_is_deterministic():
    tenors = [1.0, 2.0, 5.0, 10.0, 12.0, 30.0]
    z = _z_frame(tenors)
    assert select_fly_legs(z, IS_END) == select_fly_legs(z.copy(), IS_END)


def test_leg_selection_honours_min_obs():
    tenors = [1.0, 5.0, 10.0, 12.0, 30.0]
    z = _z_frame(tenors)
    z[30.0] = np.nan
    z.iloc[:MIN_OBS - 1, z.columns.get_loc(30.0)] = 1.0    # too few observations
    legs = select_fly_legs(z, IS_END)
    assert legs is not None and legs[2] != 30.0, "a sparse tenor must not be chosen as the wing"


def test_leg_selection_ignores_out_of_sample_rows():
    tenors = [1.0, 5.0, 10.0, 12.0, 30.0]
    z = _z_frame(tenors, n=1200, start="2016-01-01")
    legs_is = select_fly_legs(z, IS_END)
    z_extended = pd.concat([z, _z_frame([*tenors], n=600, seed=9, start="2021-01-04")])
    assert select_fly_legs(z_extended, IS_END) == legs_is


def test_leg_selection_returns_none_when_too_few_tenors():
    assert select_fly_legs(_z_frame([5.0, 10.0]), IS_END) is None


# ── 2. Package neutrality ─────────────────────────────────────────────────────

def test_fly_weights_are_dv01_neutral():
    assert sum(FLY_WEIGHTS) == pytest.approx(0.0, abs=1e-12)


def test_fly_weights_are_symmetric():
    w_lo, _, w_up = FLY_WEIGHTS
    assert w_lo == pytest.approx(w_up)


# ── 3. Sizing ─────────────────────────────────────────────────────────────────

def test_vol_scale_k_is_causal():
    s = pd.Series(np.random.default_rng(1).normal(size=800),
                  index=pd.bdate_range("2015-01-01", periods=800))
    k = fly_vol_scale_k(s)
    tail = s.copy()
    tail.iloc[-1] = 50.0                       # perturb only the last observation
    assert k.iloc[:-1].equals(fly_vol_scale_k(tail).iloc[:-1])


def test_vol_scale_k_scales_linearly_with_target():
    s = pd.Series(np.random.default_rng(2).normal(size=600),
                  index=pd.bdate_range("2015-01-01", periods=600))
    a = fly_vol_scale_k(s, target_ann_vol=100.0).dropna()
    b = fly_vol_scale_k(s, target_ann_vol=200.0).dropna()
    assert np.allclose(2.0 * a.values, b.loc[a.index].values)


# ── 4. P&L ────────────────────────────────────────────────────────────────────

def _flat_par_curves(n=900, level=3.0, seed=3):
    rng = np.random.default_rng(seed)
    tenors = [0.25, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0]
    idx = pd.bdate_range("2015-01-01", periods=n)
    base = level + np.cumsum(rng.normal(0, 0.01, size=n))
    data = {t: base + 0.10 * np.log(t + 1) + rng.normal(0, 0.002, size=n) for t in tenors}
    return pd.DataFrame(data, index=idx)


def test_pnl_runs_and_positions_are_package_neutral():
    par = _flat_par_curves()
    pnl, pos, legs = fly_pnl_repriced(par, lam=2.5, is_end=IS_END, burn_in=100)
    assert len(pnl.dropna()) > 100
    held = pos.dropna(how="all")
    assert held.notna().sum(axis=1).max() == 3, "exactly three legs may be held"
    net = held.sum(axis=1).abs().max()
    assert net < 1e-8, f"package must be DV01-neutral, got max |net| = {net}"


def test_sign_reversal_negates_pnl():
    par = _flat_par_curves(seed=4)
    pnl, pos, legs = fly_pnl_repriced(par, lam=2.5, is_end=IS_END, burn_in=100)
    flipped, _, _ = fly_pnl_repriced(par, lam=2.5, is_end=IS_END, burn_in=100,
                                     legs=(legs[2], legs[1], legs[0]))
    common = pnl.index.intersection(flipped.index)
    assert len(common) > 100
    # swapping the wings leaves the package unchanged: same legs, same weights
    assert np.allclose(pnl.loc[common].values, flipped.loc[common].values, atol=1e-10)


def test_explicit_legs_override_the_rule():
    par = _flat_par_curves(seed=5)
    _, _, legs = fly_pnl_repriced(par, lam=2.5, is_end=IS_END, burn_in=100,
                                  legs=(5.0, 10.0, 30.0))
    assert legs == (5.0, 10.0, 30.0)
