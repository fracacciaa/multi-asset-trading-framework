"""
Tests for fx/blend.py.

All blend-algorithm tests use synthetic managed returns (random data) to avoid
depending on the Bloomberg parquet files.  load_managed_returns is tested against
the real production parquets.
"""

import numpy as np
import pandas as pd
import pytest

from equities.blend import (
    BlendWeights, build_state_features,
    equal_weight, sharpe_heuristic, msrr_ridge,
    enet_blend, c_enet_blend, e_enet_blend, inv_msfe_blend,
    pcr_blend, pls_blend, pool_blend, fit_static,
    MLMSRR, LinearDynamicBlend, GBRTBlend, RandomForestBlend, DMABlend,
)
from fx.blend import load_managed_returns, combined_return, SIGNAL_FILES, _SIGNALS_DIR

_SIGNAL_NAMES = list(SIGNAL_FILES.keys())   # 6 production signals


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_F():
    """T=500, K=6 synthetic managed returns (matches FX signal count)."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2010-01-04", periods=500, freq="B")
    return pd.DataFrame(
        rng.standard_normal((500, 6)) * 3,
        index=dates,
        columns=_SIGNAL_NAMES,
    )


@pytest.fixture
def train_test(synthetic_F):
    n = int(len(synthetic_F) * 0.7)
    return synthetic_F.iloc[:n], synthetic_F.iloc[n:]


# ── load_managed_returns ──────────────────────────────────────────────────────

def test_load_managed_returns_shape():
    F = load_managed_returns()
    assert F.shape[1] == 6
    assert set(F.columns) == set(_SIGNAL_NAMES)


def test_load_managed_returns_no_nan():
    F = load_managed_returns()
    assert F.notna().all().all(), "F should have no NaN rows after inner join"


def test_load_managed_returns_sorted():
    F = load_managed_returns()
    assert F.index.is_monotonic_increasing


def test_load_managed_returns_start_date():
    F = load_managed_returns()
    # carry_bnp_e3 first valid is 2006-08-25; F cannot start before that
    assert F.index[0].year >= 2006


def test_load_managed_returns_pair_filter():
    """Passing a custom pair list changes managed-return magnitudes."""
    pairs_full = None   # default 19 pairs
    pairs_small = ["EURUSD", "USDJPY", "GBPUSD"]
    F_full  = load_managed_returns(pairs=pairs_full)
    F_small = load_managed_returns(pairs=pairs_small)
    assert F_small.shape[1] == 6
    assert not F_small.equals(F_full)


# ── combined_return ───────────────────────────────────────────────────────────

def test_combined_return_shape(synthetic_F):
    bw = equal_weight(list(synthetic_F.columns))
    r = combined_return(synthetic_F, bw)
    assert isinstance(r, pd.Series)
    assert len(r) == len(synthetic_F)


def test_combined_return_equal_weight(synthetic_F):
    bw = equal_weight(list(synthetic_F.columns))
    r = combined_return(synthetic_F, bw)
    expected = synthetic_F.mean(axis=1)
    pd.testing.assert_series_equal(r.rename(None), expected.rename(None), check_names=False)


# ── Static blend methods on synthetic F ──────────────────────────────────────

def test_sharpe_heuristic_finite(synthetic_F):
    bw = sharpe_heuristic(synthetic_F)
    assert bw.weights.notna().all()
    assert abs(bw.weights.abs().sum() - 1.0) < 1e-6


def test_msrr_ridge_finite(synthetic_F):
    bw = msrr_ridge(synthetic_F, lam=0.01)
    assert bw.weights.notna().all()
    assert abs(bw.weights.abs().sum() - 1.0) < 1e-6


def test_enet_blend_fx(synthetic_F):
    bw = enet_blend(synthetic_F, alpha=1e-4)
    assert bw.weights.notna().all()
    assert abs(bw.weights.abs().sum() - 1.0) < 1e-6


def test_c_enet_blend_fx(synthetic_F):
    bw = c_enet_blend(synthetic_F, alpha=1e-4)
    assert bw.weights.notna().all()


def test_e_enet_blend_fx(synthetic_F):
    bw = e_enet_blend(synthetic_F, alpha=1e-4)
    assert bw.weights.notna().all()


def test_inv_msfe_blend_fx(synthetic_F):
    bw = inv_msfe_blend(synthetic_F)
    assert bw.weights.notna().all()
    assert abs(bw.weights.sum() - 1.0) < 1e-6


def test_pcr_blend_fx(synthetic_F):
    bw = pcr_blend(synthetic_F, n_components=3)
    assert bw.weights.notna().all()


def test_pls_blend_fx(synthetic_F):
    bw = pls_blend(synthetic_F, n_components=3)
    assert bw.weights.notna().all()


def test_fit_static_fx(synthetic_F):
    static = fit_static(
        {s: synthetic_F[[s]].rename(columns={s: "asset"}) for s in _SIGNAL_NAMES},
        pd.DataFrame(
            np.random.default_rng(2).standard_normal((len(synthetic_F), 1)) * 0.001,
            index=synthetic_F.index, columns=["asset"],
        ),
        enet_alpha=1e-4,
    )
    expected_keys = {
        "equal_weight", "sh_p1", "sh_p2", "msrr_ols", "msrr_ridge",
        "enet", "lasso", "c_enet", "e_enet", "e_lasso",
        "inv_msfe", "pcr", "pls",
    }
    assert set(static.keys()) == expected_keys


# ── OOS train/test split discipline ──────────────────────────────────────────

def test_no_lookahead_in_state_features(synthetic_F):
    """z_t must not depend on F[t]: perturbing F at t must not affect z at t or earlier."""
    z = build_state_features(synthetic_F)
    F2 = synthetic_F.copy()
    F2.iloc[-1] = F2.iloc[-1] * 100
    z2 = build_state_features(F2)
    pd.testing.assert_frame_equal(z.iloc[:-1], z2.iloc[:-1])


def test_static_oos_split(synthetic_F, train_test):
    F_tr, F_te = train_test
    bw = msrr_ridge(F_tr, lam=0.01)
    r_oos = combined_return(F_te, bw)
    assert r_oos.notna().any()
    assert len(r_oos) == len(F_te)


# ── MLMSRR on synthetic FX data ───────────────────────────────────────────────

def test_mlmsrr_fx_oos(synthetic_F, train_test):
    F_tr, F_te = train_test
    z = build_state_features(synthetic_F)
    z_tr = z.loc[F_tr.index].dropna()
    z_te = z.loc[F_te.index].dropna()
    common_tr = F_tr.index.intersection(z_tr.index)
    common_te = F_te.index.intersection(z_te.index)

    ml = MLMSRR(n_signals=6, hidden_size=8, objective="quadratic",
                n_epochs=10, seeds=[42])
    ml.fit(F_tr.loc[common_tr], z_tr.loc[common_tr])
    w = ml.predict_weights(z_te.loc[common_te])
    assert w.shape == (len(common_te), 6)
    assert w.notna().all().all()


# ── LinearDynamicBlend on synthetic FX data ───────────────────────────────────

def test_linear_dynamic_blend_fx_oos(synthetic_F, train_test):
    F_tr, F_te = train_test
    z = build_state_features(synthetic_F)
    z_tr = z.loc[F_tr.index].dropna()
    z_te = z.loc[F_te.index].dropna()
    common_tr = F_tr.index.intersection(z_tr.index)
    common_te = F_te.index.intersection(z_te.index)

    ld = LinearDynamicBlend(ridge_lam=0.01)
    ld.fit(F_tr.loc[common_tr], z_tr.loc[common_tr])
    w = ld.predict_weights(z_te.loc[common_te])
    assert w.shape == (len(common_te), 6)


# ── DMABlend on synthetic FX data ─────────────────────────────────────────────

def test_dma_blend_fx_weights_sum_to_one(synthetic_F, train_test):
    F_tr, F_te = train_test
    dma = DMABlend(alpha=0.95)
    dma.fit(F_tr)
    w = dma.predict_weights(F_te)
    row_sums = w.sum(axis=1)
    assert (row_sums - 1.0).abs().max() < 1e-6


def test_dma_blend_fx_no_collapse(synthetic_F, train_test):
    """No single signal should dominate weights every day."""
    F_tr, F_te = train_test
    dma = DMABlend(alpha=0.90)
    dma.fit(F_tr)
    w = dma.predict_weights(F_te)
    max_weight_per_row = w.max(axis=1)
    assert (max_weight_per_row < 0.99).mean() > 0.5


# ── GBRTBlend on synthetic FX data ────────────────────────────────────────────

def test_gbrt_blend_fx_finite(synthetic_F, train_test):
    F_tr, F_te = train_test
    z = build_state_features(synthetic_F)
    z_tr = z.loc[F_tr.index].dropna()
    z_te = z.loc[F_te.index].dropna()
    common_tr = F_tr.index.intersection(z_tr.index)
    common_te = F_te.index.intersection(z_te.index)

    gb = GBRTBlend(n_estimators=10, max_depth=2)
    gb.fit(F_tr.loc[common_tr], z_tr.loc[common_tr])
    w = gb.predict_weights(z_te.loc[common_te])
    assert w.notna().all().all()
    assert w.shape == (len(common_te), 6)


# ── RandomForestBlend on synthetic FX data ────────────────────────────────────

def test_rf_blend_fx_finite(synthetic_F, train_test):
    F_tr, F_te = train_test
    z = build_state_features(synthetic_F)
    z_tr = z.loc[F_tr.index].dropna()
    z_te = z.loc[F_te.index].dropna()
    common_tr = F_tr.index.intersection(z_tr.index)
    common_te = F_te.index.intersection(z_te.index)

    rf = RandomForestBlend(n_estimators=10, max_depth=3)
    rf.fit(F_tr.loc[common_tr], z_tr.loc[common_tr])
    w = rf.predict_weights(z_te.loc[common_te])
    assert w.notna().all().all()
    assert w.shape == (len(common_te), 6)
