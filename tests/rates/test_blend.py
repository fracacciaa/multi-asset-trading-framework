"""
Tests for rates/blend.py.

All tests use synthetic managed returns (random data) to avoid depending on
the Bloomberg parquet files.  load_managed_returns is tested separately with
a tmp directory pointing to real parquets.
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
from rates.blend import load_managed_returns, combined_return, SIGNAL_FILES, _SIGNALS_DIR


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_F():
    """T=400, K=3 synthetic managed returns (matches rates signal count)."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2010-01-04", periods=400, freq="B")
    return pd.DataFrame(
        rng.standard_normal((400, 3)) * 5,   # bp scale
        index=dates,
        columns=["momentum", "carry", "mean_reversion"],
    )


@pytest.fixture
def train_test(synthetic_F):
    n = int(len(synthetic_F) * 0.7)
    return synthetic_F.iloc[:n], synthetic_F.iloc[n:]


# ── load_managed_returns ──────────────────────────────────────────────────────

def test_load_managed_returns_shape():
    F = load_managed_returns()
    assert F.shape[1] == 3
    assert set(F.columns) == {"momentum", "carry", "mean_reversion"}


def test_load_managed_returns_no_nan():
    F = load_managed_returns()
    assert F.notna().all().all(), "F should have no NaN rows after inner join"


def test_load_managed_returns_sorted():
    F = load_managed_returns()
    assert F.index.is_monotonic_increasing


def test_load_managed_returns_market_filter():
    F_all = load_managed_returns()
    F_us  = load_managed_returns(markets=["US", "EUR"])
    # fewer markets → different (smaller) absolute magnitude per day
    assert F_us.shape[1] == 3
    assert not F_us.equals(F_all)


def test_load_managed_returns_date_range():
    F = load_managed_returns()
    # carry starts 2007; F cannot start before carry's first date
    assert F.index[0].year >= 2007


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


# ── Static blend methods on synthetic rates F ────────────────────────────────

def test_sharpe_heuristic_finite(synthetic_F):
    bw = sharpe_heuristic(synthetic_F)
    assert bw.weights.notna().all()
    assert abs(bw.weights.abs().sum() - 1.0) < 1e-6


def test_msrr_ridge_finite(synthetic_F):
    bw = msrr_ridge(synthetic_F, lam=0.01)
    assert bw.weights.notna().all()
    assert abs(bw.weights.abs().sum() - 1.0) < 1e-6


def test_enet_blend_rates(synthetic_F):
    bw = enet_blend(synthetic_F, alpha=1e-4)
    assert bw.weights.notna().all()
    assert abs(bw.weights.abs().sum() - 1.0) < 1e-6


def test_c_enet_blend_rates(synthetic_F):
    bw = c_enet_blend(synthetic_F, alpha=1e-4)
    assert bw.weights.notna().all()


def test_inv_msfe_blend_rates(synthetic_F):
    bw = inv_msfe_blend(synthetic_F)
    assert bw.weights.notna().all()
    assert abs(bw.weights.sum() - 1.0) < 1e-6


def test_pcr_blend_rates(synthetic_F):
    bw = pcr_blend(synthetic_F, n_components=2)
    assert bw.weights.notna().all()


def test_fit_static_rates(synthetic_F):
    static = fit_static(
        {"momentum": synthetic_F[["momentum"]].rename(columns={"momentum": "asset"}),
         "carry":    synthetic_F[["carry"]].rename(columns={"carry": "asset"}),
         "mean_reversion": synthetic_F[["mean_reversion"]].rename(columns={"mean_reversion": "asset"})},
        pd.DataFrame(
            np.random.default_rng(1).standard_normal((len(synthetic_F), 1)) * 0.001,
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
    for name, bw in static.items():
        assert bw.weights.notna().all() or bw.weights.isna().all(), name


# ── OOS train/test split discipline ──────────────────────────────────────────

def test_no_lookahead_in_state_features(synthetic_F):
    """z_t must not use F[t]: perturbing F at t must not affect z at dates <= t."""
    z = build_state_features(synthetic_F)
    F2 = synthetic_F.copy()
    F2.iloc[-1] = F2.iloc[-1] * 100   # big change on last day
    z2 = build_state_features(F2)
    # All dates except the last should be unaffected
    pd.testing.assert_frame_equal(z.iloc[:-1], z2.iloc[:-1])


def test_static_oos_split(synthetic_F, train_test):
    F_tr, F_te = train_test
    bw_train = msrr_ridge(F_tr, lam=0.01)
    r_oos = combined_return(F_te, bw_train)
    assert r_oos.notna().any()
    assert len(r_oos) == len(F_te)


# ── MLMSRR on synthetic rates data ───────────────────────────────────────────

def test_mlmsrr_rates_oos(synthetic_F, train_test):
    F_tr, F_te = train_test
    z = build_state_features(synthetic_F)
    z_tr = z.loc[F_tr.index].dropna()
    z_te = z.loc[F_te.index].dropna()
    common_tr = F_tr.index.intersection(z_tr.index)
    common_te = F_te.index.intersection(z_te.index)

    ml = MLMSRR(n_signals=3, hidden_size=8, objective="quadratic",
                n_epochs=10, seeds=[42])
    ml.fit(F_tr.loc[common_tr], z_tr.loc[common_tr])
    w = ml.predict_weights(z_te.loc[common_te])
    assert w.shape == (len(common_te), 3)
    assert w.notna().all().all()


# ── LinearDynamicBlend on synthetic rates data ────────────────────────────────

def test_linear_dynamic_blend_rates_oos(synthetic_F, train_test):
    F_tr, F_te = train_test
    z = build_state_features(synthetic_F)
    z_tr = z.loc[F_tr.index].dropna()
    z_te = z.loc[F_te.index].dropna()
    common_tr = F_tr.index.intersection(z_tr.index)
    common_te = F_te.index.intersection(z_te.index)

    ld = LinearDynamicBlend(ridge_lam=0.01)
    ld.fit(F_tr.loc[common_tr], z_tr.loc[common_tr])
    w = ld.predict_weights(z_te.loc[common_te])
    assert w.shape == (len(common_te), 3)


# ── DMABlend on synthetic rates data ─────────────────────────────────────────

def test_dma_blend_rates_weights_sum_to_one(synthetic_F, train_test):
    F_tr, F_te = train_test
    dma = DMABlend(alpha=0.95)
    dma.fit(F_tr)
    w = dma.predict_weights(F_te)
    row_sums = w.sum(axis=1)
    assert (row_sums - 1.0).abs().max() < 1e-6


def test_dma_blend_rates_no_collapse(synthetic_F, train_test):
    """No single signal should monopolise weights every day."""
    F_tr, F_te = train_test
    dma = DMABlend(alpha=0.90)
    dma.fit(F_tr)
    w = dma.predict_weights(F_te)
    max_weight_per_row = w.max(axis=1)
    assert (max_weight_per_row < 0.99).mean() > 0.5


# ── GBRTBlend on synthetic rates data ────────────────────────────────────────

def test_gbrt_blend_rates_finite(synthetic_F, train_test):
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
    assert w.shape == (len(common_te), 3)
