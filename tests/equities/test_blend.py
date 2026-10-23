"""
Tests for equities/blend.py — all use synthetic deterministic fixtures.
No Bloomberg dependency.
"""

import math

import numpy as np
import pandas as pd
import pytest

from equities.blend import (
    BlendWeights,
    DMABlend,
    GBRTBlend,
    LinearDynamicBlend,
    MLMSRR,
    RandomForestBlend,
    STMOM,
    _xs_positions,
    build_managed_returns,
    build_state_features,
    c_enet_blend,
    combine_signals,
    e_enet_blend,
    enet_blend,
    equal_weight,
    fit_static,
    inv_msfe_blend,
    managed_return,
    msrr_ridge,
    pcr_blend,
    pls_blend,
    pool_blend,
    sharpe_heuristic,
    winsorize_signals,
)


# ── Fixtures ────────────────────────────────────────────────────────────────────

def _make_signals(T=300, N=20, K=4, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=T, freq="B")
    tickers = [f"T{i}" for i in range(N)]
    names = [f"sig{k}" for k in range(K)]
    signals = {
        name: pd.DataFrame(rng.standard_normal((T, N)), index=dates, columns=tickers)
        for name in names
    }
    fwd_ret = pd.DataFrame(
        rng.standard_normal((T, N)) * 0.01, index=dates, columns=tickers
    )
    return signals, fwd_ret, names, dates, tickers


# ── xs positions ────────────────────────────────────────────────────────────────

def test_xs_positions_gross_exposure():
    """Gross exposure should be 1 on every date (up to floating point)."""
    signals, _, _, _, _ = _make_signals()
    sig = list(signals.values())[0]
    pos = _xs_positions(sig)
    gross = pos.abs().sum(axis=1).dropna()
    np.testing.assert_allclose(gross.values, 1.0, atol=1e-12)


def test_xs_positions_dollar_neutral():
    """Positions should sum to ~0 (dollar-neutral by construction of z-score)."""
    signals, _, _, _, _ = _make_signals()
    sig = list(signals.values())[0]
    pos = _xs_positions(sig)
    net = pos.sum(axis=1).dropna()
    np.testing.assert_allclose(net.values, 0.0, atol=1e-10)


# ── equal weight ────────────────────────────────────────────────────────────────

def test_equal_weight_values():
    names = ["a", "b", "c", "d"]
    bw = equal_weight(names)
    assert set(bw.weights.index) == set(names)
    np.testing.assert_allclose(bw.weights.values, 0.25, atol=1e-12)


def test_equal_weight_abs_sum():
    bw = equal_weight(["x", "y", "z"])
    assert abs(bw.weights.abs().sum() - 1.0) < 1e-12


# ── sharpe heuristic ────────────────────────────────────────────────────────────

def test_sharpe_heuristic_positive():
    """Signal with higher Sharpe should receive higher weight."""
    signals, fwd_ret, names, _, _ = _make_signals(seed=7)
    F = build_managed_returns(signals, fwd_ret)

    # Boost sig0 to have clearly positive Sharpe
    F_biased = F.copy()
    F_biased["sig0"] = F_biased["sig0"].abs() + 0.05

    bw = sharpe_heuristic(F_biased, power=1)
    assert bw.weights["sig0"] > bw.weights["sig1"]


def test_sharpe_heuristic_all_negative_fallback():
    """All-negative Sharpe → fallback to equal weight."""
    signals, fwd_ret, names, _, _ = _make_signals(seed=1)
    F = build_managed_returns(signals, fwd_ret)
    F_neg = -F.abs() - 0.1  # force negative mean for all
    bw = sharpe_heuristic(F_neg, power=1)
    assert bw.diagnostics["fallback_to_ew"] is True
    np.testing.assert_allclose(
        bw.weights.values, 1.0 / len(F.columns), atol=1e-12
    )


def test_sharpe_heuristic_power2_more_concentrated():
    """power=2 should concentrate weight more than power=1."""
    signals, fwd_ret, names, _, _ = _make_signals(seed=3)
    F = build_managed_returns(signals, fwd_ret)
    # Ensure at least some positive Sharpe
    F = F.abs()
    bw1 = sharpe_heuristic(F, power=1)
    bw2 = sharpe_heuristic(F, power=2)
    # Concentration: max weight should be higher under power=2
    assert bw2.weights.max() >= bw1.weights.max()


# ── ridge MSRR ──────────────────────────────────────────────────────────────────

def test_msrr_ridge_algebra():
    """msrr_ridge must equal solve(M + λI, μ) exactly."""
    signals, fwd_ret, names, _, _ = _make_signals(seed=2)
    F = build_managed_returns(signals, fwd_ret)
    lam = 0.05

    Fv = F.values.astype(float)
    T = len(Fv)
    mu = Fv.mean(axis=0)
    M = Fv.T @ Fv / T
    K = M.shape[0]
    expected = np.linalg.solve(M + lam * np.eye(K), mu)
    expected /= np.abs(expected).sum()

    bw = msrr_ridge(F, lam)
    np.testing.assert_allclose(bw.weights.values, expected, atol=1e-8)


def test_msrr_ols_direction_matches_mv():
    """
    OLS MSRR (λ=0) direction must match classical max-Sharpe (Σ⁻¹μ).

    From spec §4.1: M⁻¹μ ∝ Σ⁻¹μ by Sherman-Morrison.
    We verify cos similarity ≈ 1 (or −1 if sign flips, but for positive-mean
    sleeves direction should be the same).
    """
    rng = np.random.default_rng(99)
    T, K = 500, 4
    dates = pd.date_range("2018-01-01", periods=T, freq="B")
    F_arr = rng.standard_normal((T, K)) * 0.01 + 0.001  # positive mean
    F = pd.DataFrame(F_arr, index=dates, columns=[f"f{k}" for k in range(K)])

    bw_ols = msrr_ridge(F, lam=0.0)
    b_ols = bw_ols.weights.values

    mu = F_arr.mean(axis=0)
    sigma = np.cov(F_arr.T)
    b_mv = np.linalg.solve(sigma, mu)
    b_mv /= np.abs(b_mv).sum()

    cos_sim = np.dot(b_ols, b_mv) / (np.linalg.norm(b_ols) * np.linalg.norm(b_mv))
    assert abs(cos_sim) > 1 - 1e-6, f"cos_sim={cos_sim:.8f}, expected ≈ ±1"


def test_msrr_ridge_lambda0_same_as_ols():
    """msrr_ridge(lam=0) method label should be 'msrr_ols'."""
    signals, fwd_ret, _, _, _ = _make_signals()
    F = build_managed_returns(signals, fwd_ret)
    bw = msrr_ridge(F, lam=0.0)
    assert bw.method == "msrr_ols"


def test_msrr_ridge_finite_weights_singular():
    """K≥T (near-singular M): lstsq fallback must produce finite weights."""
    rng = np.random.default_rng(5)
    T, K = 10, 20  # more signals than observations
    dates = pd.date_range("2020-01-01", periods=T, freq="B")
    F = pd.DataFrame(
        rng.standard_normal((T, K)) * 0.01,
        index=dates,
        columns=[f"f{k}" for k in range(K)],
    )
    bw = msrr_ridge(F, lam=0.0)
    assert np.all(np.isfinite(bw.weights.values))


# ── no-leakage (static) ─────────────────────────────────────────────────────────

def test_static_no_leakage():
    """Static estimators are pure functions of F_train; future data is irrelevant."""
    signals, fwd_ret, names, dates, tickers = _make_signals(T=300, seed=10)
    F_train = build_managed_returns(signals, fwd_ret).iloc[:150]

    bw_before = msrr_ridge(F_train, lam=0.01)

    # Simulate appending future data to signals (should not change anything)
    rng = np.random.default_rng(99)
    extra_dates = pd.date_range(dates[-1] + pd.offsets.BDay(1), periods=50, freq="B")
    for name in names:
        extra = pd.DataFrame(
            rng.standard_normal((50, len(tickers))), index=extra_dates, columns=tickers
        )
        signals[name] = pd.concat([signals[name], extra])

    bw_after = msrr_ridge(F_train, lam=0.01)
    np.testing.assert_array_equal(bw_before.weights.values, bw_after.weights.values)


# ── combine_signals ─────────────────────────────────────────────────────────────

def test_combine_signals_shape():
    signals, _, names, dates, tickers = _make_signals()
    bw = equal_weight(names)
    combo = combine_signals(signals, bw)
    assert combo.shape == (len(dates), len(tickers))
    assert combo.isna().all(axis=1).sum() == 0


def test_combine_signals_missing_key_skipped():
    """A weight referencing a non-existent signal is silently skipped."""
    signals, _, names, _, _ = _make_signals(K=3)
    w = pd.Series({"sig0": 0.5, "sig1": 0.3, "ghost": 0.2})
    bw = BlendWeights(w, "custom")
    combo = combine_signals(signals, bw)
    assert combo.isna().all(axis=1).sum() == 0


def test_combine_signals_no_match_raises():
    signals, _, _, _, _ = _make_signals()
    w = pd.Series({"NONEXISTENT": 1.0})
    bw = BlendWeights(w, "bad")
    with pytest.raises(ValueError, match="No signals match"):
        combine_signals(signals, bw)


# ── fit_static convenience ──────────────────────────────────────────────────────

def test_fit_static_returns_all_methods():
    signals, fwd_ret, _, _, _ = _make_signals()
    result = fit_static(signals, fwd_ret)
    expected_keys = {
        "equal_weight", "sh_p1", "sh_p2", "msrr_ols", "msrr_ridge",
        "enet", "lasso", "c_enet", "e_enet", "e_lasso",
        "inv_msfe", "pcr", "pls",
    }
    assert set(result.keys()) == expected_keys
    for bw in result.values():
        assert np.all(np.isfinite(bw.weights.values))


# ── state features ──────────────────────────────────────────────────────────────

def test_state_features_shape():
    signals, fwd_ret, _, _, _ = _make_signals(K=4)
    F = build_managed_returns(signals, fwd_ret)
    z = build_state_features(F, lookback=21)
    assert z.shape[0] == len(F)
    assert z.shape[1] == len(F.columns) + 2  # K rolling Sharpes + dispersion + avg_ret


def test_state_features_no_lookahead():
    """Perturbing F after date t must not change z at dates ≤ t."""
    signals, fwd_ret, _, _, _ = _make_signals(K=4)
    F = build_managed_returns(signals, fwd_ret)
    z_orig = build_state_features(F, lookback=21)

    # Perturb all rows after the midpoint
    mid = len(F) // 2
    F_perturbed = F.copy()
    F_perturbed.iloc[mid:] *= 100.0
    z_perturbed = build_state_features(F_perturbed, lookback=21)

    # z values up to (mid-1) must be identical
    pd.testing.assert_frame_equal(
        z_orig.iloc[:mid], z_perturbed.iloc[:mid], check_exact=False, atol=1e-10
    )


# ── ML MSRR ─────────────────────────────────────────────────────────────────────

def test_mlmsrr_seed_reproducibility():
    """Same seed → identical weights."""
    signals, fwd_ret, _, _, _ = _make_signals(K=3, T=200)
    F = build_managed_returns(signals, fwd_ret)
    z = build_state_features(F, lookback=21)
    common = F.index.intersection(z.dropna().index)
    F_c, z_c = F.loc[common], z.loc[common]

    ml1 = MLMSRR(n_signals=3, hidden_size=8, n_epochs=10, seeds=[42])
    ml1.fit(F_c, z_c)
    w1 = ml1.predict_weights(z_c).dropna().values

    ml2 = MLMSRR(n_signals=3, hidden_size=8, n_epochs=10, seeds=[42])
    ml2.fit(F_c, z_c)
    w2 = ml2.predict_weights(z_c).dropna().values

    np.testing.assert_array_equal(w1, w2)


def test_mlmsrr_quadratic_plausibility():
    """
    One synthetic dominant sleeve: after training, it should receive
    strictly more weight than a pure-noise sleeve on average.
    """
    rng = np.random.default_rng(7)
    T, K = 400, 3
    dates = pd.date_range("2018-01-01", periods=T, freq="B")
    names = [f"f{k}" for k in range(K)]

    # f0 has clear positive alpha; others are noise
    alpha = np.full(T, 0.02)
    F_arr = rng.standard_normal((T, K)) * 0.005
    F_arr[:, 0] += alpha
    F = pd.DataFrame(F_arr, index=dates, columns=names)

    z = build_state_features(F, lookback=21)
    common = F.index.intersection(z.dropna().index)
    F_c, z_c = F.loc[common], z.loc[common]

    ml = MLMSRR(
        n_signals=K,
        hidden_size=8,
        objective="quadratic",
        n_epochs=200,
        seeds=[42],
        lam=1e-4,
        patience=50,
    )
    ml.fit(F_c, z_c)
    w = ml.predict_weights(z_c).dropna()
    avg_w = w.mean()
    assert avg_w["f0"] > avg_w["f1"], (
        f"Dominant sleeve f0 avg weight {avg_w['f0']:.4f} "
        f"not > noise sleeve f1 {avg_w['f1']:.4f}"
    )


def test_mlmsrr_combine_signals_shape():
    signals, fwd_ret, _, dates, tickers = _make_signals(K=3, T=200)
    F = build_managed_returns(signals, fwd_ret)
    z = build_state_features(F, lookback=21)
    common = F.index.intersection(z.dropna().index)

    ml = MLMSRR(n_signals=3, hidden_size=8, n_epochs=10, seeds=[42])
    ml.fit(F.loc[common], z.loc[common])

    combo = ml.combine_signals(signals, z)
    assert combo.shape == (len(dates), len(tickers))


def test_mlmsrr_sharpe_objective_runs():
    """Sharpe objective should train without raising."""
    signals, fwd_ret, _, _, _ = _make_signals(K=3, T=200)
    F = build_managed_returns(signals, fwd_ret)
    z = build_state_features(F, lookback=21)
    common = F.index.intersection(z.dropna().index)

    ml = MLMSRR(
        n_signals=3, hidden_size=8, objective="sharpe", n_epochs=10, seeds=[42]
    )
    ml.fit(F.loc[common], z.loc[common])
    w = ml.predict_weights(z.loc[common]).dropna()
    assert np.all(np.isfinite(w.values))


def test_mlmsrr_softmax_weights_sum_to_one():
    """With softmax activation, blend weights must sum to 1 on every date."""
    signals, fwd_ret, _, _, _ = _make_signals(K=4, T=200)
    F = build_managed_returns(signals, fwd_ret)
    z = build_state_features(F, lookback=21)
    common = F.index.intersection(z.dropna().index)

    ml = MLMSRR(n_signals=4, hidden_size=8, output_activation="softmax",
                n_epochs=10, seeds=[42])
    ml.fit(F.loc[common], z.loc[common])
    w = ml.predict_weights(z.loc[common]).dropna()
    row_sums = w.sum(axis=1)
    np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-5)


# ── ENet / LASSO blends ──────────────────────────────────────────────────────

def test_enet_blend_finite():
    """ENet weights must be finite and |w|.sum() = 1."""
    signals, fwd_ret, _, _, _ = _make_signals()
    F = build_managed_returns(signals, fwd_ret)
    bw = enet_blend(F, alpha=0.01, l1_ratio=0.5)
    assert np.all(np.isfinite(bw.weights.values))
    np.testing.assert_allclose(bw.weights.abs().sum(), 1.0, atol=1e-12)


def test_enet_blend_non_negative():
    """With non_negative=True, all weights must be ≥ 0."""
    signals, fwd_ret, _, _, _ = _make_signals()
    F = build_managed_returns(signals, fwd_ret)
    bw = enet_blend(F, alpha=0.001, l1_ratio=0.5, non_negative=True)
    assert np.all(bw.weights.values >= -1e-12)


def test_enet_lasso_sparsity():
    """LASSO (l1_ratio=1) with a dominant sleeve should zero-out weaker sleeves."""
    rng = np.random.default_rng(11)
    T, K = 300, 4
    dates = pd.date_range("2018-01-01", periods=T, freq="B")
    alpha = np.full(T, 0.05)
    F_arr = rng.standard_normal((T, K)) * 0.001
    F_arr[:, 0] += alpha
    F = pd.DataFrame(F_arr, index=dates, columns=[f"f{k}" for k in range(K)])

    bw = enet_blend(F, alpha=0.1, l1_ratio=1.0)
    assert bw.diagnostics["n_selected"] < K, "LASSO should zero out at least one sleeve"


# ── C-ENet ───────────────────────────────────────────────────────────────────

def test_c_enet_selected_equal_weight():
    """When exactly 2 signals are selected, each should have weight 0.5."""
    rng = np.random.default_rng(22)
    T = 400
    dates = pd.date_range("2018-01-01", periods=T, freq="B")
    # Build F where sig0 and sig1 have strong positive expected return; sig2 is noise
    F_arr = rng.standard_normal((T, 3)) * 0.001
    F_arr[:, 0] += 0.05
    F_arr[:, 1] += 0.04
    F = pd.DataFrame(F_arr, index=dates, columns=["sig0", "sig1", "noise"])

    bw = c_enet_blend(F, alpha=0.05, l1_ratio=0.9)
    # If exactly 2 selected, they must each be 0.5 after normalisation
    n_sel = bw.diagnostics["n_selected"]
    active = [k for k, w in bw.weights.items() if w > 1e-8]
    if n_sel == 2 and len(active) == 2:
        np.testing.assert_allclose(bw.weights[active[0]], 0.5, atol=1e-10)
        np.testing.assert_allclose(bw.weights[active[1]], 0.5, atol=1e-10)


def test_c_enet_fallback_to_ew():
    """Very large alpha zeros all signals → fallback to equal weight."""
    signals, fwd_ret, names, _, _ = _make_signals(K=3)
    F = build_managed_returns(signals, fwd_ret)
    bw = c_enet_blend(F, alpha=1e6, l1_ratio=0.5)
    assert bw.diagnostics["fallback"] is True
    np.testing.assert_allclose(bw.weights.values, 1.0 / 3, atol=1e-10)


# ── Egalitarian ENet ─────────────────────────────────────────────────────────

def test_e_enet_selected_equal_weight():
    """Active signals after e_enet_blend should each receive weight 1/l."""
    signals, fwd_ret, _, _, _ = _make_signals()
    F = build_managed_returns(signals, fwd_ret)
    bw = e_enet_blend(F, alpha=0.01, l1_ratio=0.8)
    active = [k for k, w in bw.weights.items() if abs(w) > 1e-8]
    l = len(active)
    if l > 0:
        for name in active:
            np.testing.assert_allclose(bw.weights[name], 1.0 / l, atol=1e-10)


# ── Inverse-MSFE ─────────────────────────────────────────────────────────────

def test_inv_msfe_finite():
    """inv_msfe weights must be finite, sum to 1, and rank correctly."""
    rng = np.random.default_rng(33)
    T, K = 300, 3
    dates = pd.date_range("2018-01-01", periods=T, freq="B")
    F_mean = rng.standard_normal(T) * 0.01
    F_arr = np.column_stack([F_mean + rng.standard_normal(T) * e for e in [0.001, 0.01, 0.1]])
    F = pd.DataFrame(F_arr, index=dates, columns=["close", "mid", "far"])

    bw = inv_msfe_blend(F)
    assert np.all(np.isfinite(bw.weights.values))
    np.testing.assert_allclose(bw.weights.abs().sum(), 1.0, atol=1e-12)
    # Signal closest to ensemble mean (lowest MSFE) should get highest weight
    assert bw.weights["close"] > bw.weights["far"]


# ── PCR / PLS ────────────────────────────────────────────────────────────────

def test_pcr_finite():
    """PCR blend must return finite weights and report var_explained."""
    signals, fwd_ret, _, _, _ = _make_signals(K=4)
    F = build_managed_returns(signals, fwd_ret)
    bw = pcr_blend(F, n_components=2)
    assert np.all(np.isfinite(bw.weights.values))
    assert 0.0 <= bw.diagnostics["var_explained"] <= 1.0


def test_pls_finite():
    """PLS blend must return finite weights."""
    signals, fwd_ret, _, _, _ = _make_signals(K=4)
    F = build_managed_returns(signals, fwd_ret)
    bw = pls_blend(F, n_components=2)
    assert np.all(np.isfinite(bw.weights.values))


# ── Winsorize signals ────────────────────────────────────────────────────────

def test_winsorize_signals_clips():
    """After winsorization, no value should lie outside [q, 1-q] cross-sectional range."""
    signals, _, names, _, _ = _make_signals(K=3, T=100)
    q = 0.10
    wsig = winsorize_signals(signals, q=q)
    for name in names:
        orig = signals[name]
        clip = wsig[name]
        lo = orig.quantile(q, axis=1)
        hi = orig.quantile(1.0 - q, axis=1)
        # Clipped values must not exceed the original quantile bounds
        assert (clip.ge(lo, axis=0) | clip.isna()).all().all()
        assert (clip.le(hi, axis=0) | clip.isna()).all().all()


# ── Pool blend ───────────────────────────────────────────────────────────────

def test_pool_blend_mean():
    """Pooled mean weights must equal the per-signal average of input weights."""
    signals, fwd_ret, _, _, _ = _make_signals(K=4)
    F = build_managed_returns(signals, fwd_ret)
    bw1 = sharpe_heuristic(F, power=1)
    bw2 = msrr_ridge(F, lam=0.01)

    pool = pool_blend([bw1, bw2], method="mean")
    expected = (bw1.weights + bw2.weights) / 2
    expected = expected / expected.abs().sum()
    np.testing.assert_allclose(pool.weights.values, expected.values, atol=1e-10)


def test_pool_blend_methods():
    """Median and trimmed_mean pooling should produce finite weights."""
    signals, fwd_ret, _, _, _ = _make_signals(K=4)
    F = build_managed_returns(signals, fwd_ret)
    blend_list = [sharpe_heuristic(F, p) for p in [1, 2]]
    for method in ("median", "trimmed_mean"):
        pool = pool_blend(blend_list, method=method)
        assert np.all(np.isfinite(pool.weights.values))


# ── MLMSRR Huber objective ───────────────────────────────────────────────────

def test_mlmsrr_huber_runs():
    """Huber objective should train without error and produce finite weights."""
    signals, fwd_ret, _, _, _ = _make_signals(K=3, T=200)
    F = build_managed_returns(signals, fwd_ret)
    z = build_state_features(F, lookback=21)
    common = F.index.intersection(z.dropna().index)

    ml = MLMSRR(n_signals=3, hidden_size=8, objective="huber", n_epochs=10, seeds=[42])
    ml.fit(F.loc[common], z.loc[common])
    w = ml.predict_weights(z.loc[common]).dropna()
    assert np.all(np.isfinite(w.values))


# ── GBRT / RF blends ─────────────────────────────────────────────────────────

def test_gbrt_blend_finite():
    """GBRTBlend.predict_weights must return a finite T × K DataFrame."""
    signals, fwd_ret, _, _, _ = _make_signals(K=3, T=200)
    F = build_managed_returns(signals, fwd_ret)
    z = build_state_features(F, lookback=21)
    common = F.index.intersection(z.dropna().index)

    gb = GBRTBlend(n_estimators=20, max_depth=2)
    gb.fit(F.loc[common], z.loc[common])
    w = gb.predict_weights(z.loc[common])
    assert w.shape == (len(common), 3)
    assert np.all(np.isfinite(w.dropna().values))


def test_rf_blend_finite():
    """RandomForestBlend.predict_weights must return a finite T × K DataFrame."""
    signals, fwd_ret, _, _, _ = _make_signals(K=3, T=200)
    F = build_managed_returns(signals, fwd_ret)
    z = build_state_features(F, lookback=21)
    common = F.index.intersection(z.dropna().index)

    rf = RandomForestBlend(n_estimators=20, max_depth=3)
    rf.fit(F.loc[common], z.loc[common])
    w = rf.predict_weights(z.loc[common])
    assert w.shape == (len(common), 3)
    assert np.all(np.isfinite(w.dropna().values))


# ── LinearDynamicBlend ───────────────────────────────────────────────────────

def test_linear_dynamic_blend_shape():
    """predict_weights must return T × K DataFrame with finite values."""
    signals, fwd_ret, _, _, _ = _make_signals(K=3, T=200)
    F = build_managed_returns(signals, fwd_ret)
    z = build_state_features(F, lookback=21)
    common = F.index.intersection(z.dropna().index)

    ld = LinearDynamicBlend(ridge_lam=0.01)
    ld.fit(F.loc[common], z.loc[common])
    w = ld.predict_weights(z.loc[common])
    assert w.shape == (len(common), 3)
    assert np.all(np.isfinite(w.dropna().values))


def test_linear_dynamic_blend_no_lookahead():
    """Perturbing z after the midpoint must not change weights at earlier dates."""
    signals, fwd_ret, _, _, _ = _make_signals(K=3, T=200)
    F = build_managed_returns(signals, fwd_ret)
    z = build_state_features(F, lookback=21)
    common = F.index.intersection(z.dropna().index)

    ld = LinearDynamicBlend()
    ld.fit(F.loc[common], z.loc[common])
    w_orig = ld.predict_weights(z.loc[common])

    # Perturb z after midpoint
    mid = len(common) // 2
    z_perturbed = z.loc[common].copy()
    z_perturbed.iloc[mid:] *= 100.0
    w_perturbed = ld.predict_weights(z_perturbed)

    # Weights up to (mid-1) must be identical — fit() was not re-called
    pd.testing.assert_frame_equal(
        w_orig.iloc[:mid], w_perturbed.iloc[:mid], check_exact=False, atol=1e-10
    )


# ── DMABlend ─────────────────────────────────────────────────────────────────

def test_dma_blend_weights_sum_to_one():
    """DMA probability weights must sum to 1 on every date."""
    signals, fwd_ret, _, _, _ = _make_signals(K=4, T=200)
    F = build_managed_returns(signals, fwd_ret)

    dma = DMABlend(alpha=0.99)
    dma.fit(F)
    w = dma.predict_weights(F)
    row_sums = w.sum(axis=1)
    np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-8)


def test_dma_blend_forgetting():
    """
    Fast forgetting should abandon a previously dominant signal more quickly
    after a regime switch.

    sig0 dominates first half → sig1 dominates second half.
    By the end, fast forgetting (α=0.1) should hold less sig0 weight than slow (α=0.99).
    """
    rng = np.random.default_rng(55)
    T, K = 300, 2
    dates = pd.date_range("2018-01-01", periods=T, freq="B")
    F_arr = rng.standard_normal((T, K)) * 0.001
    F_arr[: T // 2, 0] += 0.05   # sig0 dominant in first half
    F_arr[T // 2:, 1] += 0.05    # sig1 dominant in second half
    F = pd.DataFrame(F_arr, index=dates, columns=["sig0", "sig1"])

    dma_fast = DMABlend(alpha=0.1)
    dma_fast.fit(F)

    dma_slow = DMABlend(alpha=0.99)
    dma_slow.fit(F)

    w_fast_end = dma_fast._weights_history["sig0"].iloc[-1]
    w_slow_end = dma_slow._weights_history["sig0"].iloc[-1]
    assert w_fast_end <= w_slow_end, (
        f"Fast forgetting sig0 weight {w_fast_end:.4f} should be ≤ slow {w_slow_end:.4f}"
    )


# ── STMOM ────────────────────────────────────────────────────────────────────

def test_stmom_output_range():
    """All STMOM positions must lie in (−1, 1)."""
    signals, fwd_ret, _, _, tickers = _make_signals(K=3, T=150, N=10)
    st = STMOM(n_signals=3, tau=1, hidden_size=0, n_epochs=5, seeds=[42])
    st.fit(signals, fwd_ret)
    pos = st.predict_positions(signals).dropna()
    assert pos.shape[1] == len(tickers)
    assert (pos.abs() <= 1.0 + 1e-6).all().all()


def test_stmom_output_shape():
    """STMOM positions shape must match (T, N)."""
    signals, fwd_ret, _, dates, tickers = _make_signals(K=3, T=150, N=10)
    st = STMOM(n_signals=3, tau=1, hidden_size=8, n_epochs=5, seeds=[42])
    st.fit(signals, fwd_ret)
    pos = st.predict_positions(signals)
    assert pos.shape == (len(dates), len(tickers))


def test_stmom_seed_reproducibility():
    """Same seed must produce identical positions."""
    signals, fwd_ret, _, _, _ = _make_signals(K=3, T=150, N=10)

    st1 = STMOM(n_signals=3, tau=1, hidden_size=0, n_epochs=5, seeds=[99])
    st1.fit(signals, fwd_ret)
    p1 = st1.predict_positions(signals).values

    st2 = STMOM(n_signals=3, tau=1, hidden_size=0, n_epochs=5, seeds=[99])
    st2.fit(signals, fwd_ret)
    p2 = st2.predict_positions(signals).values

    np.testing.assert_array_equal(p1, p2)


# ── fit_static extended ──────────────────────────────────────────────────────

def test_fit_static_extended():
    """Extended fit_static must include all new keys with finite weights."""
    signals, fwd_ret, _, _, _ = _make_signals(K=4)
    result = fit_static(signals, fwd_ret)
    new_keys = {"enet", "lasso", "c_enet", "e_enet", "e_lasso", "inv_msfe", "pcr", "pls"}
    for key in new_keys:
        assert key in result, f"Missing key: {key}"
        assert np.all(np.isfinite(result[key].weights.values)), f"Non-finite weights for {key}"
