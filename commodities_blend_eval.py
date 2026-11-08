#!/usr/bin/env python3
"""
Commodities blend evaluation: IS / OOS Sharpe, MDD, turnover for all blend methods.

Blend weights fitted on IS data only; OOS is a pure held-out test.

  IS  : start → 2020-12-31
  OOS : 2021-01-01 → present

Managed returns (daily) are vol-normalised to unit annual vol using IS σ.
Turnover is computed from actual blended per-commodity positions (book-scaled),
so it correctly captures cross-signal netting.

Usage:
    from commodities_blend_eval import try_comm_blends
    df = try_comm_blends()

    python commodities_blend_eval.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

IS_END    = "2020-12-31"
OOS_START = "2021-01-01"
TD        = 252


# ── helpers ───────────────────────────────────────────────────────────────────

def _sharpe(r: pd.Series) -> float:
    r = r[r.ne(0)].dropna()
    if len(r) < 60 or r.std() < 1e-12:
        return np.nan
    return float(r.mean() / r.std() * np.sqrt(TD))


def _mdd(r: pd.Series) -> float:
    cum = r.dropna().cumsum()
    if cum.empty:
        return np.nan
    return float((cum - cum.cummax()).min())


def _blend_ret(F: pd.DataFrame, w: pd.Series) -> pd.Series:
    w = w.reindex(F.columns).fillna(0.0)
    total = w.abs().sum()
    if total > 1e-12:
        w = w / total
    return (F * w.values).sum(axis=1)


def _dyn_blend_ret(F: pd.DataFrame, w_df: pd.DataFrame) -> pd.Series:
    idx = F.index.intersection(w_df.dropna().index)
    if not len(idx):
        return pd.Series(dtype=float)
    w_ = w_df.loc[idx]
    norms = w_.abs().sum(axis=1).replace(0.0, np.nan)
    return (F.loc[idx] * w_.div(norms, axis=0)).sum(axis=1)


def _combined_to(
    w_k: dict[str, pd.DataFrame],
    alpha: pd.Series | pd.DataFrame,
    names: list[str],
    is_end: str,
) -> tuple[float, float]:
    """
    Annual one-way turnover for a blend, split by IS / OOS.

    alpha: constant pd.Series (static) or time-varying pd.DataFrame T×K (dynamic).
    Returns (to_is, to_oos) in units of ann. notional turnover (same scale as run_comm_signals).
    """
    ts_cut   = pd.Timestamp(is_end)
    all_cols = w_k[names[0]].columns
    all_idx  = w_k[names[0]].index

    if isinstance(alpha, pd.Series):
        a = alpha.reindex(names).fillna(0.0)
        total = a.abs().sum()
        if total < 1e-12:
            return np.nan, np.nan
        a = a / total
        W = sum(float(a[k]) * w_k[k].reindex(index=all_idx, columns=all_cols).fillna(0.0)
                for k in names if k in w_k)
    else:
        # Dynamic: alpha_df is T × K, possibly sparser than all_idx
        a_df = (alpha.reindex(columns=names).fillna(0.0)
                     .reindex(all_idx).ffill().fillna(0.0))
        norms = a_df.abs().sum(axis=1).replace(0.0, np.nan)
        a_norm = a_df.div(norms, axis=0).fillna(0.0)
        W = pd.DataFrame(0.0, index=all_idx, columns=all_cols)
        for k in names:
            if k not in w_k:
                continue
            W += w_k[k].reindex(index=all_idx, columns=all_cols).fillna(0.0).mul(a_norm[k], axis=0)

    to_is  = float(W.loc[W.index <= ts_cut].diff().abs().sum(axis=1).mean() * TD)
    to_oos = float(W.loc[W.index >  ts_cut].diff().abs().sum(axis=1).mean() * TD)
    return to_is, to_oos


# ── rolling MSRR ─────────────────────────────────────────────────────────────

class _RollingMSRR:
    """
    Expanding-window MSRR: b_t = (M_t + λI)⁻¹ μ_t, refitted daily.

    M_t = F[:t].T @ F[:t] / t,  μ_t = mean(F[:t]).
    Weights at t use strictly past data (causal, no lookahead).

    lam_grid: values of λ to search over IS.  Pass [0.0] for OLS (no tuning).
    min_obs:  expanding-window warmup before weights are produced.
    """

    def __init__(self, lam_grid: list[float], min_obs: int = 252, penalty: str = "l2"):
        self.lam_grid = lam_grid
        self.min_obs  = min_obs
        self.penalty  = penalty   # "l2" (ridge / ols) or "l1" (lasso)
        self.lam_: float | None = None

    def _solve(self, Fv: np.ndarray, lam: float) -> np.ndarray:
        T, K = Fv.shape
        if self.penalty == "l1":
            from sklearn.linear_model import ElasticNet
            m = ElasticNet(alpha=lam, l1_ratio=1.0, fit_intercept=False, max_iter=10_000)
            m.fit(Fv, np.ones(T))
            b = m.coef_.copy()
            return b if np.abs(b).sum() > 1e-12 else np.full(K, 1.0 / K)
        mu = Fv.mean(axis=0)
        M  = Fv.T @ Fv / T
        b, *_ = np.linalg.lstsq(M + lam * np.eye(K), mu, rcond=None)
        return b

    def _expanding_weights(self, F: pd.DataFrame, lam: float) -> pd.DataFrame:
        """Expanding-window weight matrix (NaN during warmup)."""
        K  = F.shape[1]
        Fv = F.values.astype(float)
        rows = []
        for t in range(len(F)):
            if t < self.min_obs:
                rows.append([np.nan] * K)
            else:
                rows.append(self._solve(Fv[:t], lam).tolist())
        return pd.DataFrame(rows, index=F.index, columns=F.columns)

    def fit(self, F_is: pd.DataFrame) -> "_RollingMSRR":
        """Grid-search λ on IS via expanding-window IS Sharpe."""
        best_sr, best_lam = -np.inf, self.lam_grid[0]
        for lam in self.lam_grid:
            w_df = self._expanding_weights(F_is, lam)
            sr   = _sharpe(_dyn_blend_ret(F_is, w_df))
            if sr > best_sr:
                best_sr, best_lam = sr, lam
        self.lam_ = best_lam
        return self

    def weights_is(self, F_is: pd.DataFrame) -> pd.DataFrame:
        """Expanding weights over IS data."""
        return self._expanding_weights(F_is, self.lam_)

    def weights_oos(self, F_full: pd.DataFrame, oos_index: pd.Index) -> pd.DataFrame:
        """
        Expanding weights using full history (IS+OOS), sliced to oos_index.
        F_full must be the concatenation of IS and OOS managed returns.
        """
        return self._expanding_weights(F_full, self.lam_).reindex(oos_index)


# ── weight matrix builder ─────────────────────────────────────────────────────

def _build_weight_matrices(params: dict) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Run the full production pipeline for each signal and return book-scaled
    daily weight matrices (T × instruments) keyed by parquet signal name.
    """
    from commodities.data import fetch_prices, fetch_term_structure, fetch_tickers
    from commodities.backtest import (
        specific_contract_returns, _build_signals,
        vol_scale, rebalance, book_scale,
    )
    p = params
    prices  = fetch_prices()
    ts_data = fetch_term_structure()
    tk      = fetch_tickers()
    returns = specific_contract_returns(ts_data, tk).reindex(columns=prices.columns)
    raw     = _build_signals(prices, ts_data, tk, p)

    _name_map = {
        "tsmom_252":         "tsmom_252",
        "xs_carry_enh_f1f2": "xs_carry_enh",
        "curvature":         "curvature",
    }
    w_k = {}
    for raw_key, blend_key in _name_map.items():
        if raw_key not in raw:
            continue
        sig = raw[raw_key].reindex(index=returns.index, columns=returns.columns)
        w = vol_scale(sig, returns, target_vol=p["asset_vol"], com=p["vol_com"])
        w = rebalance(w, rule=p["rebal"])
        w = book_scale(w, returns, target_vol=p["book_vol"], cov_hl=p["cov_hl"], k_max=p["k_max"])
        w_k[blend_key] = w

    return w_k, returns


# ── main function ─────────────────────────────────────────────────────────────

def try_comm_blends(
    is_end: str = IS_END,
    params: dict | None = None,
) -> pd.DataFrame:
    """
    IS / OOS Sharpe, MDD, turnover for all blend methods on the 3 production signals.

    Blend weights fitted on IS only; OOS is a pure held-out test.

    Returns a DataFrame indexed by method with columns:
        type, IS_sharpe, IS_mdd, IS_to_pa, OOS_sharpe, OOS_mdd, OOS_to_pa
    """
    from commodities.backtest import PARAMS as _PARAMS
    from commodities.blend import load_managed_returns
    from equities.blend import (
        equal_weight, sharpe_heuristic, msrr_ridge,
        enet_blend, c_enet_blend, e_enet_blend,
        inv_msfe_blend, pcr_blend, pls_blend, pool_blend,
        build_state_features,
        DMABlend, LinearDynamicBlend, GBRTBlend, RandomForestBlend, MLMSRR,
    )

    p      = params or _PARAMS
    ts_cut = pd.Timestamp(is_end)
    oos_start = str((ts_cut + pd.Timedelta(days=1)).date())

    # ── Managed returns ───────────────────────────────────────────────────────
    print("Loading managed returns …")
    F     = load_managed_returns()
    F_is  = F[F.index <= ts_cut]
    F_oos = F[F.index >  ts_cut]
    names = list(F.columns)

    sigma   = F_is.std() * np.sqrt(TD)
    F_is_n  = F_is  / sigma
    F_oos_n = F_oos / sigma

    print(f"IS  : {len(F_is):,} days  ({F_is.index[0].date()} → {F_is.index[-1].date()})")
    print(f"OOS : {len(F_oos):,} days  ({F_oos.index[0].date()} → {F_oos.index[-1].date()})")
    print(f"Signals : {names}\n")

    # ── Weight matrices for turnover ──────────────────────────────────────────
    print("Building weight matrices …")
    w_k, _ = _build_weight_matrices(p)

    # ── Static blends (fit on IS) ─────────────────────────────────────────────
    print("Fitting static blends …")
    static_blends: dict = {
        "equal_weight": equal_weight(names),
        "sh_p1":        sharpe_heuristic(F_is_n, power=1),
        "sh_p2":        sharpe_heuristic(F_is_n, power=2),
        "msrr_ols":     msrr_ridge(F_is_n, lam=0.0),
        "msrr_ridge":   msrr_ridge(F_is_n, lam=0.01),
        "enet":         enet_blend(F_is_n),
        "lasso":        enet_blend(F_is_n, l1_ratio=1.0),
        "c_enet":       c_enet_blend(F_is_n),
        "e_enet":       e_enet_blend(F_is_n),
        "e_lasso":      e_enet_blend(F_is_n, l1_ratio=1.0),
        "inv_msfe":     inv_msfe_blend(F_is_n),
        "pcr":          pcr_blend(F_is_n),
        "pls":          pls_blend(F_is_n),
    }
    pool_blends: dict = {
        "pool_mean":    pool_blend(list(static_blends.values()), method="mean"),
        "pool_median":  pool_blend(list(static_blends.values()), method="median"),
        "pool_trimmed": pool_blend(list(static_blends.values()), method="trimmed_mean"),
    }

    # ── Rolling MSRR (expanding window, lam fixed after IS grid search) ────────
    F_full_n = pd.concat([F_is_n, F_oos_n])

    print("Fitting rolling MSRR (OLS, lam=0) …")
    roll_ols = _RollingMSRR(lam_grid=[0.0], min_obs=252)
    roll_ols.fit(F_is_n)
    roll_ols_w_is   = roll_ols.weights_is(F_is_n)
    roll_ols_w_oos  = roll_ols.weights_oos(F_full_n, F_oos_n.index)
    roll_ols_w_full = pd.concat([roll_ols_w_is, roll_ols_w_oos]).sort_index()

    print("Fitting rolling MSRR (ridge, IS grid search) …")
    roll_ridge = _RollingMSRR(
        lam_grid=[0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0], min_obs=252
    )
    roll_ridge.fit(F_is_n)
    print(f"  → selected lam = {roll_ridge.lam_}")
    roll_ridge_w_is   = roll_ridge.weights_is(F_is_n)
    roll_ridge_w_oos  = roll_ridge.weights_oos(F_full_n, F_oos_n.index)
    roll_ridge_w_full = pd.concat([roll_ridge_w_is, roll_ridge_w_oos]).sort_index()

    # ── State features for dynamic methods ────────────────────────────────────
    print("Building state features (lookback=252d) …")
    F_full = pd.concat([F_is_n, F_oos_n])
    z_full = build_state_features(F_full, lookback=252)
    z_is   = z_full.loc[F_is_n.index].dropna()
    z_oos  = z_full.loc[F_oos_n.index]

    # ── DMA ───────────────────────────────────────────────────────────────────
    print("Fitting DMA …")
    dma = DMABlend(alpha=0.99)
    dma.fit(F_is_n)
    dma_w_is   = dma._weights_history
    dma_w_oos  = dma.predict_weights(F_oos_n)
    dma_w_full = pd.concat([dma_w_is, dma_w_oos]).sort_index()

    # ── LinearDynamicBlend ────────────────────────────────────────────────────
    print("Fitting LinearDynamicBlend …")
    ldb = LinearDynamicBlend(ridge_lam=0.01)
    ldb.fit(F_is_n, z_is)
    ldb_w_is   = ldb.predict_weights(z_is)
    ldb_w_oos  = ldb.predict_weights(z_oos)
    ldb_w_full = pd.concat([ldb_w_is, ldb_w_oos]).sort_index()

    # ── GBRTBlend ─────────────────────────────────────────────────────────────
    print("Fitting GBRTBlend …")
    gbrt = GBRTBlend(n_estimators=100, max_depth=2, ridge_lam=0.01)
    gbrt.fit(F_is_n, z_is)
    gbrt_w_is   = gbrt.predict_weights(z_is)
    gbrt_w_oos  = gbrt.predict_weights(z_oos)
    gbrt_w_full = pd.concat([gbrt_w_is, gbrt_w_oos]).sort_index()

    # ── RandomForestBlend ─────────────────────────────────────────────────────
    print("Fitting RandomForestBlend …")
    rf = RandomForestBlend(n_estimators=100, max_depth=3, ridge_lam=0.01)
    rf.fit(F_is_n, z_is)
    rf_w_is   = rf.predict_weights(z_is)
    rf_w_oos  = rf.predict_weights(z_oos)
    rf_w_full = pd.concat([rf_w_is, rf_w_oos]).sort_index()

    # ── MLMSRR variants ───────────────────────────────────────────────────────
    mlmsrr_cfgs = {
        "mlmsrr_quad":   dict(objective="quadratic", output_activation="norm_abs"),
        "mlmsrr_sharpe": dict(objective="sharpe",    output_activation="norm_abs"),
        "mlmsrr_huber":  dict(objective="huber",     output_activation="norm_abs"),
    }
    mlmsrr_weights: dict[str, pd.DataFrame] = {}
    for label, cfg in mlmsrr_cfgs.items():
        print(f"Fitting {label} …")
        m = MLMSRR(
            n_signals=len(names),
            hidden_size=16,
            lam=1e-2,
            n_epochs=1000,
            patience=50,
            seeds=[42, 123],
            **cfg,
        )
        m.fit(F_is_n, z_is)
        mlmsrr_weights[label] = pd.concat([
            m.predict_weights(z_is),
            m.predict_weights(z_oos),
        ]).sort_index()

    # ── Collect results ───────────────────────────────────────────────────────
    rows = []

    def _add(method, type_, pnl_is, pnl_oos, alpha):
        is_dyn = isinstance(alpha, pd.DataFrame)
        to_is, to_oos = _combined_to(w_k, alpha, names, is_end)
        rows.append({
            "method":     method,
            "type":       type_,
            "IS_sharpe":  _sharpe(pnl_is),
            "IS_mdd":     _mdd(pnl_is),
            "IS_to_pa":   to_is,
            "OOS_sharpe": _sharpe(pnl_oos),
            "OOS_mdd":    _mdd(pnl_oos),
            "OOS_to_pa":  to_oos,
        })

    # Individual sleeves
    for sig in names:
        _add(sig, "sleeve", F_is_n[sig], F_oos_n[sig], pd.Series({sig: 1.0}))

    # Static blends
    for name, bw in static_blends.items():
        _add(name, "static",
             _blend_ret(F_is_n, bw.weights), _blend_ret(F_oos_n, bw.weights),
             bw.weights)

    # Pool blends
    for name, bw in pool_blends.items():
        _add(name, "pool",
             _blend_ret(F_is_n, bw.weights), _blend_ret(F_oos_n, bw.weights),
             bw.weights)

    # Rolling MSRR
    _add("rolling_msrr_ols",   "rolling",
         _dyn_blend_ret(F_is_n, roll_ols_w_is),   _dyn_blend_ret(F_oos_n, roll_ols_w_oos),
         roll_ols_w_full)
    _add("rolling_msrr_ridge", "rolling",
         _dyn_blend_ret(F_is_n, roll_ridge_w_is), _dyn_blend_ret(F_oos_n, roll_ridge_w_oos),
         roll_ridge_w_full)

    # Dynamic blends
    _add("dma", "dynamic",
         _dyn_blend_ret(F_is_n, dma_w_is), _dyn_blend_ret(F_oos_n, dma_w_oos),
         dma_w_full)
    _add("linear_dynamic", "dynamic",
         _dyn_blend_ret(F_is_n, ldb_w_is), _dyn_blend_ret(F_oos_n, ldb_w_oos),
         ldb_w_full)
    _add("gbrt", "dynamic",
         _dyn_blend_ret(F_is_n, gbrt_w_is), _dyn_blend_ret(F_oos_n, gbrt_w_oos),
         gbrt_w_full)
    _add("rf", "dynamic",
         _dyn_blend_ret(F_is_n, rf_w_is), _dyn_blend_ret(F_oos_n, rf_w_oos),
         rf_w_full)
    for label, w_full in mlmsrr_weights.items():
        _add(label, "dynamic",
             _dyn_blend_ret(F_is_n, w_full.loc[w_full.index <= ts_cut]),
             _dyn_blend_ret(F_oos_n, w_full.loc[w_full.index >  ts_cut]),
             w_full)

    # ── Print table ───────────────────────────────────────────────────────────
    W = 90
    sep = "  " + "─" * (W - 2)

    def _fs(v): return f"{v:+.2f}" if v == v else "   n/a"
    def _fm(v): return f"{v*100:+.1f}%" if v == v else "    n/a"
    def _ft(v): return f"{v:.1f}x"  if v == v else "   n/a"

    print()
    print("=" * W)
    print("  COMMODITIES BLEND EVALUATION")
    print(f"  IS ≤ {is_end}   |   OOS ≥ {oos_start}")
    print(f"  Daily managed returns, vol-normalised (IS σ)")
    print("=" * W)
    print(f"  {'Method':<20}  {'Type':<8}  "
          f"{'IS':>26}   {'OOS':>26}")
    print(f"  {'':20}  {'':8}  "
          f"{'Sharpe':>7}  {'MDD':>8}  {'TO':>7}   "
          f"{'Sharpe':>7}  {'MDD':>8}  {'TO':>7}")
    print(sep)

    prev_type = None
    for r in rows:
        if prev_type is not None and r["type"] != prev_type:
            print(sep)
        prev_type = r["type"]
        print(
            f"  {r['method']:<20}  {r['type']:<8}  "
            f"{_fs(r['IS_sharpe']):>7}  {_fm(r['IS_mdd']):>8}  {_ft(r['IS_to_pa']):>7}   "
            f"{_fs(r['OOS_sharpe']):>7}  {_fm(r['OOS_mdd']):>8}  {_ft(r['OOS_to_pa']):>7}"
        )
    print("=" * W)
    print("  Sharpe gross (vol-norm)  |  MDD in vol-norm units  |  TO annual one-way (blended positions)\n")

    # ── Blend weights table (static methods) ─────────────────────────────────
    col_w = 14
    all_static = {**static_blends, **pool_blends}
    print("  BLEND WEIGHTS (IS-fitted, static + pool)")
    print("  " + "─" * (22 + col_w * len(names)))
    print(f"  {'Method':<20}" + "".join(f"  {s:>{col_w}}" for s in names))
    print("  " + "─" * (22 + col_w * len(names)))
    for name, bw in all_static.items():
        w_str = "".join(f"  {bw.weights.get(s, 0.0):>+{col_w}.4f}" for s in names)
        print(f"  {name:<20}{w_str}")
    print("  " + "─" * (22 + col_w * len(names)))

    # ── Rolling MSRR average weights ──────────────────────────────────────────
    print(f"\n  ROLLING MSRR — selected λ and average blend weights")
    print("  " + "─" * (22 + col_w * len(names)))
    print(f"  {'Method':<20}" + "".join(f"  {s:>{col_w}}" for s in names))
    print("  " + "─" * (22 + col_w * len(names)))
    for label, lam_val, w_is, w_oos in [
        ("rolling_msrr_ols",   roll_ols.lam_,   roll_ols_w_is,   roll_ols_w_oos),
        ("rolling_msrr_ridge", roll_ridge.lam_,  roll_ridge_w_is, roll_ridge_w_oos),
    ]:
        for period, w_df in [("IS avg", w_is), ("OOS avg", w_oos)]:
            avg = w_df.dropna().mean()
            norm = avg.abs().sum()
            avg_n = (avg / norm) if norm > 1e-12 else avg
            w_str = "".join(f"  {avg_n.get(s, 0.0):>+{col_w}.4f}" for s in names)
            tag = f"{label} {period} (λ={lam_val})"
            print(f"  {tag:<20}{w_str}")
    print("  " + "─" * (22 + col_w * len(names)))

    return pd.DataFrame(rows).set_index("method")


_LAM_GRID_L2 = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
_LAM_GRID_L1 = [1e-5, 1e-4, 0.001, 0.005, 0.01, 0.05, 0.1]


def compare_msrr(
    is_end: str = IS_END,
    params: dict | None = None,
) -> pd.DataFrame:
    """
    Focused 6-way table: OLS / Lasso / Ridge  ×  static / rolling.

    Static  : λ chosen by IS Sharpe grid search, fit once on full IS.
    Rolling : λ chosen by IS expanding-window Sharpe, weights refit daily.

    Returns DataFrame with IS/OOS Sharpe, MDD, turnover, and selected λ.
    """
    from commodities.backtest import PARAMS as _PARAMS
    from commodities.blend import load_managed_returns
    from equities.blend import msrr_ridge as _msrr_ridge, msrr_lasso as _msrr_lasso

    p      = params or _PARAMS
    ts_cut = pd.Timestamp(is_end)
    oos_start = str((ts_cut + pd.Timedelta(days=1)).date())

    print("Loading managed returns …")
    F     = load_managed_returns()
    F_is  = F[F.index <= ts_cut]
    F_oos = F[F.index >  ts_cut]
    sigma = F_is.std() * np.sqrt(TD)
    F_is_n, F_oos_n = F_is / sigma, F_oos / sigma
    F_full_n = pd.concat([F_is_n, F_oos_n])
    names = list(F.columns)

    print(f"IS  : {len(F_is):,} days  ({F_is.index[0].date()} → {F_is.index[-1].date()})")
    print(f"OOS : {len(F_oos):,} days  ({F_oos.index[0].date()} → {F_oos.index[-1].date()})\n")

    print("Building weight matrices …")
    w_k, _ = _build_weight_matrices(p)

    rows = []

    # ── Static ────────────────────────────────────────────────────────────────
    # For each penalty: pick best lam by IS Sharpe, fit once on full IS.

    # OLS (no tuning)
    bw_ols = _msrr_ridge(F_is_n, lam=0.0)
    to_is, to_oos = _combined_to(w_k, bw_ols.weights, names, is_end)
    rows.append({
        "method": "static_ols",   "penalty": "OLS",   "formulation": "static",
        "IS_sharpe":  _sharpe(_blend_ret(F_is_n, bw_ols.weights)),
        "IS_mdd":     _mdd(_blend_ret(F_is_n, bw_ols.weights)),
        "IS_to_pa":   to_is,
        "OOS_sharpe": _sharpe(_blend_ret(F_oos_n, bw_ols.weights)),
        "OOS_mdd":    _mdd(_blend_ret(F_oos_n, bw_ols.weights)),
        "OOS_to_pa":  to_oos,
        "lam": 0.0,
    })

    # Lasso (IS grid search)
    best_sr, best_lam = -np.inf, _LAM_GRID_L1[0]
    for lam in _LAM_GRID_L1:
        bw = _msrr_lasso(F_is_n, lam)
        sr = _sharpe(_blend_ret(F_is_n, bw.weights))
        if sr > best_sr:
            best_sr, best_lam = sr, lam
    bw_lasso = _msrr_lasso(F_is_n, best_lam)
    print(f"static_lasso  → IS best lam = {best_lam}")
    to_is, to_oos = _combined_to(w_k, bw_lasso.weights, names, is_end)
    rows.append({
        "method": "static_lasso", "penalty": "LASSO", "formulation": "static",
        "IS_sharpe":  _sharpe(_blend_ret(F_is_n, bw_lasso.weights)),
        "IS_mdd":     _mdd(_blend_ret(F_is_n, bw_lasso.weights)),
        "IS_to_pa":   to_is,
        "OOS_sharpe": _sharpe(_blend_ret(F_oos_n, bw_lasso.weights)),
        "OOS_mdd":    _mdd(_blend_ret(F_oos_n, bw_lasso.weights)),
        "OOS_to_pa":  to_oos,
        "lam": best_lam,
    })

    # Ridge (IS grid search)
    best_sr, best_lam = -np.inf, _LAM_GRID_L2[0]
    for lam in _LAM_GRID_L2:
        bw = _msrr_ridge(F_is_n, lam)
        sr = _sharpe(_blend_ret(F_is_n, bw.weights))
        if sr > best_sr:
            best_sr, best_lam = sr, lam
    bw_ridge = _msrr_ridge(F_is_n, best_lam)
    print(f"static_ridge  → IS best lam = {best_lam}")
    to_is, to_oos = _combined_to(w_k, bw_ridge.weights, names, is_end)
    rows.append({
        "method": "static_ridge", "penalty": "Ridge", "formulation": "static",
        "IS_sharpe":  _sharpe(_blend_ret(F_is_n, bw_ridge.weights)),
        "IS_mdd":     _mdd(_blend_ret(F_is_n, bw_ridge.weights)),
        "IS_to_pa":   to_is,
        "OOS_sharpe": _sharpe(_blend_ret(F_oos_n, bw_ridge.weights)),
        "OOS_mdd":    _mdd(_blend_ret(F_oos_n, bw_ridge.weights)),
        "OOS_to_pa":  to_oos,
        "lam": best_lam,
    })

    # ── Rolling (expanding window) ────────────────────────────────────────────
    rolling_w_is: dict[str, pd.DataFrame] = {}   # saved for weights table
    for label, pen_label, pen_code, grid in [
        ("rolling_ols",   "OLS",   "l2", [0.0]),
        ("rolling_lasso", "LASSO", "l1", _LAM_GRID_L1),
        ("rolling_ridge", "Ridge", "l2", _LAM_GRID_L2),
    ]:
        print(f"Fitting {label} …")
        roll = _RollingMSRR(lam_grid=grid, min_obs=252, penalty=pen_code)
        roll.fit(F_is_n)
        print(f"  → selected lam = {roll.lam_}")
        w_is   = roll.weights_is(F_is_n)
        w_oos  = roll.weights_oos(F_full_n, F_oos_n.index)
        w_full = pd.concat([w_is, w_oos]).sort_index()
        rolling_w_is[label] = w_is
        to_is, to_oos = _combined_to(w_k, w_full, names, is_end)
        rows.append({
            "method": label, "penalty": pen_label, "formulation": "rolling",
            "IS_sharpe":  _sharpe(_dyn_blend_ret(F_is_n, w_is)),
            "IS_mdd":     _mdd(_dyn_blend_ret(F_is_n, w_is)),
            "IS_to_pa":   to_is,
            "OOS_sharpe": _sharpe(_dyn_blend_ret(F_oos_n, w_oos)),
            "OOS_mdd":    _mdd(_dyn_blend_ret(F_oos_n, w_oos)),
            "OOS_to_pa":  to_oos,
            "lam":        roll.lam_,
        })

    # ── Print ─────────────────────────────────────────────────────────────────
    W = 100
    sep = "  " + "─" * (W - 2)

    def _fs(v): return f"{v:+.2f}" if v == v else "   n/a"
    def _fm(v): return f"{v*100:+.1f}%" if v == v else "    n/a"
    def _ft(v): return f"{v:.1f}x"  if v == v else "   n/a"

    print()
    print("=" * W)
    print("  MSRR COMPARISON  —  Static vs Rolling × OLS / Lasso / Ridge")
    print(f"  IS ≤ {is_end}   |   OOS ≥ {oos_start}   |   λ tuned on IS Sharpe")
    print("=" * W)
    print(f"  {'Method':<20}  {'Penalty':<6}  {'Form':<8}  "
          f"{'IS':>26}   {'OOS':>26}   {'λ':>9}")
    print(f"  {'':20}  {'':6}  {'':8}  "
          f"{'SR':>7}  {'MDD':>8}  {'TO':>7}   "
          f"{'SR':>7}  {'MDD':>8}  {'TO':>7}   {'':>9}")
    print(sep)

    prev_form = None
    for r in rows:
        if prev_form is not None and r["formulation"] != prev_form:
            print(sep)
        prev_form = r["formulation"]
        print(
            f"  {r['method']:<20}  {r['penalty']:<6}  {r['formulation']:<8}  "
            f"{_fs(r['IS_sharpe']):>7}  {_fm(r['IS_mdd']):>8}  {_ft(r['IS_to_pa']):>7}   "
            f"{_fs(r['OOS_sharpe']):>7}  {_fm(r['OOS_mdd']):>8}  {_ft(r['OOS_to_pa']):>7}   "
            f"{r['lam']:>9.5f}"
        )
    print("=" * W)

    # ── Weights table ─────────────────────────────────────────────────────────
    col_w = 14
    static_bws = {"static_ols": bw_ols, "static_lasso": bw_lasso, "static_ridge": bw_ridge}
    print(f"\n  BLEND WEIGHTS  (rolling = IS average, normalised)")
    print("  " + "─" * (38 + col_w * len(names)))
    print(f"  {'Method':<20}  {'Form':<8}  {'λ':>5}" + "".join(f"  {s:>{col_w}}" for s in names))
    print("  " + "─" * (38 + col_w * len(names)))
    for r in rows:
        m = r["method"]
        if r["formulation"] == "static":
            w = static_bws[m].weights
            w_norm = w / w.abs().sum() if w.abs().sum() > 1e-12 else w
        else:
            avg = rolling_w_is[m].dropna().mean()
            n   = avg.abs().sum()
            w_norm = avg / n if n > 1e-12 else avg
        w_str = "".join(f"  {w_norm.get(s, 0.0):>+{col_w}.4f}" for s in names)
        print(f"  {m:<20}  {r['formulation']:<8}  {r['lam']:>5.4f}{w_str}")
    print("  " + "─" * (38 + col_w * len(names)))

    return pd.DataFrame(rows).set_index("method")


def main():
    try_comm_blends()


if __name__ == "__main__":
    main()
