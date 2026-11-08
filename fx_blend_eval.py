#!/usr/bin/env python3
"""
FX blend evaluation: IS / OOS Sharpe, MDD, and turnover for all blend methods.

Blend weights are fitted on IS data only; OOS is a pure held-out test.

  IS  : start → 2020-12-31
  OOS : 2021-01-01 → present

Managed returns (monthly) are built from three production signals:
  momentum   Momentum XS f=9m h=1           (monthly weights → rx_m)
  mr_ns_loo  MR NS-LOO monthly VF-TS         (daily positions → rx_d → monthly sum)
  carry_xs   Carry XS (view XS monthly)      (monthly weights → rx_m)

Before fitting, each sleeve is vol-normalised to unit annual vol using IS σ
(no look-ahead). Blend weights therefore allocate risk budget.

Turnover is computed from actual blended monthly positions (vol-scaled to match
the managed-return convention), so it correctly accounts for cross-signal netting.

ML/dynamic methods (LinearDynamicBlend, GBRT, RF, MLMSRR) use state features z_t
built from lagged rolling Sharpe / dispersion of the managed returns (lookback=12m).
pool_blend aggregates the 13 static blend weights (mean / median / trimmed_mean).
STMOM maps per-pair signal features directly to positions; fitted on IS monthly
positions + IS forward returns (rx_m); PnL vol-normalised to IS σ for comparability.

Usage:
    python fx_blend_eval.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

IS_END      = "2020-12-31"
OOS_START   = "2021-01-01"
SIGNALS_DIR = Path("fx/signals")


# ── helpers ───────────────────────────────────────────────────────────────────

def annualized_sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 12 or r.std() < 1e-12:
        return np.nan
    return float(r.mean() / r.std() * np.sqrt(12))


def mdd(pnl: pd.Series) -> float:
    cum = pnl.dropna().cumsum()
    if cum.empty:
        return np.nan
    return float((cum - cum.cummax()).min())


def blend_return(F: pd.DataFrame, weights) -> pd.Series:
    w = weights.weights.reindex(F.columns).fillna(0.0)
    total = w.abs().sum()
    if total > 1e-12:
        w = w / total
    return (F * w.values).sum(axis=1)


def vol_normalise(
    F_is: pd.DataFrame, F_oos: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    sigma = F_is.std() * np.sqrt(12)
    sigma = sigma.replace(0.0, np.nan)
    return F_is.div(sigma), F_oos.div(sigma), sigma


def _dynamic_return(F: pd.DataFrame, w_df: pd.DataFrame) -> pd.Series:
    """Apply time-varying blend weights to managed returns."""
    idx = F.index.intersection(w_df.dropna().index)
    w_ = w_df.loc[idx]
    row_norm = w_.abs().sum(axis=1).replace(0.0, np.nan)
    w_ = w_.div(row_norm, axis=0)
    return (F.loc[idx] * w_).sum(axis=1)


def _dynamic_turnover(
    w_df: pd.DataFrame,
    monthly_pos: dict,
    vol_scales: pd.Series,
    common_idx: list,
    common_pairs: list,
    is_end: str,
    period: str,
    names: list,
) -> float:
    """
    Annual one-way turnover for a dynamic blend (time-varying w_df).
    Builds actual blended monthly positions (vol-scaled) and diffs them.
    """
    ts = pd.Timestamp(is_end)
    idx = [t for t in common_idx
           if (t <= ts if period == "is" else t > ts)
           and t in w_df.index
           and not w_df.loc[t].isna().any()]
    if not idx:
        return np.nan

    w_period = w_df.loc[idx].copy()
    norms = w_period.abs().sum(axis=1).replace(0.0, np.nan)
    w_period = w_period.div(norms, axis=0)

    blend = pd.DataFrame(0.0, index=idx, columns=common_pairs)
    for k in names:
        if k not in monthly_pos or k not in vol_scales.index:
            continue
        pos_k = monthly_pos[k].reindex(index=idx, columns=common_pairs).fillna(0.0)
        blend = blend + pos_k.mul(w_period[k] / vol_scales[k], axis=0)

    return float(blend.diff().abs().sum(axis=1).mean() * 12)


# ── managed returns ───────────────────────────────────────────────────────────

def build_fx_managed_returns(signals_dir: Path = SIGNALS_DIR) -> pd.DataFrame:
    sdir = signals_dir
    rx_m = pd.read_parquet(sdir / "returns.parquet")["rx_m"]
    rx_d = pd.read_parquet(sdir / "rx_d.parquet")

    w_mom = pd.read_parquet(sdir / "signal_a_f9_h1.parquet")
    pnl_mom = (w_mom.shift(1) * rx_m.reindex(columns=w_mom.columns)).sum(axis=1)

    pos_mr = pd.read_parquet(sdir / "mr_ns_loo_positions.parquet")
    pnl_mr = (
        (pos_mr * rx_d.reindex(columns=pos_mr.columns)).sum(axis=1)
        .resample("BME").sum()
    )

    w_cxs = pd.read_parquet(sdir / "view_xs_carry_m.parquet")
    pnl_carry_xs = (w_cxs.shift(1) * rx_m.reindex(columns=w_cxs.columns)).sum(axis=1)

    F = pd.DataFrame({
        "momentum":  pnl_mom,
        "mr_ns_loo": pnl_mr,
        "carry_xs":  pnl_carry_xs,
    }).dropna()
    F.index.name = "date"
    return F


# ── monthly positions (for turnover) ─────────────────────────────────────────

def _load_monthly_positions(sdir: Path) -> dict[str, pd.DataFrame]:
    return {
        "momentum":  pd.read_parquet(sdir / "signal_a_f9_h1.parquet"),
        "mr_ns_loo": (pd.read_parquet(sdir / "mr_ns_loo_positions.parquet")
                      .resample("BME").last()),
        "carry_xs":  pd.read_parquet(sdir / "view_xs_carry_m.parquet"),
    }


def _blended_positions(
    blend_weights,
    monthly_pos: dict,
    vol_scales: pd.Series,
    common_idx: list,
    common_pairs: list,
) -> pd.DataFrame:
    combo = pd.DataFrame(0.0, index=common_idx, columns=common_pairs)
    for k, w_k in blend_weights.weights.items():
        if k not in monthly_pos or k not in vol_scales.index:
            continue
        pos_k = monthly_pos[k].reindex(index=common_idx, columns=common_pairs).fillna(0.0)
        combo = combo + (w_k / vol_scales[k]) * pos_k
    return combo


def _pos_turnover(pos: pd.DataFrame, is_end: str, period: str) -> float:
    p = pos[pos.index <= is_end] if period == "is" else pos[pos.index > is_end]
    return float(p.diff().abs().sum(axis=1).mean() * 12)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Building monthly managed returns …")
    F = build_fx_managed_returns()

    F_is  = F[F.index <= IS_END]
    F_oos = F[F.index >  IS_END]

    print(f"IS  : {len(F_is):>3} months  ({F_is.index[0].date()} → {F_is.index[-1].date()})")
    print(f"OOS : {len(F_oos):>3} months  ({F_oos.index[0].date()} → {F_oos.index[-1].date()})")
    print(f"Signals : {list(F.columns)}\n")

    F_is_n, F_oos_n, vol_scales = vol_normalise(F_is, F_oos)

    monthly_pos  = _load_monthly_positions(SIGNALS_DIR)
    common_pairs = sorted(set.intersection(*[set(p.columns) for p in monthly_pos.values()]))
    common_idx   = sorted(set.intersection(*[set(p.index)   for p in monthly_pos.values()]))

    from equities.blend import (
        equal_weight, sharpe_heuristic, msrr_ridge,
        enet_blend, c_enet_blend, e_enet_blend,
        inv_msfe_blend, pcr_blend, pls_blend,
        pool_blend,
        build_state_features,
        DMABlend, LinearDynamicBlend,
        GBRTBlend, RandomForestBlend, MLMSRR, STMOM,
    )

    names = list(F_is_n.columns)

    # ── static blends ─────────────────────────────────────────────────────────
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

    # ── pool blends: ensemble of all static methods ───────────────────────────
    pool_blends: dict = {
        "pool_mean":    pool_blend(list(static_blends.values()), method="mean"),
        "pool_median":  pool_blend(list(static_blends.values()), method="median"),
        "pool_trimmed": pool_blend(list(static_blends.values()), method="trimmed_mean"),
    }

    # ── state features for ML/dynamic methods (lookback = 12 months) ──────────
    print("Building state features (lookback=12m) …")
    F_full = pd.concat([F_is_n, F_oos_n])
    z_full = build_state_features(F_full, lookback=12)
    z_is   = z_full.loc[F_is_n.index].dropna()
    z_oos  = z_full.loc[F_oos_n.index]

    # ── DMA ───────────────────────────────────────────────────────────────────
    print("Fitting DMA …")
    dma = DMABlend(alpha=0.99)
    dma.fit(F_is_n)
    dma_w_is  = dma._weights_history
    dma_w_oos = dma.predict_weights(F_oos_n)

    # ── LinearDynamicBlend ────────────────────────────────────────────────────
    print("Fitting LinearDynamicBlend …")
    ldb = LinearDynamicBlend(ridge_lam=0.01)
    ldb.fit(F_is_n, z_is)
    ldb_w_is  = ldb.predict_weights(z_is)
    ldb_w_oos = ldb.predict_weights(z_oos)

    # ── GBRTBlend ─────────────────────────────────────────────────────────────
    print("Fitting GBRTBlend …")
    gbrt = GBRTBlend(n_estimators=100, max_depth=2, ridge_lam=0.01)
    gbrt.fit(F_is_n, z_is)
    gbrt_w_is  = gbrt.predict_weights(z_is)
    gbrt_w_oos = gbrt.predict_weights(z_oos)

    # ── RandomForestBlend ─────────────────────────────────────────────────────
    print("Fitting RandomForestBlend …")
    rf = RandomForestBlend(n_estimators=100, max_depth=3, ridge_lam=0.01)
    rf.fit(F_is_n, z_is)
    rf_w_is  = rf.predict_weights(z_is)
    rf_w_oos = rf.predict_weights(z_oos)

    # ── MLMSRR variants ───────────────────────────────────────────────────────
    mlmsrr_cfgs = {
        "mlmsrr_quad":   dict(objective="quadratic", output_activation="norm_abs"),
        "mlmsrr_sharpe": dict(objective="sharpe",    output_activation="norm_abs"),
        "mlmsrr_huber":  dict(objective="huber",     output_activation="norm_abs"),
    }
    mlmsrr_weights: dict[str, tuple] = {}
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
        mlmsrr_weights[label] = (m.predict_weights(z_is), m.predict_weights(z_oos))

    # ── STMOM ─────────────────────────────────────────────────────────────────
    print("Fitting STMOM …")
    rx_m_full = pd.read_parquet(SIGNALS_DIR / "returns.parquet")["rx_m"]
    rx_m_is   = rx_m_full.loc[rx_m_full.index <= IS_END]
    rx_m_oos  = rx_m_full.loc[rx_m_full.index >  IS_END]

    # common tickers across all monthly signals and rx_m
    _stmom_tickers = sorted(
        set.intersection(*[set(v.columns) for v in monthly_pos.values()])
        & set(rx_m_full.columns)
    )

    signals_is  = {k: v.loc[v.index <= IS_END, _stmom_tickers] for k, v in monthly_pos.items()}
    signals_oos = {k: v.loc[v.index >  IS_END, _stmom_tickers] for k, v in monthly_pos.items()}

    # forward_returns: position at t earns rx_m at t+1 → shift(-1)
    fwd_ret_is = rx_m_is.reindex(columns=_stmom_tickers).shift(-1)

    stmom = STMOM(n_signals=len(names), tau=1, hidden_size=0, seeds=[42, 123])
    stmom.fit(signals_is, fwd_ret_is)

    stmom_pos_is  = stmom.predict_positions(signals_is)
    stmom_pos_oos = stmom.predict_positions(signals_oos)

    # PnL: pos.shift(1) * rx_m (positions formed at t-1 earn rx_m at t)
    _srx = rx_m_full.reindex(columns=_stmom_tickers).fillna(0.0)
    stmom_pnl_is  = (stmom_pos_is.shift(1)  * _srx).sum(axis=1)
    stmom_pnl_oos = (stmom_pos_oos.shift(1) * _srx).sum(axis=1)
    stmom_pnl_is  = stmom_pnl_is.loc[stmom_pnl_is.index   <= IS_END].dropna()
    stmom_pnl_oos = stmom_pnl_oos.loc[stmom_pnl_oos.index >  IS_END].dropna()

    # vol-normalise using IS σ so MDD is on the same scale as other methods
    _stmom_sigma = stmom_pnl_is.std() * np.sqrt(12)
    if _stmom_sigma > 1e-12:
        stmom_pnl_is  = stmom_pnl_is  / _stmom_sigma
        stmom_pnl_oos = stmom_pnl_oos / _stmom_sigma

    # turnover: raw monthly position turnover (sum |Δpos| per pair per month × 12)
    _spos_is  = stmom_pos_is.loc[stmom_pos_is.index   <= IS_END]
    _spos_oos = stmom_pos_oos.loc[stmom_pos_oos.index >  IS_END]
    stmom_to_is  = float(_spos_is.diff().abs().sum(axis=1).mean()  * 12)
    stmom_to_oos = float(_spos_oos.diff().abs().sum(axis=1).mean() * 12)

    # ── collect results ───────────────────────────────────────────────────────
    rows = []

    def _row(method, type_, pnl_is, pnl_oos, w_is, w_oos, is_static=True):
        if is_static:
            pos_b = _blended_positions(w_is, monthly_pos, vol_scales, common_idx, common_pairs)
            to_is  = _pos_turnover(pos_b, IS_END, "is")
            to_oos = _pos_turnover(pos_b, IS_END, "oos")
        else:
            to_is  = _dynamic_turnover(w_is,  monthly_pos, vol_scales,
                                       common_idx, common_pairs, IS_END, "is",  names)
            to_oos = _dynamic_turnover(w_oos, monthly_pos, vol_scales,
                                       common_idx, common_pairs, IS_END, "oos", names)
        return {
            "method": method, "type": type_,
            "IS_Sharpe":  annualized_sharpe(pnl_is),
            "IS_MDD":     mdd(pnl_is),
            "IS_TO":      to_is,
            "OOS_Sharpe": annualized_sharpe(pnl_oos),
            "OOS_MDD":    mdd(pnl_oos),
            "OOS_TO":     to_oos,
        }

    # sleeves (use a single-signal "static" blend weight for position blending)
    for sig in names:
        from equities.blend import BlendWeights
        bw = BlendWeights(pd.Series({sig: 1.0}), sig)
        rows.append(_row(sig, "sleeve",
                         F_is_n[sig], F_oos_n[sig], bw, None, is_static=True))

    # static blends
    for name, bw in static_blends.items():
        rows.append(_row(name, "static",
                         blend_return(F_is_n, bw), blend_return(F_oos_n, bw),
                         bw, None, is_static=True))

    # pool blends (ensemble of all static weights)
    for name, bw in pool_blends.items():
        rows.append(_row(name, "pool",
                         blend_return(F_is_n, bw), blend_return(F_oos_n, bw),
                         bw, None, is_static=True))

    # dynamic blends
    rows.append(_row("dma", "dynamic",
                     _dynamic_return(F_is_n, dma_w_is),
                     _dynamic_return(F_oos_n, dma_w_oos),
                     dma_w_is, dma_w_oos, is_static=False))

    rows.append(_row("linear_dynamic", "dynamic",
                     _dynamic_return(F_is_n, ldb_w_is),
                     _dynamic_return(F_oos_n, ldb_w_oos),
                     ldb_w_is, ldb_w_oos, is_static=False))

    rows.append(_row("gbrt", "dynamic",
                     _dynamic_return(F_is_n, gbrt_w_is),
                     _dynamic_return(F_oos_n, gbrt_w_oos),
                     gbrt_w_is, gbrt_w_oos, is_static=False))

    rows.append(_row("rf", "dynamic",
                     _dynamic_return(F_is_n, rf_w_is),
                     _dynamic_return(F_oos_n, rf_w_oos),
                     rf_w_is, rf_w_oos, is_static=False))

    for label, (w_is, w_oos) in mlmsrr_weights.items():
        rows.append(_row(label, "dynamic",
                         _dynamic_return(F_is_n, w_is),
                         _dynamic_return(F_oos_n, w_oos),
                         w_is, w_oos, is_static=False))

    # STMOM (direct position model, PnL already vol-normalised above)
    rows.append({
        "method": "stmom", "type": "dynamic",
        "IS_Sharpe":  annualized_sharpe(stmom_pnl_is),
        "IS_MDD":     mdd(stmom_pnl_is),
        "IS_TO":      stmom_to_is,
        "OOS_Sharpe": annualized_sharpe(stmom_pnl_oos),
        "OOS_MDD":    mdd(stmom_pnl_oos),
        "OOS_TO":     stmom_to_oos,
    })

    # ── print table ───────────────────────────────────────────────────────────
    W = 86
    hdr = (f"{'Signal/Method':<20}  {'Type':<8}  "
           f"{'IS':>26}   {'OOS':>26}")
    sub = (f"{'':20}  {'':8}  "
           f"{'Sharpe':>7}  {'MDD':>8}  {'TO':>7}   "
           f"{'Sharpe':>7}  {'MDD':>8}  {'TO':>7}")
    sep = "─" * W

    def _fs(v): return f"{v:+.2f}" if v == v else "  n/a"
    def _fm(v): return f"{v*100:+.1f}%" if v == v else "    n/a"
    def _ft(v): return f"{v:.1f}x"  if v == v else "  n/a"

    print()
    print("=" * W)
    print("  FX BLEND EVALUATION")
    print(f"  IS ≤ {IS_END}   |   OOS ≥ {OOS_START}")
    print(f"  Monthly managed returns, vol-normalised (IS σ)")
    print("=" * W)
    print(hdr)
    print(sub)
    print(sep)

    prev_type = None
    for r in rows:
        if prev_type is not None and r["type"] != prev_type:
            print(sep)
        prev_type = r["type"]
        print(
            f"  {r['method']:<18}  {r['type']:<8}  "
            f"{_fs(r['IS_Sharpe']):>7}  {_fm(r['IS_MDD']):>8}  {_ft(r['IS_TO']):>7}   "
            f"{_fs(r['OOS_Sharpe']):>7}  {_fm(r['OOS_MDD']):>8}  {_ft(r['OOS_TO']):>7}"
        )

    print("=" * W)
    print("  MDD in vol-normalised return units  |  TO = annual one-way (blended positions)\n")

    # ── weights table (static only) ───────────────────────────────────────────
    col_w = 14
    print("  BLEND WEIGHTS (IS-fitted, static methods)")
    print("  " + "─" * (22 + col_w * len(names)))
    print(f"  {'Method':<20}" + "".join(f"  {s:>{col_w}}" for s in names))
    print("  " + "─" * (22 + col_w * len(names)))
    for name, bw in static_blends.items():
        w_str = "".join(f"  {bw.weights.get(s, 0.0):>+{col_w}.4f}" for s in names)
        print(f"  {name:<20}{w_str}")
    print("  " + "─" * (22 + col_w * len(names)))


if __name__ == "__main__":
    main()
