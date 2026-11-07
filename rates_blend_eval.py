#!/usr/bin/env python3
"""
Rates blend evaluation: IS / OOS Sharpe for all blend methods.

Blend weights are fitted on IS data only; OOS is a pure held-out test.

  IS  : start → 2018-12-31
  OOS : 2019-01-01 → present

Managed returns F (T × 3) are vol-normalised to unit annual vol using IS
statistics before any estimator is fitted.  This puts momentum, carry and
mean_reversion on a common risk scale — the blend then allocates *risk budget*
across signals rather than raw PnL units.

Usage:
    python rates_blend_eval.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

IS_END    = "2018-12-31"
OOS_START = "2019-01-01"


# ── helpers ───────────────────────────────────────────────────────────────────

def annualized_sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 30 or r.std() < 1e-12:
        return np.nan
    return float(r.mean() / r.std() * np.sqrt(252))


def blend_return(F: pd.DataFrame, weights: "BlendWeights") -> pd.Series:  # noqa: F821
    w = weights.weights.reindex(F.columns).fillna(0.0)
    total = w.abs().sum()
    if total > 1e-12:
        w = w / total
    return (F * w.values).sum(axis=1)


def vol_normalise(F_is: pd.DataFrame, F_oos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scale each column to unit annualised vol using IS σ only (no lookahead)."""
    sigma = F_is.std() * np.sqrt(252)
    sigma = sigma.replace(0.0, np.nan)
    return F_is.div(sigma), F_oos.div(sigma)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    from rates.blend import load_managed_returns
    from equities.blend import (
        equal_weight, sharpe_heuristic, msrr_ridge,
        enet_blend, c_enet_blend, e_enet_blend,
        inv_msfe_blend, pcr_blend, pls_blend,
        DMABlend,
    )

    # ── load managed returns ──────────────────────────────────────────────────
    F = load_managed_returns()
    F_is  = F.loc[F.index <= IS_END]
    F_oos = F.loc[F.index >= OOS_START]

    names = list(F.columns)
    print(f"IS  observations : {len(F_is):,}  ({F_is.index[0].date()} → {F_is.index[-1].date()})")
    print(f"OOS observations : {len(F_oos):,}  ({F_oos.index[0].date()} → {F_oos.index[-1].date()})")
    print(f"Signals          : {names}\n")

    # ── vol-normalise (IS σ applied to both splits) ───────────────────────────
    F_is_n, F_oos_n = vol_normalise(F_is, F_oos)

    # ── fit all static blend methods on IS ────────────────────────────────────
    static_blends: dict = {
        "equal_weight" : equal_weight(names),
        "sh_p1"        : sharpe_heuristic(F_is_n, power=1),
        "sh_p2"        : sharpe_heuristic(F_is_n, power=2),
        "msrr_ols"     : msrr_ridge(F_is_n, lam=0.0),
        "msrr_ridge"   : msrr_ridge(F_is_n, lam=0.01),
        "enet"         : enet_blend(F_is_n),
        "lasso"        : enet_blend(F_is_n, l1_ratio=1.0),
        "c_enet"       : c_enet_blend(F_is_n),
        "e_enet"       : e_enet_blend(F_is_n),
        "e_lasso"      : e_enet_blend(F_is_n, l1_ratio=1.0),
        "inv_msfe"     : inv_msfe_blend(F_is_n),
        "pcr"          : pcr_blend(F_is_n),
        "pls"          : pls_blend(F_is_n),
    }

    # ── DMA: fit on IS, predict OOS ───────────────────────────────────────────
    dma = DMABlend(alpha=0.99)
    dma.fit(F_is_n)
    dma_w_is  = dma._weights_history
    dma_w_oos = dma.predict_weights(F_oos_n)

    def dma_blended_return(F: pd.DataFrame, w_df: pd.DataFrame) -> pd.Series:
        idx = F.index.intersection(w_df.index)
        F_, w_ = F.loc[idx], w_df.loc[idx]
        row_norm = w_.abs().sum(axis=1).replace(0.0, np.nan)
        return (F_ * w_.div(row_norm, axis=0)).sum(axis=1)

    # ── collect results ───────────────────────────────────────────────────────
    rows = []

    for sig in names:
        rows.append({
            "method"    : sig,
            "type"      : "sleeve",
            "IS_Sharpe" : annualized_sharpe(F_is_n[sig]),
            "OOS_Sharpe": annualized_sharpe(F_oos_n[sig]),
        })

    for name, bw in static_blends.items():
        rows.append({
            "method"    : name,
            "type"      : "static",
            "IS_Sharpe" : annualized_sharpe(blend_return(F_is_n,  bw)),
            "OOS_Sharpe": annualized_sharpe(blend_return(F_oos_n, bw)),
        })

    rows.append({
        "method"    : "dma",
        "type"      : "dynamic",
        "IS_Sharpe" : annualized_sharpe(dma_blended_return(F_is_n,  dma_w_is)),
        "OOS_Sharpe": annualized_sharpe(dma_blended_return(F_oos_n, dma_w_oos)),
    })

    # ── print Sharpe table ────────────────────────────────────────────────────
    W = 58
    print("=" * W)
    print("  RATES BLEND EVALUATION  (F vol-normalised to unit ann. vol)")
    print(f"  IS ≤ {IS_END}   |   OOS ≥ {OOS_START}")
    print("=" * W)
    print(f"  {'Method':<20}  {'Type':<8}  {'IS SR':>8}  {'OOS SR':>8}")
    print("  " + "─" * (W - 2))

    prev_type = None
    for r in rows:
        if prev_type is not None and r["type"] != prev_type:
            print("  " + "─" * (W - 2))
        prev_type = r["type"]
        is_s  = f"{r['IS_Sharpe']:+.2f}"  if not np.isnan(r["IS_Sharpe"])  else "   n/a"
        oos_s = f"{r['OOS_Sharpe']:+.2f}" if not np.isnan(r["OOS_Sharpe"]) else "   n/a"
        print(f"  {r['method']:<20}  {r['type']:<8}  {is_s:>8}  {oos_s:>8}")

    print("=" * W)

    # ── blend weights table ───────────────────────────────────────────────────
    col_w = 16
    print(f"\n  BLEND WEIGHTS  (IS-fitted static methods)")
    print("  " + "─" * (22 + col_w * len(names)))
    print(f"  {'Method':<20}" + "".join(f"  {s:>{col_w}}" for s in names))
    print("  " + "─" * (22 + col_w * len(names)))
    for name, bw in static_blends.items():
        w_str = "".join(f"  {bw.weights.get(s, 0.0):>+{col_w}.4f}" for s in names)
        print(f"  {name:<20}{w_str}")
    print("  " + "─" * (22 + col_w * len(names)))

    # ── DMA average weights ───────────────────────────────────────────────────
    print(f"\n  DMA AVERAGE WEIGHTS")
    print("  " + "─" * (22 + col_w * len(names)))
    for label, w_df in [("IS avg", dma_w_is), ("OOS avg", dma_w_oos)]:
        avg = w_df.mean()
        w_str = "".join(f"  {avg.get(s, 0.0):>+{col_w}.4f}" for s in names)
        print(f"  {label:<20}{w_str}")
    print("  " + "─" * (22 + col_w * len(names)))


if __name__ == "__main__":
    main()
