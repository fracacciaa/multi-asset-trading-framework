#!/usr/bin/env python3
"""
Equities blend evaluation: IS / OOS Sharpe ratios for all blend methods.

Blend weights are estimated on IS data only; OOS is a pure held-out test.

  IS  : 2005-02-01 → 2020-12-31  (after VF warmup)
  OOS : 2021-01-01 → present

Usage:
    python equities_blend_eval.py             # use cached positions if available
    python equities_blend_eval.py --rebuild   # force rebuild VF positions
"""

from __future__ import annotations
import argparse
import os
import pickle
import sys
import numpy as np
import pandas as pd

IS_END    = "2020-12-31"
OOS_START = "2021-01-01"
CACHE_PKL = "equities/cache/vf_positions.pkl"

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


# ── data / cache ──────────────────────────────────────────────────────────────

def load_or_build(rebuild: bool) -> dict[str, pd.DataFrame]:
    os.makedirs(os.path.dirname(CACHE_PKL), exist_ok=True)

    if not rebuild and os.path.exists(CACHE_PKL):
        print("Loading cached VF positions …")
        with open(CACHE_PKL, "rb") as f:
            return pickle.load(f)

    print("Building VF positions — this takes ~5 min …")
    from equities.data import fetch_prices, fetch_fundamentals
    from equities.signals import build_vf_positions

    prices       = fetch_prices(tickers=None)
    fundamentals = fetch_fundamentals(tickers=None)
    returns      = prices.pct_change()

    positions = build_vf_positions(
        prices=prices,
        fundamentals=fundamentals,
        returns=returns,
        warmup=504,
        rebal_freq=21,
        cov_lookback=504,
        vol_halflife=21,
        cor_halflife=126,
    )
    with open(CACHE_PKL, "wb") as f:
        pickle.dump(positions, f)
    print(f"Cached to {CACHE_PKL}\n")
    return positions


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    positions = load_or_build(args.rebuild)

    from equities.data import fetch_prices
    prices  = fetch_prices(tickers=None)
    returns = prices.pct_change().dropna(how="all")

    # ── split positions and returns at IS / OOS boundary ─────────────────────
    pos_is  = {k: v.loc[v.index <= IS_END]   for k, v in positions.items()}
    pos_oos = {k: v.loc[v.index >= OOS_START] for k, v in positions.items()}
    ret_is  = returns.loc[returns.index <= IS_END]
    ret_oos = returns.loc[returns.index >= OOS_START]

    from equities.blend import (
        build_managed_returns,
        equal_weight, sharpe_heuristic, msrr_ridge,
        enet_blend, c_enet_blend, e_enet_blend,
        inv_msfe_blend, pcr_blend, pls_blend,
        DMABlend,
    )

    # ── managed returns: T × K daily portfolio returns per sleeve ─────────────
    F_is  = build_managed_returns(pos_is,  ret_is)
    F_oos = build_managed_returns(pos_oos, ret_oos)

    names = list(F_is.columns)
    print(f"IS  observations : {len(F_is):,}  ({F_is.index[0].date()} → {F_is.index[-1].date()})")
    print(f"OOS observations : {len(F_oos):,}  ({F_oos.index[0].date()} → {F_oos.index[-1].date()})")
    print(f"Signals : {names}\n")

    # ── fit static blend methods on IS ────────────────────────────────────────
    static_blends: dict[str, "BlendWeights"] = {  # noqa: F821
        "equal_weight" : equal_weight(names),
        "sh_p1"        : sharpe_heuristic(F_is, power=1),
        "sh_p2"        : sharpe_heuristic(F_is, power=2),
        "msrr_ols"     : msrr_ridge(F_is, lam=0.0),
        "msrr_ridge"   : msrr_ridge(F_is, lam=0.01),
        "enet"         : enet_blend(F_is),
        "lasso"        : enet_blend(F_is, l1_ratio=1.0),
        "c_enet"       : c_enet_blend(F_is),
        "e_enet"       : e_enet_blend(F_is),
        "e_lasso"      : e_enet_blend(F_is, l1_ratio=1.0),
        "inv_msfe"     : inv_msfe_blend(F_is),
        "pcr"          : pcr_blend(F_is),
        "pls"          : pls_blend(F_is),
    }

    # ── DMA: fit on IS, predict OOS ───────────────────────────────────────────
    dma = DMABlend(alpha=0.99)
    dma.fit(F_is)
    dma_w_is  = dma._weights_history
    dma_w_oos = dma.predict_weights(F_oos)

    def dma_blended_return(F: pd.DataFrame, w_df: pd.DataFrame) -> pd.Series:
        idx = F.index.intersection(w_df.index)
        F_ = F.loc[idx]
        w_ = w_df.loc[idx]
        row_norm = w_.abs().sum(axis=1).replace(0.0, np.nan)
        w_ = w_.div(row_norm, axis=0)
        return (F_ * w_).sum(axis=1)

    # ── collect results ───────────────────────────────────────────────────────
    rows = []

    # individual sleeves (pre-blend)
    for sig in names:
        rows.append({
            "method"    : sig,
            "type"      : "sleeve",
            "IS_Sharpe" : annualized_sharpe(F_is[sig]),
            "OOS_Sharpe": annualized_sharpe(F_oos[sig]),
        })

    # static blends
    for name, bw in static_blends.items():
        rows.append({
            "method"    : name,
            "type"      : "static",
            "IS_Sharpe" : annualized_sharpe(blend_return(F_is,  bw)),
            "OOS_Sharpe": annualized_sharpe(blend_return(F_oos, bw)),
        })

    # DMA
    rows.append({
        "method"    : "dma",
        "type"      : "dynamic",
        "IS_Sharpe" : annualized_sharpe(dma_blended_return(F_is,  dma_w_is)),
        "OOS_Sharpe": annualized_sharpe(dma_blended_return(F_oos, dma_w_oos)),
    })

    # ── print table ───────────────────────────────────────────────────────────
    W = 58
    hdr = f"  {'Method':<20}  {'Type':<8}  {'IS Sharpe':>10}  {'OOS Sharpe':>10}"

    print("=" * W)
    print("  EQUITIES BLEND EVALUATION")
    print(f"  IS ≤ {IS_END}   |   OOS ≥ {OOS_START}")
    print("=" * W)
    print(hdr)
    print("  " + "─" * (W - 2))

    prev_type = None
    for r in rows:
        if prev_type is not None and r["type"] != prev_type:
            print("  " + "─" * (W - 2))
        prev_type = r["type"]

        is_s  = f"{r['IS_Sharpe']:+.2f}"  if not np.isnan(r["IS_Sharpe"])  else "   n/a"
        oos_s = f"{r['OOS_Sharpe']:+.2f}" if not np.isnan(r["OOS_Sharpe"]) else "   n/a"
        print(f"  {r['method']:<20}  {r['type']:<8}  {is_s:>10}  {oos_s:>10}")

    print("=" * W)

    # ── blend weights table ───────────────────────────────────────────────────
    col_w = 12
    print("\n  BLEND WEIGHTS (IS-fitted, static methods)")
    print("  " + "─" * (22 + col_w * len(names)))
    hdr2 = f"  {'Method':<20}" + "".join(f"  {s:>{col_w}}" for s in names)
    print(hdr2)
    print("  " + "─" * (22 + col_w * len(names)))
    for name, bw in static_blends.items():
        w_str = "".join(f"  {bw.weights.get(s, 0.0):>+{col_w}.4f}" for s in names)
        print(f"  {name:<20}{w_str}")
    print("  " + "─" * (22 + col_w * len(names)))


if __name__ == "__main__":
    main()
