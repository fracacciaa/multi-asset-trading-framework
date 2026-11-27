"""
Classic NS MR — direction-only (sign-based, fixed notional) sweep.

Instead of position ∝ z_score (daily proportional resizing), each tenor holds:
  position_t = fixed_notional_tenor × sign(z_t)  when |z_t| > dead_band
  position_t = 0                                   when |z_t| ≤ dead_band

fixed_notional_tenor = IS median of |target_dv01_tenor| so the total capital
deployed matches the continuous signal when fully invested.

TC is charged only when a tenor's position changes (sign flip or dead-band
transition).  Compared with the continuous approach this gives far fewer
trading events when dead_band > 0, but even dead_band=0 trades only on sign
crossings (~12/yr per tenor) rather than daily.

PnL approximation:
  pnl_dir_t ≈ pnl_gross_t × frac_active_t
where frac_active = (Σ_tenor I_active × fixed_k) / (Σ_tenor fixed_k).
This holds exactly when all active tenors have the same sign as the continuous
target (true except during brief sign-crossing windows).
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from rates.blend import TC_SCENARIOS, IS_END, OOS_START
from rates.production.ois_data import OIS_MARKETS

SIG_DIR = os.path.join(os.path.dirname(__file__), "signals")

DEAD_BANDS = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00]


def _direction_positions(
    z_df: pd.DataFrame,
    fixed_k: pd.Series,
    dead_band: float,
) -> pd.DataFrame:
    """
    Per-tenor position matrix.
    Entry: |z| exits dead_band (crosses dead_band outward).
    Exit:  |z| re-enters dead_band (crosses dead_band inward, toward zero).
    At dead_band=0: change on every sign crossing.
    """
    arr  = z_df.to_numpy(dtype=float)
    out  = np.zeros_like(arr)
    fk   = fixed_k.to_numpy(dtype=float)
    for j in range(arr.shape[1]):
        s = 0.0  # current held position (+fk, -fk, or 0)
        for i in range(arr.shape[0]):
            x = arr[i, j]
            if not np.isfinite(x):
                s = 0.0
                out[i, j] = np.nan
                continue
            if dead_band == 0.0:
                # always in-position, sign-driven
                s = fk[j] * float(np.sign(x)) if x != 0 else s
            else:
                if s != 0.0 and abs(x) <= dead_band:
                    s = 0.0
                if s == 0.0 and abs(x) > dead_band:
                    s = fk[j] * float(np.sign(x))
            out[i, j] = s
    return pd.DataFrame(out, index=z_df.index, columns=z_df.columns)


def _sr(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if len(r) > 30 and r.std() > 1e-12 else float("nan")


def run_sweep(markets: list[str] | None = None) -> pd.DataFrame:
    zs_all  = pd.read_parquet(os.path.join(SIG_DIR, "mean_reversion_zscores.parquet"))
    pos_all = pd.read_parquet(os.path.join(SIG_DIR, "mean_reversion_dv01_pos.parquet"))
    pnl_all = pd.read_parquet(os.path.join(SIG_DIR, "mean_reversion_pnl.parquet"))

    mkts = markets or [m for m in OIS_MARKETS if m in pnl_all.columns]

    h_best = TC_SCENARIOS["best"]
    h_base = TC_SCENARIOS["base"]

    rows = []
    for mkt in mkts:
        if mkt not in zs_all.columns.get_level_values("market"):
            continue

        zs_m   = zs_all.xs(mkt, level="market", axis=1)
        pos_m  = pos_all.xs(mkt, level="market", axis=1)
        pnl_m  = pnl_all[mkt].dropna()

        active = pos_m.columns[pos_m.abs().sum() > 0]
        if len(active) == 0:
            continue

        zs_act  = zs_m[active].reindex(pnl_m.index)
        pos_act = pos_m[active].reindex(pnl_m.index).fillna(0.0)

        # fixed notional per tenor = IS median of |continuous target|
        fixed_k = pos_act.loc[pos_act.index <= IS_END].abs().median()
        fixed_k = fixed_k.replace(0.0, np.nan).dropna()
        active  = fixed_k.index
        zs_act  = zs_act[active]
        pos_act = pos_act[active]

        total_fixed_k = fixed_k.sum()
        h_mkt_best = h_best.get(mkt, 2.0)
        h_mkt_base = h_base.get(mkt, 2.0)

        for dead_band in DEAD_BANDS:
            dir_pos = _direction_positions(zs_act, fixed_k, dead_band).reindex(pnl_m.index).fillna(0.0)

            # fraction of capacity active each day
            frac_active = dir_pos.abs().sum(axis=1) / total_fixed_k
            pnl_dir = pnl_m * frac_active

            # TC: per-tenor changes (sign flips + dead-band transitions)
            tc_daily = dir_pos.diff().abs().sum(axis=1)

            for period, sl in [("IS", slice(None, IS_END)), ("OOS", slice(OOS_START, None))]:
                p = pnl_dir.loc[sl].dropna()
                if len(p) < 30:
                    continue
                vol = p.std() * np.sqrt(252)
                if vol < 1e-12:
                    continue
                g_mean    = p.mean() * 252
                tc_b_ann  = tc_daily.reindex(p.index).fillna(0.0).mean() * 252 * h_mkt_best
                tc_bs_ann = tc_daily.reindex(p.index).fillna(0.0).mean() * 252 * h_mkt_base

                rows.append(dict(
                    mkt=mkt, dead_band=dead_band, period=period,
                    gross_sr    = _sr(p),
                    net_sr_best = (g_mean - tc_b_ann) / vol,
                    net_sr_base = (g_mean - tc_bs_ann) / vol,
                    dv01_pd     = tc_daily.reindex(p.index).fillna(0.0).mean(),
                    pct_active  = float(frac_active.reindex(p.index).fillna(0.0).mean()) * 100,
                ))

        print(f"  {mkt} done", flush=True)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Running classic NS direction-only sweep...\n")
    df = run_sweep()

    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.2f}".format)

    is_df  = df[df.period == "IS"].copy()
    oos_df = df[df.period == "OOS"].copy()

    best_is = (is_df.loc[is_df.groupby("mkt")["net_sr_best"].idxmax()]
                    .set_index("mkt")[["dead_band", "gross_sr", "net_sr_best",
                                       "net_sr_base", "dv01_pd", "pct_active"]])

    oos_at_best = (oos_df.merge(
        best_is.reset_index()[["mkt", "dead_band"]],
        on=["mkt", "dead_band"], how="inner"
    ).set_index("mkt")[["dead_band", "gross_sr", "net_sr_best",
                         "net_sr_base", "dv01_pd", "pct_active"]])

    print("=== Best IS dead-band per market ===")
    print(best_is.round(2).to_string())
    print()
    print("=== OOS at best-IS dead-band ===")
    print(oos_at_best.round(2).to_string())

    print("\n=== Cross-market average by dead_band ===")
    cols = ["gross_sr", "net_sr_best", "net_sr_base", "dv01_pd", "pct_active"]
    agg = df.groupby(["dead_band", "period"])[cols].mean().round(2)
    print(agg.to_string())
