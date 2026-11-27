"""
Classic NS whole-curve MR: hysteresis threshold sweep.

Applies per-tenor hysteresis (enter |z_tenor| > entry_z, exit |z_tenor| < exit_z)
to the committed z-scores and DV01 position parquets.  Reports gross/net Sharpe
under the frozen TC model (TC charged only on per-tenor entry/exit events).

PnL approximation: pnl_gross × fraction_of_dv01_active.  Since each tenor's PnL
contribution is proportional to its held DV01, this is a good first-order estimate.
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

ENTRY_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
EXIT_GRID  = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]

THRESHOLDS = [(e, x) for e in ENTRY_GRID for x in EXIT_GRID if x < e]
THRESHOLDS = [(0.0, 0.0)] + THRESHOLDS   # continuous baseline


def _per_tenor_indicator(z_df: pd.DataFrame, entry: float, exit_z: float) -> pd.DataFrame:
    """
    Per-tenor 0/1 indicator matrix.  For each tenor independently:
    enter when |z| > entry, exit when |z| < exit_z or z crosses zero.
    entry=0 → always active.
    """
    if entry == 0.0:
        return pd.DataFrame(
            np.where(z_df.notna(), 1.0, np.nan),
            index=z_df.index, columns=z_df.columns,
        )
    arr = z_df.to_numpy(dtype=float)
    out = np.zeros_like(arr)
    for j in range(arr.shape[1]):
        s, pentry = 0, 0.0
        for i in range(arr.shape[0]):
            x = arr[i, j]
            if not np.isfinite(x):
                s = 0; out[i, j] = np.nan; continue
            if s == 1:
                if abs(x) < exit_z or x * pentry <= 0.0:
                    s = 0
            if s == 0 and abs(x) > entry:
                s = 1; pentry = float(np.sign(x))
            out[i, j] = float(s)
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

        # active tenors only (non-zero DV01 positions)
        active = pos_m.columns[pos_m.abs().sum() > 0]
        if len(active) == 0:
            continue

        zs_act  = zs_m[active].reindex(pnl_m.index)
        pos_act = pos_m[active].reindex(pnl_m.index).fillna(0.0)
        total_abs_dv01 = pos_act.abs().sum(axis=1).replace(0.0, np.nan)

        h_mkt_best = h_best.get(mkt, 2.0)
        h_mkt_base = h_base.get(mkt, 2.0)

        for entry_z, exit_z in THRESHOLDS:
            I = _per_tenor_indicator(zs_act, entry_z, exit_z).reindex(pnl_m.index).fillna(0.0)

            # fraction of DV01 capacity active each day
            held_abs = pos_act.abs().multiply(I)
            frac_active = held_abs.sum(axis=1) / total_abs_dv01.fillna(1.0)

            pnl_hyst = pnl_m * frac_active.reindex(pnl_m.index).fillna(0.0)

            # frozen TC: pay on per-tenor entry/exit transitions only
            held_dv01   = pos_act.multiply(I)
            tc_frozen   = held_dv01.diff().abs().sum(axis=1)

            for period, sl in [("IS", slice(None, IS_END)), ("OOS", slice(OOS_START, None))]:
                p = pnl_hyst.loc[sl].dropna()
                if len(p) < 30:
                    continue
                vol = p.std() * np.sqrt(252)
                if vol < 1e-12:
                    continue
                g_mean    = p.mean() * 252
                tc_f_best = tc_frozen.reindex(p.index).fillna(0.0).mean() * 252 * h_mkt_best
                tc_f_base = tc_frozen.reindex(p.index).fillna(0.0).mean() * 252 * h_mkt_base

                rows.append(dict(
                    mkt=mkt, entry_z=entry_z, exit_z=exit_z, period=period,
                    gross_sr        = _sr(p),
                    net_sr_frz_best = (g_mean - tc_f_best) / vol,
                    net_sr_frz_base = (g_mean - tc_f_base) / vol,
                    dv01_pd_frz     = tc_frozen.reindex(p.index).fillna(0.0).mean(),
                    pct_active      = float(frac_active.reindex(p.index).fillna(0.0).mean()) * 100,
                ))

        print(f"  {mkt} done", flush=True)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Running classic NS MR hysteresis grid search...\n")
    df = run_sweep()

    pd.set_option("display.width", 180)
    pd.set_option("display.float_format", "{:.2f}".format)

    is_df  = df[df.period == "IS"].copy()
    oos_df = df[df.period == "OOS"].copy()

    # best IS config per market (by net_sr_frz_best)
    best_is = (is_df.loc[is_df.groupby("mkt")["net_sr_frz_best"].idxmax()]
                    .set_index("mkt")[["entry_z", "exit_z", "gross_sr",
                                       "net_sr_frz_best", "net_sr_frz_base",
                                       "dv01_pd_frz", "pct_active"]])

    # OOS performance at those same (entry_z, exit_z) params
    oos_at_best = (oos_df.merge(
        best_is.reset_index()[["mkt", "entry_z", "exit_z"]],
        on=["mkt", "entry_z", "exit_z"], how="inner"
    ).set_index("mkt")[["entry_z", "exit_z", "gross_sr",
                         "net_sr_frz_best", "net_sr_frz_base",
                         "dv01_pd_frz", "pct_active"]])

    print("=== Best IS params per market ===")
    print(best_is.round(2).to_string())
    print()
    print("=== OOS performance at best-IS params ===")
    print(oos_at_best.round(2).to_string())

    print("\n=== Full IS grid (all markets averaged) ===")
    cols = ["gross_sr", "net_sr_frz_best", "net_sr_frz_base", "dv01_pd_frz", "pct_active"]
    agg = is_df.groupby(["entry_z", "exit_z"])[cols].mean().round(2)
    print(agg.to_string())
