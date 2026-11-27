"""
Classic NS MR — direction-only with asymmetric entry/exit hysteresis.

Position per tenor:
  - Flat when |z| ≤ exit_z  (or before first entry)
  - fixed_notional × sign(z_at_entry) when |z| > entry_z  → enter, freeze
  - Hold frozen until |z| drops back to exit_z, then close

fixed_notional_tenor = IS median of |target_dv01_tenor|.

TC charged only on entry and exit events (no within-trade rebalancing).
PnL ≈ pnl_gross × frac_active (fraction of fixed capacity deployed).
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from rates.blend import TC_SCENARIOS, IS_END, OOS_START
from rates.production.ois_data import OIS_MARKETS

SIG_DIR = os.path.join(os.path.dirname(__file__), "signals")

ENTRY_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
EXIT_GRID  = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
THRESHOLDS = [(e, x) for e in ENTRY_GRID for x in EXIT_GRID if x < e]


def _dir_hyst_positions(
    z_df: pd.DataFrame,
    fixed_k: pd.Series,
    entry: float,
    exit_z: float,
) -> pd.DataFrame:
    """
    Per-tenor direction positions with asymmetric entry/exit.
    Returns DataFrame of same shape as z_df with values in {-fixed_k, 0, +fixed_k}.
    """
    arr = z_df.to_numpy(dtype=float)
    out = np.zeros_like(arr)
    fk  = fixed_k.to_numpy(dtype=float)
    for j in range(arr.shape[1]):
        s = 0.0
        for i in range(arr.shape[0]):
            x = arr[i, j]
            if not np.isfinite(x):
                s = 0.0; out[i, j] = np.nan; continue
            if s != 0.0 and abs(x) <= exit_z:
                s = 0.0
            if s == 0.0 and abs(x) > entry:
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

        zs_m  = zs_all.xs(mkt, level="market", axis=1)
        pos_m = pos_all.xs(mkt, level="market", axis=1)
        pnl_m = pnl_all[mkt].dropna()

        active  = pos_m.columns[pos_m.abs().sum() > 0]
        zs_act  = zs_m[active].reindex(pnl_m.index)
        pos_act = pos_m[active].reindex(pnl_m.index).fillna(0.0)

        fixed_k = pos_act.loc[pos_act.index <= IS_END].abs().median()
        fixed_k = fixed_k.replace(0.0, np.nan).dropna()
        active  = fixed_k.index
        zs_act  = zs_act[active]

        total_k     = fixed_k.sum()
        h_mkt_best  = h_best.get(mkt, 2.0)
        h_mkt_base  = h_base.get(mkt, 2.0)

        for entry_z, exit_z in THRESHOLDS:
            pos_dir = _dir_hyst_positions(zs_act, fixed_k, entry_z, exit_z).reindex(pnl_m.index).fillna(0.0)

            frac_active = pos_dir.abs().sum(axis=1) / total_k
            pnl_dir     = pnl_m * frac_active
            tc_daily    = pos_dir.diff().abs().sum(axis=1)

            for period, sl in [("IS", slice(None, IS_END)), ("OOS", slice(OOS_START, None))]:
                p = pnl_dir.loc[sl].dropna()
                if len(p) < 30:
                    continue
                vol = p.std() * np.sqrt(252)
                if vol < 1e-12:
                    continue
                g_mean = p.mean() * 252
                tc_b   = tc_daily.reindex(p.index).fillna(0.0).mean() * 252 * h_mkt_best
                tc_bs  = tc_daily.reindex(p.index).fillna(0.0).mean() * 252 * h_mkt_base

                rows.append(dict(
                    mkt=mkt, entry_z=entry_z, exit_z=exit_z, period=period,
                    gross_sr    = _sr(p),
                    net_sr_best = (g_mean - tc_b)  / vol,
                    net_sr_base = (g_mean - tc_bs) / vol,
                    dv01_pd     = tc_daily.reindex(p.index).fillna(0.0).mean(),
                    pct_active  = float(frac_active.reindex(p.index).fillna(0.0).mean()) * 100,
                ))

        print(f"  {mkt} done", flush=True)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Running NS direction + hysteresis grid search...\n")
    df = run_sweep()

    pd.set_option("display.width", 180)
    pd.set_option("display.float_format", "{:.2f}".format)

    is_df  = df[df.period == "IS"].copy()
    oos_df = df[df.period == "OOS"].copy()

    best_is = (is_df.loc[is_df.groupby("mkt")["net_sr_best"].idxmax()]
                    .set_index("mkt")[["entry_z", "exit_z", "gross_sr",
                                       "net_sr_best", "net_sr_base",
                                       "dv01_pd", "pct_active"]])

    oos_at_best = (oos_df.merge(
        best_is.reset_index()[["mkt", "entry_z", "exit_z"]],
        on=["mkt", "entry_z", "exit_z"], how="inner"
    ).set_index("mkt")[["entry_z", "exit_z", "gross_sr",
                         "net_sr_best", "net_sr_base",
                         "dv01_pd", "pct_active"]])

    print("=== Best IS params per market ===")
    print(best_is.round(2).to_string())
    print()
    print("=== OOS at best-IS params ===")
    print(oos_at_best.round(2).to_string())

    print("\n=== Cross-market IS average across grid ===")
    cols = ["gross_sr", "net_sr_best", "net_sr_base", "dv01_pd", "pct_active"]
    agg = is_df.groupby(["entry_z", "exit_z"])[cols].mean().round(2)
    print(agg.to_string())
