"""
Butterfly hysteresis threshold sweep.

For each market: build fly_resid from committed z-scores, apply a regime
hysteresis filter (enter |fly_resid| > entry_z × σ_fly, exit at zero crossing),
and report gross/net Sharpe and turnover under two TC models:

  continuous : daily rebalancing within-position (high turnover, upper bound on cost)
  frozen     : position frozen at entry size, TC charged only on entry/exit events
               (realistic for a butterfly held as a single package)

PnL approximation: pnl_gross × I(holding).  Assumes position size changes slowly
enough that the frozen-size error is small.  Exact for the frozen model.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from rates.production.ns_classic.fly_signal import (
    select_fly_legs, fly_vol_scale_k,
    FLY_WEIGHTS, MIN_OBS, DEFAULT_TARGET_VOL, DEFAULT_EWMA_HL,
)
from rates.blend import TC_SCENARIOS, IS_END, OOS_START
from rates.production.ois_data import OIS_MARKETS

SIG_DIR = os.path.join(os.path.dirname(__file__), "signals")

# (entry_z, exit_z) pairs to sweep
THRESHOLDS = [
    (0.0,  0.0),
    (1.0,  0.0),
    (1.5,  0.0),
    (2.0,  0.0),
    (2.0,  0.5),
    (2.0,  1.0),
    (2.5,  0.5),
    (2.5,  1.0),
]


def _hysteresis_indicator(z: pd.Series, entry: float, exit_z: float = 0.0) -> pd.Series:
    """
    Returns 0/1 indicator: 1 = holding, 0 = flat.
    Enter when |z| > entry; exit when |z| < exit_z (same side as entry).
    exit_z=0 means exit only when z crosses zero.
    entry=0.0 means always in-position.
    """
    if entry == 0.0:
        return pd.Series(np.where(z.notna(), 1.0, np.nan), index=z.index)
    vals = z.to_numpy(dtype=float)
    out  = np.zeros(len(vals))
    s    = 0
    prev_entry_sign = 0.0
    for i, x in enumerate(vals):
        if not np.isfinite(x):
            s = 0
            out[i] = np.nan
            continue
        if s == 1:
            # exit when z has reverted past exit_z toward zero (same side)
            if abs(x) < exit_z or x * prev_entry_sign <= 0.0:
                s = 0
        if s == 0 and abs(x) > entry:
            s = 1
            prev_entry_sign = float(np.sign(x))
        out[i] = float(s)
    return pd.Series(out, index=z.index)


def _sr(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if len(r) > 30 and r.std() > 1e-12 else float("nan")


def run_sweep(markets: list[str] | None = None) -> pd.DataFrame:
    zs_all  = pd.read_parquet(os.path.join(SIG_DIR, "mean_reversion_zscores.parquet"))
    pnl_all = pd.read_parquet(os.path.join(SIG_DIR, "mean_reversion_fly_pnl.parquet"))

    mkts = markets or [m for m in OIS_MARKETS if m in pnl_all.columns]

    h_best = TC_SCENARIOS["best"]
    h_base = TC_SCENARIOS["base"]

    rows = []
    for mkt in mkts:
        if mkt not in zs_all.columns.get_level_values("market"):
            continue

        zs_m  = zs_all.xs(mkt, level="market", axis=1)
        pnl_m = pnl_all[mkt].dropna()

        legs = select_fly_legs(zs_m, IS_END, min_obs=MIN_OBS)
        if legs is None:
            continue
        lower, body, upper = legs
        w_lo, w_bd, w_up = FLY_WEIGHTS

        zz = zs_m[[lower, body, upper]].dropna()
        fly_resid = w_lo * zz[lower] + w_bd * zz[body] + w_up * zz[upper]
        k_arr = fly_vol_scale_k(fly_resid, DEFAULT_TARGET_VOL, halflife=DEFAULT_EWMA_HL)

        # target DV01 per leg (continuous, as in fly_signal.py)
        target = {
            lower: -(k_arr * fly_resid) * w_lo,
            body:  -(k_arr * fly_resid) * w_bd,
            upper: -(k_arr * fly_resid) * w_up,
        }
        target_df = pd.DataFrame(target).reindex(pnl_m.index).fillna(0.0)
        total_abs_dv01 = target_df.abs().sum(axis=1)  # sum |leg DV01| ≈ 2×|k×fly_resid|

        h_mkt_best = h_best.get(mkt, 2.0)
        h_mkt_base = h_base.get(mkt, 2.0)

        # normalize by expanding std of fly_resid level (causal, in z-score units)
        # fly_resid is built from individual z-scores so std ≈ 1.2; this gives
        # a true z-score where |fly_z|>2 fires ~5-10% of the time
        fly_std = fly_resid.expanding(min_periods=DEFAULT_EWMA_HL).std()
        fly_z   = (fly_resid / fly_std.replace(0, np.nan)).reindex(pnl_m.index)

        for entry_z, exit_z in THRESHOLDS:
            I_hold = _hysteresis_indicator(fly_z, entry_z, exit_z).reindex(pnl_m.index).fillna(0.0)

            pnl_hyst = pnl_m * I_hold

            # ── TC model 1: continuous (daily rebalancing when in-position) ──────
            held_df  = target_df.multiply(I_hold, axis=0)
            tc_cont  = held_df.diff().abs().sum(axis=1)

            # ── TC model 2: frozen (TC only on entry/exit events) ─────────────
            transitions = I_hold.diff().abs().fillna(0.0)
            tc_frozen   = total_abs_dv01 * transitions

            for period, sl in [("IS", slice(None, IS_END)), ("OOS", slice(OOS_START, None))]:
                p = pnl_hyst.loc[sl].dropna()
                if len(p) < 30:
                    continue
                vol    = p.std() * np.sqrt(252)
                if vol < 1e-12:
                    continue
                g_mean = p.mean() * 252
                i_act  = I_hold.reindex(p.index).fillna(0.0)

                tc_c_best = tc_cont.reindex(p.index).fillna(0.0).mean() * 252 * h_mkt_best
                tc_c_base = tc_cont.reindex(p.index).fillna(0.0).mean() * 252 * h_mkt_base
                tc_f_best = tc_frozen.reindex(p.index).fillna(0.0).mean() * 252 * h_mkt_best
                tc_f_base = tc_frozen.reindex(p.index).fillna(0.0).mean() * 252 * h_mkt_base

                rows.append(dict(
                    mkt=mkt, entry_z=entry_z, exit_z=exit_z, period=period,
                    gross_sr         = _sr(p),
                    net_sr_frz_best  = (g_mean - tc_f_best) / vol,
                    net_sr_frz_base  = (g_mean - tc_f_base) / vol,
                    net_sr_cont_best = (g_mean - tc_c_best) / vol,
                    dv01_pd_frz      = tc_frozen.reindex(p.index).fillna(0.0).mean(),
                    pct_in_mkt       = float(i_act.mean()) * 100,
                ))

        print(f"  {mkt} done", flush=True)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Running butterfly hysteresis sweep...\n")
    df = run_sweep()

    pd.set_option("display.width", 180)
    pd.set_option("display.float_format", "{:.2f}".format)

    print("\n=== Cross-market equal-weight average ===")
    cols = ["gross_sr", "net_sr_frz_best", "net_sr_frz_base", "net_sr_cont_best",
            "dv01_pd_frz", "pct_in_mkt"]
    agg = df.groupby(["entry_z", "exit_z", "period"])[cols].mean().round(2)
    print(agg.to_string())

    print("\n=== Per-market IS results ===")
    is_df = df[df.period == "IS"].drop(columns="period")
    print(is_df.to_string(index=False))
