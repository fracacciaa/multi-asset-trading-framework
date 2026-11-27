"""
TC comparison across all 4 NS MR signal structures.

Signals
-------
classic   full-curve continuous daily rebalancing  (~15 tenors per market)
direction full-curve direction + asymmetric hysteresis (entry=2.5, exit=0.25)
fly       3-leg DV01-neutral butterfly (outer-body-outer per market)
spread    2-leg DV01-neutral spread (outer two fly legs per market)

Metrics reported per signal × market
--------------------------------------
gross_sr         annualised Sharpe, no TC
net_sr_best/base/worst  TC deducted at market-specific half-spreads
dv01_pd          mean DV01 traded per day  (Σ_legs |Δpos| / day)
trades_tenor_yr  mean actual position changes per leg per year
n_legs           number of instruments traded per market
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from rates.blend import TC_SCENARIOS, IS_END, OOS_START
from rates.production.ois_data import OIS_MARKETS, fetch_ois_curves
from rates.production.ns_classic.bootstrap import bootstrap_curves
from rates.production.ns_classic.ns_fit import select_lambda
from rates.production.ns_classic.spread_signal import spread_pnl_repriced

SIG_DIR  = os.path.join(os.path.dirname(__file__), "signals")
_LAM_TRAIN_END = "2015-12-31"
_MR_TARGET_VOL = 100.0
_MR_BURN_IN    = 252


def _sr(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if len(r) > 30 and r.std() > 1e-12 else float("nan")


def _net_sr(pnl: pd.Series, tc_daily: pd.Series, h: float, period_slice) -> float:
    p  = pnl.loc[period_slice].dropna()
    tc = tc_daily.reindex(p.index).fillna(0.0)
    vol = p.std() * np.sqrt(252)
    if len(p) < 30 or vol < 1e-12:
        return float("nan")
    net = (p.mean() - tc.mean() * h) * 252 / vol
    return float(net)


def _dv01_pd(pos_df: pd.DataFrame, period_slice) -> float:
    sub = pos_df.loc[period_slice].dropna(how="all")
    if sub.empty:
        return float("nan")
    return float(sub.diff().abs().sum(axis=1).mean())


def _trades_per_tenor_yr(pos_df: pd.DataFrame, period_slice) -> float:
    """Mean number of actual position changes per leg per year."""
    sub = pos_df.loc[period_slice].dropna(how="all")
    if len(sub) < 2:
        return float("nan")
    changes = (sub.diff().abs() > 1e-8).sum(axis=0)
    yrs = len(sub) / 252.0
    # only count legs that ever have non-zero positions
    active_legs = sub.columns[sub.abs().sum() > 0]
    if len(active_legs) == 0:
        return float("nan")
    return float(changes[active_legs].mean() / yrs)


def _n_active_legs(pos_df: pd.DataFrame) -> int:
    return int((pos_df.abs().sum() > 0).sum())


def run_signal(name: str, pnl_all: pd.DataFrame, pos_all: pd.DataFrame) -> list[dict]:
    """Compute metrics for one signal using already-loaded pnl and pos DataFrames."""
    mkts = [m for m in OIS_MARKETS if m in pnl_all.columns]
    rows = []
    h_best  = TC_SCENARIOS["best"]
    h_base  = TC_SCENARIOS["base"]
    h_worst = TC_SCENARIOS["worst"]

    for mkt in mkts:
        pnl_m = pnl_all[mkt].dropna()
        if pnl_m.empty:
            continue

        # positions: handle both flat (date × market) and MultiIndex (date × (market,tenor))
        if isinstance(pos_all.columns, pd.MultiIndex):
            if mkt not in pos_all.columns.get_level_values("market"):
                continue
            pos_m = pos_all.xs(mkt, level="market", axis=1).reindex(pnl_m.index).fillna(0.0)
        else:
            if mkt not in pos_all.columns:
                continue
            pos_m = pos_all[[mkt]].reindex(pnl_m.index).fillna(0.0)

        active_legs = pos_m.columns[pos_m.abs().sum() > 0]
        if len(active_legs) == 0:
            continue
        pos_m = pos_m[active_legs]

        tc_daily = pos_m.diff().abs().sum(axis=1)
        h_b   = h_best.get(mkt, 2.0)
        h_bs  = h_base.get(mkt, 2.0)
        h_w   = h_worst.get(mkt, 2.0)
        n_legs = len(active_legs)

        for period, sl in [("IS", slice(None, IS_END)), ("OOS", slice(OOS_START, None))]:
            p = pnl_m.loc[sl].dropna()
            if len(p) < 30:
                continue
            rows.append(dict(
                signal=name, mkt=mkt, period=period,
                gross_sr         = _sr(p),
                net_sr_best      = _net_sr(pnl_m, tc_daily, h_b,  sl),
                net_sr_base      = _net_sr(pnl_m, tc_daily, h_bs, sl),
                net_sr_worst     = _net_sr(pnl_m, tc_daily, h_w,  sl),
                dv01_pd          = _dv01_pd(pos_m, sl),
                trades_tenor_yr  = _trades_per_tenor_yr(pos_m, sl),
                n_legs           = n_legs,
            ))

    return rows


def compute_spread_signal(mkts: list[str], fly_pos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full-repricing spread PnL and DV01 positions (outer fly legs per market).
    Returns (pnl_df [date×market], pos_df [date×(market,tenor)]).
    """
    pnl_dict: dict[str, pd.Series]    = {}
    pos_dict: dict[str, pd.DataFrame] = {}

    for mkt in mkts:
        # outer fly legs for this market
        active = fly_pos.xs(mkt, level="market", axis=1).columns
        active = [c for c in active if fly_pos.xs(mkt, level="market", axis=1)[c].abs().sum() > 0]
        if len(active) < 2:
            print(f"  {mkt}: skip (< 2 active fly legs)")
            continue
        T1, T2 = float(min(active)), float(max(active))

        try:
            par_mkt = fetch_ois_curves(mkt)
            zeros   = bootstrap_curves(par_mkt)
            try:
                lam = select_lambda(zeros, end=_LAM_TRAIN_END)["best_lambda"]
            except Exception:
                lam = 2.5

            pnl, sp_z, _, k_arr = spread_pnl_repriced(
                par_mkt, lam, T1=T1, T2=T2,
                target_ann_vol=_MR_TARGET_VOL,
                burn_in=_MR_BURN_IN,
            )
            pnl_dict[mkt] = pnl

            # Held DV01 positions: pos_T1 = k × sp_z, pos_T2 = -k × sp_z
            held = k_arr * sp_z
            pos_df = pd.DataFrame({T1: held, T2: -held}, index=pnl.index)
            pos_dict[mkt] = pos_df

            print(f"  {mkt}: T1={T1}Y T2={T2}Y  λ={lam}  n={pnl.dropna().__len__()}")
        except Exception as exc:
            print(f"  {mkt}: FAILED — {exc}")

    if not pnl_dict:
        return pd.DataFrame(), pd.DataFrame()

    pnl_df = pd.DataFrame(pnl_dict)
    pnl_df.index.name = "date"

    pos_all = pd.concat(pos_dict, axis=1)
    pos_all.columns.names = ["market", "tenor"]

    return pnl_df, pos_all


def main():
    pd.set_option("display.width", 180)
    pd.set_option("display.float_format", "{:.2f}".format)

    # ── Load existing parquets ────────────────────────────────────────────────
    print("Loading parquets...")
    classic_pnl  = pd.read_parquet(f"{SIG_DIR}/mean_reversion_pnl.parquet")
    classic_pos  = pd.read_parquet(f"{SIG_DIR}/mean_reversion_dv01_pos.parquet")
    dir_pnl      = pd.read_parquet(f"{SIG_DIR}/mean_reversion_dir_pnl.parquet")
    dir_pos      = pd.read_parquet(f"{SIG_DIR}/mean_reversion_dir_dv01_pos.parquet")
    fly_pnl      = pd.read_parquet(f"{SIG_DIR}/mean_reversion_fly_pnl.parquet")
    fly_pos      = pd.read_parquet(f"{SIG_DIR}/mean_reversion_fly_dv01_pos.parquet")

    # ── Compute spread signal (full repricing) ────────────────────────────────
    print("\nComputing spread signal (full repricing)...")
    mkts = [m for m in OIS_MARKETS if m in classic_pnl.columns]
    spread_pnl, spread_pos = compute_spread_signal(mkts, fly_pos)

    # ── Collect all metrics ───────────────────────────────────────────────────
    print("\nComputing metrics...")
    rows = []
    rows += run_signal("classic",   classic_pnl, classic_pos)
    rows += run_signal("direction", dir_pnl,     dir_pos)
    rows += run_signal("fly",       fly_pnl,     fly_pos)
    if not spread_pnl.empty:
        rows += run_signal("spread", spread_pnl, spread_pos)

    df = pd.DataFrame(rows)

    # ── Print per-signal market-level tables ──────────────────────────────────
    W = 110
    for sig in ["classic", "direction", "fly", "spread"]:
        sub = df[df.signal == sig]
        if sub.empty:
            continue
        for period in ["IS", "OOS"]:
            p = sub[sub.period == period].set_index("mkt").sort_index()
            if p.empty:
                continue
            cols = ["gross_sr", "net_sr_best", "net_sr_base", "net_sr_worst",
                    "dv01_pd", "trades_tenor_yr", "n_legs"]
            print(f"\n{'='*W}")
            print(f"  SIGNAL: {sig.upper():10s}  |  {period}")
            print(f"  {'mkt':<18}  {'gross_sr':>8}  {'net_best':>8}  {'net_base':>8}  {'net_worst':>9}  {'dv01/d':>8}  {'trades/t/yr':>12}  {'n_legs':>6}")
            print(f"  {'-'*W}")
            for mkt, row in p[cols].iterrows():
                print(
                    f"  {mkt:<18}  {row.gross_sr:>8.2f}  {row.net_sr_best:>8.2f}  "
                    f"{row.net_sr_base:>8.2f}  {row.net_sr_worst:>9.2f}  "
                    f"{row.dv01_pd:>8.2f}  {row.trades_tenor_yr:>12.1f}  {int(row.n_legs):>6d}"
                )

    # ── Cross-market summary ──────────────────────────────────────────────────
    print(f"\n\n{'='*W}")
    print("  CROSS-MARKET SUMMARY (mean across markets)")
    print(f"{'='*W}")
    metric_cols = ["gross_sr", "net_sr_best", "net_sr_base", "net_sr_worst",
                   "dv01_pd", "trades_tenor_yr", "n_legs"]
    summary = (df.groupby(["signal", "period"])[metric_cols]
                 .mean()
                 .round(2))
    print(summary.to_string())

    # ── TC degradation summary ────────────────────────────────────────────────
    print(f"\n\n{'='*W}")
    print("  TC DEGRADATION  (gross_sr − net_sr, mean across markets)")
    print(f"{'='*W}")
    for period in ["IS", "OOS"]:
        sub = df[df.period == period].copy()
        sub["deg_best"]  = sub["gross_sr"] - sub["net_sr_best"]
        sub["deg_base"]  = sub["gross_sr"] - sub["net_sr_base"]
        sub["deg_worst"] = sub["gross_sr"] - sub["net_sr_worst"]
        agg = sub.groupby("signal")[["deg_best", "deg_base", "deg_worst"]].mean().round(3)
        print(f"\n  {period}:")
        print(agg.to_string())


if __name__ == "__main__":
    main()
