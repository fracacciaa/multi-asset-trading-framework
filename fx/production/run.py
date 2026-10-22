"""
FX signal pipeline.

Monthly outputs (date × pair, ~334 rows):
  returns.parquet              s, fd_1m, rx_m, surface columns
  signal_a_f9_h1.parquet       momentum: XS f=9m h=1 (Menkhoff, best train/test)

Daily outputs (date × pair, ~7241 rows):
  rx_d.parquet
  signal_a_daily.parquet                momentum: EWMA hl selected by IS Sharpe (see manifest), XS P6–P1
  signal_d_E_only_weights_daily.parquet carry: Dupuy E-only leverage
  signal_e_carry_daily.parquet          carry: base XS + E3 aggregate de-risk (columns: base_weights, e3_weights)

Mean-reversion outputs (daily):
  mr_ns_loo_signal.parquet          NS LOO VF TS signal (ns3·loo·grid·vfts·agg), date × pair
  mr_ns_loo_positions.parquet       VF TS positions, date × pair (monthly rebalance)
  mr_ns_loo_positions_daily.parquet XS positions, date × pair (daily rebalance, n=4)
  mr_fly_signal.parquet        Best NS fly (ns3_fixed·smooth·fly_1m3m6m) z-score per pair
  mr_fly_positions.parquet     XS positions per pair for the fly signal

View-formation outputs (prefix view_):
  view_xs_carry_m.parquet / view_ts_carry_m.parquet
  view_xs_carry_daily.parquet / view_ts_carry_daily.parquet

Plus manifest.json.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from fx.production.data import (
    build_panels,
    CARRY_PAIRS,
    PRIMARY_START,
    OPTION_START,
    FCU_BASE,
    NO_FWD_PAIRS,
    TC_BP,
)
from fx.production.signal_a import (
    build_signal_a, build_signal_a_daily, build_signal_a_grid_daily, portfolio_returns,
)
from fx.production.signal_d import build_signal_d_daily
from fx.production.signal_e import build_signal_e_daily
from fx.production.formations import xs_weights, ts_weights
from fx.ns_fwd import (
    build_fwd_panel,
    compute_xs_positions,
    _calibrate_lam,
    build_ns_loo_vfts_signal,
    LAM_GRID,
    _TRAIN_END,
)
from fx.ns_rv import build_rv_signals, compute_halflife_table

OUTPUT_DIR = Path(__file__).parent.parent / "signals"


def run(
    pairs=None,
    start=PRIMARY_START,
    end=None,
    output_dir: Path | None = None,
    path: str | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run the full FX signal pipeline.

    Parameters
    ----------
    pairs      Currency pairs to include (default: all 20 CARRY_PAIRS).
    start      Sample start date.
    end        Sample end date (None = latest available).
    output_dir Directory to write parquet and manifest (None = fx/signals/).
    path       Path to the Bloomberg Excel file (None = default in data.py).
    verbose    Print progress and diagnostic tables.

    Returns
    -------
    dict with all signal DataFrames and diagnostics.
    """
    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if pairs is None:
        pairs = CARRY_PAIRS

    def _log(msg: str):
        if verbose:
            print(msg, flush=True)

    # ── Step 1: data panels ────────────────────────────────────────────────────
    _log("Loading data panels...")
    load_kw = {} if path is None else {"path": path}
    panels = build_panels(pairs=pairs, start=start, end=end, **load_kw)

    if panels["excluded"]:
        _log(f"  Excluded: {panels['excluded']}")

    _log(f"  Divisors:")
    for pair, dv in sorted(panels["divisors"].items()):
        _log(f"    {pair}: {dv:.0f}")

    rx_m   = panels["rx_m"]
    rx_d   = panels["rx_d"]
    fd_m   = panels["fd_1m"]
    s_m    = panels["s"]
    atm_m  = panels["atm_1m"]
    rr25_m = panels["rr25_1m"]
    rr10_m = panels["rr10_1m"]
    bf25_m = panels["bf25_1m"]
    bf10_m = panels["bf10_1m"]

    _log(f"  rx_m shape: {rx_m.shape}, date range: {rx_m.index[0]} – {rx_m.index[-1]}")
    if not atm_m.empty:
        _log(f"  atm_1m start: {atm_m.first_valid_index()}")

    # ── Step 2: Signal A — momentum ───────────────────────────────────────────
    _log("Signal A: XS momentum f=9 h=1...")
    sig_a = build_signal_a(rx_m, f=9, h=1)

    # monotonicity diagnostic (f=6 canonical check)
    port_ret = portfolio_returns(rx_m, f=6)
    means    = port_ret.mean()
    monotone = all(means[i+1] >= means[i] - 1e-6 for i in range(1, 5) if i in means and i+1 in means)
    _log(f"  Monotone P1→P6 (f=6 diagnostic): {monotone}")

    # ── Step 3: daily auxiliary series ────────────────────────────────────────
    _log("Building daily auxiliary series (ffill from month-end)...")
    fd_d   = fd_m.resample("B").ffill()
    rr25_d = rr25_m.resample("B").ffill() if not rr25_m.empty else pd.DataFrame()
    atm_d  = atm_m.resample("B").ffill()  if not atm_m.empty  else pd.DataFrame()
    _log(f"  fd_d shape: {fd_d.shape}")

    # ── Step 6: daily carry signals ───────────────────────────────────────────
    _log("Signal A daily: selecting halflife by IS Sharpe...")
    _hl_grid = build_signal_a_grid_daily(rx_d)
    _hl_is_srs = {}
    for _hl, _w in _hl_grid.items():
        _is = (_w.shift(1) * rx_d).sum(axis=1).loc[:_TRAIN_END].dropna()
        if len(_is) > 60:
            _hl_is_srs[_hl] = float(_is.mean() / _is.std() * np.sqrt(252))
    _best_hl = max(_hl_is_srs, key=_hl_is_srs.get)
    _log(f"  Selected hl={_best_hl}d (IS SR={_hl_is_srs[_best_hl]:.3f})")
    sig_a_d = build_signal_a_daily(rx_d, halflife_days=_best_hl)

    _log("Signal D daily (rolling MV carry)...")
    sig_d_d = build_signal_d_daily(fd_d, rx_d)
    _log(f"  MV floor hit rate (daily): {sig_d_d['MV_floor_pct']:.1%}")

    _log("Signal E daily (crash-risk carry)...")
    opt_mask_d = fd_d.index >= OPTION_START
    sig_e_d = build_signal_e_daily(
        fd_d=fd_d[opt_mask_d],
        rr25_d=rr25_d[opt_mask_d] if not rr25_d.empty else pd.DataFrame(),
        atm_d=atm_d[opt_mask_d]   if not atm_d.empty  else pd.DataFrame(),
    )
    _log(f"  fd/RR cross-sectional corr (daily): {sig_e_d.get('cs_corr_fd_rr', float('nan')):.3f}")

    # ── Write outputs ──────────────────────────────────────────────────────────
    _log("Writing parquet files...")

    def _save(name: str, df: pd.DataFrame):
        if df.empty:
            _log(f"  Skipped (empty): {name}")
            return
        p = out_dir / name
        df.to_parquet(p)
        _log(f"  Wrote {name} {df.shape}")

    # returns
    ret_frames = {"s": s_m, "fd_1m": fd_m, "rx_m": rx_m}
    if not rr25_m.empty:
        ret_frames.update({
            "rr25_1m": rr25_m, "rr10_1m": rr10_m,
            "bf25_1m": bf25_m, "bf10_1m": bf10_m, "atm_1m": atm_m,
        })
    _save("returns.parquet", pd.concat(ret_frames, axis=1))
    _save("rx_d.parquet", rx_d)

    # Signal A monthly
    _save("signal_a_f9_h1.parquet", sig_a)

    # ── Daily outputs ──────────────────────────────────────────────────────────
    _log("Writing daily parquet files...")

    # Signal A daily
    _save("signal_a_daily.parquet", sig_a_d)

    # Signal D daily — E-only only
    if "weights_E_only" in sig_d_d:
        _save("signal_d_E_only_weights_daily.parquet", sig_d_d["weights_E_only"])

    # Signal E daily — base and E3 only
    e_keep = {k: v for k, v in sig_e_d.items()
              if isinstance(v, pd.DataFrame) and k in ("base_weights", "e3_weights")}
    if e_keep:
        _save("signal_e_carry_daily.parquet", pd.concat(e_keep, axis=1))

    # ── Step 7: mean-reversion signals ────────────────────────────────────────
    _log("Mean-reversion: loading multi-tenor forward panel...")
    fwd_panels = build_fwd_panel(pairs=pairs, start=start, end=end,
                                  **({"path": path} if path else {}))

    _log("Mean-reversion: calibrating NS λ (training ≤ 2018)...")
    lam_fixed = _calibrate_lam(fwd_panels, LAM_GRID, k=3)
    hl_table  = compute_halflife_table(fwd_panels, lam_fixed)

    _log("Mean-reversion: NS LOO VF TS signal (ns3·loo·grid·vfts·agg)...")
    sig_ns_loo, pos_ns_loo = build_ns_loo_vfts_signal(fwd_panels, rx_d)
    _save("mr_ns_loo_signal.parquet",    sig_ns_loo)
    _save("mr_ns_loo_positions.parquet", pos_ns_loo)
    pos_ns_loo_daily = compute_xs_positions(sig_ns_loo, n=4)
    _save("mr_ns_loo_positions_daily.parquet", pos_ns_loo_daily)

    _log("Mean-reversion: NS fly signal (ns3_fixed·smooth·fly_1m3m6m)...")
    fly_sigs = build_rv_signals(
        model="ns3_fixed", pairs=pairs, start=start, end=end,
        smooth=True, halflife_table=hl_table, lam_fixed=lam_fixed,
        fwd_panels=fwd_panels,
    )
    fly_raw = fly_sigs.get("fly_1m3m6m", pd.DataFrame())
    if not fly_raw.empty:
        vol_fly = fly_raw.shift(1).ewm(halflife=63, min_periods=30).std()
        fly_z   = -(fly_raw / vol_fly.clip(lower=1e-8)).clip(-3.0, 3.0)
        fly_pos = compute_xs_positions(fly_z, n=4)
        _save("mr_fly_signal.parquet",    fly_z)
        _save("mr_fly_positions.parquet", fly_pos)

    # ── View formations on carry score ────────────────────────────────────────
    _log("Writing carry view formations...")
    _save("view_xs_carry_m.parquet",      xs_weights(fd_m))
    _save("view_ts_carry_m.parquet",      ts_weights(fd_m))
    _save("view_xs_carry_daily.parquet",  xs_weights(fd_d))
    _save("view_ts_carry_daily.parquet",  ts_weights(fd_d))

    # ── Manifest ───────────────────────────────────────────────────────────────
    manifest = {
        "fcu_base":           sorted(FCU_BASE),
        "no_fwd_pairs":       sorted(NO_FWD_PAIRS),
        "excluded":           panels["excluded"],
        "divisors":           {k: float(v) for k, v in panels["divisors"].items()},
        "tc_bp_placeholders": TC_BP,
        "primary_start":      PRIMARY_START,
        "option_start":       OPTION_START,
        "rr_orientation":     "foreign_oriented (FCU call - FCU put)",
        "mid_only_note":      (
            "No bid/ask data. Dupuy attractive-currency filter is vacuous with mid quotes. "
            "TC_BP values are flat placeholders, not estimates."
        ),
        "momentum": {
            "monthly": {"signal": "XS f=9m h=1", "train_sr": 0.51, "test_sr": 0.56},
            "daily":   {"halflife_days": _best_hl,
                        "is_sr": round(_hl_is_srs[_best_hl], 3),
                        "is_sr_by_hl": {k: round(v, 3) for k, v in sorted(_hl_is_srs.items())}},
        },
        "mv_floor_hit_rate_daily":  sig_d_d["MV_floor_pct"],
        "cs_corr_fd_rr":            sig_e_d.get("cs_corr_fd_rr", None),
        "portfolio_monotone":       monotone,
        "daily_carry": {
            "fd_source":        "fd_m forward-filled to business daily",
            "option_source":    "rr25_m, atm_m forward-filled to business daily",
            "dupuy_lag_days":   21,
            "dupuy_mv_window":  21,
            "e3_min_periods":   756,
        },
        "mean_reversion": {
            "ns_loo_model":      "ns3·loo·grid·vfts·agg",
            "ns_loo_rebalance":  "monthly VF TS (desired_vol=10%)",
            "ns_loo_pnl_type":   "rx_d",
            "oos_sr_ns_loo":     0.450,
            "ns_loo_corr_carry": -0.024,
            "fly_model":         "ns3_fixed·smooth·fly_1m3m6m",
            "fly_n_pairs":       4,
            "fly_pnl_type":      "rx_d",
            "oos_sr_fly":        0.366,
        },
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    _log("  Wrote manifest.json")

    _log("Done.")
    return {
        "panels":        panels,
        "signal_a":      sig_a,
        "signal_d_d":    sig_d_d,
        "signal_e_d":    sig_e_d,
        "fwd_panels":       fwd_panels,
        "signal_a_d":           sig_a_d,
        "mr_ns_loo_signal":     sig_ns_loo,
        "mr_ns_loo_pos":        pos_ns_loo,
        "mr_ns_loo_pos_daily":  pos_ns_loo_daily,
        "mr_fly_signal": fly_z   if not fly_raw.empty else pd.DataFrame(),
        "mr_fly_pos":    fly_pos if not fly_raw.empty else pd.DataFrame(),
        "manifest":      manifest,
    }


if __name__ == "__main__":
    run(verbose=True)
