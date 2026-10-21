"""
FX signal pipeline: builds all five signals and writes output to signals/.

Output (date × pair parquet files):
  returns.parquet       s, fd_1m, rx_m, rx_d, surface columns
  factors.parquet       carry/dollar monthly+daily, carry_weights
  decomposition.parquet b_carry, b_dollar, systematic, idiosyncratic
  signal_a_menkhoff.parquet  weights by (f, h)
  signal_b_zhang.parquet     sys_mom, idio_mom, factor_mom_carry, factor_mom_dollar
  signal_c_tsmom.parquet     vol-scaled and equal-weight by k
  signal_d_dupuy.parquet     E, MV, signal, weights for all three modes
  signal_e_bnp.parquet       E1, E2 (rr/bf/rr_bf), E3

Plus manifest.json.

Build order (spec §10):
  1. data + orientation → 2. returns → 3. Signal A → 4. Signal C →
  5. factors + Signal D → 6. Signal E → 7. Signal B → 8. no-lookahead tests
"""

from __future__ import annotations

import json
import os
import sys
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
    build_signal_a_grid,
    portfolio_returns,
)
from fx.production.signal_b import build_signal_b
from fx.production.signal_c import build_signal_c_grid
from fx.production.signal_d import build_signal_d
from fx.production.signal_e import build_signal_e, build_option_moment_signals

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

    # ── Step 2: Signal A ───────────────────────────────────────────────────────
    _log("Signal A: Menkhoff cross-sectional momentum...")
    sig_a = build_signal_a_grid(rx_m)

    # monotonicity check (replication diagnostic)
    port_ret = portfolio_returns(rx_m, f=6)
    means = port_ret.mean()
    _log(f"  Portfolio means (f=6): {means.to_dict()}")
    monotone = all(means[i+1] >= means[i] - 1e-6 for i in range(1, 5) if i in means and i+1 in means)
    _log(f"  Monotone P1→P6: {monotone}")

    # ── Step 3: Signal C ───────────────────────────────────────────────────────
    _log("Signal C: MOP TSMOM...")
    sig_c = build_signal_c_grid(rx_m, rx_d)

    # ── Step 4: Signal D ───────────────────────────────────────────────────────
    _log("Signal D: Dupuy managed carry...")
    sig_d = build_signal_d(fd_m, rx_d)
    _log(f"  MV floor hit rate: {sig_d['MV_floor_pct']:.1%}")

    # ── Step 5: Signal E ───────────────────────────────────────────────────────
    _log("Signal E: BNP crash-risk conditioned carry...")
    # restrict to option-surface start
    opt_mask = s_m.index >= OPTION_START
    sig_e = build_signal_e(
        fd_m=fd_m[opt_mask],
        rr25_m=rr25_m[opt_mask] if not rr25_m.empty else rr25_m,
        rr10_m=rr10_m[opt_mask] if not rr10_m.empty else rr10_m,
        bf25_m=bf25_m[opt_mask] if not bf25_m.empty else bf25_m,
        bf10_m=bf10_m[opt_mask] if not bf10_m.empty else bf10_m,
        atm_m=atm_m[opt_mask] if not atm_m.empty else atm_m,
        s_m=s_m[opt_mask],
    )
    _log(f"  fd/RR cross-sectional corr: {sig_e.get('cs_corr_fd_rr', float('nan')):.3f}")

    sig_e_ext = {}
    if not rr25_m.empty:
        sig_e_ext = build_option_moment_signals(
            fd_m[opt_mask], atm_m[opt_mask], rr25_m[opt_mask],
            bf25_m[opt_mask], bf10_m[opt_mask],
        )

    # ── Step 6: Signal B ───────────────────────────────────────────────────────
    _log("Signal B: Zhang factor decomposition + factor momentum...")
    sig_b = build_signal_b(rx_m, rx_d, fd_m)
    _log(f"  Conditional means: {sig_b['conditional_means']}")

    # ── Write outputs ──────────────────────────────────────────────────────────
    _log("Writing parquet files...")

    def _save(name: str, df: pd.DataFrame):
        if df.empty:
            _log(f"  Skipped (empty): {name}")
            return
        p = out_dir / name
        df.to_parquet(p)
        _log(f"  Wrote {name} {df.shape}")

    # returns.parquet — long format with surface
    ret_frames = {
        "s":      s_m,
        "fd_1m":  fd_m,
        "rx_m":   rx_m,
    }
    if not rr25_m.empty:
        ret_frames["rr25_1m"] = rr25_m
        ret_frames["rr10_1m"] = rr10_m
        ret_frames["bf25_1m"] = bf25_m
        ret_frames["bf10_1m"] = bf10_m
        ret_frames["atm_1m"]  = atm_m

    returns_out = pd.concat(ret_frames, axis=1)
    _save("returns.parquet", returns_out)

    # rx_d saved separately (daily, large)
    _save("rx_d.parquet", rx_d)

    # factors.parquet
    factors_out = pd.concat({
        "carry_m":    sig_b["carry_m"].rename("carry_m"),
        "dollar_m":   sig_b["dollar_m"].rename("dollar_m"),
        "carry_d":    sig_b["carry_d"].rename("carry_d"),
        "dollar_d":   sig_b["dollar_d"].rename("dollar_d"),
    }, axis=1) if sig_b else pd.DataFrame()
    _save("factors.parquet", factors_out)

    # decomposition.parquet
    decomp_out = pd.concat({
        "b_carry":      sig_b["b_carry"],
        "b_dollar":     sig_b["b_dollar"],
        "systematic":   sig_b["systematic"],
        "idiosyncratic": sig_b["idiosyncratic"],
    }, axis=1)
    _save("decomposition.parquet", decomp_out)

    # signal_a_menkhoff.parquet (f=6, h=1 primary, all on disk)
    a_primary = sig_a[(6, 1)]
    _save("signal_a_menkhoff.parquet", a_primary)
    for (f, h), w in sig_a.items():
        _save(f"signal_a_f{f}_h{h}.parquet", w)

    # signal_c_tsmom.parquet (k=12, h=1 primary)
    c_primary_vs = sig_c[(12, 1)]["vol_scaled"]
    c_primary_ew = sig_c[(12, 1)]["equal_weight"]
    _save("signal_c_tsmom_vol_k12.parquet", c_primary_vs)
    _save("signal_c_tsmom_ew_k12.parquet", c_primary_ew)
    for (k, h), variants in sig_c.items():
        _save(f"signal_c_vs_k{k}_h{h}.parquet", variants["vol_scaled"])
        _save(f"signal_c_ew_k{k}_h{h}.parquet", variants["equal_weight"])

    # signal_d_dupuy.parquet
    d_out = pd.concat({
        "E":               sig_d.get("E", pd.Series(dtype=float)),
        "MV":              sig_d["MV"],
        "signal_E_over_V": sig_d.get("signal_E_over_V", pd.Series(dtype=float)),
    }, axis=1)
    _save("signal_d_dupuy.parquet", d_out)
    for mode in ("E_over_V", "E_only", "V_only"):
        key = f"weights_{mode}"
        if key in sig_d:
            _save(f"signal_d_{mode}_weights.parquet", sig_d[key])

    # signal_e_bnp.parquet
    e_weights = {k: v for k, v in sig_e.items() if isinstance(v, pd.DataFrame) and "weights" in k}
    if e_weights:
        e_out = pd.concat(e_weights, axis=1)
        _save("signal_e_bnp.parquet", e_out)

    # signal_b_zhang.parquet (factor momentum)
    b_parts = {}
    for h in (1, 3, 12):
        key = f"factor_mom_h{h}"
        if key in sig_b:
            b_parts[f"carry_h{h}"]  = sig_b[key]["carry"]
            b_parts[f"dollar_h{h}"] = sig_b[key]["dollar"]
    if b_parts:
        _save("signal_b_zhang.parquet", pd.concat(b_parts, axis=1))

    # ── Manifest ───────────────────────────────────────────────────────────────
    manifest = {
        "fcu_base":            sorted(FCU_BASE),
        "no_fwd_pairs":        sorted(NO_FWD_PAIRS),
        "excluded":            panels["excluded"],
        "divisors":            {k: float(v) for k, v in panels["divisors"].items()},
        "tc_bp_placeholders":  TC_BP,
        "primary_start":       PRIMARY_START,
        "option_start":        OPTION_START,
        "rr_orientation":      "foreign_oriented (FCU call - FCU put)",
        "mid_only_note":       (
            "No bid/ask data. Dupuy attractive-currency filter is vacuous with mid quotes. "
            "TC_BP values are flat placeholders, not estimates."
        ),
        "mv_floor_hit_rate":   sig_d["MV_floor_pct"],
        "cs_corr_fd_rr":       sig_e.get("cs_corr_fd_rr", None),
        "portfolio_monotone":  monotone,
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    _log(f"  Wrote manifest.json")

    _log("Done.")
    return {
        "panels":   panels,
        "signal_a": sig_a,
        "signal_b": sig_b,
        "signal_c": sig_c,
        "signal_d": sig_d,
        "signal_e": sig_e,
        "manifest": manifest,
    }


if __name__ == "__main__":
    run(verbose=True)
