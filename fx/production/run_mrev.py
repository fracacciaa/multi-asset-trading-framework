"""
Mean-reversion signal pipeline: Family R (short-term reversal) and Family V (PPP value).

Build order (spec §10):
  1. past_return + carry-contamination correlation check
  2. R1 and R2 unconditional
  3. §8 economic significance gate (before inspecting Sharpe)
  4. Conditional masks (large-move, vol regime)
  5. Family V: nominal-only proxy if CPI unavailable, else full PPP model
  6. Value + carry combinations
  7. Write outputs to signals/mrev/

Outputs (signals/mrev/):
  returns_daily.parquet      daily spot return, total return, trailing vol
  signal_r1_xs.parquet       R1 cross-sectional reversal weights by (k, h, mode)
  signal_r2_ts.parquet       R2 time-series positions by (k, h)
  signal_r_conditional.parquet  R1 conditioned on large-move and vol regime
  signal_v_ppp.parquet       q, q_bar, HL and 2HL signals, value weights
  signal_c_valcarry.parquet  C1, C2, C3 value+carry combinations
  diagnostics.parquet        edge_bp, turnover, breakeven_bp per spec and pair
  manifest_mrev.json         CPI source, nominal_only flag, cost assumptions, etc.
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
from fx.production.signal_mrev_r import build_signal_r, trailing_vol
from fx.production.signal_mrev_v import build_signal_v, G10_ONLY
from fx.production.signal_d import carry_weights

OUTPUT_DIR = Path(__file__).parent.parent / "signals" / "mrev"


def run_mrev(
    pairs: list[str] | None = None,
    start: str = PRIMARY_START,
    end: str | None = None,
    rpi: pd.DataFrame | None = None,
    output_dir: Path | None = None,
    path: str | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run the mean-reversion signal pipeline.

    Parameters
    ----------
    pairs      Currency pairs (default: all 20 CARRY_PAIRS).
    start      Sample start date.
    end        Sample end date (None = latest available).
    rpi        Log price ratio log(p_f/p_us) as date × pair monthly DataFrame.
               None triggers Route 3 (nominal-only, G10 only).
    output_dir Output directory (default: fx/signals/mrev/).
    path       Path to the Bloomberg Excel file.
    verbose    Print progress and diagnostics.

    Returns
    -------
    dict with panels, signal_r, signal_v keys.
    """
    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if pairs is None:
        pairs = CARRY_PAIRS

    def _log(msg: str):
        if verbose:
            print(msg, flush=True)

    # ── Step 1: load data panels ──────────────────────────────────────────────
    _log("Loading data panels...")
    load_kw = {} if path is None else {"path": path}
    panels = build_panels(pairs=pairs, start=start, end=end, **load_kw)

    if panels["excluded"]:
        _log(f"  Excluded: {panels['excluded']}")

    s_m    = panels["s"]        # log spot, month-end × pair
    fd_m   = panels["fd_1m"]
    rx_d   = panels["rx_d"]     # daily excess returns
    atm_m  = panels["atm_1m"]
    rr25_m = panels["rr25_1m"]

    # daily log spot (same underlying as s_m but daily)
    # build_panels stores raw daily in rx_d derivation; reconstruct from the
    # forward-discount and the daily rx_d path
    # The production data.py computes rx_d from s_d.diff() internally;
    # we need the daily log spot separately.
    # Reconstruct: s_d = s_m reindexed daily by forward-fill then adjust by rx_d
    # Actually we can get daily s from rx_d:  rx_d = fd_prev/D - Δs  → Δs = fd_prev/D - rx_d
    # This is algebraically correct but error-prone.  Use the simpler route:
    # the daily spot can be recovered from rx_d and fd, or we just resample s_m
    # to business daily for the R signals.  For reversal we need *daily* moves,
    # so we recompute spot from scratch using the raw loader.
    _log("  Rebuilding daily spot series for Family R...")
    from fx.data import fetch_spot
    s_d_dict: dict[str, pd.Series] = {}
    for pair in s_m.columns:
        try:
            raw = fetch_spot(pair, start=start, end=end, **load_kw)
            if raw.empty:
                continue
            from fx.production.data import _to_fcu_per_usd
            fcu = _to_fcu_per_usd(pair, raw)
            fcu = fcu[fcu > 0]
            s_d_dict[pair] = np.log(fcu)
        except Exception as e:
            warnings.warn(f"{pair}: {e}")

    if not s_d_dict:
        _log("  No daily spot data available; Family R skipped.")
        s_d = pd.DataFrame()
    else:
        s_d = pd.DataFrame(s_d_dict).sort_index()

    _log(f"  s_d shape: {s_d.shape}, range: {s_d.index[0]} – {s_d.index[-1]}" if not s_d.empty else "  s_d empty")

    # ── Step 2: §2.1 carry contamination check ────────────────────────────────
    _log("Carry contamination check (total-ranked R1 vs spot-ranked R1)...")

    # base carry weights for the contamination correlation diagnostic
    carry_w_m = carry_weights(fd_m, n=4)

    # ── Step 3: build Family R ────────────────────────────────────────────────
    _log("Building Family R signals (R1, R2, conditional)...")
    if s_d.empty:
        _log("  Skipped (no daily spot).")
        sig_r: dict = {}
    else:
        sig_r = build_signal_r(
            s_d=s_d,
            fd_1m_m=fd_m,
            rx_d=rx_d,
            atm_1m_m=atm_m if not atm_m.empty else None,
            rr25_1m_m=rr25_m if not rr25_m.empty else None,
            carry_w_m=carry_w_m,
            tc_bp=TC_BP,
        )
        cc = sig_r.get("carry_contamination_corr")
        _log(f"  spot-vs-total R1 corr (k=10): {cc:.3f}" if cc is not None else "  carry contamination corr unavailable")
        tvc = sig_r.get("total_vs_carry_corr")
        _log(f"  total-R1 vs carry corr: {tvc:.3f}" if tvc is not None else "")

        # §8 breakeven table for primary specs
        diag = sig_r.get("diagnostics", {})
        for key in ["r1_spot_k5_h5", "r1_spot_k10_h5", "r1_spot_k20_h20",
                    "r2_k5_h5", "r2_k10_h5"]:
            if key in diag:
                d = diag[key]
                _log(f"  {key}: edge={d['edge_bp']:.1f}bp  breakeven={d['breakeven_bp']:.1f}bp")

    # ── Step 4: build Family V ────────────────────────────────────────────────
    _log("Building Family V signals (PPP value)...")
    nominal_only = rpi is None
    if nominal_only:
        _log("  CPI not provided → nominal-only proxy (Route 3, G10 only)")

    sig_v = build_signal_v(
        s_m=s_m,
        fd_m=fd_m,
        rpi=rpi,
        carry_w_m=carry_w_m,
        halflife_grid=(2.0, 3.0, 4.0, 5.0),
        h_primary=12,
        n=4,
    )

    if "regressions" in sig_v:
        reg = sig_v["regressions"]
        _log("  PPP regression α1 (expect ≈ -0.12 at h=6, ≈ -0.52 at h=24):")
        for k, v in sorted(reg.items()):
            _log(f"    {k}: {v:.3f}")

    # ── Step 5: write outputs ─────────────────────────────────────────────────
    _log("Writing parquet files to signals/mrev/...")

    def _save(name: str, df: pd.DataFrame):
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            _log(f"  Skipped (empty): {name}")
            return
        p = out_dir / name
        df.to_parquet(p)
        _log(f"  Wrote {name} {df.shape}")

    # returns_daily.parquet
    if not s_d.empty:
        vol_d = trailing_vol(s_d)
        ds_d  = -s_d.diff()    # positive = foreign appreciated
        ret_daily_frames = {"spot_return": ds_d, "trailing_vol": vol_d}
        if rx_d is not None:
            fd_d = fd_m.reindex(s_d.index, method="ffill")
            ret_daily_frames["total_return"] = ds_d + fd_d / 21.0
        _save("returns_daily.parquet", pd.concat(ret_daily_frames, axis=1))

    # signal_r1_xs.parquet — primary specs (spot mode)
    r1_parts = {k: v for k, v in sig_r.items()
                if isinstance(v, pd.DataFrame) and k.startswith("r1_spot_")}
    if r1_parts:
        _save("signal_r1_xs.parquet", pd.concat(r1_parts, axis=1))

    # signal_r2_ts.parquet
    r2_parts = {k: v for k, v in sig_r.items()
                if isinstance(v, pd.DataFrame) and k.startswith("r2_k")}
    if r2_parts:
        _save("signal_r2_ts.parquet", pd.concat(r2_parts, axis=1))

    # signal_r_conditional.parquet
    r_cond = {k: v for k, v in sig_r.items()
              if isinstance(v, pd.DataFrame) and "cond" in k}
    if r_cond:
        _save("signal_r_conditional.parquet", pd.concat(r_cond, axis=1))

    # signal_v_ppp.parquet
    v_frames: dict[str, pd.DataFrame] = {}
    for key in ("q", "q_bar", "misalignment", "hl_primary", "nominal_value_signal"):
        if key in sig_v and isinstance(sig_v[key], pd.DataFrame):
            v_frames[key] = sig_v[key]
    for key in sig_v:
        if key.startswith(("weights_hl_", "weights_2hl_", "weights_nominal_",
                            "hl_", "2hl_")):
            if isinstance(sig_v[key], pd.DataFrame):
                v_frames[key] = sig_v[key]
    if v_frames:
        _save("signal_v_ppp.parquet", pd.concat(v_frames, axis=1))

    # signal_c_valcarry.parquet
    c_frames = {k: v for k in ("c1", "c2", "c3", "c1_nominal", "c2_nominal", "c3_nominal")
                if k in sig_v and isinstance(sig_v[k], pd.DataFrame)
                for v in [sig_v[k]]}
    if c_frames:
        _save("signal_c_valcarry.parquet", pd.concat(c_frames, axis=1))

    # diagnostics.parquet
    diag_rows = []
    for key, d in sig_r.get("diagnostics", {}).items():
        row = {"signal": key, **{k: v for k, v in d.items()
                                  if not isinstance(v, dict)}}
        diag_rows.append(row)
    if diag_rows:
        _save("diagnostics.parquet", pd.DataFrame(diag_rows).set_index("signal"))

    # ── Manifest ──────────────────────────────────────────────────────────────
    manifest = {
        "nominal_only":         nominal_only,
        "nominal_only_universe": sorted(G10_ONLY) if nominal_only else None,
        "cpi_source":           "none — nominal-only proxy applied" if nominal_only
                                else "injected via rpi parameter",
        "cpi_lag_months":       2,
        "halflife_years":       [2.0, 3.0, 4.0, 5.0],
        "min_periods_recursive_mean": 60,
        "r_lookbacks_days":     [5, 10, 20],
        "r_holdings_days":      [5, 20],
        "large_move_thresholds": [1.5, 2.0, 2.5],
        "vol_window_days":      60,
        "carry_contamination_corr": sig_r.get("carry_contamination_corr"),
        "total_vs_carry_corr":  sig_r.get("total_vs_carry_corr"),
        "mask_hit_rates":       sig_r.get("mask_hit_rate", {}),
        "tc_bp_placeholders":   TC_BP,
        "excluded_ndf":         sorted(NO_FWD_PAIRS),
        "primary_start":        start,
        "option_start":         OPTION_START,
        "note_no_bid_ask":      (
            "PX_LAST only. Breakeven_bp uses flat TC_BP placeholders, not estimates. "
            "Check whether pairs carrying reversal results are the wide-spread ones."
        ),
    }
    manifest_path = out_dir / "manifest_mrev.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    _log("  Wrote manifest_mrev.json")
    _log("Done.")

    return {
        "panels":   panels,
        "signal_r": sig_r,
        "signal_v": sig_v,
        "manifest": manifest,
    }


if __name__ == "__main__":
    run_mrev(verbose=True)
