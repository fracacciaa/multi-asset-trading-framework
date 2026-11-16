"""
Rates signal pipeline.

Mirror of fx/production/run.py.  Run once; all downstream Sharpe queries
read parquet instead of re-running NS calibration and fit loops.

Outputs written to rates/signals/:
  ns_factors.parquet          date × (market, factor)       NS level/slope/curvature
  momentum_a_daily.parquet    date × (market, tenor)        MOP TSMOM position sizes
  momentum_a_daily_pnl.parquet date × market                momentum daily PnL, full repricing (bp)
  mean_reversion_pnl.parquet  date × market                 NS classic MR daily PnL, full repricing (bp)
  mean_reversion_zscores.parquet date × (market, tenor)     NS classic MR z-scores (signal + TC proxy)
  carry_ois_par.parquet       date × (market, tenor)        within-curve carry OIS positions
  carry_ois_par_pnl.parquet   date × market                 within-curve carry OIS PnL, full repricing (bp)
  rx_m.parquet                date × (market, tenor)        Sihvonen monthly log excess returns (bp)
  ret_d.parquet               date × (market, tenor)        daily CM returns (bp)
  manifest.json

Sharpe recipe (any *_pnl file):
  pnl = pd.read_parquet("rates/signals/carry_ois_par_pnl.parquet")
  sr  = pnl[market].dropna().mean() / pnl[market].dropna().std() * np.sqrt(252)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from rates.production.nelson_siegel import extract_factors_all_markets
from rates.production.ns_classic.bootstrap import bootstrap_curves
from rates.production.ns_classic.ns_fit import select_lambda
from rates.production.ns_classic.pnl import classic_ns_pnl_repriced
from rates.production.momentum.returns import build_all_return_panels
from rates.production.momentum.signal_a import build_signal_a_daily
from rates.production.momentum.signal_a import DM_MARKETS as MOM_MARKETS
from rates.production.momentum.pnl_repriced import momentum_pnl_all_repriced
from rates.production.carry.carry_ois_par import OIS_PAR_MARKETS
from rates.production.carry.signal import build_positions
from rates.production.carry.carry_ois_par import ois_par_carry_panel, BENCHMARK_TENORS as CARRY_TENORS
from rates.production.carry.pnl_repriced import ois_par_carry_pnl_repriced
from rates.production.ois_data import OIS_MARKETS, fetch_ois_curves
from rates.production.yc_builder import build_zero_curve

OUTPUT_DIR = Path(__file__).parent.parent / "signals"

# NS classic MR constants — λ frozen on this IS training window
_LAM_TRAIN_END = "2015-12-31"
_MR_TARGET_VOL = 100.0   # bp/yr vol target per market
_MR_BURN_IN    = 252     # trading days before signal goes live


def run(
    markets: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    output_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run the full rates signal pipeline and serialize outputs to parquet.

    Parameters
    ----------
    markets    Markets to include (default: all OIS_MARKETS).
    start      Sample start date (None = full history).
    end        Sample end date (None = latest available).
    output_dir Directory for parquet outputs (default: rates/signals/).
    verbose    Print progress.

    Returns
    -------
    dict with all signal DataFrames and manifest.
    """
    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if markets is None:
        markets = OIS_MARKETS

    def _log(msg: str):
        if verbose:
            print(msg, flush=True)

    def _save(name: str, df: pd.DataFrame | pd.Series):
        if isinstance(df, (pd.DataFrame, pd.Series)) and df.empty:
            _log(f"  Skipped (empty): {name}")
            return
        p = out_dir / name
        df.to_parquet(p)
        _log(f"  Wrote {name}  {getattr(df, 'shape', '?')}")

    # ── Step 0: fetch par curves + build zero curves ───────────────────────────
    _log("Zero curves: bootstrapping + interpolating all markets...")
    par_curves:  dict = {}
    zero_curves: dict = {}
    for mkt in markets:
        try:
            raw = fetch_ois_curves(mkt, start=start, end=end)
            par_curves[mkt]  = raw
            zero_curves[mkt] = build_zero_curve(raw)
        except Exception as exc:
            _log(f"  Warning: zero curve failed for {mkt}: {exc}")
    _log(f"  Built zero curves for {len(zero_curves)} markets")

    # ── Step 1: return panels (needed by momentum and PnL computation) ─────────
    _log("Return panels: daily CM + monthly Sihvonen log excess returns...")
    all_panels = build_all_return_panels(
        markets=markets, start=start, end=end, zero_curves=zero_curves,
    )
    _log(f"  {len(all_panels)} markets loaded")

    rx_m_all = pd.concat(
        {m: p["rx_m"] for m, p in all_panels.items() if p.get("rx_m") is not None},
        axis=1,
    )
    rx_m_all.columns.names = ["market", "tenor"]
    _save("rx_m.parquet", rx_m_all)

    ret_d_all = pd.concat(
        {m: p["ret_d"] for m, p in all_panels.items() if p.get("ret_d") is not None},
        axis=1,
    )
    ret_d_all.columns.names = ["market", "tenor"]
    _save("ret_d.parquet", ret_d_all)

    # ── Step 2: Momentum (daily TSMOM) ────────────────────────────────────────
    _log("Momentum Signal A-daily: daily TSMOM positions (252-day window)...")
    mom_panels  = {m: all_panels[m] for m in MOM_MARKETS if m in all_panels}
    mom_a_daily = build_signal_a_daily(mom_panels)
    _save("momentum_a_daily.parquet", mom_a_daily)
    if not mom_a_daily.empty:
        _log(f"  date range: {mom_a_daily.index[0].date()} – {mom_a_daily.index[-1].date()}")

    _log("Momentum Signal A-daily PnL: full swap repricing (pos × swap_pv × 10000)...")
    mom_a_daily_pnl = momentum_pnl_all_repriced(
        markets=list(mom_panels.keys()), formation=252, start=start, end=end,
    )
    _save("momentum_a_daily_pnl.parquet", mom_a_daily_pnl)

    # ── Step 3: NS factors ─────────────────────────────────────────────────────
    _log("NS factors: calibrating λ + batch-extracting level/slope/curvature...")
    ns_factors = extract_factors_all_markets(
        markets=markets, start=start, end=end, model="ns_calibrated",
        zero_curves=zero_curves,
    )
    _save("ns_factors.parquet", ns_factors)
    _log(f"  shape: {ns_factors.shape}")

    # ── Step 4: Within-curve carry OIS ────────────────────────────────────────
    _log("Within-curve carry OIS: positions (5Y + 10Y, daily rebalance, lag=1)...")
    carry_mkts = [m for m in OIS_PAR_MARKETS if m in markets]

    carry_pos_frames: dict = {}
    for mkt in carry_mkts:
        try:
            panel = ois_par_carry_panel(mkt, tenors=CARRY_TENORS,
                                        horizons={"3M": 0.25},
                                        start=start, end=end)
            if panel.empty:
                continue
            sub        = panel.xs("3M", level="horizon")
            carry_wide = sub["total_carry_bp"].unstack("tenor")
            dmod_wide  = sub["dmod"].unstack("tenor")
            pos = build_positions(carry_wide, dmod_wide,
                                  rebalance="daily", lag=1,
                                  xs_mode="z", neutralize=("level",),
                                  gross=1.0, duration_adj="per_dur")
            carry_pos_frames[mkt] = pos
        except Exception as exc:
            _log(f"  Warning: carry position failed for {mkt}: {exc}")

    if carry_pos_frames:
        carry_pos = pd.concat(carry_pos_frames, axis=1)
        carry_pos.columns.names = ["market", "tenor"]
    else:
        carry_pos = pd.DataFrame()
    _save("carry_ois_par.parquet", carry_pos)

    _log("Within-curve carry OIS PnL: full swap repricing (pos × swap_pv × 10000)...")
    carry_pnl = ois_par_carry_pnl_repriced(
        markets=carry_mkts, tenors=CARRY_TENORS, start=start, end=end,
    )
    _save("carry_ois_par_pnl.parquet", carry_pnl)
    if not carry_pnl.empty:
        _log(f"  {len(carry_pnl)} days, {len(carry_pnl.columns)} markets")

    # ── Step 5: Mean reversion — NS classic MR, full swap repricing ───────────
    # λ selected once on data through _LAM_TRAIN_END then frozen.
    # Falls back to λ=2.5 if selection fails (robust default near 5Y curvature peak).
    _log(f"NS classic MR: λ selection (train end {_LAM_TRAIN_END}) + full swap repricing...")
    mr_pnl_dict:    dict[str, pd.Series]    = {}
    mr_zscore_dict: dict[str, pd.DataFrame] = {}

    for mkt in markets:
        if mkt not in par_curves:
            continue
        try:
            par_mkt   = par_curves[mkt]
            zeros_mkt = bootstrap_curves(par_mkt)
            try:
                lr  = select_lambda(zeros_mkt, end=_LAM_TRAIN_END)
                lam = lr["best_lambda"]
            except Exception:
                lam = 2.5
            pnl, z_scores, _ = classic_ns_pnl_repriced(
                par_mkt, lam,
                target_ann_vol=_MR_TARGET_VOL,
                burn_in=_MR_BURN_IN,
            )
            mr_pnl_dict[mkt]    = pnl
            mr_zscore_dict[mkt] = z_scores
            _log(f"  {mkt}: λ={lam}  n={pnl.dropna().__len__()}")
        except Exception as exc:
            _log(f"  Warning: NS classic MR failed for {mkt}: {exc}")

    mr_pnl = pd.DataFrame(mr_pnl_dict)
    mr_pnl.index.name = "date"
    _save("mean_reversion_pnl.parquet", mr_pnl)

    # z_scores saved as MultiIndex (market, tenor) — used as TC position proxy
    if mr_zscore_dict:
        mr_zscores = pd.concat(mr_zscore_dict, axis=1)
        mr_zscores.columns.names = ["market", "tenor"]
    else:
        mr_zscores = pd.DataFrame()
    _save("mean_reversion_zscores.parquet", mr_zscores)
    # backward-compat alias read by rates/blend.py for TC estimation
    _save("mean_reversion_dv01_pos.parquet", mr_zscores)

    # ── Manifest ───────────────────────────────────────────────────────────────
    manifest = {
        "markets":     sorted(markets),
        "carry_markets": sorted(carry_mkts),
        "mom_markets": sorted(mom_panels.keys()),
        "mr_markets":  sorted(mr_pnl_dict.keys()),
        "start":       str(start) if start else None,
        "end":         str(end) if end else None,
        "ns_model":    "ns_calibrated",
        "carry_tenors": CARRY_TENORS,
        "mr_lam_train_end": _LAM_TRAIN_END,
        "signals": [
            "momentum_a_daily",
            "carry_within_curve_ois",
            "mean_reversion_ns_classic",
        ],
        "files": {
            "ns_factors":               "date × (market,factor) MultiIndex cols; level/slope/curvature",
            "carry_ois_par":            "date × (market,tenor) MultiIndex cols; within-curve carry OIS positions",
            "carry_ois_par_pnl":        "date × market; full repricing pos × swap_pv × 10000 — use for Sharpe (daily, √252)",
            "mean_reversion_pnl":       "date × market; NS classic MR full repricing — use for Sharpe (daily, √252)",
            "mean_reversion_zscores":   "date × (market,tenor) MultiIndex; NS classic z-scores — signal + TC proxy",
            "mean_reversion_dv01_pos":  "alias for mean_reversion_zscores.parquet (backward compat)",
            "momentum_a_daily":         "date × (market,tenor); daily TSMOM positions (252d window) — signal only",
            "momentum_a_daily_pnl":     "date × market; full repricing pos × swap_pv × 10000 — use for Sharpe (daily, √252)",
            "rx_m":                     "date × (market,tenor); Sihvonen monthly log excess returns (bp)",
            "ret_d":                    "date × (market,tenor); daily CM constant-maturity returns (bp)",
        },
        "sharpe_recipe": (
            "Always use the *_pnl files. "
            "sr = pnl[market].dropna().mean() / pnl[market].dropna().std() * sqrt(252). "
            "carry_ois_par_pnl: full swap repricing, DV01-neutral 5Y+10Y cross-section, daily rebalance. "
            "mean_reversion_pnl: NS classic MR full repricing, factor-neutral NS residuals, "
            "λ frozen on data through 2015-12-31, 100 bp/yr vol target per market. "
            "momentum_a_daily_pnl: full swap repricing, per-tenor vol-scaled positions, 252d formation."
        ),
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    _log("  Wrote manifest.json")

    _log("Done.")
    return {
        "ns_factors":              ns_factors,
        "carry_ois_par":           carry_pos,
        "carry_ois_par_pnl":       carry_pnl,
        "mean_reversion_pnl":      mr_pnl,
        "mean_reversion_zscores":  mr_zscores,
        "momentum_a_daily":        mom_a_daily,
        "momentum_a_daily_pnl":    mom_a_daily_pnl,
        "rx_m":                    rx_m_all,
        "ret_d":                   ret_d_all,
        "all_panels":              all_panels,
        "manifest":                manifest,
    }


if __name__ == "__main__":
    run(verbose=True)
