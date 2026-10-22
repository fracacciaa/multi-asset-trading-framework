"""
Rates signal pipeline.

Mirror of fx/production/run.py.  Run once; all downstream Sharpe queries
read parquet instead of re-running NS calibration and fit loops.

Outputs written to rates/signals/:
  ns_factors.parquet     date × (market, factor)  NS level/slope/curvature
  carry.parquet          date × market             within-curve carry (−β₂)
  mean_reversion.parquet date × market             factor-neutral NS residual z-score
  carry_cc.parquet       date × market             cross-country carry positions
  momentum_a.parquet     date × (market, tenor)    MOP TSMOM position sizes
  rx_m.parquet           date × (market, tenor)    Sihvonen monthly log excess returns (bp)
  ret_d.parquet          date × (market, tenor)    daily CM returns (bp)
  ois_carry.parquet      (market,date,tenor,horiz) OIS static carry panel (long format)
  manifest.json

Sharpe recipe (any signal):
  df = pd.read_parquet("rates/signals/rx_m.parquet")
  sig = pd.read_parquet("rates/signals/carry.parquet")
  pnl = sig.shift(1).reindex(df.index, method="ffill") * df[("US", 10.0)]
  sr  = pnl.mean() / pnl.std() * np.sqrt(252)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from rates.production.nelson_siegel import extract_factors_all_markets
from rates.production.signals import carry, carry_cross_country
from rates.production.mean_reversion.signal import mean_reversion_rv, mean_reversion_pnl
from rates.production.carry.experiment import within_curve_carry_pnl
from rates.production.momentum.returns import build_all_return_panels
from rates.production.momentum.signal_a import build_signal_a_daily
from rates.production.momentum.signal_a import DM_MARKETS as MOM_MARKETS
from rates.production.carry.carry_ois import ois_carry_all_markets
from rates.production.carry.cross_country import DM_MARKETS as CC_DM_MARKETS
from rates.production.ois_data import OIS_MARKETS, fetch_ois_curves
from rates.production.yc_builder import build_zero_curve

OUTPUT_DIR = Path(__file__).parent.parent / "signals"


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

    # ── Step 0: build zero curves once (reused by all downstream steps) ────────
    _log("Zero curves: bootstrapping + interpolating all markets...")
    zero_curves: dict = {}
    for mkt in markets:
        try:
            raw = fetch_ois_curves(mkt, start=start, end=end)
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

    _log("Momentum Signal A-daily PnL: pos(t-1) × ret_d(t), no forward-fill...")
    if not mom_a_daily.empty and not ret_d_all.empty:
        common_cols = mom_a_daily.columns.intersection(ret_d_all.columns)
        common_idx  = mom_a_daily.index.intersection(ret_d_all.index)
        pos_shifted = mom_a_daily.loc[common_idx, common_cols].shift(1)
        mom_a_daily_pnl = (
            pos_shifted * ret_d_all.loc[common_idx, common_cols]
        ).T.groupby(level="market").sum().T
        mom_a_daily_pnl.index.name = "date"
    else:
        mom_a_daily_pnl = pd.DataFrame()
    _save("momentum_a_daily_pnl.parquet", mom_a_daily_pnl)

    # ── Step 3: NS factors ─────────────────────────────────────────────────────
    _log("NS factors: calibrating λ + batch-extracting level/slope/curvature...")
    ns_factors = extract_factors_all_markets(
        markets=markets, start=start, end=end, model="ns_calibrated",
        zero_curves=zero_curves,
    )
    _save("ns_factors.parquet", ns_factors)
    _log(f"  shape: {ns_factors.shape}")

    # ── Step 4: Within-curve carry (−β₂) ──────────────────────────────────────
    _log("Carry: within-curve NS slope signal...")
    carry_sig = carry(ns_factors)
    _save("carry.parquet", carry_sig)

    _log("Carry PnL: per-tenor DV01-neutral positions × cm_total_return_bp...")
    carry_pnl = within_curve_carry_pnl(markets=markets, start=start, end=end)
    _save("carry_pnl.parquet", carry_pnl)

    # ── Step 5: Mean reversion ─────────────────────────────────────────────────
    _log("Mean reversion: factor-neutral NS residuals...")
    mr_sig = mean_reversion_rv(markets=markets, start=start, end=end,
                               zero_curves=zero_curves)
    _save("mean_reversion.parquet", mr_sig)

    _log("Mean reversion PnL: per-tenor pos(t-1) × −Δresid(t)...")
    mr_pnl = mean_reversion_pnl(markets=markets, start=start, end=end,
                                zero_curves=zero_curves)
    _save("mean_reversion_pnl.parquet", mr_pnl)

    # ── Step 6: Cross-country carry ────────────────────────────────────────────
    _log("Cross-country carry: DV01-neutral positions (DM only)...")
    cc_mkts = [m for m in CC_DM_MARKETS if m in markets]
    cc_sig = carry_cross_country(markets=cc_mkts, start=start, end=end)
    _save("carry_cc.parquet", cc_sig)

    _log("Cross-country carry PnL: per-market pos × 10Y CM return...")
    if not cc_sig.empty and not ret_d_all.empty:
        cc_ret_10y: dict[str, pd.Series] = {}
        mkt_level = ret_d_all.columns.get_level_values("market")
        for mkt in cc_sig.columns:
            if mkt not in mkt_level:
                continue
            mkt_tenors = ret_d_all[mkt].columns.to_numpy(float)
            closest    = float(mkt_tenors[np.argmin(np.abs(mkt_tenors - 10.0))])
            cc_ret_10y[mkt] = ret_d_all[(mkt, closest)]
        if cc_ret_10y:
            cc_ret_df  = pd.DataFrame(cc_ret_10y)
            common_idx = cc_sig.index.intersection(cc_ret_df.index)
            # cc_sig already has lag=1 applied in carry_cross_country
            cc_pnl = cc_sig.loc[common_idx] * cc_ret_df.loc[common_idx]
            cc_pnl.index.name = "date"
        else:
            cc_pnl = pd.DataFrame()
    else:
        cc_pnl = pd.DataFrame()
    _save("carry_cc_pnl.parquet", cc_pnl)

    # ── Step 7: OIS carry panel ────────────────────────────────────────────────
    _log("OIS carry: static carry panel (long format)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ois_carry = ois_carry_all_markets(
            markets=markets, start=start, end=end, zero_curves=zero_curves,
        )
    _save("ois_carry.parquet", ois_carry)
    if not ois_carry.empty:
        _log(f"  {len(ois_carry):,} rows, {ois_carry.index.get_level_values('market').nunique()} markets")

    # ── Manifest ───────────────────────────────────────────────────────────────
    manifest = {
        "markets":     sorted(markets),
        "cc_markets":  sorted(cc_mkts),
        "mom_markets": sorted(mom_panels.keys()),
        "start":       str(start) if start else None,
        "end":         str(end) if end else None,
        "ns_model":    "ns_calibrated",
        "carry_cc_lam": 0.3,
        "signals": [
            "mean_reversion",
            "momentum_a_daily",
        ],
        "files": {
            "ns_factors":           "date × (market,factor) MultiIndex cols; level/slope/curvature",
            "carry":                "date × market; within-curve carry scalar (−β₂) — signal only",
            "carry_pnl":            "date × market; per-tenor DV01-neutral pos × cm_total_return_bp — use for Sharpe (daily, √252)",
            "mean_reversion":       "date × market; factor-neutral NS residual z-score — signal only",
            "mean_reversion_pnl":   "date × market; per-tenor pos(t-1)×(−Δresid(t)) summed — use for Sharpe (daily, √252)",
            "carry_cc":             "date × market; cross-country carry positions — signal only",
            "carry_cc_pnl":         "date × market; cc pos(t-1)×10Y ret_d(t) — use for Sharpe (daily, √252)",
            "momentum_a_daily":     "date × (market,tenor); daily TSMOM positions (252d window) — signal only",
            "momentum_a_daily_pnl": "date × market; daily pos(t-1)×ret_d(t) summed — use for Sharpe (daily, √252)",
            "rx_m":                 "date × (market,tenor); Sihvonen monthly log excess returns (bp)",
            "ret_d":                "date × (market,tenor); daily CM constant-maturity returns (bp)",
            "ois_carry":            "long format (market,date,tenor,horizon); static carry (bp)",
        },
        "sharpe_recipe": (
            "Always use the *_pnl files. "
            "sr = pnl[market].dropna().mean() / pnl[market].dropna().std() * sqrt(252). "
            "mean_reversion_pnl: per-tenor z-score × (−Δresid), summed. "
            "momentum_a_daily_pnl: per-tenor vol-scaled positions × daily CM returns, summed."
        ),
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    _log("  Wrote manifest.json")

    _log("Done.")
    return {
        "ns_factors":         ns_factors,
        "carry":              carry_sig,
        "carry_pnl":          carry_pnl,
        "mean_reversion":     mr_sig,
        "mean_reversion_pnl": mr_pnl,
        "carry_cc":             cc_sig,
        "carry_cc_pnl":         cc_pnl,
        "momentum_a_daily":     mom_a_daily,
        "momentum_a_daily_pnl": mom_a_daily_pnl,
        "rx_m":               rx_m_all,
        "ret_d":              ret_d_all,
        "ois_carry":          ois_carry,
        "all_panels":         all_panels,
        "manifest":           manifest,
    }


if __name__ == "__main__":
    run(verbose=True)
