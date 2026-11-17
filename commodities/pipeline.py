"""
End-to-end commodities pipeline with IS / OOS reporting (daily, Sharpe √252).

run_commodities_pipeline() (1) builds the per-instrument signal panels (and
optionally rewrites the signal parquets), then prints Sharpe, MDD and turnover
for IS (≤ blend.IS_END) and OOS (≥ blend.OOS_START) at four levels:

  1. markets     per signal × per instrument
  2. signals     inverse-vol cross-instrument blend of each signal
  3. composites  static-Ridge (IS-fit) blends of every subset (≥2) of the signals
  4. TC-aware    tc_sweep_comm scenarios (best / base / worst), both methods

Conventions: see core/pipeline_report.py.  Turnover is instrument-level on the
weekly-rebalanced vol-scaled weights, scaled by the inv-vol instrument
multipliers (and Ridge weights for composites).  The TC table is different in
kind: tc_sweep_comm applies a uniform annual cost drag per scenario and reports
blend-level (signal-weight) turnover, not instrument-level.
"""

from __future__ import annotations

import contextlib
import io
import itertools
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from core import pipeline_report as R
from commodities import blend as _b


def run_commodities_pipeline(
    rebuild_signals: bool = True,
    signals_dir: Optional[str | Path] = None,
    params: Optional[dict] = None,
    lam: float = 0.01,
    lookback: int = 252,
    min_obs: int = 252,
    vol_target: float = 0.10,
    scenarios: Optional[dict] = None,
) -> dict:
    """
    Run the whole commodities pipeline and print IS/OOS Sharpe, MDD, turnover
    for instruments, signals, signal composites and the TC-aware scenarios.

    Parameters
    ----------
    rebuild_signals  True rewrites the signal parquets in signals_dir from the
                     fresh panels; False reads the existing parquets for the
                     signal / composite / TC stages (the per-instrument panels
                     are always recomputed — they are needed for stages 1-3 turnover).
    signals_dir      Parquet directory (default commodities/signals/).
    params           Pipeline parameters (default blend.BLEND_PARAMS).
    lam              Ridge penalty for the static-Ridge composites.
    lookback, min_obs  Rolling-Ridge settings for the TC sweep's adaptive method.
    vol_target       Annual vol used for MDD scaling.
    scenarios        TC scenarios (default blend.TC_SCENARIOS_COMM: best/base/worst).

    Returns
    -------
    dict with DataFrames "markets" (MultiIndex signal × instrument), "signals",
    "composites", "tc" (tc_sweep_comm metrics), "tc_returns".
    """
    sdir = Path(signals_dir) if signals_dir else _b._SIGNALS_DIR
    cfg = R.ReportCfg(ann=252, is_end=_b.IS_END, oos_start=_b.OOS_START,
                      vol_target=vol_target, min_obs=30)

    # ── stage 0: panels (+ parquets) ──────────────────────────────────────────
    print("Stage 0 — building per-instrument signal panels...")
    panels = _b._signal_panels(params)
    if rebuild_signals:
        print("  rewriting signal parquets...")
        _b.build_signal_parquets(output_dir=sdir, params=params, panels=panels)
    else:
        print(f"  reusing parquets in {sdir}")

    # ── stage 1: instruments ──────────────────────────────────────────────────
    market_rows: dict[tuple, dict] = {}
    R.header("STAGE 1 — INSTRUMENTS (per signal)", 22, cfg.sub)
    for sig, (w, F_i) in panels.items():
        print(f"  [{sig}]")
        for inst in F_i.columns:
            r = F_i[inst].dropna()
            sigma = R.ann_sigma(r, cfg)
            if len(r) == 0 or not np.isfinite(sigma) or sigma < 1e-12:
                continue
            p = w[[inst]]
            row = R.metrics(r / sigma, lambda idx, p=p: R.pos_turnover(p, idx, cfg), cfg)
            market_rows[(sig, inst)] = row
            R.print_row(f"  {inst}", row, 22)
    print("=" * R.WIDTH)

    # ── stage 2: signals (inv-vol across instruments) ─────────────────────────
    F = _b.load_managed_returns(signals_dir=sdir)
    sigma_F = (F.loc[F.index <= cfg.is_end].std() * np.sqrt(cfg.ann)).replace(0.0, np.nan)
    F_n = F.div(sigma_F)

    posw: dict[str, pd.DataFrame] = {}
    for sig, (w, F_i) in panels.items():
        v = _b._instrument_multipliers(F_i, _b.CROSS_INSTRUMENT_BLEND[sig], cfg.is_end)
        posw[sig] = w.mul(v.reindex(w.columns).fillna(0.0), axis=1)

    signal_rows: dict[str, dict] = {}
    R.header("STAGE 2 — SIGNALS (inverse-vol across instruments, common dates)", 22, cfg.sub)
    for sig in F.columns:
        row = R.metrics(F_n[sig], lambda idx, s=sig: R.pos_turnover(posw.get(s), idx, cfg), cfg)
        signal_rows[sig] = row
        R.print_row(sig, row, 22)
    print("=" * R.WIDTH)

    # ── stage 3: composites (static Ridge fit on IS over signal subsets) ──────
    composite_rows: dict[str, dict] = {}
    R.header("STAGE 3 — COMPOSITES (static Ridge MSRR, IS-fit)", 38,
             f"{cfg.sub}  |  λ={lam}")
    F_is_n = F_n.loc[F_n.index <= cfg.is_end]
    names = list(F.columns)
    subsets = [c for k in range(2, len(names) + 1) for c in itertools.combinations(names, k)]
    for subset in subsets:
        cols = list(subset)
        bw = _b.msrr_ridge(F_is_n[cols], lam=lam)
        w_s = bw.weights.reindex(cols).fillna(0.0)
        w_s = w_s / w_s.abs().sum()
        r = (F_n[cols] * w_s).sum(axis=1)
        w_df = pd.DataFrame(np.tile(w_s.values, (len(F_n), 1)), index=F_n.index, columns=cols)
        row = R.metrics(
            r,
            lambda idx, w=w_df, cols=cols: R.composite_turnover(
                w, {k: posw[k] for k in cols if k in posw}, sigma_F, idx, cfg),
            cfg,
        )
        label = " + ".join(cols)
        composite_rows[label] = row
        R.print_row(label, row, 38)
    print("=" * R.WIDTH)

    # ── stage 4: TC-aware scenarios ───────────────────────────────────────────
    with contextlib.redirect_stdout(io.StringIO()):
        tc = _b.run_comm_tc(
            scenarios=scenarios, signals_dir=sdir, lookback=lookback,
            min_obs=min_obs, vol_target=vol_target, lam=lam,
        )
    R.print_tc(tc["metrics"],
               "STAGE 4 — TC-AWARE SCENARIOS (all 3 signals, uniform annual cost drag)", cfg)

    return {
        "markets":    R.to_frame(market_rows, ["signal", "instrument"]),
        "signals":    R.to_frame(signal_rows, ["signal"]),
        "composites": R.to_frame(composite_rows, ["composite"]),
        "tc":         tc["metrics"],
        "tc_returns": tc["returns"],
    }


if __name__ == "__main__":
    run_commodities_pipeline()
