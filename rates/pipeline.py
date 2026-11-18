"""
End-to-end rates pipeline with IS / OOS reporting.

run_rates_pipeline() (1) rebuilds the signal parquets, then prints Sharpe, MDD
and turnover for IS (≤ blend.IS_END) and OOS (≥ blend.OOS_START) at four levels:

  1. markets     per signal × per market
  2. signals     inverse-vol cross-market blend of each signal
  3. composites  rolling-Ridge blends of every subset (≥2) of the signals
  4. TC-aware    tc_sweep_rates scenarios (best / base / worst), fixed weights

Conventions: see core/pipeline_report.py.  Signal and composite turnover use
positions scaled by the inv-vol market weights (and by the Ridge weights for
composites); the TC table's turnover comes from tc_sweep_rates, which does not
apply the market weights.  Markets with <30 IS obs are scaled by their
full-sample σ.
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
from rates import blend as _b


def run_rates_pipeline(
    markets: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    rebuild_signals: bool = True,
    signals_dir: Optional[str | Path] = None,
    lookback: int = 252,
    min_obs: int = 63,
    vol_target: float = 0.10,
    scenarios: Optional[dict] = None,
    verbose: bool = True,
) -> dict:
    """
    Run the whole rates pipeline and print IS/OOS Sharpe, MDD, turnover for
    markets, signals, signal composites and the TC-aware scenarios.

    Parameters
    ----------
    markets, start, end  Passed to the signal pipeline (rates/production/run.py).
    rebuild_signals      False reuses existing parquets in signals_dir.
    signals_dir          Parquet directory (default rates/signals/).
    lookback, min_obs    Rolling Ridge MSRR settings for composites and TC sweep.
    vol_target           Annual vol used for MDD scaling.
    scenarios            TC scenarios (default blend.TC_SCENARIOS: best/base/worst).
    verbose              Print signal-pipeline progress.

    Returns
    -------
    dict with DataFrames "markets" (MultiIndex signal × market), "signals",
    "composites", "tc" (tc_sweep_rates metrics), "tc_returns".
    """
    from rates.production.run import run as _run_signals, OUTPUT_DIR

    sig_dir = Path(signals_dir) if signals_dir else OUTPUT_DIR
    cfg = R.ReportCfg(ann=252, is_end=_b.IS_END, oos_start=_b.OOS_START, vol_target=vol_target)

    # ── stage 0: signals ──────────────────────────────────────────────────────
    if rebuild_signals:
        print("Stage 0 — rebuilding signal parquets...")
        _run_signals(markets=markets, start=start, end=end,
                     output_dir=sig_dir, verbose=verbose)
    else:
        print(f"Stage 0 — reusing parquets in {sig_dir}")

    pnl = {n: pd.read_parquet(sig_dir / f).sort_index() for n, f in _b.SIGNAL_FILES.items()}
    pos = {n: pd.read_parquet(sig_dir / f).sort_index() for n, f in _b._POS_FILES.items()
           if (sig_dir / f).exists()}
    for d in (*pnl.values(), *pos.values()):
        d.index = pd.to_datetime(d.index)

    # ── stage 1: markets ──────────────────────────────────────────────────────
    market_rows: dict[tuple, dict] = {}
    R.header("STAGE 1 — MARKETS (per signal)", 22, cfg.sub)
    for sig, df in pnl.items():
        print(f"  [{sig}]")
        for mkt in df.columns:
            r = df[mkt].dropna()
            sigma = R.ann_sigma(r, cfg)
            if not np.isfinite(sigma) or sigma < 1e-12:
                continue
            p = (pos[sig].xs(mkt, level=0, axis=1)
                 if sig in pos and mkt in pos[sig].columns.get_level_values(0) else None)
            row = R.metrics(r / sigma, lambda idx, p=p: R.pos_turnover(p, idx, cfg), cfg)
            market_rows[(sig, mkt)] = row
            R.print_row(f"  {mkt}", row, 22)
    print("=" * R.WIDTH)

    # ── stage 2: signals (inv-vol across markets) ─────────────────────────────
    F = _b.load_managed_returns(signals_dir=sig_dir)
    sigma_F = (F.loc[F.index <= cfg.is_end].std() * np.sqrt(cfg.ann)).replace(0.0, np.nan)
    F_n = F.div(sigma_F)

    posw: dict[str, pd.DataFrame] = {}
    for sig in pnl:
        w = _b._inv_vol_weights(pnl[sig], cfg.is_end)
        if sig in pos and w is not None:
            posw[sig] = R.weight_positions(pos[sig], w)

    signal_rows: dict[str, dict] = {}
    R.header("STAGE 2 — SIGNALS (inverse-vol across markets, common dates)", 22, cfg.sub)
    for sig in F.columns:
        row = R.metrics(F_n[sig], lambda idx, s=sig: R.pos_turnover(posw.get(s), idx, cfg), cfg)
        signal_rows[sig] = row
        R.print_row(sig, row, 22)
    print("=" * R.WIDTH)

    # ── stage 3: composites (rolling Ridge over signal subsets) ───────────────
    composite_rows: dict[str, dict] = {}
    R.header("STAGE 3 — COMPOSITES (rolling Ridge MSRR)", 38,
             f"{cfg.sub}  |  lookback={lookback}d")
    names = list(F.columns)
    subsets = [c for k in range(2, len(names) + 1) for c in itertools.combinations(names, k)]
    for subset in subsets:
        cols = list(subset)
        Fs = F_n[cols]
        w_raw = _b.fit_rolling_ridge(lookback=lookback, min_obs=min_obs)._roll(Fs)
        w = w_raw.div(w_raw.abs().sum(axis=1).replace(0.0, np.nan), axis=0)
        r = _b.rolling_combined_return(Fs, w)
        row = R.metrics(
            r,
            lambda idx, w=w, cols=cols: R.composite_turnover(
                w, {k: posw[k] for k in cols if k in posw}, sigma_F, idx, cfg),
            cfg,
        )
        label = " + ".join(cols)
        composite_rows[label] = row
        R.print_row(label, row, 38)
    print("=" * R.WIDTH)

    # ── stage 4: TC-aware scenarios ───────────────────────────────────────────
    with contextlib.redirect_stdout(io.StringIO()):
        tc = _b.run_rates_tc(
            scenarios=scenarios, signals_dir=sig_dir,
            lookback=lookback, min_obs=min_obs, vol_target=vol_target,
        )
    R.print_tc(tc["metrics"],
               "STAGE 4 — TC-AWARE SCENARIOS (all 3 signals, market-specific half-spreads)", cfg)

    return {
        "markets":    R.to_frame(market_rows, ["signal", "market"]),
        "signals":    R.to_frame(signal_rows, ["signal"]),
        "composites": R.to_frame(composite_rows, ["composite"]),
        "tc":         tc["metrics"],
        "tc_returns": tc["returns"],
    }


if __name__ == "__main__":
    run_rates_pipeline()
