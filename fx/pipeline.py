"""
End-to-end FX pipeline with IS / OOS reporting (monthly frequency, Sharpe √12).

run_fx_pipeline() (1) rebuilds the signal parquets, then prints Sharpe, MDD and
turnover for IS (≤ blend.IS_END) and OOS (≥ blend.OOS_START) at four levels:

  1. markets     per signal × per currency pair
  2. signals     inverse-vol cross-pair blend of each signal (momentum, mr_ns_loo, carry_xs)
  3. composites  rolling-Ridge blends of every subset (≥2) of the signals
  4. TC-aware    tc_sweep_fx scenarios (best / base / worst), fixed weights

Conventions: see core/pipeline_report.py.  Positions are month-end snapshots
(mr_ns_loo positions are sampled at the last business day of each month, as in
run_fx_prod), so mr_ns_loo turnover excludes intra-month trading.  Signal and
composite turnover use positions scaled by the inv-vol pair weights; the TC
table's turnover comes from tc_sweep_fx, which applies the same pair weights.
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
from fx import blend as _b


def run_fx_pipeline(
    pairs: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    rebuild_signals: bool = True,
    signals_dir: Optional[str | Path] = None,
    lookback: int = 36,
    min_obs: int = 36,
    tc_min_obs: Optional[int] = None,
    vol_target: float = 0.10,
    scenarios: Optional[dict] = None,
    verbose: bool = True,
) -> dict:
    """
    Run the whole FX pipeline and print IS/OOS Sharpe, MDD, turnover for
    pairs, signals, signal composites and the TC-aware scenarios.

    Parameters
    ----------
    pairs, start, end  Passed to the signal pipeline (fx/production/run.py).
    rebuild_signals    False reuses existing parquets in signals_dir.
    signals_dir        Parquet directory (default fx/signals/).
    lookback, min_obs  Rolling Ridge MSRR settings for composites (months;
                       defaults match run_fx_prod).
    tc_min_obs         min_obs for the TC sweep (default: same as min_obs, so net and
                       gross share the same IS sample).
    vol_target         Annual vol used for MDD scaling.
    scenarios          TC scenarios (default blend.TC_SCENARIOS_FX: best/base/worst).
    verbose            Print signal-pipeline progress.

    Returns
    -------
    dict with DataFrames "markets" (MultiIndex signal × pair), "signals",
    "composites", "tc" (tc_sweep_fx metrics), "tc_returns".
    """
    from fx.production.run import run as _run_signals, OUTPUT_DIR

    sdir = Path(signals_dir) if signals_dir else OUTPUT_DIR
    cfg = R.ReportCfg(ann=12, is_end=_b.IS_END, oos_start=_b.OOS_START,
                      vol_target=vol_target, min_obs=12)

    # ── stage 0: signals ──────────────────────────────────────────────────────
    if rebuild_signals:
        print("Stage 0 — rebuilding signal parquets...")
        kw = {k: v for k, v in (("pairs", pairs), ("start", start), ("end", end)) if v is not None}
        _run_signals(output_dir=sdir, verbose=verbose, **kw)
    else:
        print(f"Stage 0 — reusing parquets in {sdir}")

    rx_m = pd.read_parquet(sdir / "returns.parquet")["rx_m"]
    rx_d = pd.read_parquet(sdir / _b._RX_FILE)

    pos_mom = pd.read_parquet(sdir / "signal_a_f9_h1.parquet")
    pos_mr_d = pd.read_parquet(sdir / "mr_ns_loo_positions.parquet")
    pos_cxs = pd.read_parquet(sdir / "view_xs_carry_m.parquet")

    # Month-end position panels (turnover) and per-pair monthly PnL (same formulas
    # as load_monthly_managed_returns)
    pos = {
        "momentum":  pos_mom,
        "mr_ns_loo": pos_mr_d.resample("BME").last(),
        "carry_xs":  pos_cxs,
    }
    mr_live = pos_mr_d.notna().resample("BME").max().astype(bool)
    pnl = {
        "momentum":  pos_mom.shift(1) * rx_m.reindex(columns=pos_mom.columns),
        "mr_ns_loo": (pos_mr_d * rx_d.reindex(columns=pos_mr_d.columns)).resample("BME").sum(),
        "carry_xs":  pos_cxs.shift(1) * rx_m.reindex(columns=pos_cxs.columns),
    }
    if pairs is not None:
        pnl = {k: v[[p for p in pairs if p in v.columns]] for k, v in pnl.items()}

    # ── stage 1: pairs ────────────────────────────────────────────────────────
    market_rows: dict[tuple, dict] = {}
    R.header("STAGE 1 — PAIRS (per signal)", 22, cfg.sub)
    for sig, df in pnl.items():
        print(f"  [{sig}]")
        for pair in df.columns:
            r = df[pair]
            if sig == "mr_ns_loo":
                r = r.where(mr_live[pair].reindex(r.index).fillna(False))
            r = r.dropna()
            sigma = R.ann_sigma(r, cfg)
            if len(r) == 0 or not np.isfinite(sigma) or sigma < 1e-12:
                continue
            p = pos[sig][[pair]] if pair in pos[sig].columns else None
            row = R.metrics(r / sigma, lambda idx, p=p: R.pos_turnover(p, idx, cfg), cfg)
            market_rows[(sig, pair)] = row
            R.print_row(f"  {pair}", row, 22)
    print("=" * R.WIDTH)

    # ── stage 2: signals (inv-vol across pairs) ───────────────────────────────
    F = _b.load_monthly_managed_returns(signals_dir=sdir)
    sigma_F = (F.loc[F.index <= cfg.is_end].std() * np.sqrt(cfg.ann)).replace(0.0, np.nan)
    F_n = F.div(sigma_F)

    posw: dict[str, pd.DataFrame] = {}
    for sig in pnl:
        w = _b._pair_inv_vol_weights(pnl[sig], cfg.is_end)
        if w is not None:
            posw[sig] = R.weight_positions(pos[sig], w)

    signal_rows: dict[str, dict] = {}
    R.header("STAGE 2 — SIGNALS (inverse-vol across pairs, common dates)", 22, cfg.sub)
    for sig in F.columns:
        row = R.metrics(F_n[sig], lambda idx, s=sig: R.pos_turnover(posw.get(s), idx, cfg), cfg)
        signal_rows[sig] = row
        R.print_row(sig, row, 22)
    print("=" * R.WIDTH)

    # ── stage 3: composites (rolling Ridge over signal subsets) ───────────────
    composite_rows: dict[str, dict] = {}
    R.header("STAGE 3 — COMPOSITES (rolling Ridge MSRR)", 38,
             f"{cfg.sub}  |  lookback={lookback}m")
    F_is = F_n.loc[F_n.index <= cfg.is_end]
    F_oos = F_n.loc[F_n.index > cfg.is_end]
    names = list(F.columns)
    subsets = [c for k in range(2, len(names) + 1) for c in itertools.combinations(names, k)]
    for subset in subsets:
        cols = list(subset)
        model = _b.RollingMSRR(penalty="ridge", lam=0.01, lookback=lookback, min_obs=min_obs)
        w_all = pd.concat(_b.roll_weights_full(model, F_is[cols], F_oos[cols])).sort_index()
        r = _b.combined_return_rolling(F_n[cols], w_all)
        row = R.metrics(
            r,
            lambda idx, w=w_all, cols=cols: R.composite_turnover(
                w, {k: posw[k] for k in cols if k in posw}, sigma_F, idx, cfg),
            cfg,
        )
        label = " + ".join(cols)
        composite_rows[label] = row
        R.print_row(label, row, 38)
    print("=" * R.WIDTH)

    # ── stage 4: TC-aware scenarios ───────────────────────────────────────────
    with contextlib.redirect_stdout(io.StringIO()):
        tc = _b.run_fx_tc(
            scenarios=scenarios, signals_dir=sdir,
            lookback=lookback, min_obs=tc_min_obs or min_obs, vol_target=vol_target,
        )
    R.print_tc(tc["metrics"],
               "STAGE 4 — TC-AWARE SCENARIOS (all 3 signals, pair-specific forward spreads)", cfg)

    return {
        "markets":    R.to_frame(market_rows, ["signal", "pair"]),
        "signals":    R.to_frame(signal_rows, ["signal"]),
        "composites": R.to_frame(composite_rows, ["composite"]),
        "tc":         tc["metrics"],
        "tc_returns": tc["returns"],
    }


if __name__ == "__main__":
    run_fx_pipeline()
