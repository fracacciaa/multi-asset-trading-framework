"""
rates_signal_sharpe_tt.py — Train/test Sharpe by country for all 7 momentum signals.

Signals
-------
  A    MOP TSMOM         sign(12m rx_m)   × rx_m   tenors {2,5,10}Y
  B1   Sihvonen own-mat  sign(1m  rx_m)   × rx_m   tenors 1-10Y
  B2   Sihvonen factor   sign(rxbar_1m)   × rx_m   tenors 1-10Y   [common factor]
  B3   yield-change      sign(−1m yc_m)   × rx_m   tenors 1-10Y
  B4   PC1 momentum      sign(−Δpc1)      × rx_m   tenors 1-10Y   [PCA: first 40% IS]
  C_lo Durham long-only  LP weights × bucket_ret   6 buckets      [duration-matched]
  C_ls Durham long-short LP weights × bucket_ret   6 buckets      [zero-duration]

All signals: lag = 1 month (position at t-1, return at t).
Train: ≤ 2018-12-31   Test: ≥ 2019-01-01
PnL per country: mean across tenors (A–B4), portfolio return (C).
Sharpe: annualised = sqrt(12), min 24 obs.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")

from rates.production.momentum.returns import build_all_return_panels
from rates.production.momentum.signal_b import pc1_change
from rates.production.momentum.signal_c import build_signal_c, _bucket_returns
from rates.production.carry.experiment import within_curve_carry_pnl
from rates.production.mean_reversion.signal import mean_reversion_pnl

# ── Config ────────────────────────────────────────────────────────────────────

TRAIN_END    = "2018-12-31"
TEST_START   = "2019-01-01"
MIN_OBS      = 24
ANN          = np.sqrt(12)
ANN_D        = np.sqrt(252)
MOP_TENORS   = [2.0, 5.0, 10.0]
DM_MARKETS   = ["US", "EUR", "UK", "Japan", "Canada", "Australia", "New Zealand", "Sweden"]
_SIGNALS_DIR = Path(__file__).parents[2] / "signals"


# ── Sharpe helpers ─────────────────────────────────────────────────────────────

def _sharpe(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < MIN_OBS:
        return np.nan
    sd = s.std()
    return float(s.mean() / sd * ANN) if sd > 1e-10 else np.nan


def _tt(s: pd.Series) -> tuple[float, float]:
    return _sharpe(s[s.index <= TRAIN_END]), _sharpe(s[s.index >= TEST_START])


def _agg_tt(pnl_dict: dict) -> tuple[float, float]:
    if not pnl_dict:
        return np.nan, np.nan
    return _tt(pd.DataFrame(pnl_dict).mean(axis=1))


def _sharpe_d(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < MIN_OBS:
        return np.nan
    sd = s.std()
    return float(s.mean() / sd * ANN_D) if sd > 1e-10 else np.nan


def _tt_d(s: pd.Series) -> tuple[float, float]:
    return _sharpe_d(s[s.index <= TRAIN_END]), _sharpe_d(s[s.index >= TEST_START])


def _agg_tt_d(pnl_dict: dict) -> tuple[float, float]:
    """Equal-weight across independent per-market books."""
    if not pnl_dict:
        return np.nan, np.nan
    return _tt_d(pd.DataFrame(pnl_dict).mean(axis=1))


def _agg_tt_d_sum(pnl_dict: dict) -> tuple[float, float]:
    """Sum across markets — correct for a cross-sectional zero-net portfolio."""
    if not pnl_dict:
        return np.nan, np.nan
    return _tt_d(pd.DataFrame(pnl_dict).sum(axis=1))


# ── Printer ────────────────────────────────────────────────────────────────────

def _print(name: str, rows: list[tuple[str, float, float]]) -> None:
    w = 56
    print(f"\n{'=' * w}")
    print(f"  {name}")
    print(f"{'=' * w}")
    print(f"  {'Country':<22}  {'Train SR':>8}  {'Test SR':>8}")
    print(f"  {'-' * 22}  {'-' * 8}  {'-' * 8}")
    for i, (label, tr, te) in enumerate(rows):
        if label == "AGGREGATE":
            print(f"  {'-' * 22}  {'-' * 8}  {'-' * 8}")
        tr_s = f"{tr:+.3f}" if not np.isnan(tr) else "    n/a"
        te_s = f"{te:+.3f}" if not np.isnan(te) else "    n/a"
        print(f"  {label:<22}  {tr_s:>8}  {te_s:>8}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading return panels for all DM markets...", flush=True)
    all_panels = build_all_return_panels(markets=DM_MARKETS)
    print(f"  Loaded {len(all_panels)} markets.\n")

    # ── Signal A: MOP TSMOM ────────────────────────────────────────────────────
    rows_a: list = []
    pnl_a: dict = {}
    for mkt, p in all_panels.items():
        rx_m = p["rx_m"]
        cols = [t for t in MOP_TENORS if t in rx_m.columns]
        if not cols:
            continue
        sig = np.sign(rx_m[cols].shift(1).rolling(12).sum())
        pnl = (sig * rx_m[cols]).mean(axis=1)
        rows_a.append((mkt, *_tt(pnl)))
        pnl_a[mkt] = pnl
    rows_a.append(("AGGREGATE", *_agg_tt(pnl_a)))
    _print("A — MOP TSMOM  (sign 12m rx, {2,5,10}Y)", rows_a)

    # ── Signal B1: own-maturity momentum ───────────────────────────────────────
    rows_b1: list = []
    pnl_b1: dict = {}
    for mkt, p in all_panels.items():
        rx_m = p["rx_m"]
        sig = np.sign(rx_m.shift(1))
        pnl = (sig * rx_m).mean(axis=1)
        rows_b1.append((mkt, *_tt(pnl)))
        pnl_b1[mkt] = pnl
    rows_b1.append(("AGGREGATE", *_agg_tt(pnl_b1)))
    _print("B1 — Sihvonen own-maturity  (sign 1m rx, 1-10Y)", rows_b1)

    # ── Signal B2: factor momentum ─────────────────────────────────────────────
    rows_b2: list = []
    pnl_b2: dict = {}
    for mkt, p in all_panels.items():
        rx_m = p["rx_m"]
        rxbar = rx_m.mean(axis=1)
        sig = np.sign(rxbar.shift(1))
        pnl = rx_m.multiply(sig, axis=0).mean(axis=1)
        rows_b2.append((mkt, *_tt(pnl)))
        pnl_b2[mkt] = pnl
    rows_b2.append(("AGGREGATE", *_agg_tt(pnl_b2)))
    _print("B2 — Sihvonen factor  (sign avg-rx across maturities, 1-10Y)", rows_b2)

    # ── Signal B3: yield-change momentum ──────────────────────────────────────
    rows_b3: list = []
    pnl_b3: dict = {}
    for mkt, p in all_panels.items():
        rx_m = p["rx_m"]
        yc_m = p["yc_m"]
        sig = np.sign(-yc_m.shift(1))
        pnl = (sig * rx_m).mean(axis=1)
        rows_b3.append((mkt, *_tt(pnl)))
        pnl_b3[mkt] = pnl
    rows_b3.append(("AGGREGATE", *_agg_tt(pnl_b3)))
    _print("B3 — Sihvonen yield-change  (sign −yc, 1-10Y)", rows_b3)

    # ── Signal B4: PC1 momentum ────────────────────────────────────────────────
    rows_b4: list = []
    pnl_b4: dict = {}
    for mkt, p in all_panels.items():
        rx_m = p["rx_m"]
        z_m  = p["z_m"]
        pc_df = pc1_change(z_m, train_frac=0.40)
        if pc_df.empty or "pc1" not in pc_df.columns:
            continue
        sig = np.sign(pc_df["pc1"].shift(1))
        idx = sig.dropna().index.intersection(rx_m.index)
        if len(idx) < MIN_OBS:
            continue
        pnl = rx_m.loc[idx].multiply(sig.loc[idx], axis=0).mean(axis=1)
        rows_b4.append((mkt, *_tt(pnl)))
        pnl_b4[mkt] = pnl
    rows_b4.append(("AGGREGATE", *_agg_tt(pnl_b4)))
    _print("B4 — Sihvonen PC1  (sign −Δpc1, 1-10Y, PCA fit: first 40%)", rows_b4)

    # ── Signal C: Durham LP ────────────────────────────────────────────────────
    print("Building Signal C Durham LP...", flush=True)
    c_results = build_signal_c(all_panels)

    rows_clo: list = []
    rows_cls: list = []
    pnl_clo: dict = {}
    pnl_cls: dict = {}

    for mkt, res in c_results.items():
        p = all_panels[mkt]
        ret_b, _ = _bucket_returns(p["ret_m"], p["z_m"])

        for variant, rows_list, pnl_dict in [
            ("long_only",  rows_clo, pnl_clo),
            ("long_short", rows_cls, pnl_cls),
        ]:
            weights = res[variant]
            common  = weights.index.intersection(ret_b.index)
            pnl     = (weights.reindex(common).shift(1) * ret_b.reindex(common)).sum(axis=1)
            rows_list.append((mkt, *_tt(pnl)))
            pnl_dict[mkt] = pnl

    rows_clo.append(("AGGREGATE", *_agg_tt(pnl_clo)))
    rows_cls.append(("AGGREGATE", *_agg_tt(pnl_cls)))
    _print("C long-only  — Durham XS LP  (duration-matched, 6 buckets)", rows_clo)
    _print("C long-short — Durham XS LP  (zero-duration,   6 buckets)", rows_cls)

    # ── Within-curve carry: daily DV01-neutral per-tenor PnL ─────────────────
    print("Building within-curve carry PnL (DV01-neutral, BASELINE config)...", flush=True)
    wcc_pnl = within_curve_carry_pnl(markets=DM_MARKETS)
    rows_wcc: list = []
    pnl_wcc: dict = {}
    for mkt in DM_MARKETS:
        if mkt not in wcc_pnl.columns:
            rows_wcc.append((mkt, np.nan, np.nan))
            continue
        rows_wcc.append((mkt, *_tt_d(wcc_pnl[mkt])))
        pnl_wcc[mkt] = wcc_pnl[mkt]
    rows_wcc.append(("AGGREGATE", *_agg_tt_d(pnl_wcc)))
    _print("Within-curve carry  (reprice, dur-adj z-score, DV01-neutral, √252)", rows_wcc)

    # ── Mean reversion: daily factor-neutral NS residual PnL ─────────────────
    print("Building mean-reversion PnL (factor-neutral NS residuals)...", flush=True)
    mr_pnl = mean_reversion_pnl(markets=DM_MARKETS)
    rows_mr: list = []
    pnl_mr: dict = {}
    for mkt in DM_MARKETS:
        if mkt not in mr_pnl.columns:
            rows_mr.append((mkt, np.nan, np.nan))
            continue
        rows_mr.append((mkt, *_tt_d(mr_pnl[mkt])))
        pnl_mr[mkt] = mr_pnl[mkt]
    rows_mr.append(("AGGREGATE", *_agg_tt_d(pnl_mr)))
    _print("Mean reversion  (factor-neutral NS resid, EWMA z-score, √252)", rows_mr)

    # ── Cross-country carry: daily pos × 10Y return (loaded from parquet) ────
    print("Building cross-country carry PnL (from parquet)...", flush=True)
    try:
        cc_pos   = pd.read_parquet(_SIGNALS_DIR / "carry_cc.parquet")
        ret_d_cc = pd.read_parquet(_SIGNALS_DIR / "ret_d.parquet")
        mkt_level = ret_d_cc.columns.get_level_values("market")
        rows_cc: list = []
        pnl_cc: dict = {}
        for mkt in cc_pos.columns:
            if mkt not in mkt_level:
                continue
            mkt_tenors = ret_d_cc[mkt].columns.to_numpy(float)
            closest = float(mkt_tenors[np.argmin(np.abs(mkt_tenors - 10.0))])
            ret_10y = ret_d_cc[(mkt, closest)]
            common  = cc_pos.index.intersection(ret_10y.index)
            pnl     = cc_pos.loc[common, mkt] * ret_10y.loc[common]
            rows_cc.append((mkt, *_tt_d(pnl)))
            pnl_cc[mkt] = pnl
        rows_cc.append(("AGGREGATE (portfolio)", *_agg_tt_d_sum(pnl_cc)))
        _print("Cross-country carry  (DV01-neutral, pos×10Y ret, √252)", rows_cc)
    except FileNotFoundError as exc:
        print(f"  Skipping CC carry (parquet not found): {exc}", flush=True)


if __name__ == "__main__":
    main()
