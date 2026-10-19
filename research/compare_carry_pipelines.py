"""
Head-to-head comparison: Pipeline A (bootstrap carry) vs Pipeline B (NS carry).

Same data, same PnL engine (cm_total_return_bp), same rebalance/lag/neutralization.
Only the carry SIGNAL differs:
  Pipeline A: carry_dv01_bp from bootstrapped zero curve (carry_ois.ois_carry_panel)
  Pipeline B: carry_reprice from NS-fitted curve (experiment.run_carry_spec, baseline)

For Pipeline A we build positions the same way Pipeline B does:
  1. Cross-sectional z-score across tenors (within one curve, one date)
  2. DV01-neutral projection
  3. Monthly rebalance, lag=1

PnL is computed with cm_total_return_bp (uses NS fit from prev date for roll-down).
This means BOTH pipelines use the same returns engine — the only difference is
what drives the SIGNAL that determines position direction.
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from rates.ois_data import fetch_ois_curves
from rates.carry_ois import ois_carry_panel
from src.nsrv.ns_core import fit_ns
from src.nsrv.carry.experiment import run_carry_spec, BASELINE
from src.nsrv.carry.signal import xs_score, dv01_positions, _rebalance_dates, build_positions
from src.nsrv.carry.returns import cm_total_return_bp, compute_dur_panel, pnl_book

MARKETS = ["US", "EUR", "UK"]
LAM = 0.3
BASIS = "ns"
MIN_TENORS = 4


# ── Pipeline A position builder ───────────────────────────────────────────────

def _build_pipeline_a_positions(
    carry_wide: pd.DataFrame,   # date × tenor of carry_dv01_bp
    dur_wide: pd.DataFrame,     # date × tenor of duration (years) - same tenors
    rebalance: str = "monthly",
    lag: int = 1,
) -> pd.DataFrame:
    """
    Build DV01-neutral positions from Pipeline A carry_dv01_bp signal.
    Mirrors Pipeline B's build_positions logic exactly.
    """
    tenors = carry_wide.columns
    dates = carry_wide.index
    reb_dates = _rebalance_dates(dates, rebalance)

    pos_rows = {}
    held = None

    for date in dates:
        if date in reb_dates:
            c_row = carry_wide.loc[date]
            d_row = dur_wide.loc[date]
            valid = c_row.notna() & d_row.notna() & (d_row > 1e-8)
            if valid.sum() >= 2:
                c_adj = c_row[valid]         # already per-duration (carry_dv01_bp)
                score = xs_score(c_adj, mode="z")
                held = dv01_positions(score, d_row[valid], gross=1.0)
                held = held.reindex(tenors)

        pos_rows[date] = held if held is not None else pd.Series(np.nan, index=tenors)

    pos_df = pd.DataFrame(pos_rows).T
    pos_df.index.name = "date"
    return pos_df.shift(lag)


def _compute_pnl_from_positions(
    pos_df: pd.DataFrame,
    curves: pd.DataFrame,
    funding: pd.Series,
    tenors: np.ndarray,
) -> tuple[pd.Series, pd.Series]:
    """
    PnL using cm_total_return_bp. Fits NS on each prev date for roll-down.
    Same logic as experiment._compute_pnl_series.
    """
    fits = {}
    for date in curves.index:
        row = curves.loc[date].dropna()
        if len(row) < MIN_TENORS:
            continue
        tau = row.index.to_numpy(float)
        y_bp = row.to_numpy(float) * 100.0
        try:
            fits[date] = fit_ns(tau, y_bp, lam=LAM, basis=BASIS)
        except Exception:
            pass

    dates_list = list(pos_df.index)
    pnl_carry = pd.Series(0.0, index=pos_df.index, dtype=float)
    pnl_yldchg = pd.Series(0.0, index=pos_df.index, dtype=float)

    for i in range(1, len(dates_list)):
        date = dates_list[i]
        prev = dates_list[i - 1]

        pos = pos_df.loc[date].reindex(tenors).to_numpy(float)
        if np.isnan(pos).any():
            continue
        if prev not in fits or prev not in curves.index or date not in curves.index:
            continue
        if date not in funding.index or prev not in funding.index:
            continue

        fit_prev = fits[prev]
        row_prev = curves.loc[prev].reindex(tenors)
        row_now = curves.loc[date].reindex(tenors)
        if row_prev.isna().any() or row_now.isna().any():
            continue

        y_prev = row_prev.to_numpy(float) * 100.0
        y_now = row_now.to_numpy(float) * 100.0
        rf_prev = float(funding.loc[prev]) * 100.0
        dur_prev = compute_dur_panel(fit_prev, tenors)

        rc, ryc = cm_total_return_bp(y_prev, y_now, fit_prev, tenors, rf_prev, dur_prev)
        pc, pyc = pnl_book(pos, rc, ryc)
        pnl_carry[date] = pc
        pnl_yldchg[date] = pyc

    return pnl_carry, pnl_yldchg


# ── Stats ──────────────────────────────────────────────────────────────────────

def _stats(total: pd.Series, carry: pd.Series, label: str) -> dict:
    total = total.dropna().replace(0, np.nan).dropna()
    if len(total) < 21:
        return {}
    ann = 252
    sharpe = total.mean() / total.std(ddof=1) * np.sqrt(ann)
    cum = total.cumsum()
    max_dd = float((cum - cum.cummax()).min())
    hit = float((total > 0).mean())
    carry_ann = float(carry.reindex(total.index).mean() * ann)
    yldchg_ann = float((total - carry).reindex(total.index).mean() * ann)

    def _dd_in(s, y0, y1):
        sub = s.loc[f"{y0}-01-01":f"{y1}-12-31"]
        if len(sub) == 0:
            return np.nan
        c = sub.cumsum()
        return float((c - c.cummax()).min())

    return {
        "label": label,
        "sharpe": sharpe,
        "max_dd_bp": max_dd,
        "hit_rate": hit,
        "carry_ann_bp": carry_ann,
        "yldchg_ann_bp": yldchg_ann,
        "dd_2013": _dd_in(total, 2013, 2013),
        "dd_2022": _dd_in(total, 2022, 2022),
        "n_days": len(total),
    }


def _print_comparison(rows: list[dict]) -> None:
    W = 105
    print(f"\n{'='*W}")
    print(f"  PIPELINE A (bootstrap carry) vs PIPELINE B (NS carry)  —  same PnL engine, same positions logic")
    print(f"{'='*W}")
    hdr = (
        f"{'Market':<8} {'Pipeline':<14} {'Sharpe':>7} {'MaxDD':>8} "
        f"{'CarryAnn':>9} {'YldChgAnn':>10} {'HitRate':>8} "
        f"{'dd2013':>8} {'dd2022':>8} {'N':>6}"
    )
    print(hdr)
    print("-" * W)
    prev_mkt = None
    for r in rows:
        if not r:
            continue
        mkt = r["label"].split(" ")[0]
        pipe = r["label"].split(" ", 1)[1] if " " in r["label"] else ""
        if prev_mkt and mkt != prev_mkt:
            print()
        dd13 = f"{r['dd_2013']:>8.1f}" if not np.isnan(r['dd_2013']) else "     n/a"
        dd22 = f"{r['dd_2022']:>8.1f}" if not np.isnan(r['dd_2022']) else "     n/a"
        print(
            f"{mkt:<8} {pipe:<14} {r['sharpe']:>7.3f} {r['max_dd_bp']:>8.1f} "
            f"{r['carry_ann_bp']:>9.1f} {r['yldchg_ann_bp']:>10.1f} {r['hit_rate']:>8.3f} "
            f"{dd13} {dd22} {r['n_days']:>6}"
        )
        prev_mkt = mkt


# ── Main ───────────────────────────────────────────────────────────────────────

from src.nsrv.carry.experiment import _compute_carry_panel, _compute_pnl_series

all_rows = []
pnl_by_market: dict[str, dict] = {}   # {market: {"A": total_pnl, "B": total_pnl}}

for market in MARKETS:
    print(f"\n{'─'*60}")
    print(f"  {market}")
    print(f"{'─'*60}")

    curves = fetch_ois_curves(market)
    funding_tenor = 0.25
    if funding_tenor in curves.columns:
        funding = curves[funding_tenor]
    else:
        nearest = min(curves.columns, key=lambda x: abs(x - funding_tenor))
        funding = curves[nearest]

    pnl_by_market[market] = {}

    # ── Pipeline B (NS / baseline) ─────────────────────────────────────────────
    print(f"  Running Pipeline B (NS baseline)...", flush=True)
    try:
        carry_df_b, dur_df_b, fits_b = _compute_carry_panel(curves, funding, BASELINE, LAM, BASIS)
        tenors_b = carry_df_b.columns.to_numpy(float)
        pos_b = build_positions(
            carry_df_b, dur_df_b,
            rebalance=BASELINE.rebalance, lag=BASELINE.lag,
            xs_mode=BASELINE.xs_mode, neutralize=BASELINE.neutralize,
            gross=BASELINE.gross_dv01,
        )
        pc_b, pyc_b = _compute_pnl_series(curves, funding, pos_b, fits_b, tenors_b)
        total_b = pc_b + pyc_b
        pnl_by_market[market]["B"] = total_b
        row_b = _stats(total_b, pc_b, f"{market} B_NS_reprice")
        all_rows.append(row_b)
        print(f"    Sharpe={row_b.get('sharpe', 0):.3f}  n={row_b.get('n_days', 0)}")
    except Exception as exc:
        print(f"    Pipeline B failed: {exc}")
        all_rows.append({})

    # ── Pipeline A (bootstrap carry → same position logic) ────────────────────
    print(f"  Running Pipeline A (bootstrap carry)...", flush=True)
    try:
        panel_a = ois_carry_panel(market)
        if panel_a.empty:
            raise ValueError("empty panel")

        # Wide: date × tenor, carry_dv01_bp at 3M horizon
        sub = panel_a.xs("3M", level="horizon")["carry_dv01_bp"].unstack("tenor")
        sub.index = pd.to_datetime(sub.index)
        sub = sub.sort_index()

        tenors_a = sub.columns.to_numpy(float)
        dur_rows_a = {}
        for date in curves.index:
            row_c = curves.loc[date].reindex(tenors_a).dropna()
            if len(row_c) < MIN_TENORS:
                continue
            tau = row_c.index.to_numpy(float)
            y_bp = row_c.to_numpy(float) * 100.0
            try:
                fit = fit_ns(tau, y_bp, lam=LAM, basis=BASIS)
                dur_rows_a[date] = pd.Series(
                    compute_dur_panel(fit, tau), index=tau
                ).reindex(tenors_a)
            except Exception:
                pass
        dur_df_a = pd.DataFrame(dur_rows_a).T.reindex(sub.index)

        sub = sub.reindex(curves.index).dropna(how="all")
        dur_df_a = dur_df_a.reindex(sub.index)

        pos_a = _build_pipeline_a_positions(sub, dur_df_a, rebalance="monthly", lag=1)
        pc_a, pyc_a = _compute_pnl_from_positions(pos_a, curves, funding, tenors_a)
        total_a = pc_a + pyc_a
        pnl_by_market[market]["A"] = total_a
        row_a = _stats(total_a, pc_a, f"{market} A_bootstrap")
        all_rows.append(row_a)
        print(f"    Sharpe={row_a.get('sharpe', 0):.3f}  n={row_a.get('n_days', 0)}")
    except Exception as exc:
        print(f"    Pipeline A failed: {exc}")
        all_rows.append({})

# ── Final comparison table ────────────────────────────────────────────────────

_print_comparison(all_rows)

# ── PnL correlation between A and B ──────────────────────────────────────────
print(f"\n{'='*65}")
print("  PnL CORRELATION  A vs B  (daily, same market)")
print(f"{'='*65}")
print(f"  {'Market':<10} {'Corr(A,B)':>10}  {'N_common':>10}")
print(f"  {'-'*35}")
for mkt, d in pnl_by_market.items():
    if "A" in d and "B" in d:
        common = d["A"].dropna().index.intersection(d["B"].dropna().index)
        if len(common) > 30:
            c = float(d["A"].loc[common].corr(d["B"].loc[common]))
            print(f"  {mkt:<10} {c:>10.3f}  {len(common):>10}")
        else:
            print(f"  {mkt:<10}       n/a  {len(common):>10}")
    else:
        print(f"  {mkt:<10}       n/a         n/a")
