"""
mom_sharpe.py — OOS Sharpe per signal, market, tenor.

Signal A  MOP TSMOM:          sign(12m log excess return) × rx_m
Signal B1 Sihvonen own-mat:   sign(1m rx) × rx_m           [h=1 sign form]
Signal B3 Sihvonen yieldchg:  sign(−1m yc) × rx_m
Signal C  Durham long-short:  LP weights × cm bucket returns

Returns basis:
  A, B1, B3 → Sihvonen log excess returns (rx_m, exact bootstrap, monthly)
  C         → constant-maturity cm_return_bp at Durham bucket midpoints

Vol-scaling is omitted here: the Sharpe of sign(signal) × return is invariant
to vol scaling and directly measures directional accuracy.

OOS: 2019-01-01 → today.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from rates.ois_data import fetch_ois_curves
from rates.yc_builder import build_zero_curve
from rates.momentum.returns import cm_return_bp, swap_duration, _dfs_from_zero_row
from rates.momentum.signal_c import (
    MIDPOINTS, N_BUCKETS, durham_momentum, durham_long_short,
)

OOS_START  = "2019-01-01"
DM_MARKETS = ["US", "EUR", "UK", "Japan", "Canada", "Australia", "New Zealand", "Sweden"]

# Sihvonen tenor grid (months and years)
SIH_MONTHS = list(range(12, 121, 12))
SIH_YEARS  = [m / 12.0 for m in SIH_MONTHS]   # 1.0 … 10.0

# Signal A uses tenors available in the 1-10Y Sihvonen range
MOP_TENORS = [2.0, 5.0, 10.0]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sharpe(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 24:
        return np.nan
    sd = s.std()
    return float(s.mean() / sd * np.sqrt(12)) if sd > 1e-10 else np.nan


def _month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby([s.index.year, s.index.month]).last().values)


# ── Panel builders ────────────────────────────────────────────────────────────

def _build_rx_panels(market: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly Sihvonen log excess returns and yield-change component."""
    par  = fetch_ois_curves(market)
    zero = build_zero_curve(par)
    me   = _month_ends(zero.index)
    zm   = zero.reindex(me).dropna(how="all")
    dates = list(zm.index)

    rx_rows: dict = {}
    yc_rows: dict = {}

    for i in range(len(dates) - 1):
        t, t1 = dates[i], dates[i + 1]
        z0 = zm.loc[t].dropna() / 100.0    # % → decimal
        z1 = zm.loc[t1].dropna() / 100.0

        t0t, t0v = z0.index.to_numpy(float), z0.to_numpy(float)
        t1t, t1v = z1.index.to_numpy(float), z1.to_numpy(float)

        def z_t(τ, ts=t0t, ys=t0v):   return float(np.interp(τ, ts, ys))
        def z_tp1(τ, ts=t1t, ys=t1v): return float(np.interp(τ, ts, ys))

        rx_row: dict = {}
        yc_row: dict = {}
        for n, yr in zip(SIH_MONTHS, SIH_YEARS):
            yn_t    = z_t(n / 12.0) / 12.0
            ynm1_t  = z_t((n - 1) / 12.0) / 12.0
            ynm1_t1 = z_tp1((n - 1) / 12.0) / 12.0
            y1_t    = z_t(1.0 / 12.0) / 12.0
            rx_row[yr] = (-(n - 1) * ynm1_t1 + n * yn_t - y1_t) * 1e4
            yc_row[yr] = (n - 1) * (ynm1_t1 - ynm1_t) * 1e4

        rx_rows[t] = rx_row
        yc_rows[t] = yc_row

    rx_m = pd.DataFrame(rx_rows).T
    yc_m = pd.DataFrame(yc_rows).T
    rx_m.index.name = yc_m.index.name = "date"
    return rx_m, yc_m


def _build_bucket_panels(market: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly cm_return_bp and durations at the 6 Durham bucket midpoints."""
    par    = fetch_ois_curves(market)
    par_bp = par * 100.0
    zero   = build_zero_curve(par)
    me     = _month_ends(par.index)
    pm     = par_bp.reindex(me).dropna(how="all")
    zm     = zero.reindex(me).dropna(how="all")
    dates  = sorted(set(pm.index) & set(zm.index))
    rf_col = float(min(par.columns))
    dt_m   = 1.0 / 12

    ret_rows: dict = {}
    dur_rows: dict = {}

    for i in range(len(dates) - 1):
        t, t1 = dates[i], dates[i + 1]
        r   = pm.loc[t].dropna()
        r1  = pm.loc[t1].dropna()
        z   = zm.loc[t].dropna()
        dfs = _dfs_from_zero_row(z)
        rf  = float(r.get(rf_col, r.iloc[0]))
        rt, rv   = r.index.to_numpy(float),  r.to_numpy(float)
        r1t, r1v = r1.index.to_numpy(float), r1.to_numpy(float)

        rr: dict = {}
        rd: dict = {}
        for tau in MIDPOINTS:
            if tau < rt[0] or tau > rt[-1]:
                rr[tau] = rd[tau] = np.nan
                continue
            s_now  = float(np.interp(tau, rt, rv))
            s_next = float(np.interp(tau, r1t, r1v))
            s_aged = float(np.interp(max(tau - dt_m, rt[0]), rt, rv))
            dur    = swap_duration(dfs, tau)
            rr[tau] = cm_return_bp(s_now, s_aged, s_next, dur, rf, dt_m)
            rd[tau] = dur

        ret_rows[t] = rr
        dur_rows[t] = rd

    ret_b = pd.DataFrame(ret_rows).T
    dur_b = pd.DataFrame(dur_rows).T
    ret_b.index.name = dur_b.index.name = "date"
    return ret_b, dur_b


def _signal_c_pnl(ret_b: pd.DataFrame, dur_b: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    """Long-short LP PnL series.  Positions lagged 1 month (no lookahead)."""
    rbar   = durham_momentum(ret_b, n=n)
    w_rows: dict = {}
    for t in rbar.dropna(how="all").index:
        rb = rbar.loc[t].to_numpy(float)
        d  = dur_b.loc[t].to_numpy(float) if t in dur_b.index else np.full(N_BUCKETS, np.nan)
        if np.all(np.isfinite(rb)) and np.all(np.isfinite(d)):
            w_rows[t] = durham_long_short(rb, d)
    if not w_rows:
        return pd.DataFrame()
    w = pd.DataFrame(w_rows, index=MIDPOINTS).T
    w.index.name = "date"
    # Shift: position set at t, earns ret_b[t] (opened at t, closed at t+1)
    pnl = w.shift(1) * ret_b.reindex(w.index)
    pnl.index.name = "date"
    return pnl


# ── Sharpe printers ───────────────────────────────────────────────────────────

def _print_table(rows: dict, label: str, col_names, portfolio_key="portfolio") -> None:
    df = pd.DataFrame(rows).T
    fmt = {c: "{:6.2f}".format for c in df.columns}
    print(f"\n{'='*80}")
    print(label)
    print("="*80)
    print(df.to_string(float_format=lambda x: f"{x:6.2f}"))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    rx_panels: dict  = {}
    yc_panels: dict  = {}
    bkt_ret:   dict  = {}
    bkt_dur:   dict  = {}

    for mkt in DM_MARKETS:
        print(f"  Loading {mkt}...", flush=True)
        try:
            rx_panels[mkt], yc_panels[mkt] = _build_rx_panels(mkt)
        except Exception as e:
            print(f"    rx FAILED: {e}")
        try:
            bkt_ret[mkt], bkt_dur[mkt] = _build_bucket_panels(mkt)
        except Exception as e:
            print(f"    bucket FAILED: {e}")

    # ── Signal A ──────────────────────────────────────────────────────────────
    rows_a: dict = {}
    for mkt, rx_m in rx_panels.items():
        sig = np.sign(rx_m.shift(1).rolling(12).sum())
        pnl = sig * rx_m
        tenors = [t for t in MOP_TENORS if t in pnl.columns]
        row = {t: _sharpe(pnl[t].loc[OOS_START:]) for t in tenors}
        row["port"] = _sharpe(pnl[tenors].sum(axis=1).loc[OOS_START:])
        rows_a[mkt] = row
    _print_table(rows_a, f"Signal A: MOP TSMOM  (sign 12m, {OOS_START}–today)", MOP_TENORS)

    # ── Signal B1 ─────────────────────────────────────────────────────────────
    rows_b1: dict = {}
    for mkt, rx_m in rx_panels.items():
        sig = np.sign(rx_m.shift(1))
        pnl = sig * rx_m
        tenors = [t for t in SIH_YEARS if t in pnl.columns]
        row = {t: _sharpe(pnl[t].loc[OOS_START:]) for t in tenors}
        row["port"] = _sharpe(pnl[tenors].sum(axis=1).loc[OOS_START:])
        rows_b1[mkt] = row
    _print_table(rows_b1, f"Signal B1: Sihvonen own-maturity  (sign 1m, {OOS_START}–today)", SIH_YEARS)

    # ── Signal B3 ─────────────────────────────────────────────────────────────
    rows_b3: dict = {}
    for mkt in rx_panels:
        rx_m, yc_m = rx_panels[mkt], yc_panels[mkt]
        sig = np.sign(-yc_m.shift(1))
        pnl = sig * rx_m
        tenors = [t for t in SIH_YEARS if t in pnl.columns]
        row = {t: _sharpe(pnl[t].loc[OOS_START:]) for t in tenors}
        row["port"] = _sharpe(pnl[tenors].sum(axis=1).loc[OOS_START:])
        rows_b3[mkt] = row
    _print_table(rows_b3, f"Signal B3: Sihvonen yield-change  (sign −1m yc, {OOS_START}–today)", SIH_YEARS)

    # ── Signal C ──────────────────────────────────────────────────────────────
    rows_c: dict = {}
    for mkt in DM_MARKETS:
        if mkt not in bkt_ret:
            continue
        pnl = _signal_c_pnl(bkt_ret[mkt], bkt_dur[mkt])
        if pnl.empty:
            continue
        pnl_oos = pnl.loc[OOS_START:]
        row = {tau: _sharpe(pnl_oos[tau]) for tau in MIDPOINTS if tau in pnl_oos.columns}
        row["port"] = _sharpe(pnl_oos[[t for t in MIDPOINTS if t in pnl_oos.columns]].sum(axis=1))
        rows_c[mkt] = row
    _print_table(rows_c, f"Signal C: Durham long-short  (LP, {OOS_START}–today)", MIDPOINTS)


if __name__ == "__main__":
    main()
