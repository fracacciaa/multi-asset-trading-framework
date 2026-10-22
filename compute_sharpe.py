"""
Rates signal Sharpe ratio table.

Reads *_pnl.parquet files from rates/signals/ — the per-instrument PnL
series produced by rates/production/run.py.

Daily signals  → annualise with √252
Monthly signals → annualise with √12  (B/C momentum)

Usage:  python compute_sharpe.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

SIGNALS_DIR = Path("rates/signals")
TRAIN_END   = "2018-12-31"
TEST_START  = "2019-01-01"
MIN_OBS     = 30

# (file_stem, label, freq)
PNL_FILES = [
    ("carry_pnl",                   "Carry (within-curve)",         "D"),
    ("mean_reversion_pnl",          "Mean Reversion",               "D"),
    ("carry_cc_pnl",                "Carry CC",                     "D"),
    ("momentum_a_pnl",              "Momentum A (MOP monthly)",     "D"),
    ("momentum_a_daily_pnl",        "Momentum A-daily (252d)",      "D"),
    ("momentum_a_vf_ts_pnl",        "Momentum A-VF-TS",             "D"),
    ("momentum_a_vf_xs_pnl",        "Momentum A-VF-XS",             "D"),
    ("momentum_b_b1_own_pnl",       "Momentum B1 (own)",            "M"),
    ("momentum_b_b2_factor_pnl",    "Momentum B2 (factor)",         "M"),
    ("momentum_b_b3_yieldchg_pnl",  "Momentum B3 (yldchg)",         "M"),
    ("momentum_b_b4_pc1_pnl",       "Momentum B4 (PC1)",            "M"),
    ("momentum_c_long_only_pnl",    "Momentum C (long-only)",       "M"),
    ("momentum_c_long_short_pnl",   "Momentum C (long-short)",      "M"),
]


def sr(pnl: pd.Series, freq: str) -> float:
    s = pnl.dropna()
    if len(s) < MIN_OBS:
        return np.nan
    std = s.std()
    if std < 1e-12:
        return np.nan
    ann = np.sqrt(252) if freq == "D" else np.sqrt(12)
    return float(s.mean() / std * ann)


def load_df(stem: str) -> pd.DataFrame | None:
    p = SIGNALS_DIR / f"{stem}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    return df if isinstance(df, pd.DataFrame) else df.to_frame()


def fmt(x: float) -> str:
    return f"{x:+.2f}" if not np.isnan(x) else " n/a"


def print_signal_table(label: str, freq: str, df: pd.DataFrame):
    ann  = np.sqrt(252) if freq == "D" else np.sqrt(12)
    mkts = list(df.columns)

    tr_vals = {m: sr(df[m].loc[df.index <= TRAIN_END], freq) for m in mkts}
    te_vals = {m: sr(df[m].loc[df.index >= TEST_START], freq) for m in mkts}

    # aggregate = equal-weight mean of per-market PnL
    agg      = df.mean(axis=1)
    tr_agg   = sr(agg.loc[agg.index <= TRAIN_END], freq)
    te_agg   = sr(agg.loc[agg.index >= TEST_START], freq)

    col_w = 8
    header = f"  {'Market':<20}"
    for m in mkts:
        header += f"  {m:>{col_w}}"
    header += f"  {'AVG':>{col_w}}"
    sep = "  " + "-" * 20 + ("  " + "-" * col_w) * (len(mkts) + 1)

    print(f"\n{'─' * 72}")
    print(f"  {label}  [{freq}]")
    print(f"{'─' * 72}")
    print(header)
    print(sep)

    train_row = f"  {'Train (≤ 2018)':<20}"
    test_row  = f"  {'Test  (≥ 2019)':<20}"
    for m in mkts:
        train_row += f"  {fmt(tr_vals[m]):>{col_w}}"
        test_row  += f"  {fmt(te_vals[m]):>{col_w}}"
    train_row += f"  {fmt(tr_agg):>{col_w}}"
    test_row  += f"  {fmt(te_agg):>{col_w}}"

    print(train_row)
    print(test_row)


if __name__ == "__main__":
    print("\n" + "=" * 72)
    print("  RATES PRODUCTION SIGNAL SHARPE RATIOS")
    print("  Per-instrument PnL (no single-tenor proxies)")
    print("=" * 72)

    missing = []
    for stem, label, freq in PNL_FILES:
        df = load_df(stem)
        if df is None:
            missing.append(stem)
            continue
        print_signal_table(label, freq, df)

    if missing:
        print(f"\n[Not yet computed: {', '.join(missing)}]")
        print("Run:  python -m rates.production.run")
