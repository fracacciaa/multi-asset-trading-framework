#!/usr/bin/env python3
"""
Signal predictive regression: does each VF signal predict forward returns?

At each rebalance date t, pairs:
  x_{i,t}  =  VF position for ticker i          (from cached positions)
  y_{i,t}  =  cumulative return over next period  (rd+1 → next rebal date)

Pooled OLS with date-clustered standard errors; also reports IC / ICIR.

Usage:
    python equities_signal_regression.py
    python equities_signal_regression.py --rebuild
"""

from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import statsmodels.api as sm

IS_END    = "2020-12-31"
OOS_START = "2021-01-01"

_VF_DATA_CACHE = "equities/cache/vf_data.pkl"
_VF_POS_CACHE  = "equities/cache/vf_positions.pkl"


# ── data loading ──────────────────────────────────────────────────────────────

def _load_or_build_positions(rebuild: bool) -> dict[str, pd.DataFrame]:
    if not rebuild and os.path.exists(_VF_POS_CACHE):
        print("Loading cached VF positions …")
        with open(_VF_POS_CACHE, "rb") as f:
            return pickle.load(f)

    print("Building VF positions — this takes ~5 min …")
    from equities.data import fetch_prices, fetch_fundamentals
    from equities.signals import build_vf_positions

    prices       = fetch_prices(tickers=None)
    fundamentals = fetch_fundamentals(tickers=None)
    returns      = prices.pct_change(fill_method=None)
    positions    = build_vf_positions(
        prices=prices, fundamentals=fundamentals, returns=returns
    )
    os.makedirs(os.path.dirname(_VF_POS_CACHE), exist_ok=True)
    with open(_VF_POS_CACHE, "wb") as f:
        pickle.dump(positions, f)
    print(f"Cached to {_VF_POS_CACHE}\n")
    return positions


def _load_rebal_dates(positions: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    if os.path.exists(_VF_DATA_CACHE):
        with open(_VF_DATA_CACHE, "rb") as f:
            vf_data = pickle.load(f)
        return vf_data["rebal_dates"]
    # fallback: infer from position changes in first signal
    first = next(iter(positions.values()))
    changed = first.diff().abs().sum(axis=1) > 1e-6
    return first.index[changed]


# ── panel construction ────────────────────────────────────────────────────────

def build_panel(
    pos_df: pd.DataFrame,
    returns: pd.DataFrame,
    rebal_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    One row per (rebal_date, ticker) with non-zero position.
    Forward return = cumulative return from rd+1 to the next rebal date.
    """
    all_dates = returns.index
    rows: list[dict] = []

    for i, rd in enumerate(rebal_dates):
        if rd not in pos_df.index:
            continue

        next_rd    = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else all_dates[-1]
        hold_dates = all_dates[(all_dates > rd) & (all_dates <= next_rd)]
        if hold_dates.empty:
            continue

        pos = pos_df.loc[rd]
        pos = pos[pos.abs() > 1e-8].dropna()
        if pos.empty:
            continue

        common = pos.index.intersection(returns.columns)
        if common.empty:
            continue

        fwd_ret = (1 + returns.loc[hold_dates, common]).prod() - 1

        for ticker in common:
            rows.append({
                "date":       rd,
                "ticker":     ticker,
                "position":   float(pos[ticker]),
                "fwd_return": float(fwd_ret[ticker]),
            })

    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "position", "fwd_return"])
    return pd.DataFrame(rows)


# ── regression & IC ───────────────────────────────────────────────────────────

def run_ols(panel: pd.DataFrame) -> dict:
    """Pooled OLS with date-clustered standard errors."""
    df = panel.dropna()
    if df["date"].nunique() < 10 or len(df) < 50:
        return {}

    X = sm.add_constant(df["position"])
    y = df["fwd_return"]
    res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df["date"]})

    return {
        "beta":    float(res.params["position"]),
        "t_stat":  float(res.tvalues["position"]),
        "p_value": float(res.pvalues["position"]),
        "r2":      float(res.rsquared),
        "n_obs":   len(df),
        "n_dates": int(df["date"].nunique()),
    }


def run_ic(panel: pd.DataFrame) -> dict:
    """Spearman IC per rebalance date → mean, t-stat, ICIR."""
    df = panel.dropna()
    ics: list[float] = []

    for _, grp in df.groupby("date"):
        if len(grp) < 10:
            continue
        ic = grp["position"].rank().corr(grp["fwd_return"].rank())
        ics.append(float(ic))

    if not ics:
        return {}

    arr    = np.array(ics)
    n      = len(arr)
    mean   = float(arr.mean())
    std    = float(arr.std(ddof=1))
    t_stat = mean / (std / np.sqrt(n)) if std > 1e-12 else np.nan
    icir   = mean / std                 if std > 1e-12 else np.nan

    return {
        "ic_mean":  mean,
        "ic_std":   std,
        "ic_tstat": t_stat,
        "icir":     icir,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def _load_raw_signals():
    if not os.path.exists(_VF_DATA_CACHE):
        return None, None
    with open(_VF_DATA_CACHE, "rb") as f:
        vf_data = pickle.load(f)
    return vf_data.get("raw_signals"), vf_data.get("rebal_dates")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild VF positions from scratch")
    parser.add_argument("--raw", action="store_true",
                        help="Use raw signals (pre-VF) instead of VF positions")
    args = parser.parse_args()

    from equities.data import fetch_prices
    prices  = fetch_prices(tickers=None)
    returns = prices.pct_change(fill_method=None).dropna(how="all")

    if args.raw:
        signals, rebal_dates = _load_raw_signals()
        if signals is None:
            print("No vf_data cache found — run without --raw first to build it.")
            return
        title = "RAW SIGNALS (pre-ViewFormation)"
    else:
        positions   = _load_or_build_positions(args.rebuild)
        signals     = positions
        rebal_dates = _load_rebal_dates(positions)
        title       = "VF POSITIONS (post-ViewFormation)"

    rd_is  = rebal_dates[rebal_dates <= IS_END]
    rd_oos = rebal_dates[rebal_dates >= OOS_START]

    results: dict[str, dict] = {}

    for name, sig_df in signals.items():
        panel_is  = build_panel(sig_df, returns, rd_is)
        panel_oos = build_panel(sig_df, returns, rd_oos)

        results[name] = {
            "is":  {**run_ols(panel_is),  **run_ic(panel_is)},
            "oos": {**run_ols(panel_oos), **run_ic(panel_oos)},
        }

    _print_table(results, title)


def _print_table(results: dict, title: str = "VF POSITIONS (post-ViewFormation)") -> None:
    W = 94
    print("\n" + "=" * W)
    print(f"  SIGNAL PREDICTIVE REGRESSION  ({title} → next-period cumulative returns)")
    print(f"  IS ≤ {IS_END}   |   OOS ≥ {OOS_START}")
    print("=" * W)

    hdr = (
        f"  {'Signal':<18}  {'β':>8}  {'t-stat':>8}  {'p-val':>7}"
        f"  {'R²':>7}  {'IC':>7}  {'IC t':>7}  {'ICIR':>7}  {'N obs':>8}  {'N dates':>7}"
    )

    for split in ("is", "oos"):
        label = "IN-SAMPLE" if split == "is" else "OUT-OF-SAMPLE"
        print(f"\n  {label}")
        print(hdr)
        print("  " + "─" * (W - 2))

        for name, v in results.items():
            s = v[split]
            if not s:
                print(f"  {name:<18}  {'n/a':>8}")
                continue
            print(
                f"  {name:<18}"
                f"  {s.get('beta',    np.nan):>+8.5f}"
                f"  {s.get('t_stat',  np.nan):>+8.2f}"
                f"  {s.get('p_value', np.nan):>7.4f}"
                f"  {s.get('r2',      np.nan) * 100:>6.3f}%"
                f"  {s.get('ic_mean', np.nan):>+7.4f}"
                f"  {s.get('ic_tstat',np.nan):>+7.2f}"
                f"  {s.get('icir',    np.nan):>+7.3f}"
                f"  {s.get('n_obs',   0):>8,}"
                f"  {s.get('n_dates', 0):>7,}"
            )

        print("  " + "─" * (W - 2))

    print("\n" + "=" * W + "\n")


if __name__ == "__main__":
    main()
