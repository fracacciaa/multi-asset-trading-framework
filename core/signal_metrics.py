"""
Per-asset-class signal and composite IS / OOS metrics.

Entry point
-----------
run_signal_metrics(verbose=True) -> pd.DataFrame

Prints one table per asset class (rates, FX, commodities, equities) showing
IS and OOS Sharpe and MDD (at 10% vol target) for each individual signal and
the blend composite.  Returns a tidy DataFrame with all results.

Equities runs full position-level backtests; the other three asset classes
load directly from parquet caches and are fast.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

IS_END    = "2020-12-31"
OOS_START = "2021-01-01"
VOL_TARGET = 0.10
W = 76


# ── Metric helpers ─────────────────────────────────────────────────────────────

def _sharpe(r: pd.Series, freq: int) -> float:
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(freq)) if len(r) > 20 and r.std() > 1e-12 else float("nan")


def _mdd(r: pd.Series, freq: int, vol_target: float = VOL_TARGET) -> float:
    r = r.dropna()
    if r.empty:
        return float("nan")
    ann_vol = r.std() * np.sqrt(freq)
    if ann_vol < 1e-12:
        return float("nan")
    r_scaled = r * (vol_target / ann_vol)
    cum = (1 + r_scaled).cumprod()
    return float((cum / cum.cummax() - 1).min() * 100)


def _row(r: pd.Series, freq: int, is_end: str, oos_start: str) -> dict:
    r_is  = r.loc[r.index <= is_end].dropna()
    r_oos = r.loc[r.index >= oos_start].dropna()
    return {
        "IS_SR":  _sharpe(r_is,  freq),
        "IS_MDD": _mdd(r_is,  freq),
        "OOS_SR": _sharpe(r_oos, freq),
        "OOS_MDD":_mdd(r_oos, freq),
    }


# ── Print helpers ──────────────────────────────────────────────────────────────

def _print_header(title: str, is_end: str, oos_start: str, freq_label: str) -> None:
    print()
    print("=" * W)
    print(f"  {title}  ({freq_label})")
    print(f"  IS ≤ {is_end}   |   OOS ≥ {oos_start}   |   MDD at {VOL_TARGET*100:.0f}% vol")
    print("=" * W)
    print(f"  {'Signal':<30}  {'IS SR':>7}  {'IS MDD':>8}  {'OOS SR':>7}  {'OOS MDD':>8}")
    print("  " + "─" * (W - 2))


def _print_row(label: str, row: dict, composite: bool = False) -> None:
    tag = " [composite]" if composite else ""
    sr  = lambda v: f"{v:+.2f}" if np.isfinite(v) else "   n/a"
    dd  = lambda v: f"{v:+.1f}%" if np.isfinite(v) else "    n/a"
    print(f"  {label+tag:<30}  {sr(row['IS_SR']):>7}  {dd(row['IS_MDD']):>8}"
          f"  {sr(row['OOS_SR']):>7}  {dd(row['OOS_MDD']):>8}")


def _print_footer() -> None:
    print("=" * W)


# ── Per-asset loaders ──────────────────────────────────────────────────────────

def _rates_rows(is_end: str, oos_start: str) -> list[tuple[str, dict]]:
    from rates.blend import run_blend
    res  = run_blend()
    F    = pd.concat([res["F_is"], res["F_oos"]]).sort_index()
    r    = pd.concat([res["r_is"], res["r_oos"]]).sort_index()
    rows = [(col, _row(F[col], 252, is_end, oos_start)) for col in F.columns]
    rows.append(("combined", _row(r, 252, is_end, oos_start)))
    return rows


def _fx_rows(is_end: str, oos_start: str) -> list[tuple[str, dict]]:
    from fx.blend import run_fx_blend
    res  = run_fx_blend()
    F    = pd.concat([res["F_is"], res["F_oos"]]).sort_index()
    r    = pd.concat([res["r_is"], res["r_oos"]]).sort_index()
    rows = [(col, _row(F[col], 12, is_end, oos_start)) for col in F.columns]
    rows.append(("combined", _row(r, 12, is_end, oos_start)))
    return rows


def _comm_rows(is_end: str, oos_start: str) -> list[tuple[str, dict]]:
    from commodities.blend import run_comm_blend
    res  = run_comm_blend()
    F    = pd.concat([res["F_is"], res["F_oos"]]).sort_index()
    r    = pd.concat([res["r_is"], res["r_oos"]]).sort_index()
    rows = [(col, _row(F[col], 252, is_end, oos_start)) for col in F.columns]
    rows.append(("combined", _row(r, 252, is_end, oos_start)))
    return rows


def _eq_rows(is_end: str, oos_start: str) -> list[tuple[str, dict]]:
    from equities.production.pipeline import (
        _load_or_build_positions, _load_or_build_vf_data,
        PRODUCTION_SIGNALS, _make_signal_fn,
    )
    from equities.data import fetch_prices
    from equities.blend import build_managed_returns, msrr_ridge, combine_signals
    from core.backtest import run_backtest

    positions = _load_or_build_positions(rebuild=False, verbose=False)
    positions = {k: v for k, v in positions.items() if k in PRODUCTION_SIGNALS}
    vf_rebal  = set(_load_or_build_vf_data(None, None, None,
                                            rebuild=False, verbose=False)["rebal_dates"])
    prices  = fetch_prices(tickers=None)
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    ret_is  = returns.loc[returns.index <= is_end]
    ret_oos = returns.loc[returns.index >= oos_start]

    pos_is      = {k: v.loc[v.index <= is_end] for k, v in positions.items()}
    F_is_raw    = build_managed_returns(pos_is, ret_is)
    bw_all      = msrr_ridge(F_is_raw, lam=0.01)

    rows: list[tuple[str, dict]] = []
    for sig in PRODUCTION_SIGNALS:
        bw_s   = msrr_ridge(F_is_raw[[sig]], lam=0.0)
        comb_s = combine_signals({sig: positions[sig]}, bw_s)
        fn_s   = _make_signal_fn(comb_s)
        r_s    = pd.concat([
            run_backtest(ret_is,  fn_s, rebalance_dates=vf_rebal,
                         transaction_cost_bps=0.0)["daily_return"],
            run_backtest(ret_oos, fn_s, rebalance_dates=vf_rebal,
                         transaction_cost_bps=0.0)["daily_return"],
        ]).sort_index()
        rows.append((sig, _row(r_s, 252, is_end, oos_start)))

    fn_all = _make_signal_fn(combine_signals(positions, bw_all))
    r_all  = pd.concat([
        run_backtest(ret_is,  fn_all, rebalance_dates=vf_rebal,
                     transaction_cost_bps=0.0)["daily_return"],
        run_backtest(ret_oos, fn_all, rebalance_dates=vf_rebal,
                     transaction_cost_bps=0.0)["daily_return"],
    ]).sort_index()
    rows.append(("combined", _row(r_all, 252, is_end, oos_start)))
    return rows


# ── Entry point ────────────────────────────────────────────────────────────────

def run_signal_metrics(
    is_end:    str   = IS_END,
    oos_start: str   = OOS_START,
    vol_target: float = VOL_TARGET,
    include_equities: bool = True,
    verbose:   bool  = True,
) -> pd.DataFrame:
    """
    IS / OOS Sharpe and MDD for every signal and composite across all asset classes.

    Loads each asset class via its production blend function and computes Sharpe
    (annualised) and MDD (peak-to-trough at `vol_target` annual vol) for the IS
    and OOS periods.

    Parameters
    ----------
    is_end            Last IS date.  Default: "2020-12-31".
    oos_start         First OOS date.  Default: "2021-01-01".
    vol_target        Annual vol target for MDD scaling.  Default: 0.10.
    include_equities  Run equities position-level backtests.  Default: True.
                      Set False for a fast run (rates + FX + commodities only).
    verbose           Print tables.  Default: True.

    Returns
    -------
    pd.DataFrame with MultiIndex (asset_class, signal) and columns
    [IS_SR, IS_MDD, OOS_SR, OOS_MDD].
    """
    import contextlib, io as _io

    global VOL_TARGET
    VOL_TARGET = vol_target

    ASSET_SPECS = [
        ("rates",       "RATES",       "rolling Ridge MSRR", "daily",   _rates_rows),
        ("fx",          "FX",          "rolling Ridge MSRR", "monthly", _fx_rows),
        ("commodities", "COMMODITIES", "static Ridge MSRR",  "daily",   _comm_rows),
    ]
    if include_equities:
        ASSET_SPECS.append(
            ("equities", "EQUITIES", "msrr_ridge, TC=0", "daily", _eq_rows)
        )

    all_records: list[dict] = []

    for asset, label, method, freq_label, loader_fn in ASSET_SPECS:
        if verbose:
            print(f"Loading {asset} …")
        with contextlib.redirect_stdout(_io.StringIO()):
            rows = loader_fn(is_end, oos_start)

        if verbose:
            _print_header(f"{label}  —  {method}", is_end, oos_start, freq_label)
            for name, row in rows:
                _print_row(name, row, composite=(name == "combined"))
            _print_footer()

        for name, row in rows:
            all_records.append({
                "asset_class": asset,
                "signal":      name,
                **row,
            })

    df = pd.DataFrame(all_records).set_index(["asset_class", "signal"])
    return df
