"""
Production equities pipeline entry point.

run_eq_signals()
    1. Fetch prices + fundamentals
    2. Build all 4 signals through ViewFormation  (build_vf_data + apply_vf_params)
    3. Run a VF-schedule-aligned backtest per signal
    4. Report IS / OOS: Sharpe, max drawdown, annualized turnover

run_vf_ablation()
    Same data, same covariance matrices, but sweeps VF step configurations.
    Fast because covariances are cached; only the VF application is repeated.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

IS_END    = "2020-12-31"
OOS_START = "2021-01-01"

_VF_DATA_CACHE = os.path.join(os.path.dirname(__file__), "cache", "vf_data.pkl")

_XS_ABLATIONS: dict[str, dict] = {
    "baseline":   {},
    "no_grinold": {"use_grinold": False},
    "no_demean":  {"use_demean":  False},
    "no_ortho":   {"use_ortho":   False},
    "no_tanh":    {"use_tanh":    False},
    "with_rank":  {"use_rank":    True},
}

_TS_ABLATIONS: dict[str, dict] = {
    "baseline":  {},
    "no_ortho":  {"use_ortho": False},
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_or_build_vf_data(prices, fundamentals, returns, rebuild=False, verbose=True):
    os.makedirs(os.path.dirname(_VF_DATA_CACHE), exist_ok=True)
    if not rebuild and os.path.exists(_VF_DATA_CACHE):
        if verbose:
            print("Loading cached VF data …")
        with open(_VF_DATA_CACHE, "rb") as fh:
            return pickle.load(fh)

    if verbose:
        print("Building VF data — this takes a few minutes …")
    from equities.signals import build_vf_data
    vf_data = build_vf_data(prices=prices, fundamentals=fundamentals, returns=returns)
    with open(_VF_DATA_CACHE, "wb") as fh:
        pickle.dump(vf_data, fh)
    if verbose:
        print(f"Cached to {_VF_DATA_CACHE}\n")
    return vf_data


def _make_signal_fn(pos_df: pd.DataFrame):
    def signal_fn(date: pd.Timestamp) -> pd.Series:
        avail = pos_df.index[pos_df.index <= date]
        if avail.empty:
            return pd.Series(dtype=float)
        return pos_df.loc[avail[-1]]
    return signal_fn


def _stats_from_bt(bt: pd.DataFrame) -> dict:
    from performance import sharpe_ratio, max_drawdown
    return {
        "sharpe":       sharpe_ratio(bt["daily_return"]),
        "max_drawdown": max_drawdown(bt["nav"]),
        "ann_turnover": float(bt["turnover"].mean() * 252),
    }


def _annualized_sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 30 or r.std() < 1e-12:
        return np.nan
    return float(r.mean() / r.std() * np.sqrt(252))


# ── public API ────────────────────────────────────────────────────────────────

def run_eq_signals(
    is_end: str = IS_END,
    oos_start: str = OOS_START,
    rebuild: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Full equities pipeline: data → signals → ViewFormation → IS/OOS backtest.

    Rebalancing is aligned with the VF schedule (every 21 trading days after
    a 2-year warmup). TC is not applied — this evaluates signal quality.

    Returns
    -------
    dict keyed by signal name, each with:
        {"is":  {"sharpe", "max_drawdown", "ann_turnover"},
         "oos": {"sharpe", "max_drawdown", "ann_turnover"}}
    """
    from equities.data import fetch_prices, fetch_fundamentals
    from equities.signals import apply_vf_params
    from backtest import run_backtest

    prices       = fetch_prices(tickers=None)
    fundamentals = fetch_fundamentals(tickers=None)
    returns      = prices.pct_change(fill_method=None).dropna(how="all")

    vf_data   = _load_or_build_vf_data(prices, fundamentals, returns, rebuild, verbose)
    positions = apply_vf_params(vf_data)

    vf_rebal_dates = set(vf_data["rebal_dates"])
    ret_is  = returns.loc[returns.index <= is_end]
    ret_oos = returns.loc[returns.index >= oos_start]

    result: dict[str, dict] = {}

    for name, pos_df in positions.items():
        sig_fn = _make_signal_fn(pos_df)
        bt_is  = run_backtest(ret_is,  sig_fn, rebalance_dates=vf_rebal_dates,
                              transaction_cost_bps=0.0)
        bt_oos = run_backtest(ret_oos, sig_fn, rebalance_dates=vf_rebal_dates,
                              transaction_cost_bps=0.0)
        result[name] = {
            "is":  _stats_from_bt(bt_is),
            "oos": _stats_from_bt(bt_oos),
        }

    if verbose:
        _print_signals_table(result, is_end, oos_start)

    return result


def run_vf_ablation(
    is_end: str = IS_END,
    oos_start: str = OOS_START,
    rebuild: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Ablation study: test which ViewFormation steps help or hurt Sharpe.

    Uses managed returns (no TC, no weight drift) for speed.
    XS signals (momentum, mean_reversion, carry) are tested against _XS_ABLATIONS.
    TS signal (tsmom) is tested against _TS_ABLATIONS.

    Returns
    -------
    dict[config_name, dict[signal_name, {"is_sharpe", "oos_sharpe"}]]
    """
    from equities.data import fetch_prices, fetch_fundamentals
    from equities.signals import apply_vf_params
    from equities.blend import managed_return

    prices       = fetch_prices(tickers=None)
    fundamentals = fetch_fundamentals(tickers=None)
    returns      = prices.pct_change(fill_method=None).dropna(how="all")

    vf_data = _load_or_build_vf_data(prices, fundamentals, returns, rebuild, verbose)

    _XS_SIGNALS = {"momentum", "mean_reversion", "carry"}
    _TS_SIGNALS = {"tsmom"}

    # Determine which ablation set applies to each signal
    # XS configs vary xs_params; TS configs vary ts_params
    all_configs: dict[str, tuple[dict, dict]] = {}
    for name, xs_p in _XS_ABLATIONS.items():
        all_configs[f"xs/{name}"] = (xs_p, {})
    for name, ts_p in _TS_ABLATIONS.items():
        all_configs[f"ts/{name}"] = ({}, ts_p)

    results: dict[str, dict[str, dict]] = {}

    for config_key, (xs_p, ts_p) in all_configs.items():
        positions = apply_vf_params(vf_data, xs_params=xs_p, ts_params=ts_p or None)
        results[config_key] = {}
        for name, pos_df in positions.items():
            F     = managed_return(pos_df, returns)
            F_is  = F.loc[F.index <= is_end]
            F_oos = F.loc[F.index >= oos_start]
            results[config_key][name] = {
                "is_sharpe":  _annualized_sharpe(F_is),
                "oos_sharpe": _annualized_sharpe(F_oos),
            }

    if verbose:
        _print_ablation_table(results, is_end, oos_start)

    return results


# ── print helpers ─────────────────────────────────────────────────────────────

def _print_signals_table(result: dict, is_end: str, oos_start: str) -> None:
    W = 72
    print("=" * W)
    print("  EQUITIES SIGNALS — IS / OOS EVALUATION")
    print(f"  IS ≤ {is_end}   |   OOS ≥ {oos_start}")
    print("=" * W)
    print(f"  {'Signal':<18}  {'IS Sharpe':>10}  {'IS MaxDD':>9}  {'IS TO':>7}"
          f"  {'OOS Sharpe':>10}  {'OOS MaxDD':>9}  {'OOS TO':>7}")
    print("  " + "─" * (W - 2))

    for name, stats in result.items():
        i, o = stats["is"], stats["oos"]
        print(
            f"  {name:<18}"
            f"  {i['sharpe']:>+10.2f}"
            f"  {i['max_drawdown']*100:>8.1f}%"
            f"  {i['ann_turnover']:>6.1f}x"
            f"  {o['sharpe']:>+10.2f}"
            f"  {o['max_drawdown']*100:>8.1f}%"
            f"  {o['ann_turnover']:>6.1f}x"
        )

    print("=" * W)


def _print_ablation_table(results: dict, is_end: str, oos_start: str) -> None:
    from equities.signals import _SIGNAL_KINDS

    # Group: XS signals share xs/* configs; TS signals share ts/* configs
    xs_signals = [s for s, k in _SIGNAL_KINDS.items() if k == "xs"]
    ts_signals = [s for s, k in _SIGNAL_KINDS.items() if k == "ts"]

    xs_configs = {k.split("/")[1]: v for k, v in results.items() if k.startswith("xs/")}
    ts_configs = {k.split("/")[1]: v for k, v in results.items() if k.startswith("ts/")}

    W = 56

    def _block(signal_name: str, configs: dict) -> None:
        kind = _SIGNAL_KINDS[signal_name].upper()
        print(f"\n  {signal_name.upper()}  ({kind})")
        print("  " + "─" * (W - 2))
        print(f"  {'Config':<16}  {'IS Sharpe':>10}  {'OOS Sharpe':>10}")
        print("  " + "─" * (W - 2))
        for cfg_name, signal_stats in configs.items():
            s = signal_stats.get(signal_name, {})
            is_s  = f"{s['is_sharpe']:+.2f}"  if s and np.isfinite(s["is_sharpe"])  else "   n/a"
            oos_s = f"{s['oos_sharpe']:+.2f}" if s and np.isfinite(s["oos_sharpe"]) else "   n/a"
            print(f"  {cfg_name:<16}  {is_s:>10}  {oos_s:>10}")
        print("  " + "─" * (W - 2))

    print("\n" + "=" * W)
    print("  VF STEP ABLATION")
    print(f"  IS ≤ {is_end}  |  OOS ≥ {oos_start}")
    print("=" * W)

    for sig in xs_signals:
        _block(sig, xs_configs)
    for sig in ts_signals:
        _block(sig, ts_configs)
