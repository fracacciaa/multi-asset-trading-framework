"""
Production equities pipeline entry point.

run_eq_signals()
    1. Fetch prices + fundamentals
    2. Build all signals through ViewFormation  (build_vf_data + apply_vf_params)
    3. Run a VF-schedule-aligned backtest per signal
    4. Report IS / OOS: Sharpe, max drawdown, annualized turnover
    Production signal set: PRODUCTION_SIGNALS

try_eq_blend()
    Try all blend methods on PRODUCTION_SIGNALS managed returns.
    Reports IS / OOS: Sharpe, MDD, blend-weight turnover.
    Covers: equal-weight, Sharpe heuristic, MSRR (OLS/Ridge/LASSO),
    ENet/LASSO variants, inv-MSFE, PCR, PLS, Rolling MSRR, DMA.

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

PRODUCTION_SIGNALS: list[str] = ["mean_reversion", "carry", "tsmom", "threshold_mean_reversion"]

RESEARCH_SIGNALS: list[str] = [
    # production baseline (all 4, including momentum)
    "momentum", "tsmom", "mean_reversion", "carry",
    # momentum extensions
    "residual_momentum", "fip_momentum", "high_52w",
    # mean-reversion extensions
    "industry_reversal", "medium_reversal",
    "residual_reversal", "vol_scaled_reversal",
    "threshold_mean_reversion",
    # carry extensions
    "earnings_yield_all", "payout_momentum",
]

_SIGNAL_FAMILIES: dict[str, str] = {
    "momentum":            "momentum",
    "tsmom":               "momentum",
    "residual_momentum":   "momentum",
    "fip_momentum":        "momentum",
    "high_52w":            "momentum",
    "mean_reversion":      "mean_rev",
    "industry_reversal":   "mean_rev",
    "medium_reversal":     "mean_rev",
    "residual_reversal":   "mean_rev",
    "vol_scaled_reversal":        "mean_rev",
    "threshold_mean_reversion":   "mean_rev",
    "carry":               "carry",
    "earnings_yield_all":  "carry",
    "payout_momentum":     "carry",
}

_VF_DATA_CACHE          = os.path.join(os.path.dirname(__file__), "cache", "vf_data.pkl")
_VF_DATA_RESEARCH_CACHE = os.path.join(os.path.dirname(__file__), "cache", "vf_data_research.pkl")
_VF_POSITIONS_CACHE     = os.path.join(os.path.dirname(__file__), "cache", "vf_positions.pkl")

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


def _load_or_build_research_vf_data(prices, fundamentals, rebuild=False, verbose=True):
    """
    Build research VF data by augmenting the production cache with 7 new signals.

    Reuses the already-computed covariance matrices from the production cache so
    only the signal computation + VF application steps are repeated (~30s vs ~5min).
    """
    os.makedirs(os.path.dirname(_VF_DATA_RESEARCH_CACHE), exist_ok=True)
    if not rebuild and os.path.exists(_VF_DATA_RESEARCH_CACHE):
        if verbose:
            print("Loading cached research VF data …")
        with open(_VF_DATA_RESEARCH_CACHE, "rb") as fh:
            return pickle.load(fh)

    if verbose:
        print("Computing research signals and augmenting production VF cache …")

    # Load production VF data to reuse covariance matrices
    if not os.path.exists(_VF_DATA_CACHE):
        raise FileNotFoundError(
            f"Production VF cache not found at {_VF_DATA_CACHE}. "
            "Run run_eq_signals() first to build it."
        )
    with open(_VF_DATA_CACHE, "rb") as fh:
        prod_vf = pickle.load(fh)

    from equities.signals import (
        residual_momentum, fip_momentum, high_52w,
        industry_reversal, medium_reversal,
        residual_reversal, vol_scaled_reversal,
        threshold_mean_reversion,
        earnings_yield_all, payout_momentum,
    )

    if verbose:
        print("  computing 10 research signals …")
    extra = {
        "residual_momentum":        residual_momentum(prices),
        "fip_momentum":             fip_momentum(prices),
        "high_52w":                 high_52w(prices),
        "industry_reversal":        industry_reversal(prices),
        "medium_reversal":          medium_reversal(prices),
        "residual_reversal":        residual_reversal(prices),
        "vol_scaled_reversal":      vol_scaled_reversal(prices),
        "threshold_mean_reversion": threshold_mean_reversion(prices),
        "earnings_yield_all":       earnings_yield_all(prices, fundamentals["eps_fwd"]),
        "payout_momentum":          payout_momentum(prices, fundamentals["market_cap"]),
    }

    research_vf = {
        "raw_signals":  {**prod_vf["raw_signals"], **extra},
        "cov_matrices": prod_vf["cov_matrices"],
        "all_dates":    prod_vf["all_dates"],
        "rebal_dates":  prod_vf["rebal_dates"],
        "columns":      prod_vf["columns"],
    }

    with open(_VF_DATA_RESEARCH_CACHE, "wb") as fh:
        pickle.dump(research_vf, fh)
    if verbose:
        print(f"Cached to {_VF_DATA_RESEARCH_CACHE}\n")
    return research_vf


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
    positions = {k: v for k, v in positions.items() if k in PRODUCTION_SIGNALS}

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


# ── research signal evaluation ───────────────────────────────────────────────

def run_eq_research_signals(
    is_end: str = IS_END,
    oos_start: str = OOS_START,
    rebuild: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Extended signal evaluation: 4 production signals + 7 research variants.

    Reuses the production VF covariance cache so only signal computation and
    ViewFormation application need to run from scratch (~1–2 min on first call).

    New signals per family
    ----------------------
    Momentum   : residual_momentum, fip_momentum, high_52w
    Mean-rev   : industry_reversal, medium_reversal
    Carry      : earnings_yield_all, payout_momentum

    Returns
    -------
    dict keyed by signal name, each with:
        {"is": {"sharpe", "max_drawdown", "ann_turnover"},
         "oos": {"sharpe", "max_drawdown", "ann_turnover"},
         "family": str, "is_production": bool}
    """
    from equities.data import fetch_prices, fetch_fundamentals
    from equities.signals import apply_vf_params
    from backtest import run_backtest

    prices       = fetch_prices(tickers=None)
    fundamentals = fetch_fundamentals(tickers=None)
    returns      = prices.pct_change(fill_method=None).dropna(how="all")

    vf_data  = _load_or_build_research_vf_data(prices, fundamentals, rebuild, verbose)
    positions = apply_vf_params(vf_data)

    vf_rebal_dates = set(vf_data["rebal_dates"])
    ret_is  = returns.loc[returns.index <= is_end]
    ret_oos = returns.loc[returns.index >= oos_start]

    production_set = set(PRODUCTION_SIGNALS) | {"momentum"}
    result: dict[str, dict] = {}

    for name, pos_df in positions.items():
        sig_fn = _make_signal_fn(pos_df)
        bt_is  = run_backtest(ret_is,  sig_fn, rebalance_dates=vf_rebal_dates,
                              transaction_cost_bps=0.0)
        bt_oos = run_backtest(ret_oos, sig_fn, rebalance_dates=vf_rebal_dates,
                              transaction_cost_bps=0.0)
        result[name] = {
            "is":           _stats_from_bt(bt_is),
            "oos":          _stats_from_bt(bt_oos),
            "family":       _SIGNAL_FAMILIES.get(name, "other"),
            "is_production": name in production_set,
        }

    if verbose:
        _print_research_table(result, is_end, oos_start)

    return result


def _print_research_table(result: dict, is_end: str, oos_start: str) -> None:
    W = 86
    FAMILIES = ["momentum", "mean_rev", "carry", "other"]
    FAMILY_LABELS = {"momentum": "MOMENTUM", "mean_rev": "MEAN REVERSION",
                     "carry": "CARRY", "other": "OTHER"}

    print("\n" + "=" * W)
    print("  EQUITIES RESEARCH SIGNALS — IS / OOS EVALUATION")
    print(f"  IS ≤ {is_end}   |   OOS ≥ {oos_start}   |   * = production signal")
    print("=" * W)
    print(f"  {'Signal':<24}  {'IS Sharpe':>10}  {'IS MaxDD':>9}  {'IS TO':>7}"
          f"  {'OOS Sharpe':>10}  {'OOS MaxDD':>9}  {'OOS TO':>7}")
    print("  " + "─" * (W - 2))

    for fam in FAMILIES:
        members = [(n, v) for n, v in result.items() if v["family"] == fam]
        if not members:
            continue
        # production signals first, then research
        members.sort(key=lambda kv: (not kv[1]["is_production"], kv[0]))
        print(f"\n  ── {FAMILY_LABELS[fam]} ──")
        for name, stats in members:
            tag = "*" if stats["is_production"] else " "
            i, o = stats["is"], stats["oos"]
            print(
                f"  {tag}{name:<23}"
                f"  {i['sharpe']:>+10.2f}"
                f"  {i['max_drawdown']*100:>8.1f}%"
                f"  {i['ann_turnover']:>6.1f}x"
                f"  {o['sharpe']:>+10.2f}"
                f"  {o['max_drawdown']*100:>8.1f}%"
                f"  {o['ann_turnover']:>6.1f}x"
            )

    print("\n" + "=" * W)


# ── positions cache helper ────────────────────────────────────────────────────

def _load_or_build_positions(rebuild: bool = False, verbose: bool = True) -> dict:
    os.makedirs(os.path.dirname(_VF_POSITIONS_CACHE), exist_ok=True)
    if not rebuild and os.path.exists(_VF_POSITIONS_CACHE):
        if verbose:
            print("Loading cached VF positions …")
        with open(_VF_POSITIONS_CACHE, "rb") as fh:
            return pickle.load(fh)

    if verbose:
        print("Building VF positions — this takes ~5 min …")
    from equities.data import fetch_prices, fetch_fundamentals
    from equities.signals import build_vf_positions

    prices       = fetch_prices(tickers=None)
    fundamentals = fetch_fundamentals(tickers=None)
    returns      = prices.pct_change(fill_method=None)

    positions = build_vf_positions(
        prices=prices, fundamentals=fundamentals, returns=returns,
        warmup=504, rebal_freq=21, cov_lookback=504,
        vol_halflife=21, cor_halflife=126,
    )
    with open(_VF_POSITIONS_CACHE, "wb") as fh:
        pickle.dump(positions, fh)
    if verbose:
        print(f"Cached to {_VF_POSITIONS_CACHE}\n")
    return positions


# ── try_eq_blend ──────────────────────────────────────────────────────────────

def try_eq_blend(
    is_end: str = IS_END,
    oos_start: str = OOS_START,
    lookback: int = 252,
    min_obs: int = 63,
    rebuild: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Evaluate all blend methods on PRODUCTION_SIGNALS via full position-level backtest.

    Blend weights are estimated on IS vol-normalised managed returns only (no look-ahead).
    Each blended position DataFrame is then passed through run_backtest to produce the
    same Sharpe / MDD / annualised turnover metrics as run_eq_signals.

    Returns
    -------
    dict keyed by method name, each value:
        {"is": {"sharpe", "max_drawdown", "ann_turnover"},
         "oos": {"sharpe", "max_drawdown", "ann_turnover"},
         "type": str}
    """
    from equities.data import fetch_prices
    from equities.blend import (
        build_managed_returns, combine_signals,
        equal_weight, sharpe_heuristic, msrr_ridge, msrr_lasso,
        enet_blend, c_enet_blend, e_enet_blend,
        inv_msfe_blend, pcr_blend, pls_blend,
        RollingMSRR, DMABlend,
    )
    from backtest import run_backtest

    positions = _load_or_build_positions(rebuild, verbose)
    positions = {k: v for k, v in positions.items() if k in PRODUCTION_SIGNALS}

    # rebalance dates from vf_data cache (no rebuild, just reads pickle)
    vf_data = _load_or_build_vf_data(None, None, None, rebuild=False, verbose=False)
    vf_rebal_dates = set(vf_data["rebal_dates"])

    prices  = fetch_prices(tickers=None)
    returns = prices.pct_change(fill_method=None).dropna(how="all")

    ret_is  = returns.loc[returns.index <= is_end]
    ret_oos = returns.loc[returns.index >= oos_start]

    # vol-normalised managed returns for weight estimation (IS σ only)
    pos_is  = {k: v.loc[v.index <= is_end]    for k, v in positions.items()}
    pos_oos = {k: v.loc[v.index >= oos_start] for k, v in positions.items()}
    F_is_raw = build_managed_returns(pos_is,  ret_is)
    F_oos_raw = build_managed_returns(pos_oos, ret_oos)
    sigma  = F_is_raw.std().replace(0.0, np.nan)
    F_is   = F_is_raw.div(sigma)
    F_oos  = F_oos_raw.div(sigma)
    F_full = pd.concat([F_is, F_oos]).sort_index()

    names       = list(F_is.columns)
    all_tickers = sorted({t for pos in positions.values() for t in pos.columns})

    # ── backtest helpers ──────────────────────────────────────────────────────

    def _static_sig_fn(bw):
        combined = combine_signals(positions, bw)
        return _make_signal_fn(combined)

    def _rolling_sig_fn(weights_df):
        """Signal function backed by time-varying blend weights."""
        def signal_fn(date):
            avail = weights_df.index[weights_df.index <= date]
            if avail.empty:
                return pd.Series(dtype=float)
            w = weights_df.loc[avail[-1]].dropna()
            total = w.abs().sum()
            if total < 1e-12:
                return pd.Series(dtype=float)
            w = w / total
            row = pd.Series(0.0, index=all_tickers)
            for sig_name, wk in w.items():
                if sig_name not in positions:
                    continue
                pos = positions[sig_name]
                avail_p = pos.index[pos.index <= date]
                if avail_p.empty:
                    continue
                row = row.add(wk * pos.loc[avail_p[-1]].reindex(all_tickers).fillna(0.0),
                              fill_value=0.0)
            return row
        return signal_fn

    def _run(sig_fn):
        bt_is  = run_backtest(ret_is,  sig_fn, rebalance_dates=vf_rebal_dates,
                              transaction_cost_bps=0.0)
        bt_oos = run_backtest(ret_oos, sig_fn, rebalance_dates=vf_rebal_dates,
                              transaction_cost_bps=0.0)
        return _stats_from_bt(bt_is), _stats_from_bt(bt_oos)

    result: dict[str, dict] = {}

    # ── static blends ─────────────────────────────────────────────────────────
    static_blends = {
        "equal_weight": equal_weight(names),
        "sh_p1":        sharpe_heuristic(F_is, power=1),
        "sh_p2":        sharpe_heuristic(F_is, power=2),
        "msrr_ols":     msrr_ridge(F_is, lam=0.0),
        "msrr_ridge":   msrr_ridge(F_is, lam=0.01),
        "msrr_lasso":   msrr_lasso(F_is, lam=0.01),
        "enet":         enet_blend(F_is),
        "lasso":        enet_blend(F_is, l1_ratio=1.0),
        "c_enet":       c_enet_blend(F_is),
        "e_enet":       e_enet_blend(F_is),
        "e_lasso":      e_enet_blend(F_is, l1_ratio=1.0),
        "inv_msfe":     inv_msfe_blend(F_is),
        "pcr":          pcr_blend(F_is),
        "pls":          pls_blend(F_is),
    }
    for name, bw in static_blends.items():
        s_is, s_oos = _run(_static_sig_fn(bw))
        result[name] = {"is": s_is, "oos": s_oos, "type": "static"}

    # ── rolling MSRR ─────────────────────────────────────────────────────────
    rolling_mdls = {
        "roll_msrr_ols":   RollingMSRR(penalty="ols",                lookback=lookback, min_obs=min_obs),
        "roll_msrr_ridge": RollingMSRR(penalty="ridge", lam=0.01,    lookback=lookback, min_obs=min_obs),
        "roll_msrr_lasso": RollingMSRR(penalty="lasso", alpha=0.01,  lookback=lookback, min_obs=min_obs),
    }
    for name, mdl in rolling_mdls.items():
        w_all = mdl._roll(F_full)
        s_is, s_oos = _run(_rolling_sig_fn(w_all))
        result[name] = {"is": s_is, "oos": s_oos, "type": "rolling"}

    # ── DMA ──────────────────────────────────────────────────────────────────
    dma = DMABlend(alpha=0.99)
    dma.fit(F_is)
    w_dma_all = pd.concat([dma._weights_history, dma.predict_weights(F_oos)]).sort_index()
    s_is, s_oos = _run(_rolling_sig_fn(w_dma_all))
    result["dma"] = {"is": s_is, "oos": s_oos, "type": "dynamic"}

    if verbose:
        _print_blend_table(result, is_end, oos_start)

    return result


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


def _print_blend_table(result: dict, is_end: str, oos_start: str) -> None:
    def _fsr(v):  return f"{v:+.2f}"        if v is not None and not np.isnan(v) else "   n/a"
    def _fmdd(v): return f"{v*100:.1f}%"    if v is not None and not np.isnan(v) else "   n/a"
    def _fto(v):  return f"{v:.1f}x"        if v is not None and not np.isnan(v) else "   n/a"

    W = 97
    print("=" * W)
    print("  EQUITIES BLEND EVALUATION")
    print(f"  IS ≤ {is_end}   |   OOS ≥ {oos_start}")
    print("=" * W)
    print(
        f"  {'Method':<22}  {'Type':<8}"
        f"  {'IS Sharpe':>10}  {'IS MDD':>8}  {'IS TO':>7}"
        f"  {'OOS Sharpe':>10}  {'OOS MDD':>8}  {'OOS TO':>7}"
    )
    print("  " + "─" * (W - 2))

    prev_type = None
    for name, stats in result.items():
        t = stats["type"]
        if prev_type is not None and t != prev_type:
            print("  " + "─" * (W - 2))
        prev_type = t
        i, o = stats["is"], stats["oos"]
        print(
            f"  {name:<22}  {t:<8}"
            f"  {_fsr(i['sharpe']):>10}  {_fmdd(i['max_drawdown']):>8}  {_fto(i['ann_turnover']):>7}"
            f"  {_fsr(o['sharpe']):>10}  {_fmdd(o['max_drawdown']):>8}  {_fto(o['ann_turnover']):>7}"
        )

    print("=" * W)
    print("  MDD = NAV peak-to-trough (%). TO = annualised one-way position turnover.")


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
