"""
Commodity signal research harness: full 18-signal comparison.

Production signals live in commodities.backtest.run_comparison (4 signals).
This module runs the complete universe for exploratory / robustness work.

Usage
-----
    from commodities.research import run_full_comparison
    results = run_full_comparison()
    print(results.to_string())
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from commodities.data import (
    fetch_prices,
    fetch_prices_power,
    fetch_term_structure,
    fetch_term_structure_far,
    fetch_term_structure_power,
    fetch_tickers,
    fetch_tickers_far,
    fetch_tickers_power,
    tenor_from_tickers,
)
from commodities.signals import (
    COMMODITY_SECTORS,
    tsmom,
    ma_crossover,
    channel_breakout,
    xs_carry,
    xs_carry_enhanced,
    curve_curvature,
    momentum,
    mean_reversion,
)
from commodities.backtest import (
    PARAMS,
    TD,
    roll_safe_returns,
    vol_scale,
    rebalance,
    book_scale,
    run_backtest,
    evaluate,
    _subperiod_sharpes,
)


def _build_all_signals(
    prices: pd.DataFrame,
    ts: dict,
    tk: dict,
    p: dict,
    ts_far: dict = None,
    tk_far: dict = None,
) -> dict[str, pd.DataFrame]:
    """Full 18-signal set for research comparisons."""
    F1 = ts["front"]
    F2 = ts["second"]
    F4 = ts["fourth"] if "fourth" in ts else None

    tenor_12 = tenor_from_tickers(tk["front"], tk["second"])
    tenor_14 = tenor_from_tickers(tk["front"], tk["fourth"]) if "fourth" in tk else None
    front_tickers = tk["front"].reindex(columns=prices.columns)

    sigs: dict[str, pd.DataFrame] = {}

    # ── Trend ──────────────────────────────────────────────────────────────────
    sigs["tsmom_21"]     = tsmom(prices, lookbacks=(21,),           mode=p["mode"], clip=p["clip"])
    sigs["tsmom_63"]     = tsmom(prices, lookbacks=(63,),           mode=p["mode"], clip=p["clip"])
    sigs["tsmom_252"]    = tsmom(prices, lookbacks=(252,),          mode=p["mode"], clip=p["clip"])
    sigs["tsmom_blend"]  = tsmom(prices, lookbacks=p["lookbacks"],  mode=p["mode"], clip=p["clip"])
    sigs["tsmom_scaled"] = tsmom(prices, lookbacks=p["lookbacks"],  mode="clip",    clip=p["clip"])

    sigs["ma_20_120"]    = ma_crossover(prices, fast=20,  slow=120)
    sigs["ma_50_200"]    = ma_crossover(prices, fast=50,  slow=200)

    sigs["breakout_55"]  = channel_breakout(prices, window=55)
    sigs["breakout_120"] = channel_breakout(prices, window=120)

    # ── Carry (baseline, tenor-normalised) ────────────────────────────────────
    sigs["xs_carry_f1f2"] = xs_carry(F1, F2, tenor=tenor_12).reindex(columns=prices.columns)

    if F4 is not None:
        f4 = F4.reindex(columns=prices.columns)
        sigs["xs_carry_f1f4"] = xs_carry(
            F1.reindex(columns=prices.columns), f4, tenor=tenor_14,
        ).reindex(columns=prices.columns)
        sigs["curvature"] = curve_curvature(
            F1.reindex(columns=prices.columns),
            F2.reindex(columns=prices.columns),
            f4,
        ).reindex(columns=prices.columns)

    # ── Carry (enhanced: sector-neutral + seasonal + roll filter) ─────────────
    sigs["xs_carry_enh_f1f2"] = xs_carry_enhanced(
        F1.reindex(columns=prices.columns),
        F2.reindex(columns=prices.columns),
        tenor=tenor_12,
        tickers_front=front_tickers,
        sector_map=COMMODITY_SECTORS,
        seasonal=True,
        roll_filter_days=5,
    ).reindex(columns=prices.columns)

    if F4 is not None:
        sigs["xs_carry_enh_f1f4"] = xs_carry_enhanced(
            F1.reindex(columns=prices.columns),
            F4.reindex(columns=prices.columns),
            tenor=tenor_14,
            tickers_front=front_tickers,
            sector_map=COMMODITY_SECTORS,
            seasonal=True,
            roll_filter_days=5,
        ).reindex(columns=prices.columns)

    if ts_far is not None and "eighth" in ts_far:
        F8 = ts_far["eighth"].reindex(columns=prices.columns)
        tk_f8 = tk_far.get("eighth") if tk_far is not None else None
        tenor_18 = tenor_from_tickers(tk["front"], tk_f8) if tk_f8 is not None else None
        sigs["xs_carry_f1f8"] = xs_carry(
            F1.reindex(columns=prices.columns), F8, tenor=tenor_18,
        ).reindex(columns=prices.columns)
        sigs["xs_carry_enh_f1f8"] = xs_carry_enhanced(
            F1.reindex(columns=prices.columns), F8,
            tenor=tenor_18,
            tickers_front=front_tickers,
            sector_map=COMMODITY_SECTORS,
            seasonal=True,
            roll_filter_days=5,
        ).reindex(columns=prices.columns)

    # ── Legacy ─────────────────────────────────────────────────────────────────
    sigs["mom_magnitude"] = momentum(prices)
    sigs["meanrev_21"]    = mean_reversion(prices)

    return sigs


def run_full_comparison(
    start: Optional[str] = None,
    end: Optional[str] = None,
    params: dict = PARAMS,
    include_power: bool = False,
) -> pd.DataFrame:
    """
    Run all 18 signals through the full pipeline and return a summary DataFrame.

    Columns: sharpe, ann_ret, ann_vol, max_dd, skew, calmar, n_days,
             avg_turnover_pa, + one column per 3-year sub-period block.
    """
    p = params

    prices_gen = fetch_prices(start=start, end=end)
    ts_gen     = fetch_term_structure(start=start, end=end)
    tk_gen     = fetch_tickers(start=start, end=end)
    ts_far     = fetch_term_structure_far(start=start, end=end)
    tk_far     = fetch_tickers_far(start=start, end=end)

    if include_power:
        prices_pw = fetch_prices_power(start=start, end=end)
        ts_pw     = fetch_term_structure_power(start=start, end=end)
        tk_pw     = fetch_tickers_power(start=start, end=end)

        prices = pd.concat([prices_gen, prices_pw], axis=1).sort_index()
        ts     = {k: pd.concat([ts_gen[k], ts_pw[k]], axis=1).sort_index() for k in ts_gen}
        tk     = {k: pd.concat([tk_gen[k], tk_pw.get(k, pd.DataFrame())], axis=1).sort_index()
                  for k in tk_gen}
    else:
        prices, ts, tk = prices_gen, ts_gen, tk_gen

    front_tickers = tk["front"].reindex(columns=prices.columns)
    returns = roll_safe_returns(prices, front_tickers)

    raw_signals = _build_all_signals(prices, ts, tk, p, ts_far=ts_far, tk_far=tk_far)

    rows: dict[str, dict] = {}
    for label, raw_sig in raw_signals.items():
        sig_aligned = raw_sig.reindex(index=returns.index, columns=returns.columns)
        w = vol_scale(sig_aligned, returns, target_vol=p["asset_vol"], com=p["vol_com"])
        w = rebalance(w, rule=p["rebal"])
        w = book_scale(w, returns, target_vol=p["book_vol"], cov_hl=p["cov_hl"], k_max=p["k_max"])
        bt = run_backtest(w, returns, lag=p["lag"], cost_bps=p["cost_bps"])
        pnl = bt["pnl"]
        row = evaluate(pnl)
        row["avg_turnover_pa"] = float(bt["turnover"].mean() * TD)
        row.update({f"sr_{yr}": v for yr, v in _subperiod_sharpes(pnl).items()})
        rows[label] = row

    result = pd.DataFrame(rows).T
    result.index.name = "signal"
    return result.sort_values("sharpe", ascending=False)
