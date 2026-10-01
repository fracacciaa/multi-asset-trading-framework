"""
Research backtest wrapper.
Runs run_backtest() using raw signal DataFrames as portfolio weights,
without requiring the full optimization pipeline.

Each asset class contributes a daily view (signal row). Missing views are held
at the last available weights for that asset class — no forced liquidation.
"""

import pandas as pd
import numpy as np
from backtest import run_backtest


def run_research_backtest(
    signal_dfs: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    rebalance_freq: str = "M",
    transaction_cost_bps: float = 5.0,
    initial_capital: float = 1_000_000.0,
) -> pd.DataFrame:
    """
    Research-mode walk-forward backtest.

    signal_dfs:           dict mapping asset_class -> signal DataFrame (date x instrument).
                          Signal values are used directly as portfolio weights.
                          Instruments not present in `returns` are ignored.
    returns:              daily returns DataFrame (dates x instruments)
    rebalance_freq:       pandas offset alias ('M', 'W', 'Q', ...)
    transaction_cost_bps: one-way cost in basis points
    initial_capital:      starting NAV

    Returns the same DataFrame as run_backtest():
        nav, daily_return, turnover
    """
    prev_weights: dict[str, pd.Series] = {}

    def signal_fn(date: pd.Timestamp) -> pd.Series:
        for ac, sig_df in signal_dfs.items():
            # Use data available up to (and including) the rebalance date
            available = sig_df.loc[:date].dropna(how="all")
            if len(available) == 0:
                continue
            view = available.iloc[-1].dropna()
            if len(view) == 0:
                continue
            # Keep only instruments that exist in the returns universe
            view = view.reindex(returns.columns).dropna()
            if len(view) > 0:
                prev_weights[ac] = view

        if not prev_weights:
            return pd.Series(dtype=float)

        return pd.concat(list(prev_weights.values()))

    return run_backtest(
        returns=returns,
        signal_fn=signal_fn,
        rebalance_freq=rebalance_freq,
        transaction_cost_bps=transaction_cost_bps,
        initial_capital=initial_capital,
    )
