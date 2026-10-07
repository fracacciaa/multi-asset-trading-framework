"""
Research backtest wrapper.
Runs run_backtest() with signals processed through ViewFormationXS.

Every signal always goes through the full view formation pipeline — the same
path that would be used in production. Raw signals are never used directly as
weights.
"""

import pandas as pd
import numpy as np
from typing import Optional

from backtest import run_backtest
from optimization import estimate_covariance
from view_formation import ViewFormationXS, _DEFAULT_PARAMS


def run_research_backtest(
    signal_dfs: dict,
    returns: pd.DataFrame,
    vf_params: Optional[dict] = None,
    cov_window: int = 756,
    vol_halflife: int = 21,
    cor_halflife: int = 63,
    shrinkage_intensity: float = 0.0,
    rebalance_freq: str = "M",
    transaction_cost_bps: float = 5.0,
    initial_capital: float = 1_000_000.0,
) -> pd.DataFrame:
    """
    Research-mode walk-forward backtest with ViewFormationXS.

    signal_dfs:           dict mapping asset_class -> signal DataFrame (date × ticker).
                          Signals should be raw (not z-scored); VF handles normalization.
    returns:              daily returns DataFrame (date × ticker), used for both the
                          backtest P&L and covariance estimation.
    vf_params:            ViewFormationXS params dict; defaults to tuned params.
    cov_window:           rows of return history for covariance estimation (default 756 = 3yr)
    vol_halflife:         EWM halflife for volatility in the risk model (default 21)
    cor_halflife:         EWM halflife for correlation in the risk model (default 63)
    shrinkage_intensity:  linear shrinkage toward identity (default 0.0)
    rebalance_freq:       pandas offset alias ('M', 'W', 'Q', ...)
    transaction_cost_bps: one-way cost in basis points
    initial_capital:      starting NAV

    Returns the same DataFrame as run_backtest(): nav, daily_return, turnover
    """
    params = {**_DEFAULT_PARAMS, **(vf_params or {})}

    def signal_fn(date: pd.Timestamp) -> pd.Series:
        # Collect the latest available signal row for each asset class
        signals = {}
        for ac, sig_df in signal_dfs.items():
            available = sig_df.loc[:date].dropna(how="all")
            if available.empty:
                continue
            sig = available.iloc[-1].dropna()
            if len(sig) >= 10:
                signals[ac] = sig

        if not signals:
            return pd.Series(dtype=float)

        # Combine signals across asset classes (simple average for now)
        combined = pd.concat(list(signals.values())).groupby(level=0).mean()

        # Estimate covariance from returns up to this date
        ret_history = returns.loc[:date].dropna(how="all").iloc[-cov_window:]
        if len(ret_history) < vol_halflife * 2:
            return pd.Series(dtype=float)

        cov = estimate_covariance(
            ret_history,
            vol_halflife=vol_halflife,
            cor_halflife=cor_halflife,
            shrinkage_intensity=shrinkage_intensity,
        )

        common = combined.index.intersection(cov.index)
        if len(common) < 10:
            return pd.Series(dtype=float)

        return ViewFormationXS(
            combined.loc[common], cov.loc[common, common], params
        ).generate_view()

    return run_backtest(
        returns=returns,
        signal_fn=signal_fn,
        rebalance_freq=rebalance_freq,
        transaction_cost_bps=transaction_cost_bps,
        initial_capital=initial_capital,
    )
