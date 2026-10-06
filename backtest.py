"""
Backtest engine.
Simulates the strategy walk-forward: rebalance on a fixed schedule,
apply transaction costs, and record daily P&L.
"""

import pandas as pd
import numpy as np
from typing import Callable


def run_backtest(
    returns: pd.DataFrame,
    signal_fn: Callable[[pd.Timestamp], pd.Series],
    rebalance_freq: str = "M",
    transaction_cost_bps: float = 5.0,
    initial_capital: float = 1_000_000.0,
) -> pd.DataFrame:
    """
    Walk-forward backtest.

    returns:              daily returns DataFrame (dates x instruments)
    signal_fn:            callable — takes a rebalance date, returns target weights
                          as a pd.Series indexed by instrument. Return None or an
                          empty Series to hold current weights for that period.
    rebalance_freq:       pandas offset alias ('M', 'W', 'Q', ...)
    transaction_cost_bps: one-way cost in basis points
    initial_capital:      starting NAV

    Returns a DataFrame indexed by date with columns:
        nav           — portfolio value
        daily_return  — net daily return (TC deducted on rebalance days)
        turnover      — one-way turnover on rebalance days, 0 otherwise
    """
    trading_days = returns.index
    freq = rebalance_freq.rstrip("E") + "E" if rebalance_freq in ("M", "Q", "Y") else rebalance_freq
    rebalance_dates = set(returns.resample(freq).last().index)

    navs = np.empty(len(trading_days))
    daily_rets = np.zeros(len(trading_days))
    turnovers = np.zeros(len(trading_days))

    nav = float(initial_capital)
    weights = pd.Series(0.0, index=returns.columns)

    for i, date in enumerate(trading_days):
        nav_prev = nav

        # Accrue daily return at current weights
        day_rets = returns.loc[date].reindex(weights.index).fillna(0.0)
        port_ret = float((weights * day_rets).sum())
        nav *= (1 + port_ret)

        # Drift weights to end-of-day values
        if abs(1 + port_ret) > 1e-12:
            weights = weights * (1 + day_rets) / (1 + port_ret)

        # Rebalance at close — new weights take effect the next open
        if date in rebalance_dates:
            new_weights = signal_fn(date)
            if new_weights is not None and len(new_weights) > 0:
                new_weights = new_weights.reindex(returns.columns).fillna(0.0)
                tc = apply_transaction_costs(new_weights, weights, transaction_cost_bps)
                to = float((new_weights - weights).abs().sum() / 2)
                nav *= (1 - tc)
                turnovers[i] = to
                weights = new_weights

        navs[i] = nav
        daily_rets[i] = nav / nav_prev - 1

    return pd.DataFrame(
        {"nav": navs, "daily_return": daily_rets, "turnover": turnovers},
        index=trading_days,
    )


def apply_transaction_costs(
    weights_new: pd.Series,
    weights_old: pd.Series,
    cost_bps: float,
) -> float:
    """Return total transaction cost as a fraction of NAV."""
    turnover = (weights_new - weights_old).abs().sum() / 2
    return float(turnover * cost_bps / 10_000)
