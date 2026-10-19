"""
Research backtest wrapper.
Runs run_backtest() with signals processed through view formation.

Each signal always goes through the full view formation pipeline — the same
path that would be used in production. Raw signals are never used directly as
weights.

Supports weight-level blending of multiple (signal, VF) pairs: each config is
run through its own VF independently, then the resulting weight vectors are
averaged and re-scaled to the target vol.
"""

import pandas as pd
import numpy as np
from typing import Optional

from backtest import run_backtest
from optimization import estimate_covariance
from view_formation import ViewFormationXS, ViewFormationTS, _DEFAULT_PARAMS, _DEFAULT_TS_PARAMS


def run_research_backtest(
    signal_configs: list,
    returns: pd.DataFrame,
    desired_vol: float = 0.10,
    cov_window: int = 756,
    vol_halflife: int = 21,
    cor_halflife: int = 63,
    shrinkage_intensity: float = 0.0,
    rebalance_freq: str = "M",
    transaction_cost_bps: float = 5.0,
    initial_capital: float = 1_000_000.0,
) -> pd.DataFrame:
    """
    Research-mode walk-forward backtest with weight-level signal blending.

    signal_configs:       list of (signal_dfs, vf_class, vf_params) tuples.
                          signal_dfs:  dict mapping asset_class -> signal DataFrame
                                       (date × ticker, raw — VF handles normalization).
                          vf_class:    ViewFormationXS or ViewFormationTS.
                          vf_params:   optional param overrides for this VF; None = defaults.
                          Each config is run through its own VF independently;
                          resulting weight vectors are averaged then re-scaled to desired_vol.
    returns:              daily returns DataFrame (date × ticker).
    desired_vol:          annualized target vol for the blended output (default 0.10).
    cov_window:           rows of return history for covariance estimation (default 756 = 3yr)
    vol_halflife:         EWM halflife for volatility in the risk model (default 21)
    cor_halflife:         EWM halflife for correlation in the risk model (default 63)
    shrinkage_intensity:  linear shrinkage toward identity (default 0.0)
    rebalance_freq:       pandas offset alias ('M', 'W', 'Q', ...)
    transaction_cost_bps: one-way cost in basis points
    initial_capital:      starting NAV

    Returns the same DataFrame as run_backtest(): nav, daily_return, turnover.

    Single-signal example:
        run_research_backtest(
            [({"csmom": csmom_df}, ViewFormationXS, None)],
            returns,
        )

    Multi-signal blend example:
        run_research_backtest(
            [
                ({"csmom": csmom_df}, ViewFormationXS, None),
                ({"tsmom": tsmom_df}, ViewFormationTS, None),
                ({"mr_xs": mr_df},    ViewFormationXS, None),
                ({"mr_ts": mr_df},    ViewFormationTS, None),
            ],
            returns,
        )
    """

    def signal_fn(date: pd.Timestamp) -> pd.Series:
        # Estimate covariance once — shared across all configs
        ret_history = returns.loc[:date].dropna(how="all").iloc[-cov_window:]
        if len(ret_history) < vol_halflife * 2:
            return pd.Series(dtype=float)

        cov = estimate_covariance(
            ret_history,
            vol_halflife=vol_halflife,
            cor_halflife=cor_halflife,
            shrinkage_intensity=shrinkage_intensity,
        )

        views = []
        for signal_dfs, vf_class, vf_params in signal_configs:
            # Collect latest available signal row for each asset class in this config
            signals = {}
            for ac, sig_df in signal_dfs.items():
                available = sig_df.loc[:date].dropna(how="all")
                if available.empty:
                    continue
                sig = available.iloc[-1].dropna()
                if len(sig) >= 10:
                    signals[ac] = sig

            if not signals:
                continue

            combined = pd.concat(list(signals.values())).groupby(level=0).mean()
            common = combined.index.intersection(cov.index)
            if len(common) < 10:
                continue

            default = _DEFAULT_TS_PARAMS if vf_class is ViewFormationTS else _DEFAULT_PARAMS
            params = {**default, **(vf_params or {})}

            view = vf_class(
                combined.loc[common], cov.loc[common, common], params
            ).generate_view()
            views.append(view)

        if not views:
            return pd.Series(dtype=float)

        # Blend: align on union of tickers, fill missing with 0, equal-weight average
        blended = pd.concat(views, axis=1).fillna(0.0).mean(axis=1)

        # Re-scale to desired_vol — averaging N views reduces vol by ~1/√N
        w = blended.reindex(cov.index).fillna(0.0).values
        port_vol = np.sqrt(max(float(w @ cov.values @ w), 1e-12))
        target_daily = desired_vol / np.sqrt(252)
        return blended * (target_daily / port_vol)

    return run_backtest(
        returns=returns,
        signal_fn=signal_fn,
        rebalance_freq=rebalance_freq,
        transaction_cost_bps=transaction_cost_bps,
        initial_capital=initial_capital,
    )
