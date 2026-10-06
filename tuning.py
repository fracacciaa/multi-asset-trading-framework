"""
Hyperparameter tuning for the ViewFormationXS pipeline.

Runs a grid search over the joint space of risk-model and view-formation
parameters, evaluates each combination on the validation period, and returns
the best parameter set (maximizing Sharpe ratio).

Train / Val / Test split (recommended for 2005–2026 data):
    Train : 2005 – 2014
    Val   : 2015 – 2019
    Test  : 2020 – 2026  (never touched during tuning)
"""

import itertools
import pandas as pd
import numpy as np
from typing import Optional

from optimization import estimate_covariance
from view_formation import ViewFormationXS
from backtest import run_backtest, apply_transaction_costs
import performance


_DEFAULT_GRID = {
    "vol_halflife": [10, 21, 42],
    "cor_halflife": [63, 126, 252],
    "shrinkage":    [False, True],
    "tanh_scale":   [0.75, 1.25, 2.0],
    "pct_prop":     [0.0, 0.5, 1.0],
}


def tune_view_formation(
    signal_df: pd.DataFrame,
    returns: pd.DataFrame,
    train_end: str,
    val_end: str,
    param_grid: Optional[dict] = None,
    rebalance_freq: str = "M",
    transaction_cost_bps: float = 5.0,
    cov_window: int = 756,
) -> dict:
    """
    Grid-search over ViewFormationXS + risk-model hyperparameters.

    Evaluates each combination on the validation period and returns the
    param set with the highest Sharpe ratio.

    Parameters
    ----------
    signal_df      : date × ticker raw signal (e.g. momentum z-scores)
    returns        : date × ticker daily returns
    train_end      : last date of training period (str, e.g. '2014-12-31')
    val_end        : last date of validation period (str, e.g. '2019-12-31')
    param_grid     : dict of lists; defaults to _DEFAULT_GRID
    rebalance_freq : pandas offset alias for the backtest
    transaction_cost_bps : one-way TC in bps
    cov_window     : max lookback (rows) for covariance estimation

    Returns
    -------
    dict with keys: best_params, val_sharpe, results_df (val-period NAV)
    """
    grid = param_grid or _DEFAULT_GRID

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"Tuning: {len(combos)} combinations over val period {train_end} → {val_end}")

    best_sharpe = -np.inf
    best_params = None
    best_results = None

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))

        risk_params  = {k: params[k] for k in ("vol_halflife", "cor_halflife", "shrinkage")}
        view_params  = {k: params[k] for k in ("tanh_scale", "pct_prop")}
        view_params["use_rank"] = True
        view_params["desired_vol"] = 0.10

        try:
            signal_fn = _make_signal_fn(
                signal_df, returns, risk_params, view_params, cov_window
            )
            # Backtest runs over train+val; we score only the val slice
            val_returns = returns.loc[train_end:val_end]
            results = run_backtest(
                returns=val_returns,
                signal_fn=signal_fn,
                rebalance_freq=rebalance_freq,
                transaction_cost_bps=transaction_cost_bps,
            )
            sharpe = performance.sharpe_ratio(results["daily_return"].dropna())
        except Exception:
            sharpe = -np.inf

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = params
            best_results = results

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(combos)} done  |  best so far: {best_sharpe:.3f}")

    print(f"\nBest params: {best_params}")
    print(f"Val Sharpe:  {best_sharpe:.3f}")

    return {
        "best_params":  best_params,
        "val_sharpe":   best_sharpe,
        "results_df":   best_results,
    }


def _make_signal_fn(
    signal_df: pd.DataFrame,
    returns: pd.DataFrame,
    risk_params: dict,
    view_params: dict,
    cov_window: int,
):
    """Return a signal_fn closure suitable for run_backtest."""

    def signal_fn(date: pd.Timestamp) -> pd.Series:
        sig_row = signal_df.loc[:date].dropna(how="all")
        if len(sig_row) == 0:
            return pd.Series(dtype=float)
        signal = sig_row.iloc[-1].dropna()

        # Build risk model from returns up to this date (capped at cov_window rows)
        ret_history = returns.loc[:date].dropna(how="all").iloc[-cov_window:]
        if len(ret_history) < risk_params["vol_halflife"] * 2:
            return pd.Series(dtype=float)

        cov = estimate_covariance(
            ret_history,
            vol_halflife=risk_params["vol_halflife"],
            cor_halflife=risk_params["cor_halflife"],
            shrinkage=risk_params["shrinkage"],
        )

        common = signal.index.intersection(cov.index)
        if len(common) < 10:
            return pd.Series(dtype=float)

        vf = ViewFormationXS(signal.loc[common], cov.loc[common, common], view_params)
        return vf.generate_view()

    return signal_fn
