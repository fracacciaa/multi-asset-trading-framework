"""
Entry point.
Orchestrates data loading, signal generation, optimization, backtest and performance reporting.
"""

import pandas as pd

from equities import data as eq_data, signals as eq_signals
from rates import data as ra_data, signals as ra_signals
from fx import data as fx_data, signals as fx_signals
from commodities import data as co_data, signals as co_signals
from credit import data as cr_data, signals as cr_signals

import optimization
import constraints
import backtest
import performance


# ── Configuration ────────────────────────────────────────────────────────────

START = "2010-01-01"
END   = "2024-12-31"
REBALANCE_FREQ = "M"
TRANSACTION_COST_BPS = 5.0
RISK_FREE_RATE = 0.0


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data() -> dict:
    return {
        "equities": {
            "prices":       eq_data.fetch_prices(eq_data.fetch_universe(), START, END),
            "fundamentals": eq_data.fetch_fundamentals(eq_data.fetch_universe(), START, END),
        },
        "rates": {
            "yields": ra_data.fetch_yields([], [], START, END),
        },
        "fx": {
            "spot":     fx_data.fetch_spot(fx_data.fetch_universe(), START, END),
            "forwards": fx_data.fetch_forwards(fx_data.fetch_universe(), ["1M", "3M"], START, END),
        },
        "commodities": {
            "futures":        co_data.fetch_futures(co_data.fetch_universe(), START, END),
            "term_structure": co_data.fetch_term_structure(co_data.fetch_universe(), START, END),
        },
        "credit": {
            "spreads": cr_data.fetch_cds_spreads(cr_data.fetch_universe(), ["5Y"], START, END),
        },
    }


# ── Signal generation ─────────────────────────────────────────────────────────

def compute_signals(data: dict) -> dict[str, pd.DataFrame]:
    return {
        "equities":    eq_signals.combined_signal(data["equities"]["prices"], data["equities"]["fundamentals"]),
        "rates":       ra_signals.combined_signal(data["rates"]["yields"]),
        "fx":          fx_signals.combined_signal(data["fx"]["spot"], data["fx"]["forwards"]),
        "commodities": co_signals.combined_signal(data["commodities"]["futures"], data["commodities"]["term_structure"]),
        "credit":      cr_signals.combined_signal(data["credit"]["spreads"]),
    }


# ── Optimization ──────────────────────────────────────────────────────────────

def build_target_weights(signals_snapshot: dict[str, pd.Series], returns_history: pd.DataFrame) -> pd.Series:
    mu, omega = optimization.build_views(signals_snapshot)
    cov = optimization.estimate_covariance(returns_history)
    cons = constraints.build_constraints(universe=list(mu.index))
    return optimization.optimize(mu, cov, cons)


# ── Backtest ──────────────────────────────────────────────────────────────────

def run(data: dict) -> pd.DataFrame:
    signals_ts = compute_signals(data)

    all_returns = pd.concat(
        [data["equities"]["prices"].pct_change()],
        axis=1,
    ).dropna()

    def signal_fn(date: pd.Timestamp) -> pd.Series:
        snapshot = {ac: s.loc[:date].iloc[-1] for ac, s in signals_ts.items()}
        history = all_returns.loc[:date]
        return build_target_weights(snapshot, history)

    results = backtest.run_backtest(
        returns=all_returns,
        signal_fn=signal_fn,
        rebalance_freq=REBALANCE_FREQ,
        transaction_cost_bps=TRANSACTION_COST_BPS,
    )
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    data = load_data()

    print("Running backtest...")
    results = run(data)

    print("Performance summary:")
    stats = performance.summary(results["nav"])
    print(stats.to_string())
