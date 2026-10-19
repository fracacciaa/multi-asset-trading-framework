"""
RV signal Sharpe table — all 4 NS/NSS models × 3 structures × VFXS + VFTS.

Runs a walk-forward backtest for each (model, structure, VF) combination and
reports annualised Sharpe ratios.
"""

import numpy as np
import pandas as pd

from backtest_research import run_research_backtest
from view_formation import ViewFormationXS, ViewFormationTS
from rates.production.mean_reversion.rv import (
    build_rv_signals, build_rv_returns,
)
from rates.production.nelson_siegel import calibrate_lambda
from rates.production.ois_data import OIS_MARKETS

# ── Config ─────────────────────────────────────────────────────────────────────

MODELS     = ["ns_calibrated", "ns_free", "nss_calibrated", "nss_free"]
STRUCTURES = ["slope_2s10s", "fly_2s5s10s", "fly_5s10s20s"]
START      = "2010-01-01"   # before this fewer than 10 OIS markets exist → backtest skips all dates
END        = None           # full history

# ── Build ──────────────────────────────────────────────────────────────────────

def _sharpe(daily_returns: pd.Series) -> float:
    s = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 60:
        return np.nan
    std = s.std()
    if std < 1e-12:
        return np.nan
    return float(s.mean() / std * np.sqrt(252))


def _winsorize(ret_df: pd.DataFrame, n_sigma: float = 5.0) -> pd.DataFrame:
    """Clip each column at ±n_sigma to prevent extreme jumps from dominating the cov."""
    return ret_df.apply(lambda c: c.clip(lower=c.mean() - n_sigma * c.std(),
                                          upper=c.mean() + n_sigma * c.std()))


def rv_sharpe_table(
    markets=None,
    start: str = START,
    end=END,
    models=None,
    rebalance_freq: str = "M",
) -> pd.DataFrame:
    if markets is None:
        markets = OIS_MARKETS
    if models is None:
        models = MODELS

    print("Calibrating lambdas...", flush=True)
    lambdas = calibrate_lambda(markets=markets, start=start, end=end)

    rows = []

    for model in models:
        print(f"  Building {model}...", flush=True)
        signals = build_rv_signals(model=model, markets=markets,
                                    start=start, end=end, lambdas=lambdas)
        returns = build_rv_returns(model=model, markets=markets,
                                    start=start, end=end, lambdas=lambdas)

        for struct in STRUCTURES:
            sig_df = signals[struct]
            ret_df = _winsorize(returns[struct])

            for vf_class, vf_name in [(ViewFormationXS, "XS"), (ViewFormationTS, "TS")]:
                result = run_research_backtest(
                    signal_configs=[({struct: sig_df}, vf_class, None)],
                    returns=ret_df,
                    rebalance_freq=rebalance_freq,
                    transaction_cost_bps=0.0,   # signal quality only, no tc drag
                )
                sh = _sharpe(result["daily_return"])
                rows.append({
                    "structure": struct,
                    "model":     model,
                    "vf":        vf_name,
                    "sharpe":    sh,
                })
                print(f"    {struct} / {vf_name}: {sh:.2f}", flush=True)

    df = pd.DataFrame(rows)
    table = df.pivot_table(
        index="structure",
        columns=["model", "vf"],
        values="sharpe",
    )
    table.columns.names = ["model", "vf"]
    table.index.name    = "structure"
    return table


if __name__ == "__main__":
    print(f"Running RV Sharpe table (start={START})...\n", flush=True)
    table = rv_sharpe_table()
    pd.set_option("display.float_format", "{:.2f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 140)
    print()
    print(table.to_string())
