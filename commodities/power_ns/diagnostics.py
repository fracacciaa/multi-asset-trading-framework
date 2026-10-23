"""
Research diagnostics: ADF mean-reversion tests, convergence speed,
rolling betas, and performance attribution.

Replicates notebook cells 43 (ADF), 60-66 (convergence speed, rolling betas),
46/47 (calendar/episode robustness, yearly performance).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from commodities.power_ns.config import (
    TRADING_DAYS,
    TRADE_MAX_RANK,
    ADF_LAGS,
    DF_CRIT,
    TRAIN_END,
    TEST_START,
)
from commodities.power_ns.ns_fit import ns_loadings
from commodities.power_ns.metrics import sharpe, performance_stats


# ── Market-level NS null-space coordinate ─────────────────────────────────────


def signed_null_coordinate(
    root: str,
    ns_residual_by_rank: dict,
    tickers: pd.DataFrame,
    MATURITY: pd.DataFrame,
    LAMBDA_TRADE: float,
    max_rank: int = TRADE_MAX_RANK,
) -> pd.Series:
    """
    Construct the unique signed NS residual coordinate for M1-M4:

        z_t = n_t' e_t

    where n_t spans the one-dimensional null space of the NS loading matrix.
    Sign orientation: force the M3 loading to be non-positive (deterministic).
    """
    if max_rank != 4:
        raise ValueError(
            "Market-level scalar ADF is defined here for M1-M4 only. "
            "M1-M5 has a two-dimensional null space."
        )

    ranks = range(1, max_rank + 1)
    e = pd.concat(
        [ns_residual_by_rank[k][root].rename(k) for k in ranks],
        axis=1,
    )
    cols = [f"{root}{k}" for k in ranks]
    idx  = e.index.intersection(tickers.index).intersection(MATURITY.index)

    E  = e.reindex(idx).to_numpy(dtype=float)
    M  = MATURITY.reindex(idx)[cols].to_numpy(dtype=float)
    TK = tickers.reindex(idx)[cols]

    good = (
        np.isfinite(E).all(axis=1)
        & np.isfinite(M).all(axis=1)
        & (M > 0).all(axis=1)
        & TK.notna().all(axis=1).to_numpy()
    )

    out = pd.Series(np.nan, index=idx, dtype=float, name=root)
    if not good.any():
        return out

    Mg = M[good]
    Eg = E[good]
    sl, cv = ns_loadings(Mg, LAMBDA_TRADE)
    X  = np.stack([np.ones_like(Mg), sl, cv], axis=2)

    U, _, _ = np.linalg.svd(X, full_matrices=True)
    n = U[:, :, 3].copy()

    # Deterministic orientation: force M3 loading non-positive.
    flip    = n[:, 2] > 0
    n[flip] *= -1.0

    z = np.sum(n * Eg, axis=1)
    out.iloc[np.flatnonzero(good)] = z
    return out


# ── Roll-safe ADF test ────────────────────────────────────────────────────────


def adf_market_test(
    z: pd.Series,
    ticker_panel: pd.DataFrame,
    n_lags: int = ADF_LAGS,
    min_obs: int = 250,
) -> dict | None:
    """
    ADF test on one market-level residual coordinate.

    A regression observation at t is accepted only when the complete
    ADF window lies inside one uninterrupted specific-contract segment.
    """
    z  = pd.Series(z)
    idx = z.index.intersection(ticker_panel.index)
    z  = z.reindex(idx)
    tk = ticker_panel.reindex(idx)
    Z  = z.to_numpy(dtype=float)

    changed = (
        tk.ne(tk.shift(1)).any(axis=1)
        | tk.isna().any(axis=1)
        | tk.shift(1).isna().any(axis=1)
    ).to_numpy()
    segment = np.cumsum(changed)
    dz      = np.r_[np.nan, np.diff(Z)]

    rows = []
    for t in range(n_lags + 1, len(Z)):
        if segment[t] != segment[t - (n_lags + 1)]:
            continue
        if not (np.isfinite(Z[t]) and np.isfinite(Z[t - 1])):
            continue
        if not np.isfinite(dz[t - n_lags: t + 1]).all():
            continue
        rows.append([dz[t], Z[t - 1], *[dz[t - L] for L in range(1, n_lags + 1)]])

    if len(rows) < min_obs:
        return None

    arr  = np.asarray(rows, dtype=float)
    y    = arr[:, 0]
    X    = sm.add_constant(arr[:, 1:])
    fit  = sm.OLS(y, X).fit()

    gamma    = float(fit.params[1])
    adf_stat = float(fit.tvalues[1])
    halflife = (
        np.log(0.5) / np.log(1.0 + gamma)
        if -1 < gamma < 0 else np.nan
    )

    return {"N": len(arr), "ADF stat": adf_stat, "gamma": gamma, "halflife": halflife}


def run_market_adfs(
    price_universe: list[str],
    ns_residual_by_rank: dict,
    tickers: pd.DataFrame,
    MATURITY: pd.DataFrame,
    LAMBDA_TRADE: float,
    n_lags: int = ADF_LAGS,
    max_rank: int = TRADE_MAX_RANK,
) -> pd.DataFrame:
    """Run roll-safe ADF for every market and return a summary DataFrame."""
    rows = []
    for root in price_universe:
        z    = signed_null_coordinate(root, ns_residual_by_rank, tickers, MATURITY, LAMBDA_TRADE, max_rank)
        cols = [f"{root}{k}" for k in range(1, max_rank + 1)]
        tk_p = tickers.reindex(z.index)[cols]
        res  = adf_market_test(z, tk_p, n_lags=n_lags)
        if res is not None:
            rows.append({"Market": root, **res})

    return (
        pd.DataFrame(rows).set_index("Market").sort_values("ADF stat")
        if rows else pd.DataFrame()
    )


# ── Convergence speed analysis ────────────────────────────────────────────────


def rolling_convergence_speed(
    z: pd.Series,
    halflife_window: int = 252,
    min_obs: int = 63,
) -> pd.Series:
    """
    Estimate rolling mean-reversion half-life (days) from a trailing OLS on z.

    Fits AR(1):  dz_t = a + gamma * z_{t-1} + e_t  in a rolling window.
    Returns half-life = log(0.5) / log(1 + gamma), NaN when gamma >= 0.
    """
    z     = pd.Series(z).dropna()
    dz    = z.diff()
    z_lag = z.shift(1)

    n     = len(z)
    hl    = pd.Series(np.nan, index=z.index, dtype=float)

    for t in range(halflife_window, n):
        sl_y  = dz.iloc[t - halflife_window: t].dropna()
        sl_x  = z_lag.reindex(sl_y.index).dropna()
        idx_c = sl_y.index.intersection(sl_x.index)
        if len(idx_c) < min_obs:
            continue
        y_c, x_c = sl_y.loc[idx_c].values, sl_x.loc[idx_c].values
        X     = sm.add_constant(x_c)
        try:
            fit   = sm.OLS(y_c, X).fit()
            gamma = float(fit.params[1])
            if -1 < gamma < 0:
                hl.iloc[t] = np.log(0.5) / np.log(1.0 + gamma)
        except Exception:
            pass

    return hl


def run_all_convergence_speeds(
    price_universe: list[str],
    ns_residual_by_rank: dict,
    tickers: pd.DataFrame,
    MATURITY: pd.DataFrame,
    LAMBDA_TRADE: float,
    halflife_window: int = 252,
    min_obs: int = 63,
) -> pd.DataFrame:
    """Compute rolling half-life for every market and return as a DataFrame."""
    out = {}
    for root in price_universe:
        z     = signed_null_coordinate(root, ns_residual_by_rank, tickers, MATURITY, LAMBDA_TRADE)
        out[root] = rolling_convergence_speed(z, halflife_window, min_obs)
    return pd.DataFrame(out)


# ── Yearly performance breakdown ──────────────────────────────────────────────


def yearly_performance(
    market_gross: pd.DataFrame,
    price_universe: list[str],
) -> pd.DataFrame:
    """
    Annual gross Sharpe ratio by calendar year.

    Returns DataFrame(index=year, columns=["Sharpe", "Mean", "N"]).
    """
    portfolio = market_gross.mean(axis=1).dropna()
    by_year   = portfolio.groupby(portfolio.index.year)
    rows      = []
    for yr, grp in by_year:
        if len(grp) < 21:
            continue
        s = sharpe(grp)
        rows.append({"Year": yr, "Sharpe": s, "Mean": grp.mean() * TRADING_DAYS, "N": len(grp)})
    return pd.DataFrame(rows).set_index("Year") if rows else pd.DataFrame()


# ── Calendar-day-of-month robustness ─────────────────────────────────────────


def day_of_month_breakdown(gross: pd.Series) -> pd.DataFrame:
    """P&L statistics grouped by day-of-month."""
    df = pd.DataFrame({"pnl": gross, "day": gross.index.day})
    stats = df.groupby("day")["pnl"].agg(["count", "mean", "std"])
    stats["ann mean"] = stats["mean"] * TRADING_DAYS
    stats["Sharpe"]   = stats["mean"] / stats["std"] * np.sqrt(TRADING_DAYS)
    return stats[["count", "ann mean", "Sharpe"]]


# ── Implementation delay ladder ───────────────────────────────────────────────


def implementation_delay_table(
    ns_residual_by_rank: dict,
    tickers: pd.DataFrame,
    MATURITY: pd.DataFrame,
    LAMBDA_TRADE: float,
    price_universe: list[str],
    residual_weights_rev: dict,
    prices: pd.DataFrame,
    BUSINESS_DAYS_TO_MONTH_END: pd.DataFrame,
    lag_grid: tuple = (1, 2, 3, 5),
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """
    Sharpe vs implementation lag.

    Runs the full residual portfolio at each lag value and reports mean Sharpe
    across markets in TRAIN and TEST windows.
    """
    from commodities.power_ns.execution import run_residual_portfolio

    rows = []
    for lag in lag_grid:
        book  = run_residual_portfolio(
            residual_weights_rev, prices, tickers, BUSINESS_DAYS_TO_MONTH_END,
            price_universe, lag=lag, cost_bps=cost_bps,
        )
        mg    = book["market_gross"]
        delay = lag - 1
        rows.append({
            "lag":    lag,
            "delay (days)": delay,
            "TRAIN Sharpe": sharpe(book["gross"].loc[:TRAIN_END].dropna()),
            "TEST Sharpe":  sharpe(book["gross"].loc[TEST_START:].dropna()),
            "FULL Sharpe":  sharpe(book["gross"].dropna()),
        })

    return pd.DataFrame(rows).set_index("lag")
