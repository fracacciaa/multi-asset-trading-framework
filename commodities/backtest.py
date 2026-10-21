"""
Commodity futures backtest: portfolio construction, risk scaling, and
signal comparison harness.

Typical usage
-------------
    from commodities.backtest import run_comparison, PARAMS
    results = run_comparison()
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
    fetch_term_structure_power,
    fetch_tickers,
    fetch_tickers_power,
)
from commodities.signals import (
    tsmom,
    ma_crossover,
    channel_breakout,
    xs_carry,
    curve_slope,
    curve_curvature,
    momentum,
    mean_reversion,
)

TD = 252  # trading days per year

PARAMS = dict(
    lookbacks=(21, 63, 252),
    mode="sign",
    clip=2.0,
    vol_com=60,
    asset_vol=0.40,
    book_vol=0.10,
    cov_hl=252,
    rebal="W-FRI",
    lag=1,
    cost_bps=5.0,
    rank=1,
    k_max=10.0,
)


# ── Returns ───────────────────────────────────────────────────────────────────

def roll_safe_returns(
    prices: pd.DataFrame,
    tickers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Daily excess returns with NaN on roll dates.

    Roll dates are rows where the front-month ticker changes from the prior
    day; the price jump on that day reflects the spread between expiring and
    new contract, not a tradeable P&L.
    """
    ret = prices.pct_change()
    roll_mask = tickers.ne(tickers.shift(1)) & tickers.notna() & tickers.shift(1).notna()
    return ret.where(~roll_mask)


# ── Volatility and scaling ────────────────────────────────────────────────────

def ex_ante_vol(
    ret: pd.DataFrame,
    com: int = 60,
    td: int = TD,
) -> pd.DataFrame:
    """EWMA annualised ex-ante vol (HOP eq. 2)."""
    min_p = max(120, 2 * com)
    mu = ret.ewm(com=com, min_periods=min_p).mean()
    return np.sqrt(((ret - mu) ** 2).ewm(com=com, min_periods=min_p).mean() * td)


def vol_scale(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    target_vol: float = 0.40,
    com: int = 60,
    td: int = TD,
) -> pd.DataFrame:
    """
    Scale each asset's raw signal so the expected position contributes
    target_vol annualised volatility.

    weights = signal * target_vol / sigma_hat
    """
    sigma = ex_ante_vol(returns, com=com, td=td)
    sigma = sigma.reindex(signal.index, method="ffill").replace(0, np.nan)
    return signal * target_vol / sigma


def rebalance(
    weights: pd.DataFrame,
    rule: Optional[str] = "W-FRI",
) -> pd.DataFrame:
    """
    Freeze weights between rebalance dates.

    rule=None → daily updates.  rule follows pd.resample offset aliases.
    """
    if rule is None:
        return weights
    rebal_dates = weights.resample(rule).last().dropna(how="all").index
    return (weights.reindex(rebal_dates)
                   .reindex(weights.index)
                   .ffill())


def book_scale(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    target_vol: float = 0.10,
    cov_hl: int = TD,
    td: int = TD,
    k_max: float = 10.0,
) -> pd.DataFrame:
    """
    Scale the whole book to target_vol using an EWMA covariance estimate.

    k_t = clip(target_vol / sqrt(w_t' Σ_t w_t * td), 0, k_max)
    Covariance uses the same EWMA half-life as cov_hl trading days.
    """
    r = returns.fillna(0.0)
    com = cov_hl / np.log(2)          # half-life to centre-of-mass

    W = weights.reindex(r.index).fillna(0.0).to_numpy(float)
    R = r.to_numpy(float)

    # Build rolling EWMA covariance row by row
    n = R.shape[1]
    alpha = 1.0 / (1.0 + com)
    Sigma = np.zeros((n, n))
    mu_r  = np.zeros(n)
    k_arr = np.zeros(len(W))

    for t in range(len(W)):
        mu_r  = (1 - alpha) * mu_r  + alpha * R[t]
        dev   = R[t] - mu_r
        Sigma = (1 - alpha) * Sigma + alpha * np.outer(dev, dev)
        var_p = float(W[t] @ Sigma @ W[t]) * td
        if var_p > 1e-12:
            k_arr[t] = np.clip(target_vol / np.sqrt(var_p), 0.0, k_max)

    return pd.DataFrame(
        W * k_arr[:, None],
        index=weights.index,
        columns=weights.columns,
    )


# ── Backtest runner ───────────────────────────────────────────────────────────

def run_backtest(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    lag: int = 1,
    cost_bps: float = 5.0,
) -> dict:
    """
    Execute a set of weights, returning PnL net of linear round-trip costs.

    weights : date × ticker (signals already scaled and book-scaled)
    returns : date × ticker daily excess returns
    lag     : execution delay in trading days (1 = trade next open)
    """
    w = weights.shift(lag).fillna(0.0)
    ret_aligned = returns.reindex(w.index).fillna(0.0)
    pnl = (w * ret_aligned).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    pnl_net = pnl - turnover * cost_bps * 1e-4
    return dict(pnl=pnl_net, weights=w, turnover=turnover)


# ── Performance metrics ───────────────────────────────────────────────────────

def sharpe(pnl: pd.Series, td: int = TD) -> float:
    """Annualised Sharpe on non-zero, non-NaN days."""
    s = pnl[pnl.ne(0)].dropna()
    return float(s.mean() / s.std() * np.sqrt(td)) if len(s) > 120 else np.nan


def max_drawdown(pnl: pd.Series) -> float:
    """Worst peak-to-trough on cumulative PnL."""
    cum = pnl.cumsum()
    return float((cum - cum.cummax()).min())


def calmar(pnl: pd.Series, td: int = TD) -> float:
    mdd = max_drawdown(pnl)
    if mdd >= 0:
        return np.nan
    ann_ret = float(pnl.mean() * td)
    return ann_ret / abs(mdd)


def evaluate(pnl: pd.Series, td: int = TD) -> dict:
    s = pnl[pnl.ne(0)].dropna()
    if len(s) < 120:
        return {k: np.nan for k in ("sharpe", "ann_ret", "ann_vol", "max_dd", "skew", "calmar", "n_days")}
    return dict(
        sharpe=sharpe(s, td),
        ann_ret=float(s.mean() * td),
        ann_vol=float(s.std() * np.sqrt(td)),
        max_dd=max_drawdown(s),
        skew=float(s.skew()),
        calmar=calmar(s, td),
        n_days=len(s),
    )


def _subperiod_sharpes(pnl: pd.Series, td: int = TD) -> dict:
    """Sharpe by 3-year block (label = start year of block)."""
    blocks = (pnl.index.year // 3) * 3
    return {str(yr): sharpe(grp, td) for yr, grp in pnl.groupby(blocks)}


# ── Signal definitions for comparison ────────────────────────────────────────

def _build_signals(
    prices: pd.DataFrame,
    ts: dict,
    p: dict,
) -> dict[str, pd.DataFrame]:
    """
    Return a dict of raw (unscaled) signal DataFrames keyed by signal label.
    All signals are defined on the intersection of columns present in prices and ts.
    """
    F1 = ts["front"]
    F2 = ts["second"]
    F4 = ts["fourth"] if "fourth" in ts else None

    sigs: dict[str, pd.DataFrame] = {}

    # ── Trend signals ──────────────────────────────────────────────────────────
    sigs["tsmom_21"]    = tsmom(prices, lookbacks=(21,),  mode=p["mode"], clip=p["clip"])
    sigs["tsmom_63"]    = tsmom(prices, lookbacks=(63,),  mode=p["mode"], clip=p["clip"])
    sigs["tsmom_252"]   = tsmom(prices, lookbacks=(252,), mode=p["mode"], clip=p["clip"])
    sigs["tsmom_blend"] = tsmom(prices, lookbacks=p["lookbacks"], mode=p["mode"], clip=p["clip"])
    sigs["tsmom_scaled"]= tsmom(prices, lookbacks=p["lookbacks"], mode="clip",  clip=p["clip"])

    sigs["ma_20_120"]   = ma_crossover(prices, fast=20,  slow=120)
    sigs["ma_50_200"]   = ma_crossover(prices, fast=50,  slow=200)

    sigs["breakout_55"]  = channel_breakout(prices, window=55)
    sigs["breakout_120"] = channel_breakout(prices, window=120)

    # ── Carry signals (cross-sectional: relative rank within universe) ─────────
    sigs["xs_carry_f1f2"] = xs_carry(F1, F2).reindex(columns=prices.columns)
    if F4 is not None:
        f4_aligned = F4.reindex(columns=prices.columns)
        sigs["xs_carry_f1f4"] = xs_carry(F1.reindex(columns=prices.columns),
                                          f4_aligned).reindex(columns=prices.columns)
        sigs["curvature"]     = curve_curvature(F1.reindex(columns=prices.columns),
                                                F2.reindex(columns=prices.columns),
                                                f4_aligned).reindex(columns=prices.columns)

    # ── Existing signals ───────────────────────────────────────────────────────
    sigs["mom_magnitude"] = momentum(prices)
    sigs["meanrev_21"]    = mean_reversion(prices)

    return sigs


# ── Main comparison runner ────────────────────────────────────────────────────

def run_comparison(
    start: Optional[str] = None,
    end: Optional[str] = None,
    rank: int = 1,
    params: dict = PARAMS,
    include_power: bool = False,
) -> pd.DataFrame:
    """
    Run all 14 signals through the full vol-scaling and book-scaling pipeline
    and return a summary DataFrame.

    Columns: sharpe, ann_ret, ann_vol, max_dd, skew, calmar, n_days,
             + one column per 3-year sub-period block.

    Parameters
    ----------
    start / end  : date strings for slicing returns.
    rank         : front-month rank (1 = standard; 2 for robustness run).
    params       : hyperparameter dict (defaults to PARAMS).
    include_power: if True, add the 4 liquid power instruments to the universe.
    """
    p = params

    # ── Load data ──────────────────────────────────────────────────────────────
    prices_gen = fetch_prices(start=start, end=end)
    ts_gen     = fetch_term_structure(start=start, end=end)
    tk_gen     = fetch_tickers(start=start, end=end)

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

    # ── Roll-safe returns ──────────────────────────────────────────────────────
    front_tickers = tk["front"].reindex(columns=prices.columns)
    returns = roll_safe_returns(prices, front_tickers)

    # ── Build raw signals ──────────────────────────────────────────────────────
    raw_signals = _build_signals(prices, ts, p)

    # ── Run each signal through the pipeline ──────────────────────────────────
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

    # Sort by Sharpe descending
    result = result.sort_values("sharpe", ascending=False)
    return result
