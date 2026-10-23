"""
Performance-metric infrastructure.

Replicates notebook cell 18: sharpe, drawdown, performance_stats, book_metrics,
block bootstrap, HAC mean test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import skew, kurtosis, norm

from commodities.power_ns.config import TRADING_DAYS, DEFAULT_NW_LAGS, NW_LAG_GRID

# P&L convention used throughout the notebook.
PNL_MODE = "additive"

# ME vs M alias: pandas >= 2.2 uses "ME".
MONTH_END = "ME" if tuple(int(x) for x in pd.__version__.split(".")[:2]) >= (2, 2) else "M"


# ── Bootstrap helpers ─────────────────────────────────────────────────────────


def circular_blocks(n: int, block_length: int, rng) -> np.ndarray:
    """Return circular moving-block bootstrap indices of length n."""
    n, block_length = int(n), int(block_length)
    if n <= 0 or block_length <= 0:
        raise ValueError("n and block_length must be positive")
    n_blocks = int(np.ceil(n / block_length))
    starts   = rng.integers(0, n, n_blocks)
    return np.concatenate([
        np.arange(start, start + block_length) % n for start in starts
    ])[:n]


def block_bootstrap_sharpe_se(
    r,
    block_length: int = 21,
    n_boot: int = 300,
    seed: int = 123,
) -> float:
    """Circular-block bootstrap SE for annualized Sharpe."""
    x = pd.Series(r).dropna().to_numpy(float)
    if len(x) < max(60, 2 * block_length):
        return np.nan
    rng  = np.random.default_rng(seed)
    vals = np.empty(int(n_boot), dtype=float)
    for b in range(int(n_boot)):
        ix   = circular_blocks(len(x), block_length, rng)
        xb   = x[ix]
        sd   = np.std(xb, ddof=1)
        vals[b] = np.mean(xb) / sd * np.sqrt(TRADING_DAYS) if sd > 0 else np.nan
    return float(np.nanstd(vals, ddof=1))


# ── HAC mean test ─────────────────────────────────────────────────────────────


def mean_regression(r, nw_lags: int = DEFAULT_NW_LAGS):
    """Intercept-only OLS with Newey-West HAC standard errors."""
    r = pd.Series(r).dropna()
    if len(r) < 2:
        return None
    X = np.ones((len(r), 1))
    return sm.OLS(r.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": int(nw_lags)})


def hac_mean_sensitivity(
    r,
    lags=NW_LAG_GRID,
    trading_days: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Mean-return inference across several HAC bandwidths."""
    x    = pd.Series(r).dropna()
    rows = []
    for L in lags:
        reg = mean_regression(x, nw_lags=L)
        if reg is None:
            continue
        rows.append({
            "NW lags":   int(L),
            "N":         len(x),
            "Ann. mean": float(reg.params[0]) * trading_days,
            "t-stat":    float(reg.tvalues[0]),
            "p-value":   float(reg.pvalues[0]),
        })
    return pd.DataFrame(rows).set_index("NW lags") if rows else pd.DataFrame()


def lo_adjusted_sharpe(r, max_lag: int = 21, annualization: int = TRADING_DAYS) -> float:
    """Serial-correlation-adjusted Sharpe using a Bartlett long-run variance factor."""
    x = pd.Series(r).dropna()
    if len(x) <= max_lag + 5 or x.std(ddof=1) <= 0:
        return np.nan
    rho     = np.array([x.autocorr(k) for k in range(1, max_lag + 1)], float)
    weights = 1.0 - np.arange(1, max_lag + 1) / (max_lag + 1.0)
    lrv     = 1.0 + 2.0 * np.nansum(weights * rho)
    if not np.isfinite(lrv) or lrv <= 0:
        return np.nan
    return float(x.mean() / x.std(ddof=1) * np.sqrt(annualization / lrv))


# ── Core statistics ───────────────────────────────────────────────────────────


def sharpe(
    r,
    window=None,
    min_periods=None,
    annualization: int = TRADING_DAYS,
    ddof: int = 1,
    min_obs: int = 2,
):
    """Canonical annualized Sharpe for Series or DataFrame."""
    x = r.copy() if isinstance(r, (pd.Series, pd.DataFrame)) else pd.Series(r)

    if window is None:
        count = x.count()
        mu    = x.mean()
        sd    = x.std(ddof=ddof)
        out   = mu / sd * np.sqrt(annualization)
        if np.isscalar(out):
            if count < min_obs or not np.isfinite(sd) or sd <= 0:
                return np.nan
            return float(out)
        return out.where((count >= min_obs) & sd.gt(0))

    if window == "expanding":
        mp  = min_obs if min_periods is None else int(min_periods)
        obj = x.expanding(min_periods=mp)
    else:
        window = int(window)
        mp     = window if min_periods is None else int(min_periods)
        obj    = x.rolling(window, min_periods=mp)

    mu  = obj.mean()
    sd  = obj.std(ddof=ddof)
    out = mu / sd * np.sqrt(annualization)
    return out.where(sd > 0)


def drawdown(r, mode: str = PNL_MODE) -> pd.Series | pd.DataFrame:
    """Drawdown path. mode='additive' (cumsum) or 'compound' (cumprod)."""
    x = r.copy() if isinstance(r, (pd.Series, pd.DataFrame)) else pd.Series(r)
    if mode == "additive":
        curve = x.cumsum()
        return curve - curve.cummax()
    if mode == "compound":
        wealth = (1.0 + x).cumprod()
        return wealth / wealth.cummax() - 1.0
    raise ValueError(f"Unknown P&L mode: {mode}")


def performance_stats(
    r,
    nw_lags: int = DEFAULT_NW_LAGS,
    trading_days: int = TRADING_DAYS,
) -> pd.Series:
    """Full-sample statistics for one additive P&L / return Series."""
    r = pd.Series(r).dropna()
    if len(r) < 2:
        return pd.Series(dtype=float)

    reg            = mean_regression(r, nw_lags=nw_lags)
    alpha_daily    = reg.params[0]
    alpha_tstat    = reg.tvalues[0]
    alpha_pvalue   = reg.pvalues[0]
    ann_mean       = alpha_daily * trading_days
    ann_vol        = r.std(ddof=1) * np.sqrt(trading_days)
    downside       = r[r < 0]
    downside_vol   = downside.std(ddof=1) * np.sqrt(trading_days) if len(downside) > 1 else np.nan
    sharpe_ratio   = sharpe(r, annualization=trading_days)
    gains          = r[r > 0].sum()
    losses         = -r[r < 0].sum()
    omega          = gains / losses if losses > 0 else np.nan
    skewness       = skew(r, bias=False)
    excess_kurtosis = kurtosis(r, fisher=True, bias=False)
    q01            = r.quantile(0.01)
    hist_var_99    = -q01
    hist_es_99     = -r[r <= q01].mean() if (r <= q01).any() else np.nan
    monthly        = r.resample(MONTH_END).sum()
    positive_months = (monthly > 0).mean() if len(monthly) else np.nan
    max_drawdown   = drawdown(r).min()

    return pd.Series({
        "N":                    len(r),
        "Mean":                 ann_mean,
        "Alpha daily":          alpha_daily,
        "Alpha t-stat (NW)":    alpha_tstat,
        "Alpha p-value (NW)":   alpha_pvalue,
        "NW lags":              int(nw_lags),
        "Volatility":           ann_vol,
        "Downside volatility":  downside_vol,
        "Sharpe ratio":         sharpe_ratio,
        "Skewness":             skewness,
        "Excess kurtosis":      excess_kurtosis,
        "99% historical VaR":   hist_var_99,
        "99% historical ES":    hist_es_99,
        "% positive months":    positive_months,
        "Maximum drawdown":     max_drawdown,
    })


def book_metrics(book: dict, period=slice(None), cost_bps=None, aum=None) -> pd.Series:
    """Canonical portfolio/book summary from one gross/turnover position path."""
    gross    = book["gross"].loc[period]
    turnover = book.get("turnover", pd.Series(0.0, index=gross.index, dtype=float)).reindex(gross.index)

    if cost_bps is None and "net" in book:
        net = book["net"].reindex(gross.index)
    else:
        c   = 0.0 if cost_bps is None else float(cost_bps)
        net = gross - c / 1e4 * turnover

    net      = net.dropna()
    gross    = gross.reindex(net.index)
    turnover = turnover.reindex(net.index)

    sn      = performance_stats(net)
    sg      = performance_stats(gross)
    ann_turn = turnover.mean() * TRADING_DAYS

    out = {
        "N":                sn.get("N", np.nan),
        "Gross Sharpe":     sg.get("Sharpe ratio", np.nan),
        "Net Sharpe":       sn.get("Sharpe ratio", np.nan),
        "Net Mean":         sn.get("Mean", np.nan),
        "Net Vol":          sn.get("Volatility", np.nan),
        "MaxDD (additive)": sn.get("Maximum drawdown", np.nan),
        "Annual turnover":  ann_turn,
        "Break-even bps":   sg.get("Mean", np.nan) / ann_turn * 1e4
                            if np.isfinite(ann_turn) and ann_turn > 0 else np.nan,
    }
    if aum is not None:
        out["AUM"]           = float(aum)
        out["Annual $ P&L"]  = sn.get("Mean", np.nan) * float(aum)
    if "alloc" in book:
        out["Mean utilisation"] = book["alloc"].reindex(net.index).sum(axis=1).mean()
    if "capped" in book:
        out["Markets capped (avg)"] = book["capped"].reindex(net.index).sum(axis=1).mean()
    if "selected_depth" in book:
        out["Average selected depth"] = book["selected_depth"].reindex(net.index).stack().mean()
    return pd.Series(out)
