"""
Performance analytics.
Computes standard statistics to evaluate backtest results.
"""

import pandas as pd
import numpy as np


def annualized_return(nav: pd.Series, periods_per_year: int = 252) -> float:
    total = nav.iloc[-1] / nav.iloc[0]
    n_years = len(nav) / periods_per_year
    return total ** (1 / n_years) - 1

def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    return returns.std() * np.sqrt(periods_per_year)

def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    excess = returns - risk_free / periods_per_year
    if excess.std() == 0:
        return np.nan
    return excess.mean() / excess.std() * np.sqrt(periods_per_year)

def max_drawdown(nav: pd.Series) -> float:
    rolling_max = nav.cummax()
    drawdown = nav / rolling_max - 1
    return drawdown.min()

def calmar_ratio(nav: pd.Series, returns: pd.Series, periods_per_year: int = 252) -> float:
    mdd = max_drawdown(nav)
    if mdd == 0:
        return np.nan
    return annualized_return(nav, periods_per_year) / abs(mdd)

def sortino_ratio(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    excess = returns - risk_free / periods_per_year
    downside = excess[excess < 0].std()
    if downside == 0:
        return np.nan
    return excess.mean() / downside * np.sqrt(periods_per_year)

def hit_rate(returns: pd.Series) -> float:
    return (returns > 0).mean()

def average_win_loss(returns: pd.Series) -> float:
    wins = returns[returns > 0].mean()
    losses = returns[returns < 0].mean()
    if losses == 0:
        return np.nan
    return abs(wins / losses)

def skewness(returns: pd.Series) -> float:
    return returns.skew()

def kurtosis(returns: pd.Series) -> float:
    return returns.kurt()

def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    return returns.quantile(1 - confidence)

def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    var = value_at_risk(returns, confidence)
    return returns[returns <= var].mean()

def summary(nav: pd.Series, returns=None, risk_free: float = 0.0) -> pd.Series:
    """
    Return a Series of all key performance metrics.
    If returns is None it is inferred from nav.
    """
    if returns is None:
        returns = nav.pct_change().dropna()

    return pd.Series(
        {
            "Annualized Return": annualized_return(nav),
            "Annualized Volatility": annualized_volatility(returns),
            "Sharpe Ratio": sharpe_ratio(returns, risk_free),
            "Sortino Ratio": sortino_ratio(returns, risk_free),
            "Max Drawdown": max_drawdown(nav),
            "Calmar Ratio": calmar_ratio(nav, returns),
            "Hit Rate": hit_rate(returns),
            "Avg Win/Loss": average_win_loss(returns),
            "Skewness": skewness(returns),
            "Kurtosis": kurtosis(returns),
            "VaR (95%)": value_at_risk(returns),
            "CVaR / ES (95%)": expected_shortfall(returns),
        }
    )
