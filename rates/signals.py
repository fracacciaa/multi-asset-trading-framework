"""
Rates signals derived from Nelson-Siegel / Nelson-Siegel-Svensson factors.

All signals are cross-market (date × market DataFrames), each market treated
independently (ViewFormationTS). Positive signal → long duration (receive rates).

Three signals, each using the best-fit model for its purpose:
  1. Momentum  — NS_fixed  — TSMOM on β₁ (level), 12-1m skip=21
  2. Carry     — NS_fixed  — −β₂ (slope factor)
  3. MeanRev   — NSS_free  — −(β₁ − rolling_mean), NSS for more accurate level

The mean-reversion signal uses NSS_free because the additional curvature factor
prevents the level estimate from absorbing hump-shaped mid-curve features.
"""

import numpy as np
import pandas as pd
from typing import Optional

from rates.nelson_siegel import build_factors_all_markets
from rates.ois_data import fetch_ois_returns, OIS_MARKETS


def _get_factor(factors: pd.DataFrame, factor_name: str) -> pd.DataFrame:
    """Extract a single factor across all markets as a date × market DataFrame."""
    return factors.xs(factor_name, axis=1, level="factor")


def momentum(
    factors: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
) -> pd.DataFrame:
    """
    Time-series momentum on β₁ (level), 12-1m skip=21.

    Signal = β₁.shift(1+skip) − β₁.shift(lookback+1+skip)   [absolute bp change]

    Positive → level rose → rates expected to keep rising → short bonds.
    Negative → level fell → rates expected to keep falling → long bonds.
    """
    level = _get_factor(factors, "level")
    return (level.shift(1 + skip) - level.shift(lookback + 1 + skip)).rename_axis("date")


def carry(factors: pd.DataFrame) -> pd.DataFrame:
    """
    Carry from roll-down: −β₂ (slope factor from NS).

    Normal (upward-sloping) curve: β₂ < 0 → −β₂ > 0 → positive carry.
    Inverted curve: β₂ > 0 → −β₂ < 0 → negative carry.
    """
    slope = _get_factor(factors, "slope")
    return (-slope).rename_axis("date")


def mean_reversion(
    factors: pd.DataFrame,
    lookback: int = 252,
) -> pd.DataFrame:
    """
    Mean reversion on β₁ (level factor from NSS).

    Signal = −(β₁ − rolling_mean(β₁, lookback))

    Rates above their 1-year average → expect reversion lower → long bonds.
    NSS level (β₁) is a cleaner long-run yield estimate than NS level
    because the two curvature terms absorb hump-shaped curve features
    that would otherwise contaminate β₁.
    """
    level = _get_factor(factors, "level")
    rolling_mean = level.rolling(lookback, min_periods=lookback // 2).mean()
    return (-(level - rolling_mean)).rename_axis("date")


def build_signals(
    markets: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    mom_lookback: int = 252,
    mom_skip: int = 21,
    mr_lookback: int = 252,
    use_ois: bool = True,
    mr_model: str = "ns_free",
) -> dict[str, pd.DataFrame]:
    """
    Run the full rates signal pipeline.

    Momentum and carry use NS_fixed (fast, OLS).
    Mean reversion uses `mr_model` (default: NS_free) for a more accurate
    level estimate. NS_free adapts λ to unusual curve shapes (flat, inverted)
    without the β₁ sign instability seen in NSS_free at near-zero rates.

    Parameters
    ----------
    mr_model : model for mean reversion level — 'ns_fixed', 'ns_free',
               'nss_fixed', or 'nss_free'

    Returns
    -------
    dict with keys 'momentum', 'carry', 'mean_reversion'.
    Each value is a date × market DataFrame of raw (unnormalised) signals.
    """
    # NS_fixed for momentum + carry
    ns_factors = build_factors_all_markets(
        markets=markets, start=start, end=end,
        model="ns_fixed", use_ois=use_ois,
    )

    # Separate model for mean reversion level
    if mr_model == "ns_fixed":
        mr_factors = ns_factors
    else:
        mr_factors = build_factors_all_markets(
            markets=markets, start=start, end=end,
            model=mr_model, use_ois=use_ois,
        )

    return {
        "momentum":       momentum(ns_factors, lookback=mom_lookback, skip=mom_skip),
        "carry":          carry(ns_factors),
        "mean_reversion": mean_reversion(mr_factors, lookback=mr_lookback),
    }
