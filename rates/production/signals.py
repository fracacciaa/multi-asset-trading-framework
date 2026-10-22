"""
Rates signals derived from Nelson-Siegel / Nelson-Siegel-Svensson factors.

All signals are cross-market (date × market DataFrames), each market treated
independently (ViewFormationTS). Positive signal → long duration (receive rates).

Four signals:
  1. Momentum  — MOP Signal A — sign(12m log excess return) on {2,5,10}Y OIS tenors,
                 equal-weight average across tenors, monthly frequency
  2. Carry     — NS_fixed  — −β₂ (slope factor)
  3. MeanRev   — ex_raw RV — factor-neutral NS residuals, per-tenor EWMA z-score
  4. Carry CC  — cross-country DV01-neutral carry (DM only)

The mean-reversion signal uses factor-neutral NS residuals (ex_raw): each tenor's
residual is projected onto the complement of the DV01-weighted NS factor space,
then EWMA z-scored. The per-market scalar is the mean of per-tenor z-scores,
representing aggregate factor-neutral cheapness of the curve.
Lambda is calibrated once on the training window (≤ RV_TRAIN_END).

The momentum signal (MOP) uses bootstrapped zero-coupon log excess returns
(Sihvonen eq 1) rather than NS level-factor changes.  OOS Sharpe at 2Y: 0.55–0.90
across US/EUR/UK/Japan/CA/AU/NZ.  Spec: rates/docs/ois_momentum_signals_spec.md.
"""

import numpy as np
import pandas as pd
from typing import Optional

from rates.production.nelson_siegel import extract_factors_all_markets
from rates.production.ois_data import OIS_MARKETS, fetch_ois_curves
from rates.production.mean_reversion.signal import mean_reversion_rv

# ── Cross-country carry constants ──────────────────────────────────────────────

CC_LAM             = 0.3
CC_BASIS           = "ns"
CC_TAU_TARGET      = 10.0
CC_VOL_LOOKBACK    = 252
CC_MIN_OBS         = 63
CC_MIN_MARKETS     = 4
CC_LAG             = 1


# ── Signal functions ───────────────────────────────────────────────────────────

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


def momentum_mop(
    markets: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    tenors: Optional[list[float]] = None,
    k: int = 12,
) -> pd.DataFrame:
    """
    MOP time-series momentum on log excess swap returns (Signal A).

    signal[t] = mean_tenors( sign( Σ_{i=1}^{k} rx_m[t-i] ) )

    rx_m is the monthly Sihvonen log excess return (exact OIS bootstrap, eq 1).
    Positive signal → positive momentum → long duration (receive fixed).

    Markets default to DM-8 (same as cross-country carry).
    Tenors default to {2, 5, 10}Y — primary set mirroring MOP bond futures.
    Signal is monthly (month-end); forward-fills between rebalances.

    OOS Sharpe (2019–today): US 0.49, EUR 0.41, AU 0.52, Japan 0.59, NZ 0.37.
    """
    from rates.production.momentum.returns import build_rx_monthly
    from rates.production.carry.cross_country import DM_MARKETS

    if markets is None:
        markets = DM_MARKETS
    if tenors is None:
        tenors = [2.0, 5.0, 10.0]

    mkt_series: dict[str, pd.Series] = {}
    for market in markets:
        try:
            panels = build_rx_monthly(market, start=start, end=end)
            rx_m   = panels["rx_m"]
            cols   = [t for t in tenors if t in rx_m.columns]
            if not cols:
                continue
            sig = np.sign(rx_m[cols].shift(1).rolling(k).sum())
            mkt_series[market] = sig.mean(axis=1)
        except Exception:
            continue

    if not mkt_series:
        return pd.DataFrame()

    df = pd.DataFrame(mkt_series)
    df.index.name = "date"
    return df


def carry_cross_country(
    markets: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Cross-country carry position book (DV01-neutral, CC DV01-only variant).

    For each DM market: fit NS daily, compute carry at 10Y via carry_reprice,
    divide by duration to get yield-space carry.  Then cross-section across
    countries: z-score, inverse-vol size, zero-net, DV01-project.

    Returns
    -------
    date × market DataFrame of position weights.
    Positions are ready-to-trade: inverse-vol sized, zero-net, DV01-neutral,
    normalised to gross = 1.0, lagged by 1 day.  Positive = receive fixed
    (long duration); negative = pay fixed (short duration).
    """
    from rates.production.ns_core import fit_ns, fit_ns_batch
    from rates.production.carry.cross_country import (
        country_carry_at_target, build_cross_country_positions, DM_MARKETS,
    )
    from rates.production.carry.experiment import BASELINE

    if markets is None:
        markets = DM_MARKETS

    curves_dict: dict = {}
    funding_dict: dict = {}
    fits_dict: dict = {}

    for mkt in markets:
        try:
            c = fetch_ois_curves(mkt, start=start, end=end)
            col = 0.25 if 0.25 in c.columns else min(c.columns, key=lambda x: abs(x - 0.25))
            curves_dict[mkt]  = c
            funding_dict[mkt] = c[col]

            tau_all = c.columns.to_numpy(float)
            Y_all   = c.values * 100.0               # % → bp, shape (T, N)
            full_rows   = np.all(np.isfinite(Y_all), axis=1)
            enough_rows = np.sum(np.isfinite(Y_all), axis=1) >= 4

            fits: dict = {}

            # Batch the dates where all tenors are valid (vast majority)
            if full_rows.any():
                for date, fit in zip(
                    c.index[full_rows],
                    fit_ns_batch(tau_all, Y_all[full_rows], lam=CC_LAM, basis=CC_BASIS),
                ):
                    fits[date] = fit

            # Per-row fallback for partial dates (some tenors NaN)
            for i in np.where(enough_rows & ~full_rows)[0]:
                row = c.iloc[i].dropna()
                if len(row) < 4:
                    continue
                try:
                    fits[c.index[i]] = fit_ns(
                        row.index.to_numpy(float),
                        row.to_numpy(float) * 100.0,
                        lam=CC_LAM, basis=CC_BASIS,
                    )
                except Exception:
                    pass

            if len(fits) >= CC_MIN_OBS:
                fits_dict[mkt] = fits
        except Exception:
            pass

    cc_series: dict = {}
    for mkt in markets:
        if mkt not in fits_dict:
            continue
        rf_bp = funding_dict[mkt] * 100.0
        carry_ts = country_carry_at_target(
            fits_dict[mkt], curves_dict[mkt], rf_bp, CC_TAU_TARGET, BASELINE,
        )
        if len(carry_ts) > 0:
            cc_series[mkt] = carry_ts

    if len(cc_series) < 2:
        return pd.DataFrame()

    country_carry_df = pd.DataFrame(cc_series).sort_index()

    pos_df = build_cross_country_positions(
        country_carry_df,
        curves_dict,
        tau_vol_ref=CC_TAU_TARGET,
        universe="DM",
        vol_lookback=CC_VOL_LOOKBACK,
        min_obs=CC_MIN_OBS,
        fits_dict=fits_dict,
        global_beta_neutralize=False,
    )

    valid = pos_df.notna().sum(axis=1) >= CC_MIN_MARKETS
    pos_df = pos_df.where(valid)
    gross  = pos_df.abs().sum(axis=1).replace(0, np.nan)
    pos_df = pos_df.div(gross, axis=0).shift(CC_LAG)
    pos_df.index.name = "date"
    return pos_df


def build_signals(
    markets: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    mom_tenors: Optional[list[float]] = None,
    mom_k: int = 12,
) -> dict[str, pd.DataFrame]:
    """
    Run the full rates signal pipeline.

    Momentum uses MOP Signal A (log excess returns on {2,5,10}Y OIS tenors).
    Carry uses ns_calibrated (fast, OLS).
    Mean reversion uses the ex_raw factor-neutral NS residual signal.
    Cross-country carry uses carry_reprice on NS-fitted 10Y curves with DV01
    neutralisation; returns a ready-to-trade position book.

    Returns
    -------
    dict with keys:
      'momentum'            — date × market monthly signal (month-end indexed)
      'carry'               — date × market raw signal (unnormalised)
      'mean_reversion'      — date × market raw signal (unnormalised)
      'carry_cross_country' — date × market position weights (ready-to-trade)
    """
    ns_factors = extract_factors_all_markets(
        markets=markets, start=start, end=end,
        model="ns_calibrated",
    )

    return {
        "momentum":            momentum_mop(markets=markets, start=start, end=end,
                                            tenors=mom_tenors, k=mom_k),
        "carry":               carry(ns_factors),
        "mean_reversion":      mean_reversion_rv(markets=markets, start=start, end=end),
        "carry_cross_country": carry_cross_country(markets=markets, start=start, end=end),
    }
