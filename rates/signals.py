"""
Rates signals derived from Nelson-Siegel / Nelson-Siegel-Svensson factors.

All signals are cross-market (date × market DataFrames), each market treated
independently (ViewFormationTS). Positive signal → long duration (receive rates).

Three signals:
  1. Momentum  — NS_fixed  — TSMOM on β₁ (level), 12-1m skip=21
  2. Carry     — NS_fixed  — −β₂ (slope factor)
  3. MeanRev   — ex_raw RV — factor-neutral NS residuals, per-tenor EWMA z-score

The mean-reversion signal uses factor-neutral NS residuals (ex_raw): each tenor's
residual is projected onto the complement of the DV01-weighted NS factor space,
then EWMA z-scored. The per-market scalar is the mean of per-tenor z-scores,
representing aggregate factor-neutral cheapness of the curve.
Lambda is calibrated once on the training window (≤ RV_TRAIN_END).
"""

import numpy as np
import pandas as pd
from typing import Optional

from rates.nelson_siegel import (
    extract_factors_all_markets, calibrate_lambda,
    extract_factors, _reconstruct_fitted, _ns_loadings,
)
from rates.ois_data import OIS_MARKETS, fetch_ois_curves
from rates.yc_builder import build_zero_curve

# ── RV mean-reversion constants ────────────────────────────────────────────────

RV_TRAIN_END  = "2018-12-31"   # lambda calibration window
RV_EWMA_HL    = 63             # z-score EWMA halflife (days)
RV_MIN_PERIODS = 30
RV_VOL_FLOOR  = 1e-4           # % — prevents division by near-zero vol
RV_Z_CAP      = 3.0            # per-tenor z-score clip


# ── RV helpers (factor-neutral projection) ─────────────────────────────────────

def _dv01_mat(z_arr: np.ndarray, tenors: np.ndarray) -> np.ndarray:
    """DV01 matrix (T, N) from yield array (T, N) in %."""
    return tenors[None, :] * np.exp(-z_arr / 100.0 * tenors[None, :])


def _factor_neutral_residuals(
    resid: np.ndarray,    # (T, N) NS residuals in %
    z_arr: np.ndarray,    # (T, N) zero yields in %
    tenors: np.ndarray,   # (N,)
    ns_load: np.ndarray,  # (N, 3) NS loading matrix (fixed)
) -> np.ndarray:
    """
    Project each date's NS residual onto the complement of the DV01-weighted
    NS factor space.  Returns s_perp (T, N) — factor-neutral residuals.
    """
    dv01 = _dv01_mat(z_arr, tenors)                          # (T, N)
    D    = dv01[:, :, None] * ns_load[None, :, :]            # (T, N, 3)
    DtD  = np.einsum('tni,tnj->tij', D, D)                   # (T, 3, 3)
    Dts  = np.einsum('tni,tn->ti',   D, resid)               # (T, 3)
    coef = np.linalg.solve(DtD, Dts[:, :, None]).squeeze(-1) # (T, 3)
    return resid - np.einsum('tni,ti->tn', D, coef)          # (T, N)


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


def mean_reversion_rv(
    markets: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    lambdas: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Factor-neutral NS residual mean-reversion signal (ex_raw).

    For each market and tenor: z-score of the DV01-weighted factor-neutral NS
    residual, using an EWMA vol (hl=63d) estimated on lagged data.
    Per-market scalar: mean of per-tenor clipped z-scores (aggregate cheapness).

    NS lambda is calibrated once on data ≤ RV_TRAIN_END and held fixed OOS.
    Positive → curve is cheap on average → long duration.
    """
    if markets is None:
        markets = OIS_MARKETS
    if lambdas is None:
        lambdas = calibrate_lambda(markets=markets, end=RV_TRAIN_END)

    mkt_series: dict[str, pd.Series] = {}
    for market in markets:
        raw = fetch_ois_curves(market, start=start, end=end)
        zc  = build_zero_curve(raw)

        # Observed tenors ≥ 1Y present in bootstrapped curve
        raw_cols = fetch_ois_curves(market).columns.values.astype(float)
        avail = np.array([t for t in np.sort(raw_cols[raw_cols >= 1.0])
                          if t in zc.columns])
        if len(avail) < 3:
            continue
        zc_obs = zc[list(avail)]

        ns_lam  = lambdas["ns"][market]
        ns_load = _ns_loadings(avail, ns_lam)

        # In-sample NS residuals on observed tenors
        fac    = extract_factors(zc, model="ns_calibrated", lam=ns_lam)
        fitted = _reconstruct_fitted(zc, fac, "ns_calibrated", lam=ns_lam)
        common = zc_obs.index.intersection(fitted.index)
        resid  = zc_obs.loc[common].subtract(fitted.loc[common, list(avail)])

        # Factor neutralisation
        z_arr   = zc_obs.loc[resid.index, list(avail)].values.astype(float)
        r_arr   = resid.values.astype(float)
        has_nan = ~(np.all(np.isfinite(r_arr), axis=1) &
                    np.all(np.isfinite(z_arr), axis=1))
        r_arr[has_nan] = 0.0
        z_arr[has_nan] = 3.0

        s_perp = _factor_neutral_residuals(r_arr, z_arr, avail, ns_load)
        s_perp[has_nan] = np.nan
        s_df = pd.DataFrame(s_perp, index=resid.index, columns=avail)

        # EWMA z-score (lagged vol so no lookahead)
        vol  = s_df.shift(1).ewm(halflife=RV_EWMA_HL,
                                  min_periods=RV_MIN_PERIODS).std()
        z_df = s_df / vol.clip(lower=RV_VOL_FLOOR)

        # Aggregate to scalar: mean of clipped per-tenor z-scores
        mkt_series[market] = z_df.clip(-RV_Z_CAP, RV_Z_CAP).mean(axis=1)

    df = pd.concat(mkt_series.values(), axis=1, keys=mkt_series.keys())
    df.index.name = "date"
    return df


def build_signals(
    markets: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    mom_lookback: int = 252,
    mom_skip: int = 21,
) -> dict[str, pd.DataFrame]:
    """
    Run the full rates signal pipeline.

    Momentum and carry use ns_calibrated (fast, OLS).
    Mean reversion uses the ex_raw factor-neutral NS residual signal.

    Returns
    -------
    dict with keys 'momentum', 'carry', 'mean_reversion'.
    Each value is a date × market DataFrame of raw (unnormalised) signals.
    """
    ns_factors = extract_factors_all_markets(
        markets=markets, start=start, end=end,
        model="ns_calibrated",
    )

    return {
        "momentum":       momentum(ns_factors, lookback=mom_lookback, skip=mom_skip),
        "carry":          carry(ns_factors),
        "mean_reversion": mean_reversion_rv(markets=markets, start=start, end=end),
    }
