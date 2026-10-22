"""
Factor-neutral NS residual mean-reversion signal (ex_raw RV).

For each market and tenor: z-score of the DV01-weighted factor-neutral NS
residual, using an EWMA vol (hl=63d) estimated on lagged data.

Two public entry points:

  mean_reversion_rv(...)
      Per-market scalar = mean of per-tenor clipped z-scores.
      Used for cross-market signal combination.

  mean_reversion_pnl(...)
      Per-market daily PnL = sum_tenors[ pos(t-1) × (−Δresid(t)) ].
      Correct backtest return: each tenor traded against its own residual
      change, not an averaged proxy.  This is what the Sharpe figures refer to.

NS lambda is calibrated once on data ≤ RV_TRAIN_END and held fixed OOS.
Positive signal / positive PnL → curve is cheap → long duration (receive).
"""

import numpy as np
import pandas as pd
from typing import Optional

from rates.production.nelson_siegel import (
    calibrate_lambda, extract_factors, _reconstruct_fitted, _ns_loadings,
)
from rates.production.ois_data import OIS_MARKETS, fetch_ois_curves
from rates.production.yc_builder import build_zero_curve

RV_TRAIN_END   = "2018-12-31"
RV_EWMA_HL     = 63
RV_MIN_PERIODS = 30
RV_VOL_FLOOR   = 1e-4
RV_Z_CAP       = 3.0


def _dv01_mat(z_arr: np.ndarray, tenors: np.ndarray) -> np.ndarray:
    """DV01 matrix (T, N) from yield array (T, N) in %."""
    return tenors[None, :] * np.exp(-z_arr / 100.0 * tenors[None, :])


def _factor_neutral_residuals(
    resid: np.ndarray,
    z_arr: np.ndarray,
    tenors: np.ndarray,
    ns_load: np.ndarray,
) -> np.ndarray:
    """
    Project each date's NS residual onto the complement of the DV01-weighted
    NS factor space.  Returns s_perp (T, N) — factor-neutral residuals.
    """
    dv01 = _dv01_mat(z_arr, tenors)
    D    = dv01[:, :, None] * ns_load[None, :, :]
    DtD  = np.einsum('tni,tnj->tij', D, D)
    Dts  = np.einsum('tni,tn->ti',   D, resid)
    coef = np.linalg.solve(DtD, Dts[:, :, None]).squeeze(-1)
    return resid - np.einsum('tni,ti->tn', D, coef)


def _market_z_scores(
    market: str,
    lambdas: dict,
    start: Optional[str],
    end: Optional[str],
    zero_curves: Optional[dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build per-tenor clipped z-scores and raw factor-neutral residuals for one market.

    Returns
    -------
    z_df : date × tenor DataFrame of clipped z-scores (positions)
    s_df : date × tenor DataFrame of raw factor-neutral residuals (for PnL)
    """
    if zero_curves is not None and market in zero_curves:
        zc = zero_curves[market]
    else:
        raw = fetch_ois_curves(market, start=start, end=end)
        zc  = build_zero_curve(raw)

    raw_cols = fetch_ois_curves(market).columns.values.astype(float)
    avail = np.array([t for t in np.sort(raw_cols[raw_cols >= 1.0])
                      if t in zc.columns])
    if len(avail) < 3:
        return pd.DataFrame(), pd.DataFrame()
    zc_obs = zc[list(avail)]

    ns_lam  = lambdas["ns"][market]
    ns_load = _ns_loadings(avail, ns_lam)

    fac    = extract_factors(zc, model="ns_calibrated", lam=ns_lam)
    fitted = _reconstruct_fitted(zc, fac, "ns_calibrated", lam=ns_lam)
    common = zc_obs.index.intersection(fitted.index)
    resid  = zc_obs.loc[common].subtract(fitted.loc[common, list(avail)])

    z_arr   = zc_obs.loc[resid.index, list(avail)].values.astype(float)
    r_arr   = resid.values.astype(float)
    has_nan = ~(np.all(np.isfinite(r_arr), axis=1) &
                np.all(np.isfinite(z_arr), axis=1))
    r_arr[has_nan] = 0.0
    z_arr[has_nan] = 3.0

    s_perp = _factor_neutral_residuals(r_arr, z_arr, avail, ns_load)
    s_perp[has_nan] = np.nan
    s_df = pd.DataFrame(s_perp, index=resid.index, columns=avail)

    vol  = s_df.shift(1).ewm(halflife=RV_EWMA_HL, min_periods=RV_MIN_PERIODS).std()
    z_df = (s_df / vol.clip(lower=RV_VOL_FLOOR)).clip(-RV_Z_CAP, RV_Z_CAP)

    return z_df, s_df


def mean_reversion_rv(
    markets: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    lambdas: Optional[dict] = None,
    zero_curves: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Per-market scalar signal = mean of per-tenor clipped z-scores.

    Used for cross-market signal combination.  For backtesting use
    mean_reversion_pnl(), which trades each tenor against its own residual.

    NS lambda calibrated once on data ≤ RV_TRAIN_END (held fixed OOS).
    Positive → curve cheap on average → long duration.
    """
    if markets is None:
        markets = OIS_MARKETS
    if lambdas is None:
        lambdas = calibrate_lambda(markets=markets, end=RV_TRAIN_END)

    mkt_series: dict[str, pd.Series] = {}
    for market in markets:
        z_df, _ = _market_z_scores(market, lambdas, start, end, zero_curves)
        if z_df.empty:
            continue
        mkt_series[market] = z_df.mean(axis=1)

    df = pd.concat(mkt_series.values(), axis=1, keys=mkt_series.keys())
    df.index.name = "date"
    return df


def mean_reversion_pnl(
    markets: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    lambdas: Optional[dict] = None,
    zero_curves: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Per-market daily PnL using per-tenor positions and residual returns.

    For each tenor: pos(t-1) × (−Δresid(t)).
    Per-market PnL = sum across tenors.

    This is the correct backtest: each tenor is traded against its own
    factor-neutral residual change, not a yield proxy or cross-tenor average.
    Positive PnL when cheap tenors mean-revert (residuals fall back to model).

    NS lambda calibrated once on data ≤ RV_TRAIN_END (held fixed OOS).
    """
    if markets is None:
        markets = OIS_MARKETS
    if lambdas is None:
        lambdas = calibrate_lambda(markets=markets, end=RV_TRAIN_END)

    mkt_series: dict[str, pd.Series] = {}
    for market in markets:
        z_df, s_df = _market_z_scores(market, lambdas, start, end, zero_curves)
        if z_df.empty:
            continue
        pnl = (z_df.shift(1) * (-s_df.diff())).sum(axis=1)
        mkt_series[market] = pnl

    df = pd.concat(mkt_series.values(), axis=1, keys=mkt_series.keys())
    df.index.name = "date"
    return df
