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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Build per-tenor clipped z-scores and raw factor-neutral residuals for one market.

    Returns
    -------
    z_df    : date × tenor DataFrame of clipped z-scores (signal)
    s_df    : date × tenor DataFrame of raw factor-neutral residuals
    zc_obs  : date × tenor DataFrame of zero curve yields at observed tenors (%)
    avail   : np.ndarray of available tenor values in years (>= 1Y)
    zc      : full bootstrapped zero curve (all tenors, for funding rate)
    ns_load : (N, 3) NS loading matrix at the fixed lambda [level, slope, curvature]
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
        empty = pd.DataFrame()
        return empty, empty, empty, avail, zc, np.empty((0, 3))
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

    return z_df, s_df, zc_obs.loc[common], avail, zc, ns_load


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
        z_df, _, _, _, _, _ = _market_z_scores(market, lambdas, start, end, zero_curves)
        if z_df.empty:
            continue
        mkt_series[market] = z_df.mean(axis=1)

    df = pd.concat(mkt_series.values(), axis=1, keys=mkt_series.keys())
    df.index.name = "date"
    return df


def _factor_neutral_positions(
    z_arr: np.ndarray,
    dv01_arr: np.ndarray,
    ns_load: np.ndarray,
) -> np.ndarray:
    """
    Re-project z-scores onto the complement of the DV01-weighted NS factor space.

    Vol-normalising s_perp per tenor breaks factor-neutrality because each tenor
    is divided by its own EWMA vol.  This re-projection restores it.

    w = z − D(DᵀD)⁻¹Dᵀz   where D[t,n,k] = dv01[t,n] × ns_load[n,k]

    Returns w (T, N) — the actual hedged position weights.
    """
    D   = dv01_arr[:, :, None] * ns_load[None, :, :]         # (T, N, 3)
    DtD = np.einsum('tni,tnj->tij', D, D)                     # (T, 3, 3)
    Dtz = np.einsum('tni,tn->ti',   D, z_arr)                 # (T, 3)
    coef = np.linalg.solve(DtD, Dtz[:, :, None]).squeeze(-1)  # (T, 3)
    return z_arr - np.einsum('tni,ti->tn', D, coef)           # (T, N)


def mean_reversion_pnl(
    markets: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    lambdas: Optional[dict] = None,
    zero_curves: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Per-market daily P&L of the DV01-weighted factor-neutral mean-reversion position.

    Position w(t-1, τ) is z_df re-projected onto the complement of the DV01-weighted
    NS factor space at t-1 (vol-normalising s_perp breaks factor-neutrality; this
    restores it so the hedge is frozen at t-1's loadings).

    For each tenor and day:
      pos(t-1, τ)       = w(t-1, τ) × DV01(t-1, τ)
      pnl_mtm(t, τ)     = pos(t-1, τ) × (−Δy(t, τ))               [actual yield move]
      pnl_carry(t, τ)   = pos(t-1, τ) × (y(t-1,τ) − y_fund(t-1)) / 252

    Per-market P&L (bp) = [Σ_τ (pnl_mtm + pnl_carry)] × 100

    DV01(τ,t) = τ × exp(−y(t,τ)/100 × τ)  — sensitivity per 1% yield move.
    y_fund    = shortest bootstrapped zero-curve tenor (overnight proxy), computed
                from the zero curve directly (no carry parquet).
    Carry adds the yield spread over funding; note cheap tenors may earn less carry
    (opposite sign to the signal), which will reduce Sharpe relative to the gross.

    Units: bp per unit of aggregate z-score exposure.
    NS lambda calibrated once on data ≤ RV_TRAIN_END (held fixed OOS).
    """
    if markets is None:
        markets = OIS_MARKETS
    if lambdas is None:
        lambdas = calibrate_lambda(markets=markets, end=RV_TRAIN_END)

    mkt_series: dict[str, pd.Series] = {}
    for market in markets:
        z_df, _, zc_obs, avail, zc_full, ns_load = _market_z_scores(
            market, lambdas, start, end, zero_curves
        )
        if z_df.empty:
            continue

        z_arr    = z_df.values.astype(float)
        dv01_arr = _dv01_mat(zc_obs.values.astype(float), avail)   # (T, N)

        # Re-project z onto the complement of DV01-weighted NS factor space
        has_nan = ~np.all(np.isfinite(z_arr), axis=1)
        z_safe  = np.where(has_nan[:, None], 0.0, z_arr)
        w_arr   = _factor_neutral_positions(z_safe, dv01_arr, ns_load)
        w_arr[has_nan] = np.nan

        w_df    = pd.DataFrame(w_arr, index=z_df.index, columns=avail)
        dv01_df = pd.DataFrame(dv01_arr, index=zc_obs.index, columns=avail)

        # Actual yield changes from the zero curve (not residual diffs)
        delta_y = zc_obs.diff()

        # Income carry from the zero curve: (y − y_fund) / 252
        # y_fund = shortest bootstrapped tenor (overnight proxy)
        fund_tenor = min(zc_full.columns)
        funding    = zc_full[fund_tenor].reindex(zc_obs.index)
        carry_spread = zc_obs.sub(funding, axis=0) / 252.0  # % per day

        # DV01-scaled lagged position
        pos_lagged = w_df.shift(1) * dv01_df.shift(1)

        # P&L in bp (×100 converts DV01-code × % → bp)
        pnl_mtm   = (pos_lagged * (-delta_y)).sum(axis=1)
        pnl_carry = (pos_lagged * carry_spread.shift(1)).sum(axis=1)
        mkt_series[market] = (pnl_mtm + pnl_carry) * 100.0

    df = pd.concat(mkt_series.values(), axis=1, keys=mkt_series.keys())
    df.index.name = "date"
    return df


def mean_reversion_dv01_pos(
    markets: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    lambdas: Optional[dict] = None,
    zero_curves: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Per-market per-tenor DV01-scaled positions for transaction cost calculation.

    Returns w × DV01 per tenor per market (date × (market, tenor) MultiIndex).
    w is the factor-neutral re-projection of z_df (restoring the hedge frozen at
    t-1's DV01-weighted NS loading structure).

    TC formula: |Δ(w × DV01)| × h_bp  → cost in bp  (consistent with pnl in bp).
    """
    if markets is None:
        markets = OIS_MARKETS
    if lambdas is None:
        lambdas = calibrate_lambda(markets=markets, end=RV_TRAIN_END)

    mkt_frames: dict[str, pd.DataFrame] = {}
    for market in markets:
        z_df, _, zc_obs, avail, _, ns_load = _market_z_scores(
            market, lambdas, start, end, zero_curves
        )
        if z_df.empty:
            continue

        z_arr    = z_df.values.astype(float)
        dv01_arr = _dv01_mat(zc_obs.values.astype(float), avail)

        has_nan = ~np.all(np.isfinite(z_arr), axis=1)
        z_safe  = np.where(has_nan[:, None], 0.0, z_arr)
        w_arr   = _factor_neutral_positions(z_safe, dv01_arr, ns_load)
        w_arr[has_nan] = np.nan

        w_df    = pd.DataFrame(w_arr, index=z_df.index, columns=avail)
        dv01_df = pd.DataFrame(dv01_arr, index=zc_obs.index, columns=avail)
        mkt_frames[market] = w_df * dv01_df

    df = pd.concat(mkt_frames.values(), axis=1, keys=mkt_frames.keys())
    df.columns.names = ["market", "tenor"]
    df.index.name = "date"
    return df
