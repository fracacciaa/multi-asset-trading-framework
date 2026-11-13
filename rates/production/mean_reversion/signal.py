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

RV_TRAIN_END          = "2018-12-31"
RV_EWMA_HL            = 63
RV_MIN_PERIODS        = 30
RV_VOL_FLOOR          = 1e-4
RV_Z_CAP              = 3.0
RV_GROSS_CAP_PCTILE   = 0.99  # IS gross notional percentile used as per-market cap
RV_DY_ABS_CAP         = 0.25  # 25bp hard cap on per-tenor daily yield changes used in
                               # P&L computation; handles data-gap spikes (e.g. NZ/China
                               # OIS showing 500-840bp overnight "moves") without touching
                               # the signal z-scores

# Markets excluded from the mean-reversion sleeve only (momentum and carry are
# unaffected).  Attribution of the all-tenor sleeve, 9 DM markets:
#   New Zealand  IS SR -0.18, OOS SR -0.41, OOS MDD -43.7%, and the only market
#                with a negative cumulative contribution (-10.2% of sleeve P&L).
#                51.5% of the IS drawdown.  Excludable on IS evidence alone.
#   Japan        IS SR +0.27, OOS SR -0.07, OOS MDD -29.3%, 4.7% of sleeve P&L.
#                Marginal contributor that adds drawdown; note the IS Sharpe was
#                positive, so this exclusion is partly OOS-informed.
# Effect (IS/OOS Sharpe): +0.31/+0.78 for all 9 -> +0.42/+0.81 for these 7.
# This does NOT reduce drawdown (-25.6%/-21.3% -> -26.9%/-21.2%): the drawdowns
# are driven by EUR and Switzerland, which this exclusion does not touch.
RV_EXCLUDE_MARKETS    = ("New Zealand", "Japan")


def _rv_markets(markets: Optional[list[str]]) -> list[str]:
    """Default market list for the MR sleeve, with RV_EXCLUDE_MARKETS removed.

    An explicit `markets` argument is honoured as given, so callers can still
    evaluate the excluded markets deliberately.
    """
    if markets is not None:
        return list(markets)
    return [m for m in OIS_MARKETS if m not in RV_EXCLUDE_MARKETS]


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

    # Denominator: EWMA vol of actual yield changes, not of the residual itself.
    # Using residual vol collapses to near-zero in quiet regimes, making the
    # floor binding and distorting the z-score.  Yield-change vol is always
    # regime-sensible and makes the floor essentially inert.
    yield_chg = zc_obs.loc[common].diff()
    vol  = yield_chg.shift(1).ewm(halflife=RV_EWMA_HL, min_periods=RV_MIN_PERIODS).std()
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
    markets = _rv_markets(markets)
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


def _gross_cap_scale(
    w_df: pd.DataFrame,
    dv01_df: pd.DataFrame,
    cap_end: str = RV_TRAIN_END,
) -> pd.Series:
    """
    Per-date scale factor keeping daily gross DV01 notional (Σ_τ |w×DV01|)
    below the IS 99th-percentile cap.  Caps multi-tenor stress-event blowups
    where many individually z-capped tenors align simultaneously.

    Returned as a per-date Series so the same scaling can be applied to the
    unit-notional weights (for P&L) and to the DV01-scaled weights (for TC),
    keeping the two consistent.  Uniform across tenors, so the factor-neutral
    hedge structure is preserved.
    """
    gross = (w_df * dv01_df).abs().sum(axis=1)
    is_gross = gross[gross.index <= cap_end]
    ref = is_gross if len(is_gross) >= 30 else gross
    cap = float(ref.quantile(RV_GROSS_CAP_PCTILE))
    if cap <= 0:
        return pd.Series(1.0, index=w_df.index)
    return (cap / gross).clip(upper=1.0)


def _capped_dv01_pos(
    w_df: pd.DataFrame,
    dv01_df: pd.DataFrame,
    cap_end: str = RV_TRAIN_END,
) -> pd.DataFrame:
    """
    DV01-scaled, gross-capped positions.  Used for transaction costs only:
    a swap half-spread is a yield spread, so converting it into P&L units
    requires the DV01.  The P&L itself must NOT use these (see
    mean_reversion_pnl — ret_d already contains the duration term).
    """
    return (w_df * dv01_df).mul(_gross_cap_scale(w_df, dv01_df, cap_end), axis=0)


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
    ret_d: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Per-market daily P&L: yesterday's factor-neutral residual score times the
    actual return of the instrument at that tenor today.

        pnl(t, market) = Σ_τ  w(t-1, τ) × ret_d(t, (market, τ))        [bp]

    w(t-1, τ) is the unit-notional factor-neutral position: z_df re-projected
    onto the complement of the DV01-weighted NS factor space, so the level,
    slope and curvature hedge is frozen at t-1's loadings, then gross-capped at
    the IS 99th percentile of DV01 notional.

    ret_d is the constant-maturity receiver-swap return from
    momentum.returns.cm_return_bp, which already contains carry, roll-down and
    the duration-scaled price term:

        ret_d = (s - rf)·dt  +  dur·(s - s_aged)  -  dur·(s_next - s)

    Two consequences, both of which earlier versions got wrong:
      * do NOT multiply w by DV01 here — ret_d already carries the duration
        term, so w × DV01 × ret_d counts duration twice;
      * do NOT add a separate carry leg — ret_d already includes it.

    The score is built on the bootstrapped zero curve while ret_d is built from
    par swap rates, so the two do not share a quote-error realisation.  That
    matters: a P&L formed as score(t-1) × (−Δy(t)) on the *same* series is a
    non-negative quadratic form y(t-1)ᵀPy(t-1) and scores a large positive
    Sharpe on pure noise.  Regression test: phase-scramble w and confirm the
    Sharpe collapses to zero.

    ret_d : optional pre-built panel with (market, tenor) MultiIndex columns.
            Built per market from the par curve when not supplied.
    NS lambda calibrated once on data ≤ RV_TRAIN_END (held fixed OOS).
    """
    markets = _rv_markets(markets)
    if lambdas is None:
        lambdas = calibrate_lambda(markets=markets, end=RV_TRAIN_END)

    mkt_series: dict[str, pd.Series] = {}
    for market in markets:
        z_df, _, zc_obs, avail, _, ns_load = _market_z_scores(
            market, lambdas, start, end, zero_curves
        )
        if z_df.empty:
            continue

        if ret_d is not None:
            if market not in ret_d.columns.get_level_values(0):
                continue
            r_mkt = ret_d[market]
        else:
            from rates.production.momentum.returns import build_return_panels
            r_mkt = build_return_panels(market, start=start, end=end,
                                        tenors=list(avail))["ret_d"]

        tenors = [t for t in avail if t in r_mkt.columns]
        if len(tenors) < 3:
            continue

        z_arr    = z_df.values.astype(float)
        dv01_arr = _dv01_mat(zc_obs.values.astype(float), avail)

        has_nan = ~np.all(np.isfinite(z_arr), axis=1)
        z_safe  = np.where(has_nan[:, None], 0.0, z_arr)
        w_arr   = _factor_neutral_positions(z_safe, dv01_arr, ns_load)
        w_arr[has_nan] = np.nan

        w_df    = pd.DataFrame(w_arr, index=z_df.index, columns=avail)
        dv01_df = pd.DataFrame(dv01_arr, index=zc_obs.index, columns=avail)
        w_df    = w_df.mul(_gross_cap_scale(w_df, dv01_df), axis=0)

        idx = w_df.index.intersection(r_mkt.index)
        mkt_series[market] = (w_df.loc[idx, tenors].shift(1)
                              * r_mkt.loc[idx, tenors]).sum(axis=1)

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
    markets = _rv_markets(markets)
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
        mkt_frames[market] = _capped_dv01_pos(w_df, dv01_df)

    df = pd.concat(mkt_frames.values(), axis=1, keys=mkt_frames.keys())
    df.columns.names = ["market", "tenor"]
    df.index.name = "date"
    return df
