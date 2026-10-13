"""
Relative-value signals from Nelson-Siegel model residuals.

Residual at tenor τ:  resid(t, τ) = y_actual(t, τ) − y_fitted(t, τ)
  > 0  → actual yield above model → tenor is "cheap"  (receive / buy)
  < 0  → actual yield below model → tenor is "rich"   (pay / sell)

Three duration-neutral structures (applied per market, per model):

  slope_2s10s   long 1 DV01 at 2Y, short 1 DV01 at 10Y (10Y leg adjusted)
                signal  = resid(2) − w(t) × resid(10)
                w(t)    = DV01_2(t) / DV01_10(t)

  fly_2s5s10s   long wings (2Y, 10Y), short body (5Y adjusted)
                signal  = resid(2) + resid(10) − w(t) × resid(5)
                w(t)    = [DV01_2(t) + DV01_10(t)] / DV01_5(t)

  fly_5s10s20s  long wings (5Y, 20Y), short body (10Y adjusted)
                signal  = resid(5) + resid(20) − w(t) × resid(10)
                w(t)    = [DV01_5(t) + DV01_20(t)] / DV01_10(t)

DV01 uses the full zero-coupon discount factor:
  DV01(τ, t) ∝ τ × exp(−r_τ(t) × τ / 100)
so hedge ratios are time-varying and accurate at any yield level (critical for EM).

Positive signal → long the described structure.
"""

import numpy as np
import pandas as pd
from typing import Optional

from rates.nelson_siegel import (
    extract_factors, calibrate_lambda, _reconstruct_fitted,
)
from rates.yc_builder import build_zero_curve
from rates.ois_data import fetch_ois_curves, OIS_MARKETS

DM_MARKETS = ["US", "EUR", "UK", "Japan", "Canada", "Australia", "New Zealand", "Sweden", "Switzerland"]
EM_MARKETS = ["India", "China"]

# 0.5Y-spaced tenor grid: 0.5Y, 1.0Y, …, 30.0Y  (60 points)
HALF_YEAR_TENORS: np.ndarray = np.arange(0.5, 30.5, 0.5)


# ── DV01 helper ────────────────────────────────────────────────────────────────

def _dv01(zc: pd.DataFrame, tenor: float) -> pd.Series:
    """
    DV01 of a ZCB at `tenor` years.  Units: years × discount factor (price-bpv).
    Yields in zc are in percent, so divide by 100 before exp.
    """
    r = zc[tenor] / 100.0
    return tenor * np.exp(-r * tenor)


# ── Signal constructors ────────────────────────────────────────────────────────

def slope_2s10s(resid: pd.DataFrame, zc: pd.DataFrame) -> pd.Series:
    """
    Duration-neutral 2s10s slope bet.

    Long 2Y, short 10Y (sized so DV01 of 10Y position = DV01 of 2Y position).
    Positive → 2Y cheap relative to 10Y vs model.
    """
    common = resid.index.intersection(zc.index)
    w = _dv01(zc, 2.0).loc[common] / _dv01(zc, 10.0).loc[common]
    return (resid.loc[common, 2.0] - w * resid.loc[common, 10.0]).rename("slope_2s10s")


def fly_2s5s10s(resid: pd.DataFrame, zc: pd.DataFrame) -> pd.Series:
    """
    Duration-neutral 2s5s10s butterfly.

    Long wings (2Y, 10Y at 1 unit each), short body (5Y adjusted).
    Positive → 5Y rich relative to wings vs model.
    """
    common = resid.index.intersection(zc.index)
    w = (_dv01(zc, 2.0).loc[common] + _dv01(zc, 10.0).loc[common]) / _dv01(zc, 5.0).loc[common]
    signal = resid.loc[common, 2.0] + resid.loc[common, 10.0] - w * resid.loc[common, 5.0]
    return signal.rename("fly_2s5s10s")


def fly_5s10s20s(resid: pd.DataFrame, zc: pd.DataFrame) -> pd.Series:
    """
    Duration-neutral 5s10s20s butterfly.

    Long wings (5Y, 20Y at 1 unit each), short body (10Y adjusted).
    Positive → 10Y rich relative to wings vs model.
    """
    common = resid.index.intersection(zc.index)
    w = (_dv01(zc, 5.0).loc[common] + _dv01(zc, 20.0).loc[common]) / _dv01(zc, 10.0).loc[common]
    signal = resid.loc[common, 5.0] + resid.loc[common, 20.0] - w * resid.loc[common, 10.0]
    return signal.rename("fly_5s10s20s")


# ── AR(1) half-life ────────────────────────────────────────────────────────────

def _ar1_halflife(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 100:
        return np.nan
    rho = np.polyfit(s.iloc[:-1].values, s.iloc[1:].values, 1)[0]
    if rho <= 0 or rho >= 1:
        return np.nan
    return np.log(0.5) / np.log(rho)


def compute_halflife_table(
    markets: Optional[list[str]] = None,
    model: str = "ns_calibrated",
    start: Optional[str] = None,
    end: Optional[str] = None,
    lambdas: Optional[dict] = None,
    tenors: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    AR(1) half-life (business days) of NS residuals per market × tenor.

    Returns a market × tenor (float years) DataFrame.
    Used to set per-tenor EWM smoothing in build_rv_signals / build_rv_returns.
    """
    if markets is None:
        markets = OIS_MARKETS
    if lambdas is None:
        lambdas = calibrate_lambda(markets=markets, start=start, end=end)

    rows = {}
    for mkt in markets:
        resid, _ = _market_residuals(mkt, model, lambdas, start=start, end=end,
                                     tenors=tenors)
        rows[mkt] = {col: _ar1_halflife(resid[col]) for col in resid.columns}

    df = pd.DataFrame(rows).T
    df.index.name = "market"
    return df


# ── Per-market residual computation ───────────────────────────────────────────

def _market_residuals(
    market: str,
    model: str,
    lambdas: dict,
    start: Optional[str] = None,
    end: Optional[str] = None,
    halflife_map: Optional[dict[float, float]] = None,
    tenors: Optional[np.ndarray] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (residuals, zero_curve) for one market under one model.
    Both are date × tenor DataFrames on the same index.

    tenors       : if provided, subset the zero curve to these tenors before
                   fitting NS/NSS. Only tenors present in the bootstrapped
                   curve are used (missing ones are silently dropped).
    halflife_map : {tenor_years: halflife_days} — if provided, each tenor's
                   residuals are smoothed with EWM(halflife=...) before return.
                   Covers DM/EM-specific mean-reversion speeds.
    """
    raw = fetch_ois_curves(market, start=start, end=end)
    zc  = build_zero_curve(raw)

    if tenors is not None:
        avail = [t for t in tenors if t in zc.columns]
        zc = zc[avail]

    lam_m = lam1_m = lam2_m = None
    if model == "ns_calibrated":
        lam_m = lambdas["ns"][market]
        fac   = extract_factors(zc, model=model, lam=lam_m)
    elif model == "nss_calibrated":
        lam1_m, lam2_m = lambdas["nss"][market]
        fac   = extract_factors(zc, model=model, lam1=lam1_m, lam2=lam2_m)
    else:
        fac = extract_factors(zc, model=model)

    fitted = _reconstruct_fitted(zc, fac, model,
                                  lam=lam_m, lam1=lam1_m, lam2=lam2_m)
    resid  = zc.loc[fitted.index].sub(fitted)

    if halflife_map:
        for tenor, hl in halflife_map.items():
            if tenor in resid.columns and np.isfinite(hl) and hl > 0:
                resid[tenor] = resid[tenor].ewm(halflife=hl, min_periods=int(hl)).mean()

    return resid, zc.loc[fitted.index]


# ── Portfolio-level builders ───────────────────────────────────────────────────

def build_rv_signals(
    model: str = "ns_calibrated",
    markets: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    lambdas: Optional[dict] = None,
    smooth: bool = False,
    halflife_table: Optional[pd.DataFrame] = None,
) -> dict[str, pd.DataFrame]:
    """
    Build all three RV signals for all markets under one NS/NSS model.

    Parameters
    ----------
    model          : 'ns_calibrated' | 'ns_free' | 'nss_calibrated' | 'nss_free'
    lambdas        : output of calibrate_lambda(); auto-computed if None
    smooth         : if True, smooth each tenor's residuals with EWM(halflife)
                     calibrated to the AR(1) mean-reversion speed of that tenor.
                     DM and EM markets each get their own per-tenor halflife.
    halflife_table : pre-computed output of compute_halflife_table(); auto-
                     computed if smooth=True and this is None.

    Returns
    -------
    dict with keys 'slope_2s10s', 'fly_2s5s10s', 'fly_5s10s20s'.
    Each value is a date × market DataFrame of raw (unnormalised) signals.
    """
    if markets is None:
        markets = OIS_MARKETS
    if lambdas is None:
        lambdas = calibrate_lambda(markets=markets, start=start, end=end)
    if smooth and halflife_table is None:
        halflife_table = compute_halflife_table(
            markets=markets, model=model, start=start, end=end, lambdas=lambdas
        )

    slope_d, fly1_d, fly2_d = {}, {}, {}

    for market in markets:
        hl_map = halflife_table.loc[market].to_dict() if smooth else None
        resid, zc = _market_residuals(market, model, lambdas,
                                       start=start, end=end, halflife_map=hl_map)
        slope_d[market] = slope_2s10s(resid, zc).rename(market)
        fly1_d[market]  = fly_2s5s10s(resid, zc).rename(market)
        fly2_d[market]  = fly_5s10s20s(resid, zc).rename(market)

    def _concat(d: dict) -> pd.DataFrame:
        df = pd.concat(d.values(), axis=1)
        df.index.name = "date"
        return df

    return {
        "slope_2s10s":  _concat(slope_d),
        "fly_2s5s10s":  _concat(fly1_d),
        "fly_5s10s20s": _concat(fly2_d),
    }


def build_rv_returns(
    model: str = "ns_calibrated",
    markets: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    lambdas: Optional[dict] = None,
    smooth: bool = False,
    halflife_table: Optional[pd.DataFrame] = None,
) -> dict[str, pd.DataFrame]:
    """
    Daily P&L of being long each structure for one day.

    Uses the *previous day's* hedge ratio so the hedge is fixed over the holding
    period — i.e. return(t) = Δresid_short(t) − w(t-1) × Δresid_long(t).
    This is the correct mark-to-market return of a position entered at t-1.

    smooth / halflife_table : same semantics as build_rv_signals — smooth the
    residuals before differencing so returns reflect the smoothed signal's P&L.

    Returns
    -------
    Same key structure as build_rv_signals; values are first-differenced.
    Units: basis points (same as the residuals / signals).
    """
    if markets is None:
        markets = OIS_MARKETS
    if lambdas is None:
        lambdas = calibrate_lambda(markets=markets, start=start, end=end)
    if smooth and halflife_table is None:
        halflife_table = compute_halflife_table(
            markets=markets, model=model, start=start, end=end, lambdas=lambdas
        )

    slope_d, fly1_d, fly2_d = {}, {}, {}

    for market in markets:
        hl_map = halflife_table.loc[market].to_dict() if smooth else None
        resid, zc = _market_residuals(market, model, lambdas,
                                       start=start, end=end, halflife_map=hl_map)
        common = resid.index.intersection(zc.index)
        r, z   = resid.loc[common], zc.loc[common]

        # Lagged hedge ratios (position fixed at start of each day)
        w_slope = (_dv01(z, 2.0)  / _dv01(z, 10.0)).shift(1)
        w_fly1  = ((_dv01(z, 2.0) + _dv01(z, 10.0)) / _dv01(z, 5.0)).shift(1)
        w_fly2  = ((_dv01(z, 5.0) + _dv01(z, 20.0)) / _dv01(z, 10.0)).shift(1)

        slope_d[market] = (r[2.0].diff()  - w_slope * r[10.0].diff()).rename(market)
        fly1_d[market]  = (r[2.0].diff()  + r[10.0].diff() - w_fly1 * r[5.0].diff()).rename(market)
        fly2_d[market]  = (r[5.0].diff()  + r[20.0].diff() - w_fly2 * r[10.0].diff()).rename(market)

    def _concat(d: dict) -> pd.DataFrame:
        df = pd.concat(d.values(), axis=1)
        df.index.name = "date"
        return df

    return {
        "slope_2s10s":  _concat(slope_d),
        "fly_2s5s10s":  _concat(fly1_d),
        "fly_5s10s20s": _concat(fly2_d),
    }


def build_rv_signals_all_models(
    markets: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Build all three RV signals for all four NS/NSS models.

    Calibration runs once and is shared across both calibrated models.

    Returns
    -------
    {model: {'slope_2s10s': df, 'fly_2s5s10s': df, 'fly_5s10s20s': df}}
    """
    if markets is None:
        markets = OIS_MARKETS

    lambdas = calibrate_lambda(markets=markets, start=start, end=end)

    return {
        m: build_rv_signals(model=m, markets=markets,
                             start=start, end=end, lambdas=lambdas)
        for m in ["ns_calibrated", "ns_free", "nss_calibrated", "nss_free"]
    }


# ── Grid-based RV signals (0.5Y-spaced tenors, no pre-defined structures) ────

def build_rv_signals_grid(
    model: str = "ns_calibrated",
    markets: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    lambdas: Optional[dict] = None,
    tenors: Optional[np.ndarray] = None,
    smooth: bool = False,
    halflife_table: Optional[pd.DataFrame] = None,
) -> dict[str, pd.DataFrame]:
    """
    NS residuals on a uniform tenor grid, one DataFrame per market.

    NS is fitted using only the specified tenors (default: HALF_YEAR_TENORS,
    0.5Y … 30Y in 0.5Y steps). Each tenor's residual is an independent signal
    — no hardcoded slope/fly structures and no short/medium/long division.

    Positive residual → yield above fitted curve → tenor is cheap (receive).
    Negative residual → yield below fitted curve → tenor is rich (pay).

    Parameters
    ----------
    tenors : tenor grid in years; defaults to HALF_YEAR_TENORS (60 points)

    Returns
    -------
    dict[market → date × tenor DataFrame of raw NS residuals (bps × 100)]
    """
    if markets is None:
        markets = OIS_MARKETS
    if tenors is None:
        tenors = HALF_YEAR_TENORS
    if lambdas is None:
        lambdas = calibrate_lambda(markets=markets, start=start, end=end)
    if smooth and halflife_table is None:
        halflife_table = compute_halflife_table(
            markets=markets, model=model, start=start, end=end,
            lambdas=lambdas, tenors=tenors,
        )

    out: dict[str, pd.DataFrame] = {}
    for market in markets:
        hl_map = halflife_table.loc[market].to_dict() if smooth else None
        resid, _ = _market_residuals(
            market, model, lambdas,
            start=start, end=end, halflife_map=hl_map, tenors=tenors,
        )
        resid.index.name = "date"
        out[market] = resid

    return out


def build_rv_returns_grid(
    model: str = "ns_calibrated",
    markets: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    lambdas: Optional[dict] = None,
    tenors: Optional[np.ndarray] = None,
    smooth: bool = False,
    halflife_table: Optional[pd.DataFrame] = None,
) -> dict[str, pd.DataFrame]:
    """
    Daily P&L of being long each tenor point for one day, on the grid.

    Uses the previous day's DV01-scaled residual change:
      return(t, τ) = Δresid(t, τ)  [basis points]

    Grid tenors default to HALF_YEAR_TENORS (same as build_rv_signals_grid).

    Returns
    -------
    dict[market → date × tenor DataFrame of first-differenced residuals]
    """
    signals = build_rv_signals_grid(
        model=model, markets=markets, start=start, end=end,
        lambdas=lambdas, tenors=tenors, smooth=smooth,
        halflife_table=halflife_table,
    )
    return {mkt: df.diff() for mkt, df in signals.items()}
