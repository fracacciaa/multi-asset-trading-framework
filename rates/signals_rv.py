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


# ── Per-market residual computation ───────────────────────────────────────────

def _market_residuals(
    market: str,
    model: str,
    lambdas: dict,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (residuals, zero_curve) for one market under one model.
    Both are date × tenor DataFrames on the same index.
    """
    raw = fetch_ois_curves(market, start=start, end=end)
    zc  = build_zero_curve(raw)

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
    return resid, zc.loc[fitted.index]


# ── Portfolio-level builders ───────────────────────────────────────────────────

def build_rv_signals(
    model: str = "ns_calibrated",
    markets: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    lambdas: Optional[dict] = None,
) -> dict[str, pd.DataFrame]:
    """
    Build all three RV signals for all markets under one NS/NSS model.

    Parameters
    ----------
    model   : 'ns_calibrated' | 'ns_free' | 'nss_calibrated' | 'nss_free'
    lambdas : output of calibrate_lambda(); auto-computed if None

    Returns
    -------
    dict with keys 'slope_2s10s', 'fly_2s5s10s', 'fly_5s10s20s'.
    Each value is a date × market DataFrame of raw (unnormalised) signals.
    """
    if markets is None:
        markets = OIS_MARKETS
    if lambdas is None:
        lambdas = calibrate_lambda(markets=markets, start=start, end=end)

    slope_d, fly1_d, fly2_d = {}, {}, {}

    for market in markets:
        resid, zc = _market_residuals(market, model, lambdas,
                                       start=start, end=end)
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
