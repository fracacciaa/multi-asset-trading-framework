"""
Nelson-Siegel (NS) and Nelson-Siegel-Svensson (NSS) yield curve factor extraction.

Four models, differing in how λ is determined:
  ns_calibrated  : y(τ) = β₁ + β₂·L(τ;λ) + β₃·C(τ;λ)
                   λ = per-market historical median (see calibrate_lambda)
  ns_free        : same, λ optimised per date via bounded scalar search
  nss_calibrated : y(τ) = β₁ + β₂·L₁ + β₃·C₁ + β₄·C₂
                   (λ₁, λ₂) = per-market historical medians
  nss_free       : same, (λ₁, λ₂) jointly optimised per date (L-BFGS-B)

Loadings:
  L(τ;λ)  = (1 − exp(−τ/λ)) / (τ/λ)
  C(τ;λ)  = L(τ;λ) − exp(−τ/λ)

λ calibration (calibrate_lambda):
  Run NS_free / NSS_free on every subsample-th date, compute per-market medians.
  Markets with fewer than _MIN_CALIBRATION_DATES valid fits fall back to the
  cross-market global median (used for short-history EMs: Israel, Philippines).

Factor naming (all models):
  β₁ = level    (yield as τ→∞)
  β₂ = slope    (negative for normal upward-sloping curve)
  β₃ = curvature / curvature1
  β₄ = curvature2   (NSS only)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from statsmodels.tsa.stattools import adfuller
from typing import Optional

from rates.production.yc_builder import build_zero_curve, TARGET_TENORS
from rates.production.ois_data import fetch_ois_curves, OIS_MARKETS

# ── Constants ─────────────────────────────────────────────────────────────────

_LAM_INIT_NS   = 2.0
_LAM1_INIT_NSS = 1.0
_LAM2_INIT_NSS = 5.0

_LAM1_BOUNDS = (0.1, 3.0)    # NSS λ₁: short-end hump
_LAM2_BOUNDS = (4.0, 15.0)   # NSS λ₂: long-end hump

MIN_GRID_PTS = 10
_MIN_CALIBRATION_DATES = 100  # below this, per-market median is unreliable

# Index in TARGET_TENORS closest to 10Y
_TEN_YR_IDX: int = int(np.argmin(np.abs(TARGET_TENORS - 10.0)))

ModelType = str  # 'ns_calibrated' | 'ns_free' | 'nss_calibrated' | 'nss_free' | 'ns_ewm' | 'nss_ewm'


# ── Loading matrices ───────────────────────────────────────────────────────────

def _ns_loadings(tenors: np.ndarray, lam: float) -> np.ndarray:
    """(N×3) design matrix [1, L(τ;λ), C(τ;λ)]."""
    x = tenors / lam
    exp_x = np.exp(-x)
    L = np.where(x == 0, 1.0, (1.0 - exp_x) / x)
    C = L - exp_x
    return np.column_stack([np.ones(len(tenors)), L, C])


def _nss_loadings(tenors: np.ndarray, lam1: float, lam2: float) -> np.ndarray:
    """(N×4) design matrix [1, L(τ;λ₁), C(τ;λ₁), C(τ;λ₂)]."""
    x1 = tenors / lam1
    exp1 = np.exp(-x1)
    L1 = np.where(x1 == 0, 1.0, (1.0 - exp1) / x1)
    C1 = L1 - exp1

    x2 = tenors / lam2
    exp2 = np.exp(-x2)
    L2 = np.where(x2 == 0, 1.0, (1.0 - exp2) / x2)
    C2 = L2 - exp2

    return np.column_stack([np.ones(len(tenors)), L1, C1, C2])


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """OLS fit. Returns (betas, rss)."""
    betas, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    rss = float(np.sum((X @ betas - y) ** 2))
    return betas, rss


# ── Per-row fitting functions ──────────────────────────────────────────────────

def fit_ns_fixed(
    tenors: np.ndarray,
    yields: np.ndarray,
    lam: float = _LAM_INIT_NS,
) -> Optional[dict]:
    """NS OLS with fixed λ. Returns dict{betas, lambda, rss} or None."""
    mask = np.isfinite(yields)
    if mask.sum() < MIN_GRID_PTS:
        return None
    t, y = tenors[mask], yields[mask]
    X = _ns_loadings(t, lam)
    betas, rss = _ols(X, y)
    return {"betas": betas, "lambda": lam, "rss": rss}


def fit_ns_free(
    tenors: np.ndarray,
    yields: np.ndarray,
    lam_init: float = _LAM_INIT_NS,
) -> Optional[dict]:
    """NS with λ optimised via bounded scalar search."""
    mask = np.isfinite(yields)
    if mask.sum() < MIN_GRID_PTS:
        return None
    t, y = tenors[mask], yields[mask]

    def rss_of_lam(lam):
        b, _ = _ols(_ns_loadings(t, lam), y)
        return float(np.sum((_ns_loadings(t, lam) @ b - y) ** 2))

    res = minimize_scalar(rss_of_lam, bounds=(0.1, 10.0), method="bounded")
    lam = float(res.x)
    X = _ns_loadings(t, lam)
    betas, rss = _ols(X, y)
    return {"betas": betas, "lambda": lam, "rss": rss}


def fit_nss_fixed(
    tenors: np.ndarray,
    yields: np.ndarray,
    lam1: float = _LAM1_INIT_NSS,
    lam2: float = _LAM2_INIT_NSS,
) -> Optional[dict]:
    """NSS OLS with fixed (λ₁, λ₂)."""
    mask = np.isfinite(yields)
    if mask.sum() < MIN_GRID_PTS:
        return None
    t, y = tenors[mask], yields[mask]
    X = _nss_loadings(t, lam1, lam2)
    betas, rss = _ols(X, y)
    return {"betas": betas, "lambda1": lam1, "lambda2": lam2, "rss": rss}


def fit_nss_free(
    tenors: np.ndarray,
    yields: np.ndarray,
    lam1_init: float = _LAM1_INIT_NSS,
    lam2_init: float = _LAM2_INIT_NSS,
) -> Optional[dict]:
    """
    NSS with (λ₁, λ₂) jointly optimised via L-BFGS-B.

    Bounds keep λ₁ in [0.1, 3.0] and λ₂ in [4.0, 15.0] to prevent near-
    degenerate collinear curvature columns. Falls back to fixed (1.0, 5.0)
    if the optimal β₁ is implausibly large or negative.
    """
    mask = np.isfinite(yields)
    if mask.sum() < MIN_GRID_PTS:
        return None
    t, y = tenors[mask], yields[mask]

    def rss_of_lams(lams):
        X = _nss_loadings(t, lams[0], lams[1])
        b, _ = _ols(X, y)
        return float(np.sum((X @ b - y) ** 2))

    x0 = [
        float(np.clip(lam1_init, *_LAM1_BOUNDS)),
        float(np.clip(lam2_init, *_LAM2_BOUNDS)),
    ]
    res = minimize(
        rss_of_lams, x0=x0,
        bounds=[_LAM1_BOUNDS, _LAM2_BOUNDS],
        method="L-BFGS-B",
        options={"maxiter": 300, "ftol": 1e-10},
    )
    best_lam1, best_lam2 = float(res.x[0]), float(res.x[1])
    X = _nss_loadings(t, best_lam1, best_lam2)
    betas, rss = _ols(X, y)

    if not (-5.0 < betas[0] < 50.0):
        X = _nss_loadings(t, _LAM1_INIT_NSS, _LAM2_INIT_NSS)
        betas, rss = _ols(X, y)
        best_lam1, best_lam2 = _LAM1_INIT_NSS, _LAM2_INIT_NSS

    return {"betas": betas, "lambda1": best_lam1, "lambda2": best_lam2, "rss": rss}


# ── Lambda calibration ─────────────────────────────────────────────────────────

def calibrate_lambda(
    markets: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    subsample: int = 20,
) -> dict:
    """
    Calibrate per-market λ by running free-λ models on a subsampled history.

    For each market, computes median of optimal λ values across all subsampled
    dates. Markets with fewer than _MIN_CALIBRATION_DATES valid fits fall back
    to the cross-market global median.

    Parameters
    ----------
    subsample : fit every N-th date (20 → ~12 samples/year for a daily series)

    Returns
    -------
    {"ns":  {market: lambda_float},
     "nss": {market: (lambda1_float, lambda2_float)}}
    """
    if markets is None:
        markets = OIS_MARKETS

    ns_lams:  dict[str, list[float]]               = {m: [] for m in markets}
    nss_lams: dict[str, list[tuple[float, float]]] = {m: [] for m in markets}

    for market in markets:
        raw = fetch_ois_curves(market, start=start, end=end)
        zc  = build_zero_curve(raw)

        for _, row in zc.iloc[::subsample].iterrows():
            y = row.values.astype(float)

            r = fit_ns_free(TARGET_TENORS, y)
            if r is not None:
                ns_lams[market].append(r["lambda"])

            r = fit_nss_free(TARGET_TENORS, y)
            if r is not None:
                nss_lams[market].append((r["lambda1"], r["lambda2"]))

    # Global fallback medians (cross-market)
    all_ns  = [v for vals in ns_lams.values()  for v in vals]
    all_l1  = [v[0] for vals in nss_lams.values() for v in vals]
    all_l2  = [v[1] for vals in nss_lams.values() for v in vals]
    g_ns   = float(np.median(all_ns))  if all_ns  else _LAM_INIT_NS
    g_lam1 = float(np.median(all_l1)) if all_l1 else _LAM1_INIT_NSS
    g_lam2 = float(np.median(all_l2)) if all_l2 else _LAM2_INIT_NSS

    ns_result:  dict[str, float]               = {}
    nss_result: dict[str, tuple[float, float]] = {}

    for market in markets:
        ns_v  = ns_lams[market]
        nss_v = nss_lams[market]

        ns_result[market] = (float(np.median(ns_v))
                             if len(ns_v) >= _MIN_CALIBRATION_DATES else g_ns)
        nss_result[market] = (
            (float(np.median([v[0] for v in nss_v])),
             float(np.median([v[1] for v in nss_v])))
            if len(nss_v) >= _MIN_CALIBRATION_DATES
            else (g_lam1, g_lam2)
        )

    return {"ns": ns_result, "nss": nss_result}


# ── Factor extraction ──────────────────────────────────────────────────────────

def _factor_columns(model: str) -> list[str]:
    return (["level", "slope", "curvature1", "curvature2"] if model.startswith("nss")
            else ["level", "slope", "curvature"])


def extract_factors(
    zero_curve: pd.DataFrame,
    model: ModelType = "ns_calibrated",
    lam: Optional[float] = None,
    lam1: Optional[float] = None,
    lam2: Optional[float] = None,
) -> pd.DataFrame:
    """
    Fit NS/NSS to every row of a zero curve DataFrame.

    Parameters
    ----------
    zero_curve : date × 70-tenor DataFrame from build_zero_curve()
    model      : 'ns_calibrated' | 'ns_free' | 'nss_calibrated' | 'nss_free'
    lam        : fixed λ for ns_calibrated (from calibrate_lambda output)
    lam1, lam2 : fixed (λ₁, λ₂) for nss_calibrated

    Returns
    -------
    date × factor DataFrame.
    NS models  : [level, slope, curvature]              (+ lambda for ns_free)
    NSS models : [level, slope, curvature1, curvature2] (+ lambda1, lambda2)
    Free-λ variants warm-start from the previous day's lambda.
    """
    if model == "ns_calibrated" and lam is None:
        raise ValueError("ns_calibrated requires lam=; call calibrate_lambda() first")
    if model == "nss_calibrated" and (lam1 is None or lam2 is None):
        raise ValueError("nss_calibrated requires lam1= and lam2=; call calibrate_lambda() first")

    factor_cols  = _factor_columns(model)
    prev_lam     = lam  if lam  is not None else _LAM_INIT_NS
    prev_lam1    = lam1 if lam1 is not None else _LAM1_INIT_NSS
    prev_lam2    = lam2 if lam2 is not None else _LAM2_INIT_NSS

    tenors = zero_curve.columns.values.astype(float)
    rows = []
    for date, row in zero_curve.iterrows():
        y = row.values.astype(float)

        if model == "ns_calibrated":
            result = fit_ns_fixed(tenors, y, lam=lam)
        elif model == "ns_free":
            result = fit_ns_free(tenors, y, lam_init=prev_lam)
            if result is not None:
                prev_lam = result["lambda"]
        elif model == "nss_calibrated":
            result = fit_nss_fixed(tenors, y, lam1=lam1, lam2=lam2)
        elif model == "nss_free":
            result = fit_nss_free(tenors, y,
                                   lam1_init=prev_lam1, lam2_init=prev_lam2)
            if result is not None:
                prev_lam1 = result["lambda1"]
                prev_lam2 = result["lambda2"]
        else:
            raise ValueError(f"Unknown model: {model!r}. "
                             "Choose: ns_calibrated, ns_free, nss_calibrated, nss_free")

        if result is not None:
            d = dict(zip(factor_cols, result["betas"]))
            if model == "ns_free":
                d["lambda"] = result["lambda"]
            elif model in ("nss_calibrated", "nss_free"):
                d["lambda1"] = result["lambda1"]
                d["lambda2"] = result["lambda2"]
            rows.append(pd.Series(d, name=date))
        else:
            rows.append(pd.Series({c: np.nan for c in factor_cols}, name=date))

    df = pd.DataFrame(rows)
    df.index.name = "date"
    return df


def extract_factors_all_markets(
    markets: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    model: ModelType = "ns_calibrated",
    lambdas: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Build zero curves and extract NS/NSS factors for all markets in one call.

    If model is 'ns_calibrated' or 'nss_calibrated' and lambdas is not
    provided, calibrate_lambda() runs automatically on the full date range.

    Parameters
    ----------
    lambdas : output of calibrate_lambda(). Reuse to avoid repeated calibration.

    Returns
    -------
    date × (market, factor) MultiIndex DataFrame
    """
    if markets is None:
        markets = OIS_MARKETS

    if model in ("ns_calibrated", "nss_calibrated") and lambdas is None:
        lambdas = calibrate_lambda(markets=markets, start=start, end=end)

    frames = {}
    for market in markets:
        raw = fetch_ois_curves(market, start=start, end=end)
        zc  = build_zero_curve(raw)

        if model == "ns_calibrated":
            df = extract_factors(zc, model=model, lam=lambdas["ns"][market])
        elif model == "nss_calibrated":
            lam1_m, lam2_m = lambdas["nss"][market]
            df = extract_factors(zc, model=model, lam1=lam1_m, lam2=lam2_m)
        else:
            df = extract_factors(zc, model=model)

        frames[market] = df

    result = pd.concat(frames, axis=1)
    result.columns.names = ["market", "factor"]
    return result


# ── Benchmarking helpers ───────────────────────────────────────────────────────

def _adf_pval(series: pd.Series) -> float:
    """ADF p-value for a time series. Returns 1.0 on failure or thin data."""
    s = series.dropna()
    if len(s) < 20:
        return 1.0
    try:
        return float(adfuller(s, autolag="AIC")[1])
    except Exception:
        return 1.0


def _reconstruct_fitted(
    zc: pd.DataFrame,
    factors: pd.DataFrame,
    model: str,
    lam: Optional[float] = None,
    lam1: Optional[float] = None,
    lam2: Optional[float] = None,
) -> pd.DataFrame:
    """
    Reconstruct fitted zero yields from factor loadings.

    Returns a date × tenor DataFrame aligned to the intersection of zc and
    valid (non-NaN) factor rows.
    """
    fc    = _factor_columns(model)
    valid = factors.dropna(subset=fc)
    common = zc.index.intersection(valid.index)
    betas_arr = valid.loc[common, fc].values  # (N, 3 or 4)
    tenors = zc.columns.values.astype(float)

    if model == "ns_calibrated":
        X = _ns_loadings(tenors, lam)
        fitted_arr = (X @ betas_arr.T).T

    elif model == "nss_calibrated":
        X = _nss_loadings(tenors, lam1, lam2)
        fitted_arr = (X @ betas_arr.T).T

    elif model in ("ns_free", "ns_ewm", "ns_blend"):
        lams = valid.loc[common, "lambda"].values
        fitted_arr = np.array([
            _ns_loadings(tenors, float(l)) @ b
            for l, b in zip(lams, betas_arr)
        ])

    else:  # nss_free, nss_ewm, nss_blend
        lam1s = valid.loc[common, "lambda1"].values
        lam2s = valid.loc[common, "lambda2"].values
        fitted_arr = np.array([
            _nss_loadings(tenors, float(l1), float(l2)) @ b
            for l1, l2, b in zip(lam1s, lam2s, betas_arr)
        ])

    return pd.DataFrame(fitted_arr, index=common, columns=tenors)


def extract_factors_ewm(
    zero_curve: pd.DataFrame,
    ns_or_nss: str = "ns",
    halflife: float = 252,
    lam_init: Optional[float] = None,
    lam1_init: Optional[float] = None,
    lam2_init: Optional[float] = None,
) -> pd.DataFrame:
    """
    NS or NSS factor extraction with EWM-smoothed λ.

    Steps:
      1. Run the free-λ model on every date (warm-started) to collect raw λ(t).
      2. Apply causal EWM smoothing: λ̃(t) = ewm(halflife).mean() of the raw series.
      3. Re-fit OLS per date using λ̃(t) as a fixed input.

    Returns the same column structure as ns_free / nss_free (factors + smoothed
    lambda columns), so _reconstruct_fitted works with model='ns_ewm'/'nss_ewm'.

    Parameters
    ----------
    ns_or_nss : 'ns' or 'nss'
    halflife  : EWM halflife in trading days (controls adaptation speed)
    lam_init  : warm-start for NS free-λ optimisation (optional)
    lam1_init, lam2_init : warm-start for NSS (optional)
    """
    tenors = zero_curve.columns.values.astype(float)
    prev_lam  = lam_init  if lam_init  is not None else _LAM_INIT_NS
    prev_lam1 = lam1_init if lam1_init is not None else _LAM1_INIT_NSS
    prev_lam2 = lam2_init if lam2_init is not None else _LAM2_INIT_NSS

    # ── pass 1: collect raw daily optimal λ ──────────────────────────────────
    raw_lam, raw_lam1, raw_lam2, dates = [], [], [], []
    for date, row in zero_curve.iterrows():
        y = row.values.astype(float)
        if ns_or_nss == "ns":
            r = fit_ns_free(tenors, y, lam_init=prev_lam)
            if r is not None:
                prev_lam = r["lambda"]
                raw_lam.append(r["lambda"])
            else:
                raw_lam.append(np.nan)
        else:
            r = fit_nss_free(tenors, y, lam1_init=prev_lam1, lam2_init=prev_lam2)
            if r is not None:
                prev_lam1, prev_lam2 = r["lambda1"], r["lambda2"]
                raw_lam1.append(r["lambda1"])
                raw_lam2.append(r["lambda2"])
            else:
                raw_lam1.append(np.nan)
                raw_lam2.append(np.nan)
        dates.append(date)

    # ── pass 2: EWM smoothing (causal) ───────────────────────────────────────
    if ns_or_nss == "ns":
        lam_s = (pd.Series(raw_lam, index=dates)
                   .ewm(halflife=halflife, min_periods=1).mean())
    else:
        lam1_s = (pd.Series(raw_lam1, index=dates)
                    .ewm(halflife=halflife, min_periods=1).mean())
        lam2_s = (pd.Series(raw_lam2, index=dates)
                    .ewm(halflife=halflife, min_periods=1).mean())

    # ── pass 3: OLS per date using smoothed λ̃(t) ────────────────────────────
    if ns_or_nss == "ns":
        fc = ["level", "slope", "curvature"]
    else:
        fc = ["level", "slope", "curvature1", "curvature2"]

    rows = []
    for date, row in zero_curve.iterrows():
        y = row.values.astype(float)
        if ns_or_nss == "ns":
            lam_t = float(lam_s.loc[date])
            result = fit_ns_fixed(tenors, y, lam=lam_t)
            if result is not None:
                d = dict(zip(fc, result["betas"]))
                d["lambda"] = lam_t
                rows.append(pd.Series(d, name=date))
            else:
                rows.append(pd.Series({c: np.nan for c in fc + ["lambda"]}, name=date))
        else:
            lam1_t = float(lam1_s.loc[date])
            lam2_t = float(lam2_s.loc[date])
            result = fit_nss_fixed(tenors, y, lam1=lam1_t, lam2=lam2_t)
            if result is not None:
                d = dict(zip(fc, result["betas"]))
                d["lambda1"] = lam1_t
                d["lambda2"] = lam2_t
                rows.append(pd.Series(d, name=date))
            else:
                rows.append(pd.Series(
                    {c: np.nan for c in fc + ["lambda1", "lambda2"]}, name=date))

    df = pd.DataFrame(rows)
    df.index.name = "date"
    return df


def extract_factors_blend(
    zero_curve: pd.DataFrame,
    ns_or_nss: str = "ns",
    alpha: float = 0.5,
    lam_calibrated: Optional[float] = None,
    lam1_calibrated: Optional[float] = None,
    lam2_calibrated: Optional[float] = None,
    free_factors: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    NS or NSS factor extraction with blended λ:
      λ_blend(t) = α × λ_free(t) + (1−α) × λ_calibrated

    α = 0 → pure calibrated;  α = 1 → pure free.

    The free optimisation runs once (or can be passed in as `free_factors` to
    avoid re-running when sweeping α). Re-fits OLS per date with the blended λ.

    Returns same column structure as ns_free / nss_free (including per-date
    lambda columns), so _reconstruct_fitted works with model='ns_blend'/'nss_blend'.

    Parameters
    ----------
    alpha            : blend weight on the free lambda (0 = calibrated, 1 = free)
    lam_calibrated   : fixed NS lambda (from calibrate_lambda)
    lam1_calibrated,
    lam2_calibrated  : fixed NSS lambdas
    free_factors     : pre-computed output of extract_factors(model='ns_free' /
                       'nss_free') — reuse to avoid re-running the optimisation
    """
    tenors = zero_curve.columns.values.astype(float)

    # ── get per-date free lambdas (run once or reuse) ─────────────────────────
    if free_factors is None:
        free_model = "ns_free" if ns_or_nss == "ns" else "nss_free"
        free_factors = extract_factors(zero_curve, model=free_model)

    # ── blend ─────────────────────────────────────────────────────────────────
    if ns_or_nss == "ns":
        if lam_calibrated is None:
            raise ValueError("lam_calibrated required for ns_blend")
        lam_blend = (alpha * free_factors["lambda"]
                     + (1 - alpha) * lam_calibrated)
        fc = ["level", "slope", "curvature"]
    else:
        if lam1_calibrated is None or lam2_calibrated is None:
            raise ValueError("lam1_calibrated and lam2_calibrated required for nss_blend")
        lam1_blend = (alpha * free_factors["lambda1"]
                      + (1 - alpha) * lam1_calibrated)
        lam2_blend = (alpha * free_factors["lambda2"]
                      + (1 - alpha) * lam2_calibrated)
        fc = ["level", "slope", "curvature1", "curvature2"]

    # ── re-fit OLS per date with blended λ ───────────────────────────────────
    rows = []
    for date, row in zero_curve.iterrows():
        if date not in free_factors.index:
            continue
        y = row.values.astype(float)
        if ns_or_nss == "ns":
            lam_t = float(lam_blend.loc[date])
            result = fit_ns_fixed(tenors, y, lam=lam_t)
            if result is not None:
                d = dict(zip(fc, result["betas"]))
                d["lambda"] = lam_t
                rows.append(pd.Series(d, name=date))
            else:
                rows.append(pd.Series({c: np.nan for c in fc + ["lambda"]}, name=date))
        else:
            l1_t = float(lam1_blend.loc[date])
            l2_t = float(lam2_blend.loc[date])
            result = fit_nss_fixed(tenors, y, lam1=l1_t, lam2=l2_t)
            if result is not None:
                d = dict(zip(fc, result["betas"]))
                d["lambda1"] = l1_t
                d["lambda2"] = l2_t
                rows.append(pd.Series(d, name=date))
            else:
                rows.append(pd.Series(
                    {c: np.nan for c in fc + ["lambda1", "lambda2"]}, name=date))

    df = pd.DataFrame(rows)
    df.index.name = "date"
    return df


# ── Benchmarking ───────────────────────────────────────────────────────────────

def benchmark_models(
    markets: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    models: Optional[list[str]] = None,
    subsample: int = 10,
) -> pd.DataFrame:
    """
    Compare NS/NSS models across markets on fit quality and factor properties.

    Metrics (mean across markets):
      fit_rmse_bps         : RMSE of fitted vs actual yields in basis points
                             (subsampled for speed — every subsample-th date)
      level_stability      : σ of daily Δβ₁
      slope_stability      : σ of daily Δβ₂
      level_r2_10y         : r² of β₁ vs 10Y zero rate
      resid_stationary_pct : % of 70 tenors whose residual series is I(0) (ADF p<0.05)
      beta1_adf_pval       : median ADF p-value for β₁ across markets
      beta2_adf_pval       : median ADF p-value for β₂ across markets

    Parameters
    ----------
    subsample : every N-th date for RMSE; ADF and stability use the full history
    """
    if markets is None:
        markets = OIS_MARKETS
    if models is None:
        models = ["ns_calibrated", "ns_free", "nss_calibrated", "nss_free"]

    # Calibrate once; use 2× subsample to keep calibration fast
    lambdas = calibrate_lambda(markets=markets, start=start, end=end,
                               subsample=subsample * 2)

    rows = []
    for model in models:
        rmse_vals, level_stab, slope_stab = [], [], []
        level_r2, resid_stat, b1_pvals, b2_pvals = [], [], [], []

        for market in markets:
            raw = fetch_ois_curves(market, start=start, end=end)
            zc  = build_zero_curve(raw)

            lam_m = lam1_m = lam2_m = None
            if model == "ns_calibrated":
                lam_m = lambdas["ns"][market]
                fac = extract_factors(zc, model=model, lam=lam_m)
            elif model == "nss_calibrated":
                lam1_m, lam2_m = lambdas["nss"][market]
                fac = extract_factors(zc, model=model, lam1=lam1_m, lam2=lam2_m)
            else:
                fac = extract_factors(zc, model=model)

            # Factor stability (full history)
            level_stab.append(float(fac["level"].diff().std()))
            slope_stab.append(float(fac["slope"].diff().std()))

            # Level r² vs 10Y zero (full history)
            zc_10y  = zc.iloc[:, _TEN_YR_IDX]
            lvl     = fac["level"].dropna()
            common  = lvl.index.intersection(zc_10y.dropna().index)
            if len(common) > 30:
                r2 = float(np.corrcoef(lvl.loc[common].values,
                                       zc_10y.loc[common].values)[0, 1] ** 2)
                level_r2.append(r2)

            # Factor ADF tests (full history)
            b1_pvals.append(_adf_pval(fac["level"]))
            b2_pvals.append(_adf_pval(fac["slope"]))

            # Reconstruct fitted yields
            fitted = _reconstruct_fitted(zc, fac, model,
                                          lam=lam_m, lam1=lam1_m, lam2=lam2_m)
            resid  = zc.loc[fitted.index].sub(fitted)   # date × 70 tenors

            # RMSE on subsampled dates (bps)
            sub_resid = resid.iloc[::subsample]
            rmse_per_date = sub_resid.apply(
                lambda r: float(np.sqrt(np.nanmean(r.values ** 2))) * 100, axis=1
            )
            if len(rmse_per_date) > 0:
                rmse_vals.append(float(rmse_per_date.mean()))

            # Residual stationarity: ADF on full residual time series per tenor
            n_stat = sum(1 for col in resid.columns if _adf_pval(resid[col]) < 0.05)
            resid_stat.append(100.0 * n_stat / len(TARGET_TENORS))

        rows.append({
            "model":                model,
            "fit_rmse_bps":         float(np.nanmean(rmse_vals)),
            "level_stability":      float(np.nanmean(level_stab)),
            "slope_stability":      float(np.nanmean(slope_stab)),
            "level_r2_10y":         float(np.nanmean(level_r2)),
            "resid_stationary_pct": float(np.nanmean(resid_stat)),
            "beta1_adf_pval":       float(np.nanmedian(b1_pvals)),
            "beta2_adf_pval":       float(np.nanmedian(b2_pvals)),
        })

    return pd.DataFrame(rows).set_index("model")
