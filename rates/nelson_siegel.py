"""
Nelson-Siegel (NS) and Nelson-Siegel-Svensson (NSS) yield curve factor extraction.

Four models, differing in number of factors and how λ is estimated:

  NS_fixed  : y(τ) = β₁ + β₂·L(τ;λ) + β₃·C(τ;λ)                  λ fixed = 2.0
  NS_free   : same functional form, λ optimised per day (scalar search)
  NSS_fixed : y(τ) = β₁ + β₂·L₁ + β₃·C₁ + β₄·C₂                  λ₁=1.0, λ₂=5.0 fixed
  NSS_free  : same 4-factor form, (λ₁, λ₂) optimised per day (2-D search)

Loadings:
  L(τ;λ)  = (1 − exp(−τ/λ)) / (τ/λ)
  C(τ;λ)  = L(τ;λ) − exp(−τ/λ)          ← curvature factor (hump at ≈1.5λ)

NS factors:  β₁=level (y as τ→∞), β₂=slope, β₃=curvature
NSS factors: β₁=level, β₂=slope, β₃=first curvature, β₄=second curvature

For fixed-λ variants the β's are computed by OLS (no iteration).
For free-λ variants λ is found by minimising RSS, with β's as nested OLS.

Default model for mean-reversion: NSS_free (most accurate level estimate).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from typing import Optional, Literal

from rates.data import fetch_curves, MARKET_PREFIXES
from rates.bootstrap import bootstrap_curves, COUPON_FREQ
from rates.interpolation import build_full_curves, TARGET_TENORS
from rates.ois_data import fetch_ois_curves, OIS_COUPON_FREQ, OIS_MARKETS

# ── Constants ─────────────────────────────────────────────────────────────────

LAMBDA_NS  = 2.0   # NS fixed λ: curvature hump at ~3Y
LAMBDA1_NSS = 1.0  # NSS fixed λ₁: first hump at ~1.5Y
LAMBDA2_NSS = 5.0  # NSS fixed λ₂: second hump at ~7.5Y
MIN_GRID_PTS = 10

ModelType = Literal["ns_fixed", "ns_free", "nss_fixed", "nss_free"]


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
    lam: float = LAMBDA_NS,
) -> Optional[dict]:
    """NS OLS with fixed λ. Returns dict with betas, lambda, rss."""
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
    lam_init: float = LAMBDA_NS,
) -> Optional[dict]:
    """NS with λ optimised via bounded scalar search."""
    mask = np.isfinite(yields)
    if mask.sum() < MIN_GRID_PTS:
        return None
    t, y = tenors[mask], yields[mask]

    def rss_of_lam(lam):
        X = _ns_loadings(t, lam)
        betas, _ = _ols(X, y)
        return float(np.sum((X @ betas - y) ** 2))

    res = minimize_scalar(rss_of_lam, bounds=(0.1, 10.0), method="bounded")
    lam = float(res.x)
    X = _ns_loadings(t, lam)
    betas, rss = _ols(X, y)
    return {"betas": betas, "lambda": lam, "rss": rss}


def fit_nss_fixed(
    tenors: np.ndarray,
    yields: np.ndarray,
    lam1: float = LAMBDA1_NSS,
    lam2: float = LAMBDA2_NSS,
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
    lam1_init: float = LAMBDA1_NSS,
    lam2_init: float = LAMBDA2_NSS,
) -> Optional[dict]:
    """
    NSS with (λ₁, λ₂) jointly optimised via L-BFGS-B.

    Bounds enforce λ₁ ∈ [0.1, 3.0] and λ₂ ∈ [4.0, 15.0] to prevent
    near-degenerate solutions where λ₁ ≈ λ₂ causes collinear curvature
    columns and numerically unstable betas. The separation ≥ 1 year ensures
    the two curvature factors peak at meaningfully different maturities.

    Multi-start from 4 seeds; returns the global best.
    """
    mask = np.isfinite(yields)
    if mask.sum() < MIN_GRID_PTS:
        return None
    t, y = tenors[mask], yields[mask]

    # λ₁ controls short-end hump, λ₂ controls long-end hump.
    # Keep them in separate bands so they never cross or approach each other.
    LAM1_BOUNDS = (0.1, 3.0)
    LAM2_BOUNDS = (4.0, 15.0)

    def rss_of_lams(lams):
        lam1, lam2 = lams
        X = _nss_loadings(t, lam1, lam2)
        betas, _ = _ols(X, y)
        return float(np.sum((X @ betas - y) ** 2))

    x0 = [
        max(LAM1_BOUNDS[0], min(LAM1_BOUNDS[1], lam1_init)),
        max(LAM2_BOUNDS[0], min(LAM2_BOUNDS[1], lam2_init)),
    ]

    res = minimize(
        rss_of_lams,
        x0=x0,
        bounds=[LAM1_BOUNDS, LAM2_BOUNDS],
        method="L-BFGS-B",
        options={"maxiter": 300, "ftol": 1e-10},
    )
    best_lam1, best_lam2 = float(res.x[0]), float(res.x[1])

    X = _nss_loadings(t, best_lam1, best_lam2)
    betas, rss = _ols(X, y)

    # If β₁ is implausible (negative or > 50%), the model found a degenerate
    # solution (level absorbed by large β₄·C₂). Fall back to nss_fixed.
    if not (-5.0 < betas[0] < 50.0):
        X_fb = _nss_loadings(t, LAMBDA1_NSS, LAMBDA2_NSS)
        betas, rss = _ols(X_fb, y)
        best_lam1, best_lam2 = LAMBDA1_NSS, LAMBDA2_NSS

    return {"betas": betas, "lambda1": best_lam1, "lambda2": best_lam2, "rss": rss}


# ── Backward-compatible wrapper ────────────────────────────────────────────────

def fit_nelson_siegel(
    tenors: np.ndarray,
    yields: np.ndarray,
    lam: float = LAMBDA_NS,
) -> Optional[np.ndarray]:
    """NS OLS with fixed λ. Returns array([β₁, β₂, β₃]) or None."""
    result = fit_ns_fixed(tenors, yields, lam=lam)
    return result["betas"] if result else None


# ── Full-curve factor extraction ───────────────────────────────────────────────

def _factor_columns(model: ModelType) -> list[str]:
    if model.startswith("ns"):
        return ["level", "slope", "curvature"]
    return ["level", "slope", "curvature1", "curvature2"]


def extract_factors(
    market: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    model: ModelType = "ns_fixed",
    use_ois: bool = True,
) -> pd.DataFrame:
    """
    Full pipeline for one market: rates → bootstrap → MP interpolation → NS/NSS fit.

    Parameters
    ----------
    model    : 'ns_fixed' | 'ns_free' | 'nss_fixed' | 'nss_free'
    use_ois  : True = OIS rates (default), False = government par yields

    Returns
    -------
    date × factor DataFrame. NS: [level, slope, curvature].
    NSS: [level, slope, curvature1, curvature2].
    Free-λ models also include lambda (ns) or lambda1/lambda2 (nss) columns.
    """
    if use_ois:
        raw = fetch_ois_curves(market, start=start, end=end)
        coupon_freq = OIS_COUPON_FREQ
        min_tenors = 6
    else:
        raw = fetch_curves(market, start=start, end=end)
        coupon_freq = COUPON_FREQ[market]
        min_tenors = 4

    zeros = bootstrap_curves(raw, coupon_freq=coupon_freq, min_tenors=min_tenors)
    dense = build_full_curves(zeros, method="monotone_preserving")

    fit_fns = {
        "ns_fixed":  lambda t, y: fit_ns_fixed(t, y),
        "ns_free":   lambda t, y: fit_ns_free(t, y),
        "nss_fixed": lambda t, y: fit_nss_fixed(t, y),
        "nss_free":  lambda t, y: fit_nss_free(t, y),
    }
    fit_fn = fit_fns[model]

    # For free-λ models, warm-start from previous day's lambdas
    prev_lam: Optional[dict] = None
    rows = []
    for date, row in dense.iterrows():
        if model == "ns_free" and prev_lam:
            result = fit_ns_free(TARGET_TENORS, row.values,
                                  lam_init=prev_lam.get("lambda", LAMBDA_NS))
        elif model == "nss_free" and prev_lam:
            result = fit_nss_free(TARGET_TENORS, row.values,
                                   lam1_init=prev_lam.get("lambda1", LAMBDA1_NSS),
                                   lam2_init=prev_lam.get("lambda2", LAMBDA2_NSS))
        else:
            result = fit_fn(TARGET_TENORS, row.values)

        if result is not None:
            prev_lam = result
            factor_cols = _factor_columns(model)
            d = dict(zip(factor_cols, result["betas"]))
            if model == "ns_free":
                d["lambda"] = result["lambda"]
            elif model in ("nss_fixed", "nss_free"):
                d["lambda1"] = result["lambda1"]
                d["lambda2"] = result["lambda2"]
            rows.append(pd.Series(d, name=date))
        else:
            factor_cols = _factor_columns(model)
            rows.append(pd.Series({c: np.nan for c in factor_cols}, name=date))
            prev_lam = None

    df = pd.DataFrame(rows)
    df.index.name = "date"
    return df


def build_factors_all_markets(
    markets: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    model: ModelType = "ns_fixed",
    use_ois: bool = True,
) -> pd.DataFrame:
    """
    Extract NS/NSS factors for all (or selected) markets.

    Returns date × (market, factor) MultiIndex column DataFrame.
    """
    if markets is None:
        markets = OIS_MARKETS if use_ois else list(MARKET_PREFIXES.values())

    frames = {}
    for market in markets:
        df = extract_factors(market, start=start, end=end,
                             model=model, use_ois=use_ois)
        frames[market] = df

    result = pd.concat(frames, axis=1)
    result.columns.names = ["market", "factor"]
    return result
