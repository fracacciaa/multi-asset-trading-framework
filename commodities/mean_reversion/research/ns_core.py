"""
NS/NSS factor extraction for commodity futures term structures.

The analogue of a yield curve is the log-futures price curve:
  p(n) = log(F_n),  n ∈ {1, …, N}  (contract slot)
  τ_n  = n / 12     (months-to-expiry in years)

Four model variants matching the rates NS library:
  ns_calibrated  — 3-factor NS, per-commodity median λ fixed OOS
  ns_free        — 3-factor NS, λ optimised per date
  bc_calibrated  — 4-factor Bjork-Christensen, per-commodity median λ fixed
  bc_free        — 4-factor BC, λ optimised per date

Residuals: in-sample and LOO (via rates.production.ns_core.fit_ns).
Factor-neutral projection removes the NS factor component from residuals,
isolating idiosyncratic contract mispricing.

λ convention: lam in 1/years, x = τ × lam  (same as rates/production/ns_core.py)
Slope loading peaks at τ = 1/lam years.  For 1–8 month tenors a lam ≈ 6 (1/yr)
puts the peak at ~2 months, in the middle of the curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from typing import Optional

from rates.production.ns_core import fit_ns, LOADINGS

# ── Tenor grid ─────────────────────────────────────────────────────────────────

MAX_CONTRACTS = 8

def tenors_for(n: int) -> np.ndarray:
    """τ array in years for n contract slots (τ_i = i/12)."""
    return np.arange(1, n + 1) / 12.0


# ── λ search bounds (lam in 1/years) ──────────────────────────────────────────
# Commodity curves span 1–8 months (0.083–0.667 yr).
# Slope peak = 1/lam: lam=6 → peak at 2 months; lam=3 → 4 months.

_LAM_BOUNDS = (1.0, 30.0)
_LAM_INIT   = 6.0

MIN_CONTRACTS = 3     # minimum non-NaN contracts to attempt a fit
TRAIN_END     = "2018-12-31"
_MIN_CALIB_FITS = 30  # below this, fall back to cross-commodity global median


# ── Low-level fits ─────────────────────────────────────────────────────────────

def _ols(X: np.ndarray, y: np.ndarray):
    b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return b, float(np.sum((X @ b - y) ** 2))


def _fit_fixed(tenors: np.ndarray, y: np.ndarray, lam: float, basis: str = "ns"):
    mask = np.isfinite(y)
    if mask.sum() < MIN_CONTRACTS:
        return None
    X = LOADINGS[basis](tenors[mask], lam)
    b, rss = _ols(X, y[mask])
    return {"betas": b, "lambda": lam, "rss": rss, "basis": basis}


def _fit_free(tenors: np.ndarray, y: np.ndarray, lam_init: float = _LAM_INIT,
              basis: str = "ns"):
    mask = np.isfinite(y)
    if mask.sum() < MIN_CONTRACTS:
        return None
    t, yv = tenors[mask], y[mask]

    def _rss(lam):
        X = LOADINGS[basis](t, lam)
        b, _ = _ols(X, yv)
        return float(np.sum((X @ b - yv) ** 2))

    res = minimize_scalar(_rss, bounds=_LAM_BOUNDS, method="bounded")
    lam = float(res.x)
    X = LOADINGS[basis](t, lam)
    b, rss = _ols(X, yv)
    return {"betas": b, "lambda": lam, "rss": rss, "basis": basis}


# ── Lambda calibration ─────────────────────────────────────────────────────────

def calibrate_lambda(
    log_curves: dict[str, pd.DataFrame],
    train_end: str = TRAIN_END,
    subsample: int = 20,
) -> dict[str, dict]:
    """
    Per-commodity median λ calibrated on log-price curves up to train_end.

    log_curves : {commodity → date × N DataFrame of log futures prices}

    Returns
    -------
    {"ns": {commodity: lam_float},
     "bc": {commodity: lam_float}}
    """
    ns_lams: dict[str, list] = {}
    bc_lams: dict[str, list] = {}

    for com, df in log_curves.items():
        sub = df.loc[:train_end].dropna(how="all")
        ns_lams[com] = []
        bc_lams[com] = []
        tenors = tenors_for(df.shape[1])

        for _, row in sub.iloc[::subsample].iterrows():
            y = row.values.astype(float)
            r = _fit_free(tenors, y, basis="ns")
            if r is not None:
                ns_lams[com].append(r["lambda"])
            r2 = _fit_free(tenors, y, basis="bc")
            if r2 is not None:
                bc_lams[com].append(r2["lambda"])

    # Global fallback medians
    all_ns = [v for vs in ns_lams.values() for v in vs]
    all_bc = [v for vs in bc_lams.values() for v in vs]
    g_ns = float(np.median(all_ns)) if all_ns else _LAM_INIT
    g_bc = float(np.median(all_bc)) if all_bc else _LAM_INIT

    ns_res, bc_res = {}, {}
    for com in log_curves:
        ns_res[com] = (float(np.median(ns_lams[com]))
                       if len(ns_lams[com]) >= _MIN_CALIB_FITS else g_ns)
        bc_res[com] = (float(np.median(bc_lams[com]))
                       if len(bc_lams[com]) >= _MIN_CALIB_FITS else g_bc)

    return {"ns": ns_res, "bc": bc_res}


# ── Factor extraction ──────────────────────────────────────────────────────────

def _basis_for(model: str) -> str:
    return "bc" if model.startswith("bc") else "ns"


def extract_factors(
    log_curve: pd.DataFrame,
    model: str = "ns_calibrated",
    lam: Optional[float] = None,
) -> pd.DataFrame:
    """
    Fit NS or BC to every row of a log-price term structure.

    log_curve : date × N DataFrame of log futures prices
    lam       : fixed λ for *_calibrated models
    Returns   : date × factor DataFrame (level, slope, curvature [, curv2 for bc])
    """
    basis = _basis_for(model)
    is_free = model.endswith("_free")
    n = log_curve.shape[1]
    tenors = tenors_for(n)
    prev_lam = lam if lam is not None else _LAM_INIT

    n_factors = 3 if basis == "ns" else 4
    fc = (["level", "slope", "curvature"] if n_factors == 3
          else ["level", "slope", "curvature1", "curvature2"])

    rows = []
    for date, row in log_curve.iterrows():
        y = row.values.astype(float)
        if is_free:
            r = _fit_free(tenors, y, lam_init=prev_lam, basis=basis)
            if r is not None:
                prev_lam = r["lambda"]
        else:
            r = _fit_fixed(tenors, y, lam=prev_lam, basis=basis)

        if r is not None:
            d = dict(zip(fc, r["betas"]))
            d["lambda"] = r["lambda"]
            rows.append(pd.Series(d, name=date))
        else:
            rows.append(pd.Series({c: np.nan for c in fc + ["lambda"]}, name=date))

    return pd.DataFrame(rows)


# ── Residuals (in-sample + LOO) ────────────────────────────────────────────────

def extract_residuals(
    log_curve: pd.DataFrame,
    model: str = "ns_calibrated",
    lam: Optional[float] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute in-sample and LOO residuals for one commodity's log-price curve.

    LOO residual corrects for self-absorption: resid_loo = resid / (1 - h_ii).
    With N=8 contracts and 3–4 factors the hat diagonal can exceed 0.3 at
    the curve endpoints, making the correction material.

    Returns (resid_is, resid_loo): both date × contract DataFrames.
    """
    basis = _basis_for(model)
    is_free = model.endswith("_free")
    n = log_curve.shape[1]
    tenors = tenors_for(n)
    prev_lam = lam if lam is not None else _LAM_INIT
    cols = list(range(1, n + 1))

    is_rows, loo_rows = [], []

    for date, row in log_curve.iterrows():
        y = row.values.astype(float)
        mask = np.isfinite(y)

        if mask.sum() < MIN_CONTRACTS:
            nan_row = pd.Series(np.nan, index=cols, name=date)
            is_rows.append(nan_row)
            loo_rows.append(nan_row.copy())
            continue

        t_obs, y_obs = tenors[mask], y[mask]

        if is_free:
            r_pre = _fit_free(t_obs, y_obs, lam_init=prev_lam, basis=basis)
            lam_t = r_pre["lambda"] if r_pre is not None else prev_lam
            if r_pre is not None:
                prev_lam = lam_t
        else:
            lam_t = prev_lam

        # fit_ns from rates.production.ns_core gives LOO for free
        ns_result = fit_ns(t_obs, y_obs, lam=lam_t, basis=basis)

        is_arr  = np.full(n, np.nan)
        loo_arr = np.full(n, np.nan)
        is_arr[mask]  = ns_result.resid
        loo_arr[mask] = ns_result.resid_loo

        is_rows.append(pd.Series(is_arr, index=cols, name=date))
        loo_rows.append(pd.Series(loo_arr, index=cols, name=date))

    return pd.DataFrame(is_rows), pd.DataFrame(loo_rows)


# ── Factor-neutral projection ──────────────────────────────────────────────────

def factor_neutral_residuals(
    resid: pd.DataFrame,
    tenors: np.ndarray,
    lam: float,
    basis: str = "ns",
) -> pd.DataFrame:
    """
    Project each date's residual onto the complement of the NS loading space.

    Removes the component of the residual that the NS factors can explain,
    leaving only idiosyncratic contract mispricing.  Equal observation weights
    (no DV01 analogue for commodities).

    Returns s_perp: date × contract DataFrame, same shape as resid.
    """
    X_full = LOADINGS[basis](tenors, lam)   # (N, k) — full-grid loadings
    arr = resid.values.astype(float)
    out = np.full_like(arr, np.nan)

    for i in range(len(arr)):
        row = arr[i]
        mask = np.isfinite(row)
        if mask.sum() < 3:
            continue
        X = X_full[mask]                    # (m, k)
        r = row[mask]
        try:
            coef = np.linalg.solve(X.T @ X, X.T @ r)
            out[i, mask] = r - X @ coef
        except np.linalg.LinAlgError:
            pass

    return pd.DataFrame(out, index=resid.index, columns=resid.columns)
