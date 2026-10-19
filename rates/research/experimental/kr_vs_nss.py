"""
KRR vs NS/NSS comparison — factor-neutral OOS Sharpe (2019→today).

Models compared:
  ns_cal   : NS with λ calibrated on train (≤2018)
  ns_free  : NS with λ optimised every date
  nss_cal  : NSS with (λ₁,λ₂) calibrated on train
  nss_free : NSS with (λ₁,λ₂) optimised every date
  kr       : Kernel Ridge Regression (cubic spline kernel, μ tuned on train)

All models:
  • Fit on the full 70-tenor interpolated zero curve
  • Residuals extracted at observed long-end tenors (≥1Y raw OIS quotes)
  • Factor-neutralised: DV01-weighted projection onto NS level/slope/curvature
    using calibrated λ per market  (consistent basis across all models)
  • P&L(t) = −signal(t) · Δresid(t+1)  (signal at t, realisation at t→t+1)
"""

import numpy as np
import pandas as pd

from rates.production.ois_data      import fetch_ois_curves, OIS_MARKETS
from rates.production.yc_builder    import build_zero_curve
from rates.production.nelson_siegel import (
    calibrate_lambda, extract_factors, _reconstruct_fitted, _ns_loadings,
)
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernel_ridge import extract_residuals_kr

TRAIN_END = "2018-12-31"
OOS_START = "2019-01-01"
MU_GRID   = [1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]


# ── helpers ────────────────────────────────────────────────────────────────────

def _long_end_tenors(market: str) -> np.ndarray:
    """≥1Y observed tenors for this market, inferred from raw OIS data."""
    raw = fetch_ois_curves(market)
    cols = raw.columns.values.astype(float)
    return np.sort(cols[cols >= 1.0])


def _fn_pnl(
    resid   : np.ndarray,   # (T, N)
    z_arr   : np.ndarray,   # (T, N) zero rates at those tenors (%)
    tenors  : np.ndarray,   # (N,)
    ns_load : np.ndarray,   # (N, 3) NS loading matrix (fixed per market)
) -> np.ndarray:
    """
    Vectorised factor-neutral P&L, length T−1.

    pnl[i] = −s_perp[i] · (resid[i+1] − resid[i])
    where s_perp is DV01-weighted 3PC-neutral residual at date i.
    """
    dv01 = tenors[None, :] * np.exp(-z_arr / 100.0 * tenors[None, :])  # (T, N)

    D   = dv01[:, :, None] * ns_load[None, :, :]               # (T, N, 3)
    DtD = np.einsum('tni,tnj->tij', D, D)                       # (T, 3, 3)
    Dts = np.einsum('tni,tn->ti',   D, resid)                   # (T, 3)
    coef   = np.linalg.solve(DtD, Dts[:, :, None]).squeeze(-1)  # (T, 3)
    s_perp = resid - np.einsum('tni,ti->tn', D, coef)           # (T, N)

    delta  = np.diff(resid, axis=0)                              # (T-1, N)
    pnl    = -np.einsum('tn,tn->t', s_perp[:-1], delta)         # (T-1,)

    bad = ~np.all(np.isfinite(resid), axis=1)
    pnl[bad[1:] | bad[:-1]] = np.nan
    return pnl


def _sharpe(pnl: np.ndarray, oos_start_idx: int) -> float:
    """Annualised Sharpe over pnl[oos_start_idx:]."""
    v = pnl[oos_start_idx:]
    v = v[np.isfinite(v)]
    if len(v) < 60:
        return np.nan
    return float(v.mean() / v.std() * np.sqrt(252))


# ── per-market computation ──────────────────────────────────────────────────────

def _compute_resid(
    zc     : pd.DataFrame,   # full 70-tenor zero curve
    avail  : np.ndarray,     # observed long-end tenors to trade
    model  : str,
    kw     : dict,           # kwargs for extract_factors / _reconstruct_fitted
    mu_kr  : float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (resid, zc_at_avail) for one model at the observed tenors.
    """
    if model == "kr":
        resid, _ = extract_residuals_kr(zc, mu=mu_kr, eval_tenors=avail)
        return resid, zc.loc[resid.index, list(avail)]

    fac = extract_factors(zc, **kw)
    fit = _reconstruct_fitted(
        zc, fac, kw["model"],
        lam=kw.get("lam"), lam1=kw.get("lam1"), lam2=kw.get("lam2"),
    )
    resid_full = zc.loc[fit.index].sub(fit)
    resid      = resid_full[list(avail)]
    return resid, zc.loc[fit.index, list(avail)]


def _market_results(
    market  : str,
    lambdas : dict,
    mu_kr   : float,
) -> dict[str, float]:
    """OOS factor-neutral Sharpe for all 5 models for one market."""
    raw = fetch_ois_curves(market)
    zc  = build_zero_curve(raw)

    avail   = np.array([t for t in _long_end_tenors(market) if t in zc.columns])
    ns_lam  = lambdas["ns"][market]
    nss_l1, nss_l2 = lambdas["nss"][market]
    ns_load = _ns_loadings(avail, ns_lam)

    specs = [
        ("ns_cal",   "ns",  dict(model="ns_calibrated",  lam=ns_lam)),
        ("ns_free",  "ns",  dict(model="ns_free")),
        ("nss_cal",  "nss", dict(model="nss_calibrated", lam1=nss_l1, lam2=nss_l2)),
        ("nss_free", "nss", dict(model="nss_free")),
        ("kr",       "kr",  {}),
    ]

    results: dict[str, float] = {}
    for label, _, kw in specs:
        resid, zc_obs = _compute_resid(zc, avail, label, kw, mu_kr)
        r_arr  = resid.values.astype(float)
        z_arr  = zc_obs.values.astype(float)
        pnl    = _fn_pnl(r_arr, z_arr, avail, ns_load)
        oos_i  = resid.index.searchsorted(OOS_START)
        results[label] = _sharpe(pnl, oos_i)

    return results


# ── μ calibration ──────────────────────────────────────────────────────────────

def _tune_mu(market: str, lambdas: dict) -> float:
    """Pick μ maximising factor-neutral Sharpe on train period (≤2018)."""
    raw = fetch_ois_curves(market, end=TRAIN_END)
    zc  = build_zero_curve(raw)

    avail   = np.array([t for t in _long_end_tenors(market) if t in zc.columns])
    ns_lam  = lambdas["ns"][market]
    ns_load = _ns_loadings(avail, ns_lam)

    best_mu, best_sh = MU_GRID[0], -np.inf
    for mu in MU_GRID:
        resid, _ = extract_residuals_kr(zc, mu=mu, eval_tenors=avail)
        r_arr  = resid.values.astype(float)
        z_arr  = zc[list(avail)].loc[resid.index].values.astype(float)
        pnl    = _fn_pnl(r_arr, z_arr, avail, ns_load)
        sh     = _sharpe(pnl, 0)   # full train-period Sharpe
        if sh > best_sh:
            best_mu, best_sh = mu, sh

    return best_mu


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Calibrating NS/NSS lambdas on train (≤2018)...", flush=True)
    lambdas = calibrate_lambda(end=TRAIN_END)
    for mkt in OIS_MARKETS:
        ns_l  = lambdas["ns"][mkt]
        l1,l2 = lambdas["nss"][mkt]
        print(f"  {mkt}: ns_λ={ns_l:.3f}  nss_λ₁={l1:.3f} λ₂={l2:.3f}", flush=True)

    print("\nTuning KRR μ on train period...", flush=True)
    mu_map: dict[str, float] = {}
    for mkt in OIS_MARKETS:
        mu_map[mkt] = _tune_mu(mkt, lambdas)
        print(f"  {mkt}: μ={mu_map[mkt]}", flush=True)

    print("\nRunning OOS (≥2019-01-01)...", flush=True)
    rows: dict[str, dict] = {}
    for mkt in OIS_MARKETS:
        print(f"  {mkt}...", flush=True)
        rows[mkt] = _market_results(mkt, lambdas, mu_map[mkt])

    df = pd.DataFrame(rows).T[["ns_cal", "ns_free", "nss_cal", "nss_free", "kr"]]
    df["mean"] = df.mean(axis=1)
    df = df.sort_values("mean", ascending=False)

    print(f"\nOOS Sharpe ({OOS_START}→today) — factor-neutral, observed tenors\n")
    pd.set_option("display.float_format", "{:.2f}".format)
    pd.set_option("display.width", 120)
    print(df.to_string())


if __name__ == "__main__":
    main()
