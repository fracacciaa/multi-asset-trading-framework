"""
Nelson-Siegel fit for the classic-NS signal.

Three design decisions:

  1.  λ fixed a priori, not optimised per date.
      With λ fixed, β is linear OLS — deterministic and fast.  Per-date
      λ optimisation adds noise, makes the residual series non-stationary
      across λ-regime shifts, and re-introduces look-ahead when you pick λ
      by overall fit quality.

  2.  DV01 weighting (not equal weighting).
      Minimises risk-weighted error: each tenor's fit penalty is proportional
      to its actual dollar sensitivity.  Equal weighting lets the 30Y point
      dominate in yield space while contributing little to a tradeable position
      and leaves the front end systematically mispriced.
      DV01_i = t_i × exp(−z_i/100 × t_i)  (zero-coupon sensitivity, bps)

  3.  Sub-1Y excluded from the fit by default.
      NS has three factors and cannot represent the kinked front end that
      arises during policy turning points (priced FOMC/MPC dates).  Fitting
      through it throws large residuals that reflect model misspecification,
      not genuine RV — a directional macro bet in RV clothing.
      Set min_tenor=0.0 to include the front end, or use Svensson.

Lambda selection:
  Treat λ as a small discrete hyperparameter.  For a 30Y curve, a sensible
  grid places curvature peaks near 2.5Y / 4.5Y / 7Y / 11Y:
      λ ∈ {1.4, 2.5, 4.0, 6.0}
  Select on residual half-life (mean-reversion speed), not on Sharpe.
  Half-life is a property of the residual process and far more stable
  across regimes than Sharpe.  Freeze the chosen λ; if re-estimation is
  needed, do it annually on trailing data only.

Public API
----------
  ns_basis(t, lam)                   →  (n, 3) design matrix
  ns_resid(t, z, lam, w)             →  (resids, beta)
  half_life(resid)                   →  float (trading days)
  fit_ns_panel(zeros, lam, ...)      →  (betas_df, resids_df)
  select_lambda(zeros, ...)          →  dict
  beta_diagnostics(betas, zeros)     →  DataFrame
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

LAMBDA_GRID: list[float] = [1.4, 2.5, 4.0, 6.0]
DEFAULT_MIN_TENOR: float = 1.0
MIN_FIT_TENORS:    int   = 5


# ── Loading matrix ─────────────────────────────────────────────────────────────

def ns_basis(t: np.ndarray, lam: float) -> np.ndarray:
    """
    (n, 3) NS design matrix [level, slope, curvature].

    λ in years; the curvature loading (1−e^{−x})/x − e^{−x} peaks near t ≈ 1.79λ.
    Small-x limit (1−e^{−x})/x → 1 is handled by clipping x from below.
    """
    t = np.asarray(t, dtype=float)
    x = np.maximum(t / lam, 1e-9)
    e = np.exp(-x)
    L = (1.0 - e) / x
    C = L - e
    return np.column_stack([np.ones_like(t), L, C])


# ── DV01 weights ───────────────────────────────────────────────────────────────

def dv01_weights(t: np.ndarray, z_pct: np.ndarray) -> np.ndarray:
    """
    Zero-coupon DV01 weights: w_i = t_i × exp(−z_i/100 × t_i).

    Clipped from below at 1e-8 so the weight matrix is always positive definite.
    """
    t = np.asarray(t, dtype=float)
    z = np.asarray(z_pct, dtype=float)
    return np.maximum(t * np.exp(-z / 100.0 * t), 1e-8)


# ── Single-row fit ─────────────────────────────────────────────────────────────

def ns_resid(
    t: np.ndarray,
    z: np.ndarray,
    lam: float,
    w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Weighted OLS NS fit for one date.

    Solves: min_β Σ_i w_i × (z_i − X_i β)²
    via sqrt-weighting: X_w = X × √w, z_w = z × √w, then standard OLS.

    Parameters
    ----------
    t   : (n,) tenors in years (already filtered to fit range)
    z   : (n,) zero rates in percent
    lam : fixed NS lambda (years)
    w   : (n,) non-negative observation weights

    Returns
    -------
    resid : (n,) signed residuals z − X β  (percent, observed minus fitted)
    beta  : (3,) [β₀ level, β₁ slope, β₂ curvature]
    """
    X  = ns_basis(t, lam)
    sw = np.sqrt(np.asarray(w, dtype=float))
    beta, *_ = np.linalg.lstsq(X * sw[:, None], z * sw, rcond=None)
    return z - X @ beta, beta


# ── Half-life estimator ────────────────────────────────────────────────────────

def half_life(resid: np.ndarray) -> float:
    """
    Half-life (trading days) of an AR(1) mean-reverting residual series.

    Estimates the mean-reversion coefficient φ from:
        Δr_t = φ × (r_{t-1} − r̄) + ε_t
    via demeaned OLS (equivalent to including a constant in the regression).

    Returns np.inf if the series is non-stationary (φ ≥ 0),
    explosive (φ < −2), or too short to estimate reliably.
    """
    r = np.asarray(resid, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 10:
        return np.inf
    x    = r[:-1]
    y    = np.diff(r)
    x_dm = x - x.mean()
    denom = np.dot(x_dm, x_dm)
    if denom < 1e-12:
        return np.inf
    phi = np.dot(x_dm, y - y.mean()) / denom
    # Require -1 < phi < 0: stable monotone decay.  phi < -1 is oscillatory
    # (|1+phi| < 1 but 1+phi < 0) so log(1+phi) is undefined.
    if -1.0 < phi < 0.0:
        return np.log(0.5) / np.log(1.0 + phi)
    return np.inf


# ── Panel fit ──────────────────────────────────────────────────────────────────

def fit_ns_panel(
    zeros: pd.DataFrame,
    lam: float,
    min_tenor: float = DEFAULT_MIN_TENOR,
    weight_dv01: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run NS fit on every row of a zero-rate panel.

    Parameters
    ----------
    zeros       : date × tenor DataFrame of zero rates in percent.
                  Columns are tenors in years (from bootstrap_curves).
                  NaN means the tenor was not observed on that date.
    lam         : fixed NS lambda (years)
    min_tenor   : exclude tenors strictly below this from the fit.
                  Default 1.0 — avoids policy-date front-end misfit.
    weight_dv01 : weight observations by DV01 (recommended); else equal

    Returns
    -------
    betas  : date × ['beta0', 'beta1', 'beta2'] DataFrame.
             NaN rows where fewer than MIN_FIT_TENORS valid points were available.
    resids : date × tenor DataFrame of NS residuals in percent (observed − fitted).
             NaN for tenors excluded from the fit or with missing zero rates.
             The residual grid matches the columns of `zeros`.
    """
    all_tenors = zeros.columns.to_numpy(dtype=float)
    fit_mask   = all_tenors >= min_tenor

    beta_rows  = []
    resid_rows = []

    for date, row in zeros.iterrows():
        z_all = row.to_numpy(dtype=float)
        valid = np.isfinite(z_all) & fit_mask

        if valid.sum() < MIN_FIT_TENORS:
            beta_rows.append(
                pd.Series({'beta0': np.nan, 'beta1': np.nan, 'beta2': np.nan}, name=date)
            )
            resid_rows.append(pd.Series(np.nan, index=all_tenors, name=date))
            continue

        t_fit = all_tenors[valid]
        z_fit = z_all[valid]
        w     = dv01_weights(t_fit, z_fit) if weight_dv01 else np.ones(len(t_fit))

        r, beta = ns_resid(t_fit, z_fit, lam, w)

        r_full       = np.full(len(all_tenors), np.nan)
        r_full[valid] = r

        beta_rows.append(
            pd.Series({'beta0': beta[0], 'beta1': beta[1], 'beta2': beta[2]}, name=date)
        )
        resid_rows.append(pd.Series(r_full, index=all_tenors, name=date))

    betas  = pd.DataFrame(beta_rows);  betas.index.name  = "date"
    resids = pd.DataFrame(resid_rows); resids.index.name = "date"
    return betas, resids


# ── Lambda selection ───────────────────────────────────────────────────────────

def select_lambda(
    zeros: pd.DataFrame,
    lambdas: list[float] = LAMBDA_GRID,
    min_tenor: float = DEFAULT_MIN_TENOR,
    end: Optional[str] = None,
    weight_dv01: bool = True,
) -> dict:
    """
    Select λ from a discrete grid by minimising average residual half-life.

    Shorter half-life = faster mean reversion = signal is usable within the
    intended holding period.  This criterion is a property of the residual
    process and far more stable across regimes than Sharpe-based selection.

    Parameters
    ----------
    zeros     : date × tenor zero-rate panel (from bootstrap_curves)
    lambdas   : candidate λ values in years.  Default grid places curvature
                peaks near 2.5Y / 4.5Y / 7.2Y / 10.8Y.
    min_tenor : passed to fit_ns_panel; tenors below this are excluded
    end       : use only dates ≤ end for selection (training window).
                None = use full history.  Always freeze λ OOS.
    weight_dv01 : passed to fit_ns_panel

    Returns
    -------
    dict with:
      'best_lambda'  : float — λ with shortest mean finite half-life
      'half_lives'   : {lam: mean_hl_days} — scalar per candidate
      'per_tenor_hl' : {lam: pd.Series(tenor → hl_days)}
      'table'        : DataFrame sorted by mean_half_life_days (diagnostic view)
    """
    train = zeros.loc[:end] if end is not None else zeros
    all_cols = zeros.columns.to_numpy(dtype=float)
    fit_cols  = all_cols[all_cols >= min_tenor]

    results: dict[float, dict] = {}
    for lam in lambdas:
        _, resids = fit_ns_panel(train, lam, min_tenor=min_tenor, weight_dv01=weight_dv01)

        hl_map: dict[float, float] = {}
        for t in fit_cols:
            if t in resids.columns:
                hl_map[t] = half_life(resids[t].to_numpy())

        finite = [v for v in hl_map.values() if np.isfinite(v)]
        results[lam] = {
            'mean_hl':  float(np.mean(finite)) if finite else np.inf,
            'per_tenor': hl_map,
        }

    best_lam = min(lambdas, key=lambda l: results[l]['mean_hl'])

    table = pd.DataFrame([
        {'lambda': lam, 'mean_half_life_days': results[lam]['mean_hl'],
         'curvature_peak_yr': round(1.793 * lam, 1)}
        for lam in lambdas
    ]).sort_values('mean_half_life_days').reset_index(drop=True)

    return {
        'best_lambda':  best_lam,
        'half_lives':   {lam: results[lam]['mean_hl'] for lam in lambdas},
        'per_tenor_hl': {lam: pd.Series(results[lam]['per_tenor']) for lam in lambdas},
        'table':        table,
    }


# ── Beta diagnostics ───────────────────────────────────────────────────────────

def beta_diagnostics(
    betas: pd.DataFrame,
    zeros: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sanity-check that NS factors are well-behaved for the chosen λ.

    β₀ should track the long-end zero rate; β₀ + β₁ should track the
    short-end zero rate; both should be smooth (small daily changes relative
    to level).  If not, λ is badly scaled for the tenor range.

    Returns a one-row DataFrame with:
      corr_beta0_longrate    : ρ(β₀, longest available zero rate)
      corr_b0b1_shortrate    : ρ(β₀+β₁, shortest available zero rate ≥ 1Y)
      beta0_daily_std_bps    : σ(Δβ₀) in basis points (smoothness of level)
      beta1_daily_std_bps    : σ(Δβ₁) in basis points (smoothness of slope)
      beta2_daily_std_bps    : σ(Δβ₂) in basis points (smoothness of curvature)
      pct_dates_valid        : % of dates where all three betas are finite
    """
    b0 = betas['beta0'].dropna()
    b1 = betas['beta1'].dropna()
    b2 = betas['beta2'].dropna()

    tenors = zeros.columns.to_numpy(dtype=float)
    long_t = tenors[-1]
    # Shortest zero-curve tenor ≥ 1Y for the short-rate proxy
    ge1    = tenors[tenors >= 1.0]
    short_t = ge1[0] if len(ge1) > 0 else tenors[0]

    def _corr(s1: pd.Series, s2: pd.Series) -> float:
        common = s1.index.intersection(s2.index)
        if len(common) < 20:
            return np.nan
        return float(np.corrcoef(s1.loc[common].values, s2.loc[common].values)[0, 1])

    long_rate  = zeros[long_t].dropna()
    short_rate = zeros[short_t].dropna()
    b0b1 = (betas['beta0'] + betas['beta1']).dropna()

    n_valid = betas[['beta0', 'beta1', 'beta2']].dropna().shape[0]

    return pd.DataFrame({
        'corr_beta0_longrate':   [_corr(b0, long_rate)],
        'corr_b0b1_shortrate':   [_corr(b0b1, short_rate)],
        'beta0_daily_std_bps':   [round(float(b0.diff().std() * 100), 2)],
        'beta1_daily_std_bps':   [round(float(b1.diff().std() * 100), 2)],
        'beta2_daily_std_bps':   [round(float(b2.diff().std() * 100), 2)],
        'pct_dates_valid':       [round(100.0 * n_valid / max(len(betas), 1), 1)],
    })
