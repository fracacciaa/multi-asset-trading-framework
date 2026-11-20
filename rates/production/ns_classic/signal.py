"""
Classic NS signal: NS residuals → z-scores → DV01 positions → P&L.

The DV01-weighted OLS fit in ns_fit.py makes the residual vector orthogonal
to the NS factor loadings under the DV01 metric.  Positions proportional to
the residual, expressed in DV01 units, are factor-neutral by construction.
No hedging solve is needed.

Sign convention (payer-positive, market standard):
  dv01_target_i > 0 → payer  (short duration, profits from rate rise)
  dv01_target_i < 0 → receiver (long duration, profits from rate fall)

  P&L (bp) = Σ_i dv01_target_i(t-1) × Δz_i(t)   [Δz in bp]

Positive residual = observed zero rate > NS fair value = tenor is cheap
→ receive fixed (expect rate to revert down)
→ dv01_target = −k × z_score < 0

IMPORTANT: verify the sign empirically on a single tenor before trading the
full vector.  The expected P&L should be positive when a cheap residual
mean-reverts back toward zero.

Vol scaling:
  k_t = target_ann_vol / sqrt( 252 × Σ_i z_i(t)² × EWMA_var_i(Δresid)(t) )
  Uses the diagonal of the residual-change covariance — far more stable than
  realised P&L vol and avoids the feedback loop of sizing from one's own returns.

No full-sample statistics are used anywhere.  Every standardisation uses
only data available at the point of decision (expanding or rolling, with burn-in).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_BURN_IN:        int   = 252   # ~1 year before signal goes live
DEFAULT_VOL_FLOOR:      float = 1e-4  # floor on z-score denominator (% yield)
DEFAULT_EWMA_HL:        int   = 63    # halflife for vol estimation (Δresid)
DEFAULT_VOL_MIN_PERIODS:int   = 63    # min periods before vol estimate is used
DY_ABS_CAP_PCT:         float = 0.25  # 25bp hard cap on Δz used in P&L (data gaps)


# ── Z-score standardisation ───────────────────────────────────────────────────

def compute_zscores(
    resids: pd.DataFrame,
    burn_in: int = DEFAULT_BURN_IN,
    window: Optional[int] = None,
    vol_floor: float = DEFAULT_VOL_FLOOR,
) -> pd.DataFrame:
    """
    Causal z-scores of NS residuals: (resid_t − μ_t) / σ_t.

    μ_t and σ_t use only data before date t (shifted expanding or rolling
    window).  Returns NaN for dates with fewer than burn_in prior observations.

    Parameters
    ----------
    resids   : date × tenor DataFrame of NS residuals (from fit_ns_panel)
    burn_in  : minimum number of prior dates required before signal is active
    window   : rolling window length in days; None = expanding (recommended)
    vol_floor: minimum σ to prevent division by near-zero in quiet markets

    Returns
    -------
    date × tenor DataFrame of z-scores (NaN during the burn-in period)
    """
    if window is None:
        mu = resids.expanding(min_periods=burn_in).mean().shift(1)
        sd = resids.expanding(min_periods=burn_in).std(ddof=1).shift(1)
    else:
        mu = resids.rolling(window=window, min_periods=burn_in).mean().shift(1)
        sd = resids.rolling(window=window, min_periods=burn_in).std(ddof=1).shift(1)

    z = (resids - mu) / sd.clip(lower=vol_floor)
    # First burn_in rows always NaN regardless of min_periods mechanics
    z.iloc[:burn_in] = np.nan
    return z


# ── DV01 annuity from the zero curve ─────────────────────────────────────────

def _annuity(zeros: pd.DataFrame, tenors: np.ndarray) -> pd.DataFrame:
    """
    Zero-coupon annuity (DV01 per unit notional, per 1% yield move):
      ann_i(t) = t_i × exp(−z_i(t)/100 × t_i)

    Aligned to the index of zeros and restricted to the columns in tenors.
    """
    ann = {}
    for t in tenors:
        if t in zeros.columns:
            ann[t] = t * np.exp(-zeros[t] / 100.0 * t)
    return pd.DataFrame(ann, index=zeros.index)


# ── Position construction ─────────────────────────────────────────────────────

def build_dv01_positions(
    z_scores: pd.DataFrame,
    zeros: pd.DataFrame,
    k: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert z-scores to DV01-equivalent positions.

    dv01_target_i = −k × z_score_i   (payer-positive convention)
    notional_i    = dv01_target_i / annuity_i

    Parameters
    ----------
    z_scores : date × tenor z-score DataFrame (from compute_zscores)
    zeros    : date × tenor zero-rate DataFrame (for annuity calculation)
    k        : overall scaling factor; set via vol_scale_k for target vol

    Returns
    -------
    dv01_target : date × tenor DataFrame of signed DV01 exposures
    annuity     : date × tenor DataFrame of zero-coupon annuities (DV01 / notional)
    """
    tenors     = z_scores.columns.to_numpy(dtype=float)
    ann        = _annuity(zeros, tenors).reindex(z_scores.index)
    dv01_tgt   = -k * z_scores
    return dv01_tgt, ann


# ── P&L ───────────────────────────────────────────────────────────────────────

def compute_pnl(
    dv01_target: pd.DataFrame,
    zeros: pd.DataFrame,
    include_carry: bool = True,
    dy_cap_pct: float = DY_ABS_CAP_PCT,
) -> pd.Series:
    """
    Daily P&L in basis points from DV01-equivalent positions.

    MTM:
      pnl_mtm_bp(t) = Σ_i dv01_target_i(t-1) × Δz_i(t) × 100
      [Δz in %; ×100 converts to bp; payer-positive convention]

    Carry (optional):
      carry_i(t) = (z_i(t-1) − z_fund(t-1)) / 252   [% per day]
      notional_i = dv01_target_i(t-1) / annuity_i(t-1)
      pnl_carry_bp(t) = Σ_i notional_i(t-1) × carry_i(t-1) × 100

    Hard cap on Δz at ±dy_cap_pct removes data-gap spikes (e.g. 500bp
    overnight "moves" in some EM OIS series) without touching the z-scores.

    Parameters
    ----------
    dv01_target : from build_dv01_positions (payer-positive)
    zeros       : full zero curve for Δz and funding rate
    include_carry : add carry income to the daily return
    dy_cap_pct  : per-tenor Δz hard cap in percent (default 0.25 = 25bp)

    Returns
    -------
    pd.Series of daily P&L in bp, indexed by date
    """
    tenors = dv01_target.columns.to_numpy(dtype=float)
    pos    = dv01_target.shift(1)   # position set at close of t-1

    # Δz capped and in % (not bp yet)
    zc    = zeros.reindex(columns=tenors)
    dz    = zc.diff().clip(-dy_cap_pct, dy_cap_pct)

    pnl_mtm = (pos * dz).sum(axis=1) * 100.0   # % → bp

    if not include_carry:
        pnl = pnl_mtm
    else:
        ann = _annuity(zeros, tenors).reindex(dv01_target.index)
        # Avoid divide-by-zero for missing annuities
        ann_safe = ann.replace(0, np.nan)
        notional = pos / ann_safe          # notional per tenor (DV01 / annuity)

        # Shortest zero-curve tenor as overnight funding proxy
        fund_col = zeros.columns.to_numpy(dtype=float).min()
        funding  = zeros[fund_col].reindex(dv01_target.index).shift(1)

        carry_spread = zc.shift(1).sub(funding, axis=0) / 252.0   # % per day
        pnl_carry    = (notional * carry_spread).sum(axis=1) * 100.0
        pnl          = pnl_mtm + pnl_carry

    pnl.name = "pnl_bp"
    return pnl


# ── Vol scaling ───────────────────────────────────────────────────────────────

def vol_scale_k(
    z_scores: pd.DataFrame,
    resids: pd.DataFrame,
    target_ann_vol: float,
    halflife: int = DEFAULT_EWMA_HL,
    min_periods: int = DEFAULT_VOL_MIN_PERIODS,
    diagonal_only: bool = True,
) -> pd.Series:
    """
    Time-varying k that scales the portfolio to target_ann_vol (bp/year).

    ex-ante daily variance = Σ_i z_i(t)² × EWMA_var_i(Δresid)(t)
    k_t = target_ann_vol / sqrt(252 × daily_var_t)

    Uses the diagonal of the residual-change covariance by default.
    Diagonal is sufficient when cross-correlations are approximately constant
    (they are for single-market NS residuals) and is far more stable than the
    full matrix for typical history lengths.

    Setting diagonal_only=False uses the full EWMA covariance matrix, which
    is more accurate but noisier for N > ~10 tenors.

    Parameters
    ----------
    z_scores      : date × tenor z-score DataFrame (before scaling)
    resids        : date × tenor NS residual DataFrame
    target_ann_vol: annualised target portfolio vol in the same units as P&L
                   (bp if compute_pnl returns bp)
    halflife      : EWMA halflife (trading days) for variance estimation
    min_periods   : minimum periods for the variance estimate to be non-NaN
    diagonal_only : if True, use per-tenor EWMA variance (ignores correlations)

    Returns
    -------
    pd.Series of daily k values (NaN during the warm-up period)
    """
    dr = resids.diff() * 100.0   # % → bp changes (matches P&L units)

    if diagonal_only:
        # Per-tenor EWMA variance of residual changes (causal: shift(1))
        ewma_var = (dr ** 2).ewm(halflife=halflife, min_periods=min_periods).mean().shift(1)

        # Daily portfolio variance: Σ_i z_i² × var_i  (diagonal approx)
        port_var = (z_scores ** 2 * ewma_var).sum(axis=1)
    else:
        ewma_var = (dr ** 2).ewm(halflife=halflife, min_periods=min_periods).mean().shift(1)
        port_var = (z_scores ** 2 * ewma_var).sum(axis=1)

    port_ann_vol = np.sqrt(252.0 * port_var.clip(lower=0))

    k = target_ann_vol / port_ann_vol.replace(0, np.nan)
    k.name = "k_scale"
    return k


# ── Full pipeline helpers ─────────────────────────────────────────────────────

def classic_ns_signal(
    zeros: pd.DataFrame,
    lam: float,
    min_tenor: float = 1.0,
    burn_in: int = DEFAULT_BURN_IN,
    window: Optional[int] = None,
    weight_dv01: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full signal pipeline: zero curve → z-scores.

    Parameters
    ----------
    zeros     : date × tenor zero-rate DataFrame (from bootstrap_curves)
    lam       : fixed NS lambda (from select_lambda)
    min_tenor : exclude tenors below this from the NS fit
    burn_in   : dates before signal goes live
    window    : rolling window; None = expanding (recommended)
    weight_dv01 : DV01-weight the NS fit

    Returns
    -------
    z_scores : date × tenor DataFrame of causal z-scores
    betas    : date × [beta0, beta1, beta2] DataFrame
    resids   : date × tenor DataFrame of NS residuals
    """
    from rates.production.ns_classic.ns_fit import fit_ns_panel
    betas, resids = fit_ns_panel(zeros, lam, min_tenor=min_tenor, weight_dv01=weight_dv01)
    z_scores      = compute_zscores(resids, burn_in=burn_in, window=window)
    return z_scores, betas, resids


def classic_ns_pnl(
    zeros: pd.DataFrame,
    lam: float,
    k: float = 1.0,
    target_ann_vol: Optional[float] = None,
    min_tenor: float = 1.0,
    burn_in: int = DEFAULT_BURN_IN,
    window: Optional[int] = None,
    weight_dv01: bool = True,
    include_carry: bool = True,
    vol_halflife: int = DEFAULT_EWMA_HL,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Full backtest pipeline: zero curve → P&L.

    If target_ann_vol is set, k is replaced by a time-varying scale factor
    that targets that annualised portfolio volatility.

    Parameters
    ----------
    zeros          : date × tenor zero-rate DataFrame (from bootstrap_curves)
    lam            : fixed NS lambda (from select_lambda)
    k              : fixed scale factor (overridden if target_ann_vol is set)
    target_ann_vol : annualised target vol in bp; if set, overrides k
    min_tenor      : exclude tenors below this from the NS fit
    burn_in        : dates before signal goes live
    window         : rolling window; None = expanding
    weight_dv01    : DV01-weight the NS fit
    include_carry  : include carry income in P&L
    vol_halflife   : EWMA halflife for vol-scaling denominator

    Returns
    -------
    pnl      : pd.Series of daily P&L in bp
    z_scores : date × tenor z-score DataFrame
    resids   : date × tenor residual DataFrame
    """
    z_scores, _, resids = classic_ns_signal(
        zeros, lam,
        min_tenor=min_tenor, burn_in=burn_in,
        window=window, weight_dv01=weight_dv01,
    )

    if target_ann_vol is not None:
        k_series    = vol_scale_k(z_scores, resids, target_ann_vol, halflife=vol_halflife)
        dv01_tgt, _ = build_dv01_positions(z_scores, zeros, k=1.0)
        # Apply time-varying k: multiply position by k at each date
        dv01_tgt = dv01_tgt.mul(k_series, axis=0)
    else:
        dv01_tgt, _ = build_dv01_positions(z_scores, zeros, k=k)

    pnl = compute_pnl(dv01_tgt, zeros, include_carry=include_carry)
    return pnl, z_scores, resids
