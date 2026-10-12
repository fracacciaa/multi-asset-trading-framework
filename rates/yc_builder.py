"""
Yield Curve Builder — bootstrap + interpolation pipeline.

Three-step process per market:
  1. Data      : fetch_ois_curves() from ois_data.py
  2. Bootstrap : OIS par rates → continuously compounded zero rates
                 at observed tenors (1M, 2M, ..., 1Y, 2Y, ..., 30Y)
  3. Interpolate: zero rates at observed tenors → dense target grid

Target grid:
  - Monthly  : 1/12, 2/12, ..., 12/12  (preserves all 1M–12M OIS observations)
  - Semi-annual: 1.5, 2.0, ..., 30.0   (matches OIS data frequency post-1Y)
  Total: 70 tenors.

Three interpolation methods; use benchmark_interpolation() to compare.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from typing import Optional

from rates.ois_data import fetch_ois_curves, OIS_MARKETS, OIS_COUPON_FREQ

# ── Target tenor grid ─────────────────────────────────────────────────────────

_short = np.array([m / 12.0 for m in range(1, 13)])  # 1M … 12M
_long  = np.arange(1.5, 30.5, 0.5)                    # 1.5Y … 30.0Y
TARGET_TENORS: np.ndarray = np.concatenate([_short, _long])  # 70 points


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2 — Bootstrap (par OIS rates → zero coupon rates at observed tenors)
# ═══════════════════════════════════════════════════════════════════════════════

def _loglinear_discount(
    knot_t: np.ndarray, knot_d: np.ndarray, t: float
) -> float:
    """Log-linear interpolation / flat-forward extrapolation of discount factors."""
    if t <= knot_t[0]:
        return knot_d[0]
    if t >= knot_t[-1]:
        if len(knot_t) < 2:
            return knot_d[-1]
        log_d = (np.log(knot_d[-1])
                 + (np.log(knot_d[-1]) - np.log(knot_d[-2]))
                 / (knot_t[-1] - knot_t[-2])
                 * (t - knot_t[-1]))
        return np.exp(log_d)
    i = np.searchsorted(knot_t, t, side="right") - 1
    frac = (t - knot_t[i]) / (knot_t[i + 1] - knot_t[i])
    log_d = (1 - frac) * np.log(knot_d[i]) + frac * np.log(knot_d[i + 1])
    return np.exp(log_d)


def bootstrap(
    tenors: np.ndarray,
    par_yields: np.ndarray,
    coupon_freq: int = OIS_COUPON_FREQ,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bootstrap par yields to continuously compounded zero rates.

    Parameters
    ----------
    tenors      : maturities in years, strictly increasing
    par_yields  : annual par yields in percent (5.0 = 5%)
    coupon_freq : coupon payments per year (1 = annual for all OIS markets)

    Returns
    -------
    (tenors, zero_rates) — same grid, zero rates in percent (c.c.)
    """
    t = np.asarray(tenors, dtype=float)
    r = np.asarray(par_yields, dtype=float)

    dt            = 1.0 / coupon_freq
    zero_cpn_max  = dt          # tenors ≤ dt treated as zero coupon

    knot_t = np.array([0.0])
    knot_d = np.array([1.0])
    zero_rates = np.empty(len(t))

    for i in range(len(t)):
        tau  = t[i]
        r_pa = r[i] / 100.0
        c    = r_pa / coupon_freq

        if tau <= zero_cpn_max:
            d_tau = np.exp(-r_pa * tau)
        else:
            n_periods    = int(round(tau * coupon_freq))
            coupon_dates = np.array([j * dt for j in range(1, n_periods + 1)])
            coupon_sum   = sum(
                _loglinear_discount(knot_t, knot_d, tc)
                for tc in coupon_dates[:-1]
            )
            d_tau = (1.0 - c * coupon_sum) / (1.0 + c)

        d_tau          = max(d_tau, 1e-6)
        zero_rates[i]  = -np.log(d_tau) / tau * 100.0
        knot_t         = np.append(knot_t, tau)
        knot_d         = np.append(knot_d, d_tau)

    return t, zero_rates


def _bootstrap_row(
    row: pd.Series,
    coupon_freq: int = OIS_COUPON_FREQ,
    min_tenors: int = 6,
) -> Optional[pd.Series]:
    valid = row.dropna().sort_index()
    if len(valid) < min_tenors:
        return None
    t_arr = valid.index.to_numpy(dtype=float)
    r_arr = valid.to_numpy(dtype=float)
    _, z = bootstrap(t_arr, r_arr, coupon_freq=coupon_freq)
    return pd.Series(z, index=t_arr, name=row.name)


def bootstrap_curves(
    curves: pd.DataFrame,
    coupon_freq: int = OIS_COUPON_FREQ,
    min_tenors: int = 6,
) -> pd.DataFrame:
    """
    Bootstrap every row of a date × tenor par yield DataFrame.

    Returns date × tenor DataFrame of continuously compounded zero rates (%).
    Rows with insufficient tenors are returned as NaN.
    """
    rows = []
    for date, row in curves.iterrows():
        s = _bootstrap_row(row, coupon_freq=coupon_freq, min_tenors=min_tenors)
        rows.append(s.rename(date) if s is not None
                    else pd.Series(np.nan, index=curves.columns, name=date))
    result = pd.DataFrame(rows)
    result.index.name = "date"
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3 — Interpolation (zero rates at observed tenors → TARGET_TENORS grid)
# All methods operate on log-discount factors A(t) = −r(t)·t / 100.
# Converting to A(t) before interpolating makes non-negative forward rates natural.
# ═══════════════════════════════════════════════════════════════════════════════

def _monotone_preserving(
    tenors: np.ndarray, yields: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """
    Cubic Hermite on A(t) = −r(t)·t with Fritsch-Carlson monotonicity clamping.
    Guarantees: non-negative forward rates + continuous forward curve.
    Reference: Du Preez & Maré (2013).
    """
    t = np.asarray(tenors, dtype=float)
    y = np.asarray(yields, dtype=float)
    A = -y / 100.0 * t
    n = len(t)

    # Instantaneous forward estimates at knots
    f = np.empty(n)
    if n == 1:
        f[0] = y[0] / 100.0
    elif n == 2:
        f[0] = f[1] = (A[1] - A[0]) / (t[1] - t[0])
    else:
        for i in range(1, n - 1):
            h_l = t[i] - t[i - 1]
            h_r = t[i + 1] - t[i]
            s_l = (A[i] - A[i - 1]) / h_l
            s_r = (A[i + 1] - A[i]) / h_r
            f[i] = (h_r * s_l + h_l * s_r) / (h_l + h_r)
        f[0]     = (A[1] - A[0]) / (t[1] - t[0])
        f[n - 1] = (A[n - 1] - A[n - 2]) / (t[n - 1] - t[n - 2])

    # Fritsch-Carlson monotonicity clamping (ensures f(t) ≥ 0)
    for i in range(n - 1):
        h     = t[i + 1] - t[i]
        delta = (A[i + 1] - A[i]) / h
        if abs(delta) < 1e-14:
            f[i] = f[i + 1] = 0.0
            continue
        alpha = f[i]     / delta
        beta  = f[i + 1] / delta
        if alpha < 0.0:
            f[i] = 0.0
            alpha = 0.0
        if beta < 0.0:
            f[i + 1] = 0.0
            beta = 0.0
        norm2 = alpha ** 2 + beta ** 2
        if norm2 > 9.0:
            tau          = 3.0 / np.sqrt(norm2)
            f[i]         = tau * alpha * delta
            f[i + 1]     = tau * beta  * delta

    def _A(t_star: float) -> float:
        if t_star <= t[0]:
            return A[0] + f[0] * (t_star - t[0])
        if t_star >= t[-1]:
            return A[-1] + f[-1] * (t_star - t[-1])
        idx = min(np.searchsorted(t, t_star, side="right") - 1, n - 2)
        h   = t[idx + 1] - t[idx]
        s   = (t_star - t[idx]) / h
        h00 = (1 - s) ** 2 * (1 + 2 * s)
        h10 = s * (1 - s) ** 2
        h01 = s ** 2 * (3 - 2 * s)
        h11 = s ** 2 * (s - 1)
        return A[idx] * h00 + f[idx] * h * h10 + A[idx + 1] * h01 + f[idx + 1] * h * h11

    out = np.empty(len(target))
    for k, t_star in enumerate(target):
        if t_star <= 0:
            out[k] = y[0]
        else:
            out[k] = -_A(t_star) / t_star * 100.0
    return out


def _monotone_convex(
    tenors: np.ndarray, yields: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """
    Quadratic forward-rate interpolation with 8-case monotone fix.
    Guarantees: non-negative + monotone forward rates.
    Reference: Hagan & West (2006).
    Uses 5-point Gauss-Legendre quadrature (exact for piecewise-quadratic forwards).
    """
    t = np.asarray(tenors, dtype=float)
    y = np.asarray(yields, dtype=float)
    A = -y / 100.0 * t
    n = len(t)

    t_ext = np.concatenate([[0.0], t])
    A_ext = np.concatenate([[0.0], A])

    # Discrete interval-average forward rates
    f_d = np.array([
        -(A_ext[i + 1] - A_ext[i]) / (t_ext[i + 1] - t_ext[i])
        for i in range(n)
    ])

    # Point forward rates at knots
    f_pt = np.empty(n + 1)
    for i in range(1, n):
        h_l    = t_ext[i]     - t_ext[i - 1]
        h_r    = t_ext[i + 1] - t_ext[i]
        f_pt[i] = (h_r * f_d[i - 1] + h_l * f_d[i]) / (h_l + h_r)
    f_pt[0] = f_d[0] - 0.5 * (f_pt[1]     - f_d[0])
    f_pt[n] = f_d[-1] - 0.5 * (f_pt[n - 1] - f_d[-1])

    # Clamp to enforce non-negativity
    f_pt[0] = min(max(0.0, f_pt[0]), 2.0 * f_d[0])
    for i in range(1, n):
        f_pt[i] = min(max(0.0, f_pt[i]), 2.0 * min(f_d[i - 1], f_d[i]))
    f_pt[n] = min(max(0.0, f_pt[n]), 2.0 * f_d[-1])

    def _fwd(t_star: float) -> float:
        if t_star <= t_ext[0]:
            return f_d[0]
        if t_star >= t_ext[-1]:
            return f_d[-1]
        i  = min(np.searchsorted(t_ext, t_star, side="right") - 1, n - 1)
        h  = t_ext[i + 1] - t_ext[i]
        x  = (t_star - t_ext[i]) / h
        g0 = f_pt[i]     - f_d[i]
        g1 = f_pt[i + 1] - f_d[i]
        if g0 == 0.0 and g1 == 0.0:
            g = 0.0
        elif g0 >= 0.0 and g1 >= 0.0:
            phi = g1 / (g0 + g1) if (g0 + g1) else 0.5
            A_  = -0.5 * (phi * g0 + (1 - phi) * g1)
            g   = (A_ + (g0 - A_) * ((phi - x) / phi) ** 2 if x <= phi and phi > 0
                   else A_ + (g1 - A_) * ((x - phi) / (1 - phi)) ** 2 if phi < 1
                   else g1)
        elif g0 <= 0.0 and g1 <= 0.0:
            phi = g1 / (g0 + g1) if (g0 + g1) else 0.5
            A_  = -0.5 * (phi * g0 + (1 - phi) * g1)
            g   = (A_ + (g0 - A_) * ((phi - x) / phi) ** 2 if x <= phi and phi > 0
                   else A_ + (g1 - A_) * ((x - phi) / (1 - phi)) ** 2 if phi < 1
                   else g1)
        elif g0 >= 0.0 and g1 < 0.0:
            phi = max(0.0, min(1.0, 1.0 + 3.0 * g0 / (g1 - g0) if g1 != g0 else 1.0))
            g   = g0 if x <= phi else (g0 + (g1 - g0) * (x - phi) / (1 - phi) if phi < 1 else g1)
        else:
            phi = max(0.0, min(1.0, -3.0 * g0 / (g1 - g0) if g1 != g0 else 0.0))
            g   = g1 if x >= phi else (g0 + (g1 - g0) * (x - phi) / (1 - phi) if phi < 1 else g1)
        return f_d[i] + g

    # 5-point Gauss-Legendre quadrature weights and nodes
    _gl_x = np.array([0.0, 0.5384693101056831, -0.5384693101056831,
                       0.9061798459386640, -0.9061798459386640])
    _gl_w = np.array([0.5688888888888889, 0.4786286704993665,  0.4786286704993665,
                       0.2369268850561891,  0.2369268850561891])

    def _integrate(a: float, b: float) -> float:
        mid  = 0.5 * (a + b)
        half = 0.5 * (b - a)
        return half * float(np.dot(_gl_w, [_fwd(mid + half * xi) for xi in _gl_x]))

    A_knot = np.zeros(n + 1)
    for i in range(n):
        A_knot[i + 1] = A_knot[i] - _integrate(t_ext[i], t_ext[i + 1])

    def _A(t_star: float) -> float:
        if t_star <= t_ext[0]:
            return A_knot[0] - f_d[0] * (t_star - t_ext[0])
        if t_star >= t_ext[-1]:
            return A_knot[n] - f_d[-1] * (t_star - t_ext[-1])
        i = min(np.searchsorted(t_ext, t_star, side="right") - 1, n - 1)
        return A_knot[i] - _integrate(t_ext[i], t_star)

    out = np.empty(len(target))
    for k, t_star in enumerate(target):
        if t_star <= 0:
            out[k] = y[0]
        else:
            out[k] = -_A(t_star) / t_star * 100.0
    return out


def _natural_cubic_spline(
    tenors: np.ndarray, yields: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """
    Natural cubic spline on A(t) = −r(t)·t / 100.

    'Natural' boundary: d²A/dt² = 0 at both endpoints, which implies linear
    A(t) extrapolation beyond the last knot → constant forward rate (well-behaved).
    Does NOT enforce forward rate positivity; benchmark will quantify the impact.
    """
    t = np.asarray(tenors, dtype=float)
    y = np.asarray(yields, dtype=float)

    # Include origin: A(0) = 0  (P(0) = 1)
    t_ext = np.concatenate([[0.0], t])
    A_ext = np.concatenate([[0.0], -y / 100.0 * t])

    cs = CubicSpline(t_ext, A_ext, bc_type="natural")

    out = np.empty(len(target))
    for k, t_star in enumerate(target):
        if t_star <= 0:
            out[k] = y[0]
        else:
            out[k] = -float(cs(t_star)) / t_star * 100.0
    return out


# ── Interpolation dispatcher ──────────────────────────────────────────────────

INTERP_METHODS: dict[str, callable] = {
    "monotone_preserving":  _monotone_preserving,
    "monotone_convex":      _monotone_convex,
    "natural_cubic_spline": _natural_cubic_spline,
}


def _interpolate_row(
    row: pd.Series,
    method: str,
    target: np.ndarray,
    min_tenors: int,
) -> Optional[pd.Series]:
    valid = row.dropna().sort_index()
    if len(valid) < min_tenors:
        return None
    out = INTERP_METHODS[method](
        valid.index.to_numpy(dtype=float),
        valid.to_numpy(dtype=float),
        target,
    )
    return pd.Series(out, index=target, name=row.name)


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def build_zero_curve(
    raw_curves: pd.DataFrame,
    method: str = "monotone_preserving",
    coupon_freq: int = OIS_COUPON_FREQ,
    min_tenors: int = 6,
    target: np.ndarray = TARGET_TENORS,
) -> pd.DataFrame:
    """
    Single-market pipeline: raw OIS par rates → zero coupon curve.

    Parameters
    ----------
    raw_curves  : date × tenor (years) DataFrame from fetch_ois_curves()
    method      : 'monotone_preserving' | 'monotone_convex' | 'natural_cubic_spline'
    coupon_freq : 1 for all OIS markets
    min_tenors  : minimum observed tenors required per row
    target      : output tenor grid (default: TARGET_TENORS, 70 points)

    Returns
    -------
    date × tenor DataFrame of continuously compounded zero rates (%).
    Columns match `target` (years).
    """
    if method not in INTERP_METHODS:
        raise ValueError(f"Unknown method {method!r}. Choose from {list(INTERP_METHODS)}")

    zeros = bootstrap_curves(raw_curves, coupon_freq=coupon_freq, min_tenors=min_tenors)

    rows = []
    for date, row in zeros.iterrows():
        s = _interpolate_row(row, method=method, target=target, min_tenors=min_tenors)
        rows.append(
            s.rename(date) if s is not None
            else pd.Series(np.nan, index=target, name=date)
        )

    result = pd.DataFrame(rows)
    result.index.name = "date"
    return result


def build_zero_curves_all_markets(
    markets: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    method: str = "monotone_preserving",
) -> pd.DataFrame:
    """
    Build zero coupon curves for all (or selected) OIS markets.

    Returns date × (market, tenor) MultiIndex column DataFrame.
    Each market's curve spans TARGET_TENORS (70 tenors, 1M → 30Y).
    """
    if markets is None:
        markets = OIS_MARKETS

    frames = {}
    for market in markets:
        raw = fetch_ois_curves(market, start=start, end=end)
        frames[market] = build_zero_curve(raw, method=method)

    result = pd.concat(frames, axis=1)
    result.columns.names = ["market", "tenor"]
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmarking
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_interpolation(
    markets: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    methods: Optional[list[str]] = None,
    subsample: int = 20,
) -> pd.DataFrame:
    """
    Leave-one-out cross-validation across all markets and dates.

    For each market × sampled date, holds out each interior observed tenor
    (excluding the first and last as anchors), re-interpolates from the
    remaining zero rates, and measures the error at the held-out point.

    Three metrics per method × market:
      loo_rmse_bps      : RMSE of LOO errors in basis points
      fwd_neg_pct       : % of dates where any forward rate is negative
      fwd_oscillation   : mean |Δ²f / Δt| — forward rate smoothness

    Parameters
    ----------
    subsample : use every Nth date to limit runtime (20 → ~5% of dates)

    Returns
    -------
    DataFrame with per-market rows plus an '*** OVERALL ***' summary per method.
    """
    if markets is None:
        markets = OIS_MARKETS
    if methods is None:
        methods = list(INTERP_METHODS)

    records = []

    for market in markets:
        raw   = fetch_ois_curves(market, start=start, end=end)
        zeros = bootstrap_curves(raw, coupon_freq=OIS_COUPON_FREQ, min_tenors=6)
        dates = zeros.dropna(how="all").index[::subsample]

        for method in methods:
            fn = INTERP_METHODS[method]
            sq_errors:  list[float] = []
            neg_days:   int         = 0
            total_days: int         = 0
            osc_vals:   list[float] = []

            for date in dates:
                row   = zeros.loc[date].dropna().sort_index()
                t_obs = row.index.to_numpy(dtype=float)
                z_obs = row.to_numpy(dtype=float)
                if len(t_obs) < 5:
                    continue

                # ── LOO-CV: hold out each interior tenor ──────────────────────
                for i in range(1, len(t_obs) - 1):
                    t_held = t_obs[i]
                    z_held = z_obs[i]
                    t_red  = np.delete(t_obs, i)
                    z_red  = np.delete(z_obs, i)
                    try:
                        z_hat = fn(t_red, z_red, np.array([t_held]))
                        sq_errors.append((z_hat[0] - z_held) ** 2)
                    except Exception:
                        pass

                # ── Forward rate quality ──────────────────────────────────────
                try:
                    dense = fn(t_obs, z_obs, TARGET_TENORS)
                    A_dense = -dense / 100.0 * TARGET_TENORS
                    fwd  = -np.diff(A_dense) / np.diff(TARGET_TENORS)  # approx f(t)
                    neg_days   += int(np.any(fwd < -1e-6))
                    total_days += 1
                    # Forward rate smoothness: mean |d²f/dt²|
                    osc = np.mean(np.abs(np.diff(fwd) / np.diff(TARGET_TENORS[1:])))
                    osc_vals.append(osc)
                except Exception:
                    total_days += 1

            records.append({
                "method":          method,
                "market":          market,
                "loo_rmse_bps":    round(np.sqrt(np.mean(sq_errors)) * 100, 4)
                                   if sq_errors else float("nan"),
                "fwd_neg_pct":     round(100 * neg_days / total_days, 2)
                                   if total_days else float("nan"),
                "fwd_oscillation": round(float(np.mean(osc_vals)), 6)
                                   if osc_vals else float("nan"),
            })

    detail = pd.DataFrame(records)

    # Overall summary: mean across markets per method
    summary = (
        detail.groupby("method")[["loo_rmse_bps", "fwd_neg_pct", "fwd_oscillation"]]
        .mean()
        .round(4)
        .reset_index()
    )
    summary.insert(1, "market", "*** OVERALL ***")

    return (
        pd.concat([detail, summary], ignore_index=True)
        .sort_values(["method", "market"])
        .reset_index(drop=True)
    )
