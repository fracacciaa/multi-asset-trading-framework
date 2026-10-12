"""
Yield curve interpolation methods.

Two approaches from the literature, both operating on continuously compounded
spot rates converted to/from log discount factors A(t) = ln(P(t)) = -r(t)·t:

  1. Monotone Preserving (Du Preez & Maré 2013)
     Cubic Hermite on A(t) with Fritsch-Carlson clamping.
     Guarantees: positive forward rates + continuous forward curve.

  2. Monotone Convex (Hagan & West 2006)
     Quadratic interpolation on discrete forward rates with 8-case monotone fix.
     Guarantees: positive forward rates + monotone forward curve.

Both return yields at any target set of maturities. Extrapolation is flat
(constant rate beyond the last knot).
"""

import numpy as np
import pandas as pd
from typing import Optional


# ── Target grid ───────────────────────────────────────────────────────────────

TARGET_TENORS = np.arange(0.5, 30.5, 0.5)   # 0.5, 1.0, 1.5, …, 30.0


# ── Method 1: Monotone Preserving ─────────────────────────────────────────────

def monotone_preserving(
    tenors: np.ndarray,
    yields: np.ndarray,
    target: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Interpolate yields via cubic Hermite on log-discount A(t) = -r(t)·t,
    with Fritsch-Carlson monotonicity clamping.

    Parameters
    ----------
    tenors : observed maturities (years), strictly increasing
    yields : continuously compounded spot rates (%) at those maturities
    target : output maturities; defaults to TARGET_TENORS

    Returns
    -------
    Interpolated yields (%) at target maturities.
    """
    if target is None:
        target = TARGET_TENORS

    t = np.asarray(tenors, dtype=float)
    y = np.asarray(yields, dtype=float)

    # Convert spot rates → log discount factors: A(t) = -r(t)·t / 100
    A = -y / 100.0 * t

    n = len(t)

    # ── Estimate instantaneous forward rates f_i at knots ────────────────────
    # f(t) = -dA/dt; estimate as slope of quadratic through 3 adjacent points
    f = np.empty(n)
    if n == 1:
        f[0] = y[0] / 100.0
    elif n == 2:
        # Linear A → constant forward
        f[0] = f[1] = (A[1] - A[0]) / (t[1] - t[0])
    else:
        # Interior points: weighted harmonic of adjacent slopes
        for i in range(1, n - 1):
            h_l = t[i] - t[i - 1]
            h_r = t[i + 1] - t[i]
            s_l = (A[i] - A[i - 1]) / h_l
            s_r = (A[i + 1] - A[i]) / h_r
            # Quadratic through three points, evaluated at t[i]
            f[i] = (h_r * s_l + h_l * s_r) / (h_l + h_r)

        # Endpoints: extrapolate from first/last interior quadratic
        f[0] = s_l = (A[1] - A[0]) / (t[1] - t[0])
        f[n - 1] = s_r = (A[n - 1] - A[n - 2]) / (t[n - 1] - t[n - 2])

    # ── Fritsch-Carlson monotonicity clamping ─────────────────────────────────
    for i in range(n - 1):
        h = t[i + 1] - t[i]
        delta = (A[i + 1] - A[i]) / h   # must be ≤ 0 (discount factors decrease)
        if abs(delta) < 1e-14:
            f[i] = f[i + 1] = 0.0
            continue
        alpha = f[i] / delta
        beta  = f[i + 1] / delta
        # Ensure non-negative forwards: alpha, beta in [0, 3]
        if alpha < 0.0:
            f[i] = 0.0
            alpha = 0.0
        if beta < 0.0:
            f[i + 1] = 0.0
            beta = 0.0
        # Fritsch-Carlson ellipse constraint: α² + β² ≤ 9
        norm2 = alpha ** 2 + beta ** 2
        if norm2 > 9.0:
            tau = 3.0 / np.sqrt(norm2)
            f[i]     = tau * alpha * delta
            f[i + 1] = tau * beta  * delta

    # ── Cubic Hermite interpolation ───────────────────────────────────────────
    def _interp_A(t_star):
        """Evaluate A(t_star) via cubic Hermite."""
        if t_star <= t[0]:
            # Constant-rate extrapolation below first knot
            return A[0] + f[0] * (t_star - t[0])
        if t_star >= t[-1]:
            # Constant-forward extrapolation beyond last knot
            return A[-1] + f[-1] * (t_star - t[-1])
        i = np.searchsorted(t, t_star, side="right") - 1
        i = min(i, n - 2)
        h = t[i + 1] - t[i]
        s = (t_star - t[i]) / h
        h1 = (1 - s) ** 2 * (1 + 2 * s)
        h2 = s ** 2 * (3 - 2 * s)
        h3 = s * (1 - s) ** 2
        h4 = s ** 2 * (s - 1)
        return A[i] * h1 + A[i + 1] * h2 + f[i] * h * h3 + f[i + 1] * h * h4

    # Evaluate at each target maturity and convert back to yield
    out = np.empty(len(target))
    for k, t_star in enumerate(target):
        if t_star <= 0:
            out[k] = y[0]
            continue
        A_star = _interp_A(t_star)
        out[k] = -A_star / t_star * 100.0   # back to percent

    return out


# ── Method 2: Monotone Convex ─────────────────────────────────────────────────

def monotone_convex(
    tenors: np.ndarray,
    yields: np.ndarray,
    target: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Interpolate yields via quadratic forward-rate interpolation with
    8-case monotone fix (Hagan & West 2006 Monotone Convex method).

    Parameters
    ----------
    tenors : observed maturities (years), strictly increasing
    yields : continuously compounded spot rates (%) at those maturities
    target : output maturities; defaults to TARGET_TENORS

    Returns
    -------
    Interpolated yields (%) at target maturities.
    """
    if target is None:
        target = TARGET_TENORS

    t = np.asarray(tenors, dtype=float)
    y = np.asarray(yields, dtype=float)

    # Convert spot rates → log discount factors A(t) = -r(t)·t / 100
    A = -y / 100.0 * t
    n = len(t)

    # ── Discrete forward rates (interval averages) ───────────────────────────
    # f^d_i = -(A[i] - A[i-1]) / (t[i] - t[i-1])  (positive = rate positive)
    # Treat A[0] as corresponding to t[0] with A at 0 extrapolated
    # Insert a virtual t[0] = 0 point with A[0] = 0 (P(0) = 1 → A = 0)
    t_ext = np.concatenate([[0.0], t])
    A_ext = np.concatenate([[0.0], A])
    n_ext = len(t_ext)

    # Discrete forwards per interval [t_{i-1}, t_i] for i = 1..n
    f_d = np.empty(n)
    for i in range(n):
        h = t_ext[i + 1] - t_ext[i]
        f_d[i] = -(A_ext[i + 1] - A_ext[i]) / h   # forward rate > 0 when yields > 0

    # ── Convert discrete → point forwards at knots ───────────────────────────
    f_pt = np.empty(n + 1)   # at t_ext[0..n]

    # Interior knots (i = 1..n-1 in original, index 1..n-1 in t_ext)
    for i in range(1, n):
        h_l = t_ext[i] - t_ext[i - 1]
        h_r = t_ext[i + 1] - t_ext[i]
        f_pt[i] = (h_r * f_d[i - 1] + h_l * f_d[i]) / (h_l + h_r)

    # Endpoints
    f_pt[0] = f_d[0] - 0.5 * (f_pt[1] - f_d[0])
    f_pt[n] = f_d[-1] - 0.5 * (f_pt[n - 1] - f_d[-1])

    # Enforce non-negativity (clamp to [0, 2·min(adjacent f^d)])
    f_pt[0] = min(max(0.0, f_pt[0]), 2.0 * f_d[0])
    for i in range(1, n):
        f_pt[i] = min(max(0.0, f_pt[i]), 2.0 * min(f_d[i - 1], f_d[i]))
    f_pt[n] = min(max(0.0, f_pt[n]), 2.0 * f_d[-1])

    # ── Quadratic interpolation with 8-case monotone fix ─────────────────────
    def _forward(t_star):
        """Instantaneous forward rate at t_star."""
        if t_star <= t_ext[0]:
            return f_d[0]
        if t_star >= t_ext[-1]:
            return f_d[-1]
        i = np.searchsorted(t_ext, t_star, side="right") - 1
        i = min(i, n - 1)   # interval index: [t_ext[i], t_ext[i+1]]
        h = t_ext[i + 1] - t_ext[i]
        x = (t_star - t_ext[i]) / h   # in [0, 1]

        g0 = f_pt[i] - f_d[i]     # deviation at left endpoint
        g1 = f_pt[i + 1] - f_d[i]  # deviation at right endpoint

        # 8-case quadratic
        if g0 == 0.0 and g1 == 0.0:
            g = 0.0
        elif g0 >= 0.0 and g1 >= 0.0:
            # Case (iv): interior maximum → parabolic bump
            phi = g1 / (g1 + g0) if (g0 + g1) != 0 else 0.5
            A_  = -0.5 * (phi * g0 + (1 - phi) * g1)
            if x <= phi:
                g = A_ + (g0 - A_) * ((phi - x) / phi) ** 2 if phi > 0 else g0
            else:
                g = A_ + (g1 - A_) * ((x - phi) / (1 - phi)) ** 2 if phi < 1 else g1
        elif g0 <= 0.0 and g1 <= 0.0:
            # Case (i): interior minimum → parabolic dip
            phi = g1 / (g1 + g0) if (g0 + g1) != 0 else 0.5
            A_  = -0.5 * (phi * g0 + (1 - phi) * g1)
            if x <= phi:
                g = A_ + (g0 - A_) * ((phi - x) / phi) ** 2 if phi > 0 else g0
            else:
                g = A_ + (g1 - A_) * ((x - phi) / (1 - phi)) ** 2 if phi < 1 else g1
        elif g0 >= 0.0 and g1 < 0.0:
            # Case (ii): monotone increasing with local max
            phi = 1.0 + 3.0 * g0 / (g1 - g0) if g1 != g0 else 1.0
            phi = max(0.0, min(1.0, phi))
            if x <= phi:
                g = g0
            else:
                g = g0 + (g1 - g0) * (x - phi) / (1 - phi) if phi < 1 else g1
        else:
            # Case (iii): monotone decreasing with local min
            phi = -3.0 * g0 / (g1 - g0) if g1 != g0 else 0.0
            phi = max(0.0, min(1.0, phi))
            if x >= phi:
                g = g1
            else:
                g = g0 + (g1 - g0) * (x - phi) / (1 - phi) if phi < 1 else g1

        return f_d[i] + g

    # Precompute A at each knot using 5-point Gauss-Legendre quadrature
    # (exact for polynomials up to degree 9; forward is piecewise quadratic → exact)
    _gl5_x = np.array([0.0, 0.5384693101056831, -0.5384693101056831,
                        0.9061798459386640, -0.9061798459386640])
    _gl5_w = np.array([0.5688888888888889, 0.4786286704993665,  0.4786286704993665,
                        0.2369268850561891,  0.2369268850561891])

    def _integrate_piece(a, b):
        """∫_a^b f(u) du using 5-point GL."""
        mid = 0.5 * (a + b)
        half = 0.5 * (b - a)
        pts = mid + half * _gl5_x
        return half * np.dot(_gl5_w, [_forward(p) for p in pts])

    # A_knotval[i] = A at t_ext[i] (exact, using cumulative GL integration)
    A_knotval = np.empty(n + 1)
    A_knotval[0] = 0.0   # A(0) = ln(P(0)) = 0
    for i in range(n):
        A_knotval[i + 1] = A_knotval[i] - _integrate_piece(t_ext[i], t_ext[i + 1])

    def _interp_A(t_star):
        """A(t_star) via precomputed knot values + GL partial piece."""
        if t_star <= t_ext[0]:
            return A_knotval[0] - f_d[0] * (t_star - t_ext[0])
        if t_star >= t_ext[-1]:
            return A_knotval[n] - f_d[-1] * (t_star - t_ext[-1])
        i = np.searchsorted(t_ext, t_star, side="right") - 1
        i = min(i, n - 1)
        return A_knotval[i] - _integrate_piece(t_ext[i], t_star)

    # Evaluate at each target maturity
    out = np.empty(len(target))
    for k, t_star in enumerate(target):
        if t_star <= 0:
            out[k] = y[0]
            continue
        A_star = _interp_A(t_star)
        out[k] = -A_star / t_star * 100.0

    return out


# ── Row-level helpers ─────────────────────────────────────────────────────────

def interpolate_curve(
    row: pd.Series,
    method: str = "monotone_preserving",
    target: Optional[np.ndarray] = None,
    min_tenors: int = 4,
) -> Optional[pd.Series]:
    """
    Interpolate a single date's yield curve.

    Parameters
    ----------
    row     : pd.Series with float tenor (years) index and yield (%) values
    method  : 'monotone_preserving' or 'monotone_convex'
    target  : output maturities; defaults to TARGET_TENORS
    min_tenors : minimum non-NaN points required

    Returns
    -------
    pd.Series of interpolated yields at target tenors, or None if insufficient data.
    """
    if target is None:
        target = TARGET_TENORS

    valid = row.dropna().sort_index()
    if len(valid) < min_tenors:
        return None

    tenors = valid.index.to_numpy(dtype=float)
    yields = valid.to_numpy(dtype=float)

    if method == "monotone_preserving":
        out = monotone_preserving(tenors, yields, target)
    elif method == "monotone_convex":
        out = monotone_convex(tenors, yields, target)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    return pd.Series(out, index=target)


def build_full_curves(
    curves: pd.DataFrame,
    method: str = "monotone_preserving",
    target: Optional[np.ndarray] = None,
    min_tenors: int = 4,
) -> pd.DataFrame:
    """
    Apply interpolation to every row of a date × tenor DataFrame.

    Returns date × target_tenor DataFrame.
    NaN rows are returned for dates with insufficient data.
    """
    if target is None:
        target = TARGET_TENORS

    rows = []
    for date, row in curves.iterrows():
        s = interpolate_curve(row, method=method, target=target, min_tenors=min_tenors)
        if s is not None:
            rows.append(s.rename(date))
        else:
            rows.append(pd.Series(np.nan, index=target, name=date))

    result = pd.DataFrame(rows)
    result.index.name = "date"
    return result
