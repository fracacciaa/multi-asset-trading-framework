"""
OIS zero-curve bootstrap for the classic-NS signal.

Three design decisions that differ from yc_builder.bootstrap:

  1.  Log-DF interpolation for intermediate coupon dates.
      Intermediate annual coupons of a T > 1Y swap are discounted using
      log-linear interpolation on already-bootstrapped discount factors
      (linear in log(DF) vs time).  Plain DF interpolation or zero-rate
      interpolation both introduce systematic forward-rate kinks near the
      quoted tenors, which corrupt the NS residuals used for the signal.

  2.  Act/360 accrual via tau_fn.
      Coupon accrual factors are computed by a caller-supplied function
      tau_fn(start_yr, end_yr) → fraction.  The default is the act/360
      approximation (end − start) × 365/360; callers with actual date
      grids may pass their own.  The existing yc_builder uses a fixed
      1/coupon_freq period, which is exactly 1.0 year for all OIS markets
      — correct for approximate but wrong at the basis-point level for
      non-round tenors near year boundaries.

  3.  T+2 spot start.
      OIS swaps settle T+2.  The maturity of a T-year swap measured from
      today is T + 2/365, not just T.  Ignoring the offset shifts the
      short-end zero curve by ~2 days, which is immaterial for multi-year
      tenors but biases the front end (1M–6M) where the NS curvature
      factor is most sensitive.  We apply the correction only at the zero-
      rate conversion step: z = −ln(df) / (T + t_plus).

Output convention: continuously compounded zero rates in percent (5.0 = 5%).
"""

import numpy as np
from typing import Callable

# Default tau function: act/360 approximation from fractional years
_ACT360: Callable[[float, float], float] = lambda s, e: (e - s) * 365.0 / 360.0

_T_PLUS_DEFAULT: float = 2.0 / 365.0   # T+2 spot offset in years


def _logdf_interp(t: float, knot_t: np.ndarray, knot_ldf: np.ndarray) -> float:
    """
    Log-linear DF interpolation: linear in log(DF) vs time.

    Uses flat extrapolation beyond the last knot (constant forward rate),
    consistent with the flat-forward behaviour of yc_builder._loglinear_discount.
    """
    if len(knot_t) == 0:
        return 0.0                         # no knots yet → DF = 1 → log(DF) = 0
    if t <= knot_t[0]:
        return knot_ldf[0]
    if t >= knot_t[-1]:
        # Flat-forward extrapolation using the last interval's slope
        if len(knot_t) >= 2:
            slope = (knot_ldf[-1] - knot_ldf[-2]) / (knot_t[-1] - knot_t[-2])
            return knot_ldf[-1] + slope * (t - knot_t[-1])
        return knot_ldf[-1]
    return float(np.interp(t, knot_t, knot_ldf))


def bootstrap_ois(
    tenors: np.ndarray,
    par: np.ndarray,
    tau_fn: Callable[[float, float], float] = _ACT360,
    t_plus: float = _T_PLUS_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bootstrap OIS par rates to discount factors and continuously compounded
    zero rates.

    Parameters
    ----------
    tenors  : (n,) maturities in years from spot, strictly ascending
              (e.g. [1/12, 2/12, ..., 1, 2, ..., 30])
    par     : (n,) OIS par rates as decimals (0.05 = 5%)
    tau_fn  : accrual day-count function(start_yr, end_yr) → fraction
              Default: act/360 approximation, (e − s) × 365/360
    t_plus  : spot offset in years applied to the zero-rate denominator
              (default 2/365 ≈ T+2); set to 0 to disable

    Returns
    -------
    tenors     : (n,) input tenors (passed through unchanged)
    dfs        : (n,) bootstrapped discount factors (< 1 for positive rates)
    zero_rates : (n,) continuously compounded zero rates, percent
                 z_i = −ln(df_i) / (T_i + t_plus) × 100
    """
    tenors = np.asarray(tenors, dtype=float)
    par    = np.asarray(par,    dtype=float)
    n      = len(tenors)

    dfs        = np.empty(n)
    knot_t     = np.empty(0, dtype=float)
    knot_ldf   = np.empty(0, dtype=float)

    for i in range(n):
        T = tenors[i]
        r = par[i]

        if T <= 1.0:
            # Zero-coupon segment: single exchange at maturity
            a   = tau_fn(0.0, T)
            df  = 1.0 / (1.0 + r * a)
        else:
            # Annual fixed leg
            # Payment times: T−(n−1), T−(n−2), ..., T  (ascending, n = round(T))
            n_coupons = int(round(T))
            t_pay = np.array([T - (n_coupons - j - 1) for j in range(n_coupons)])
            # Accrual factors: τ_j = tau_fn(previous coupon date, this coupon date)
            t_prev = np.concatenate([[0.0], t_pay[:-1]])
            a_j    = np.array([tau_fn(s, e) for s, e in zip(t_prev, t_pay)])

            # Discount intermediate coupons via log-DF interpolation
            prev_ann = sum(
                a_j[j] * np.exp(_logdf_interp(t_pay[j], knot_t, knot_ldf))
                for j in range(n_coupons - 1)
            )

            df = (1.0 - r * prev_ann) / (1.0 + r * a_j[-1])

        df = max(df, 1e-10)
        dfs[i] = df

        # Accumulate knots for the next iteration
        knot_t   = np.append(knot_t,   T)
        knot_ldf = np.append(knot_ldf, np.log(df))

    # T+2 spot correction applied only at the zero-rate conversion step
    t_eff      = tenors + t_plus
    zero_rates = -np.log(dfs) / t_eff * 100.0

    return tenors, dfs, zero_rates


def bootstrap_curves(
    curves: "pd.DataFrame",   # date × tenor (years); par yields in percent
    tau_fn: Callable[[float, float], float] = _ACT360,
    t_plus: float = _T_PLUS_DEFAULT,
    min_tenors: int = 6,
) -> "pd.DataFrame":
    """
    Apply bootstrap_ois to every row of a date × tenor par-yield DataFrame.

    Parameters
    ----------
    curves     : date × tenor (years) DataFrame; values are par yields in percent
    tau_fn     : passed through to bootstrap_ois
    t_plus     : passed through to bootstrap_ois
    min_tenors : skip rows with fewer valid observations

    Returns
    -------
    date × tenor DataFrame of continuously compounded zero rates in percent.
    Rows with insufficient data are filled with NaN.
    """
    import pandas as pd

    rows = []
    for date, row in curves.iterrows():
        valid = row.dropna().sort_index()
        if len(valid) < min_tenors:
            rows.append(pd.Series(np.nan, index=curves.columns, name=date))
            continue

        t_arr = valid.index.to_numpy(dtype=float)
        p_arr = valid.to_numpy(dtype=float) / 100.0   # % → decimal for bootstrap

        _, _, z = bootstrap_ois(t_arr, p_arr, tau_fn=tau_fn, t_plus=t_plus)

        rows.append(pd.Series(z, index=t_arr, name=date))

    result = pd.DataFrame(rows)
    result.index.name = "date"
    return result
