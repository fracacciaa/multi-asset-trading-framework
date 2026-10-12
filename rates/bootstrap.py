"""
Par yield → zero coupon rate bootstrap.

Par bonds price at 100 by definition. The bootstrap solves for the discount
factor at each maturity given all previously derived discount factors:

    d(τᵢ) = (1 − c × Σⱼ<ᵢ d(tⱼ)) / (1 + c)

where c = par_yield / freq  (coupon per period, decimal)
      tⱼ = coupon payment dates strictly before τᵢ

Intermediate coupon dates not yet bootstrapped are filled by log-linear
interpolation of existing discount factors (constant forward between knots).

Short end (τ ≤ 1/freq): no coupon — treated as zero coupon instrument:
    d(τ) = exp(−r × τ / 100)
"""

import numpy as np
import pandas as pd
from typing import Optional

# Coupon frequency per market (payments per year)
COUPON_FREQ: dict[str, int] = {
    "US":        2,   # US Treasury: semi-annual
    "Germany":   1,   # Bund: annual
    "UK":        2,   # Gilt: semi-annual
    "Japan":     2,   # JGB: semi-annual
    "France":    1,   # OAT: annual
    "Canada":    2,   # semi-annual
    "Australia": 2,   # semi-annual
}


def _loglinear_discount(knot_t: np.ndarray, knot_d: np.ndarray, t: float) -> float:
    """Log-linear interpolation/extrapolation of discount factors."""
    if t <= knot_t[0]:
        return knot_d[0]
    if t >= knot_t[-1]:
        if len(knot_t) < 2:
            return knot_d[-1]   # only origin knot: flat at 1.0
        # Flat forward extrapolation beyond last knot
        log_d = np.log(knot_d[-1]) + (np.log(knot_d[-1]) - np.log(knot_d[-2])) / \
                (knot_t[-1] - knot_t[-2]) * (t - knot_t[-1])
        return np.exp(log_d)
    i = np.searchsorted(knot_t, t, side="right") - 1
    frac = (t - knot_t[i]) / (knot_t[i + 1] - knot_t[i])
    log_d = (1 - frac) * np.log(knot_d[i]) + frac * np.log(knot_d[i + 1])
    return np.exp(log_d)


def bootstrap(
    tenors: np.ndarray,
    par_yields: np.ndarray,
    coupon_freq: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bootstrap par yields to continuously compounded zero rates.

    Parameters
    ----------
    tenors      : maturities in years, strictly increasing, ≥1 element
    par_yields  : annual par yields in percent (e.g. 5.0 means 5%)
    coupon_freq : coupon payments per year (2 = semi-annual, 1 = annual)

    Returns
    -------
    (tenors, zero_rates) — same tenor grid, zero rates in percent (c.c.)
    """
    t = np.asarray(tenors, dtype=float)
    r = np.asarray(par_yields, dtype=float)

    dt = 1.0 / coupon_freq          # interval between coupon dates
    zero_coupon_max = dt            # treat maturities ≤ dt as zero coupon

    # Bootstrapped knots: start at t=0, d=1
    knot_t = np.array([0.0])
    knot_d = np.array([1.0])

    zero_rates = np.empty(len(t))

    for i in range(len(t)):
        tau  = t[i]
        r_pa = r[i] / 100.0         # par yield as decimal
        c    = r_pa / coupon_freq   # coupon per period

        if tau <= zero_coupon_max:
            # Money market / short end: zero coupon → direct
            d_tau = np.exp(-r_pa * tau)
        else:
            # Generate all coupon dates up to and including τ
            n_periods = int(round(tau * coupon_freq))
            coupon_dates = np.array([j * dt for j in range(1, n_periods + 1)])

            # Sum discounted intermediate coupons (all except final)
            coupon_sum = 0.0
            for t_c in coupon_dates[:-1]:
                d_tc = _loglinear_discount(knot_t, knot_d, t_c)
                coupon_sum += d_tc

            # Solve for d(τ):  1 = c * coupon_sum + (1 + c) * d(τ)
            d_tau = (1.0 - c * coupon_sum) / (1.0 + c)

        # Guard against non-positive discount factors (bad input data)
        d_tau = max(d_tau, 1e-6)

        # Continuously compounded zero rate
        zero_rates[i] = -np.log(d_tau) / tau * 100.0

        # Append to known knots
        knot_t = np.append(knot_t, tau)
        knot_d = np.append(knot_d, d_tau)

    return t, zero_rates


def bootstrap_row(
    row: pd.Series,
    coupon_freq: int = 2,
    min_tenors: int = 4,
) -> Optional[pd.Series]:
    """
    Bootstrap a single date's par yield curve (a pd.Series with float-year index).

    Returns pd.Series of zero rates with same index, or None if insufficient data.
    """
    valid = row.dropna().sort_index()
    if len(valid) < min_tenors:
        return None

    tenors = valid.index.to_numpy(dtype=float)
    par_yields = valid.to_numpy(dtype=float)

    _, zero_rates = bootstrap(tenors, par_yields, coupon_freq=coupon_freq)
    return pd.Series(zero_rates, index=tenors, name=row.name)


def bootstrap_curves(
    curves: pd.DataFrame,
    coupon_freq: int = 2,
    min_tenors: int = 4,
) -> pd.DataFrame:
    """
    Bootstrap every row of a date × tenor par yield DataFrame.

    Returns date × tenor DataFrame of continuously compounded zero rates (%).
    Rows with insufficient data are returned as NaN.
    """
    rows = []
    for date, row in curves.iterrows():
        s = bootstrap_row(row, coupon_freq=coupon_freq, min_tenors=min_tenors)
        if s is not None:
            rows.append(s.rename(date))
        else:
            rows.append(pd.Series(np.nan, index=curves.columns, name=date))

    result = pd.DataFrame(rows)
    result.index.name = "date"
    return result
