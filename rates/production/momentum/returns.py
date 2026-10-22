"""
Return panels for OIS-based momentum signals.

Per-market output (all indexed by month-end date, columns = tenor in years):

  ret_d   daily constant-maturity approximation (cm_return_bp), bp
  ret_m   month-end-to-month-end constant-maturity returns, bp
  rx_m    monthly log excess returns (Sihvonen eq 1, exact bootstrap), bp
  c_m     carry component of rx_m (observable at t), bp
  yc_m    yield-change component: rx_m == c_m - yc_m to 1e-10
  z_m     monthly zero-curve panel, annual cc %, used by Signal B PC1

Long-position convention: receiving fixed.
Positive return = rates fell (receiver profits).
A swap is unfunded: the rf rate enters once, inside cm_return_bp carry,
as (s_now - rf_bp) × dt.  Do not subtract a second funding term.

Bootstrap: reuses rates.yc_builder.build_zero_curve (monotone-preserving,
annual coupon, cc for the short end).  The spec's simple-compounding stub
for tenors < 1Y differs by < 0.5 bp at normal rate levels; not material
for a return that accumulates over one month or longer.

Spec: ois_momentum_signals_spec.md §1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from rates.production.ois_data import fetch_ois_curves, OIS_SHEETS
from rates.production.yc_builder import build_zero_curve

# ── Daycount basis per curve (sheet name = curve_id) ─────────────────────────
# Used only in bootstrap_ois; the reused build_zero_curve path ignores it
# (cc convention throughout, which is standard for OIS).

DAYCOUNT: dict[str, int] = {
    "USDEFFR": 360, "USDSOFR": 360,
    "EUR": 360, "CHF": 360, "SEK": 360,
    "GBP": 365, "JPY": 365, "CAD": 365,
    "AUD": 365, "NZD": 365, "INR": 365, "CNY": 365,
}

# Sihvonen maturities: 12, 24, …, 120 months = 1–10Y annual steps
SIHVONEN_MONTHS: list[int] = list(range(12, 121, 12))
SIHVONEN_YEARS: list[float] = [m / 12.0 for m in SIHVONEN_MONTHS]


# ── Low-level helpers ─────────────────────────────────────────────────────────

def df_interp(df_dict: dict[float, float], u: float) -> float:
    """Log-linear interpolation of discount factors (piecewise-constant forward)."""
    ks = sorted(df_dict)
    if not ks:
        return 1.0
    if u <= ks[0]:
        return df_dict[ks[0]] ** (u / ks[0]) if ks[0] > 0 else 1.0
    if u >= ks[-1]:
        return df_dict[ks[-1]] ** (u / ks[-1])
    lo = max(k for k in ks if k <= u)
    hi = min(k for k in ks if k > u)
    if lo == hi:
        return df_dict[lo]
    w = (u - lo) / (hi - lo)
    return np.exp(
        (1 - w) * np.log(max(df_dict[lo], 1e-15))
        + w * np.log(max(df_dict[hi], 1e-15))
    )


def _dfs_from_zero_row(zero_row: pd.Series) -> dict[float, float]:
    """Build {tenor: df} from a zero-rate row (annual cc, %)."""
    result: dict[float, float] = {}
    for t, z in zero_row.dropna().items():
        t, z = float(t), float(z)
        if t > 0 and np.isfinite(z):
            result[t] = float(np.exp(-z / 100.0 * t))
    return result


def swap_duration(dfs: dict[float, float], tau: float) -> float:
    """
    Annuity-weighted duration of a par swap.

    Formula (spec §1.4): Σ u × DF(u) for annual payment dates u = 1 … tau.
    For tau < 1Y (single-payment stub): returns tau directly.
    """
    if tau < 1.0:
        return tau
    pay = np.arange(1.0, tau + 1e-9, 1.0)
    return float(sum(u * df_interp(dfs, u) for u in pay))


def cm_return_bp(
    s_now: float,
    s_aged: float,
    s_next: float,
    dur: float,
    rf_bp: float,
    dt: float,
) -> float:
    """
    Constant-maturity receiver swap return over one period dt.

    s_now   : par rate at tau, time t (bp)
    s_aged  : par rate at tau-dt, time t (roll target, bp)
    s_next  : par rate at tau, time t+1 (bp)
    dur     : annuity duration at t (years, from swap_duration)
    rf_bp   : funding rate at t (bp)
    dt      : step size in years

    No second funding subtraction: the swap floating leg is already the funding leg.
    """
    carry = (s_now - rf_bp) * dt
    roll  = dur * (s_now - s_aged)
    price = -dur * (s_next - s_now)
    return carry + roll + price


def log_excess_return_bp(
    z_t: object,   # callable: tau_years -> annual cc zero yield (decimal)
    z_tp1: object,
    n_months: int,
) -> float:
    """
    One-month log excess return of an n-month zero, in bp.  Sihvonen eq (1).

    z_t, z_tp1: callables tau_years -> annual cc yield as decimal (e.g. 0.025 = 2.5%).
    n: maturity in months.  Monthly yield = annual / 12.
    """
    n = n_months
    y_n_t    = z_t(n / 12.0) / 12.0
    y_nm1_t1 = z_tp1((n - 1) / 12.0) / 12.0
    y_1_t    = z_t(1.0 / 12.0) / 12.0
    return (-(n - 1) * y_nm1_t1 + n * y_n_t - y_1_t) * 1e4


def _carry_yc_bp(
    z_t: object,
    z_tp1: object,
    n_months: int,
) -> tuple[float, float]:
    """
    Carry/yield-change decomposition of the log excess return.
    Returns (c, yc) where rx == c - yc.  Sihvonen eq (5).
    """
    n = n_months
    y_n_t     = z_t(n / 12.0) / 12.0
    y_nm1_t   = z_t((n - 1) / 12.0) / 12.0
    y_nm1_t1  = z_tp1((n - 1) / 12.0) / 12.0
    y_1_t     = z_t(1.0 / 12.0) / 12.0

    c  = (-(n - 1) * y_nm1_t + n * y_n_t - y_1_t) * 1e4
    yc = (n - 1) * (y_nm1_t1 - y_nm1_t) * 1e4
    return c, yc


# ── Month-end helper ──────────────────────────────────────────────────────────

def _month_end_dates(idx: pd.DatetimeIndex) -> list:
    """Last calendar date in `idx` for each (year, month) combination."""
    s = pd.Series(idx, index=idx)
    ends = s.groupby([s.index.year, s.index.month]).last()
    return list(ends.values)


# ── Main panel builder ────────────────────────────────────────────────────────

def build_return_panels(
    market: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    tenors: Optional[list[float]] = None,
    zero_curve: Optional[pd.DataFrame] = None,
) -> dict[str, pd.DataFrame]:
    """
    Build all return panels for one OIS market.

    Parameters
    ----------
    market  : display name used by fetch_ois_curves (e.g. "US", "EUR")
    tenors  : constant-maturity tenors for ret_d / ret_m.  Defaults to all
              tenors available in the OIS curve that are also in the zero curve.

    Returns
    -------
    dict with keys:
      ret_d  daily CM returns, date × tenor (bp)
      ret_m  monthly CM returns, date × tenor (bp)
      rx_m   Sihvonen monthly log excess returns, date × {1..10}Y (bp)
      c_m    carry component of rx_m (bp)
      yc_m   yield-change component; rx_m == c_m - yc_m
      z_m    monthly zero-curve panel, date × all tenors (annual cc %)
    """
    par = fetch_ois_curves(market, start=start, end=end)    # %, date × tenor
    par_bp = par * 100.0                                     # % → bp

    zero = zero_curve if zero_curve is not None else build_zero_curve(par)

    # ── Tenor selection ───────────────────────────────────────────────────────
    if tenors is None:
        avail = sorted(t for t in par.columns if t in zero.columns)
    else:
        avail = sorted(t for t in tenors if t in par.columns and t in zero.columns)

    # Funding rate: shortest available OIS tenor
    rf_col = float(min(par.columns))

    # ── Daily CM returns (vectorised across dates) ───────────────────────────
    dates     = par.index
    dt_d      = 1.0 / 252
    par_np    = par_bp.to_numpy(float)          # (D, n_par_cols)
    par_tcols = par_bp.columns.to_numpy(float)
    zero_np   = zero.to_numpy(float)            # (D, 70)
    zero_tcols = zero.columns.to_numpy(float)

    # Pre-compute discount factors at integer years 1..30 for all dates
    int_yrs = np.arange(1.0, 31.0)
    int_idx = np.array([int(np.argmin(np.abs(zero_tcols - u))) for u in int_yrs])
    int_df_mat = np.exp(-zero_np[:, int_idx] / 100.0 * int_yrs[None, :])  # (D, 30)

    rf_col_idx = int(np.argmin(np.abs(par_tcols - rf_col)))
    rf_vec     = par_np[:, rf_col_idx]           # (D,)

    D = len(dates)
    ret_d_mat = np.full((D - 1, len(avail)), np.nan)

    for j, tau in enumerate(avail):
        tau_cidx = int(np.argmin(np.abs(par_tcols - tau)))
        s_now  = par_np[:-1, tau_cidx]
        s_next = par_np[1:,  tau_cidx]

        tau_aged = max(tau - dt_d, par_tcols[0])
        ir = int(np.clip(np.searchsorted(par_tcols, tau_aged), 1, len(par_tcols) - 1))
        il = ir - 1
        w  = ((tau_aged - par_tcols[il]) / (par_tcols[ir] - par_tcols[il])
              if par_tcols[ir] != par_tcols[il] else 0.0)
        s_aged = (1 - w) * par_np[:-1, il] + w * par_np[:-1, ir]

        if tau < 1.0:
            dur = np.full(D - 1, tau)
        else:
            n_c     = int(round(tau))
            weights = np.arange(1, n_c + 1, dtype=float)
            dur     = (int_df_mat[:-1, :n_c] * weights[None, :]).sum(axis=1)

        rf_arr = rf_vec[:-1]
        ret_d_mat[:, j] = ((s_now - rf_arr) * dt_d
                           + dur * (s_now - s_aged)
                           - dur * (s_next - s_now))

    ret_d = pd.DataFrame(ret_d_mat, index=dates[:-1], columns=avail)
    ret_d.index.name = "date"

    # ── Month-end CM returns ──────────────────────────────────────────────────
    me_dates = _month_end_dates(dates)
    me_valid = sorted(d for d in me_dates if d in par_bp.index and d in zero.index)
    dt_m = 1.0 / 12
    ret_m_rows: dict = {}

    for i in range(len(me_valid) - 1):
        t, t1 = me_valid[i], me_valid[i + 1]
        par_row  = par_bp.loc[t].dropna()
        par_row1 = par_bp.loc[t1].dropna()
        zero_row = zero.loc[t].dropna()

        dfs_t = _dfs_from_zero_row(zero_row)
        rf    = float(par_row.get(rf_col, par_row.iloc[0]))
        raw_t = par_row.index.to_numpy(float)
        raw_v = par_row.to_numpy(float)

        row = {}
        for tau in avail:
            if tau not in par_row.index or tau not in par_row1.index:
                row[tau] = np.nan
                continue
            s_now  = float(par_row[tau])
            s_next = float(par_row1[tau])
            tau_aged = max(tau - dt_m, raw_t[0])
            s_aged = float(np.interp(tau_aged, raw_t, raw_v))
            dur    = swap_duration(dfs_t, tau)
            row[tau] = cm_return_bp(s_now, s_aged, s_next, dur, rf, dt_m)

        ret_m_rows[t] = row

    ret_m = pd.DataFrame(ret_m_rows).T
    ret_m.index.name = "date"

    # ── Sihvonen log excess returns (vectorised across dates) ────────────────
    zero_me    = zero.reindex([d for d in me_valid if d in zero.index])
    zm_np      = zero_me.to_numpy(float) / 100.0   # (M, 70) annual cc decimal
    zm_cols    = zero_me.columns.to_numpy(float)    # TARGET_TENORS
    me_idx     = list(zero_me.index)
    M          = len(me_idx)

    ns_arr  = np.array(SIHVONEN_MONTHS)            # [12,24,...,120]
    yrs_n   = ns_arr / 12.0                         # [1..10]
    yrs_nm1 = (ns_arr - 1) / 12.0                  # [11/12, 23/12, ...]
    yr_1m   = 1.0 / 12.0

    def _query_cols(Z, taus):
        """Vectorised linear interp of Z (D×70) at each tau in taus. Returns (D, K)."""
        out = np.empty((len(Z), len(taus)))
        for k, tau in enumerate(taus):
            ir = int(np.clip(np.searchsorted(zm_cols, tau), 1, len(zm_cols) - 1))
            il = ir - 1
            w  = ((tau - zm_cols[il]) / (zm_cols[ir] - zm_cols[il])
                  if zm_cols[ir] != zm_cols[il] else 0.0)
            out[:, k] = (1 - w) * Z[:, il] + w * Z[:, ir]
        return out

    Z0 = zm_np[:-1]   # (M-1, 70) — zero curve at t
    Z1 = zm_np[1:]    # (M-1, 70) — zero curve at t+1

    yn_t    = _query_cols(Z0, yrs_n)              / 12.0  # (M-1, 10) monthly
    ynm1_t1 = _query_cols(Z1, yrs_nm1)            / 12.0
    ynm1_t  = _query_cols(Z0, yrs_nm1)            / 12.0
    y1_t    = _query_cols(Z0, np.array([yr_1m]))   / 12.0  # (M-1, 1)

    ns2d   = ns_arr[None, :]                               # (1, 10)
    rx_mat = (-(ns2d - 1) * ynm1_t1 + ns2d * yn_t - y1_t) * 1e4
    c_mat  = (-(ns2d - 1) * ynm1_t  + ns2d * yn_t - y1_t) * 1e4
    yc_mat = (ns2d - 1) * (ynm1_t1 - ynm1_t) * 1e4

    rx_m = pd.DataFrame(rx_mat, index=me_idx[:-1], columns=SIHVONEN_YEARS)
    c_m  = pd.DataFrame(c_mat,  index=me_idx[:-1], columns=SIHVONEN_YEARS)
    yc_m = pd.DataFrame(yc_mat, index=me_idx[:-1], columns=SIHVONEN_YEARS)
    for df in (rx_m, c_m, yc_m):
        df.index.name = "date"

    # ── Monthly zero panel (for Signal B PC1) ─────────────────────────────────
    z_m = zero.loc[[d for d in me_valid if d in zero.index]].copy()
    z_m.index.name = "date"

    return {
        "ret_d": ret_d,
        "ret_m": ret_m,
        "rx_m":  rx_m,
        "c_m":   c_m,
        "yc_m":  yc_m,
        "z_m":   z_m,
    }


def build_rx_monthly(
    market: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """
    Fast path: monthly Sihvonen log excess return panels only.

    Skips the daily cm_return_bp loop; only needs build_zero_curve (one pass).
    Use when Signal A or B is the only consumer and bucket returns are not needed.

    Returns {rx_m, c_m, yc_m, z_m} — same keys as build_return_panels but
    without ret_d / ret_m.
    """
    par  = fetch_ois_curves(market, start=start, end=end)
    zero = build_zero_curve(par)
    me   = _month_end_dates(zero.index)
    zm   = zero.reindex([d for d in me if d in zero.index]).dropna(how="all")
    dates = list(zm.index)

    zm_np   = zm.to_numpy(float) / 100.0      # (M, 70) annual cc decimal
    zm_cols = zm.columns.to_numpy(float)      # TARGET_TENORS

    ns_arr  = np.array(SIHVONEN_MONTHS)
    yrs_n   = ns_arr / 12.0
    yrs_nm1 = (ns_arr - 1) / 12.0
    yr_1m   = 1.0 / 12.0

    def _qcols(Z, taus):
        out = np.empty((len(Z), len(taus)))
        for k, tau in enumerate(taus):
            ir = int(np.clip(np.searchsorted(zm_cols, tau), 1, len(zm_cols) - 1))
            il = ir - 1
            w  = ((tau - zm_cols[il]) / (zm_cols[ir] - zm_cols[il])
                  if zm_cols[ir] != zm_cols[il] else 0.0)
            out[:, k] = (1 - w) * Z[:, il] + w * Z[:, ir]
        return out

    Z0 = zm_np[:-1];  Z1 = zm_np[1:]
    yn_t    = _qcols(Z0, yrs_n)             / 12.0
    ynm1_t1 = _qcols(Z1, yrs_nm1)           / 12.0
    ynm1_t  = _qcols(Z0, yrs_nm1)           / 12.0
    y1_t    = _qcols(Z0, np.array([yr_1m])) / 12.0

    ns2d   = ns_arr[None, :]
    rx_mat = (-(ns2d - 1) * ynm1_t1 + ns2d * yn_t - y1_t) * 1e4
    c_mat  = (-(ns2d - 1) * ynm1_t  + ns2d * yn_t - y1_t) * 1e4
    yc_mat = (ns2d - 1) * (ynm1_t1 - ynm1_t) * 1e4

    rx_m = pd.DataFrame(rx_mat, index=dates[:-1], columns=SIHVONEN_YEARS)
    c_m  = pd.DataFrame(c_mat,  index=dates[:-1], columns=SIHVONEN_YEARS)
    yc_m = pd.DataFrame(yc_mat, index=dates[:-1], columns=SIHVONEN_YEARS)
    for df in (rx_m, c_m, yc_m):
        df.index.name = "date"

    z_m = zm.copy()
    z_m.index.name = "date"
    return {"rx_m": rx_m, "c_m": c_m, "yc_m": yc_m, "z_m": z_m}


def build_all_return_panels(
    markets: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    tenors: Optional[list[float]] = None,
    zero_curves: Optional[dict] = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Build return panels for all markets.  Returns {market: panels_dict}."""
    if markets is None:
        from rates.production.ois_data import OIS_MARKETS
        markets = OIS_MARKETS

    return {
        mkt: build_return_panels(
            mkt, start=start, end=end, tenors=tenors,
            zero_curve=zero_curves.get(mkt) if zero_curves else None,
        )
        for mkt in markets
    }
