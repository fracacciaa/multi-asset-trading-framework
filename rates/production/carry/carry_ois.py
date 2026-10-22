"""
OIS carry signals — E01, E02, E04 from rates_carry_research_implementation.md.

Uses bootstrapped OIS zero curves (yc_builder) — NOT Nelson-Siegel.

Three signals per tenor × horizon:
  roll_proxy_bp    (OIS-0): par-curve roll-down proxy, no funding adjustment
                            = (K_t(T) - K_t(T-h)) × A_t(T-h,aged)  × 1e4
  static_carry_bp  (OIS-1): full static-curve carry after funding subtraction
                            = roll_proxy - r_fund × h × 1e4
  carry_dv01_bp    (OIS-4): static carry per unit DV01 (risk-normalized)
                            = static_carry_bp / (A_t(T) × 1e4)  [bp per bp-DV01]
  carry_zscore     (OIS-5): trailing-z score of static_carry_bp, min 252 obs

All carry values are for a *receiver* OIS swap (receive fixed, pay float).
Positive = receiver earns under the static (unchanged) scenario.

Formulae reference:
  §5.1  roll proxy       : RollPnL = N × A_t(T-h) × (K_t(T) - K_t(T-h))
  §5.2  static carry     : C^S = roll_proxy - r_fund × h  (no coupon in h < 1Y)
  §2    DV01-normalised  : R^DV01 = PnL / DV01 = PnL / (A_t(T) × 1e-4)
  §5.5  z-score          : (C^S - mean_past) / std_past, trailing min-252 obs

OIS conventions: annual coupon (freq=1); short end ≤ 1Y treated as zero-coupon
during bootstrap (consistent with yc_builder bootstrap()).
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import Optional

from rates.production.ois_data import fetch_ois_curves, OIS_MARKETS
from rates.production.yc_builder import build_zero_curve

# ── Default parameters ────────────────────────────────────────────────────────

BENCHMARK_TENORS: list[float] = [2.0, 5.0, 10.0, 20.0, 30.0]
HORIZONS: dict[str, float] = {"1M": 1 / 12, "3M": 3 / 12, "6M": 6 / 12}
OIS_COUPON_FREQ: int = 1   # annual coupon for all OIS markets
ZSCORE_LOOKBACK: int = 756  # 3 years trailing; minimum 252


# ── Curve primitives (all inputs in % for zero rates, decimal for rates) ─────

def _df(z_tenors: np.ndarray, z_pct: np.ndarray, t: float) -> float:
    """Discount factor P(t) = exp(-z(t)/100 × t), z interpolated in %."""
    if t <= 0:
        return 1.0
    z = float(np.interp(t, z_tenors, z_pct))
    return float(np.exp(-z / 100.0 * t))


def _annuity(z_tenors: np.ndarray, z_pct: np.ndarray, tenor: float,
             freq: int = OIS_COUPON_FREQ) -> float:
    """
    Discounted fixed-leg annuity A(T) = Σ P(j/freq) × (1/freq).
    Coupon dates: 1/freq, 2/freq, ..., T (last date = T exactly).
    """
    n = max(1, round(tenor * freq))
    coupon_dates = np.array([j / freq for j in range(1, n + 1)])
    coupon_dates[-1] = tenor  # snap last date exactly to T
    dfs = np.array([_df(z_tenors, z_pct, cd) for cd in coupon_dates])
    return float(dfs.sum()) / freq


def _par_rate_pct(z_tenors: np.ndarray, z_pct: np.ndarray, tenor: float,
                  freq: int = OIS_COUPON_FREQ) -> float:
    """
    Par rate (%) at tenor T back-calculated from bootstrapped zero curve.
    K(T) = (1 - P(T)) / A(T) × 100
    """
    ann = _annuity(z_tenors, z_pct, tenor, freq)
    df_T = _df(z_tenors, z_pct, tenor)
    if ann < 1e-10:
        return 0.0
    return (1.0 - df_T) / ann * 100.0


def _aged_annuity(z_tenors: np.ndarray, z_pct: np.ndarray,
                  tenor: float, horizon: float,
                  freq: int = OIS_COUPON_FREQ) -> float:
    """
    Annuity for the *remaining* cashflows of a swap originally at tenor T,
    aged by horizon h under the static-TTM scenario.

    Original coupon dates (TTM from entry): 1/freq, 2/freq, ..., T
    After aging h, remaining TTMs = {j/freq - h : j/freq > h}
    Under Scenario S: discount at today's zero rates for those TTMs.
    """
    n = max(1, round(tenor * freq))
    orig_dates = np.array([j / freq for j in range(1, n + 1)])
    orig_dates[-1] = tenor
    remaining_ttm = orig_dates - horizon
    remaining_ttm = remaining_ttm[remaining_ttm > 1e-6]
    if len(remaining_ttm) == 0:
        return 0.0
    dfs = np.array([_df(z_tenors, z_pct, ttm) for ttm in remaining_ttm])
    return float(dfs.sum()) / freq


# ── Per-date carry computation ────────────────────────────────────────────────

def _ois_carry_one_date(
    z_tenors: np.ndarray,
    z_pct: np.ndarray,
    rf_pct: float,
    tenors: list[float],
    horizons: dict[str, float],
) -> list[dict]:
    """
    Compute all OIS carry variants for one date across tenors × horizons.

    Parameters
    ----------
    z_tenors : array of tenor knots (years) for the zero curve
    z_pct    : array of continuously compounded zero rates in %
    rf_pct   : funding rate in % (3M OIS par rate from same date)
    tenors   : list of swap tenors in years (e.g. [2, 5, 10])
    horizons : dict of {label: h_years}

    Returns
    -------
    list of dicts, one per (tenor, horizon) pair
    """
    rows = []
    max_z = float(z_tenors[-1])

    for T in tenors:
        if T > max_z:
            continue

        K_T = _par_rate_pct(z_tenors, z_pct, T)       # entry par rate (%)
        ann_T = _annuity(z_tenors, z_pct, T)           # annuity at T (years)
        dv01 = ann_T * 1e-4                            # DV01 per unit notional

        for h_label, h in horizons.items():
            if T - h < 1.0 / OIS_COUPON_FREQ:         # would age below one coupon period
                continue

            ann_aged = _aged_annuity(z_tenors, z_pct, T, h)
            K_Tmh = _par_rate_pct(z_tenors, z_pct, T - h)  # par rate at T-h (%)

            # OIS-0: roll proxy (bp per unit notional)
            # = A(T-h,aged) × (K(T) - K(T-h)) × 100   [K in %, × 100 → bp × annuity]
            roll_bp = ann_aged * (K_T - K_Tmh) * 100.0

            # OIS-1: static carry = roll - funding (bp per unit notional)
            funding_bp = rf_pct * h * 100.0            # bp
            static_bp = roll_bp - funding_bp

            rows.append({
                "tenor":            T,
                "horizon":          h_label,
                "par_rate_pct":     K_T,
                "rf_pct":           rf_pct,
                "annuity":          ann_T,
                "aged_annuity":     ann_aged,
                "dv01":             dv01,
                "roll_proxy_bp":    roll_bp,
                "static_carry_bp":  static_bp,
                "carry_dv01_bp":    static_bp / (dv01 * 1e4) if dv01 > 1e-12 else np.nan,
            })

    return rows


# ── Panel computation (one market) ───────────────────────────────────────────

def ois_carry_panel(
    market: str,
    *,
    tenors: Optional[list[float]] = None,
    horizons: Optional[dict[str, float]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    zero_method: str = "monotone_preserving",
    zscore_lookback: int = ZSCORE_LOOKBACK,
    zero_curve: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute OIS carry signal panel for one market.

    Parameters
    ----------
    market   : OIS market name (from OIS_MARKETS)
    tenors   : swap tenors in years; default BENCHMARK_TENORS
    horizons : {label: h_years}; default HORIZONS
    start, end : date filter strings
    zero_method : interpolation method for yc_builder.build_zero_curve
    zscore_lookback : lookback window for carry_zscore (trailing obs)

    Returns
    -------
    Long-format DataFrame indexed by (date, tenor, horizon) with columns:
      par_rate_pct, rf_pct, annuity, aged_annuity, dv01,
      roll_proxy_bp, static_carry_bp, carry_dv01_bp, carry_zscore

    Notes
    -----
    - carry_zscore uses trailing-only mean/std; first zscore_lookback rows NaN
    - All carry values for *receiver* OIS (positive = receiver earns in Scenario S)
    """
    if tenors is None:
        tenors = BENCHMARK_TENORS
    if horizons is None:
        horizons = HORIZONS

    par = fetch_ois_curves(market, start=start, end=end)

    if zero_curve is not None:
        zero = zero_curve
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            zero = build_zero_curve(par, method=zero_method)

    # Funding rate: 3M (tenor 0.25) from par curve; fall back to nearest short-end zero
    funding_tenor = 0.25
    if funding_tenor in par.columns:
        rf_series = par[funding_tenor]
    else:
        nearest = min(par.columns, key=lambda x: abs(x - funding_tenor))
        rf_series = par[nearest]
        warnings.warn(
            f"{market}: 3M OIS not found; using tenor {nearest:.3f}Y as funding proxy"
        )

    # ── Batch discount-factor matrix (vectorised across all dates) ───────────
    z_np   = zero.to_numpy(float)               # (D, 70)
    z_cols = zero.columns.to_numpy(float)       # TARGET_TENORS
    D      = len(zero)

    def _cpn_dates(tenor: float) -> list:
        """Coupon dates for a swap of given tenor (freq=OIS_COUPON_FREQ, last snapped)."""
        n = max(1, round(tenor * OIS_COUPON_FREQ))
        cd = [j / OIS_COUPON_FREQ for j in range(1, n + 1)]
        cd[-1] = tenor
        return cd

    # Collect every unique time-point needed across all (T, h) combinations
    _tpts_set: set = set()
    for T in tenors:
        for cd in _cpn_dates(T):
            _tpts_set.add(round(cd, 9))
        for h in horizons.values():
            if T - h < 1.0 / OIS_COUPON_FREQ:
                continue
            for cd in _cpn_dates(T - h):           # for K_Tmh annuity
                _tpts_set.add(round(cd, 9))
            for cd in _cpn_dates(T):               # for aged annuity
                if cd - h > 1e-6:
                    _tpts_set.add(round(cd - h, 9))
    _tpts    = np.array(sorted(_tpts_set))
    _tpt_idx = {v: i for i, v in enumerate(_tpts)}

    def _key(x: float) -> float:
        return round(x, 9)

    # (D, K) discount factor matrix via vectorised linear interp on zero curve
    batch_df = np.ones((D, len(_tpts)))
    for k, tp in enumerate(_tpts):
        if tp <= 0:
            continue
        ir = int(np.clip(np.searchsorted(z_cols, tp), 1, len(z_cols) - 1))
        il = ir - 1
        w  = ((tp - z_cols[il]) / (z_cols[ir] - z_cols[il])
              if z_cols[ir] != z_cols[il] else 0.0)
        z_i = (1 - w) * z_np[:, il] + w * z_np[:, ir]     # (D,)
        batch_df[:, k] = np.exp(-z_i / 100.0 * tp)

    # Align rf to zero.index
    rf_vec     = rf_series.reindex(zero.index).to_numpy(float)  # (D,)
    valid_mask = (np.sum(np.isfinite(z_np), axis=1) >= 6) & np.isfinite(rf_vec)

    all_frames = []
    max_z = float(z_cols[-1])

    def _ann(dates_list: list) -> np.ndarray:
        """Sum of DFs for dates_list, divided by freq. Returns (D,) array."""
        return sum(batch_df[:, _tpt_idx[_key(cd)]] for cd in dates_list) / OIS_COUPON_FREQ

    for T in tenors:
        if T > max_z:
            continue
        cpn_T = _cpn_dates(T)
        ann_T = _ann(cpn_T)
        df_T  = batch_df[:, _tpt_idx[_key(T)]]
        K_T   = np.where(ann_T > 1e-10, (1.0 - df_T) / ann_T * 100.0, np.nan)
        dv01  = ann_T * 1e-4

        for h_label, h in horizons.items():
            T_mh = T - h
            if T_mh < 1.0 / OIS_COUPON_FREQ:
                continue
            cpn_Tmh  = _cpn_dates(T_mh)
            aged_cpn = [cd - h for cd in cpn_T if cd - h > 1e-6]
            if not aged_cpn:
                continue

            ann_Tmh  = _ann(cpn_Tmh)
            df_Tmh   = batch_df[:, _tpt_idx[_key(T_mh)]]
            K_Tmh    = np.where(ann_Tmh > 1e-10, (1.0 - df_Tmh) / ann_Tmh * 100.0, np.nan)

            ann_aged = _ann(aged_cpn)

            roll_bp    = ann_aged * (K_T - K_Tmh) * 100.0
            funding_bp = rf_vec * h * 100.0
            static_bp  = roll_bp - funding_bp

            frame = pd.DataFrame({
                "tenor":           T,
                "horizon":         h_label,
                "par_rate_pct":    K_T,
                "rf_pct":          rf_vec,
                "annuity":         ann_T,
                "aged_annuity":    ann_aged,
                "dv01":            dv01,
                "roll_proxy_bp":   roll_bp,
                "static_carry_bp": static_bp,
                "carry_dv01_bp":   np.where(dv01 * 1e4 > 1e-12,
                                            static_bp / (dv01 * 1e4), np.nan),
            }, index=zero.index)
            all_frames.append(frame.loc[valid_mask])

    if not all_frames:
        return pd.DataFrame()

    df = (pd.concat(all_frames)
            .reset_index()
            .rename(columns={"index": "date"})
            .set_index(["date", "tenor", "horizon"])
            .sort_index())

    # OIS-5: trailing z-score of static_carry_bp per (tenor, horizon) group
    def _zscore(s: pd.Series) -> pd.Series:
        mu = s.expanding(min_periods=252).mean().shift(1)
        sd = s.expanding(min_periods=252).std(ddof=1).shift(1)
        return (s - mu) / sd.clip(lower=1e-12)

    zs = (
        df["static_carry_bp"]
        .groupby(level=["tenor", "horizon"])
        .transform(_zscore)
    )
    df["carry_zscore"] = zs

    return df


# ── Multi-market wrapper ──────────────────────────────────────────────────────

def ois_carry_all_markets(
    markets: Optional[list[str]] = None,
    *,
    tenors: Optional[list[float]] = None,
    horizons: Optional[dict[str, float]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    zero_method: str = "monotone_preserving",
    zero_curves: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Compute OIS carry for all (or selected) markets.

    Returns long-format DataFrame with an additional 'market' level in the index.
    """
    if markets is None:
        markets = OIS_MARKETS

    frames = []
    for mkt in markets:
        try:
            panel = ois_carry_panel(
                mkt, tenors=tenors, horizons=horizons,
                start=start, end=end, zero_method=zero_method,
                zero_curve=zero_curves.get(mkt) if zero_curves else None,
            )
            if panel.empty:
                continue
            panel.index = pd.MultiIndex.from_arrays(
                [
                    [mkt] * len(panel),
                    panel.index.get_level_values("date"),
                    panel.index.get_level_values("tenor"),
                    panel.index.get_level_values("horizon"),
                ],
                names=["market", "date", "tenor", "horizon"],
            )
            frames.append(panel)
        except Exception as exc:
            warnings.warn(f"ois_carry_panel failed for {mkt}: {exc}")

    return pd.concat(frames) if frames else pd.DataFrame()
