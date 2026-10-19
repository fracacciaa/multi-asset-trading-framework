"""
Government bond carry signals — E08, E09 from rates_carry_research_implementation.md.

Uses Bloomberg generic par yields (RATESSSSS.xlsx via rates/data.py).
Funding rate from the corresponding OIS market (ois_data.py).

Section 6.2 approximation (exact repricing requires contractual cash-flows
and repo; this module implements the documented approximation, not full
dirty-price accounting):

  ExcessReturn_h ≈ (y_t(T) - r_fund) × h  +  D_mod × [y_t(T) - y_t(T-h)]

  where
    y_t(T)   : par yield at tenor T from benchmark sovereign curve (%)
    r_fund   : funding rate (%) — 3M OIS from the matched currency
    h        : horizon in years (1/12, 3/12, 6/12)
    D_mod    : modified duration ≈ par-bond formula below
    y_t(T-h) : par yield at T-h, interpolated from today's curve

All carry values in bp.
Positive = long bond earns carry above funding under static-curve scenario.

For E09 (within-country maturity-relative carry), output also includes
  carry_dmod_bp : carry / D_mod  [per-unit-duration carry, normalizes maturity bias]

Funding market map:
  US → "US" (OIS), Germany/France → "EUR", UK → "UK",
  Japan → "Japan", Canada → "Canada", Australia → "Australia"

Bond coupon frequency (for D_mod computation only):
  US, UK, Canada, Australia → 2 (semi-annual)
  Germany, France, Japan    → 1 (annual)
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import Optional

from rates.data import load_bloomberg_excel
from rates.ois_data import fetch_ois_curves

# ── Configuration ─────────────────────────────────────────────────────────────

_BOND_PATH = "/Users/francescocaccia/Downloads/RATESSSSS.xlsx"

BOND_MARKETS: list[str] = ["US", "Germany", "UK", "Japan", "France", "Canada", "Australia"]

HORIZONS: dict[str, float] = {"1M": 1 / 12, "3M": 3 / 12, "6M": 6 / 12}

BENCHMARK_TENORS: list[float] = [2.0, 5.0, 10.0, 20.0, 30.0]

# Map bond market → OIS funding market
_FUNDING_MAP: dict[str, str] = {
    "US":        "US",
    "Germany":   "EUR",
    "UK":        "UK",
    "Japan":     "Japan",
    "France":    "EUR",
    "Canada":    "Canada",
    "Australia": "Australia",
}

# Coupon frequency per market (used only for D_mod)
_COUPON_FREQ: dict[str, int] = {
    "US":        2,
    "Germany":   1,
    "UK":        2,
    "Japan":     2,
    "France":    1,
    "Canada":    2,
    "Australia": 2,
}

ZSCORE_LOOKBACK: int = 756  # ~3 years


# ── Duration approximation (vectorized) ──────────────────────────────────────

def _dmod_par_vec(y_pct_arr: np.ndarray, tenor: float, freq: int = 2) -> np.ndarray:
    """
    Modified duration of a par bond — vectorized over an array of yields.

      D_mod = [(1 - (1 + y/freq)^{-n}) / (y/freq)] / freq  /  (1 + y/freq)
              where n = round(tenor × freq)

    For y ≈ 0 (near-ZIRP), returns tenor as limit.
    """
    y = np.asarray(y_pct_arr, dtype=float) / 100.0
    n = max(1, round(tenor * freq))
    y_period = y / freq
    near_zero = y < 1e-6
    with np.errstate(divide='ignore', invalid='ignore'):
        mac_periods = np.where(
            near_zero,
            float(n),
            (1.0 - (1.0 + y_period) ** (-n)) / y_period,
        )
    mac_years = mac_periods / freq
    return np.where(near_zero, tenor, mac_years / (1.0 + y_period))


# ── Vectorized par yield interpolation ───────────────────────────────────────

def _interp_at_tenor_vec(bond_df: pd.DataFrame, target: float) -> pd.Series:
    """
    Linear interpolation of par yield at `target` tenor for ALL dates at once.

    `target` is clipped to [t_min, t_max] of the available tenor grid.
    Returns a Series aligned to bond_df.index.
    NaN propagates from the two bracketing columns (correct: missing data → NaN carry).
    """
    cols = bond_df.columns.to_numpy(float)
    t_min, t_max = cols[0], cols[-1]
    t = float(np.clip(target, t_min, t_max))

    idx = int(np.searchsorted(cols, t, side="left"))

    if idx == 0 or t <= t_min:
        return bond_df.iloc[:, 0].copy().rename(target)
    if idx >= len(cols) or t >= t_max:
        return bond_df.iloc[:, -1].copy().rename(target)

    t_lo, t_hi = cols[idx - 1], cols[idx]
    if t_hi == t_lo:
        return bond_df.iloc[:, idx - 1].copy().rename(target)

    alpha = (t - t_lo) / (t_hi - t_lo)
    y_lo = bond_df.iloc[:, idx - 1].values
    y_hi = bond_df.iloc[:, idx].values
    result = y_lo + alpha * (y_hi - y_lo)
    return pd.Series(result, index=bond_df.index, name=target)


# ── Panel computation ─────────────────────────────────────────────────────────

def bond_carry_panel(
    market: str,
    *,
    tenors: Optional[list[float]] = None,
    horizons: Optional[dict[str, float]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    zscore_lookback: int = ZSCORE_LOOKBACK,
) -> pd.DataFrame:
    """
    Compute government bond carry panel for one market.

    Parameters
    ----------
    market   : bond market name — one of BOND_MARKETS
    tenors   : maturity buckets in years; default BENCHMARK_TENORS
    horizons : {label: h_years}; default HORIZONS
    start, end : date filters
    zscore_lookback : lookback window for carry_zscore

    Returns
    -------
    Long-format DataFrame indexed by (date, tenor, horizon) with columns:
      yield_pct, rf_pct, dmod, income_bp, roll_bp, total_carry_bp,
      carry_dmod_bp, carry_zscore

    Notes
    -----
    - Approximation per §6.2: off-par, high-coupon, or repo-special bonds
      can deviate materially from this estimate.
    - Funding = 3M OIS from the matched currency in _FUNDING_MAP.
    - carry_zscore is trailing-only (shifts mu/sigma by 1 step).
    - Interpolation is linear along the tenor axis for each date (vectorized
      across all dates simultaneously); target tenors are clipped to
      [t_min, t_max] of the available knots.
    """
    if tenors is None:
        tenors = BENCHMARK_TENORS
    if horizons is None:
        horizons = HORIZONS

    # Bond yields — sort columns ascending by tenor
    all_bonds = load_bloomberg_excel(_BOND_PATH)
    if market not in all_bonds:
        raise ValueError(f"Bond market {market!r} not found. Available: {list(all_bonds)}")
    bond_df = all_bonds[market].sort_index(axis=1)
    if start:
        bond_df = bond_df.loc[start:]
    if end:
        bond_df = bond_df.loc[:end]

    # Funding from matched OIS market (3M tenor)
    ois_mkt = _FUNDING_MAP.get(market, market)
    try:
        ois_par = fetch_ois_curves(ois_mkt, start=start, end=end)
        funding_tenor = 0.25
        if funding_tenor in ois_par.columns:
            rf_series = ois_par[funding_tenor]
        else:
            nearest = min(ois_par.columns, key=lambda x: abs(x - funding_tenor))
            rf_series = ois_par[nearest]
    except Exception:
        warnings.warn(f"{market}: OIS funding unavailable; using short-end bond yield as proxy")
        short = min(bond_df.columns)
        rf_series = bond_df[short]

    coupon_freq = _COUPON_FREQ.get(market, 2)

    # Valid (T, h_label, h) combos — drop combos where T-h < 0.25
    valid_combos = [
        (T, h_label, h)
        for T in tenors
        for h_label, h in horizons.items()
        if T - h >= 0.25
    ]

    # All unique target tenors needed across all combos
    all_targets = sorted(set(
        [T for T, _, _ in valid_combos] +
        [T - h for T, _, h in valid_combos]
    ))

    # Vectorized interpolation — ONE pass per unique target tenor (not per date)
    y_cache: dict[float, pd.Series] = {t: _interp_at_tenor_vec(bond_df, t) for t in all_targets}

    # Align funding to bond date index (forward-fill up to 5 business days)
    rf_aligned = rf_series.reindex(bond_df.index, method="ffill", limit=5)
    mask_rf = rf_aligned.notna()

    frames: list[pd.DataFrame] = []
    for T, h_label, h in valid_combos:
        T_minus_h = T - h
        y_T   = y_cache[T]
        y_Tmh = y_cache[T_minus_h]

        valid = y_T.notna() & y_Tmh.notna() & mask_rf
        if not valid.any():
            continue

        y_T_v   = y_T[valid].values
        y_Tmh_v = y_Tmh[valid].values
        rf_v    = rf_aligned[valid].values
        dates_v = bond_df.index[valid]

        dmod       = _dmod_par_vec(y_T_v, T, freq=coupon_freq)
        income_bp  = (y_T_v - rf_v) / 100.0 * h * 1e4
        roll_bp    = dmod * (y_T_v - y_Tmh_v) / 100.0 * 1e4
        total_bp   = income_bp + roll_bp
        carry_dmod = np.where(dmod > 1e-6, total_bp / dmod, np.nan)

        frames.append(pd.DataFrame({
            "tenor":          T,
            "horizon":        h_label,
            "yield_pct":      y_T_v,
            "rf_pct":         rf_v,
            "dmod":           dmod,
            "income_bp":      income_bp,
            "roll_bp":        roll_bp,
            "total_carry_bp": total_bp,
            "carry_dmod_bp":  carry_dmod,
        }, index=pd.Index(dates_v, name="date")))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames).reset_index()
    df = df.set_index(["date", "tenor", "horizon"]).sort_index()

    # Trailing z-score per (tenor, horizon) group
    def _zscore(s: pd.Series) -> pd.Series:
        mu = s.expanding(min_periods=252).mean().shift(1)
        sd = s.expanding(min_periods=252).std(ddof=1).shift(1)
        return (s - mu) / sd.clip(lower=1e-12)

    df["carry_zscore"] = (
        df["total_carry_bp"]
        .groupby(level=["tenor", "horizon"])
        .transform(_zscore)
    )

    return df


# ── Multi-market wrapper ──────────────────────────────────────────────────────

def bond_carry_all_markets(
    markets: Optional[list[str]] = None,
    *,
    tenors: Optional[list[float]] = None,
    horizons: Optional[dict[str, float]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute bond carry for all (or selected) markets.

    Returns long-format DataFrame with an additional 'market' index level.
    """
    if markets is None:
        markets = BOND_MARKETS

    frames = []
    for mkt in markets:
        try:
            panel = bond_carry_panel(
                mkt, tenors=tenors, horizons=horizons, start=start, end=end
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
            warnings.warn(f"bond_carry_panel failed for {mkt}: {exc}")

    return pd.concat(frames) if frames else pd.DataFrame()
