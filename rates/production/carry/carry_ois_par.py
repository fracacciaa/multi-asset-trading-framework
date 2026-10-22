"""
OIS within-curve carry — §6.2 par-rate approximation.

Mirrors carry_bonds.bond_carry_panel() exactly but uses OIS par rates
(SWAPPPPP RATES.xlsx via ois_data.fetch_ois_curves) instead of government
bond par yields. Annual coupon convention (freq=1) for all OIS markets.

ExcessReturn_h ≈ (K_t(T) - r_fund) × h  +  D_mod × [K_t(T) - K_t(T-h)]

  K_t(T)   : OIS par rate at tenor T (%)
  r_fund   : 3M OIS rate from the same curve (%)
  h        : horizon in years (1M, 3M, 6M)
  D_mod    : modified duration of an annual par bond at yield K_t(T), tenor T
  K_t(T-h) : OIS par rate at T-h, linearly interpolated from today's curve

All carry values in bp. Positive = receiver earns carry under the static-curve
scenario (unchanged curve over h).

Design notes
------------
- No NS fitting, no zero-curve bootstrap — uses raw OIS par rates directly.
  This makes carry_ois_par directly comparable to carry_bonds (same methodology,
  different instrument), while carry_ois.py (bootstrapped zeros) is the
  more precise but NS-free version for OIS.
- freq=1 (annual coupon) for all OIS markets: consistent with OIS conventions
  in carry_ois.py and ois_data.py.
- Funding = 3M OIS (tenor 0.25) from the same market's par curve.
- Trailing z-score (carry_zscore) uses expanding window, shifted by 1 step
  to prevent lookahead; identical to carry_bonds convention.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import Optional

from rates.production.ois_data import fetch_ois_curves, OIS_MARKETS
from rates.production.carry.carry_bonds import _dmod_par_vec, _interp_at_tenor_vec

# ── Defaults ──────────────────────────────────────────────────────────────────

OIS_PAR_MARKETS: list[str] = [
    "US", "EUR", "UK", "Japan", "Canada", "Australia", "New Zealand", "Sweden", "Switzerland",
]

BENCHMARK_TENORS: list[float] = [5.0, 10.0]
HORIZONS: dict[str, float]    = {"1M": 1 / 12, "3M": 3 / 12, "6M": 6 / 12}
OIS_COUPON_FREQ: int           = 1       # annual for all OIS markets
ZSCORE_LOOKBACK: int           = 756     # ~3 years; min 252


# ── Panel computation ─────────────────────────────────────────────────────────

def ois_par_carry_panel(
    market: str,
    *,
    tenors: Optional[list[float]] = None,
    horizons: Optional[dict[str, float]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    zscore_lookback: int = ZSCORE_LOOKBACK,
) -> pd.DataFrame:
    """
    Compute OIS within-curve carry using the §6.2 par-rate approximation.

    Parameters
    ----------
    market   : OIS market name (from OIS_PAR_MARKETS)
    tenors   : swap tenors in years; default BENCHMARK_TENORS
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
    - Approximation: off-par or path-dependent carry (e.g. OIS compounding
      vs. simple accrual) is ignored.  The formula is exact for a par annual
      fixed-rate receiver under an unchanged-curve scenario.
    - Linear interpolation along the tenor axis (identical to carry_bonds).
    - carry_zscore is trailing-only (mu/sigma shifted by 1 step).
    """
    if tenors is None:
        tenors = BENCHMARK_TENORS
    if horizons is None:
        horizons = HORIZONS

    ois_df = fetch_ois_curves(market, start=start, end=end)
    ois_df = ois_df.sort_index(axis=1)

    # Funding: 3M OIS from the same curve
    funding_tenor = 0.25
    if funding_tenor in ois_df.columns:
        rf_series = ois_df[funding_tenor]
    else:
        nearest = min(ois_df.columns, key=lambda x: abs(x - funding_tenor))
        rf_series = ois_df[nearest]
        warnings.warn(
            f"{market}: 3M OIS not found; using tenor {nearest:.3f}Y as funding proxy"
        )

    # Valid (T, h_label, h) — skip combos where T-h < 1/freq (would roll below 1 coupon)
    valid_combos = [
        (T, h_label, h)
        for T in tenors
        for h_label, h in horizons.items()
        if T - h >= 1.0 / OIS_COUPON_FREQ
    ]

    all_targets = sorted(set(
        [T for T, _, _ in valid_combos] +
        [T - h for T, _, h in valid_combos]
    ))

    # One vectorised interpolation pass per unique target tenor
    y_cache: dict[float, pd.Series] = {
        t: _interp_at_tenor_vec(ois_df, t) for t in all_targets
    }

    rf_aligned = rf_series.reindex(ois_df.index, method="ffill", limit=5)
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
        dates_v = ois_df.index[valid]

        dmod      = _dmod_par_vec(y_T_v, T, freq=OIS_COUPON_FREQ)
        income_bp = (y_T_v - rf_v) / 100.0 * h * 1e4
        roll_bp   = dmod * (y_T_v - y_Tmh_v) / 100.0 * 1e4
        total_bp  = income_bp + roll_bp
        carry_dm  = np.where(dmod > 1e-6, total_bp / dmod, np.nan)

        frames.append(pd.DataFrame({
            "tenor":          T,
            "horizon":        h_label,
            "yield_pct":      y_T_v,
            "rf_pct":         rf_v,
            "dmod":           dmod,
            "income_bp":      income_bp,
            "roll_bp":        roll_bp,
            "total_carry_bp": total_bp,
            "carry_dmod_bp":  carry_dm,
        }, index=pd.Index(dates_v, name="date")))

    if not frames:
        return pd.DataFrame()

    df = (pd.concat(frames)
            .reset_index()
            .set_index(["date", "tenor", "horizon"])
            .sort_index())

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


# ── Per-market PnL (positions × daily mark-to-market) ────────────────────────

def ois_par_carry_pnl(
    markets: Optional[list[str]] = None,
    *,
    tenors: Optional[list[float]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Daily PnL for the within-curve carry OIS signal.

    For each market: build DV01-neutral carry positions (daily rebalance, lag=1),
    then mark to market each day via the §6.2 return approximation:

      ret_t(T) = pos_t(T) × [ (y_{t-1}(T) - rf_{t-1}) × 100/252
                               - D_mod(T) × (y_t(T) - y_{t-1}(T)) × 100 ]  bp

    Returns
    -------
    date × market DataFrame of daily PnL in bp.
    """
    from rates.production.carry.carry_bonds import _dmod_par_vec, _interp_at_tenor_vec
    from rates.production.carry.signal import build_positions

    if markets is None:
        markets = OIS_PAR_MARKETS
    if tenors is None:
        tenors = BENCHMARK_TENORS

    mkt_series: dict[str, pd.Series] = {}
    for market in markets:
        try:
            panel = ois_par_carry_panel(market, tenors=tenors,
                                        horizons={"3M": 0.25},
                                        start=start, end=end)
            if panel.empty:
                continue

            sub        = panel.xs("3M", level="horizon")
            carry_wide = sub["total_carry_bp"].unstack("tenor")
            dmod_wide  = sub["dmod"].unstack("tenor")

            pos_df = build_positions(carry_wide, dmod_wide,
                                     rebalance="daily", lag=1,
                                     xs_mode="z", neutralize=("level",),
                                     gross=1.0, duration_adj="per_dur")

            ois_df     = fetch_ois_curves(market, start=start, end=end)
            fund_col   = (0.25 if 0.25 in ois_df.columns
                          else min(ois_df.columns, key=lambda x: abs(x - 0.25)))
            rf_aligned = ois_df[fund_col].reindex(ois_df.index, method="ffill", limit=5)
            y_cache    = {T: _interp_at_tenor_vec(ois_df, T) for T in tenors}
            pos_ri     = pos_df.reindex(ois_df.index, method="ffill")

            common = (ois_df.index
                      .intersection(pos_ri.index)
                      .intersection(rf_aligned.dropna().index)
                      .sort_values())

            daily: dict = {}
            for i in range(1, len(common)):
                date, prev = common[i], common[i - 1]
                pos = pos_ri.loc[date]
                if pos.isna().all():
                    continue
                day = 0.0
                for T in tenors:
                    if T not in pos.index or np.isnan(pos[T]):
                        continue
                    yp = y_cache[T]
                    if prev not in yp.index or date not in yp.index:
                        continue
                    y_prev = float(yp.loc[prev])
                    y_now  = float(yp.loc[date])
                    if prev not in rf_aligned.index:
                        continue
                    rf_p = float(rf_aligned.loc[prev])
                    if any(np.isnan([y_prev, y_now, rf_p])):
                        continue
                    d = float(_dmod_par_vec(np.array([y_prev]), T,
                                            freq=OIS_COUPON_FREQ)[0])
                    day += pos[T] * ((y_prev - rf_p) * 100.0 / 252.0
                                     - d * (y_now - y_prev) * 100.0)
                daily[date] = day

            s = pd.Series(daily)
            s.index = pd.to_datetime(s.index)
            mkt_series[market] = s
        except Exception as exc:
            warnings.warn(f"ois_par_carry_pnl failed for {market}: {exc}")

    if not mkt_series:
        return pd.DataFrame()

    df = pd.DataFrame(mkt_series)
    df.index.name = "date"
    return df


# ── Multi-market wrapper ──────────────────────────────────────────────────────

def ois_par_carry_all_markets(
    markets: Optional[list[str]] = None,
    *,
    tenors: Optional[list[float]] = None,
    horizons: Optional[dict[str, float]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute OIS par carry for all (or selected) markets.

    Returns long-format DataFrame with an additional 'market' index level.
    """
    if markets is None:
        markets = OIS_PAR_MARKETS

    frames = []
    for mkt in markets:
        try:
            panel = ois_par_carry_panel(
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
            warnings.warn(f"ois_par_carry_panel failed for {mkt}: {exc}")

    return pd.concat(frames) if frames else pd.DataFrame()
