"""
Rates signals derived from Nelson-Siegel / Nelson-Siegel-Svensson factors.

All signals are cross-market (date × market DataFrames), each market treated
independently (ViewFormationTS). Positive signal → long duration (receive rates).

Four signals:
  1. Momentum  — Signal A daily — per-(market,tenor) TSMOM, 252-day window,
                 inverse-vol sized; independent bet per tenor, all OIS markets
  2. Carry     — NS_fixed  — −β₂ (slope factor)
  3. MeanRev   — ex_raw RV — factor-neutral NS residuals, per-tenor EWMA z-score
  4. Carry CC  — cross-country DV01-neutral carry (DM only)

The mean-reversion signal uses factor-neutral NS residuals (ex_raw): each tenor's
residual is projected onto the complement of the DV01-weighted NS factor space,
then EWMA z-scored. The per-market scalar is the mean of per-tenor z-scores,
representing aggregate factor-neutral cheapness of the curve.
Lambda is calibrated once on the training window (≤ RV_TRAIN_END).

The momentum signal uses daily CM returns (ret_d) for both the formation window
and vol estimation; positions are inverse-vol sized (MOP spec). Each (market,tenor)
pair is a fully independent bet; no cross-tenor averaging.
"""

import numpy as np
import pandas as pd
from typing import Optional

from rates.production.nelson_siegel import extract_factors_all_markets
from rates.production.ois_data import OIS_MARKETS, fetch_ois_curves
from rates.production.mean_reversion.signal import mean_reversion_rv

# ── Cross-country carry constants ──────────────────────────────────────────────

CC_LAM             = 0.3
CC_BASIS           = "ns"
CC_TAU_TARGET      = 10.0
CC_VOL_LOOKBACK    = 252
CC_MIN_OBS         = 63
CC_MIN_MARKETS     = 4
CC_LAG             = 1


# ── Signal functions ───────────────────────────────────────────────────────────

def _get_factor(factors: pd.DataFrame, factor_name: str) -> pd.DataFrame:
    """Extract a single factor across all markets as a date × market DataFrame."""
    return factors.xs(factor_name, axis=1, level="factor")


def momentum(
    factors: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
) -> pd.DataFrame:
    """
    Time-series momentum on β₁ (level), 12-1m skip=21.

    Signal = β₁.shift(1+skip) − β₁.shift(lookback+1+skip)   [absolute bp change]
    """
    level = _get_factor(factors, "level")
    return (level.shift(1 + skip) - level.shift(lookback + 1 + skip)).rename_axis("date")


def carry(factors: pd.DataFrame) -> pd.DataFrame:
    """
    Carry from roll-down: −β₂ (slope factor from NS).

    Normal (upward-sloping) curve: β₂ < 0 → −β₂ > 0 → positive carry.
    Inverted curve: β₂ > 0 → −β₂ < 0 → negative carry.
    """
    slope = _get_factor(factors, "slope")
    return (-slope).rename_axis("date")


def momentum_a_daily(
    markets: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    tenors: Optional[list[float]] = None,
    k: int = 252,
) -> pd.DataFrame:
    """
    Signal A daily — MOP TSMOM, independent inverse-vol-sized bet per (market, tenor).

    pos[(market, tenor), t] = sign( Σ_{i=1}^{k} ret_d[t-i] ) × (TARGET_VOL / σ_t)

    Uses daily CM returns (ret_d) for both formation window and vol estimation.
    Vol floor at 10% of TARGET_VOL to guard against sparse-data periods.
    Defaults to all OIS markets and primary tenors {2, 5, 10}Y.

    Returns a date × (market, tenor) MultiIndex DataFrame of daily position sizes.
    """
    from rates.production.momentum.returns import build_return_panels
    from rates.production.momentum.signal_a import build_signal_a_daily, TARGET_VOL_BP

    if markets is None:
        markets = OIS_MARKETS
    if tenors is None:
        tenors = [2.0, 5.0, 10.0]

    panels_by_market: dict = {}
    for market in markets:
        try:
            panels_by_market[market] = build_return_panels(
                market, start=start, end=end, tenors=tenors,
            )
        except Exception:
            continue

    sig = build_signal_a_daily(panels_by_market, tenors=tenors, k=k)

    # Cap positions at 10× (equivalent to a 10% TARGET_VOL floor on sigma)
    max_pos = TARGET_VOL_BP / (TARGET_VOL_BP * 0.1)   # = 10.0
    return sig.clip(lower=-max_pos, upper=max_pos)


def carry_cross_country(
    markets: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Cross-country carry position book (DV01-neutral, CC DV01-only variant).

    For each DM market: fit NS daily, compute carry at 10Y via carry_reprice,
    divide by duration to get yield-space carry.  Then cross-section across
    countries: z-score, inverse-vol size, zero-net, DV01-project.

    Returns
    -------
    date × market DataFrame of position weights.
    Positions are ready-to-trade: inverse-vol sized, zero-net, DV01-neutral,
    normalised to gross = 1.0, lagged by 1 day.  Positive = receive fixed
    (long duration); negative = pay fixed (short duration).
    """
    from rates.production.ns_core import fit_ns, fit_ns_batch
    from rates.production.carry.cross_country import (
        country_carry_at_target, build_cross_country_positions, DM_MARKETS,
    )
    from rates.production.carry.experiment import BASELINE

    if markets is None:
        markets = DM_MARKETS

    curves_dict: dict = {}
    funding_dict: dict = {}
    fits_dict: dict = {}

    for mkt in markets:
        try:
            c = fetch_ois_curves(mkt, start=start, end=end)
            col = 0.25 if 0.25 in c.columns else min(c.columns, key=lambda x: abs(x - 0.25))
            curves_dict[mkt]  = c
            funding_dict[mkt] = c[col]

            tau_all = c.columns.to_numpy(float)
            Y_all   = c.values * 100.0               # % → bp, shape (T, N)
            full_rows   = np.all(np.isfinite(Y_all), axis=1)
            enough_rows = np.sum(np.isfinite(Y_all), axis=1) >= 4

            fits: dict = {}

            # Batch the dates where all tenors are valid (vast majority)
            if full_rows.any():
                for date, fit in zip(
                    c.index[full_rows],
                    fit_ns_batch(tau_all, Y_all[full_rows], lam=CC_LAM, basis=CC_BASIS),
                ):
                    fits[date] = fit

            # Per-row fallback for partial dates (some tenors NaN)
            for i in np.where(enough_rows & ~full_rows)[0]:
                row = c.iloc[i].dropna()
                if len(row) < 4:
                    continue
                try:
                    fits[c.index[i]] = fit_ns(
                        row.index.to_numpy(float),
                        row.to_numpy(float) * 100.0,
                        lam=CC_LAM, basis=CC_BASIS,
                    )
                except Exception:
                    pass

            if len(fits) >= CC_MIN_OBS:
                fits_dict[mkt] = fits
        except Exception:
            pass

    cc_series: dict = {}
    for mkt in markets:
        if mkt not in fits_dict:
            continue
        rf_bp = funding_dict[mkt] * 100.0
        carry_ts = country_carry_at_target(
            fits_dict[mkt], curves_dict[mkt], rf_bp, CC_TAU_TARGET, BASELINE,
        )
        if len(carry_ts) > 0:
            cc_series[mkt] = carry_ts

    if len(cc_series) < 2:
        return pd.DataFrame()

    country_carry_df = pd.DataFrame(cc_series).sort_index()

    pos_df = build_cross_country_positions(
        country_carry_df,
        curves_dict,
        tau_vol_ref=CC_TAU_TARGET,
        universe="DM",
        vol_lookback=CC_VOL_LOOKBACK,
        min_obs=CC_MIN_OBS,
        fits_dict=fits_dict,
        global_beta_neutralize=False,
    )

    valid = pos_df.notna().sum(axis=1) >= CC_MIN_MARKETS
    pos_df = pos_df.where(valid)
    gross  = pos_df.abs().sum(axis=1).replace(0, np.nan)
    pos_df = pos_df.div(gross, axis=0).shift(CC_LAG)
    pos_df.index.name = "date"
    return pos_df


def within_curve_carry_ois(
    markets: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    tenors: Optional[list[float]] = None,
) -> pd.DataFrame:
    """
    Within-curve carry OIS position book — §6.2 par-rate approximation.

    Daily rebalance, 1-day lag, DV01-neutral cross-section of 5Y and 10Y.

    Returns date × (market, tenor) MultiIndex DataFrame of position weights.
    """
    from rates.production.carry.carry_ois_par import ois_par_carry_panel, OIS_PAR_MARKETS
    from rates.production.carry.signal import build_positions

    if markets is None:
        markets = OIS_PAR_MARKETS
    if tenors is None:
        tenors = [5.0, 10.0]

    mkt_frames: dict = {}
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
            pos = build_positions(carry_wide, dmod_wide,
                                  rebalance="daily", lag=1,
                                  xs_mode="z", neutralize=("level",),
                                  gross=1.0, duration_adj="per_dur")
            mkt_frames[market] = pos
        except Exception:
            continue

    if not mkt_frames:
        return pd.DataFrame()

    combined = pd.concat(mkt_frames, axis=1)
    combined.columns.names = ["market", "tenor"]
    return combined


def build_signals(
    markets: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    mom_tenors: Optional[list[float]] = None,
) -> dict[str, pd.DataFrame]:
    """
    Run the full rates signal pipeline.

    Returns
    -------
    dict with keys:
      'momentum'             — date × (market, tenor) daily position sizes
      'carry_within_curve'   — date × (market, tenor) daily position weights
      'mean_reversion'       — date × market raw signal (unnormalised)
    """
    return {
        "momentum":           momentum_a_daily(markets=markets, start=start, end=end,
                                               tenors=mom_tenors),
        "carry_within_curve": within_curve_carry_ois(markets=markets, start=start, end=end),
        "mean_reversion":     mean_reversion_rv(markets=markets, start=start, end=end),
    }



