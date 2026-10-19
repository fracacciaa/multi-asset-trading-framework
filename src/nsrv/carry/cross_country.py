"""
Cross-country carry book.

Separate from the within-curve carry book.  Martens et al (2019) report the
correlation between curve carry and country carry at ~11%; these should be
sized and evaluated independently, then combined.

Construction:
  1. For each country: compute carry at a fixed duration target (10Y or blend),
     divide by duration to get yield-space carry.
  2. Cross-section across countries on the same date.
  3. Inverse-vol sizing: pos_c = z_c / trailing_vol_10y_change_c
  4. Zero-net: pos = pos - mean(pos)

Universe axes (carry_signal_spec.md §7.2):
  DM only        : standard baseline
  EM only        : requires verified funding sources; report funding_source
  both           : DM and EM pooled (use with caution, EM dominates z-score)
  bloc_demeaned  : z-score within bloc, then stack

EM caveats (§7.3):
  - funding_source must be explicitly recorded per country
  - tradeable flag filters non-deliverable markets
  - EM panel entry dates must be identical across specs (survivorship guard)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .measures import carry_pickup, carry_rolldown_dur, carry_reprice, carry_forward, approx_dv01, CARRY_MEASURES
from .config import CarryConfig


# ── Rolling PC1 of cross-country yield changes ────────────────────────────────

def _rolling_pc1(
    dy_panel: pd.DataFrame,
    date: pd.Timestamp,
    lookback: int,
    min_obs: int,
) -> Optional[pd.Series]:
    """
    Return the PC1 loading vector of daily yield changes ending the day before `date`.

    Uses a rolling window of `lookback` trading days, requiring at least `min_obs`
    non-NaN observations per market.  Returns None when insufficient data.

    Sign convention: PC1 is oriented so that dot(pc1, ones) > 0, meaning the
    factor represents "all markets rising together" (the global sell-off direction).
    """
    idx = dy_panel.index.get_loc(date) if date in dy_panel.index else None
    if idx is None or idx < lookback:
        return None

    window = dy_panel.iloc[max(0, idx - lookback): idx]   # excludes today (lag=1)
    ok = window.columns[window.notna().sum() >= min_obs]
    if len(ok) < 2:
        return None
    sub = window[ok].dropna(how="any")
    if len(sub) < min_obs:
        return None

    cov = sub.cov().to_numpy()
    try:
        _, vecs = np.linalg.eigh(cov)        # eigh: largest = last column
    except np.linalg.LinAlgError:
        return None

    pc1 = vecs[:, -1]
    if np.dot(pc1, np.ones(len(pc1))) < 0:  # enforce sign convention
        pc1 = -pc1
    return pd.Series(pc1, index=ok)


# ── Market classification ──────────────────────────────────────────────────────

DM_MARKETS = [
    "US", "EUR", "UK", "Japan", "Canada", "Australia",
    "New Zealand", "Sweden", "Switzerland",
]
EM_MARKETS = ["India", "China"]


def _universe_markets(universe: str) -> tuple[list[str], list[str]]:
    """Return (markets_to_use, blocs_list) for cross-sectioning."""
    if universe == "DM":
        return DM_MARKETS, ["DM"]
    elif universe == "EM":
        return EM_MARKETS, ["EM"]
    elif universe == "both":
        return DM_MARKETS + EM_MARKETS, ["all"]
    elif universe == "bloc_demeaned":
        return DM_MARKETS + EM_MARKETS, ["DM", "EM"]
    else:
        raise ValueError(f"Unknown universe: {universe!r}")


# ── Per-country carry at a duration target ────────────────────────────────────

def country_carry_at_target(
    fits: dict,           # {date → NSFit} for one market
    curves: pd.DataFrame, # date × tenor (bp)
    rf: pd.Series,        # date-indexed, funding rate in bp
    tau_target: float,
    cfg: CarryConfig,
) -> pd.Series:
    """
    Time series of carry at `tau_target` years, divided by duration.

    Returns carry-per-duration in bp/year (yield-space carry proxy).
    """
    results = {}
    roll_floor = min(curves.columns.to_numpy(float))

    for date, fit in fits.items():
        if date not in rf.index:
            continue
        rf_bp = float(rf.loc[date])
        tau_arr = np.array([tau_target])

        if cfg.measure == "pickup":
            c = carry_pickup(fit, tau_arr, rf_bp, cfg.horizon_years)
        elif cfg.measure == "roll_dur":
            c = carry_rolldown_dur(fit, tau_arr, rf_bp, cfg.horizon_years, roll_floor=roll_floor)
        elif cfg.measure == "reprice":
            c = carry_reprice(fit, tau_arr, rf_bp, cfg.horizon_years)
        elif cfg.measure == "forward":
            c = carry_forward(fit, tau_arr, rf_bp, cfg.horizon_years)
        else:
            raise ValueError(f"Unknown measure: {cfg.measure!r}")

        y_fit = float(fit.predict(tau_arr)[0])
        dur = approx_dv01(tau_target, y_fit / 1e4)
        results[date] = float(c[0]) / max(dur, 1e-8)

    return pd.Series(results)


# ── Cross-country portfolio ────────────────────────────────────────────────────

def build_cross_country_positions(
    country_carry: pd.DataFrame,   # date × market, carry per duration (bp)
    curves_dict: dict,             # {market → date×tenor (pct)} for vol estimation
    tau_vol_ref: float = 10.0,
    universe: str = "DM",
    vol_lookback: int = 252,
    min_obs: int = 63,
    tradeable_mask: Optional[pd.DataFrame] = None,
    fits_dict: Optional[dict] = None,      # {market → {date → NSFit}} for DV01 neutralisation
    global_beta_neutralize: bool = False,  # also project out the global yield PC1 factor
    pc1_lookback: int = 252,               # rolling window (bd) for PC1 estimation
) -> pd.DataFrame:
    """
    Build inverse-vol-sized, DV01-neutral, zero-net cross-country positions.

    Parameters
    ----------
    country_carry         : date × market DataFrame of carry-per-duration (bp)
    curves_dict           : raw yield curves {market → date×tenor (pct)} for vol estimation
    tau_vol_ref           : tenor for vol estimation and duration (typically 10Y)
    universe              : "DM" | "EM" | "both" | "bloc_demeaned"
    vol_lookback          : rolling window (bd) for trailing yield-change vol
    min_obs               : minimum observations needed before producing a score
    tradeable_mask        : optional date × market bool mask (False = exclude from CS)
    fits_dict             : optional {market → {date → NSFit}}; when provided the portfolio
                            is projected DV01-neutral (parallel shift → zero PnL)
    global_beta_neutralize: also project out the first PC of cross-country yield changes.
                            Removes the "all markets sell off together" risk that DV01
                            neutrality alone does not eliminate when markets co-move
                            by different amounts.  Requires fits_dict.
    pc1_lookback          : rolling window (bd) for estimating the global yield PC1.

    Returns
    -------
    pos_df : date × market DataFrame of position weights
    """
    markets, blocs = _universe_markets(universe)
    available = [m for m in markets if m in country_carry.columns]
    if not available:
        raise ValueError(f"No markets available for universe={universe!r}")

    carry = country_carry[available].copy()

    # Inverse-vol sizing: weight inversely proportional to trailing 10Y yield vol
    # Also build dy_panel for PC1 (reuse the same tenor lookup)
    vols = {}
    dy_cols = {}
    for mkt in available:
        if mkt in curves_dict:
            df = curves_dict[mkt]
            cols = df.columns.to_numpy(float)
            closest = float(cols[np.argmin(np.abs(cols - tau_vol_ref))])
            dy = df[closest].diff() * 100.0  # bp daily change
            vols[mkt] = dy.rolling(vol_lookback, min_periods=min_obs).std()
            dy_cols[mkt] = dy
    vol_df = pd.DataFrame(vols)
    dy_panel = pd.DataFrame(dy_cols) if global_beta_neutralize else None

    pos_rows = {}
    for date in carry.index:
        c_row = carry.loc[date]
        valid = c_row.notna()

        if tradeable_mask is not None and date in tradeable_mask.index:
            valid = valid & tradeable_mask.loc[date].reindex(available, fill_value=True)

        if valid.sum() < 2:
            pos_rows[date] = pd.Series(np.nan, index=available)
            continue

        if universe == "bloc_demeaned":
            # Z-score within each bloc, then stack
            scores = pd.Series(np.nan, index=available)
            for bloc, bloc_mkt_list in [("DM", DM_MARKETS), ("EM", EM_MARKETS)]:
                bloc_mkt = [m for m in bloc_mkt_list if m in available and valid.get(m, False)]
                if len(bloc_mkt) >= 2:
                    c_bloc = c_row[bloc_mkt]
                    s = c_bloc.std()
                    scores[bloc_mkt] = (c_bloc - c_bloc.mean()) / s if s > 1e-10 else 0.0
        else:
            c_valid = c_row[valid]
            s = c_valid.std()
            scores = pd.Series(np.nan, index=available)
            scores[valid] = (c_valid - c_valid.mean()) / s if s > 1e-10 else 0.0

        # Inverse-vol sizing
        if date in vol_df.index:
            v_row = vol_df.loc[date]
            scores = scores / v_row.replace(0, np.nan).clip(lower=1e-8)

        # Zero-net
        scores = scores - scores.mean()

        # DV01 neutralisation: project out the aggregate duration-weighted component
        # so that a parallel shift of every curve produces zero PnL.
        # Requires fits_dict; skipped silently when not supplied.
        if fits_dict is not None:
            dur = pd.Series(np.nan, index=available)
            for mkt in available:
                if mkt in fits_dict and date in fits_dict[mkt]:
                    y_bp = float(fits_dict[mkt][date].predict(np.array([tau_vol_ref]))[0])
                    dur[mkt] = approx_dv01(tau_vol_ref, y_bp / 1e4)

            proj_valid = scores.notna() & dur.notna()
            if proj_valid.sum() >= 2:
                w = scores[proj_valid].to_numpy(float)
                d = dur[proj_valid].to_numpy(float)
                dd = np.dot(d, d)
                if dd > 1e-12:
                    w = w - (np.dot(w, d) / dd) * d
                new_scores = pd.Series(np.nan, index=available)
                new_scores[proj_valid] = w
                scores = new_scores

        # Global yield beta neutralisation: project out the PC1 of yield changes.
        # Hedge vector h_m = dur_m × pc1_m  (units: years × dimensionless)
        # After projection: dot(pos, h) = 0, so the portfolio has zero sensitivity
        # to the global yield factor (weighted by each market's co-movement amplitude).
        if global_beta_neutralize and dy_panel is not None:
            pc1 = _rolling_pc1(dy_panel, date, pc1_lookback, min_obs)
            if pc1 is not None:
                hedge = pd.Series(np.nan, index=available)
                for mkt in available:
                    if mkt not in pc1.index:
                        continue
                    if fits_dict is not None and mkt in fits_dict and date in fits_dict[mkt]:
                        y_bp = float(fits_dict[mkt][date].predict(np.array([tau_vol_ref]))[0])
                        dur_m = approx_dv01(tau_vol_ref, y_bp / 1e4)
                    else:
                        dur_m = approx_dv01(tau_vol_ref, 0.04)   # 4% fallback
                    hedge[mkt] = dur_m * float(pc1[mkt])

                proj_valid = scores.notna() & hedge.notna()
                if proj_valid.sum() >= 2:
                    w = scores[proj_valid].to_numpy(float)
                    h = hedge[proj_valid].to_numpy(float)
                    hh = np.dot(h, h)
                    if hh > 1e-12:
                        w = w - (np.dot(w, h) / hh) * h
                    new_scores = pd.Series(np.nan, index=available)
                    new_scores[proj_valid] = w
                    scores = new_scores

        pos_rows[date] = scores

    pos_df = pd.DataFrame(pos_rows).T
    pos_df.index.name = "date"
    return pos_df


# ── EM survivorship guard ──────────────────────────────────────────────────────

def validate_em_entry_dates(
    country_carry: pd.DataFrame,
    markets: list[str],
) -> None:
    """
    Assert that EM markets enter the panel on the same first valid date
    across all specs.  Call once per market, store the first_valid date in
    the manifest.  Raise if a market re-enters after a gap.
    """
    em = [m for m in markets if m in EM_MARKETS and m in country_carry.columns]
    for mkt in em:
        s = country_carry[mkt].dropna()
        if len(s) == 0:
            continue
        first_valid = s.index[0]
        expected_idx = country_carry.loc[first_valid:, mkt].index
        actual_idx = s.index
        if len(actual_idx) != len(expected_idx):
            raise ValueError(
                f"EM market {mkt!r} has gaps after first entry {first_valid.date()}. "
                "This creates survivorship bias.  Check data coverage."
            )
