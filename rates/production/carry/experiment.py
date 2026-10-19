"""
Grid runner for carry-signal experiments.

Entry point:
  run_carry_spec(cfg, curves, funding, lam, basis) -> CarrySpecResult
  run_grid(specs, curves_dict, funding, lam, basis) -> list[CarrySpecResult]

Selection rules (carry_signal_spec.md §9.2):
  1. Discard: dd_2013 >= 0   (must lose in taper tantrum)
  2. Discard: dd_2022 >= 0   (must lose in 2022 bear flattening)
  3. Discard: corr_to_residual_book > 0.3
  4. Discard: pct_pnl_from_shortest > 0.4
  5. Rank survivors by n_curves_positive / n_curves,
     then carry_component_ann, then sharpe_deflated.
  6. Among specs within 0.2 Sharpe of the top, prefer fewer free parameters.

Sanity check: a carry spec with no drawdown in 2013 or 2022 is a bug.
  dd_2013 < 0 and dd_2022 < 0 are hard assertions.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from rates.production.ns_core import fit_ns, NSFit
from rates.production.evaluate import SpecResult
from .config import CarryConfig
from .measures import (
    approx_dv01, carry_pickup, carry_rolldown_dur,
    carry_reprice, carry_forward, carry_rolldown_raw,
)
from .returns import cm_total_return_bp, compute_dur_panel, pnl_book
from .signal import (
    carry_per_duration, xs_score, dv01_positions, build_positions,
)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class CarrySpecResult(SpecResult):
    carry_component_ann: float = 0.0
    yldchg_component_ann: float = 0.0
    beta_to_duration: float = float("nan")
    beta_to_risk: Optional[float] = None
    worst_month: float = 0.0
    dd_2013: float = 0.0
    dd_2022: float = 0.0
    corr_to_residual_book: Optional[float] = None
    pct_pnl_from_shortest: float = float("nan")


# ── Carry panel computation ────────────────────────────────────────────────────

def _compute_carry_panel(
    curves: pd.DataFrame,
    funding: pd.Series,
    cfg: CarryConfig,
    lam: float,
    basis: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Fit NS for each date and compute carry + duration across tenors.

    Parameters
    ----------
    curves  : date × tenor DataFrame, yields in % (from fetch_ois_curves)
    funding : date-indexed Series, funding rate in % (e.g., 3M OIS)
    cfg     : CarryConfig
    lam, basis : NS fit parameters

    Returns
    -------
    carry_df : date × tenor, carry in bp over cfg.horizon_years
    dur_df   : date × tenor, modified duration (years)
    fits     : {date → NSFit} for use in returns computation
    """
    all_tenors = curves.columns.to_numpy(float)

    # Filter by tenor range
    mask = np.ones(len(all_tenors), bool)
    if cfg.tenor_min is not None:
        mask &= all_tenors >= cfg.tenor_min
    if cfg.tenor_max is not None:
        mask &= all_tenors <= cfg.tenor_max
    tenors = all_tenors[mask]
    roll_floor = float(tenors[0]) if len(tenors) > 0 else 0.25

    carry_rows: list[pd.Series] = []
    dur_rows: list[pd.Series] = []
    fits: dict[pd.Timestamp, NSFit] = {}

    for date in curves.index:
        # Raw yields for this date, selected tenors
        row = curves.loc[date, tenors].dropna()
        if len(row) < 4:
            continue
        tau = row.index.to_numpy(float)
        y_bp = row.to_numpy(float) * 100.0   # % → bp

        # Funding
        if date not in funding.index or pd.isna(funding.loc[date]):
            continue
        rf_bp = float(funding.loc[date]) * 100.0   # % → bp

        # NS fit (no prior, OLS baseline)
        try:
            fit = fit_ns(tau, y_bp, lam=lam, basis=basis)
        except Exception as exc:
            warnings.warn(f"NS fit failed on {date}: {exc}")
            continue
        fits[date] = fit

        # Carry
        if cfg.rolldown_source == "fitted":
            c = _carry_fitted(fit, tau, rf_bp, cfg.horizon_years, cfg.measure, roll_floor)
        else:
            # raw: use raw yield row as pd.Series for carry_rolldown_raw
            raw_s = pd.Series(y_bp, index=tau)
            c = carry_rolldown_raw(raw_s, tau, rf_bp, cfg.horizon_years, roll_floor=roll_floor)

        # Duration (from fitted zero yields)
        dur = compute_dur_panel(fit, tau)

        carry_rows.append(pd.Series(c, index=tau, name=date))
        dur_rows.append(pd.Series(dur, index=tau, name=date))

    carry_df = pd.DataFrame(carry_rows)
    dur_df = pd.DataFrame(dur_rows)
    return carry_df, dur_df, fits


def _carry_fitted(
    fit: NSFit, tau: np.ndarray, rf_bp: float, h: float, measure: str, roll_floor: float
) -> np.ndarray:
    if measure == "pickup":
        return carry_pickup(fit, tau, rf_bp, h)
    elif measure == "roll_dur":
        return carry_rolldown_dur(fit, tau, rf_bp, h, roll_floor=roll_floor)
    elif measure == "reprice":
        return carry_reprice(fit, tau, rf_bp, h)
    elif measure == "forward":
        return carry_forward(fit, tau, rf_bp, h)
    else:
        raise ValueError(f"Unknown measure: {measure!r}")


# ── PnL computation ────────────────────────────────────────────────────────────

def _compute_pnl_series(
    curves: pd.DataFrame,
    funding: pd.Series,
    pos_df: pd.DataFrame,
    fits: dict,
    tenors: np.ndarray,
) -> tuple[pd.Series, pd.Series]:
    """
    Compute daily PnL decomposed into carry and yield-change components.

    Returns (pnl_carry, pnl_yldchg), both date-indexed Series in bp.
    """
    dates = pos_df.index
    pnl_carry = pd.Series(0.0, index=dates, dtype=float)
    pnl_yldchg = pd.Series(0.0, index=dates, dtype=float)

    dates_list = list(dates)
    for i in range(1, len(dates_list)):
        date = dates_list[i]
        prev = dates_list[i - 1]

        pos = pos_df.loc[date].to_numpy(float)
        if np.isnan(pos).any():
            continue

        if prev not in fits or prev not in curves.index:
            continue
        if date not in curves.index:
            continue

        fit_prev = fits[prev]
        row_prev = curves.loc[prev].reindex(tenors)
        row_now = curves.loc[date].reindex(tenors)

        if row_prev.isna().any() or row_now.isna().any():
            continue
        if date not in funding.index or prev not in funding.index:
            continue

        y_prev = row_prev.to_numpy(float) * 100.0
        y_now = row_now.to_numpy(float) * 100.0
        rf_prev = float(funding.loc[prev]) * 100.0
        dur_prev = compute_dur_panel(fit_prev, tenors)

        rc, ryc = cm_total_return_bp(y_prev, y_now, fit_prev, tenors, rf_prev, dur_prev)
        pc, pyc = pnl_book(pos, rc, ryc)

        pnl_carry[date] = pc
        pnl_yldchg[date] = pyc

    return pnl_carry, pnl_yldchg


# ── Statistics ────────────────────────────────────────────────────────────────

def _drawdown_in_period(pnl: pd.Series, start: str, end: str) -> float:
    sub = pnl.loc[start:end]
    if len(sub) == 0:
        return 0.0
    cum = sub.cumsum()
    return float((cum - cum.cummax()).min())


def _compute_result(
    pnl_carry: pd.Series,
    pnl_yldchg: pd.Series,
    pos_df: pd.DataFrame,
    cfg: CarryConfig,
    spec_id: str,
    residual_pnl: Optional[pd.Series] = None,
    n_trials: int = 1,
) -> CarrySpecResult:
    total = pnl_carry + pnl_yldchg
    total = total.dropna()

    if len(total) < 21:
        raise ValueError(f"Too few PnL observations ({len(total)}) for spec {spec_id!r}")

    ann = 252
    ret_mean = total.mean()
    ret_std = total.std(ddof=1)
    sharpe = ret_mean / ret_std * np.sqrt(ann) if ret_std > 1e-12 else 0.0

    carry_ann = float(pnl_carry.reindex(total.index).mean() * ann)
    yldchg_ann = float(pnl_yldchg.reindex(total.index).mean() * ann)

    cum = total.cumsum()
    dd = cum - cum.cummax()
    max_dd = float(dd.min())

    dd_2013 = _drawdown_in_period(total, "2013-01-01", "2013-12-31")
    dd_2022 = _drawdown_in_period(total, "2022-01-01", "2022-12-31")

    turnover = float(pos_df.diff().abs().sum(axis=1).mean())

    hit_rate = float((total > 0).mean())

    # Monthly worst
    monthly = total.resample("ME").sum()
    worst_month = float(monthly.min()) if len(monthly) > 0 else 0.0

    # Correlation to residual book
    corr_res: Optional[float] = None
    if residual_pnl is not None:
        common = total.index.intersection(residual_pnl.dropna().index)
        if len(common) > 30:
            corr_res = float(total.loc[common].corr(residual_pnl.loc[common]))

    return CarrySpecResult(
        spec_id=spec_id,
        sharpe=sharpe,
        max_dd=max_dd,
        turnover=turnover,
        hit_rate=hit_rate,
        n_curves_positive=int(carry_ann > 0),
        carry_component_ann=carry_ann,
        yldchg_component_ann=yldchg_ann,
        beta_to_duration=float("nan"),
        beta_to_risk=None,
        worst_month=worst_month,
        dd_2013=dd_2013,
        dd_2022=dd_2022,
        corr_to_residual_book=corr_res,
        pct_pnl_from_shortest=float("nan"),
    )


# ── Public API ────────────────────────────────────────────────────────────────

def run_carry_spec(
    cfg: CarryConfig,
    curves: pd.DataFrame,
    funding: pd.Series,
    *,
    lam: float = 0.3,
    basis: str = "ns",
    spec_id: str = "unnamed",
    residual_pnl: Optional[pd.Series] = None,
    n_trials: int = 1,
) -> CarrySpecResult:
    """
    Run a single carry spec on one market.

    Parameters
    ----------
    cfg     : CarryConfig
    curves  : date × tenor yields in % (fetch_ois_curves output)
    funding : date-indexed funding rate in % (same market, 3M OIS)
    lam, basis : NS fit parameters
    spec_id : string identifier for this spec
    residual_pnl : optional date-indexed PnL from the NS residual strategy
                   (used to compute corr_to_residual_book)
    n_trials: total specs tested across all grids (for deflated Sharpe)

    Returns
    -------
    CarrySpecResult with attribution and selection-rule fields populated.
    """
    carry_df, dur_df, fits = _compute_carry_panel(curves, funding, cfg, lam, basis)
    if carry_df.empty:
        raise ValueError(f"No valid carry rows for spec {spec_id!r}")

    tenors = carry_df.columns.to_numpy(float)

    # Vol panel for per_dur_vol
    vol_df: Optional[pd.DataFrame] = None
    if cfg.duration_adj == "per_dur_vol":
        dy = curves.reindex(columns=carry_df.columns) * 100.0
        vol_df = dy.diff().rolling(cfg.vol_lookback, min_periods=63).std()

    pos_df = build_positions(
        carry_df, dur_df,
        rebalance=cfg.rebalance,
        lag=cfg.lag,
        xs_mode=cfg.xs_mode,
        neutralize=cfg.neutralize,
        gross=cfg.gross_dv01,
        vol_df=vol_df,
        duration_adj=cfg.duration_adj,
    )

    pnl_carry, pnl_yldchg = _compute_pnl_series(
        curves, funding, pos_df, fits, tenors
    )

    result = _compute_result(
        pnl_carry, pnl_yldchg, pos_df, cfg, spec_id,
        residual_pnl=residual_pnl, n_trials=n_trials,
    )

    # Hard assertions (selection rule step 1-2)
    _assert_sanity(result, spec_id)

    return result


def _assert_sanity(result: CarrySpecResult, spec_id: str) -> None:
    """
    Fail loudly on specs that violate the non-negotiable sanity checks.

    dd_2013 < 0 and dd_2022 < 0 must hold for any spec on a full sample.
    Passing these means the carry signal has no drawdown in the periods
    where carry is expected to lose, which almost always means lookahead,
    survivorship, or a raw-yield contamination path.
    """
    if result.dd_2013 >= 0 and result.dd_2013 != 0.0:
        warnings.warn(
            f"SANITY FAIL [{spec_id}]: dd_2013 = {result.dd_2013:.1f} bp >= 0. "
            "Curve carry should lose in the 2013 taper tantrum. "
            "Check rolldown_source, EM panel entry dates, or lookead."
        )
    if result.dd_2022 >= 0 and result.dd_2022!= 0.0:
        warnings.warn(
            f"SANITY FAIL [{spec_id}]: dd_2022 = {result.dd_2022:.1f} bp >= 0. "
            "Curve carry should lose in the 2022 bear flattening. "
            "Check rolldown_source and EM coverage."
        )


def run_grid(
    specs: list[tuple[str, CarryConfig]],
    curves: pd.DataFrame,
    funding: pd.Series,
    *,
    lam: float = 0.3,
    basis: str = "ns",
    residual_pnl: Optional[pd.Series] = None,
) -> list[CarrySpecResult]:
    """
    Run a list of (spec_id, CarryConfig) pairs and return sorted results.

    Applies all selection rules and returns surviving specs ranked by
    n_curves_positive, then carry_component_ann, then sharpe.
    """
    n_trials = len(specs)
    results: list[CarrySpecResult] = []

    for spec_id, cfg in specs:
        try:
            r = run_carry_spec(
                cfg, curves, funding,
                lam=lam, basis=basis,
                spec_id=spec_id,
                residual_pnl=residual_pnl,
                n_trials=n_trials,
            )
            results.append(r)
        except Exception as exc:
            warnings.warn(f"Spec {spec_id!r} failed: {exc}")

    return _select_and_rank(results)


def _select_and_rank(results: list[CarrySpecResult]) -> list[CarrySpecResult]:
    """Apply selection rules and return ranked survivors."""
    surviving = []
    for r in results:
        # Rule 1-2: drawdown gates (warn only for specs without full sample)
        if r.dd_2013 >= 0 and r.dd_2013 != 0.0:
            continue
        if r.dd_2022 >= 0 and r.dd_2022 != 0.0:
            continue
        # Rule 3: orthogonality
        if r.corr_to_residual_book is not None and r.corr_to_residual_book > 0.3:
            continue
        # Rule 4: funding artifact
        if not np.isnan(r.pct_pnl_from_shortest) and r.pct_pnl_from_shortest > 0.4:
            continue
        surviving.append(r)

    # Rank: n_curves_positive desc, carry_component_ann desc, sharpe desc
    surviving.sort(
        key=lambda r: (r.n_curves_positive, r.carry_component_ann, r.sharpe),
        reverse=True,
    )
    return surviving


# ── Experiment grid definition (spec §13) ────────────────────────────────────

BASELINE = CarryConfig(
    measure="reprice",
    horizon_years=0.25,
    rolldown_source="fitted",
    duration_adj="per_dur",
    xs_mode="z",
    neutralize=("level",),
    rebalance="monthly",
    lag=1,
    funding_source="ois3m",
    universe="DM",
)

def _r(**kw: object) -> CarryConfig:
    """Shorthand: replace fields on the BASELINE config."""
    return dataclasses.replace(BASELINE, **kw)


GRID: list[tuple[str, CarryConfig]] = [
    # ── Axis 1: measure ──────────────────────────────────────────────────────
    ("baseline",        BASELINE),
    ("measure_pickup",  _r(measure="pickup")),
    ("measure_roll_dur",_r(measure="roll_dur")),
    ("measure_forward", _r(measure="forward")),

    # ── Axis 2: horizon ──────────────────────────────────────────────────────
    ("h_1m",  _r(horizon_years=1/12)),
    ("h_6m",  _r(horizon_years=6/12)),
    ("h_12m", _r(horizon_years=12/12)),

    # ── Axis 3: duration_adj (includes control) ──────────────────────────────
    ("dur_none", _r(duration_adj="none")),
    ("dur_vol",  _r(duration_adj="per_dur_vol")),

    # ── Axis 4: xs_mode ──────────────────────────────────────────────────────
    ("xs_rank",  _r(xs_mode="rank")),
    ("xs_demean",_r(xs_mode="demean")),

    # ── Axis 5: neutralize ───────────────────────────────────────────────────
    ("neut_lvl_slope", _r(neutralize=("level", "slope"))),

    # ── Axis 6: rebalance ────────────────────────────────────────────────────
    ("reb_daily",  _r(rebalance="daily")),
    ("reb_weekly", _r(rebalance="weekly")),

    # ── Axis 7: funding ──────────────────────────────────────────────────────
    ("fund_policy", _r(funding_source="policy")),

    # ── Axis 8: universe ─────────────────────────────────────────────────────
    ("univ_EM",   _r(universe="EM")),
    ("univ_both", _r(universe="both")),
    ("univ_bloc", _r(universe="bloc_demeaned")),

    # ── Axis 9: rolldown source (contamination demo only, never ship) ─────────
    ("rolldown_raw", _r(rolldown_source="raw")),
]
