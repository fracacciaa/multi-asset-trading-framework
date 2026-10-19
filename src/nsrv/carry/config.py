"""
CarryConfig: all knobs for a single carry-signal specification.

Non-negotiable defaults (carry_signal_spec.md §10):
  measure         = "reprice"   (full repricing handles convexity)
  rolldown_source = "fitted"    ("raw" is a contamination demo, not shippable)
  duration_adj    = "per_dur"   (raw carry is a duration bet, not a curve bet)
  rebalance       = "monthly"   (Martens et al convention)
  lag             = 1           (required; no implicit zero-lag)
  require_funding = True        (no cross-country carry without funding series)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CarryConfig:
    # ── carry definition ─────────────────────────────────────────────────────
    measure: str = "reprice"
    # pickup | roll_dur | reprice | forward
    # "roll_dur" uses a duration approximation; "reprice" is the preferred measure

    horizon_years: float = 0.25
    # holding horizon: 1/12 (1M), 3/12 (3M), 6/12 (6M), 12/12 (12M)

    rolldown_source: str = "fitted"
    # "fitted": roll-down from NS-fitted curve (orthogonal to residual signal)
    # "raw"   : roll-down from interpolated raw yields (contamination demo only)

    # ── NS fit parameters ────────────────────────────────────────────────────
    lam: float = 0.3
    # Nelson-Siegel lambda (1/years). Inherit from residual-spec winner when available.

    basis: str = "ns"
    # ns | bc | svensson_adj

    parent_fit_spec: Optional[str] = None
    # spec_id of the NS fit that generated the curves used here (for manifest)

    # ── duration and cross-section ───────────────────────────────────────────
    duration_adj: str = "per_dur"
    # none       : raw carry (control; will show high beta_to_duration)
    # per_dur    : carry / D
    # per_dur_vol: carry / (D × trailing_sd_yield_change)

    xs_mode: str = "z"
    # z | rank | demean

    neutralize: tuple = ("level",)
    # ("level",)          : project out DV01-weighted level
    # ("level", "slope")  : also project out slope loading

    # ── position management ──────────────────────────────────────────────────
    rebalance: str = "monthly"
    # daily | weekly | monthly

    lag: int = 1
    # signal lag in trading days; must be >= 1

    gross_dv01: float = 1.0
    # target gross DV01 per unit notional for the position book

    # ── funding ───────────────────────────────────────────────────────────────
    funding_source: str = "ois3m"
    # ois3m | bill3m | policy
    # For DM: 3M OIS is the standard.  For EM: onshore 3M depo or policy rate.

    require_funding: bool = True
    # True: raise if a curve has no funding series
    # False: skip that curve in cross-country (within-curve carry still works without)

    coupon_reinvest: bool = False
    # Reinvest coupons at rf during the horizon (small effect for h <= 1)

    # ── tenor filter ──────────────────────────────────────────────────────────
    tenor_min: Optional[float] = None   # e.g. 1.0 to exclude sub-1Y tenors
    tenor_max: Optional[float] = None   # e.g. 30.0

    # ── cross-country universe ────────────────────────────────────────────────
    universe: str = "DM"
    # DM | EM | both | bloc_demeaned
    # Do NOT pool DM and EM in one z-score; use bloc_demeaned instead.

    vol_lookback: int = 252
    # rolling window (business days) for trailing yield-change vol

    # ── costs ─────────────────────────────────────────────────────────────────
    bid_ask_bp: Optional[dict] = None
    # {tenor_years → half-spread in bp}; None → use module defaults

    def __post_init__(self) -> None:
        if self.lag < 1:
            raise ValueError(f"lag must be >= 1, got {self.lag}")
        if self.measure not in ("pickup", "roll_dur", "reprice", "forward"):
            raise ValueError(f"Unknown measure: {self.measure}")
        if self.rolldown_source not in ("fitted", "raw"):
            raise ValueError(f"rolldown_source must be 'fitted' or 'raw'")
        if self.duration_adj not in ("none", "per_dur", "per_dur_vol"):
            raise ValueError(f"Unknown duration_adj: {self.duration_adj}")
        if self.xs_mode not in ("z", "rank", "demean"):
            raise ValueError(f"Unknown xs_mode: {self.xs_mode}")
        if self.rebalance not in ("daily", "weekly", "monthly"):
            raise ValueError(f"Unknown rebalance: {self.rebalance}")
        if self.universe not in ("DM", "EM", "both", "bloc_demeaned"):
            raise ValueError(f"Unknown universe: {self.universe}")
        if self.horizon_years <= 0:
            raise ValueError(f"horizon_years must be > 0")
