"""
Liquidity analysis and capacity-aware portfolio construction.

Replicates notebook cells 86 (tradable universe, causal_dtv_notional),
87 (OI capacity by depth), 91 (water_fill, build_dynamic_depth_book).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from commodities.power_ns.config import (
    DYNAMIC_MIN_DEPTH,
    DYNAMIC_MAX_DEPTH,
    LAG_BASE,
    COVERAGE_FLOOR,
    ROLL_OUT_BUFFER,
)

# ── Governance constants ──────────────────────────────────────────────────────

OI_CAP              = 0.05    # max fraction of open interest we hold per leg
MWH_PER_CONTRACT    = 736.0   # MWh per power futures contract

VOL_ROLL_WINDOW     = 90
VOL_MIN_PRINTS      = 250
VOL_MIN_COVER       = 0.50

LEG_WEIGHT_Q        = 0.95
LEG_WEIGHT_WINDOW   = 500
LEG_WEIGHT_MIN_OBS  = 60
LEG_WEIGHT_FLOOR    = 1e-3

OI_PRINT_WINDOW     = 20
OI_STALENESS_HALFLIFE = 60.0

NON_TRADABLE_MARKETS: dict[str, str] = {
    "ZUB": "M4 open interest unmeasurable in the capacity window; no depth-4 trade exists",
}


def get_tradable_universe(price_universe: list[str]) -> list[str]:
    """Remove markets that cannot support the full M1-M4 spread."""
    return [m for m in price_universe if m not in NON_TRADABLE_MARKETS]


# ── Dollar volume builder ─────────────────────────────────────────────────────


def causal_dtv_notional(
    col: str,
    index: pd.DatetimeIndex,
    volume: pd.DataFrame,
    prices: pd.DataFrame,
    window: int = VOL_ROLL_WINDOW,
    min_prints: int = VOL_MIN_PRINTS,
    min_cover: float = VOL_MIN_COVER,
) -> Optional[pd.Series]:
    """
    Rolling dollar volume for one contract rank, using only past data.

    Returns NaN wherever the leg has not yet qualified (insufficient prints or
    coverage), allowing callers to fall back to the OI cap.
    """
    if col not in volume.columns or col not in prices.columns:
        return None

    raw  = volume[col].reindex(index)
    seen = raw.notna()
    if not seen.any():
        return None

    first  = seen.idxmax()
    filled = raw.copy()
    filled.loc[first:] = filled.loc[first:].fillna(0.0)

    dtv      = filled.rolling(window, min_periods=20).mean()
    n_prints = seen.cumsum()
    elapsed  = pd.Series(np.arange(len(index)), index=index) - index.get_loc(first)
    coverage = n_prints / elapsed.clip(lower=1)
    eligible = (n_prints >= min_prints) & (coverage >= min_cover)

    return (dtv * prices[col].reindex(index) * MWH_PER_CONTRACT).where(eligible)


# ── OI-based capacity builder ─────────────────────────────────────────────────


def build_oi_capacity(
    open_interest: pd.DataFrame,
    prices: pd.DataFrame,
    candidate_weights_by_depth: dict,
    price_universe: list[str],
    risk_index: pd.DatetimeIndex,
    volume: Optional[pd.DataFrame] = None,
    use_dtv_cap: bool = False,
    dynamic_depths: Optional[list[int]] = None,
    tradable_universe: Optional[list[str]] = None,
) -> tuple[dict, dict]:
    """
    Build per-depth per-market causal dollar capacity.

    Returns (capacity_by_depth, binding_leg_by_depth), both dicts
    {depth: DataFrame(index=risk_index, columns=price_universe)}.

    capacity_by_depth[d][m][t] is the maximum dollar notional that market m's
    depth-d residual portfolio can hold on date t (using only data through t-1).

    DTV cap is implemented but gated by `use_dtv_cap` (notebook default: False).
    """
    if dynamic_depths is None:
        dynamic_depths = list(range(DYNAMIC_MIN_DEPTH, DYNAMIC_MAX_DEPTH + 1))
    if tradable_universe is None:
        tradable_universe = [m for m in price_universe if m not in NON_TRADABLE_MARKETS]

    _pos = pd.Series(np.arange(len(risk_index)), index=risk_index)

    # Per-leg notional capacity base (OI only, or min(OI, DTV))
    oi_notional_base: dict = {}

    for market in tradable_universe:
        for k in range(1, DYNAMIC_MAX_DEPTH + 1):
            col = f"{market}{k}"
            if col not in open_interest.columns or col not in prices.columns:
                continue

            raw   = open_interest[col].reindex(risk_index)
            level = (
                raw.dropna()
                .rolling(OI_PRINT_WINDOW, min_periods=1)
                .median()
                .reindex(risk_index)
                .ffill()
            )
            stale   = _pos - _pos.where(raw.notna()).ffill()
            haircut = np.power(0.5, stale / OI_STALENESS_HALFLIFE)
            px      = prices[col].reindex(risk_index)

            cap_oi = OI_CAP * level * px * MWH_PER_CONTRACT * haircut

            if use_dtv_cap and volume is not None:
                cap_vol = causal_dtv_notional(col, risk_index, volume, prices)
                oi_notional_base[(market, k)] = (
                    cap_oi if cap_vol is None
                    else pd.concat([cap_oi, cap_vol.fillna(np.inf)], axis=1).min(axis=1)
                )
            else:
                oi_notional_base[(market, k)] = cap_oi

    capacity_by_depth:     dict = {}
    binding_leg_by_depth:  dict = {}

    for depth in dynamic_depths:
        C = pd.DataFrame(np.nan, index=risk_index, columns=price_universe, dtype=float)
        B = pd.DataFrame(np.nan, index=risk_index, columns=price_universe, dtype=float)

        for market in tradable_universe:
            legs: dict = {}
            for k in range(1, depth + 1):
                if (market, k) not in oi_notional_base:
                    legs = {}
                    break
                wk = (
                    candidate_weights_by_depth[depth][k][market]
                    .abs()
                    .rolling(LEG_WEIGHT_WINDOW, min_periods=LEG_WEIGHT_MIN_OBS)
                    .quantile(LEG_WEIGHT_Q)
                    .ffill()
                    .clip(lower=LEG_WEIGHT_FLOOR)
                )
                legs[k] = oi_notional_base[(market, k)] / wk

            if legs:
                caps = pd.concat(legs, axis=1)
                C[market] = caps.min(axis=1).where(caps.notna().all(axis=1))
                ok = caps.notna().any(axis=1)
                if ok.any():
                    B.loc[ok, market] = caps.loc[ok].idxmin(axis=1).astype(float)

        capacity_by_depth[depth]    = C.shift(1)
        binding_leg_by_depth[depth] = B.shift(1)

    return capacity_by_depth, binding_leg_by_depth


# ── Water-fill allocator ──────────────────────────────────────────────────────


def water_fill(
    target_weights: np.ndarray,
    capacities: np.ndarray,
    aum: float,
    max_passes: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Allocate dollars subject to known capacities; unknown capacity = excluded.

    Returns (alloc, capped):
        alloc : dollar amounts (same shape as target_weights)
        capped: boolean array — True where market hit its capacity ceiling
    """
    live  = np.isfinite(target_weights) & (target_weights > 0)
    known = np.isfinite(capacities)

    if not live.any():
        return np.zeros_like(target_weights), np.zeros_like(live, dtype=bool)

    cap = np.where(known, capacities, 0.0)
    w   = np.where(live, target_weights, 0.0)
    if w.sum() <= 0:
        return np.zeros_like(w), np.zeros_like(live, dtype=bool)
    w = w / w.sum()

    alloc  = np.zeros_like(w)
    free   = live.copy()
    budget = float(aum)

    for _ in range(max_passes):
        if not free.any() or budget <= 1e-9:
            break
        wf = np.where(free, w, 0.0)
        if wf.sum() <= 0:
            break
        wf       = wf / wf.sum()
        proposed = alloc + budget * wf
        over     = free & (proposed > cap)
        if not over.any():
            alloc = np.where(free, proposed, alloc)
            break
        alloc  = np.where(over, cap, np.where(free, proposed, alloc))
        budget = aum - alloc.sum()
        free  &= ~over

    capped = known & (alloc >= cap - 1e-9) & live
    return alloc, capped


# ── Liquidity-aware portfolio builder ─────────────────────────────────────────


def build_dynamic_depth_book(
    K: pd.DataFrame,
    aum: float,
    capacity_by_depth: dict,
    candidate_weights_by_depth: dict,
    candidate_gross_by_depth: dict,
    price_universe: list[str],
    risk_index: pd.DatetimeIndex,
    run_fn,
    mode: str = "liquidity_aware",
    cost_bps: float = 2.0,
    allocation: str = "reallocated",
    dynamic_min_depth: int = DYNAMIC_MIN_DEPTH,
    dynamic_max_depth: int = DYNAMIC_MAX_DEPTH,
) -> dict:
    """
    Build a liquidity-constrained Dynamic M4/M5 portfolio.

    mode:
        "fixed4"          → always M1-M4
        "blind_deepest"   → deepest data-valid candidate ignoring capacity
        "liquidity_aware" → go deeper only if capacity >= desired dollar allocation (PRODUCTION)

    allocation:
        "reallocated"     → PRODUCTION. Water-fill redistributes shortfall across markets.
        "strict"          → each market capped independently; shortfall stays in cash.
    """
    if allocation not in ("reallocated", "strict"):
        raise ValueError(f"Unknown allocation: {allocation}")

    dynamic_depths = list(range(dynamic_min_depth, dynamic_max_depth + 1))
    Kn = K.div(K.sum(axis=1), axis=0)

    selected_depth    = pd.DataFrame(np.nan, index=risk_index, columns=price_universe, dtype=float)
    selected_capacity = pd.DataFrame(np.nan, index=risk_index, columns=price_universe, dtype=float)

    for root in price_universe:
        valid4 = (
            candidate_gross_by_depth[dynamic_min_depth][root].notna()
            & capacity_by_depth[dynamic_min_depth][root].notna()
        )
        selected_depth.loc[valid4, root]    = dynamic_min_depth
        selected_capacity.loc[valid4, root] = capacity_by_depth[dynamic_min_depth].loc[valid4, root]

        target_dollars = aum * Kn[root]

        if mode == "fixed4":
            continue

        for depth in range(dynamic_min_depth + 1, dynamic_max_depth + 1):
            valid = (
                candidate_gross_by_depth[depth][root].notna()
                & capacity_by_depth[depth][root].notna()
            )
            if mode == "blind_deepest":
                choose = valid
            elif mode == "liquidity_aware":
                choose = (
                    valid
                    & target_dollars.notna()
                    & (capacity_by_depth[depth][root] >= target_dollars)
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")

            selected_depth.loc[choose, root]    = depth
            selected_capacity.loc[choose, root] = capacity_by_depth[depth].loc[choose, root]

    Kv = Kn.to_numpy(float)
    Cv = selected_capacity.to_numpy(float)
    A  = np.zeros_like(Kv)
    C  = np.zeros_like(Kv, dtype=bool)

    for i in range(len(risk_index)):
        if not np.isfinite(Kv[i]).any():
            A[i] = np.nan
            continue
        if allocation == "strict":
            k_i  = np.where(np.isfinite(Kv[i]), Kv[i], 0.0)
            c_i  = np.where(np.isfinite(Cv[i]), Cv[i], 0.0)
            A[i] = np.minimum(k_i * aum, c_i)
            C[i] = (k_i * aum) > (c_i + 1e-9)
        else:
            A[i], C[i] = water_fill(Kv[i], Cv[i], aum)

    alloc  = pd.DataFrame(A, index=risk_index, columns=price_universe) / aum
    capped = pd.DataFrame(C, index=risk_index, columns=price_universe)

    final_weights = {
        k: pd.DataFrame(0.0, index=risk_index, columns=price_universe)
        for k in range(1, dynamic_max_depth + 1)
    }

    for root in price_universe:
        for depth in dynamic_depths:
            m = selected_depth[root].eq(depth)
            if not m.any():
                continue
            for k in range(1, depth + 1):
                final_weights[k].loc[m, root] = (
                    candidate_weights_by_depth[depth][k].loc[m, root]
                    * alloc.loc[m, root]
                )

    res = run_fn(
        final_weights,
        selected_depth,
        cost_bps=cost_bps,
    )

    return {
        "selected_depth":    selected_depth,
        "selected_capacity": selected_capacity,
        "alloc":             alloc,
        "capped":            capped,
        "final_weights":     final_weights,
        **res,
    }
