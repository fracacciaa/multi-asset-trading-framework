"""
Specific-contract execution and P&L engine.

Replicates notebook cell 35: surviving_legs, execution_contract_map,
run_contract_positions, run_residual_portfolio.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from commodities.power_ns.config import (
    TRADE_MAX_RANK,
    LAG_BASE,
    ROLL_OUT_BUFFER,
    COVERAGE_FLOOR,
)

# ── Module-level execution cache ──────────────────────────────────────────────
# Keyed by (data_version, root, max_rank, hash_of_index_bytes).
_EXECUTION_CACHE: dict = {}
_DATA_VERSION: int = 0


def bump_data_version() -> int:
    """Call after reloading prices/tickers to invalidate cached maps."""
    global _DATA_VERSION
    _DATA_VERSION += 1
    _EXECUTION_CACHE.clear()
    return _DATA_VERSION


def clear_execution_cache() -> None:
    _EXECUTION_CACHE.clear()


# ── Roll-survival mask ────────────────────────────────────────────────────────


def surviving_legs(
    BUSINESS_DAYS_TO_MONTH_END: pd.DataFrame,
    root: str,
    idx: pd.DatetimeIndex,
    buffer: int,
    max_rank: int = TRADE_MAX_RANK,
) -> pd.DataFrame:
    """
    Causal mask: a leg survives only when it is strictly more than `buffer`
    business days from the end of its delivery month.

    Returns DataFrame of booleans, index=idx, columns=range(1, max_rank+1).
    """
    cols = [f"{root}{k}" for k in range(1, max_rank + 1)]
    dtd  = BUSINESS_DAYS_TO_MONTH_END.reindex(idx).reindex(columns=cols)
    out  = pd.DataFrame(False, index=idx, columns=range(1, max_rank + 1))
    for j, k in enumerate(range(1, max_rank + 1)):
        out[k] = np.isfinite(dtd[cols[j]]) & (dtd[cols[j]] > float(buffer))
    return out


# ── Execution map (cached) ────────────────────────────────────────────────────


def execution_contract_map(
    prices: pd.DataFrame,
    tickers: pd.DataFrame,
    root: str,
    idx: pd.DatetimeIndex,
    max_rank: int,
    data_version: Optional[int] = None,
) -> dict:
    """
    Build generic→specific mapping with monotonic data-version cache key.
    """
    idx = pd.DatetimeIndex(idx)
    dv  = data_version if data_version is not None else _DATA_VERSION
    key = (dv, root, int(max_rank), hash(idx.asi8.tobytes()))
    cached = _EXECUTION_CACHE.get(key)
    if cached is not None:
        return cached

    cols    = [f"{root}{k}" for k in range(1, max_rank + 1)]
    tk      = tickers.reindex(idx).reindex(columns=cols)
    px      = prices.reindex(idx).reindex(columns=cols)

    flat_tk            = tk.to_numpy().ravel()
    labels, contracts  = pd.factorize(flat_tk, sort=False)
    rows               = np.repeat(np.arange(len(idx)), max_rank)
    vals               = px.to_numpy(float).ravel()
    valid_label        = labels >= 0

    P = np.full((len(idx), len(contracts)), np.nan)
    good_p = valid_label & np.isfinite(vals)
    if len(contracts):
        P[rows[good_p], labels[good_p]] = vals[good_p]

    result = {
        "cols": cols, "tk": tk, "px": px,
        "labels": labels, "contracts": contracts,
        "rows": rows, "valid_label": valid_label, "P": P,
    }
    _EXECUTION_CACHE[key] = result
    return result


# ── Main execution engine ─────────────────────────────────────────────────────


def run_contract_positions(
    weights_by_rank: dict,
    prices: pd.DataFrame,
    tickers: pd.DataFrame,
    BUSINESS_DAYS_TO_MONTH_END: pd.DataFrame,
    universe: list[str],
    *,
    index=None,
    max_rank: int = TRADE_MAX_RANK,
    depth_panel: Optional[pd.DataFrame] = None,
    min_depth: int = 1,
    lag: int = LAG_BASE,
    cost_bps: float = 0.0,
    coverage_floor: float = COVERAGE_FLOOR,
    ffill_limit=None,
    roll_out_buffer: int = ROLL_OUT_BUFFER,
    market_aggregation: str = "sum",
    trade_cap: Optional[dict] = None,
    trade_aum: Optional[float] = None,
) -> dict:
    """
    Canonical specific-contract execution engine.

    `depth_panel=None` → all ranks 1..max_rank active.
    `market_aggregation`:
        "sum"             — final weights already contain market allocation
        "equal_available" — unit-gross market assets are equally averaged
    """
    if ffill_limit is not None:
        raise ValueError("Forward filling is disabled. Use ffill_limit=None.")

    idx = (
        prices.index.intersection(tickers.index).sort_values()
        if index is None else pd.DatetimeIndex(index)
    )

    market_gross     = {}
    market_turnover  = {}
    market_coverage  = {}
    market_unfilled  = {}
    market_exposure  = {}
    market_active    = {}
    market_roll_flat = {}

    for root in universe:
        _emap     = execution_contract_map(prices, tickers, root, idx, max_rank)
        cols      = _emap["cols"]
        tk        = _emap["tk"]
        px        = _emap["px"]

        wdf = pd.concat(
            [weights_by_rank[k][root].rename(k) for k in range(1, max_rank + 1)],
            axis=1,
        ).reindex(idx)

        if depth_panel is None:
            formed = wdf.notna().all(axis=1) & tk.notna().all(axis=1)
            active = np.ones((len(idx), max_rank), dtype=bool)
            W      = wdf.to_numpy(float)
        else:
            dep    = depth_panel[root].reindex(idx)
            active = np.column_stack([
                dep.to_numpy(float) >= k for k in range(1, max_rank + 1)
            ])
            W0     = wdf.to_numpy(float)
            required_ok = (~active) | (np.isfinite(W0) & tk.notna().to_numpy())
            formed = (
                dep.ge(min_depth).fillna(False).to_numpy()
                & required_ok.all(axis=1)
            )
            W = np.where(active, np.nan_to_num(W0, nan=0.0), 0.0)

        formed_arr = np.array(formed, dtype=bool, copy=True)

        if roll_out_buffer:
            surv = surviving_legs(BUSINESS_DAYS_TO_MONTH_END, root, idx, roll_out_buffer, max_rank).to_numpy()
            all_survive  = ((~active) | surv).all(axis=1)
            roll_flat_raw = formed_arr & ~all_survive
            W[roll_flat_raw] = 0.0
        else:
            roll_flat_raw = np.zeros(len(idx), dtype=bool)

        labels      = _emap["labels"]
        contracts   = _emap["contracts"]
        rows        = _emap["rows"]
        valid_label = _emap["valid_label"]
        wvals       = W.ravel()

        n = len(idx)
        C = len(contracts)
        if C == 0:
            _nan = pd.Series(np.nan, index=idx, dtype=float)
            market_gross[root]     = _nan.copy()
            market_turnover[root]  = _nan.copy()
            market_coverage[root]  = _nan.copy()
            market_unfilled[root]  = pd.Series(0.0, index=idx, dtype=float)
            market_exposure[root]  = pd.Series(0.0, index=idx, dtype=float)
            market_active[root]    = pd.Series(False, index=idx, dtype=bool)
            market_roll_flat[root] = pd.Series(False, index=idx, dtype=bool)
            continue

        P       = _emap["P"]
        pos_raw = np.zeros((n, C))
        good_w  = (
            valid_label
            & np.isfinite(wvals)
            & (wvals != 0)
            & np.repeat(formed_arr, max_rank)
        )
        np.add.at(pos_raw, (rows[good_w], labels[good_w]), wvals[good_w])
        pos_raw[~formed_arr] = 0.0

        pos = np.zeros_like(pos_raw)
        if lag > 0:
            pos[lag:] = pos_raw[:-lag]
            live              = np.r_[np.zeros(lag, dtype=bool), formed_arr[:-lag]]
            intentional_flat  = np.r_[np.zeros(lag, dtype=bool), roll_flat_raw[:-lag]]
        else:
            pos[:]           = pos_raw
            live             = formed_arr.copy()
            intentional_flat = roll_flat_raw.copy()

        gap_daily = np.zeros(n)
        if trade_cap is not None:
            if trade_aum is None:
                raise ValueError("trade_cap requires trade_aum")
            capmat = np.full((n, max_rank), np.inf)
            for j, k in enumerate(range(1, max_rank + 1)):
                if (root, k) in trade_cap:
                    capmat[:, j] = trade_cap[(root, k)].reindex(idx).to_numpy(float)
            capmat = np.where(np.isfinite(capmat), capmat, np.inf)
            CAPC = np.full((n, C), np.inf)
            cvals = capmat.ravel()
            CAPC[rows[valid_label], labels[valid_label]] = cvals[valid_label]
            CAPC = np.where(np.isfinite(CAPC), CAPC / float(trade_aum), np.inf)
            held_c = np.zeros(C)
            filled = np.zeros_like(pos)
            for t in range(n):
                mv      = np.clip(pos[t] - held_c, -CAPC[t], CAPC[t])
                held_c  = held_c + mv
                filled[t] = held_c
                gap_daily[t] = np.abs(pos[t] - held_c).sum()
            pos = filled

        ret      = np.full_like(P, np.nan)
        ret[1:]  = P[1:] / P[:-1] - 1.0
        leg      = pos * ret

        gross_w  = np.abs(pos).sum(axis=1)
        seen_w   = np.abs(np.where(np.isfinite(ret), pos, 0.0)).sum(axis=1)
        coverage = np.where(
            gross_w > 0,
            seen_w / np.maximum(gross_w, 1e-12),
            np.nan,
        )
        flat_exec = intentional_flat & (gross_w <= 1e-12)
        complete  = coverage >= (coverage_floor - 1e-12)
        ok        = live & (flat_exec | complete)

        pnl  = np.where(np.isfinite(leg), leg, 0.0).sum(axis=1)
        turn = np.r_[
            np.abs(pos[0]).sum(),
            np.abs(np.diff(pos, axis=0)).sum(axis=1),
        ]

        bucket   = 0.0
        charged  = np.full(n, np.nan)
        for i in range(n):
            bucket += turn[i]
            if ok[i]:
                charged[i] = bucket
                bucket = 0.0

        active_market = live & ((gross_w > 1e-12) | flat_exec)

        market_gross[root]     = pd.Series(np.where(ok, pnl, np.nan), index=idx)
        market_turnover[root]  = pd.Series(charged, index=idx)
        coverage_report        = np.where(flat_exec, 1.0, coverage)
        market_coverage[root]  = pd.Series(np.where(live, coverage_report, np.nan), index=idx)
        market_unfilled[root]  = pd.Series(gap_daily, index=idx)
        market_exposure[root]  = pd.Series(gross_w, index=idx)
        market_active[root]    = pd.Series(active_market, index=idx)
        market_roll_flat[root] = pd.Series(intentional_flat, index=idx)

    MG = pd.DataFrame(market_gross)
    MT = pd.DataFrame(market_turnover)
    CV = pd.DataFrame(market_coverage)
    UF = pd.DataFrame(market_unfilled)
    ME = pd.DataFrame(market_exposure)
    MA = pd.DataFrame(market_active).fillna(False).astype(bool)
    RF = pd.DataFrame(market_roll_flat).fillna(False).astype(bool)

    _expected = pd.Index(universe)
    for _name, _panel in {
        "market_gross": MG, "market_turnover": MT, "market_coverage": CV,
        "market_unfilled": UF, "market_exposure": ME, "market_active": MA,
        "market_roll_flat": RF,
    }.items():
        assert _panel.columns.equals(_expected), (
            f"{_name} columns {_panel.columns.tolist()} != universe {_expected.tolist()}"
        )

    portfolio_ok = ((~MA) | MG.notna()).all(axis=1)

    if market_aggregation == "equal_available":
        MW = MA.astype(float)
        MW = MW.div(MW.sum(axis=1).replace(0, np.nan), axis=0)
    elif market_aggregation == "sum":
        MW = MA.astype(float)
    else:
        raise ValueError(f"Unknown market aggregation: {market_aggregation}")

    gross_raw = (MW * MG).sum(axis=1, min_count=1)
    turn_raw  = (MW * MT).sum(axis=1, min_count=1)
    gross     = gross_raw.where(portfolio_ok)

    turnover = pd.Series(np.nan, index=idx, dtype=float)
    bucket   = 0.0
    for i, dt in enumerate(idx):
        if np.isfinite(turn_raw.iloc[i]):
            bucket += float(turn_raw.iloc[i])
        if bool(portfolio_ok.iloc[i]) and np.isfinite(gross.iloc[i]):
            turnover.iloc[i] = bucket
            bucket = 0.0

    rate = cost_bps / 1e4

    return {
        "unfilled":         UF.sum(axis=1, min_count=1),
        "gross":            gross,
        "net":              gross - rate * turnover,
        "turnover":         turnover,
        "market_gross":     MG,
        "market_net":       MG - rate * MT,
        "market_turnover":  MT,
        "market_coverage":  CV,
        "market_exposure":  ME,
        "market_active":    MA,
        "market_roll_flat": RF,
        "portfolio_complete": portfolio_ok,
        "market_weights":   MW,
    }


def run_residual_portfolio(
    weights_by_rank: dict,
    prices: pd.DataFrame,
    tickers: pd.DataFrame,
    BUSINESS_DAYS_TO_MONTH_END: pd.DataFrame,
    universe: list[str],
    lag: int = LAG_BASE,
    cost_bps: float = 0.0,
    max_rank: int = TRADE_MAX_RANK,
    coverage_floor: float = COVERAGE_FLOOR,
    ffill_limit=None,
    roll_out_buffer: int = ROLL_OUT_BUFFER,
) -> dict:
    """Thin fixed-depth wrapper around run_contract_positions."""
    return run_contract_positions(
        weights_by_rank,
        prices,
        tickers,
        BUSINESS_DAYS_TO_MONTH_END,
        universe,
        max_rank=max_rank,
        lag=lag,
        cost_bps=cost_bps,
        coverage_floor=coverage_floor,
        ffill_limit=ffill_limit,
        roll_out_buffer=roll_out_buffer,
        market_aggregation="equal_available",
    )
