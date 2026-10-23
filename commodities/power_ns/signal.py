"""
Signal weight construction: projected NS-null residual weights, lambda selection,
dynamic M4/M5 depth.

Replicates notebook cells 36 (build_projected_residual_weights),
37 (select_lambda_trade), 69/70 (dynamic M4/M5).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from commodities.power_ns.config import (
    TRADE_MAX_RANK,
    DYNAMIC_MIN_DEPTH,
    DYNAMIC_MAX_DEPTH,
    LAMBDA_GRID,
    TRAIN_END,
    NS_CURVATURE_PEAK_X,
    LAG_BASE,
    ROLL_OUT_BUFFER,
    COVERAGE_FLOOR,
)
from commodities.power_ns.ns_fit import ns_loadings
from commodities.power_ns.metrics import sharpe


# ── Core weight builder ───────────────────────────────────────────────────────


def build_projected_residual_weights(
    residual_by_rank: dict,
    ns_results: pd.DataFrame,
    tickers: pd.DataFrame,
    prices: pd.DataFrame,
    MATURITY: pd.DataFrame,
    universe: list[str],
    depth: int,
    lambda_trade: Optional[float],
) -> tuple[dict, pd.DataFrame]:
    """
    Build price-consistent NS-null residual portfolio weights for any depth >= 4.

    Returns (weights, projected_gross):
        weights        : dict {rank: DataFrame(index=dates, columns=universe)}
        projected_gross: DataFrame(index=dates, columns=universe) — |Q| per day
    """
    idx = residual_by_rank[1].index
    weights = {
        k: pd.DataFrame(np.nan, index=idx, columns=universe, dtype=float)
        for k in range(1, depth + 1)
    }
    projected_gross = pd.DataFrame(np.nan, index=idx, columns=universe, dtype=float)

    for root in universe:
        try:
            x = ns_results.xs(root, level="Commodity").sort_index()
        except KeyError:
            continue

        dates = x.index[x["N_contracts"].ge(depth)].intersection(idx)
        if len(dates) == 0:
            continue

        e = pd.concat(
            [residual_by_rank[k][root].rename(k) for k in range(1, depth + 1)],
            axis=1,
        ).reindex(dates)

        cols = [f"{root}{k}" for k in range(1, depth + 1)]
        if any(c not in tickers.columns or c not in prices.columns for c in cols):
            continue

        E  = e.to_numpy(float)
        PX = prices.reindex(dates)[cols].to_numpy(float)
        M  = MATURITY.reindex(dates)[cols].to_numpy(float)

        good = (
            np.isfinite(E).all(axis=1)
            & np.isfinite(PX).all(axis=1)
            & (np.abs(PX) > 1e-12).all(axis=1)
            & np.isfinite(M).all(axis=1)
            & (M > 0).all(axis=1)
        )
        if not good.any():
            continue

        dates_g = dates[good]
        E_g, PX_g, M_g = E[good], PX[good], M[good]
        lam = (
            x.loc[dates_g, "lambda"].to_numpy(float)
            if lambda_trade is None
            else np.full(len(dates_g), float(lambda_trade))
        )

        slope, curv = ns_loadings(M_g, lam[:, None])
        X = np.stack([np.ones_like(M_g), slope, curv], axis=2)  # (n, depth, 3)

        U, S, _ = np.linalg.svd(X, full_matrices=True)
        N       = U[:, :, 3:]                                     # null-space basis
        coeff   = np.einsum("ndq,nd->nq", N, E_g)
        Q       = -np.einsum("ndq,nq->nd", N, coeff)

        qgross  = np.abs(Q).sum(axis=1)
        notional = Q * PX_g
        ngross  = np.abs(notional).sum(axis=1)

        ok = (
            np.isfinite(qgross) & (qgross > 1e-12)
            & np.isfinite(ngross) & (ngross > 1e-12)
        )

        W = np.full_like(notional, np.nan)
        W[ok] = notional[ok] / ngross[ok, None]

        for j, k in enumerate(range(1, depth + 1)):
            weights[k].loc[dates_g, root] = W[:, j]
        projected_gross.loc[dates_g, root] = qgross

    return weights, projected_gross


def build_residual_reversal_weights(
    residual_by_rank: dict,
    ns_results: pd.DataFrame,
    tickers: pd.DataFrame,
    prices: pd.DataFrame,
    MATURITY: pd.DataFrame,
    universe: list[str],
    trade_max_rank: int = TRADE_MAX_RANK,
    lambda_trade: Optional[float] = None,
) -> tuple[dict, pd.DataFrame, pd.Series]:
    """M1-M4 wrapper; returns (weights, projected_gross, dropped) for compatibility."""
    w, g = build_projected_residual_weights(
        residual_by_rank=residual_by_rank,
        ns_results=ns_results,
        tickers=tickers,
        prices=prices,
        MATURITY=MATURITY,
        universe=universe,
        depth=trade_max_rank,
        lambda_trade=lambda_trade,
    )
    dropped = pd.Series(dtype=float, name="gross removed")
    return w, g, dropped


# ── Lambda trade selection ────────────────────────────────────────────────────


def select_lambda_trade(
    ns_residual_by_rank: dict,
    ns_results: pd.DataFrame,
    tickers: pd.DataFrame,
    prices: pd.DataFrame,
    MATURITY: pd.DataFrame,
    BUSINESS_DAYS_TO_MONTH_END: pd.DataFrame,
    price_universe: list[str],
    lambda_grid: list[float] = LAMBDA_GRID,
    train_end: pd.Timestamp = TRAIN_END,
    ns_curvature_peak_x: float = NS_CURVATURE_PEAK_X,
    trade_max_rank: int = TRADE_MAX_RANK,
    lag: int = LAG_BASE,
) -> tuple[float, pd.DataFrame]:
    """
    Select LAMBDA_TRADE by maximizing IS Sharpe on the training window.

    Grid-searches lambda_grid; runs run_residual_portfolio for each candidate.
    Returns (LAMBDA_TRADE, lambda_choice_df).

    IMPORTANT: LAMBDA_TRADE must NOT sit at a grid edge (the optimizer asserts
    this to guard against degenerate condition numbers at the extremes).
    """
    from commodities.power_ns.execution import run_residual_portfolio

    _Mref = np.array([0.75, 1.75, 2.75, 3.75])
    rows  = []

    for lam in lambda_grid:
        w, _, _ = build_residual_reversal_weights(
            residual_by_rank=ns_residual_by_rank,
            ns_results=ns_results,
            tickers=tickers,
            prices=prices,
            MATURITY=MATURITY,
            universe=price_universe,
            trade_max_rank=trade_max_rank,
            lambda_trade=lam,
        )
        mg   = run_residual_portfolio(
            w, prices, tickers, BUSINESS_DAYS_TO_MONTH_END, price_universe,
            lag=lag, cost_bps=0.0,
        )["market_gross"]
        per  = sharpe(mg.loc[:train_end], min_obs=251)

        sl, cv = ns_loadings(_Mref, lam)
        X4     = np.column_stack([np.ones(4), sl, cv])
        v      = np.linalg.svd(X4)[0][:, 3]
        v      = v / np.abs(v).sum()
        v      = v * np.sign(-v[2])

        rows.append({
            "lambda":              lam,
            "peak (months)":       ns_curvature_peak_x / lam,
            "cond(X4)":            np.linalg.cond(X4),
            "TRAIN Sharpe mean":   per.mean(),
            "TRAIN Sharpe median": per.median(),
            "n mkts":              int(per.notna().sum()),
            "M1": v[0], "M2": v[1], "M3": v[2], "M4": v[3],
        })

    lambda_choice = pd.DataFrame(rows).set_index("lambda")
    LAMBDA_TRADE  = float(lambda_choice["TRAIN Sharpe mean"].idxmax())

    assert LAMBDA_TRADE not in (lambda_grid[0], lambda_grid[-1]), (
        "train argmax sits at a grid edge; widen LAMBDA_GRID. "
        "cond(X4) blows up at both ends, so an edge solution is a degenerate basis."
    )

    return LAMBDA_TRADE, lambda_choice


# ── Dynamic M4/M5 depth ───────────────────────────────────────────────────────


def build_dynamic_depth_weights(
    ns_residual_by_rank: dict,
    ns_results: pd.DataFrame,
    tickers: pd.DataFrame,
    prices: pd.DataFrame,
    MATURITY: pd.DataFrame,
    price_universe: list[str],
    LAMBDA_TRADE: float,
    dynamic_min_depth: int = DYNAMIC_MIN_DEPTH,
    dynamic_max_depth: int = DYNAMIC_MAX_DEPTH,
) -> tuple[dict, pd.DataFrame, dict, dict]:
    """
    Build Dynamic M4/M5 residual weights.

    Returns:
        uncon_weights             : {rank: DataFrame} — M5 if available, else M4
        uncon_depth               : DataFrame — selected depth (4 or 5) per market/date
        candidate_weights_by_depth: {depth: {rank: DataFrame}}
        candidate_gross_by_depth  : {depth: DataFrame}
    """
    candidate_weights_by_depth: dict[int, dict] = {}
    candidate_gross_by_depth: dict[int, pd.DataFrame] = {}

    for depth in range(dynamic_min_depth, dynamic_max_depth + 1):
        w, g = build_projected_residual_weights(
            residual_by_rank=ns_residual_by_rank,
            ns_results=ns_results,
            tickers=tickers,
            prices=prices,
            MATURITY=MATURITY,
            universe=price_universe,
            depth=depth,
            lambda_trade=LAMBDA_TRADE,
        )
        candidate_weights_by_depth[depth] = w
        candidate_gross_by_depth[depth]   = g

    _dead_m4 = [
        m for m in price_universe
        if candidate_gross_by_depth[dynamic_min_depth][m].notna().sum() == 0
    ]
    if _dead_m4:
        raise ValueError(f"Markets with zero M1-M4 residual candidates: {_dead_m4}")

    _have4 = candidate_gross_by_depth[dynamic_min_depth].notna()
    _have5 = candidate_gross_by_depth[dynamic_max_depth].notna()

    uncon_depth = (
        pd.DataFrame(np.nan, index=_have4.index, columns=price_universe, dtype=float)
        .mask(_have4, float(dynamic_min_depth))
        .mask(_have5, float(dynamic_max_depth))
    )

    _use5 = uncon_depth.eq(float(dynamic_max_depth))

    uncon_weights: dict[int, pd.DataFrame] = {}
    for k in range(1, dynamic_max_depth + 1):
        w5 = candidate_weights_by_depth[dynamic_max_depth][k]
        w4 = (
            candidate_weights_by_depth[dynamic_min_depth][k]
            if k <= dynamic_min_depth
            else pd.DataFrame(np.nan, index=w5.index, columns=w5.columns, dtype=float)
        )
        uncon_weights[k] = w5.where(_use5, w4)

    return uncon_weights, uncon_depth, candidate_weights_by_depth, candidate_gross_by_depth
