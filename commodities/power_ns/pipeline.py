"""
Top-level pipeline functions for the Power NS residual-reversal strategy.

run_production()     — full pipeline from data to final P&L
run_full_analysis()  — research harness including diagnostics
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from commodities.power_ns.config import (
    TRADING_DAYS,
    TRADE_MAX_RANK,
    DYNAMIC_MIN_DEPTH,
    DYNAMIC_MAX_DEPTH,
    LAG_BASE,
    ROLL_OUT_BUFFER,
    COVERAGE_FLOOR,
    LAMBDA_GRID,
    TRAIN_END,
    TEST_START,
    COV_HALFLIFE,
    COV_MIN_OBS,
    COV_EXTRA_LAG,
)


# ── Shared state builder ──────────────────────────────────────────────────────


def load_and_prepare(data_dir: Optional[Path] = None) -> dict:
    """
    Load all power data and build all derived panels needed by every function.

    Returns a state dict with keys:
        prices, tickers, open_interest, aggregate_oi, volume,
        MATURITY, BUSINESS_DAYS_TO_MONTH_END,
        price_universe, market_max_rank, TRADING_INDEX,
        ns_results,
        ns_actual_by_rank, ns_fitted_by_rank, ns_residual_by_rank,
    """
    from commodities.power_ns.data     import load_power_data
    from commodities.power_ns.contract import build_maturity_panels, validate_maturity_panels
    from commodities.power_ns.ns_fit   import estimate_ns_panel, build_ns_residual_panels
    from commodities.power_ns.config   import CURVE_MAX_RANK, MIN_OBS_FOR_RANK, LAMBDA_FIT, NS_CURVATURE_PEAK_X

    print("Loading data …")
    d = load_power_data(data_dir)
    prices        = d["prices"]
    tickers       = d["tickers"]
    open_interest = d["open_interest"]
    aggregate_oi  = d["aggregate_oi"]
    volume        = d["volume"]
    market_max_rank = d["market_max_rank"]
    price_universe  = d["price_universe"]
    TRADING_INDEX   = d["TRADING_INDEX"]

    print("Building maturity panels …")
    MATURITY, BUSINESS_DAYS_TO_MONTH_END = build_maturity_panels(tickers)
    validate_maturity_panels(tickers, MATURITY, BUSINESS_DAYS_TO_MONTH_END)

    print("Fitting NS panel (this takes a few minutes) …")
    ns_results = estimate_ns_panel(
        prices, tickers, MATURITY, price_universe,
        market_depth=market_max_rank,
        max_curve_rank=CURVE_MAX_RANK,
        lambda_fit=LAMBDA_FIT,
        ns_curvature_peak_x=NS_CURVATURE_PEAK_X,
        min_obs_for_rank=MIN_OBS_FOR_RANK,
    )

    print("Building NS residual panels …")
    ns_actual_by_rank, ns_fitted_by_rank, ns_residual_by_rank = build_ns_residual_panels(
        ns_results, prices, tickers, MATURITY, price_universe, CURVE_MAX_RANK,
    )

    return {
        "prices":                 prices,
        "tickers":                tickers,
        "open_interest":          open_interest,
        "aggregate_oi":           aggregate_oi,
        "volume":                 volume,
        "MATURITY":               MATURITY,
        "BUSINESS_DAYS_TO_MONTH_END": BUSINESS_DAYS_TO_MONTH_END,
        "price_universe":         price_universe,
        "market_max_rank":        market_max_rank,
        "TRADING_INDEX":          TRADING_INDEX,
        "ns_results":             ns_results,
        "ns_actual_by_rank":      ns_actual_by_rank,
        "ns_fitted_by_rank":      ns_fitted_by_rank,
        "ns_residual_by_rank":    ns_residual_by_rank,
    }


# ── Production pipeline ───────────────────────────────────────────────────────


def run_production(
    data_dir: Optional[Path] = None,
    cost_bps: float = 0.0,
    aum: Optional[float] = None,
    state: Optional[dict] = None,
) -> dict:
    """
    Full production pipeline.

    Steps:
    1. Load data and build NS fits (or use provided `state`)
    2. Select LAMBDA_TRADE on the training window
    3. Build Dynamic M4/M5 residual weights
    4. Run equal-available portfolio at the book level
    5. Build risk-parity sizing weights
    6. Run sized portfolio

    Returns a dict with:
        portfolio_stats   — pd.Series from performance_stats
        gross_pnl         — pd.Series
        net_pnl           — pd.Series (at cost_bps)
        turnover          — pd.Series
        market_gross      — pd.DataFrame per market
        sizing_weights    — risk-parity weight DataFrame
        LAMBDA_TRADE      — selected lambda
        lambda_choice     — full grid search DataFrame
        state             — the prepared data/fit state
    """
    from commodities.power_ns.signal     import (
        select_lambda_trade,
        build_dynamic_depth_weights,
    )
    from commodities.power_ns.execution  import run_residual_portfolio, run_contract_positions
    from commodities.power_ns.risk_model import (
        build_all_sizing_weights,
        SIZING_VARIANTS_DEFAULT,
    )
    from commodities.power_ns.metrics    import performance_stats, book_metrics

    if state is None:
        state = load_and_prepare(data_dir)

    prices                   = state["prices"]
    tickers                  = state["tickers"]
    MATURITY                 = state["MATURITY"]
    BUSINESS_DAYS_TO_MONTH_END = state["BUSINESS_DAYS_TO_MONTH_END"]
    price_universe           = state["price_universe"]
    ns_results               = state["ns_results"]
    ns_residual_by_rank      = state["ns_residual_by_rank"]

    print("Selecting LAMBDA_TRADE …")
    LAMBDA_TRADE, lambda_choice = select_lambda_trade(
        ns_residual_by_rank, ns_results, tickers, prices, MATURITY,
        BUSINESS_DAYS_TO_MONTH_END, price_universe,
        lambda_grid=LAMBDA_GRID, train_end=TRAIN_END,
    )
    print(f"  LAMBDA_TRADE = {LAMBDA_TRADE:.4f}")

    print("Building Dynamic M4/M5 weights …")
    uncon_weights, uncon_depth, candidate_weights_by_depth, candidate_gross_by_depth = (
        build_dynamic_depth_weights(
            ns_residual_by_rank, ns_results, tickers, prices, MATURITY, price_universe,
            LAMBDA_TRADE,
        )
    )

    print("Running unconstrained equal-available book …")
    uncon_book = run_contract_positions(
        uncon_weights, prices, tickers, BUSINESS_DAYS_TO_MONTH_END, price_universe,
        max_rank=DYNAMIC_MAX_DEPTH,
        depth_panel=uncon_depth,
        min_depth=DYNAMIC_MIN_DEPTH,
        lag=LAG_BASE,
        cost_bps=0.0,
        market_aggregation="equal_available",
    )

    print("Building risk-parity sizing weights …")
    # The sign-anchored traded per-market returns are the covariance input.
    # Build a proxy from the uncon_book's market_gross directly.
    RISK_INDEX = uncon_book["market_gross"].index

    sizing_weights, cov_path, cov_counts = build_all_sizing_weights(
        strategy_market_returns=uncon_book["market_gross"].fillna(0.0),
        price_universe=price_universe,
        cov_halflife=COV_HALFLIFE,
        cov_min_obs=COV_MIN_OBS,
        cov_extra_lag=COV_EXTRA_LAG,
        train_end=TRAIN_END,
        test_start=TEST_START,
    )
    rp_K = sizing_weights["risk parity"]

    print("Running risk-parity sized portfolio …")
    final_weights = {
        k: uncon_weights[k].reindex(rp_K.index) * rp_K
        for k in range(1, DYNAMIC_MAX_DEPTH + 1)
    }
    production_book = run_contract_positions(
        final_weights, prices, tickers, BUSINESS_DAYS_TO_MONTH_END, price_universe,
        max_rank=DYNAMIC_MAX_DEPTH,
        depth_panel=uncon_depth,
        min_depth=DYNAMIC_MIN_DEPTH,
        lag=LAG_BASE,
        cost_bps=cost_bps,
        market_aggregation="equal_available",
    )

    gross = production_book["gross"].dropna()
    net   = production_book["net"].dropna()

    return {
        "portfolio_stats":  performance_stats(net),
        "gross_pnl":        gross,
        "net_pnl":          net,
        "turnover":         production_book["turnover"],
        "market_gross":     production_book["market_gross"],
        "sizing_weights":   rp_K,
        "LAMBDA_TRADE":     LAMBDA_TRADE,
        "lambda_choice":    lambda_choice,
        "uncon_book":       uncon_book,
        "uncon_weights":    uncon_weights,
        "uncon_depth":      uncon_depth,
        "candidate_weights_by_depth": candidate_weights_by_depth,
        "candidate_gross_by_depth":   candidate_gross_by_depth,
        "all_sizing_weights": sizing_weights,
        "state":            state,
    }


# ── Research harness ──────────────────────────────────────────────────────────


def run_full_analysis(
    data_dir: Optional[Path] = None,
    cost_bps_grid: tuple = (0.0, 2.0, 5.0, 10.0),
    state: Optional[dict] = None,
    production_result: Optional[dict] = None,
) -> dict:
    """
    Full research harness.

    Runs production pipeline, then adds:
    - Market ADF tests
    - Cost stress test table
    - Yearly performance breakdown
    - Convergence speed panel
    - All sizing variant comparison

    Returns a dict of result DataFrames.
    """
    from commodities.power_ns.execution   import run_residual_portfolio, run_contract_positions
    from commodities.power_ns.diagnostics import (
        run_market_adfs,
        yearly_performance,
        run_all_convergence_speeds,
        implementation_delay_table,
    )
    from commodities.power_ns.metrics     import performance_stats, sharpe
    from commodities.power_ns.risk_model  import SIZING_VARIANTS_DEFAULT

    if production_result is None:
        production_result = run_production(data_dir=data_dir, cost_bps=0.0, state=state)

    state              = production_result["state"]
    prices             = state["prices"]
    tickers            = state["tickers"]
    MATURITY           = state["MATURITY"]
    BDME               = state["BUSINESS_DAYS_TO_MONTH_END"]
    price_universe     = state["price_universe"]
    ns_residual_by_rank = state["ns_residual_by_rank"]
    LAMBDA_TRADE       = production_result["LAMBDA_TRADE"]
    uncon_book         = production_result["uncon_book"]
    uncon_weights      = production_result["uncon_weights"]
    uncon_depth        = production_result["uncon_depth"]
    all_sizing_weights = production_result["all_sizing_weights"]

    # ── Fixed M1-M4 baseline book ─────────────────────────────────────────────
    from commodities.power_ns.signal import build_residual_reversal_weights

    residual_weights_rev, _, _ = build_residual_reversal_weights(
        ns_residual_by_rank, state["ns_results"], tickers, prices, MATURITY,
        price_universe, TRADE_MAX_RANK, LAMBDA_TRADE,
    )
    base_book = run_residual_portfolio(
        residual_weights_rev, prices, tickers, BDME, price_universe,
        lag=LAG_BASE, cost_bps=0.0,
    )

    # ── Market ADF tests ──────────────────────────────────────────────────────
    print("Running market ADF tests …")
    market_adf = run_market_adfs(
        price_universe, ns_residual_by_rank, tickers, MATURITY, LAMBDA_TRADE,
    )

    # ── Cost stress test ──────────────────────────────────────────────────────
    print("Running cost stress test …")
    cost_rows: dict = {}
    for c in cost_bps_grid:
        res  = run_residual_portfolio(
            residual_weights_rev, prices, tickers, BDME, price_universe,
            lag=LAG_BASE, cost_bps=c,
        )
        snet = performance_stats(res["net"].dropna())
        cost_rows[c] = {
            "Gross Sharpe":  sharpe(res["gross"].dropna()),
            "Net Sharpe":    snet["Sharpe ratio"],
            "Net Mean":      snet["Mean"],
            "Annual turnover": res["turnover"].mean() * TRADING_DAYS,
        }
    cost_table = pd.DataFrame(cost_rows).T
    cost_table.index.name = "cost_bps"

    # ── Yearly performance ────────────────────────────────────────────────────
    yearly = yearly_performance(uncon_book["market_gross"], price_universe)

    # ── Convergence speed ─────────────────────────────────────────────────────
    print("Computing convergence speed (slow) …")
    convergence_speed = run_all_convergence_speeds(
        price_universe, ns_residual_by_rank, tickers, MATURITY, LAMBDA_TRADE,
    )

    # ── Sizing variants comparison ────────────────────────────────────────────
    print("Running sizing variant comparison …")
    RISK_INDEX = uncon_book["market_gross"].index

    variant_rows = []
    for label, K in all_sizing_weights.items():
        final_w = {
            k: uncon_weights[k].reindex(K.index) * K
            for k in range(1, DYNAMIC_MAX_DEPTH + 1)
        }
        book = run_contract_positions(
            final_w, prices, tickers, BDME, price_universe,
            max_rank=DYNAMIC_MAX_DEPTH,
            depth_panel=uncon_depth,
            min_depth=DYNAMIC_MIN_DEPTH,
            lag=LAG_BASE,
            cost_bps=0.0,
            market_aggregation="equal_available",
        )
        gross_s = book["gross"]
        turn_s  = book["turnover"]
        row = {"variant": label}
        for period, sl in [("train", slice(None, TRAIN_END)), ("test", slice(TEST_START, None))]:
            g = gross_s.loc[sl].dropna()
            t = turn_s.loc[sl].dropna()
            row[f"{period} Sharpe"]  = sharpe(g)
            if period == "test":
                row["test net@2bps"] = sharpe((g - 2 / 1e4 * t.reindex(g.index).fillna(0)).dropna())
                row["test net@5bps"] = sharpe((g - 5 / 1e4 * t.reindex(g.index).fillna(0)).dropna())
                row["turnover"]      = t.mean() * TRADING_DAYS
        variant_rows.append(row)

    sizing_variants_table = pd.DataFrame(variant_rows).set_index("variant")

    # ── Per-market breakdown ──────────────────────────────────────────────────
    market_stats = pd.DataFrame({
        root: performance_stats(uncon_book["market_gross"][root].dropna())
        for root in price_universe
    }).T
    market_stats["Annualized Turnover"] = pd.Series({
        root: uncon_book["market_turnover"][root].dropna().mean() * TRADING_DAYS
        for root in price_universe
    })

    return {
        "market_adf":             market_adf,
        "cost_table":             cost_table,
        "yearly_performance":     yearly,
        "convergence_speed":      convergence_speed,
        "sizing_variants_table":  sizing_variants_table,
        "market_stats":           market_stats,
        "base_book":              base_book,
        "production_result":      production_result,
    }
