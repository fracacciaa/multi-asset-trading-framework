"""
Combined NS residual-reversal pipeline: 4 power markets + 13 general commodities.

Fixed M1-M4 depth for all markets (no dynamic M5).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from commodities.power_ns.config import (
    TRADING_DAYS,
    LAG_BASE,
    ROLL_OUT_BUFFER,
    COVERAGE_FLOOR,
    LAMBDA_GRID,
    TRAIN_END,
    TEST_START,
    COV_HALFLIFE,
    COV_MIN_OBS,
    COV_EXTRA_LAG,
    LAMBDA_FIT,
    NS_CURVATURE_PEAK_X,
    MIN_OBS_FOR_RANK,
)
from commodities.combined_ns.config import COMBINED_TRADE_MAX_RANK


# ── State builder ─────────────────────────────────────────────────────────────


def load_and_prepare(data_dir: Optional[Path] = None) -> dict:
    """
    Load combined data and build all derived panels.

    Returns the same state dict as power_ns.load_and_prepare, plus 'data'.
    """
    from commodities.combined_ns.data  import load_combined_data
    from commodities.power_ns.contract import build_maturity_panels, validate_maturity_panels
    from commodities.power_ns.ns_fit   import estimate_ns_panel, build_ns_residual_panels
    from commodities.power_ns.config   import CURVE_MAX_RANK

    print("Loading combined data …")
    d = load_combined_data(data_dir)

    prices          = d["prices"]
    tickers         = d["tickers"]
    open_interest   = d["open_interest"]
    aggregate_oi    = d["aggregate_oi"]
    market_max_rank = d["market_max_rank"]
    price_universe  = d["price_universe"]
    TRADING_INDEX   = d["TRADING_INDEX"]

    print("Building maturity panels …")
    MATURITY, BUSINESS_DAYS_TO_MONTH_END = build_maturity_panels(tickers)
    validate_maturity_panels(tickers, MATURITY, BUSINESS_DAYS_TO_MONTH_END)

    print("Fitting NS panel …")
    ns_results = estimate_ns_panel(
        prices, tickers, MATURITY, price_universe,
        market_depth=market_max_rank,
        max_curve_rank=COMBINED_TRADE_MAX_RANK,
        lambda_fit=LAMBDA_FIT,
        ns_curvature_peak_x=NS_CURVATURE_PEAK_X,
        min_obs_for_rank=MIN_OBS_FOR_RANK,
    )

    print("Building NS residual panels …")
    ns_actual_by_rank, ns_fitted_by_rank, ns_residual_by_rank = build_ns_residual_panels(
        ns_results, prices, tickers, MATURITY, price_universe, COMBINED_TRADE_MAX_RANK,
    )

    return {
        "prices":                     prices,
        "tickers":                    tickers,
        "open_interest":              open_interest,
        "aggregate_oi":               aggregate_oi,
        "MATURITY":                   MATURITY,
        "BUSINESS_DAYS_TO_MONTH_END": BUSINESS_DAYS_TO_MONTH_END,
        "price_universe":             price_universe,
        "market_max_rank":            market_max_rank,
        "TRADING_INDEX":              TRADING_INDEX,
        "ns_results":                 ns_results,
        "ns_actual_by_rank":          ns_actual_by_rank,
        "ns_fitted_by_rank":          ns_fitted_by_rank,
        "ns_residual_by_rank":        ns_residual_by_rank,
    }


# ── Production pipeline ───────────────────────────────────────────────────────


def run_production(
    data_dir: Optional[Path] = None,
    cost_bps: float = 0.0,
    aum: Optional[float] = None,
    state: Optional[dict] = None,
) -> dict:
    """
    Full combined production pipeline (M1-M4, fixed depth).

    Steps:
    1. Load combined data and build NS fits
    2. Select LAMBDA_TRADE on the training window
    3. Build M1-M4 residual weights for all 17 markets
    4. Run equal-available book
    5. Build risk-parity sizing weights
    6. Run sized portfolio

    Returns dict with portfolio_stats, gross_pnl, net_pnl, turnover,
    market_gross, sizing_weights, LAMBDA_TRADE, lambda_choice, state.
    """
    from commodities.power_ns.signal    import select_lambda_trade, build_residual_reversal_weights
    from commodities.power_ns.execution import run_contract_positions
    from commodities.power_ns.risk_model import build_all_sizing_weights
    from commodities.power_ns.metrics   import performance_stats

    if state is None:
        state = load_and_prepare(data_dir)

    prices                     = state["prices"]
    tickers                    = state["tickers"]
    MATURITY                   = state["MATURITY"]
    BUSINESS_DAYS_TO_MONTH_END = state["BUSINESS_DAYS_TO_MONTH_END"]
    price_universe             = state["price_universe"]
    ns_results                 = state["ns_results"]
    ns_residual_by_rank        = state["ns_residual_by_rank"]

    print("Selecting LAMBDA_TRADE …")
    LAMBDA_TRADE, lambda_choice = select_lambda_trade(
        ns_residual_by_rank, ns_results, tickers, prices, MATURITY,
        BUSINESS_DAYS_TO_MONTH_END, price_universe,
        lambda_grid=LAMBDA_GRID, train_end=TRAIN_END,
        trade_max_rank=COMBINED_TRADE_MAX_RANK,
    )
    print(f"  LAMBDA_TRADE = {LAMBDA_TRADE:.4f}")

    print("Building M1-M4 residual weights …")
    residual_weights, projected_gross, _ = build_residual_reversal_weights(
        ns_residual_by_rank, ns_results, tickers, prices, MATURITY,
        price_universe, trade_max_rank=COMBINED_TRADE_MAX_RANK,
        lambda_trade=LAMBDA_TRADE,
    )

    print("Running equal-available book …")
    uncon_book = run_contract_positions(
        residual_weights, prices, tickers, BUSINESS_DAYS_TO_MONTH_END, price_universe,
        max_rank=COMBINED_TRADE_MAX_RANK,
        depth_panel=None,
        min_depth=COMBINED_TRADE_MAX_RANK,
        lag=LAG_BASE,
        cost_bps=0.0,
        market_aggregation="equal_available",
    )

    # ── Equal-weight production book ──────────────────────────────────────────
    # Equal-weight is the correct sizing for a cross-asset portfolio:
    # RP over-weights low-variance gen markets (tiny NS residuals) and kills
    # performance by diluting the higher-Sharpe power markets.
    print("Running equal-weight production book …")
    production_book = run_contract_positions(
        residual_weights, prices, tickers, BUSINESS_DAYS_TO_MONTH_END, price_universe,
        max_rank=COMBINED_TRADE_MAX_RANK,
        depth_panel=None,
        min_depth=COMBINED_TRADE_MAX_RANK,
        lag=LAG_BASE,
        cost_bps=cost_bps,
        market_aggregation="equal_available",
    )

    gross = production_book["gross"].dropna()
    net   = production_book["net"].dropna()

    # ── RP sizing (research only) ─────────────────────────────────────────────
    print("Building risk-parity sizing weights (research) …")
    sizing_weights, cov_path, cov_counts = build_all_sizing_weights(
        strategy_market_returns=uncon_book["market_gross"],
        price_universe=price_universe,
        cov_halflife=COV_HALFLIFE,
        cov_min_obs=COV_MIN_OBS,
        cov_extra_lag=COV_EXTRA_LAG,
        train_end=TRAIN_END,
        test_start=TEST_START,
    )

    return {
        "portfolio_stats":    performance_stats(net),
        "gross_pnl":          gross,
        "net_pnl":            net,
        "turnover":           production_book["turnover"],
        "market_gross":       production_book["market_gross"],
        "LAMBDA_TRADE":       LAMBDA_TRADE,
        "lambda_choice":      lambda_choice,
        "uncon_book":         uncon_book,
        "residual_weights":   residual_weights,
        "projected_gross":    projected_gross,
        "all_sizing_weights": sizing_weights,
        "state":              state,
    }


def run_production_hysteresis(
    data_dir: Optional[Path] = None,
    cost_bps: float = 0.0,
    state: Optional[dict] = None,
    production_result: Optional[dict] = None,
    z_entry: float = 1.0,
    ewma_halflife: int = 63,
) -> dict:
    """
    Combined NS pipeline with hysteresis positioning.

    Positions are discrete {-1, 0, +1}: enter when |z| > z_entry (EWMA-normalised),
    hold through noise, exit only when z crosses zero.

    If production_result is provided, reuses its state and residual_weights to
    avoid re-running the expensive NS fit.

    Returns the same keys as run_production plus hyst_positions and z_scores.
    """
    from commodities.power_ns.signal    import build_residual_reversal_weights, select_lambda_trade
    from commodities.power_ns.execution import run_contract_positions
    from commodities.power_ns.metrics   import performance_stats
    from commodities.combined_ns.signal import compute_ns_alpha_panel, build_hysteresis_weights

    if production_result is not None:
        state            = production_result["state"]
        residual_weights = production_result["residual_weights"]
        LAMBDA_TRADE     = production_result["LAMBDA_TRADE"]
        lambda_choice    = production_result["lambda_choice"]
    else:
        if state is None:
            state = load_and_prepare(data_dir)

        print("Selecting LAMBDA_TRADE …")
        LAMBDA_TRADE, lambda_choice = select_lambda_trade(
            state["ns_residual_by_rank"], state["ns_results"], state["tickers"],
            state["prices"], state["MATURITY"], state["BUSINESS_DAYS_TO_MONTH_END"],
            state["price_universe"],
            lambda_grid=LAMBDA_GRID, train_end=TRAIN_END,
            trade_max_rank=COMBINED_TRADE_MAX_RANK,
        )
        print(f"  LAMBDA_TRADE = {LAMBDA_TRADE:.4f}")

        print("Building M1-M4 residual weights …")
        residual_weights, _, _ = build_residual_reversal_weights(
            state["ns_residual_by_rank"], state["ns_results"], state["tickers"],
            state["prices"], state["MATURITY"],
            state["price_universe"], trade_max_rank=COMBINED_TRADE_MAX_RANK,
            lambda_trade=LAMBDA_TRADE,
        )

    prices                     = state["prices"]
    tickers                    = state["tickers"]
    MATURITY                   = state["MATURITY"]
    BUSINESS_DAYS_TO_MONTH_END = state["BUSINESS_DAYS_TO_MONTH_END"]
    price_universe             = state["price_universe"]
    ns_results                 = state["ns_results"]
    ns_residual_by_rank        = state["ns_residual_by_rank"]

    print("Computing NS alpha panel …")
    alpha_panel = compute_ns_alpha_panel(
        ns_residual_by_rank, ns_results, MATURITY,
        price_universe, COMBINED_TRADE_MAX_RANK, LAMBDA_TRADE,
    )

    print("Building hysteresis weights …")
    hyst_weights, hyst_positions, z_scores = build_hysteresis_weights(
        residual_weights, alpha_panel,
        z_entry=z_entry, ewma_halflife=ewma_halflife,
    )

    print("Running hysteresis book …")
    hyst_book = run_contract_positions(
        hyst_weights, prices, tickers, BUSINESS_DAYS_TO_MONTH_END, price_universe,
        max_rank=COMBINED_TRADE_MAX_RANK,
        depth_panel=None,
        min_depth=COMBINED_TRADE_MAX_RANK,
        lag=LAG_BASE,
        cost_bps=cost_bps,
        market_aggregation="equal_available",
    )

    gross = hyst_book["gross"].dropna()
    net   = hyst_book["net"].dropna()

    return {
        "portfolio_stats":  performance_stats(net),
        "gross_pnl":        gross,
        "net_pnl":          net,
        "turnover":         hyst_book["turnover"],
        "market_gross":     hyst_book["market_gross"],
        "hyst_positions":   hyst_positions,
        "z_scores":         z_scores,
        "alpha_panel":      alpha_panel,
        "LAMBDA_TRADE":     LAMBDA_TRADE,
        "lambda_choice":    lambda_choice,
        "residual_weights": residual_weights,
        "state":            state,
    }
