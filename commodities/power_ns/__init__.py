"""
Power NS residual-reversal strategy package.

Public API:

    from commodities.power_ns import run_production, run_full_analysis, load_and_prepare
    from commodities.power_ns import performance_stats, sharpe, book_metrics

Usage:
    from pathlib import Path
    from commodities.power_ns import run_production

    result = run_production(data_dir=Path("~/Downloads").expanduser(), cost_bps=2.0)
    print(result["portfolio_stats"])
"""

from commodities.power_ns.pipeline import load_and_prepare, run_production, run_full_analysis
from commodities.power_ns.data     import load_power_data
from commodities.power_ns.metrics  import sharpe, drawdown, performance_stats, book_metrics
from commodities.power_ns.ns_fit   import ns_loadings, estimate_ns_panel, build_ns_residual_panels
from commodities.power_ns.contract import build_maturity_panels
from commodities.power_ns.execution import run_contract_positions, run_residual_portfolio

__all__ = [
    "load_and_prepare",
    "run_production",
    "run_full_analysis",
    "load_power_data",
    "sharpe",
    "drawdown",
    "performance_stats",
    "book_metrics",
    "ns_loadings",
    "estimate_ns_panel",
    "build_ns_residual_panels",
    "build_maturity_panels",
    "run_contract_positions",
    "run_residual_portfolio",
]
