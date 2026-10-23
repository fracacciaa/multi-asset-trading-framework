"""
Combined NS residual-reversal: 4 power markets + 13 general commodities.

Usage:
    from pathlib import Path
    from commodities.combined_ns import run_production

    result = run_production(data_dir=Path("~/Downloads").expanduser(), cost_bps=2.0)
    print(result["portfolio_stats"])
"""

from commodities.combined_ns.config   import COMBINED_MARKETS, POWER_4_MARKETS
from commodities.combined_ns.data     import load_combined_data
from commodities.combined_ns.pipeline import load_and_prepare, run_production, run_production_hysteresis

__all__ = [
    "COMBINED_MARKETS",
    "POWER_4_MARKETS",
    "load_combined_data",
    "load_and_prepare",
    "run_production",
    "run_production_hysteresis",
]
