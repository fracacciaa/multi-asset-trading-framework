"""
Full pipeline runner: data → signals → per-asset blend → cross-asset blend.

Steps
-----
1. Equities   run_eq_prod()       daily return series (IS + OOS)
2. Rates      run_blend()         daily return series (IS + OOS)
3. FX         run_fx_blend()      monthly return series (IS + OOS)
4. Commodities run_comm_blend()   daily return series (IS + OOS)
5. Cross-asset run_asset_blend()  monthly blended portfolio (Rolling Ridge MSRR)
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from asset_blend import run_asset_tc

if __name__ == "__main__":
    r = run_asset_tc(verbose=True, plot=True)
    print(f"\nProduction return (base TC, Adaptive Ridge): "
          f"{len(r)} monthly observations, "
          f"{r.index[0].strftime('%Y-%m')} → {r.index[-1].strftime('%Y-%m')}")
