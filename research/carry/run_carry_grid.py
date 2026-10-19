"""Run the carry signal grid on US and EUR OIS and print spec results."""
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from rates.ois_data import fetch_ois_curves
from rates.carry.experiment import run_grid, GRID

MARKETS = ["US", "EUR"]

def _funding(curves: pd.DataFrame) -> pd.Series:
    funding_tenor = 0.25
    if funding_tenor in curves.columns:
        return curves[funding_tenor]
    nearest = min(curves.columns, key=lambda x: abs(x - funding_tenor))
    return curves[nearest]


def _print_results(market: str, results: list) -> None:
    W = 110
    print(f"\n{'='*W}")
    print(f"  CARRY GRID — {market}  ({len(results)} surviving specs)")
    print(f"{'='*W}")
    hdr = (
        f"{'Spec':<22} {'Sharpe':>7} {'MaxDD':>8} {'CarryAnn':>9} "
        f"{'YldChgAnn':>10} {'HitRate':>8} {'Turnover':>9} "
        f"{'dd2013':>8} {'dd2022':>8} {'CorrRes':>8}"
    )
    print(hdr)
    print("-" * W)
    for r in results:
        corr = f"{r.corr_to_residual_book:.3f}" if r.corr_to_residual_book is not None else "  n/a"
        print(
            f"{r.spec_id:<22} {r.sharpe:>7.3f} {r.max_dd:>8.1f} {r.carry_component_ann:>9.1f} "
            f"{r.yldchg_component_ann:>10.1f} {r.hit_rate:>8.3f} {r.turnover:>9.4f} "
            f"{r.dd_2013:>8.1f} {r.dd_2022:>8.1f} {corr:>8}"
        )


for market in MARKETS:
    print(f"\nLoading {market} OIS curves...", flush=True)
    curves = fetch_ois_curves(market)
    funding = _funding(curves)
    print(f"  {len(curves)} dates, {len(curves.columns)} tenors", flush=True)

    print(f"Running grid ({len(GRID)} specs)...", flush=True)
    results = run_grid(GRID, curves, funding)
    _print_results(market, results)

print("\nDone.")
