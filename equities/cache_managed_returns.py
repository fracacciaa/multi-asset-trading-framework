"""
Regenerate equities/cache/managed_returns.parquet.

Run from the project root:
    python -m equities.cache_managed_returns
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from equities.data import fetch_prices, fetch_fundamentals
from equities.signals import build_vf_positions
from equities.blend import build_managed_returns
from equities.config import (
    IS_END, MANAGED_RETURNS_PATH,
    VF_WARMUP, VF_REBAL_FREQ, VF_COV_LOOKBACK, VF_VOL_HALFLIFE, VF_COR_HALFLIFE,
)


def main():
    print("Loading Bloomberg data...")
    prices       = fetch_prices(tickers=None)
    fundamentals = fetch_fundamentals(tickers=None)
    returns      = prices.pct_change().dropna(how="all")

    print("Building VF positions (monthly rebalance, warmup=504)...")
    vf_pos = build_vf_positions(
        prices, fundamentals, returns,
        warmup=VF_WARMUP, rebal_freq=VF_REBAL_FREQ,
        cov_lookback=VF_COV_LOOKBACK,
        vol_halflife=VF_VOL_HALFLIFE, cor_halflife=VF_COR_HALFLIFE,
    )

    print("Building managed returns F...")
    F = build_managed_returns(vf_pos, returns)

    os.makedirs(os.path.dirname(MANAGED_RETURNS_PATH), exist_ok=True)
    F.to_parquet(MANAGED_RETURNS_PATH)

    F_is  = F.loc[F.index <= IS_END]
    F_oos = F.loc[F.index >  IS_END]
    print(f"\nSaved  : {MANAGED_RETURNS_PATH}")
    print(f"IS_END : {IS_END}")
    print(f"IS     : {F_is.index[0].date()} → {F_is.index[-1].date()}  ({len(F_is)} rows)")
    print(f"OOS    : {F_oos.index[0].date()} → {F_oos.index[-1].date()}  ({len(F_oos)} rows)")


if __name__ == "__main__":
    main()
