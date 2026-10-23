"""
Combined data loader: merges 4 power markets (M1-M4) with 13 general
commodity markets on the union of their trading calendars.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from commodities.combined_ns.config import POWER_4_MARKETS, COMBINED_MARKETS, COMBINED_TRADE_MAX_RANK


def load_combined_data(data_dir: Optional[Path] = None) -> dict:
    """
    Load and merge power (4 markets, M1-M4) + general commodity data.

    Uses the union of the two trading calendars; each market is NaN on days
    its exchange does not trade, and the execution engine handles this naturally.

    Returns the same dict structure as load_power_data / load_gen_data:
        prices, tickers, open_interest, aggregate_oi,
        market_max_rank, price_universe, TRADING_INDEX
    """
    from commodities.power_ns.data import load_power_data
    from commodities.gen_ns.data   import load_gen_data

    if data_dir is None:
        data_dir = Path.cwd()
    data_dir = Path(data_dir).expanduser()

    print("  Loading power data (4 markets) …")
    pw = load_power_data(data_dir)

    print("  Loading general commodity data …")
    gn = load_gen_data(data_dir)

    # ── Filter power panels to 4 markets, M1-M4 only ─────────────────────────
    power_cols = [
        f"{m}{k}"
        for m in POWER_4_MARKETS
        for k in range(1, COMBINED_TRADE_MAX_RANK + 1)
    ]

    def _filter_power(df):
        keep = [c for c in power_cols if c in df.columns]
        return df[keep]

    pw_prices  = _filter_power(pw["prices"])
    pw_tickers = _filter_power(pw["tickers"])
    pw_oi      = _filter_power(pw["open_interest"])
    pw_agg_oi  = _filter_power(pw["aggregate_oi"])

    # ── Union trading calendar ────────────────────────────────────────────────
    idx = pw_prices.index.union(gn["prices"].index).sort_values()

    def _merge(left, right):
        return pd.concat(
            [left.reindex(idx), right.reindex(idx)],
            axis=1,
        )

    prices        = _merge(pw_prices,  gn["prices"])
    tickers       = _merge(pw_tickers, gn["tickers"])
    open_interest = _merge(pw_oi,      gn["open_interest"])
    aggregate_oi  = _merge(pw_agg_oi,  gn["aggregate_oi"])

    # ── Market depth (all 4 for every market in the combined universe) ─────────
    depth = {}
    for root in COMBINED_MARKETS:
        cols_needed = [f"{root}{k}" for k in range(1, COMBINED_TRADE_MAX_RANK + 1)]
        if all(c in prices.columns and c in tickers.columns for c in cols_needed):
            depth[root] = COMBINED_TRADE_MAX_RANK

    market_max_rank = pd.Series(depth, dtype=int)
    price_universe  = list(market_max_rank.index)

    return {
        "prices":          prices,
        "tickers":         tickers,
        "open_interest":   open_interest,
        "aggregate_oi":    aggregate_oi,
        "volume":          None,
        "market_max_rank": market_max_rank,
        "price_universe":  price_universe,
        "TRADING_INDEX":   idx,
    }
