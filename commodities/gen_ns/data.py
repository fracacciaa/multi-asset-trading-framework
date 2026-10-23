"""
Bloomberg Excel data loading for general commodity futures.

Loads GEN_PRICE.xlsx, GEN_TICKER.xlsx, GEN_OI.xlsx, GEN_OI_AGG.xlsx.
Same paired-column Bloomberg format as the power files; no deferred / M5+.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from commodities.power_ns.data import load_and_align_excel, resolve_input_file, keep_research_universe
from commodities.gen_ns.config import GEN_MARKETS, GEN_MAX_RANK


def load_gen_data(data_dir: Optional[Path] = None) -> dict:
    """
    Load general commodity futures data from Bloomberg Excel files.

    Returns
    -------
    dict with keys:
        prices, tickers, open_interest, aggregate_oi,
        market_max_rank, price_universe, TRADING_INDEX
    """
    if data_dir is None:
        data_dir = Path.cwd()
    data_dir = Path(data_dir).expanduser()

    def _resolve(name):
        return resolve_input_file(name, data_dir)

    prices        = load_and_align_excel(_resolve("GEN_PRICE.xlsx"),   numeric=True)
    tickers       = load_and_align_excel(_resolve("GEN_TICKER.xlsx"),  numeric=False)
    open_interest = load_and_align_excel(_resolve("GEN_OI.xlsx"),      numeric=True)
    aggregate_oi  = load_and_align_excel(_resolve("GEN_OI_AGG.xlsx"),  numeric=True)

    # Authoritative calendar: GEN_PRICE defines the trading index.
    TRADING_INDEX = prices.index.copy()
    tickers       = tickers.reindex(TRADING_INDEX)
    open_interest = open_interest.reindex(TRADING_INDEX)
    aggregate_oi  = aggregate_oi.reindex(TRADING_INDEX)

    # Filter to research universe.
    prices        = keep_research_universe(prices,        GEN_MARKETS, GEN_MAX_RANK)
    tickers       = keep_research_universe(tickers,       GEN_MARKETS, GEN_MAX_RANK)
    open_interest = keep_research_universe(open_interest, GEN_MARKETS, GEN_MAX_RANK)
    aggregate_oi  = keep_research_universe(aggregate_oi,  GEN_MARKETS, GEN_MAX_RANK)

    # Market depth: all gen markets have M1-M4 only.
    depth = {}
    for root in GEN_MARKETS:
        cols_needed = [f"{root}{k}" for k in range(1, GEN_MAX_RANK + 1)]
        if all(c in prices.columns and c in tickers.columns for c in cols_needed):
            depth[root] = GEN_MAX_RANK

    market_max_rank = pd.Series(depth, dtype=int)
    price_universe  = list(market_max_rank.index)

    return {
        "prices":          prices,
        "tickers":         tickers,
        "open_interest":   open_interest,
        "aggregate_oi":    aggregate_oi,
        "market_max_rank": market_max_rank,
        "price_universe":  price_universe,
        "TRADING_INDEX":   TRADING_INDEX,
    }
