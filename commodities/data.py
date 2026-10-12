"""
Commodities data ingestion.
Parses Bloomberg wide-format Excel exports with 4 contracts per commodity.

Layout: 13 commodities × 8 columns each (4 contracts × 2 cols: date + price).
"""

import pandas as pd
import numpy as np
from functools import lru_cache
from typing import Optional

_PATH = "/Users/francescocaccia/Downloads/GEN_PRICE.xlsx"
_STRIDE = 8   # columns per commodity
_N_CONTRACTS = 4


def load_bloomberg_excel(path: str) -> dict[str, pd.DataFrame]:
    """Parse a Bloomberg wide-format futures Excel export.

    Each commodity occupies 8 consecutive columns:
      col+0: date  col+1: F1  col+2: date  col+3: F2
      col+4: date  col+5: F3  col+6: date  col+7: F4

    Returns dict with keys 'front', 'second', 'third', 'fourth':
    each a date × ticker DataFrame, sorted and forward-filled.
    """
    raw = pd.read_excel(path, header=0)

    n_commodities = raw.shape[1] // _STRIDE
    frames: dict[str, dict[str, pd.Series]] = {
        "front": {}, "second": {}, "third": {}, "fourth": {}
    }
    contracts = ["front", "second", "third", "fourth"]

    for i in range(n_commodities):
        base = i * _STRIDE

        # Commodity name: strip trailing digit (and spaces) from the price column header
        col_name = str(raw.columns[base + 1])   # e.g. "CL1", "C 1", "W 1"
        ticker = col_name[:-1].strip()           # "CL", "C", "W"

        # Use the date column for this commodity (same for all 4 contracts within a commodity)
        dates = pd.to_datetime(raw.iloc[:, base], errors="coerce")
        valid = dates.notna()
        idx = dates[valid].values

        for c, key in enumerate(contracts):
            price_col = base + 2 * c + 1
            prices = pd.to_numeric(raw.iloc[:, price_col][valid], errors="coerce")
            frames[key][ticker] = pd.Series(prices.values, index=idx, name=ticker)

    return {
        key: pd.DataFrame(val).sort_index().ffill()
        for key, val in frames.items()
    }


@lru_cache(maxsize=1)
def _cached_load(path: str) -> dict[str, pd.DataFrame]:
    return load_bloomberg_excel(path)


def fetch_prices(
    start: Optional[str] = None,
    end: Optional[str] = None,
    path: str = _PATH,
) -> pd.DataFrame:
    """Front-month futures prices (date × ticker)."""
    df = _cached_load(path)["front"]
    return _slice(df, start, end)


def fetch_term_structure(
    start: Optional[str] = None,
    end: Optional[str] = None,
    path: str = _PATH,
) -> dict[str, pd.DataFrame]:
    """Term structure DataFrames keyed by 'front', 'second', 'third', 'fourth'."""
    data = _cached_load(path)
    return {k: _slice(v, start, end) for k, v in data.items()}


def fetch_universe(path: str = _PATH) -> list[str]:
    return list(_cached_load(path)["front"].columns)


def _slice(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if start:
        df = df.loc[start:]
    if end:
        df = df.loc[:end]
    return df
