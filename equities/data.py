"""
Equities data ingestion.
Fetches OHLCV prices and fundamentals for the equities universe.
Vendor/source to be configured via config.py.
"""

import pandas as pd


def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Return a DataFrame of adjusted close prices indexed by date."""
    raise NotImplementedError

def fetch_fundamentals(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Return point-in-time fundamental data (e.g. P/E, dividend yield) indexed by date."""
    raise NotImplementedError

def fetch_universe() -> list[str]:
    """Return the current equities universe."""
    raise NotImplementedError
