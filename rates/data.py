"""
Rates data ingestion.
Fetches yield curves, futures prices and carry data for the rates universe.
Vendor/source to be configured via config.py.
"""

import pandas as pd


def fetch_yields(tenors: list[str], countries: list[str], start: str, end: str) -> pd.DataFrame:
    """Return a DataFrame of government bond yields indexed by date."""
    raise NotImplementedError

def fetch_futures(contracts: list[str], start: str, end: str) -> pd.DataFrame:
    """Return adjusted futures prices for rate futures (e.g. Eurodollar, Bund)."""
    raise NotImplementedError

def fetch_universe() -> list[str]:
    """Return the current rates universe (country x tenor combinations)."""
    raise NotImplementedError
