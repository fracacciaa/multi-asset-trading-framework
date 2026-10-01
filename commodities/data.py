"""
Commodities data ingestion.
Fetches futures prices and term structure data for the commodities universe.
Vendor/source to be configured via config.py.
"""

import pandas as pd


def fetch_futures(contracts: list[str], start: str, end: str) -> pd.DataFrame:
    """Return front-month and second-month futures prices indexed by date."""
    raise NotImplementedError

def fetch_term_structure(contracts: list[str], start: str, end: str) -> pd.DataFrame:
    """Return full term structure (multiple expiries) for roll-yield computation."""
    raise NotImplementedError

def fetch_universe() -> list[str]:
    """Return the current commodities universe (futures contracts)."""
    raise NotImplementedError
