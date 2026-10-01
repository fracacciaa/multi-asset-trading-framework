"""
FX data ingestion.
Fetches spot rates, forward rates and interest rate differentials.
Vendor/source to be configured via config.py.
"""

import pandas as pd


def fetch_spot(pairs: list[str], start: str, end: str) -> pd.DataFrame:
    """Return spot exchange rates indexed by date."""
    raise NotImplementedError

def fetch_forwards(pairs: list[str], tenors: list[str], start: str, end: str) -> pd.DataFrame:
    """Return forward exchange rates for given tenors (e.g. 1M, 3M, 1Y)."""
    raise NotImplementedError

def fetch_interest_rate_differentials(pairs: list[str], start: str, end: str) -> pd.DataFrame:
    """Return short-rate differentials used for carry construction."""
    raise NotImplementedError

def fetch_universe() -> list[str]:
    """Return the current FX universe (currency pairs)."""
    raise NotImplementedError
