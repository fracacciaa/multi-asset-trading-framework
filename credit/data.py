"""
Credit data ingestion.
Fetches CDS spreads, bond yields and index data for the credit universe.
Vendor/source to be configured via config.py.
"""

import pandas as pd


def fetch_cds_spreads(names: list[str], tenors: list[str], start: str, end: str) -> pd.DataFrame:
    """Return CDS spreads indexed by date."""
    raise NotImplementedError

def fetch_bond_yields(issuers: list[str], start: str, end: str) -> pd.DataFrame:
    """Return corporate bond yields and OAS indexed by date."""
    raise NotImplementedError

def fetch_index_returns(indices: list[str], start: str, end: str) -> pd.DataFrame:
    """Return total return index data (e.g. IG, HY indices) indexed by date."""
    raise NotImplementedError

def fetch_universe() -> list[str]:
    """Return the current credit universe (issuers or index buckets)."""
    raise NotImplementedError
