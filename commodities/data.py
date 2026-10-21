"""
Commodities data ingestion.
Parses Bloomberg wide-format Excel exports.

Layout: N commodities × (2 × C) columns where C is the number of contracts.
Each commodity occupies a block of 2*C consecutive columns:
  col+0: date  col+1: F1  col+2: date  col+3: F2  ...  col+2c: date  col+2c+1: Fc

File map (all under ~/Downloads):
  GEN_PRICE (1).xlsx          prices,   contracts 1-4, 21 commodities
  GEN_OI.xlsx                 OI,       contracts 1-4, 21 commodities
  GEN_OI_AGG.xlsx             agg OI,   front only,    21 commodities
  GEN_TICKER (1).xlsx         tickers,  contracts 1-4, 21 commodities
  ULTIMO TOMMI BBG (2).xlsx   prices/OI/tickers, contracts 5-8, 21 commodities
"""

import pandas as pd
import numpy as np
from functools import lru_cache
from typing import Optional

_BASE = "/Users/francescocaccia/Downloads/"

_PATH_PRICE  = _BASE + "GEN_PRICE (1).xlsx"
_PATH_OI     = _BASE + "GEN_OI.xlsx"
_PATH_OI_AGG = _BASE + "GEN_OI_AGG.xlsx"
_PATH_TICKER = _BASE + "GEN_TICKER (1).xlsx"
_PATH_FAR    = _BASE + "ULTIMO TOMMI BBG (2).xlsx"

_FAR_SHEET = {"price": "gen price", "oi": "gen OI", "ticker": "gen ticker"}

_CONTRACT_NAMES = ["front", "second", "third", "fourth",
                   "fifth", "sixth", "seventh", "eighth"]

# Legacy alias kept for external callers that pin the old default path.
_PATH = _PATH_PRICE

_EXCEL_EPOCH = pd.Timestamp("1899-12-30")


def _coerce_dates(col: pd.Series) -> pd.Series:
    """
    Parse a date column that may contain Timestamps or Excel serial integers.

    Bloomberg exports sometimes store dates as Excel day-count integers (e.g.
    33220 = 1990-12-13) rather than formatted date cells.  pd.to_datetime treats
    bare integers as nanoseconds from epoch, producing wrong results, so we
    convert those cells explicitly via the Excel 1900 epoch.
    """
    result = pd.to_datetime(col, errors="coerce")
    numeric_mask = col.apply(
        lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and not pd.isna(x)
    )
    if numeric_mask.any():
        result = result.copy()
        result[numeric_mask] = col[numeric_mask].apply(
            lambda x: _EXCEL_EPOCH + pd.Timedelta(days=int(x))
        )
    return result


# ── Core parser ───────────────────────────────────────────────────────────────

def _parse_bloomberg_wide(
    raw: pd.DataFrame,
    n_contracts: int = 4,
    numeric: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Parse a Bloomberg wide-format DataFrame.

    Returns a dict keyed by contract name ('front', 'second', ...).
    With numeric=False (tickers), string values are preserved and not ffilled.
    """
    stride = 2 * n_contracts
    n_commodities = raw.shape[1] // stride
    keys = _CONTRACT_NAMES[:n_contracts]
    frames: dict[str, dict[str, pd.Series]] = {k: {} for k in keys}

    for i in range(n_commodities):
        base = i * stride
        col_name = str(raw.columns[base + 1])   # e.g. "CL1", "C 1", "W 1"
        ticker = col_name[:-1].strip()           # "CL", "C", "W"

        dates = _coerce_dates(raw.iloc[:, base])
        valid = dates.notna()
        idx = dates[valid].values

        for c, key in enumerate(keys):
            col_data = raw.iloc[:, base + 2 * c + 1][valid]
            if numeric:
                values = pd.to_numeric(col_data, errors="coerce")
            else:
                values = col_data.where(col_data.notna()).astype(object)
            frames[key][ticker] = pd.Series(values.values, index=idx, name=ticker)

    result = {}
    for key, val in frames.items():
        df = pd.DataFrame(val).sort_index()
        result[key] = df.ffill() if numeric else df
    return result


@lru_cache(maxsize=16)
def _cached_parse(
    path: str,
    sheet_name: object,    # str | int — must be hashable
    n_contracts: int,
    numeric: bool,
) -> dict[str, pd.DataFrame]:
    raw = pd.read_excel(path, sheet_name=sheet_name, header=0)
    return _parse_bloomberg_wide(raw, n_contracts=n_contracts, numeric=numeric)


def load_bloomberg_excel(path: str) -> dict[str, pd.DataFrame]:
    """Parse a Bloomberg wide-format futures price export (4 contracts)."""
    return _cached_parse(path, 0, 4, True)


# ── Prices ────────────────────────────────────────────────────────────────────

def fetch_prices(
    start: Optional[str] = None,
    end: Optional[str] = None,
    path: str = _PATH_PRICE,
) -> pd.DataFrame:
    """Front-month futures prices (date × ticker)."""
    df = _cached_parse(path, 0, 4, True)["front"]
    return _slice(df, start, end)


def fetch_term_structure(
    start: Optional[str] = None,
    end: Optional[str] = None,
    path: str = _PATH_PRICE,
) -> dict[str, pd.DataFrame]:
    """Term structure keyed 'front'–'fourth' (contracts 1–4)."""
    data = _cached_parse(path, 0, 4, True)
    return {k: _slice(v, start, end) for k, v in data.items()}


def fetch_term_structure_far(
    start: Optional[str] = None,
    end: Optional[str] = None,
    path: str = _PATH_FAR,
) -> dict[str, pd.DataFrame]:
    """Term structure keyed 'fifth'–'eighth' (contracts 5–8)."""
    data = _cached_parse(path, _FAR_SHEET["price"], 4, True)
    rename = dict(zip(["front", "second", "third", "fourth"],
                      ["fifth", "sixth", "seventh", "eighth"]))
    return {rename[k]: _slice(v, start, end) for k, v in data.items()}


def fetch_universe(path: str = _PATH_PRICE) -> list[str]:
    """List of commodity tickers available in the price file."""
    return list(_cached_parse(path, 0, 4, True)["front"].columns)


# ── Open Interest ─────────────────────────────────────────────────────────────

def fetch_open_interest(
    start: Optional[str] = None,
    end: Optional[str] = None,
    path: str = _PATH_OI,
) -> dict[str, pd.DataFrame]:
    """Open interest keyed 'front'–'fourth' (contracts 1–4)."""
    data = _cached_parse(path, 0, 4, True)
    return {k: _slice(v, start, end) for k, v in data.items()}


def fetch_oi_agg(
    start: Optional[str] = None,
    end: Optional[str] = None,
    path: str = _PATH_OI_AGG,
) -> pd.DataFrame:
    """Aggregated open interest (all expirations), front-month row (date × ticker)."""
    df = _cached_parse(path, 0, 1, True)["front"]
    return _slice(df, start, end)


def fetch_open_interest_far(
    start: Optional[str] = None,
    end: Optional[str] = None,
    path: str = _PATH_FAR,
) -> dict[str, pd.DataFrame]:
    """Open interest keyed 'fifth'–'eighth' (contracts 5–8)."""
    data = _cached_parse(path, _FAR_SHEET["oi"], 4, True)
    rename = dict(zip(["front", "second", "third", "fourth"],
                      ["fifth", "sixth", "seventh", "eighth"]))
    return {rename[k]: _slice(v, start, end) for k, v in data.items()}


# ── Tickers ───────────────────────────────────────────────────────────────────

def fetch_tickers(
    start: Optional[str] = None,
    end: Optional[str] = None,
    path: str = _PATH_TICKER,
) -> dict[str, pd.DataFrame]:
    """Active contract tickers keyed 'front'–'fourth' (contracts 1–4)."""
    data = _cached_parse(path, 0, 4, False)
    return {k: _slice(v, start, end) for k, v in data.items()}


def fetch_tickers_far(
    start: Optional[str] = None,
    end: Optional[str] = None,
    path: str = _PATH_FAR,
) -> dict[str, pd.DataFrame]:
    """Active contract tickers keyed 'fifth'–'eighth' (contracts 5–8)."""
    data = _cached_parse(path, _FAR_SHEET["ticker"], 4, False)
    rename = dict(zip(["front", "second", "third", "fourth"],
                      ["fifth", "sixth", "seventh", "eighth"]))
    return {rename[k]: _slice(v, start, end) for k, v in data.items()}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _slice(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if start:
        df = df.loc[start:]
    if end:
        df = df.loc[:end]
    return df
