"""
Equities data ingestion.
Parses Bloomberg wide-format Excel exports and exposes price / fundamental DataFrames.
"""

import os
import pandas as pd
from functools import lru_cache
from typing import Optional

_FIELDS = ["prices", "market_cap", "fcf", "shares_out", "eps_fwd"]
_FIELD_COLS = ["PX_LAST", "CUR_MKT_CAP", "TRAIL_12M_FREE_CASH_FLOW", "EQY_SH_OUT", "BEST_EPS"]
_STRIDE = 6  # columns per ticker: date + 5 fields


def load_bloomberg_excel(path: str) -> dict[str, pd.DataFrame]:
    """Parse a Bloomberg wide-format Excel export into per-field DataFrames.

    Expected layout: every 6 columns = one ticker.
    Col 0: Excel date serial  Col 1: PX_LAST  Col 2: CUR_MKT_CAP
    Col 3: TRAIL_12M_FREE_CASH_FLOW  Col 4: EQY_SH_OUT  Col 5: BEST_EPS

    Returns a dict with keys: 'prices', 'market_cap', 'fcf', 'shares_out', 'eps_fwd'.
    Each value is a date × ticker DataFrame, forward-filled and indexed by DatetimeIndex.
    """
    raw = pd.read_excel(path, header=None, engine="calamine")
    header = raw.iloc[0]
    data = raw.iloc[1:].reset_index(drop=True)

    n_tickers = raw.shape[1] // _STRIDE
    field_frames: dict[str, dict[str, pd.Series]] = {f: {} for f in _FIELDS}

    dates = None

    for i in range(n_tickers):
        base = i * _STRIDE
        ticker_label = str(header.iloc[base])
        # strip Bloomberg suffix: "AAPL UW Equity" → "AAPL"
        ticker = ticker_label.split()[0] if ticker_label else f"UNKNOWN_{i}"

        serial = pd.to_numeric(data.iloc[:, base], errors="coerce")
        if dates is None:
            dates = pd.to_datetime(serial.dropna().astype(int), unit="D", origin="1899-12-30")

        for j, field in enumerate(_FIELDS):
            col = data.iloc[:, base + 1 + j]
            field_frames[field][ticker] = pd.to_numeric(col, errors="coerce").values

    # Build index from first ticker's date column (all tickers share the same dates)
    index = pd.to_datetime(
        pd.to_numeric(data.iloc[:, 0], errors="coerce").dropna().astype(int),
        unit="D",
        origin="1899-12-30",
    )
    n_rows = len(index)

    result = {}
    for field in _FIELDS:
        df = pd.DataFrame(
            {ticker: series[:n_rows] for ticker, series in field_frames[field].items()},
            index=index,
        )
        df.index.name = "date"
        df = df.sort_index()
        # Forward-fill so weekends / missing dates carry the last known value
        df = df.ffill()
        result[field] = df

    return result


@lru_cache(maxsize=1)
def _cached_load(path: str) -> dict[str, pd.DataFrame]:
    cache_dir = os.path.join(os.path.dirname(path), ".bbg_cache")
    cache_stamp = os.path.join(cache_dir, os.path.basename(path) + ".stamp")
    cache_ok = (
        os.path.isdir(cache_dir)
        and os.path.exists(cache_stamp)
        and os.path.getmtime(cache_stamp) >= os.path.getmtime(path)
    )
    if cache_ok:
        return {
            field: pd.read_parquet(os.path.join(cache_dir, f"{field}.parquet"))
            for field in _FIELDS
        }
    data = load_bloomberg_excel(path)
    os.makedirs(cache_dir, exist_ok=True)
    for field, df in data.items():
        df.to_parquet(os.path.join(cache_dir, f"{field}.parquet"))
    open(cache_stamp, "w").close()
    return data


_SIGNALS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "signals")
_RETURNS_PARQUET = os.path.join(_SIGNALS_DIR, "returns.parquet")


def fetch_prices(
    tickers: Optional[list],
    start: Optional[str] = None,
    end: Optional[str] = None,
    path: str = "EQH saved.xlsx",
) -> pd.DataFrame:
    """
    Return adjusted close prices (date × ticker), optionally filtered.

    Prefers equities/signals/returns.parquet (daily return panel) when present,
    reconstructing a synthetic cumulative-price series so that callers' downstream
    pct_change() reproduces the original returns exactly.  Falls back to the
    Bloomberg Excel export when the parquet is absent.
    """
    if os.path.exists(_RETURNS_PARQUET):
        ret = pd.read_parquet(_RETURNS_PARQUET)
        prices = (1.0 + ret.fillna(0.0)).cumprod()
        return _slice(prices, tickers, start, end)
    from core.paths import raw as _resolve
    data = _cached_load(str(_resolve(path)))
    return _slice(data["prices"], tickers, start, end)


def fetch_fundamentals(
    tickers: Optional[list],
    start: Optional[str] = None,
    end: Optional[str] = None,
    path: str = "EQH saved.xlsx",
) -> dict:
    """Return fundamental DataFrames (market_cap, fcf, shares_out, eps_fwd), optionally filtered."""
    from core.paths import raw as _resolve
    data = _cached_load(str(_resolve(path)))
    return {f: _slice(data[f], tickers, start, end) for f in _FIELDS if f != "prices"}


def fetch_universe(path: str = "EQH saved.xlsx") -> list[str]:
    """Return all ticker symbols available in the data file."""
    from core.paths import raw as _resolve
    return list(_cached_load(str(_resolve(path)))["prices"].columns)


def _slice(
    df: pd.DataFrame,
    tickers: Optional[list],
    start: Optional[str],
    end: Optional[str],
) -> pd.DataFrame:
    if tickers:
        df = df[[t for t in tickers if t in df.columns]]
    if start:
        df = df.loc[start:]
    if end:
        df = df.loc[:end]
    return df
