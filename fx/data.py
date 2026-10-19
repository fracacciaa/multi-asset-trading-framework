"""
FX data ingestion — Bloomberg BDH export.

File: 'FXXXXXXXXX.xlsx'
Layout per sheet (one sheet per currency pair):
  Row 4  : block labels  "CATEGORY | TENOR"  at cols 0, 3, 6, ... (3-col stride)
  Row 5  : Bloomberg ticker
  Row 6  : DATE / PX_LAST column headers
  Row 7+ : data — col 3k = Excel serial date, col 3k+1 = value

31 data blocks per sheet:
  SPOT        (1 block)
  FWD         1M 3M 6M 12M 2Y   (5 blocks)  — raw Bloomberg forward points (pips)
  ATM         1M 3M 6M 1Y  2Y   (5 blocks)  — ATM implied vol, annualised %
  25RR / 25BF 1M 3M 6M 1Y  2Y   (10 blocks) — 25-delta risk-reversal / butterfly
  10RR / 10BF 1M 3M 6M 1Y  2Y   (10 blocks) — 10-delta risk-reversal / butterfly

Universe (24 pairs):
  DM (10): AUDUSD EURUSD GBPUSD NZDUSD USDCAD USDCHF USDDKK USDJPY USDNOK USDSEK
  EM (14): USDBRL USDKRW USDMXN USDSGD USDTWD USDZAR USDCZK USDCNY USDTHB USDPLN
           USDINR USDIDR USDHUF USDTRY

Quote direction:
  XXXUSD (AUDUSD EURUSD GBPUSD NZDUSD) — USD is second currency; higher = USD weaker.
  USDXXX (all others)                  — USD is first currency; higher = USD stronger.

Forward points:
  Bloomberg pips convention — divide by pip factor to get outright forward:
    XXXUSD / USDXXX with 4 decimal places → pip = 0.0001
    USDJPY / USDKRW / USDIDR               → pip = 0.01
    USDHUF / USDCZK / USDINR               → pip = 0.01
  Use pip_factor(pair) to convert; never add raw pips to spot without scaling.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

# ── File path ─────────────────────────────────────────────────────────────────

_FX_PATH = "/Users/francescocaccia/Downloads/FXXXXXXXXX.xlsx"

# ── Universe ──────────────────────────────────────────────────────────────────

FX_PAIRS: list[str] = [
    "AUDUSD", "EURUSD", "GBPUSD", "NZDUSD",
    "USDCAD", "USDCHF", "USDDKK", "USDJPY",
    "USDNOK", "USDSEK",
    "USDBRL", "USDKRW", "USDMXN", "USDSGD",
    "USDTWD", "USDZAR", "USDCZK", "USDCNY",
    "USDTHB", "USDPLN", "USDINR", "USDIDR",
    "USDHUF", "USDTRY",
]

DM_PAIRS: list[str] = [
    "AUDUSD", "EURUSD", "GBPUSD", "NZDUSD",
    "USDCAD", "USDCHF", "USDDKK", "USDJPY",
    "USDNOK", "USDSEK",
]

EM_PAIRS: list[str] = [p for p in FX_PAIRS if p not in DM_PAIRS]

# XXXUSD: USD is the second currency (EUR/GBP/AUD/NZD based)
XXXUSD_PAIRS: frozenset[str] = frozenset(["AUDUSD", "EURUSD", "GBPUSD", "NZDUSD"])

FWD_TENORS: list[str] = ["1M", "3M", "6M", "12M", "2Y"]
VOL_TENORS: list[str] = ["1M", "3M", "6M", "1Y", "2Y"]
VOL_CATEGORIES: list[str] = ["ATM", "25RR", "25BF", "10RR", "10BF"]

# ── Forward pip factors ───────────────────────────────────────────────────────

# Most pairs: 4 decimal places → pip = 0.0001
# JPY-based and some EM pairs: 2 decimal places → pip = 0.01
_PIP_100 = frozenset([
    "USDJPY", "USDKRW", "USDIDR", "USDHUF", "USDCZK",
    "USDTWD", "USDTHB", "USDINR",
])


def pip_factor(pair: str) -> float:
    """Return the pip divisor for converting Bloomberg forward points to outright."""
    return 100.0 if pair in _PIP_100 else 10_000.0


# ── Block layout ──────────────────────────────────────────────────────────────

_BLOCK_INDEX: dict[tuple[str, str], int] = {
    ("SPOT", "SPOT"): 0,
    ("FWD",  "1M"):   1,  ("FWD",  "3M"):  2,  ("FWD",  "6M"):  3,
    ("FWD",  "12M"):  4,  ("FWD",  "2Y"):  5,
    ("ATM",  "1M"):   6,  ("ATM",  "3M"):  7,  ("ATM",  "6M"):  8,
    ("ATM",  "1Y"):   9,  ("ATM",  "2Y"):  10,
    ("25RR", "1M"):  11,  ("25RR", "3M"): 12,  ("25RR", "6M"): 13,
    ("25RR", "1Y"):  14,  ("25RR", "2Y"): 15,
    ("25BF", "1M"):  16,  ("25BF", "3M"): 17,  ("25BF", "6M"): 18,
    ("25BF", "1Y"):  19,  ("25BF", "2Y"): 20,
    ("10RR", "1M"):  21,  ("10RR", "3M"): 22,  ("10RR", "6M"): 23,
    ("10RR", "1Y"):  24,  ("10RR", "2Y"): 25,
    ("10BF", "1M"):  26,  ("10BF", "3M"): 27,  ("10BF", "6M"): 28,
    ("10BF", "1Y"):  29,  ("10BF", "2Y"): 30,
}

_EXCEL_EPOCH = datetime(1899, 12, 30)


def _serial_to_dates(col: pd.Series) -> pd.DatetimeIndex:
    """Convert a column of Excel serial integers to DatetimeIndex."""
    return pd.to_datetime([
        _EXCEL_EPOCH + timedelta(days=int(d)) for d in col.astype(int)
    ])


def _read_block(raw: pd.DataFrame, block: int) -> pd.Series:
    """Extract one (date, value) block from the raw sheet, return date-indexed Series."""
    c0 = block * 3
    chunk = raw.iloc[7:, [c0, c0 + 1]].copy()
    chunk.columns = ["date", "value"]
    chunk = chunk.dropna(subset=["date"])
    chunk.index = _serial_to_dates(chunk["date"])
    chunk.index.name = "date"
    return pd.to_numeric(chunk["value"], errors="coerce").dropna()


# ── Per-pair cached loader ────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def _load_pair(pair: str, path: str = _FX_PATH) -> dict[tuple[str, str], pd.Series]:
    """Load all 31 blocks for a pair. Cached per (pair, path)."""
    raw = pd.read_excel(path, sheet_name=pair, header=None)
    out: dict[tuple[str, str], pd.Series] = {}
    for key, block in _BLOCK_INDEX.items():
        s = _read_block(raw, block)
        if not s.empty:
            out[key] = s
    return out


def _slice(s: pd.Series | pd.DataFrame, start, end):
    if start:
        s = s.loc[start:]
    if end:
        s = s.loc[:end]
    return s


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_spot(
    pair: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    path: str = _FX_PATH,
) -> pd.Series:
    """
    Spot exchange rate for one pair.

    Returns a date-indexed Series. XXXUSD pairs (EURUSD etc.) quote USD per
    foreign unit; USDXXX pairs (USDJPY etc.) quote foreign units per USD.
    """
    data = _load_pair(pair, path)
    s = data[("SPOT", "SPOT")].rename(pair)
    return _slice(s, start, end)


def fetch_forwards(
    pair: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    path: str = _FX_PATH,
) -> pd.DataFrame:
    """
    Forward points for one pair (date × tenor: 1M 3M 6M 12M 2Y).

    Units: Bloomberg raw pips. To convert to outright forward rate:
        fwd_rate = spot + forward_points / pip_factor(pair)
    """
    data = _load_pair(pair, path)
    df = pd.DataFrame(
        {tenor: data[("FWD", tenor)] for tenor in FWD_TENORS if ("FWD", tenor) in data}
    ).sort_index()
    df.index.name = "date"
    return _slice(df, start, end)


def fetch_atm_vol(
    pair: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    path: str = _FX_PATH,
) -> pd.DataFrame:
    """
    ATM implied volatility for one pair (date × tenor: 1M 3M 6M 1Y 2Y).

    Units: annualised %, e.g. 8.5 = 8.5% vol.
    """
    data = _load_pair(pair, path)
    df = pd.DataFrame(
        {tenor: data[("ATM", tenor)] for tenor in VOL_TENORS if ("ATM", tenor) in data}
    ).sort_index()
    df.index.name = "date"
    return _slice(df, start, end)


def fetch_vol_surface(
    pair: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    path: str = _FX_PATH,
) -> pd.DataFrame:
    """
    Full vol surface for one pair.

    Returns a DataFrame with MultiIndex columns (category, tenor).
    Categories: ATM 25RR 25BF 10RR 10BF.  Tenors: 1M 3M 6M 1Y 2Y.
    Units: annualised % for ATM; vol points (%) for risk-reversals and butterflies.
    """
    data = _load_pair(pair, path)
    frames: dict[tuple[str, str], pd.Series] = {}
    for cat in VOL_CATEGORIES:
        for tenor in VOL_TENORS:
            key = (cat, tenor)
            if key in data:
                frames[key] = data[key]
    df = pd.DataFrame(frames).sort_index()
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["category", "tenor"])
    df.index.name = "date"
    return _slice(df, start, end)


def fetch_fx_all(
    pair: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    path: str = _FX_PATH,
) -> dict[str, pd.Series | pd.DataFrame]:
    """
    All data for one pair in a single call.

    Returns dict with keys:
      'spot'    — pd.Series
      'fwd'     — pd.DataFrame (date × tenor), raw forward points
      'atm'     — pd.DataFrame (date × tenor), ATM vol %
      'surface' — pd.DataFrame (date × MultiIndex(category,tenor)), full vol surface
    """
    return {
        "spot":    fetch_spot(pair, start, end, path),
        "fwd":     fetch_forwards(pair, start, end, path),
        "atm":     fetch_atm_vol(pair, start, end, path),
        "surface": fetch_vol_surface(pair, start, end, path),
    }


def fetch_spot_panel(
    pairs: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    path: str = _FX_PATH,
) -> pd.DataFrame:
    """
    Spot rates for multiple pairs (date × pair).

    Defaults to the full 24-pair universe.
    """
    if pairs is None:
        pairs = FX_PAIRS
    series = {}
    for pair in pairs:
        try:
            series[pair] = fetch_spot(pair, start, end, path)
        except Exception:
            pass
    df = pd.DataFrame(series).sort_index()
    df.index.name = "date"
    return df
