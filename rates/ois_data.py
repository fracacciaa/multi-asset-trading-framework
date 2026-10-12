"""
OIS (Overnight Index Swap) rate data loader.

File: 'SWAPPPPP RATES.xlsx'
Layout per sheet:
  Row 3  : tenor labels at cols 0, 3, 6, ...  (1M, 2M, ..., 12M, 1Y, 2Y, ..., 30Y)
  Row 4  : "Date" / Bloomberg ticker / blank repeated per tenor
  Row 5+ : data — col i = Excel serial date, col i+1 = rate (%)

Stride = 3 columns per tenor.

OIS bootstrap convention: annual coupon for all markets (freq=1).
Short end ≤ 1Y treated as zero coupon (single payment at maturity).

Market universe (15 markets):
  G10 — full curve (25-26 tenors), long history:
    US          → USDEFFR  (EFFR OIS, 2001+)
    EUR         → EUR      (€STR OIS, 2007+)
    UK          → GBP      (SONIA OIS, 2000+)
    Japan       → JPY      (TONA OIS, 2002+)
    Canada      → CAD      (CORRA OIS, 2002+)
    Australia   → AUD      (AONIA OIS, 2001+)
    New Zealand → NZD      (2002+, 25 tenors)
    Sweden      → SEK      (STIBOR OIS, 2004+)

  G10 — full curve, shorter history:
    Switzerland → CHF      (SARON OIS, 2009+)
    Norway      → NOK      (NOWA OIS, 2022+)

  EM — limited tenors (15-24) but tradeable:
    Turkey      → TRY      (2007+, 15 tenors)
    India       → INR      (1999+, 15 tenors)
    Colombia    → COP      (2015+, 17 tenors)
    China       → CNY      (2008+, 11 tenors — borderline MIN_GRID_PTS)
    Israel      → ILS      (2024+, 24 tenors — 2y history only)
    Philippines → PHP      (2024+, 12 tenors — 2y history only)

Returns: synthetic constant-maturity 10Y receiver swap
  return_t ≈ −duration × Δy_t / 100
"""

import re
import numpy as np
import pandas as pd
from functools import lru_cache
from typing import Optional

_OIS_PATH = "/Users/francescocaccia/Downloads/SWAPPPPP RATES.xlsx"

# Market display name → sheet name
OIS_SHEETS: dict[str, str] = {
    # G10 — full curve, long history
    "US":          "USDEFFR",
    "EUR":         "EUR",
    "UK":          "GBP",
    "Japan":       "JPY",
    "Canada":      "CAD",
    "Australia":   "AUD",
    "New Zealand": "NZD",
    "Sweden":      "SEK",
    # G10 — full curve, shorter history
    "Switzerland": "CHF",    # 2017+
    "Norway":      "NOK",    # 2022+
    # EM
    "Turkey":      "TRY",    # 2007+, 15 tenors
    "India":       "INR",    # 1999+, 15 tenors
    "Colombia":    "COP",    # 2015+, 17 tenors
    "China":       "CNY",    # 2008+, 11 tenors
    "Israel":      "ILS",    # 2024+, 24 tenors
    "Philippines": "PHP",    # 2024+, 12 tenors
}

OIS_MARKETS = list(OIS_SHEETS.keys())

# All OIS swaps: annual fixed payments for maturities > 1Y
OIS_COUPON_FREQ = 1


def _parse_tenor(label: str) -> Optional[float]:
    """Convert tenor label ('1M', '6M', '1Y', '10Y') to years."""
    label = label.strip().upper()
    m = re.match(r"^(\d+)(M|Y)$", label)
    if not m:
        return None
    num, unit = int(m.group(1)), m.group(2)
    return num / 12.0 if unit == "M" else float(num)


def _parse_ois_sheet(path: str, sheet: str) -> pd.DataFrame:
    """
    Parse one OIS sheet into a date × tenor (float years) DataFrame of rates (%).

    Handles deduplication: 12M and 1Y both map to 1.0Y and are averaged.
    """
    raw = pd.read_excel(path, sheet_name=sheet, header=None)

    # Row 3: tenor labels at cols 0, 3, 6, ...
    tenor_row = raw.iloc[3, :]
    n_cols = len(tenor_row)

    series_map: dict[float, list[pd.Series]] = {}
    for col_i in range(0, n_cols - 1, 3):
        tenor_label = tenor_row.iloc[col_i]
        if pd.isna(tenor_label):
            continue
        tenor_y = _parse_tenor(str(tenor_label))
        if tenor_y is None:
            continue

        date_col = col_i
        rate_col = col_i + 1

        dates_raw = pd.to_numeric(raw.iloc[5:, date_col], errors="coerce").dropna()
        if len(dates_raw) == 0:
            continue
        dates = pd.to_datetime(dates_raw.astype(int), unit="D", origin="1899-12-30")
        rates = pd.to_numeric(raw.iloc[dates_raw.index, rate_col], errors="coerce")

        s = pd.Series(rates.values, index=dates.values, name=tenor_y)
        series_map.setdefault(tenor_y, []).append(s)

    # Average duplicate tenors (12M and 1Y both → 1.0Y)
    combined: dict[float, pd.Series] = {}
    for tenor_y, series_list in series_map.items():
        c = pd.concat(series_list, axis=1).mean(axis=1)
        c.index = pd.to_datetime(c.index)
        combined[tenor_y] = c

    df = pd.DataFrame(combined)
    df.index.name = "date"
    df = df.sort_index()
    df.columns = pd.Index(sorted(df.columns), dtype=float)
    df = df.ffill()
    return df


@lru_cache(maxsize=4)
def _cached_load_ois(path: str) -> dict[str, pd.DataFrame]:
    """Load all unique OIS sheets and cache."""
    sheets_needed = set(OIS_SHEETS.values())
    return {sheet: _parse_ois_sheet(path, sheet) for sheet in sheets_needed}


def fetch_ois_curves(
    market: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    path: str = _OIS_PATH,
) -> pd.DataFrame:
    """Return date × tenor (years) DataFrame of OIS rates (%) for one market."""
    sheet = OIS_SHEETS[market]
    sheets = _cached_load_ois(path)
    df = sheets[sheet]
    if start:
        df = df.loc[start:]
    if end:
        df = df.loc[:end]
    return df


def fetch_ois_returns(
    market: str,
    duration: float = 10.0,
    start: Optional[str] = None,
    end: Optional[str] = None,
    path: str = _OIS_PATH,
) -> pd.Series:
    """
    Synthetic constant-maturity 10Y receiver swap daily return.

    return_t ≈ −duration × Δy_t / 100

    Positive return = rates fell (receiver swap profits).
    Uses the closest available tenor to `duration` years.
    """
    curves = fetch_ois_curves(market, start=start, end=end, path=path)
    available = curves.columns.tolist()
    closest = min(available, key=lambda t: abs(t - duration))
    y = curves[closest]
    return (-duration * y.diff() / 100).rename(market)


def fetch_ois_universe(path: str = _OIS_PATH) -> list[str]:
    """Return list of available OIS market display names."""
    return OIS_MARKETS
