"""
Rates data ingestion.
Parses Bloomberg wide-format Excel export (alternating date/rate column pairs).
"""

import re
import pandas as pd
import numpy as np
from functools import lru_cache
from typing import Optional

_PATH = "/Users/francescocaccia/Downloads/RATESSSSS.xlsx"
_SHEET = "Sheet2"

# Bloomberg prefix → display name
MARKET_PREFIXES: dict[str, str] = {
    "USGG":  "US",
    "GDBR":  "Germany",
    "GUKG":  "UK",
    "GJGB":  "Japan",
    "GFRN":  "France",
    "GCAN":  "Canada",
    "GACGB": "Australia",
}


def _parse_ticker(name: str) -> tuple[Optional[str], Optional[float]]:
    """Return (bloomberg_prefix, tenor_years) from a Bloomberg ticker column header."""
    name = name.replace(" Index", "").strip()
    m = re.match(r"^([A-Z]+?)(\d+)(M|YR|Y?)$", name)
    if not m:
        return None, None
    prefix, num, unit = m.group(1), int(m.group(2)), m.group(3)
    tenor = num / 12.0 if unit == "M" else float(num)
    return prefix, tenor


def load_bloomberg_excel(path: str) -> dict[str, pd.DataFrame]:
    """Parse Sheet2 into per-market date × tenor DataFrames.

    Sheet2 layout: named columns are Excel-serial date columns; the immediately
    following unnamed column is the rate. Stride is irregular (extra blank columns
    between tickers are skipped by this parser).

    Returns dict keyed by display name (e.g. 'US', 'Germany') containing
    date × tenor (float years) DataFrames of continuously compounded yields (%).
    """
    raw = pd.read_excel(path, sheet_name=_SHEET, header=0)
    cols = list(raw.columns)

    # Collect (market_display, tenor) → list of Series
    series_map: dict[tuple[str, float], list[pd.Series]] = {}

    for i, col in enumerate(cols):
        if col.startswith("Unnamed"):
            continue
        prefix, tenor = _parse_ticker(col)
        if prefix is None or tenor is None:
            continue
        display = MARKET_PREFIXES.get(prefix)
        if display is None:
            continue
        if i + 1 >= len(cols):
            continue

        # Date column: convert Excel serial → datetime
        dates_raw = pd.to_numeric(raw.iloc[:, i], errors="coerce").dropna()
        dates = pd.to_datetime(dates_raw.astype(int), unit="D", origin="1899-12-30")

        # Rate column: next unnamed column
        rates = pd.to_numeric(raw.iloc[dates_raw.index, i + 1], errors="coerce")

        s = pd.Series(rates.values, index=dates.values, name=tenor)
        key = (display, tenor)
        series_map.setdefault(key, []).append(s)

    # Build per-market DataFrames; average duplicate tenors
    market_series: dict[str, dict[float, pd.Series]] = {}
    for (display, tenor), series_list in series_map.items():
        combined = pd.concat(series_list, axis=1).mean(axis=1)
        combined.name = tenor
        combined.index = pd.to_datetime(combined.index)
        market_series.setdefault(display, {})[tenor] = combined

    # Remove "year" tenors that are exact duplicates of their month-equivalent.
    # Bloomberg sometimes exports month-maturity tickers without the M suffix
    # (e.g. USGG6 = USGG6M = 6-month data, incorrectly parsed as 6-year).
    for display, tenor_map in market_series.items():
        bad = []
        for t_y, s_y in list(tenor_map.items()):
            t_m = t_y / 12.0   # what this would be if num meant months
            if t_m in tenor_map and t_m != t_y:
                s_m = tenor_map[t_m]
                common = s_y.index.intersection(s_m.index)
                if len(common) > 100 and (s_y.loc[common] - s_m.loc[common]).abs().max() < 1e-9:
                    bad.append(t_y)
        for t in bad:
            del tenor_map[t]

    result: dict[str, pd.DataFrame] = {}
    for display, tenor_map in market_series.items():
        df = pd.DataFrame(tenor_map)
        df.index.name = "date"
        df = df.sort_index()
        df.columns = sorted(df.columns)
        df = df.ffill()
        result[display] = df

    return result


@lru_cache(maxsize=1)
def _cached_load(path: str) -> dict[str, pd.DataFrame]:
    return load_bloomberg_excel(path)


def fetch_curves(
    market: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    path: str = _PATH,
) -> pd.DataFrame:
    """Return date × tenor (years) DataFrame of yields (%) for one market."""
    data = _cached_load(path)
    df = data[market]
    if start:
        df = df.loc[start:]
    if end:
        df = df.loc[:end]
    return df


def fetch_universe(path: str = _PATH) -> list[str]:
    """Return list of available market display names."""
    return list(_cached_load(path).keys())


def fetch_returns(
    market: str,
    duration: float = 10.0,
    start: Optional[str] = None,
    end: Optional[str] = None,
    path: str = _PATH,
) -> pd.Series:
    """Synthetic constant-maturity zero-coupon bond daily return.

    return_t ≈ −duration × Δy_t / 100
    Uses the closest available tenor to `duration` years.
    """
    curves = fetch_curves(market, start, end, path)
    available = curves.columns.tolist()
    closest = min(available, key=lambda t: abs(t - duration))
    y = curves[closest]
    return (-duration * y.diff() / 100).rename(market)
