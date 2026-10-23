"""
Bloomberg Excel data loading for power futures.

Replicates notebook cells 5–6: load_and_align_excel, canonicalize_power_columns,
merge_curve, resolve_input_file, and the main load_power_data() entry point.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from commodities.power_ns.config import (
    POWER_MARKETS,
    EXCLUDED_POWER_MARKETS,
    MAX_AVAILABLE_RANK,
    MIN_OBS_FOR_RANK,
)

# ── Low-level helpers ─────────────────────────────────────────────────────────


def convert_excel_dates(s: pd.Series) -> pd.Series:
    """
    Convert a Bloomberg Excel date column robustly.

    Bloomberg exports can mix true Excel datetimes and raw Excel serial numbers.
    Numeric cells are interpreted as Excel day serials (origin 1899-12-30);
    all other cells are parsed as normal datetimes.
    """
    s = s.copy()
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce")

    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    def _is_numeric(x):
        return (
            isinstance(x, (int, float, np.integer, np.floating))
            and not isinstance(x, (bool, np.bool_))
            and pd.notna(x)
        )

    numeric_mask = s.map(_is_numeric)

    if numeric_mask.any():
        serial = pd.to_numeric(s.loc[numeric_mask], errors="coerce")
        out.loc[numeric_mask] = pd.to_datetime(
            serial, unit="D", origin="1899-12-30", errors="coerce",
        ).to_numpy()

    if (~numeric_mask).any():
        out.loc[~numeric_mask] = pd.to_datetime(
            s.loc[~numeric_mask], errors="coerce",
        ).to_numpy()

    return out


def clean_contract_name(x) -> str:
    return str(x).replace(" ", "")


def load_and_align_excel(
    path: str,
    numeric: bool = True,
    sheet_name=0,
) -> pd.DataFrame:
    """Load Bloomberg paired (date, value) columns onto one DatetimeIndex."""
    raw = pd.read_excel(path, sheet_name=sheet_name)
    aligned = []

    for i in range(0, raw.shape[1] - 1, 2):
        name = clean_contract_name(raw.columns[i + 1])
        tmp = raw.iloc[:, [i, i + 1]].copy()
        tmp.columns = ["Date", name]
        tmp["Date"] = convert_excel_dates(tmp["Date"])

        if numeric:
            tmp[name] = pd.to_numeric(tmp[name], errors="coerce")

        tmp = (
            tmp.dropna(subset=["Date"])
               .drop_duplicates(subset="Date", keep="last")
               .set_index("Date")
        )
        aligned.append(tmp)

    df = pd.concat(aligned, axis=1, join="outer", sort=True).sort_index()
    df.index = pd.DatetimeIndex(df.index)
    df.index.name = "Date"
    return df


def canonicalize_power_columns(df: pd.DataFrame, max_rank: int = 8) -> pd.DataFrame:
    """Canonicalize generic columns, e.g. PHB5M → PHB5, for ranks 1..max_rank."""
    rename = {}
    for col in df.columns:
        s = clean_contract_name(col)
        m = re.fullmatch(r"([A-Za-z]+)(\d+)M?", s)
        if m and 1 <= int(m.group(2)) <= max_rank:
            rename[col] = f"{m.group(1)}{int(m.group(2))}"
        else:
            rename[col] = s
    return df.rename(columns=rename)


def merge_curve(front: pd.DataFrame, deferred: pd.DataFrame) -> pd.DataFrame:
    """
    Join M5-M8 onto the M1-M4 trading calendar.

    The M1-M4 date index is authoritative; deferred contracts are reindexed
    onto it to avoid adding artificial non-trading rows.
    """
    idx = front.index
    out = front.copy()
    for col in deferred.columns:
        if col not in out.columns:
            out[col] = deferred[col].reindex(idx)
    return out.sort_index()


def resolve_input_file(filename: str, data_dir: Optional[Path] = None) -> str:
    """
    Resolve either the canonical filename or a versioned browser copy
    such as POWER_PRICE(6).xlsx. If several versioned copies exist, use the
    highest numeric suffix deterministically.
    """
    p = Path(filename)
    if data_dir is not None:
        p = data_dir / p.name

    if p.exists():
        return str(p)

    search_dir = p.parent
    rx = re.compile(
        rf"^{re.escape(p.stem)}\((\d+)\){re.escape(p.suffix)}$",
        flags=re.IGNORECASE,
    )
    candidates = []
    for q in search_dir.glob(f"{p.stem}(*){p.suffix}"):
        m = rx.fullmatch(q.name)
        if m:
            candidates.append((int(m.group(1)), q))

    if not candidates:
        raise FileNotFoundError(
            f"Could not find {p} or a versioned copy such as "
            f"{p.stem}(N){p.suffix} in {search_dir.resolve()}"
        )

    chosen = max(candidates, key=lambda x: x[0])[1]
    return str(chosen)


# ── Universe helpers ──────────────────────────────────────────────────────────


def keep_research_universe(
    df: pd.DataFrame,
    universe: dict = POWER_MARKETS,
    max_rank: int = MAX_AVAILABLE_RANK,
) -> pd.DataFrame:
    keep = []
    roots = sorted(universe.keys(), key=len, reverse=True)
    for col in df.columns:
        s = str(col)
        for root in roots:
            tail = s[len(root):]
            if s.startswith(root) and tail.isdigit() and 1 <= int(tail) <= max_rank:
                keep.append(col)
                break
    return df.loc[:, keep].copy()


def curve_schema_depth(
    prices: pd.DataFrame,
    tickers: pd.DataFrame,
    universe: dict,
    max_available_rank: int = MAX_AVAILABLE_RANK,
) -> pd.Series:
    """
    Static schema upper bound: how deep can each market's curve go?

    A market enters the research universe only when M1-M4 columns exist in both
    panels. Deferred ranks are an upper bound; historical eligibility is determined
    causally inside estimate_ns_panel.
    """
    depth = {}
    for root in universe:
        n = 0
        for k in range(1, max_available_rank + 1):
            c = f"{root}{k}"
            if c not in prices.columns or c not in tickers.columns:
                break
            n = k
        if n >= 4:
            depth[root] = n
    return pd.Series(depth, dtype=int).sort_values(ascending=False)


# ── Main entry point ─────────────────────────────────────────────────────────


def load_power_data(data_dir: Optional[Path] = None) -> dict:
    """
    Load all power futures data files and return a consolidated dict.

    Parameters
    ----------
    data_dir : Path, optional
        Directory containing the Bloomberg Excel files. Defaults to Path.cwd().
        Pass Path("~/Downloads").expanduser() to use the standard download location.

    Returns
    -------
    dict with keys:
        prices, tickers, open_interest, aggregate_oi, volume,
        market_max_rank, price_universe, TRADING_INDEX
    """
    if data_dir is None:
        data_dir = Path.cwd()
    data_dir = Path(data_dir).expanduser()

    def _resolve(name):
        return resolve_input_file(name, data_dir)

    EXTRA_SHEETS  = {"prices": "price power", "tickers": "ticker power", "open_interest": "OI power"}
    VOLUME_SHEETS = {"front": "Sheet1", "deferred": "Sheet2"}

    prices        = canonicalize_power_columns(load_and_align_excel(_resolve("POWER_PRICE.xlsx"), True))
    tickers       = canonicalize_power_columns(load_and_align_excel(_resolve("POWER_TICKERS.xlsx"), False))
    open_interest = canonicalize_power_columns(load_and_align_excel(_resolve("POWER_OI.xlsx"), True))
    aggregate_oi  = canonicalize_power_columns(load_and_align_excel(_resolve("POWER_OI_AGG.xlsx"), True))

    volume_front    = canonicalize_power_columns(load_and_align_excel(_resolve("Volume_power.xlsx"), True, VOLUME_SHEETS["front"]))
    volume_deferred = canonicalize_power_columns(load_and_align_excel(_resolve("Volume_power.xlsx"), True, VOLUME_SHEETS["deferred"]))
    volume = merge_curve(volume_front, volume_deferred)

    extra_file = _resolve("ULTIMO_TOMMI_BBG.xlsx")
    prices        = merge_curve(prices,        canonicalize_power_columns(load_and_align_excel(extra_file, True,  EXTRA_SHEETS["prices"])))
    tickers       = merge_curve(tickers,        canonicalize_power_columns(load_and_align_excel(extra_file, False, EXTRA_SHEETS["tickers"])))
    open_interest = merge_curve(open_interest,  canonicalize_power_columns(load_and_align_excel(extra_file, True,  EXTRA_SHEETS["open_interest"])))

    # The M1-M4 price file defines the authoritative trading calendar.
    TRADING_INDEX = prices.index.copy()
    tickers       = tickers.reindex(TRADING_INDEX)
    open_interest = open_interest.reindex(TRADING_INDEX)
    aggregate_oi  = aggregate_oi.reindex(TRADING_INDEX)
    volume        = volume.reindex(TRADING_INDEX)

    # Restrict to the research universe.
    prices        = keep_research_universe(prices)
    tickers       = keep_research_universe(tickers)
    open_interest = keep_research_universe(open_interest)
    aggregate_oi  = keep_research_universe(aggregate_oi)
    volume        = keep_research_universe(volume)

    # Calendar-integrity assertions.
    _front_cal = canonicalize_power_columns(
        load_and_align_excel(_resolve("POWER_PRICE.xlsx"), True)
    ).index
    assert prices.index.equals(_front_cal), "Deferred data changed the M1-M4 trading calendar."
    assert tickers.index.equals(prices.index), "Ticker calendar differs from price calendar."
    assert open_interest.index.equals(prices.index), "OI calendar differs from price calendar."
    assert aggregate_oi.index.equals(prices.index), "Aggregate OI calendar differs."
    assert volume.index.equals(prices.index), "Volume calendar differs from price calendar."

    market_max_rank = curve_schema_depth(prices, tickers, POWER_MARKETS)
    price_universe  = list(market_max_rank.index)

    return {
        "prices":        prices,
        "tickers":       tickers,
        "open_interest": open_interest,
        "aggregate_oi":  aggregate_oi,
        "volume":        volume,
        "market_max_rank": market_max_rank,
        "price_universe":  price_universe,
        "TRADING_INDEX":   TRADING_INDEX,
    }
