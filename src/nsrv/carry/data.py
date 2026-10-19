"""
Funding-rate loader for the carry signal.

Each curve_id needs one short rate per day in bp.

Preferred sources (carry_signal_spec.md §2.2):
  DM  : 3M OIS        → ois3m   (extracted from existing OIS curve data)
  EM  : onshore 3M depo or 3M bill

For DM, the OIS 3M tenor (0.25Y) already exists in the OIS panel; no
separate file is required.  This module extracts it from the existing
fetch_ois_curves loader as the default path.

For a file-based loader (future EM data), use load_funding_from_file().

Validation rules:
  - curve_id present in curve panel but no funding → raise if require_funding=True
  - |rf| > 10000 bp → raise (impossible rate)
  - gaps ≤ 5 business days → forward-fill, record fill mask
  - gaps > 5 business days → raise
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

# Local
try:
    from rates.ois_data import fetch_ois_curves, OIS_MARKETS
except ImportError:
    fetch_ois_curves = None
    OIS_MARKETS = []

# Short-rate tenor proxies per funding source
_FUNDING_TENORS: dict[str, float] = {
    "ois3m": 0.25,
    "bill3m": 0.25,
    "policy": 1 / 12,
}

MAX_GAP_DAYS = 5
MAX_ABS_RF_BP = 10_000.0


def load_funding_from_ois(
    markets: list[str],
    source: str = "ois3m",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extract the short-rate proxy directly from the OIS curve panel.

    Returns a date × market DataFrame with funding rates in bp.
    Uses the 0.25Y (3M) OIS tenor as the default.
    """
    if fetch_ois_curves is None:
        raise ImportError("rates.ois_data not available")

    tenor = _FUNDING_TENORS.get(source, 0.25)
    frames: dict[str, pd.Series] = {}

    for mkt in markets:
        try:
            df = fetch_ois_curves(mkt, start=start, end=end)
        except Exception as exc:
            warnings.warn(f"Could not load OIS curves for {mkt}: {exc}")
            continue

        # Find closest available tenor to target
        available = df.columns.to_numpy(float)
        if len(available) == 0:
            warnings.warn(f"No OIS tenors found for {mkt}")
            continue
        closest = float(available[np.argmin(np.abs(available - tenor))])
        rf_pct = df[closest]
        rf_bp = rf_pct * 100.0  # % → bp
        frames[mkt] = rf_bp.rename(mkt)

    if not frames:
        raise ValueError("No funding series could be loaded for any market")

    result = pd.DataFrame(frames)
    result.index.name = "date"
    return result


def load_funding_from_file(
    path: str,
    *,
    units: str = "pct",
) -> pd.DataFrame:
    """
    Load funding rates from a CSV/Excel file.

    Expected columns: date, curve_id, rf
    Returns a date × curve_id DataFrame with rf in bp.
    """
    if path.endswith(".csv"):
        raw = pd.read_csv(path, parse_dates=["date"])
    else:
        raw = pd.read_excel(path, parse_dates=["date"])

    raw = raw.set_index("date")
    if "curve_id" in raw.columns and "rf" in raw.columns:
        result = raw.pivot(columns="curve_id", values="rf")
    else:
        result = raw  # assume date × curve_id layout already

    if units == "pct":
        result = result * 100.0  # % → bp
    elif units == "decimal":
        result = result * 10_000.0  # decimal → bp

    return result


def validate_funding(
    funding: pd.DataFrame,
    curve_ids: list[str],
    *,
    require_funding: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate and forward-fill the funding DataFrame.

    Returns (funding_clean, fill_mask) where fill_mask is True at filled dates.
    Raises on impossible rates or excessive gaps.
    """
    if require_funding:
        missing = [c for c in curve_ids if c not in funding.columns]
        if missing:
            raise ValueError(
                f"require_funding=True but no funding series for: {missing}"
            )

    funding = funding.reindex(columns=[c for c in curve_ids if c in funding.columns])

    # Impossible rate check
    too_large = (funding.abs() > MAX_ABS_RF_BP).any()
    if too_large.any():
        bad = list(too_large[too_large].index)
        raise ValueError(f"|rf| > {MAX_ABS_RF_BP} bp in markets: {bad}")

    # Gap check and forward-fill
    fill_mask = pd.DataFrame(False, index=funding.index, columns=funding.columns)

    for col in funding.columns:
        s = funding[col]
        nan_locs = s.isna()
        if not nan_locs.any():
            continue

        # Find runs of NaNs
        runs = (nan_locs != nan_locs.shift()).cumsum()
        for _, group in s.groupby(runs):
            if group.isna().all():
                run_len = len(group)
                if run_len > MAX_GAP_DAYS:
                    raise ValueError(
                        f"Funding gap of {run_len} days (> {MAX_GAP_DAYS}) in {col} "
                        f"around {group.index[0].date()}"
                    )
                fill_mask.loc[group.index, col] = True

    funding_clean = funding.ffill()
    return funding_clean, fill_mask
