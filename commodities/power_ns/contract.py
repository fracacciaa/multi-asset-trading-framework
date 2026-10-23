"""
Contract ticker parsing and maturity panel construction.

Replicates notebook cells 20 (parse_contract_month_year), 21
(months_to_maturity, business_days_to_delivery_month_end, vectorized panels).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from commodities.power_ns.config import MONTH_CODES

# ── Scalar helpers ────────────────────────────────────────────────────────────


def parse_contract_month_year(ticker, observation_date):
    """
    Parse Bloomberg power futures ticker into (year, month) or None.

    Handles formats: ZUBJ16, ITPQ6, PHBK9M09, PHBQ6M, PCBJ9M09, PCBQ6M.
    Uses the FIRST futures month-code/year pair after the root.
    One-digit years are resolved relative to observation_date.
    """
    if pd.isna(ticker):
        return None

    ticker = str(ticker).strip().upper()
    m = re.search(r"([FGHJKMNQUVXZ])(\d{1,2})", ticker)
    if m is None:
        return None

    month_code = m.group(1)
    year_code  = m.group(2)
    month      = MONTH_CODES[month_code]
    obs_year   = observation_date.year

    if len(year_code) == 2:
        year = 1900 + int(year_code)
        if year < obs_year - 50:
            year += 100
    else:
        digit = int(year_code)
        decade = (obs_year // 10) * 10
        candidates = [decade - 10 + digit, decade + digit, decade + 10 + digit]
        year = min(candidates, key=lambda y: abs(y - obs_year))

    return year, month


def months_to_maturity(ticker, observation_date) -> float:
    """
    Approximate NS maturity coordinate in months.

    Used ONLY by the Nelson-Siegel design matrix, not for roll protection.
    """
    parsed = parse_contract_month_year(ticker, observation_date)
    if parsed is None:
        return np.nan

    year, month = parsed
    month_distance = 12 * (year - observation_date.year) + month - observation_date.month
    days_in_month  = observation_date.days_in_month
    fraction       = (days_in_month - observation_date.day) / days_in_month
    M = month_distance + fraction
    return M if M > 0 else np.nan


def business_days_to_delivery_month_end(ticker, observation_date) -> float:
    """
    Business days remaining until the first day of the month AFTER the contract's
    delivery month (the roll deadline).

    Used for roll protection, not the NS maturity coordinate.
    """
    parsed = parse_contract_month_year(ticker, observation_date)
    if parsed is None:
        return np.nan

    year, month = parsed
    obs        = pd.Timestamp(observation_date).normalize()
    month_start = pd.Timestamp(year=int(year), month=int(month), day=1)
    next_month  = month_start + pd.offsets.MonthBegin(1)

    if obs >= next_month:
        return 0.0

    return float(np.busday_count(
        np.datetime64(obs.date()),
        np.datetime64(next_month.date()),
    ))


# ── Vectorized batch builder ──────────────────────────────────────────────────


def _vectorized_maturity_and_roll(
    ticker_series: pd.Series,
    idx: pd.DatetimeIndex,
):
    """
    Build maturity and business-days-to-roll arrays for one ticker column.

    Returns (maturity_arr, business_days_arr), both float64.
    """
    # Keep as object dtype throughout; nullable StringDtype causes issues on older pandas.
    s = ticker_series.map(
        lambda x: str(x).strip().upper() if pd.notna(x) else np.nan
    ).astype(object)

    parts    = s.str.extract(r"([FGHJKMNQUVXZ])(\d{1,2})")
    month    = pd.to_numeric(parts[0].map(MONTH_CODES), errors="coerce").to_numpy(dtype=float)
    year_num = pd.to_numeric(parts[1], errors="coerce").to_numpy(dtype=float)
    year_len = parts[1].map(lambda x: len(x) if isinstance(x, str) else np.nan).to_numpy(dtype=float)

    obs_year = idx.year.to_numpy(dtype=int)
    year     = np.full(len(idx), np.nan, dtype=float)

    two_digit = (year_len == 2) & np.isfinite(year_num)
    if two_digit.any():
        yy = 1900.0 + year_num[two_digit]
        yy = np.where(yy < obs_year[two_digit] - 50, yy + 100.0, yy)
        year[two_digit] = yy

    one_digit = (year_len == 1) & np.isfinite(year_num)
    if one_digit.any():
        digit      = year_num[one_digit]
        decade     = (obs_year[one_digit] // 10) * 10
        candidates = np.column_stack([
            decade - 10 + digit,
            decade + digit,
            decade + 10 + digit,
        ])
        pick = np.argmin(np.abs(candidates - obs_year[one_digit, None]), axis=1)
        year[one_digit] = candidates[np.arange(len(candidates)), pick]

    valid    = np.isfinite(year) & np.isfinite(month)
    maturity = np.full(len(idx), np.nan, dtype=float)
    bdays    = np.full(len(idx), np.nan, dtype=float)

    ii = np.flatnonzero(valid)
    if len(ii):
        idx_month = idx.month.to_numpy(dtype=int)
        idx_day   = idx.day.to_numpy(dtype=int)
        idx_dim   = idx.days_in_month.to_numpy(dtype=int)

        month_distance = 12.0 * (year[ii] - obs_year[ii]) + month[ii] - idx_month[ii]
        fraction       = (idx_dim[ii] - idx_day[ii]) / idx_dim[ii]
        m_vals         = month_distance + fraction
        maturity[ii]   = np.where(m_vals > 0, m_vals, np.nan)

        contract_start = pd.to_datetime(pd.DataFrame({
            "year":  year[ii].astype(int),
            "month": month[ii].astype(int),
            "day":   1,
        })).to_numpy(dtype="datetime64[D]")
        next_month = (
            pd.DatetimeIndex(contract_start.astype("datetime64[ns]"))
            + pd.offsets.MonthBegin(1)
        ).to_numpy(dtype="datetime64[D]")
        obs = idx.to_numpy(dtype="datetime64[D]")[ii]
        bd  = np.busday_count(obs, next_month).astype(float)
        bdays[ii] = np.where(obs >= next_month, 0.0, bd)

    return maturity, bdays


# ── Public panel builder ──────────────────────────────────────────────────────


def build_maturity_panels(tickers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build MATURITY and BUSINESS_DAYS_TO_MONTH_END panels from the tickers DataFrame.

    Returns (MATURITY, BUSINESS_DAYS_TO_MONTH_END), both indexed like tickers.
    """
    MATURITY                  = pd.DataFrame(index=tickers.index, columns=tickers.columns, dtype=float)
    BUSINESS_DAYS_TO_MONTH_END = pd.DataFrame(index=tickers.index, columns=tickers.columns, dtype=float)

    for col in tickers.columns:
        m_arr, b_arr = _vectorized_maturity_and_roll(tickers[col], tickers.index)
        MATURITY[col]                   = m_arr
        BUSINESS_DAYS_TO_MONTH_END[col] = b_arr

    return MATURITY, BUSINESS_DAYS_TO_MONTH_END


def validate_maturity_panels(
    tickers: pd.DataFrame,
    MATURITY: pd.DataFrame,
    BUSINESS_DAYS_TO_MONTH_END: pd.DataFrame,
    n_checks: int = 8,
    seed: int = 12345,
) -> None:
    """
    Spot-check vectorized panels against scalar reference implementations.
    Raises AssertionError on mismatch.
    """
    rng   = np.random.default_rng(seed)
    cols  = list(tickers.columns)
    n_col = min(n_checks, len(cols))
    for col in rng.choice(cols, size=n_col, replace=False):
        rows = rng.choice(len(tickers.index), size=min(20, len(tickers.index)), replace=False)
        for i in rows:
            t = tickers[col].iloc[i]
            d = tickers.index[i]
            m_s = months_to_maturity(t, d)
            b_s = business_days_to_delivery_month_end(t, d)
            m_v = MATURITY[col].iloc[i]
            b_v = BUSINESS_DAYS_TO_MONTH_END[col].iloc[i]
            assert (pd.isna(m_s) and pd.isna(m_v)) or np.isclose(m_s, m_v, atol=1e-12), (
                f"Maturity mismatch for {col} on {d}: {m_s} vs {m_v}"
            )
            assert (pd.isna(b_s) and pd.isna(b_v)) or np.isclose(b_s, b_v, atol=1e-12), (
                f"Roll-calendar mismatch for {col} on {d}: {b_s} vs {b_v}"
            )
