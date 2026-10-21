"""
FX production data: orientation, forward outrights, excess return panels.

Three orientation traps (spec §0.2):
  1. Quotation: invert XXXUSD pairs to FCU-per-USD before everything.
  2. Forward points: auto-calibrate divisor per pair (never hardcode).
  3. Risk reversals: flip USDXXX pairs so all RR = IV(FCU call) - IV(FCU put).

Excludes four NDF pairs (BRL TWD INR IDR) that have no forward data here.
Primary sample start: 1999-01-01.  Option surface start: 2003-10-01.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from fx.data import (
    FX_PAIRS,
    XXXUSD_PAIRS,
    fetch_spot,
    fetch_forwards,
    fetch_vol_surface,
    pip_factor as _hardcoded_divisor,
)

# ── Universe ──────────────────────────────────────────────────────────────────

FCU_BASE: frozenset[str] = XXXUSD_PAIRS        # AUDUSD EURUSD GBPUSD NZDUSD

# Four NDF pairs: no outright forward in this workbook
NO_FWD_PAIRS: frozenset[str] = frozenset(["USDBRL", "USDTWD", "USDINR", "USDIDR"])

CARRY_PAIRS: list[str] = [p for p in FX_PAIRS if p not in NO_FWD_PAIRS]  # 20 pairs

PRIMARY_START  = "1999-01-01"
OPTION_START   = "2003-10-01"

# synthetic round-trip transaction cost placeholders (bp)
TC_BP: dict[str, float] = {
    **{p: 2.0  for p in ["AUDUSD", "EURUSD", "GBPUSD", "NZDUSD",
                          "USDCAD", "USDCHF", "USDDKK", "USDJPY",
                          "USDNOK", "USDSEK"]},
    **{p: 5.0  for p in ["USDKRW", "USDSGD", "USDCZK", "USDCNY",
                          "USDTHB", "USDPLN", "USDHUF"]},
    **{p: 10.0 for p in ["USDMXN", "USDZAR", "USDBRL", "USDTRY"]},
}

# ── Orientation helpers ───────────────────────────────────────────────────────

def _to_fcu_per_usd(pair: str, px: pd.Series) -> pd.Series:
    return 1.0 / px if pair in FCU_BASE else px.copy()


def _rr_foreign_oriented(pair: str, rr: pd.Series) -> pd.Series:
    """Flip USDXXX pairs: Bloomberg quotes IV(USD call)-IV(USD put), we want FCU."""
    return rr.copy() if pair in FCU_BASE else -rr


# ── Divisor calibration ───────────────────────────────────────────────────────

_DIVISOR_CANDIDATES = [1.0, 10.0, 100.0, 1_000.0, 10_000.0]


def calibrate_divisor(spot: pd.Series, points: pd.Series) -> float:
    """
    Pick the divisor making the annualised 1M forward discount land near 2%.

    Rejects any candidate whose max absolute discount exceeds 60%
    (catches a wrong-order-of-magnitude divisor on extreme-carry pairs).
    Falls back to the hardcoded pip_factor if no candidate passes.
    """
    idx = spot.index.intersection(points.index)
    s = spot.loc[idx].dropna()
    p = points.loc[idx].reindex(s.index).dropna()
    s = s.reindex(p.index)
    if len(s) < 12:
        return np.nan

    best: Optional[float] = None
    best_score = np.inf
    for dv in _DIVISOR_CANDIDATES:
        fd = (p / dv) / s * 12.0
        if float(np.nanmax(np.abs(fd))) > 0.60:
            continue
        score = abs(float(np.nanmedian(np.abs(fd))) - 0.02)
        if score < best_score:
            best_score = score
            best = dv
    return best if best is not None else np.nan


# ── Outright forward ──────────────────────────────────────────────────────────

def _fwd_outright_fcu(pair: str, spot_raw: pd.Series,
                      points: pd.Series, divisor: float) -> pd.Series:
    """
    Returns the 1M forward outright in FCU-per-USD.

    Inversion for FCU-base pairs happens AFTER adding points to the quoted spot.
    Inverting spot first and then adding points is wrong (spec §1.2).
    """
    F_quoted = spot_raw + points / divisor   # still in Bloomberg quote convention
    return _to_fcu_per_usd(pair, F_quoted)


# ── Public panel builder ──────────────────────────────────────────────────────

def build_panels(
    pairs: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    path: Optional[str] = None,
) -> dict:
    """
    Build return and vol-surface panels for the given pairs.

    Returns
    -------
    dict with keys:
      s        — log spot FCU/USD, month-end × pair
      f_1m     — log 1M forward outright FCU/USD, month-end × pair
      fd_1m    — log forward discount (f_1m - s), month-end × pair
      rx_m     — monthly excess return, month-end × pair (return known at t+1)
      rx_d     — daily excess return, daily × pair
      atm_1m   — ATM 1M vol %, month-end × pair
      rr25_1m  — 25Δ 1M RR foreign-oriented, month-end × pair
      rr10_1m  — 10Δ 1M RR foreign-oriented, month-end × pair
      bf25_1m  — 25Δ 1M BF (orientation-free), month-end × pair
      bf10_1m  — 10Δ 1M BF, month-end × pair
      divisors — dict pair → divisor used
      excluded — list of pairs excluded and why
    """
    if pairs is None:
        pairs = CARRY_PAIRS
    loader_kw: dict = {}
    if path:
        loader_kw["path"] = path

    excluded: list[str] = []
    divisors: dict[str, float] = {}

    s_raw: dict[str, pd.Series] = {}   # raw daily log spot (FCU/USD)
    f1m_raw: dict[str, pd.Series] = {} # raw daily log 1M fwd (FCU/USD)
    fd1m_raw: dict[str, pd.Series] = {}
    atm: dict[str, pd.Series] = {}
    rr25: dict[str, pd.Series] = {}
    rr10: dict[str, pd.Series] = {}
    bf25: dict[str, pd.Series] = {}
    bf10: dict[str, pd.Series] = {}

    for pair in pairs:
        if pair in NO_FWD_PAIRS:
            excluded.append(pair + ":no_forward_data")
            continue
        try:
            spot_raw = fetch_spot(pair, start=start, end=end, **loader_kw)
            if spot_raw.empty or (spot_raw <= 0).any():
                spot_raw = spot_raw[spot_raw > 0]
            if len(spot_raw) < 12:
                excluded.append(pair + ":insufficient_spot")
                continue

            fwd_raw = fetch_forwards(pair, start=start, end=end, **loader_kw)
            if "1M" not in fwd_raw.columns or fwd_raw["1M"].dropna().empty:
                excluded.append(pair + ":missing_1M_forward")
                continue

            dv = calibrate_divisor(spot_raw, fwd_raw["1M"])
            if np.isnan(dv):
                dv = float(_hardcoded_divisor(pair))
                warnings.warn(f"{pair}: divisor calibration failed, using hardcoded {dv}")
            divisors[pair] = dv

            spot_fcu = _to_fcu_per_usd(pair, spot_raw)
            f1m_fcu  = _fwd_outright_fcu(pair, spot_raw, fwd_raw["1M"], dv)

            valid = spot_fcu > 0
            s_log  = np.log(spot_fcu[valid])
            f_log  = np.log(f1m_fcu.reindex(s_log.index))

            s_raw[pair]   = s_log
            f1m_raw[pair] = f_log
            fd1m_raw[pair] = f_log - s_log

            surf = fetch_vol_surface(pair, start=start, end=end, **loader_kw)
            if not surf.empty:
                def _col(cat, tenor):
                    return surf[(cat, tenor)] if (cat, tenor) in surf.columns else None

                c = _col("ATM", "1M")
                if c is not None:
                    atm[pair] = c
                c = _col("25RR", "1M")
                if c is not None:
                    rr25[pair] = _rr_foreign_oriented(pair, c)
                c = _col("10RR", "1M")
                if c is not None:
                    rr10[pair] = _rr_foreign_oriented(pair, c)
                c = _col("25BF", "1M")
                if c is not None:
                    bf25[pair] = c
                c = _col("10BF", "1M")
                if c is not None:
                    bf10[pair] = c

        except Exception as e:
            warnings.warn(f"{pair}: {e}")
            excluded.append(f"{pair}:error:{e}")

    def _panel(d: dict[str, pd.Series]) -> pd.DataFrame:
        if not d:
            return pd.DataFrame()
        return pd.DataFrame(d).sort_index()

    def _to_me(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        return df.resample("BME").last()

    # ── Daily excess returns ───────────────────────────────────────────────────
    # rx_d[d] = fd_1m[prev month-end] / business_days_in_current_month - Δs[d]
    rx_d_dict: dict[str, pd.Series] = {}
    for pair in s_raw:
        s_ser  = s_raw[pair]
        fd_ser = fd1m_raw[pair]
        fd_me  = fd_ser.resample("BME").last()
        # fd from the START of the holding month = previous month-end value
        fd_prev = fd_me.shift(1).reindex(s_ser.index, method="ffill")
        # business-day count for the CURRENT month (bfill: each day gets its own month's count)
        bdays   = s_ser.resample("BME").count()
        bdays_c = bdays.reindex(s_ser.index, method="bfill").clip(lower=1)
        rx_d_dict[pair] = fd_prev / bdays_c - s_ser.diff()

    # ── Monthly excess returns ─────────────────────────────────────────────────
    # rx_m[t+1] = f_1m[t] - s[t+1], both at month-end
    rx_m_dict: dict[str, pd.Series] = {}
    for pair in s_raw:
        s_me  = s_raw[pair].resample("BME").last()
        f_me  = f1m_raw[pair].resample("BME").last()
        rx_m_dict[pair] = f_me.shift(1) - s_me

    return {
        "s":        _to_me(_panel(s_raw)),
        "f_1m":     _to_me(_panel(f1m_raw)),
        "fd_1m":    _to_me(_panel(fd1m_raw)),
        "rx_m":     _panel(rx_m_dict),
        "rx_d":     _panel(rx_d_dict),
        "atm_1m":   _to_me(_panel(atm)),
        "rr25_1m":  _to_me(_panel(rr25)),
        "rr10_1m":  _to_me(_panel(rr10)),
        "bf25_1m":  _to_me(_panel(bf25)),
        "bf10_1m":  _to_me(_panel(bf10)),
        "divisors": divisors,
        "excluded": excluded,
    }
