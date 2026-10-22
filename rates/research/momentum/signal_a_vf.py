"""
ViewFormation wrapper for the daily TSMOM direction signal.

Input to ViewFormation: sign(252-day rolling sum of ret_d) — pure ±1 direction,
no vol scaling. ViewFormation handles all risk normalisation internally (TS: per-
instrument total vol; XS: idiosyncratic vol + Grinold + tanh).

Covariance: EWMA halflife=63d of ret_d across all (market, tenor) instruments,
estimated on lagged returns (no lookahead). Daily units as required by both VF classes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from rates.production.momentum.signal_a import PRIMARY_TENORS

COV_HL    = 63     # EWMA halflife for covariance, days
DAILY_K   = 252    # rolling window for direction signal
_SEP      = "_"    # separator for flat ticker string "market_tenor"


def _flat(market: str, tenor: float) -> str:
    return f"{market}{_SEP}{tenor}"


def _unpack(ticker: str) -> tuple[str, float]:
    parts = ticker.rsplit(_SEP, 1)
    return parts[0], float(parts[1])


def build_signal_a_vf(
    panels_by_market: dict[str, dict[str, pd.DataFrame]],
    vf_class,
    vf_params: Optional[dict] = None,
    tenors: Optional[list[float]] = None,
    k: int = DAILY_K,
    cov_halflife: int = COV_HL,
) -> pd.DataFrame:
    """
    Apply ViewFormation (XS or TS) to the daily TSMOM direction signal.

    Parameters
    ----------
    panels_by_market : output of build_all_return_panels()
    vf_class         : ViewFormationXS or ViewFormationTS
    vf_params        : dict forwarded to vf_class constructor (e.g. desired_vol)
    tenors           : constant-maturity tenors; defaults to PRIMARY_TENORS
    k                : rolling window for direction signal (default 252 days)
    cov_halflife     : EWMA halflife for covariance estimation (default 63 days)

    Returns
    -------
    DataFrame with MultiIndex columns (market, tenor), index = daily dates.
    Values are VF portfolio weights (vol-scaled to desired_vol, default 10% ann.).
    Apply shift(1) before multiplying by ret_d to get PnL.
    """
    if tenors is None:
        tenors = PRIMARY_TENORS

    # ── Build flat ret_d and direction signal ──────────────────────────────
    ret_pieces: dict[str, pd.Series] = {}
    for mkt, panels in panels_by_market.items():
        ret_d = panels["ret_d"]
        for tau in tenors:
            if tau in ret_d.columns:
                ret_pieces[_flat(mkt, tau)] = ret_d[tau]

    if not ret_pieces:
        return pd.DataFrame()

    ret_flat  = pd.DataFrame(ret_pieces).sort_index()            # date × ticker
    sign_flat = np.sign(ret_flat.rolling(k).sum())               # ±1, no vol scaling

    # ── Rolling EWMA covariance (lagged, daily units) ──────────────────────
    cov_panel = ret_flat.shift(1).ewm(halflife=cov_halflife, min_periods=30).cov()
    cov_dates = set(cov_panel.index.get_level_values(0))

    # ── Per-date VF ────────────────────────────────────────────────────────
    weights: dict[pd.Timestamp, pd.Series] = {}
    for t in sign_flat.dropna(how="all").index:
        if t not in cov_dates:
            continue
        sig_t = sign_flat.loc[t].dropna()
        if len(sig_t) < 4:
            continue
        cov_t = cov_panel.loc[t].reindex(index=sig_t.index, columns=sig_t.index)
        if cov_t.isna().all().all():
            continue
        try:
            w = vf_class(sig_t, cov_t, vf_params).generate_view()
            weights[t] = w
        except Exception:
            continue

    if not weights:
        return pd.DataFrame()

    wt_flat = pd.DataFrame(weights).T
    wt_flat.index.name = "date"

    mi = pd.MultiIndex.from_tuples(
        [_unpack(c) for c in wt_flat.columns],
        names=["market", "tenor"],
    )
    wt_flat.columns = mi
    return wt_flat
