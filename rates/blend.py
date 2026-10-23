"""
Rates signal blend layer.

Loads the three production PnL parquets, aggregates across markets, and
exposes the same blend API as equities/blend.py (all algorithms are imported
directly — no duplication).

Pipeline position:
    rates/signals/*_pnl.parquet  →  load_managed_returns()  →  F (T × 3)
    F  →  fit_static / MLMSRR / LinearDynamicBlend / …  →  blend weights
    blend weights  →  combined portfolio return

Signals (production, from rates/signals/):
    momentum        momentum_a_daily_pnl.parquet   daily TSMOM (9 markets)
    carry           carry_ois_par_pnl.parquet       within-curve OIS carry (9 markets)
    mean_reversion  mean_reversion_pnl.parquet      factor-neutral NS residuals (11 markets)

Each PnL parquet is date × market (bp, daily); load_managed_returns sums
each signal across the common market intersection to produce a single
time-series per signal (the cross-market portfolio managed return).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Re-export every blend algorithm from the equities layer ──────────────────
from equities.blend import (                                        # noqa: F401
    BlendWeights,
    equal_weight,
    sharpe_heuristic,
    msrr_ridge,
    combine_signals,
    build_managed_returns,
    build_state_features,
    enet_blend,
    c_enet_blend,
    e_enet_blend,
    inv_msfe_blend,
    pcr_blend,
    pls_blend,
    winsorize_signals,
    pool_blend,
    fit_static,
    MLMSRR,
    LinearDynamicBlend,
    GBRTBlend,
    RandomForestBlend,
    DMABlend,
    STMOM,
)

# ── Rates-specific constants ─────────────────────────────────────────────────

_SIGNALS_DIR = Path(__file__).parent / "signals"

SIGNAL_FILES: dict[str, str] = {
    "momentum":       "momentum_a_daily_pnl.parquet",
    "carry":          "carry_ois_par_pnl.parquet",
    "mean_reversion": "mean_reversion_pnl.parquet",
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_managed_returns(
    signals: dict[str, str] | None = None,
    signals_dir: str | Path | None = None,
    markets: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load rates PnL parquets and build the managed-return matrix F (T × K).

    Each signal's cross-market portfolio return at time t is the sum of
    per-market PnL values on that date (after restricting to the common
    market intersection).  This is the rates analog of
    equities.blend.build_managed_returns.

    Parameters
    ----------
    signals     Map of signal_name → parquet filename.  Default: SIGNAL_FILES.
    signals_dir Directory containing the parquet files.  Default: rates/signals/.
    markets     Restrict to these markets before summing.  Default: all common
                markets present in every signal file.

    Returns
    -------
    pd.DataFrame, shape (T, K), index=date, columns=signal names.
    Only dates where all signals are available are returned (inner join).
    """
    if signals is None:
        signals = SIGNAL_FILES
    sig_dir = Path(signals_dir) if signals_dir else _SIGNALS_DIR

    frames: dict[str, pd.DataFrame] = {}
    for name, fname in signals.items():
        path = sig_dir / fname
        frames[name] = pd.read_parquet(path)

    # Common market intersection across all signals
    if markets is None:
        common_mkts = sorted(
            set.intersection(*(set(df.columns) for df in frames.values()))
        )
    else:
        common_mkts = markets

    # Sum each signal across the common markets → scalar per date
    series: dict[str, pd.Series] = {
        name: df[common_mkts].sum(axis=1) for name, df in frames.items()
    }

    F = pd.DataFrame(series)
    F.index.name = "date"
    F = F.sort_index()

    # Inner join: only dates where every signal has data
    F = F.dropna(how="any")
    return F


def combined_return(
    F: pd.DataFrame,
    weights: "BlendWeights",
) -> pd.Series:
    """
    Apply static blend weights to the managed-return matrix.

    For rates, the combined portfolio return at time t is:
        R_t = Σ_k  w_k · F_k[t]

    Returns a pd.Series of daily portfolio returns.
    """
    w = weights.weights
    r = pd.Series(0.0, index=F.index)
    for k in w.index:
        if k in F.columns:
            r = r + w[k] * F[k]
    r.name = "combined"
    return r
