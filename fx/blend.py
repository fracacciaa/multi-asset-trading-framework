"""
FX signal blend layer.

Loads production position parquets, computes managed returns, and exposes
the same blend API as equities/blend.py (all algorithms are re-exported).

Pipeline position:
    fx/signals/*_positions*.parquet  →  load_managed_returns()  →  F (T × 6)
    F  →  fit_static / MLMSRR / LinearDynamicBlend / …  →  blend weights
    blend weights  →  combined portfolio return

Signals (production, from fx/signals/):
    momentum        signal_a_f9_h1.parquet              XS monthly f=9m h=1 (ffilled)
    carry_E_only    signal_d_E_only_weights_daily.parquet  Dupuy MV E-only (daily)
    carry_bnp_base  signal_e_carry_daily.parquet        BNP base XS weights (daily)
    carry_bnp_e3    signal_e_carry_daily.parquet        BNP E3 crash-risk adjusted (daily)
    mr_ns_loo       mr_ns_loo_positions_daily.parquet   NS LOO XS positions (daily)
    mr_fly          mr_fly_positions.parquet             NS fly XS positions (daily)

Managed return convention: F_k[t] = (pos_k[t-1] × rx_d[t]).sum(pairs)
Monthly momentum is forward-filled to business days before applying this rule.
signal_e_carry_daily uses a two-level MultiIndex (base_weights / e3_weights).

Default pair universe: 19-pair intersection that excludes USDTRY (which is absent
from both MR position files). Data starts 2006-08-25 (first date where all six
signals, including the option-data-dependent carry_bnp_e3, are available).
"""

from __future__ import annotations

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

# ── FX-specific constants ─────────────────────────────────────────────────────

_SIGNALS_DIR = Path(__file__).parent / "signals"

# Map signal name → parquet filename.
# carry_bnp_base and carry_bnp_e3 share a file; _SIGNAL_LEVEL selects the level.
SIGNAL_FILES: dict[str, str] = {
    "momentum":       "signal_a_f9_h1.parquet",
    "carry_E_only":   "signal_d_E_only_weights_daily.parquet",
    "carry_bnp_base": "signal_e_carry_daily.parquet",
    "carry_bnp_e3":   "signal_e_carry_daily.parquet",
    "mr_ns_loo":      "mr_ns_loo_positions_daily.parquet",
    "mr_fly":         "mr_fly_positions.parquet",
}

# Top-level key to select from a MultiIndex parquet (two-level columns).
_SIGNAL_LEVEL: dict[str, str] = {
    "carry_bnp_base": "base_weights",
    "carry_bnp_e3":   "e3_weights",
}

# Monthly signals that need forward-filling to daily (using business-day ffill).
_MONTHLY_SIGNALS: frozenset[str] = frozenset({"momentum"})

# Returns parquet (daily excess returns, date × pair).
_RX_FILE = "rx_d.parquet"


# ── Data loading ─────────────────────────────────────────────────────────────

def load_managed_returns(
    signals_dir: str | Path | None = None,
    pairs: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load FX production signals and build the managed-return matrix F (T × K).

    For each signal k, the managed return at time t is:
        F_k[t] = Σ_{pair}  pos_k[t-1, pair] × rx_d[t, pair]

    Monthly signals (momentum) are forward-filled to business days before
    applying the convention.  MultiIndex signals (carry_bnp) are sliced by
    their sub-key before processing.

    Parameters
    ----------
    signals_dir
        Directory containing the parquet files.  Default: fx/signals/.
    pairs
        Currency pairs to include.  Default: the 19-pair intersection of all
        signal files (excludes USDTRY which is absent from MR position files).

    Returns
    -------
    pd.DataFrame, shape (T, 6), index=date, columns=signal names.
    Only dates where all six signals are non-NaN are returned (inner join).
    Data starts 2006-08-25 (first date carry_bnp_e3 is available).
    """
    sig_dir = Path(signals_dir) if signals_dir else _SIGNALS_DIR

    rx_d = pd.read_parquet(sig_dir / _RX_FILE)

    # Cache MultiIndex parquets that are referenced by more than one signal.
    _file_cache: dict[str, pd.DataFrame] = {}

    def _load_raw(fname: str) -> pd.DataFrame:
        if fname not in _file_cache:
            _file_cache[fname] = pd.read_parquet(sig_dir / fname)
        return _file_cache[fname]

    # Determine pair universe.
    if pairs is None:
        # Build default 19-pair intersection: rx_d ∩ all signal files.
        # MR files have 19 pairs (no USDTRY); that is the binding constraint.
        mr_ns = _load_raw(SIGNAL_FILES["mr_ns_loo"])
        pairs = sorted(set(rx_d.columns) & set(mr_ns.columns))

    rx = rx_d[pairs]

    def _managed_ret(name: str) -> pd.Series:
        raw = _load_raw(SIGNAL_FILES[name])
        # Slice MultiIndex level if needed.
        if name in _SIGNAL_LEVEL:
            raw = raw[_SIGNAL_LEVEL[name]]
        # Restrict to target pairs.
        available = [p for p in pairs if p in raw.columns]
        pos = raw[available].reindex(rx.index)
        # Monthly → forward-fill to daily business days.
        if name in _MONTHLY_SIGNALS:
            pos = pos.ffill()
        # Managed return: pos[t-1] × rx[t]; min_count=1 avoids silently treating
        # NaN positions as zero (important for carry_bnp before option data starts).
        return (pos.shift(1) * rx[available]).sum(axis=1, min_count=1)

    series = {name: _managed_ret(name) for name in SIGNAL_FILES}
    F = pd.DataFrame(series)
    F.index.name = "date"
    F = F.sort_index()
    F = F.dropna(how="any")
    return F


# ── Portfolio-level combination ───────────────────────────────────────────────

def combined_return(
    F: pd.DataFrame,
    weights: "BlendWeights",
) -> pd.Series:
    """
    Apply static blend weights to the managed-return matrix.

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
