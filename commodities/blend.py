"""
Commodities signal blend layer.

Loads pre-computed vol-scaled managed-return parquets, and exposes the same
blend API as equities/blend.py (all algorithms are re-exported).

Pipeline position:
    commodities/signals/*_pnl.parquet  →  load_managed_returns()  →  F (T × 4)
    F  →  fit_static / MLMSRR / LinearDynamicBlend / …  →  blend weights
    blend weights  →  combined portfolio return

Signals (production, from commodities/signals/):
    tsmom_252        tsmom_252_pnl.parquet        TS momentum 252-day (sign)
    xs_carry         xs_carry_pnl.parquet          XS tenor-normalised F1/F2 carry
    xs_carry_enh     xs_carry_enh_pnl.parquet      XS carry + seasonal/sector/roll
    curvature        curvature_pnl.parquet         NS butterfly (F1+F4-2*F2)

Managed-return convention: for each signal k and commodity i,
    w_k[i,t] = signal_k[i,t] × target_vol / sigma_hat[i,t]   (vol-targeted)
    F_k[t]   = Σ_i  w_k[i,t-1] × r[i,t]                     (weekly W-FRI rebalance)

The NS null-space butterfly (Signal B) is excluded: its daily vol is ~10× lower
than directional signals; the vol-scaling step would apply ~10× leverage, making
its managed return non-comparable and its blend weight trivially dominant.

Parquets are pre-computed by build_signal_parquets(); loading them takes ~1 ms.
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

# ── Commodities-specific constants ────────────────────────────────────────────

_SIGNALS_DIR = Path(__file__).parent / "signals"

SIGNAL_FILES: dict[str, str] = {
    "tsmom_252":    "tsmom_252_pnl.parquet",
    "xs_carry":     "xs_carry_pnl.parquet",
    "xs_carry_enh": "xs_carry_enh_pnl.parquet",
    "curvature":    "curvature_pnl.parquet",
}

# Production pipeline parameters (must match those used in build_signal_parquets).
# Subset of backtest.PARAMS relevant to the managed-return computation.
BLEND_PARAMS = dict(
    asset_vol = 0.40,
    vol_com   = 60,
    rebal     = "W-FRI",
    lag       = 1,
    mode      = "sign",
    clip      = 2.0,
)


# ── One-time parquet builder ──────────────────────────────────────────────────

def build_signal_parquets(
    output_dir: str | Path | None = None,
    params: dict | None = None,
) -> None:
    """
    Compute vol-scaled managed returns for all four production signals and save
    them to parquet.  Call once; subsequent load_managed_returns() calls read
    from disk in ~1 ms.

    Parameters
    ----------
    output_dir  Directory for parquet output.  Default: commodities/signals/.
    params      Pipeline parameters.  Default: BLEND_PARAMS.
    """
    from commodities.data import fetch_prices, fetch_term_structure, fetch_tickers
    from commodities.backtest import (
        specific_contract_returns, vol_scale, rebalance, _build_signals,
    )

    p = params or BLEND_PARAMS
    out_dir = Path(output_dir) if output_dir else _SIGNALS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Bloomberg data …")
    prices  = fetch_prices()
    ts      = fetch_term_structure()
    tk      = fetch_tickers()

    returns = specific_contract_returns(ts, tk).reindex(columns=prices.columns)
    raw_signals = _build_signals(prices, ts, tk, p)

    # Map from _build_signals keys → SIGNAL_FILES keys
    _key_map = {
        "tsmom_252":          "tsmom_252",
        "xs_carry_f1f2":      "xs_carry",
        "xs_carry_enh_f1f2":  "xs_carry_enh",
        "curvature":          "curvature",
    }

    print("Computing managed returns …")
    for raw_key, blend_key in _key_map.items():
        if raw_key not in raw_signals:
            print(f"  Skipping {raw_key} (not available)")
            continue
        raw_sig = raw_signals[raw_key]
        sig = raw_sig.reindex(index=returns.index, columns=returns.columns)
        w = vol_scale(sig, returns, target_vol=p["asset_vol"], com=p["vol_com"])
        w = rebalance(w, rule=p["rebal"])
        # Managed return: w[t-1] × r[t], zero rows dropped
        F_k = (w.shift(p["lag"]) * returns).sum(axis=1)
        F_k = F_k.replace(0.0, np.nan).dropna()
        F_k.index.name = "date"
        F_k.name = blend_key
        path = out_dir / SIGNAL_FILES[blend_key]
        F_k.to_frame().to_parquet(path)
        sr = F_k.mean() / F_k.std() * np.sqrt(252)
        print(f"  {blend_key:15s}  SR={sr:+.3f}  rows={len(F_k)}  → {path.name}")

    print("Done.")


# ── Data loading ─────────────────────────────────────────────────────────────

def load_managed_returns(
    signals: dict[str, str] | None = None,
    signals_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load commodities managed-return parquets and return F (T × K).

    Parameters
    ----------
    signals     Map of signal_name → parquet filename.  Default: SIGNAL_FILES.
    signals_dir Directory containing the parquet files.  Default: commodities/signals/.

    Returns
    -------
    pd.DataFrame, shape (T, K), index=date, columns=signal names.
    Only dates where all signals are available are returned (inner join).

    Raises
    ------
    FileNotFoundError if parquets are missing.  Run build_signal_parquets() first.
    """
    if signals is None:
        signals = SIGNAL_FILES
    sig_dir = Path(signals_dir) if signals_dir else _SIGNALS_DIR

    series: dict[str, pd.Series] = {}
    for name, fname in signals.items():
        path = sig_dir / fname
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found.  Run commodities.blend.build_signal_parquets() first."
            )
        df = pd.read_parquet(path)
        series[name] = df.iloc[:, 0]

    F = pd.DataFrame(series)
    F.index.name = "date"
    F = F.sort_index().dropna(how="any")
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
