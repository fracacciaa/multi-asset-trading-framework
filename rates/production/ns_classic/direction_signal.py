"""
NS classic MR — direction + asymmetric hysteresis (production signal).

Position per tenor:
  flat      when |z| ≤ exit_z  (or before first entry)
  ±fixed_k  when |z| > entry_z  → enter at sign(z), freeze
  hold      until |z| drops back to exit_z, then close

fixed_k per tenor = IS median of |continuous_dv01_target|.

PnL approximation:
    pnl_dir_t ≈ pnl_gross_t × frac_active_t
where frac_active = Σ_j I_active_j × fixed_k_j / Σ_j fixed_k_j.

TC is charged only on entry and exit events (no within-trade rebalancing).
Production params: entry_z=2.5, exit_z=0.25.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ENTRY_Z: float = 2.5
EXIT_Z:  float = 0.25


def _dir_hyst_positions(
    z_df: pd.DataFrame,
    fixed_k: pd.Series,
    entry: float,
    exit_z: float,
) -> pd.DataFrame:
    """
    Per-tenor direction positions with asymmetric entry/exit hysteresis.
    Returns DataFrame same shape as z_df with values in {-fixed_k_j, 0, +fixed_k_j}.
    """
    arr = z_df.to_numpy(dtype=float)
    out = np.zeros_like(arr)
    fk  = fixed_k.to_numpy(dtype=float)
    for j in range(arr.shape[1]):
        s = 0.0
        for i in range(arr.shape[0]):
            x = arr[i, j]
            if not np.isfinite(x):
                s = 0.0; out[i, j] = np.nan; continue
            if s != 0.0 and abs(x) <= exit_z:
                s = 0.0
            if s == 0.0 and abs(x) > entry:
                s = fk[j] * float(np.sign(x))
            out[i, j] = s
    return pd.DataFrame(out, index=z_df.index, columns=z_df.columns)


def build_direction_signal(
    mr_pnl: pd.DataFrame,
    mr_zscores: pd.DataFrame,
    mr_dv01_pos: pd.DataFrame,
    is_end: str,
    entry_z: float = ENTRY_Z,
    exit_z: float = EXIT_Z,
    markets: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply direction + hysteresis to committed NS MR z-scores and positions.

    Parameters
    ----------
    mr_pnl      date × market gross PnL (from mean_reversion_pnl.parquet)
    mr_zscores  date × (market, tenor) z-scores (from mean_reversion_zscores.parquet)
    mr_dv01_pos date × (market, tenor) continuous DV01 positions (from mean_reversion_dv01_pos.parquet)
    is_end      last in-sample date (used to fix per-tenor notionals)
    entry_z     entry threshold (|z| > entry_z to open position)
    exit_z      exit threshold (|z| ≤ exit_z to close position)
    markets     restrict to these markets (default: all in mr_pnl)

    Returns
    -------
    dir_pnl     date × market direction-strategy PnL (pnl_gross × frac_active)
    dir_pos     date × (market, tenor) direction DV01 positions (±fixed_k or 0)
    """
    mkts = markets or mr_pnl.columns.tolist()
    mkt_levels = mr_zscores.columns.get_level_values("market")

    pnl_dict: dict[str, pd.Series]    = {}
    pos_dict: dict[str, pd.DataFrame] = {}

    for mkt in mkts:
        if mkt not in mkt_levels:
            continue

        pnl_m  = mr_pnl[mkt].dropna()
        zs_m   = mr_zscores.xs(mkt, level="market", axis=1)
        pos_m  = mr_dv01_pos.xs(mkt, level="market", axis=1)

        active  = pos_m.columns[pos_m.abs().sum() > 0]
        if len(active) == 0:
            continue

        zs_act  = zs_m[active].reindex(pnl_m.index)
        pos_act = pos_m[active].reindex(pnl_m.index).fillna(0.0)

        fixed_k = pos_act.loc[pos_act.index <= is_end].abs().median()
        fixed_k = fixed_k.replace(0.0, np.nan).dropna()
        if fixed_k.empty:
            continue
        active  = fixed_k.index
        zs_act  = zs_act[active]
        total_k = fixed_k.sum()

        dir_pos = _dir_hyst_positions(zs_act, fixed_k, entry_z, exit_z) \
                      .reindex(pnl_m.index).fillna(0.0)

        frac_active = dir_pos.abs().sum(axis=1) / total_k
        pnl_dict[mkt] = pnl_m * frac_active.reindex(pnl_m.index).fillna(0.0)
        pos_dict[mkt] = dir_pos

    if not pnl_dict:
        return pd.DataFrame(), pd.DataFrame()

    dir_pnl = pd.DataFrame(pnl_dict)
    dir_pnl.index.name = "date"

    dir_pos_all = pd.concat(pos_dict, axis=1)
    dir_pos_all.columns.names = ["market", "tenor"]

    return dir_pnl, dir_pos_all
