"""
Hysteresis positioning for the combined NS residual-reversal signal.

The NS null-space coordinate (alpha = -N^T e) is the same scalar quantity that
Signal B uses.  We compute it exactly (not approximated from the weight signs)
and apply EWMA z-scoring + hysteresis entry/exit logic on top.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from commodities.power_ns.ns_fit import ns_loadings

# ── Defaults (match Signal B) ─────────────────────────────────────────────────

EWMA_HALFLIFE  = 63    # ~3 months
MIN_PERIODS    = 30
Z_ENTRY        = 1.0   # |z| threshold for entry
VOL_FLOOR      = 1e-6


# ── Signed null-space coordinate ──────────────────────────────────────────────


def compute_ns_alpha_panel(
    ns_residual_by_rank: dict,
    ns_results: pd.DataFrame,
    MATURITY: pd.DataFrame,
    universe: list[str],
    depth: int,
    lambda_trade: float,
) -> pd.DataFrame:
    """
    Compute alpha = -N(t)^T e(t) per market per date.

    Sign convention: force the M3 (index-2) loading of N non-positive so that
    alpha > 0 means "butterfly is below equilibrium → expect upward reversion"
    consistently across all dates and markets.

    Returns DataFrame(index=TRADING_INDEX, columns=universe).
    """
    idx      = ns_residual_by_rank[1].index
    alpha_df = pd.DataFrame(np.nan, index=idx, columns=universe, dtype=float)

    for root in universe:
        try:
            x = ns_results.xs(root, level="Commodity").sort_index()
        except KeyError:
            continue

        dates = x.index[x["N_contracts"].ge(depth)].intersection(idx)
        if len(dates) == 0:
            continue

        e = pd.concat(
            [ns_residual_by_rank[k][root].rename(k) for k in range(1, depth + 1)],
            axis=1,
        ).reindex(dates)

        cols = [f"{root}{k}" for k in range(1, depth + 1)]
        M = MATURITY.reindex(dates)[cols].to_numpy(float)
        E = e.to_numpy(float)

        good = (
            np.isfinite(E).all(axis=1)
            & np.isfinite(M).all(axis=1)
            & (M > 0).all(axis=1)
        )
        if not good.any():
            continue

        M_g, E_g = M[good], E[good]
        slope, curv = ns_loadings(M_g, float(lambda_trade))
        X = np.stack([np.ones_like(M_g), slope, curv], axis=2)   # (n, depth, 3)

        U, _, _ = np.linalg.svd(X, full_matrices=True)
        # X is (n, depth, 3); U is (n, depth, depth); null space starts at col 3
        n_null = depth - X.shape[2]   # = depth - 3
        N = U[:, :, X.shape[2]:].copy()   # shape (n, depth, n_null)
        if N.shape[2] == 0:
            continue
        n_vec = N[:, :, 0]            # (n, depth)

        # Stable sign convention: M3 loading ≤ 0
        flip = n_vec[:, 2] > 0
        n_vec[flip] *= -1.0

        coord  = np.einsum("nd,nd->n", n_vec, E_g)   # N^T e
        alpha  = -coord                               # Signal-B convention

        out = pd.Series(np.nan, index=dates, dtype=float)
        out.iloc[np.flatnonzero(good)] = alpha
        alpha_df[root] = out

    return alpha_df


# ── Z-score (causal EWMA) ─────────────────────────────────────────────────────


def ewma_zscore(
    alpha: pd.DataFrame,
    halflife: int  = EWMA_HALFLIFE,
    min_periods: int = MIN_PERIODS,
    vol_floor: float = VOL_FLOOR,
    demean: bool = True,
) -> pd.DataFrame:
    """
    Causal EWMA z-score.

    demean=True (default):  z(t) = (alpha(t) - mu(t-1)) / sigma(t-1)
        Correct for hysteresis: z=0 means alpha is at its rolling mean
        (equilibrium), not at absolute zero. Required when the null-space
        direction N is time-varying (actual maturities) because N^T e is
        not guaranteed zero-mean unlike the fixed-N Signal B case.

    demean=False:  z(t) = alpha(t) / sigma(t-1)
        Appropriate only when alpha is zero-mean by construction (e.g.
        fixed-N projection as in ns_proj.py Signal B).
    """
    a_lag = alpha.shift(1)
    mu  = a_lag.ewm(halflife=halflife, min_periods=min_periods).mean()
    vol = a_lag.ewm(halflife=halflife, min_periods=min_periods).std()
    centered = alpha - mu if demean else alpha
    return centered / vol.clip(lower=vol_floor)


# ── Hysteresis ────────────────────────────────────────────────────────────────


def _hyst_array(z_vals: np.ndarray, entry: float) -> np.ndarray:
    """Scalar hysteresis loop over a 1-D z-score array."""
    pos = np.empty(len(z_vals), dtype=float)
    cur = 0.0
    for i in range(len(z_vals)):
        v = z_vals[i]
        if np.isnan(v):
            pos[i] = np.nan
            cur    = 0.0       # reset on gap
            continue
        if cur == 0.0:
            if   v >  entry: cur =  1.0
            elif v < -entry: cur = -1.0
        else:
            if cur > 0 and v <= 0.0: cur = 0.0
            elif cur < 0 and v >= 0.0: cur = 0.0
        pos[i] = cur
    return pos


def apply_hysteresis(
    z: pd.DataFrame,
    entry: float = Z_ENTRY,
) -> pd.DataFrame:
    """
    Apply hysteresis column-wise and return position DataFrame ∈ {-1, 0, +1, NaN}.
    """
    result = pd.DataFrame(index=z.index, columns=z.columns, dtype=float)
    for col in z.columns:
        result[col] = _hyst_array(z[col].to_numpy(dtype=float), entry)
    return result


# ── Hysteresis weight builder ─────────────────────────────────────────────────


def build_hysteresis_weights(
    residual_weights: dict,
    alpha_panel: pd.DataFrame,
    z_entry: float  = Z_ENTRY,
    ewma_halflife: int = EWMA_HALFLIFE,
    min_periods: int = MIN_PERIODS,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Gate the unit-gross butterfly weights with EWMA z-score + hysteresis.

    Weights derivation
    ------------------
    The continuous reversal weights satisfy:
        W[k] = sign(alpha) * canonical_long[k]

    where canonical_long[k] is the unit-gross +N direction.  Re-arranging:
        canonical_long[k] = sign(alpha) * W[k]

    The hysteresis butterfly weight is:
        hyst_W[k] = pos * canonical_long[k] = pos * sign(alpha) * W[k]

    This correctly delivers +N (long) when pos=+1 and -N (short) when pos=-1,
    regardless of whether the current continuous alpha temporarily changes sign
    during the hold.

    Returns
    -------
    hyst_weights  : dict {rank: DataFrame} — same structure as residual_weights
    hyst_positions: DataFrame ∈ {-1, 0, +1, NaN}
    z_scores      : DataFrame — EWMA z-scores before hysteresis
    """
    max_rank = max(residual_weights.keys())
    universe = alpha_panel.columns.tolist()
    idx      = alpha_panel.index

    z_scores = ewma_zscore(alpha_panel, halflife=ewma_halflife, min_periods=min_periods)
    hyst_pos = apply_hysteresis(z_scores, entry=z_entry)

    hyst_weights = {
        k: pd.DataFrame(np.nan, index=idx, columns=universe, dtype=float)
        for k in range(1, max_rank + 1)
    }

    for market in universe:
        pos  = hyst_pos[market].reindex(idx)
        alph = alpha_panel[market].reindex(idx)
        sign_a = np.sign(alph)

        gate = pos * sign_a      # ∈ {-1, 0, +1} — direction multiplier

        for k in range(1, max_rank + 1):
            wk = residual_weights[k][market].reindex(idx)
            hyst_weights[k][market] = gate * wk

    return hyst_weights, hyst_pos, z_scores
