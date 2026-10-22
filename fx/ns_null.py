"""
FX Forward Curve NS Null-Space Mean Reversion Signal.

For each currency pair, the annualised forward discount term structure
(1M/3M/6M/12M/2Y) lives in R^5.  A 3-factor Nelson-Siegel model spans
a 3-dimensional subspace; the orthogonal complement has dimension 2.  Any
move in this 2-dimensional null space is factor-neutral by construction —
it cannot be explained by a shift in level, slope, or curvature.

Signal:
  1. Calibrate λ per pair on training data (≤ 2018-12-31).
  2. Compute N = (5, 2) null-space basis via SVD of the NS design matrix X(τ, λ).
     Basis is constant per pair (fixed λ → fixed X → fixed N).
  3. Project the daily fd_ann curve onto N: alpha = fd_arr @ N  →  (T, 2).
  4. Z-score each null-space dimension with lagged EWMA vol (halflife=63d).
  5. Combined scalar = −mean(z1, z2) per pair per date (negate: positive
     projection = excess carry that will revert → short).
  6. Cross-sectional rank: long top-n / short bottom-n.

Two P&L measures:
  rx_d     — spot excess return (comparable with other FX signals)
  Δfd_ann  — pure in-curve convergence: pos_{t-1} × (−Δmean_alpha_t)

Compared against:
  top-4 fly_1m3m6m variants from compare_rv
  XS carry benchmark

Usage:
    from fx.ns_null import compare_null
    compare_null(start="2005-01-01")
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from fx.ns_fwd import (
    build_fwd_panel,
    compute_xs_positions,
    _calibrate_lam,
    LAM_GRID,
    TAU_YRS,
    FWD_TENORS,
    _TRAIN_END,
)
from fx.ns_rv import build_rv_signals, compute_halflife_table
from fx.production.data import build_panels, CARRY_PAIRS
from rates.production.ns_core import ns_loadings


# ── Null-space basis ───────────────────────────────────────────────────────────

def _null_basis(tau: np.ndarray, lam: float) -> np.ndarray:
    """
    2D null-space basis for the NS design matrix X(tau, lam).

    X is (n, 3), n=5 for FX → null(X^T) has dim n−3.
    Returns N: (n, n−3) — last n−3 left singular vectors of X.
    Sign: first nonzero element of each column is positive.
    """
    X = ns_loadings(tau, lam)                             # (n, 3)
    U, _, _ = np.linalg.svd(X, full_matrices=True)        # U: (n, n)
    k = X.shape[1]                                         # 3
    N = U[:, k:].copy()                                    # (n, n-k)
    for j in range(N.shape[1]):
        col = N[:, j]
        nz  = col[np.abs(col) > 1e-10]
        if len(nz) > 0 and nz[0] < 0:
            N[:, j] *= -1
    return N


# ── Build projections ──────────────────────────────────────────────────────────

def build_null_projections(
    fwd_panels: dict[str, pd.DataFrame],
    lam_fixed:  dict[str, float],
) -> dict:
    """
    Project each pair's forward discount curves onto the null-space basis.

    Parameters
    ----------
    fwd_panels  output of build_fwd_panel()
    lam_fixed   output of _calibrate_lam()

    Returns
    -------
    dict:
      'alpha' : {pair: DataFrame(date × ['ns1', 'ns2', ...])}  — raw projections
      'N'     : {pair: ndarray(n_tenors, n_null)}               — basis vectors
      'lam'   : {pair: float}                                   — calibrated λ
    """
    alpha_dict: dict[str, pd.DataFrame] = {}
    N_dict:     dict[str, np.ndarray]   = {}
    lam_dict:   dict[str, float]        = {}

    for pair, df in fwd_panels.items():
        tenors = [t for t in FWD_TENORS if t in df.columns]
        if len(tenors) < 4:         # need n > k=3 for non-trivial null space
            continue
        tau    = np.array([TAU_YRS[FWD_TENORS.index(t)] for t in tenors])
        lam    = lam_fixed.get(pair, float(np.median(LAM_GRID)))
        N      = _null_basis(tau, lam)          # (n_tenors, n_null)
        n_null = N.shape[1]

        fd_arr   = df[tenors].values            # (T, n_tenors) — already no-NaN rows
        alpha    = fd_arr @ N                   # (T, n_null)

        alpha_dict[pair] = pd.DataFrame(
            alpha,
            index=df.index,
            columns=[f"ns{j+1}" for j in range(n_null)],
        )
        N_dict[pair]   = N
        lam_dict[pair] = lam

    return {"alpha": alpha_dict, "N": N_dict, "lam": lam_dict}


# ── Signal ─────────────────────────────────────────────────────────────────────

def compute_null_signal(
    alpha_dict:  dict[str, pd.DataFrame],
    ewma_hl:     int = 63,
    min_periods: int = 30,
) -> pd.DataFrame:
    """
    Z-score and aggregate null-space projections to a scalar per-pair signal.

    For each pair:
      vol_k  = lagged EWMA std(alpha_k)     (halflife=ewma_hl)
      z_k    = clip(alpha_k / vol_k, ±3)
      signal = −mean(z1, z2)               (negate: positive = excess carry → short)

    Returns DataFrame(date × pair).
    """
    sigs: dict[str, pd.Series] = {}
    for pair, alpha_df in alpha_dict.items():
        vol = alpha_df.shift(1).ewm(halflife=ewma_hl, min_periods=min_periods).std()
        z   = (alpha_df / vol.clip(lower=1e-8)).clip(-3.0, 3.0)
        sigs[pair] = -z.mean(axis=1)

    if not sigs:
        return pd.DataFrame()
    return pd.DataFrame(sigs).sort_index()


# ── In-curve P&L ───────────────────────────────────────────────────────────────

def compute_pnl_fd(
    alpha_dict: dict[str, pd.DataFrame],
    positions:  pd.DataFrame,
) -> pd.Series:
    """
    Pure in-curve convergence P&L using changes in the mean null-space projection.

    pnl_t = sum_pair [ pos_{pair,t-1} × Δmean_alpha_{pair,t} ]

    pos < 0 (short) earns when Δalpha < 0 (projection reverts toward zero), and
    pos > 0 (long) earns when Δalpha > 0 — consistent with a standard long/short
    position.  P is measured in annualised forward discount units.

    Returns daily PnL Series (zero days excluded).
    """
    mean_alpha: dict[str, pd.Series] = {}
    for pair, alpha_df in alpha_dict.items():
        mean_alpha[pair] = alpha_df.mean(axis=1)

    ma_df  = pd.DataFrame(mean_alpha).sort_index()
    d_ma   = ma_df.diff()                               # Δmean_alpha

    common = positions.columns.intersection(d_ma.columns)
    pos_a  = positions[common].reindex(d_ma.index).fillna(0.0)
    pnl    = (pos_a.shift(1) * d_ma[common]).sum(axis=1)
    return pnl.loc[pnl != 0]


# ── Combined comparison ────────────────────────────────────────────────────────

def compare_null(
    pairs:       Optional[list[str]] = None,
    start:       Optional[str] = "2005-01-01",
    end:         Optional[str] = None,
    oos_start:   Optional[str] = "2019-01-01",
    n:           int = 4,
    path:        Optional[str] = None,
    ewma_hl:     int = 63,
    min_periods: int = 30,
) -> dict:
    """
    Compare top-4 fly_1m3m6m variants, null-space signal, and carry benchmark.

    Fly variants (all OOS):
      ns3_grid  · raw    · fly_1m3m6m
      ns3_grid  · smooth · fly_1m3m6m
      ns3_fixed · raw    · fly_1m3m6m
      ns3_fixed · smooth · fly_1m3m6m

    Null-space variants (all OOS):
      NS null (rx_d)    — standard excess-return P&L
      NS null (Δfd_ann) — pure in-curve convergence P&L

    Signal computed from `start` for EWMA warmup; SR measured on [oos_start, end].
    λ calibration uses data ≤ _TRAIN_END (2018-12-31).

    Returns dict with results DataFrame and all intermediate series.
    """
    if pairs is None:
        pairs = CARRY_PAIRS

    # ── Shared data ────────────────────────────────────────────────────────────
    print("Loading forward panel and calibrating λ...")
    fwd_panels = build_fwd_panel(pairs=pairs, start=start, end=end, path=path)
    lam_fixed  = _calibrate_lam(fwd_panels, LAM_GRID, k=3)

    print("Building null-space projections...")
    null_out   = build_null_projections(fwd_panels, lam_fixed)
    alpha_dict = null_out["alpha"]

    print("Computing AR(1) halflives for smoothed fly variants...")
    hl_table = compute_halflife_table(fwd_panels, lam_fixed)

    load_kw = {"path": path} if path else {}
    panels  = build_panels(pairs=pairs, start=start, end=end, **load_kw)
    rx_d    = panels["rx_d"]
    fd_d    = panels["fd_1m"].resample("B").ffill()

    # ── Carry benchmark ────────────────────────────────────────────────────────
    carry_pos = compute_xs_positions(fd_d, n=n)
    common_c  = carry_pos.columns.intersection(rx_d.columns)
    c_pnl     = (carry_pos[common_c].shift(1) * rx_d[common_c]).sum(axis=1)
    c_pnl     = c_pnl.loc[c_pnl != 0]
    c_oos     = c_pnl.loc[oos_start:] if oos_start else c_pnl
    c_sr      = float(c_oos.mean() / c_oos.std() * np.sqrt(252))

    # ── Null-space signal and PnL ──────────────────────────────────────────────
    null_sig = compute_null_signal(alpha_dict, ewma_hl=ewma_hl, min_periods=min_periods)
    null_pos = compute_xs_positions(null_sig, n=n)

    common_n   = null_pos.columns.intersection(rx_d.columns)
    null_pos_a = null_pos[common_n].reindex(rx_d.index).fillna(0.0)
    null_pnl_rxd = (null_pos_a.shift(1) * rx_d[common_n]).sum(axis=1)
    null_pnl_rxd = null_pnl_rxd.loc[null_pnl_rxd != 0]

    null_pnl_fd = compute_pnl_fd(alpha_dict, null_pos)

    # ── Fly_1m3m6m variants ────────────────────────────────────────────────────
    def _fly_pnl(model: str, smooth: bool) -> pd.Series:
        sigs = build_rv_signals(
            model=model, pairs=pairs, start=start, end=end,
            smooth=smooth,
            halflife_table=hl_table if smooth else None,
            lam_fixed=lam_fixed if model == "ns3_fixed" else None,
            path=path,
        )
        fly_df = sigs.get("fly_1m3m6m", pd.DataFrame())
        if fly_df.empty:
            return pd.Series(dtype=float)
        z_dict: dict[str, pd.Series] = {}
        for col in fly_df.columns:
            s = fly_df[col].dropna()
            if len(s) < min_periods:
                continue
            vol = s.shift(1).ewm(halflife=ewma_hl, min_periods=min_periods).std()
            z_dict[col] = -(s / vol.clip(lower=1e-8)).clip(-3.0, 3.0)
        if not z_dict:
            return pd.Series(dtype=float)
        z_df   = pd.DataFrame(z_dict).sort_index()
        pos    = compute_xs_positions(z_df, n=n)
        common = pos.columns.intersection(rx_d.columns)
        pos_a  = pos[common].reindex(rx_d.index).fillna(0.0)
        pnl    = (pos_a.shift(1) * rx_d[common]).sum(axis=1)
        return pnl.loc[pnl != 0]

    print("Computing fly_1m3m6m variants...")
    fly_pnls: dict[str, pd.Series] = {
        "ns3_grid·smooth·fly_1m3m6m":  _fly_pnl("ns3_grid",  True),
        "ns3_fixed·smooth·fly_1m3m6m": _fly_pnl("ns3_fixed", True),
        "ns3_grid·raw·fly_1m3m6m":     _fly_pnl("ns3_grid",  False),
        "ns3_fixed·raw·fly_1m3m6m":    _fly_pnl("ns3_fixed", False),
    }

    # ── Metrics ────────────────────────────────────────────────────────────────
    def _metrics(pnl: pd.Series, label: str) -> dict:
        oos = pnl.loc[oos_start:] if oos_start else pnl
        if len(oos) < 63:
            return dict(variant=label, SR=np.nan, skew=np.nan, maxDD=np.nan, corr=np.nan)
        sr  = float(oos.mean() / oos.std() * np.sqrt(252))
        m, v = oos.mean(), oos.std()
        skw  = float(((oos - m) ** 3).mean() / v ** 3) if v > 0 else 0.0
        cum  = oos.cumsum()
        mdd  = float((cum - cum.cummax()).min())
        idx  = oos.index.intersection(c_oos.index)
        corr = float(np.corrcoef(oos.reindex(idx), c_oos.reindex(idx))[0, 1]) \
               if len(idx) > 10 else np.nan
        return dict(variant=label, SR=sr, skew=skw, maxDD=mdd, corr=corr)

    rows = []
    for label, pnl in fly_pnls.items():
        rows.append(_metrics(pnl, label))
    rows.append(_metrics(null_pnl_rxd, "NS null (rx_d)"))
    rows.append(_metrics(null_pnl_fd,  "NS null (Δfd_ann)"))

    df_out = pd.DataFrame(rows).set_index("variant")

    # ── Print table ────────────────────────────────────────────────────────────
    oos_label = f"OOS {oos_start}→end" if oos_start else "full sample"
    print(f"\n{oos_label}")
    print(f"{'Variant':<44} {'SR':>6} {'Skew':>6} {'MaxDD':>9} {'Corr/carry':>11}")
    print("─" * 80)
    print(f"{'CARRY (benchmark)':<44} {c_sr:>+6.3f}")
    print("─" * 80)
    for _, row in df_out.sort_values("SR", ascending=False).iterrows():
        sr_str = f"{row['SR']:>+6.3f}" if pd.notna(row["SR"]) else "   nan"
        sk_str = f"{row['skew']:>+6.3f}" if pd.notna(row["skew"]) else "   nan"
        dd_str = f"{row['maxDD']:>9.4f}" if pd.notna(row["maxDD"]) else "     nan"
        cr_str = f"{row['corr']:>+11.3f}" if pd.notna(row["corr"]) else "        nan"
        print(f"{row.name:<44} {sr_str} {sk_str} {dd_str} {cr_str}")

    return {
        "df":            df_out,
        "carry_sr":      c_sr,
        "c_pnl":         c_pnl,
        "null_sig":      null_sig,
        "null_pos":      null_pos,
        "null_pnl_rxd":  null_pnl_rxd,
        "null_pnl_fd":   null_pnl_fd,
        "fly_pnls":      fly_pnls,
        "alpha_dict":    alpha_dict,
        "N_dict":        null_out["N"],
        "lam_dict":      null_out["lam"],
    }
