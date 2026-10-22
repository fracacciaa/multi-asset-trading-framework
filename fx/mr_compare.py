"""
FX mean-reversion master comparison harness.

Runs every MR signal family we have and reports OOS Sharpe in one table:

  NS_RV     — 20 variants: NS3 fixed/grid × raw/smooth × 5 structures
  NS_NULL   — 2D null-space (NS factors) via rx_d and Δfd_ann
  PCA_SCA   — PCA per-pair, aggregate z-score, via rx_d
  PCA_NULL  — PCA per-pair, null-space projection, via rx_d
  PCA_ALL   — PCA all (pair×tenor), XS-ranked, via Δfd_ann and rx_d

Data is loaded once and shared across all families.

Usage:
    from fx.mr_compare import compare_all_mr
    compare_all_mr(start="2005-01-01")
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
    _TRAIN_END,
)
from fx.ns_rv import (
    build_rv_signals,
    compute_halflife_table,
    STRUCTURES,
    MODELS,
)
from fx.ns_null import (
    build_null_projections,
    compute_null_signal   as _null_sig_ns,
    compute_pnl_fd        as _pnl_fd_ns,
)
from fx.pca_rv import (
    build_pca_factors,
    build_pca_residuals,
    compute_scalar_signal as _pca_scalar_sig,
    compute_null_signal   as _pca_null_sig,
    compute_all_residual_signal,
    xs_positions_flat,
    pnl_fd_all,
    pnl_rxd_from_pair_tenor,
)
from fx.production.data import build_panels, CARRY_PAIRS


def compare_all_mr(
    pairs:       Optional[list[str]] = None,
    start:       Optional[str] = "2005-01-01",
    end:         Optional[str] = None,
    oos_start:   Optional[str] = "2019-01-01",
    n:           int = 4,
    n_instr:     int = 20,
    path:        Optional[str] = None,
    ewma_hl:     int = 63,
    min_periods: int = 30,
) -> pd.DataFrame:
    """
    Run every FX mean-reversion signal and report OOS Sharpe.

    Parameters
    ----------
    n        Pairs per XS leg for pair-level signals (NS_RV, NS_NULL, PCA_SCA, PCA_NULL).
    n_instr  (pair, tenor) instruments per XS leg for PCA_ALL.

    Returns
    -------
    DataFrame(variant → SR / skew / maxDD / corr_carry / pnl_type), sorted by OOS SR.
    """
    if pairs is None:
        pairs = CARRY_PAIRS

    # ── Load all data once ─────────────────────────────────────────────────────
    print("Loading data...")
    fwd_panels = build_fwd_panel(pairs=pairs, start=start, end=end, path=path)
    load_kw    = {"path": path} if path else {}
    panels     = build_panels(pairs=pairs, start=start, end=end, **load_kw)
    rx_d       = panels["rx_d"]
    fd_d       = panels["fd_1m"].resample("B").ffill()

    # ── Calibrate parameters once ──────────────────────────────────────────────
    print("Calibrating NS λ and PCA factors on training data...")
    lam_fixed = _calibrate_lam(fwd_panels, LAM_GRID, k=3)
    hl_table  = compute_halflife_table(fwd_panels, lam_fixed)
    pca_fac   = build_pca_factors(fwd_panels, k=3)
    pca_res   = build_pca_residuals(fwd_panels, pca_fac)

    # ── Carry benchmark ────────────────────────────────────────────────────────
    carry_pos = compute_xs_positions(fd_d, n=n)
    common_c  = carry_pos.columns.intersection(rx_d.columns)
    c_pnl     = (carry_pos[common_c].shift(1) * rx_d[common_c]).sum(axis=1)
    c_pnl     = c_pnl.loc[c_pnl != 0]
    c_oos     = c_pnl.loc[oos_start:] if oos_start else c_pnl
    c_sr      = float(c_oos.mean() / c_oos.std() * np.sqrt(252))

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _metrics(pnl: pd.Series, label: str, pnl_type: str = "rx_d") -> dict:
        oos = pnl.loc[oos_start:] if oos_start else pnl
        if len(oos) < 63:
            return dict(variant=label, pnl_type=pnl_type,
                        SR=np.nan, skew=np.nan, maxDD=np.nan, corr=np.nan)
        sr  = float(oos.mean() / oos.std() * np.sqrt(252))
        m, v = oos.mean(), oos.std()
        skw  = float(((oos - m) ** 3).mean() / v ** 3) if v > 0 else 0.0
        cum  = oos.cumsum()
        mdd  = float((cum - cum.cummax()).min())
        idx  = oos.index.intersection(c_oos.index)
        corr = float(np.corrcoef(oos.reindex(idx), c_oos.reindex(idx))[0, 1]) \
               if len(idx) > 10 else np.nan
        return dict(variant=label, pnl_type=pnl_type,
                    SR=sr, skew=skw, maxDD=mdd, corr=corr)

    def _pnl_rxd(pos: pd.DataFrame) -> pd.Series:
        common = pos.columns.intersection(rx_d.columns)
        pos_a  = pos[common].reindex(rx_d.index).fillna(0.0)
        pnl    = (pos_a.shift(1) * rx_d[common]).sum(axis=1)
        return pnl.loc[pnl != 0]

    rows: list[dict] = []

    # ── NS_RV: 20 variants ─────────────────────────────────────────────────────
    print("NS_RV: running 20 variants (NS3 fixed/grid × raw/smooth × 5 structures)...")
    for model in MODELS:
        for smooth in [False, True]:
            smooth_tag = "smooth" if smooth else "raw"
            sigs = build_rv_signals(
                model=model, pairs=pairs, start=start, end=end,
                smooth=smooth,
                halflife_table=hl_table if smooth else None,
                lam_fixed=lam_fixed if model == "ns3_fixed" else None,
                fwd_panels=fwd_panels,
            )
            for sname in STRUCTURES:
                fly_df = sigs.get(sname, pd.DataFrame())
                if fly_df.empty:
                    continue
                z_dict: dict[str, pd.Series] = {}
                for col in fly_df.columns:
                    s = fly_df[col].dropna()
                    if len(s) < min_periods:
                        continue
                    vol = s.shift(1).ewm(halflife=ewma_hl, min_periods=min_periods).std()
                    z_dict[col] = -(s / vol.clip(lower=1e-8)).clip(-3.0, 3.0)
                if not z_dict:
                    continue
                z_df  = pd.DataFrame(z_dict).sort_index()
                pos   = compute_xs_positions(z_df, n=n)
                pnl   = _pnl_rxd(pos)
                label = f"ns_rv·{model}·{smooth_tag}·{sname}"
                rows.append(_metrics(pnl, label, "rx_d"))

    # ── NS_NULL: null-space via NS factors ─────────────────────────────────────
    print("NS_NULL: 2D null-space (NS factors)...")
    null_out     = build_null_projections(fwd_panels, lam_fixed)
    null_sig     = _null_sig_ns(null_out["alpha"], ewma_hl=ewma_hl, min_periods=min_periods)
    null_pos     = compute_xs_positions(null_sig, n=n)
    null_pnl_rxd = _pnl_rxd(null_pos)
    null_pnl_fd  = _pnl_fd_ns(null_out["alpha"], null_pos)
    rows.append(_metrics(null_pnl_rxd, "ns_null (rx_d)",   "rx_d"))
    rows.append(_metrics(null_pnl_fd,  "ns_null (Δfd_ann)", "Δfd_ann"))

    # ── PCA_SCA: per-pair aggregate z-score ────────────────────────────────────
    print("PCA_SCA: per-pair aggregate z-score...")
    sig_sca = _pca_scalar_sig(pca_res, ewma_hl=ewma_hl, min_periods=min_periods)
    pos_sca = compute_xs_positions(sig_sca, n=n)
    rows.append(_metrics(_pnl_rxd(pos_sca), "pca_scalar (rx_d)", "rx_d"))

    # ── PCA_NULL: per-pair null-space (PCA factors) ────────────────────────────
    print("PCA_NULL: per-pair null-space (PCA factors)...")
    sig_pnull = _pca_null_sig(pca_res, pca_fac, ewma_hl=ewma_hl, min_periods=min_periods)
    pos_pnull = compute_xs_positions(sig_pnull, n=n)
    rows.append(_metrics(_pnl_rxd(pos_pnull), "pca_null (rx_d)", "rx_d"))

    # ── PCA_ALL: all (pair × tenor) instruments ────────────────────────────────
    print("PCA_ALL: all (pair × tenor) instruments...")
    sig_all  = compute_all_residual_signal(pca_res, ewma_hl=ewma_hl, min_periods=min_periods)
    pos_all  = xs_positions_flat(sig_all, n=n_instr)
    pnl_afd  = pnl_fd_all(pos_all, fwd_panels)
    pnl_arxd = pnl_rxd_from_pair_tenor(pos_all, rx_d)
    rows.append(_metrics(pnl_afd,  f"pca_all (Δfd_ann) n={n_instr}", "Δfd_ann"))
    rows.append(_metrics(pnl_arxd, f"pca_all (rx_d)    n={n_instr}", "rx_d"))

    # ── Print master table ─────────────────────────────────────────────────────
    df_out = pd.DataFrame(rows).set_index("variant")

    oos_label = f"OOS {oos_start}→end" if oos_start else "full sample"
    print(f"\n{'─'*92}")
    print(f"  FX mean-reversion  |  {oos_label}  |  carry benchmark SR={c_sr:+.3f}")
    print(f"{'─'*92}")
    print(f"{'Variant':<52} {'Type':>8} {'SR':>7} {'Skew':>6} {'MaxDD':>9} {'Corr/C':>7}")
    print(f"{'─'*92}")

    for _, row in df_out.sort_values("SR", ascending=False).iterrows():
        sr_s  = f"{row['SR']:>+7.3f}" if pd.notna(row["SR"]) else "    nan"
        sk_s  = f"{row['skew']:>+6.3f}" if pd.notna(row["skew"]) else "   nan"
        dd_s  = f"{row['maxDD']:>9.4f}" if pd.notna(row["maxDD"]) else "     nan"
        cr_s  = f"{row['corr']:>+7.3f}" if pd.notna(row["corr"]) else "    nan"
        print(f"{row.name:<52} {row['pnl_type']:>8} {sr_s} {sk_s} {dd_s} {cr_s}")

    print(f"{'─'*92}")
    print(f"  * Δfd_ann P&L is in annualised forward-discount units (not spot returns).")
    print(f"    SR is scale-independent; Δfd_ann trades are delta-neutral tenor spreads.")
    print(f"{'─'*92}\n")

    return df_out
