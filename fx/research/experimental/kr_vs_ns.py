"""
KRR vs NS comparison for FX forward discount curves.

Direct adaptation of rates/research/experimental/kr_vs_nss.py for FX.

Models compared
---------------
  ns3_fixed : NS3 with λ calibrated per-pair on train (≤2018), held fixed OOS
  ns3_grid  : NS3 with λ daily grid-search (min-SSE per date)
  ns3_loo   : NS3 with LOO-corrected residuals (daily grid λ)
  kr        : KRR with cubic spline (H²-Sobolev) kernel, μ tuned per-pair on train

Signal pipeline (shared across models)
---------------------------------------
  1. Compute per-tenor residuals: actual_fd − fitted_fd
  2. EWMA z-score each tenor (hl=63d, lagged 1d, clip ±3)
  3. Aggregate: mean across tenors, negated (above-NS fit → short)
  4. Cross-sectional positioning: long top-n / short bottom-n (n=4)
  5. Backtest against rx_d

μ calibration: per-pair, on training window (≤2018), maximising per-pair
signal P&L Sharpe (position × −Δresid_agg), analogous to the factor-neutral
Sharpe criterion in rates/research/experimental/kr_vs_nss.py.

OOS: 2019-01-01 → today.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import Optional

from fx.ns_fwd import (
    build_fwd_panel,
    _calibrate_lam,
    _hat_matrices_k,
    _fit_residuals,
    _fit_residuals_loo,
    compute_xs_positions,
    LAM_GRID,
    TAU_YRS,
    FWD_TENORS,
    _TRAIN_END,
)
from fx.production.data import build_panels, CARRY_PAIRS
from rates.production.ns_core import ns_loadings

warnings.filterwarnings("ignore")

OOS_START   = "2019-01-01"
MU_GRID     = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
EWMA_HL     = 63
MIN_PERIODS = 30
N_XS        = 4


# ── KRR kernel and fitting ─────────────────────────────────────────────────────

def _kernel_matrix(s: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Cubic spline (H²-Sobolev) Gram matrix k(s_i, t_j)."""
    s = s[:, None]
    t = t[None, :]
    mn = np.minimum(s, t)
    mx = np.maximum(s, t)
    return 1.0 + mn * mx + 0.5 * mn**2 * (mx - mn / 3.0)


def _fit_kr(fd_arr: np.ndarray, tau: np.ndarray, mu: float) -> np.ndarray:
    """
    KRR residuals for a forward discount panel.

    fd_arr : (T, N) forward discounts in TAU_YRS order
    tau    : (N,) tenor in years
    mu     : ridge penalty (larger → smoother fit)

    Returns (T, N) residuals = actual_fd − KRR_fitted_fd.
    """
    n = len(tau)
    K = _kernel_matrix(tau, tau)                     # (N, N) Gram matrix
    # Hat matrix W = K (K + μI)^{-1}; both K and K+μI symmetric so W is symmetric
    W = np.linalg.solve(K + mu * np.eye(n), K).T    # (N, N)
    return fd_arr - fd_arr @ W.T


# ── Signal helpers ─────────────────────────────────────────────────────────────

def _agg_signal(resid_df: pd.DataFrame) -> pd.Series:
    """Mean of EWMA-z-scored per-tenor residuals, negated (above-NS fit = short)."""
    vol = resid_df.shift(1).ewm(halflife=EWMA_HL, min_periods=MIN_PERIODS).std()
    z   = (resid_df / vol.clip(lower=1e-8)).clip(-3.0, 3.0)
    return -z.mean(axis=1)


# ── Per-pair μ calibration ─────────────────────────────────────────────────────

def _tune_mu(
    df: pd.DataFrame,
    tenors: list[str],
    tau: np.ndarray,
    train_end: str = _TRAIN_END,
) -> float:
    """
    Tune μ per pair: maximise in-sample P&L Sharpe on training window (≤train_end).

    P&L proxy: position(t−1) × −Δresid_agg(t), analogous to the factor-neutral
    PnL Sharpe criterion in rates/research/experimental/kr_vs_nss.py.
    """
    train = df[tenors].loc[df.index <= pd.Timestamp(train_end)].dropna()
    if len(train) < 60:
        return 0.1

    best_mu, best_sr = MU_GRID[0], -np.inf
    for mu in MU_GRID:
        resid_arr = _fit_kr(train.values, tau, mu)
        resid_df  = pd.DataFrame(resid_arr, index=train.index, columns=tenors)
        sig       = _agg_signal(resid_df)
        ret_proxy = (-resid_df.sum(axis=1)).diff()        # −Δresid_agg
        pnl       = (sig.shift(1) * ret_proxy).dropna()
        if len(pnl) < 30 or pnl.std() < 1e-12:
            continue
        sr = float(pnl.mean() / pnl.std() * np.sqrt(252))
        if sr > best_sr:
            best_mu, best_sr = mu, sr

    return best_mu


# ── Build per-model signals ────────────────────────────────────────────────────

def _build_model_signals(
    fwd_panels: dict,
    lam_fixed: dict,
    mu_map: dict,
) -> dict[str, dict[str, pd.Series]]:
    """
    Build {model → {pair → signal_series}} for all four models.

    Each signal is the aggregate (mean per-tenor z-score, negated) from
    the model's residuals, computed on the full history for EWMA warmup.
    """
    H3 = _hat_matrices_k(TAU_YRS, LAM_GRID, k=3)

    sigs: dict[str, dict] = {m: {} for m in ["ns3_fixed", "ns3_grid", "ns3_loo", "kr"]}

    for pair, df in fwd_panels.items():
        tenors = [t for t in FWD_TENORS if t in df.columns]
        if len(tenors) < 3:
            continue

        tau    = np.array([TAU_YRS[FWD_TENORS.index(t)] for t in tenors])
        fd_arr = df[tenors].values
        H      = H3 if tenors == list(FWD_TENORS) else _hat_matrices_k(tau, LAM_GRID, k=3)

        # ns3_fixed — single hat matrix from per-pair calibrated λ
        lam = lam_fixed.get(pair, float(np.median(LAM_GRID)))
        X   = ns_loadings(tau, lam)
        try:
            G = np.linalg.solve(X.T @ X, X.T)
        except np.linalg.LinAlgError:
            G = np.linalg.lstsq(X, np.eye(len(tau)), rcond=None)[0].T
        H1_fixed    = X @ G
        resid_fixed = pd.DataFrame(fd_arr - (H1_fixed @ fd_arr.T).T,
                                   index=df.index, columns=tenors)
        sigs["ns3_fixed"][pair] = _agg_signal(resid_fixed)

        # ns3_grid — daily λ grid search (min-SSE per date)
        resid_grid = pd.DataFrame(_fit_residuals(fd_arr, H), index=df.index, columns=tenors)
        sigs["ns3_grid"][pair] = _agg_signal(resid_grid)

        # ns3_loo — LOO-corrected residuals (daily λ)
        resid_loo = pd.DataFrame(_fit_residuals_loo(fd_arr, H), index=df.index, columns=tenors)
        sigs["ns3_loo"][pair] = _agg_signal(resid_loo)

        # kr — KRR with per-pair tuned μ
        mu        = mu_map.get(pair, 0.1)
        resid_kr  = pd.DataFrame(_fit_kr(fd_arr, tau, mu), index=df.index, columns=tenors)
        sigs["kr"][pair] = _agg_signal(resid_kr)

    return sigs


# ── PnL and metrics ────────────────────────────────────────────────────────────

def _to_pnl(
    sigs_dict: dict[str, pd.Series],
    rx_d: pd.DataFrame,
    n: int = N_XS,
) -> pd.Series:
    sig_df = pd.DataFrame(sigs_dict).sort_index()
    if sig_df.empty:
        return pd.Series(dtype=float)
    pos    = compute_xs_positions(sig_df, n=n)
    common = pos.columns.intersection(rx_d.columns)
    pos_a  = pos[common].reindex(rx_d.index).fillna(0.0)
    pnl    = (pos_a.shift(1) * rx_d[common]).sum(axis=1)
    return pnl.loc[pnl != 0]


def _metrics(pnl: pd.Series, oos_start: str = OOS_START) -> dict:
    oos = pnl.loc[oos_start:].dropna() if oos_start else pnl.dropna()
    if len(oos) < 63:
        return dict(SR=np.nan, skew=np.nan, maxDD=np.nan)
    sr   = float(oos.mean() / oos.std() * np.sqrt(252))
    m, v = oos.mean(), oos.std()
    skw  = float(((oos - m) ** 3).mean() / v ** 3) if v > 0 else 0.0
    cum  = oos.cumsum()
    mdd  = float((cum - cum.cummax()).min())
    return dict(SR=sr, skew=skw, maxDD=mdd)


# ── Main comparison entry point ────────────────────────────────────────────────

def compare(
    pairs: Optional[list[str]] = None,
    start: str = "2005-01-01",
    end: Optional[str] = None,
    oos_start: str = OOS_START,
    path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run all four models and print a ranked OOS comparison table.

    Returns DataFrame(model → SR / skew / maxDD) sorted by SR.
    """
    if pairs is None:
        pairs = CARRY_PAIRS

    print("Loading forward panel...", flush=True)
    fwd_panels = build_fwd_panel(pairs=pairs, start=start, end=end, path=path)
    if not fwd_panels:
        raise RuntimeError("No pairs loaded — check data path and pair list.")

    print("Calibrating per-pair fixed λ on training data (≤2018)...", flush=True)
    lam_fixed = _calibrate_lam(fwd_panels, LAM_GRID, k=3)

    print("Tuning KRR μ per pair on training data (≤2018)...", flush=True)
    mu_map: dict = {}
    for pair, df in fwd_panels.items():
        tenors = [t for t in FWD_TENORS if t in df.columns]
        if len(tenors) < 3:
            mu_map[pair] = 0.1
            continue
        tau = np.array([TAU_YRS[FWD_TENORS.index(t)] for t in tenors])
        mu_map[pair] = _tune_mu(df, tenors, tau)
        print(f"  {pair}: μ={mu_map[pair]}", flush=True)

    print("Building model signals...", flush=True)
    model_sigs = _build_model_signals(fwd_panels, lam_fixed, mu_map)

    print("Loading returns panel...", flush=True)
    load_kw  = {"path": path} if path else {}
    panels   = build_panels(pairs=pairs, start=start, end=end, **load_kw)
    rx_d     = panels["rx_d"]
    fd_d     = panels["fd_1m"].resample("B").ffill()

    # Carry benchmark
    carry_pos = compute_xs_positions(fd_d, n=N_XS)
    common_c  = carry_pos.columns.intersection(rx_d.columns)
    c_pnl     = (carry_pos[common_c].shift(1) * rx_d[common_c]).sum(axis=1)
    c_pnl     = c_pnl.loc[c_pnl != 0]
    c_m       = _metrics(c_pnl, oos_start)

    print(f"\nRunning backtest for each model...\n", flush=True)
    rows = []
    for model, sigs_dict in model_sigs.items():
        pnl = _to_pnl(sigs_dict, rx_d)
        m   = _metrics(pnl, oos_start)
        rows.append({"model": model, **m})
        print(f"  {model:<14}  SR={m['SR']:+.3f}  skew={m['skew']:+.3f}"
              f"  mdd={m['maxDD']:.4f}")

    df_out = pd.DataFrame(rows).set_index("model").sort_values("SR", ascending=False)

    print(f"\nOOS {oos_start}→end — FX KRR vs NS, XS top-{N_XS}/bottom-{N_XS}\n")
    print(f"{'Model':<16} {'SR':>6} {'Skew':>6} {'MaxDD':>9}")
    print("─" * 42)
    sr_str = f"{c_m['SR']:>+6.3f}" if not np.isnan(c_m["SR"]) else "   nan"
    print(f"{'carry (bench)':<16} {sr_str} {c_m['skew']:>+6.3f} {c_m['maxDD']:>9.4f}")
    print("─" * 42)
    for _, row in df_out.iterrows():
        sr_str = f"{row['SR']:>+6.3f}" if not np.isnan(row["SR"]) else "   nan"
        print(f"{row.name:<16} {sr_str} {row['skew']:>+6.3f} {row['maxDD']:>9.4f}")

    return df_out


if __name__ == "__main__":
    compare()
