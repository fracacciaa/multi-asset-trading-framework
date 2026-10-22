"""
FX Forward Curve PCA Residual Mean-Reversion Signal.

Uses empirical PCA (instead of parametric NS) to extract the 3 dominant
modes of variation from the annualised forward discount term structure
(1M/3M/6M/12M/2Y).  Residuals from the PCA fit are the factor-neutral
component — any (pair, tenor) position built purely from residuals is
orthogonal to the main statistical factors.

Three levels of aggregation:

  pca_scalar   Collapse 5 per-tenor residual z-scores to one scalar per pair
               (mean), then XS-rank pairs.  Analogous to ns_rv agg.

  pca_null     Project per-pair residuals onto the 2D null-space basis of
               the PCA loading matrix. Same structure as ns_null but with
               empirical rather than NS factors.

  pca_all      Treat every (pair, tenor) pair as a SEPARATE instrument —
               100 instruments total.  XS-rank across all of them; positions
               within a pair are automatically factor-neutral.
               P&L: Δfd_ann (each tenor of a pair is a tradeable contract).

Note: "orthogonality to carry" is a by-product of removing the first PC
(level), not an explicit constraint.

PCA is fitted on fd_ann CHANGES (Δfd_ann) on training data ≤ 2018-12-31
to avoid look-ahead; the factor loadings are held fixed out-of-sample.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from fx.ns_fwd import (
    build_fwd_panel,
    compute_xs_positions,
    _TRAIN_END,
    FWD_TENORS,
)
from fx.production.data import build_panels, CARRY_PAIRS


# ── PCA factor extraction ──────────────────────────────────────────────────────

def build_pca_factors(
    fwd_panels: dict[str, pd.DataFrame],
    k: int = 3,
    train_end: str = _TRAIN_END,
) -> dict[str, np.ndarray]:
    """
    Per-pair PCA on fd_ann CHANGES on training data.

    Returns {pair: V (n_tenors, k)} where columns of V are the top-k
    eigenvectors of the within-pair covariance of Δfd_ann, sorted by
    descending eigenvalue.

    Using changes (not levels) avoids spurious low-frequency trend dominance
    and gives factors that match the actual variance decomposition.
    """
    out: dict[str, np.ndarray] = {}
    for pair, df in fwd_panels.items():
        train   = df[df.index <= pd.Timestamp(train_end)]
        changes = train.diff().dropna()
        if changes.shape[0] < 60 or changes.shape[1] < k + 1:
            continue
        cov              = np.cov(changes.values.T)          # (n, n)
        eigvals, eigvecs = np.linalg.eigh(cov)               # ascending order
        idx              = np.argsort(eigvals)[::-1]         # descending
        V                = eigvecs[:, idx[:k]].copy()        # (n, k)
        out[pair]        = V
    return out


def build_pca_residuals(
    fwd_panels:  dict[str, pd.DataFrame],
    pca_factors: dict[str, np.ndarray],
) -> dict[str, pd.DataFrame]:
    """
    Per-pair per-tenor residuals from projecting fd_ann onto the PCA factor space.

    Residual = (I − V Vᵀ) @ fd_ann_t   (in the row direction).

    Because the fd_ann LEVEL is projected out (not just changes), residuals
    represent the factor-neutral component of the curve level each day —
    the part expected to mean-revert.

    Returns {pair: DataFrame(date × tenors)} of residuals.
    """
    out: dict[str, pd.DataFrame] = {}
    for pair, df in fwd_panels.items():
        if pair not in pca_factors:
            continue
        V      = pca_factors[pair]       # (n, k)
        P_proj = V @ V.T                  # (n, n) projection onto factor space
        fd_arr = df.values               # (T, n)
        resid  = fd_arr - fd_arr @ P_proj  # (T, n) — factor-neutral residuals
        out[pair] = pd.DataFrame(resid, index=df.index, columns=df.columns)
    return out


# ── Signals ────────────────────────────────────────────────────────────────────

def _zscore_series(s: pd.Series, hl: int, mp: int) -> pd.Series:
    vol = s.shift(1).ewm(halflife=hl, min_periods=mp).std()
    return (s / vol.clip(lower=1e-8)).clip(-3.0, 3.0)


def compute_scalar_signal(
    pca_resids:  dict[str, pd.DataFrame],
    ewma_hl:     int = 63,
    min_periods: int = 30,
) -> pd.DataFrame:
    """
    Collapse per-tenor residual z-scores to one scalar per pair (mean).

    Negate: positive residual = curve above PCA fit = excess carry → short.
    Returns DataFrame(date × pair).
    """
    sigs: dict[str, pd.Series] = {}
    for pair, resid_df in pca_resids.items():
        vol = resid_df.shift(1).ewm(halflife=ewma_hl, min_periods=min_periods).std()
        z   = (resid_df / vol.clip(lower=1e-8)).clip(-3.0, 3.0)
        sigs[pair] = -z.mean(axis=1)
    return pd.DataFrame(sigs).sort_index() if sigs else pd.DataFrame()


def compute_null_signal(
    pca_resids:  dict[str, pd.DataFrame],
    pca_factors: dict[str, np.ndarray],
    ewma_hl:     int = 63,
    min_periods: int = 30,
) -> pd.DataFrame:
    """
    Project residuals onto the null-space of the PCA loading matrix, then
    z-score and average the 2 null-space dimensions.

    The null-space of V (n × k) is the last (n-k) columns of U from SVD(V).
    This is identical to the ns_null approach but with PCA factors.
    Returns DataFrame(date × pair).
    """
    sigs: dict[str, pd.Series] = {}
    for pair, resid_df in pca_resids.items():
        V = pca_factors.get(pair)
        if V is None:
            continue
        n, k = V.shape
        if n <= k:
            continue
        # Null-space basis: last (n-k) left singular vectors of V
        U, _, _ = np.linalg.svd(V, full_matrices=True)
        N = U[:, k:].copy()              # (n, n-k)
        for j in range(N.shape[1]):
            col = N[:, j]
            nz  = col[np.abs(col) > 1e-10]
            if len(nz) > 0 and nz[0] < 0:
                N[:, j] *= -1
        alpha = resid_df.values @ N      # (T, n-k)
        alpha_df = pd.DataFrame(alpha, index=resid_df.index,
                                columns=[f"ns{j+1}" for j in range(N.shape[1])])
        vol = alpha_df.shift(1).ewm(halflife=ewma_hl, min_periods=min_periods).std()
        z   = (alpha_df / vol.clip(lower=1e-8)).clip(-3.0, 3.0)
        sigs[pair] = -z.mean(axis=1)
    return pd.DataFrame(sigs).sort_index() if sigs else pd.DataFrame()


def compute_all_residual_signal(
    pca_resids:  dict[str, pd.DataFrame],
    ewma_hl:     int = 63,
    min_periods: int = 30,
) -> pd.DataFrame:
    """
    Treat every (pair, tenor) combination as a separate XS instrument.

    Returns DataFrame with MultiIndex columns (pair, tenor), each column a
    z-scored signal ready for cross-sectional ranking.
    Negate: positive residual = above PCA fit → short.
    """
    sigs: dict[tuple[str, str], pd.Series] = {}
    for pair, resid_df in pca_resids.items():
        for tenor in resid_df.columns:
            s = resid_df[tenor]
            z = _zscore_series(s, ewma_hl, min_periods)
            sigs[(pair, tenor)] = -z
    if not sigs:
        return pd.DataFrame()
    df = pd.DataFrame(sigs).sort_index()
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["pair", "tenor"])
    return df


# ── Positioning ────────────────────────────────────────────────────────────────

def xs_positions_flat(signal: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    XS long top-n / short bottom-n across ALL columns (pair × tenor combined).
    Returns same shape as signal, values in {−1/n, 0, +1/n}.
    """
    def _row(row: pd.Series) -> pd.Series:
        valid = row.dropna()
        w = pd.Series(0.0, index=row.index)
        if len(valid) < 2 * n:
            return w
        ranked = valid.rank()
        w.loc[ranked[ranked >= len(valid) - n + 1].index] =  1.0 / n
        w.loc[ranked[ranked <= n].index]                  = -1.0 / n
        return w
    return signal.apply(_row, axis=1)


# ── PnL from Δfd_ann ───────────────────────────────────────────────────────────

def pnl_fd_all(
    positions:   pd.DataFrame,     # MultiIndex columns (pair, tenor)
    fwd_panels:  dict[str, pd.DataFrame],
) -> pd.Series:
    """
    Daily P&L from the (pair, tenor) position matrix using Δfd_ann.

    pnl_t = sum_{pair, tenor} pos_{pair,tenor,t-1} × Δfd_ann_{pair,tenor,t}

    Returns daily P&L Series (zero days dropped).
    """
    fd_frames: dict[tuple[str, str], pd.Series] = {}
    for pair, df in fwd_panels.items():
        for tenor in df.columns:
            fd_frames[(pair, tenor)] = df[tenor]
    fd_all = pd.DataFrame(fd_frames).sort_index()
    fd_all.columns = pd.MultiIndex.from_tuples(fd_all.columns, names=["pair", "tenor"])
    d_fd = fd_all.diff()

    common   = positions.columns.intersection(d_fd.columns)
    pos_a    = positions[common].reindex(d_fd.index).fillna(0.0)
    pnl      = (pos_a.shift(1) * d_fd[common]).sum(axis=1)
    return pnl.loc[pnl != 0]


# ── pnl via rx_d (net pair exposure from pair × tenor positions) ───────────────

def pnl_rxd_from_pair_tenor(
    positions: pd.DataFrame,    # MultiIndex (pair, tenor)
    rx_d:      pd.DataFrame,    # date × pair
) -> pd.Series:
    """
    Map (pair, tenor) positions to pair-level net exposure, then earn rx_d.

    Each pair's net exposure = sum of that pair's tenor weights.
    This is the rx_d P&L of the pair position implied by the multi-tenor trade.
    """
    pairs   = positions.columns.get_level_values("pair").unique()
    net_pos = pd.DataFrame(
        {p: positions.xs(p, axis=1, level="pair").sum(axis=1) for p in pairs
         if p in positions.columns.get_level_values("pair")}
    ).sort_index()
    common = net_pos.columns.intersection(rx_d.columns)
    pos_a  = net_pos[common].reindex(rx_d.index).fillna(0.0)
    pnl    = (pos_a.shift(1) * rx_d[common]).sum(axis=1)
    return pnl.loc[pnl != 0]


# ── Full comparison ────────────────────────────────────────────────────────────

def compare_pca(
    pairs:       Optional[list[str]] = None,
    start:       Optional[str] = "2005-01-01",
    end:         Optional[str] = None,
    oos_start:   Optional[str] = "2019-01-01",
    k:           int = 3,
    n_pair:      int = 4,
    n_instr:     int = 20,
    path:        Optional[str] = None,
    ewma_hl:     int = 63,
    min_periods: int = 30,
) -> dict:
    """
    Compare three PCA residual approaches against carry.

    pca_scalar (rx_d)   — mean of per-tenor z-scores per pair → pair XS
    pca_null   (rx_d)   — null-space projection per pair → pair XS
    pca_all    (Δfd_ann)— all (pair, tenor) instruments, top-n_instr/bottom-n_instr
    pca_all    (rx_d)   — same positions, P&L via rx_d (net pair exposure)

    Parameters
    ----------
    k          Number of PCA factors to remove (default 3).
    n_pair     Pairs per XS leg for scalar and null signals.
    n_instr    (pair, tenor) instruments per XS leg for the all-residuals signal.

    Returns dict with results DataFrame and all intermediate objects.
    """
    if pairs is None:
        pairs = CARRY_PAIRS

    # ── Data ───────────────────────────────────────────────────────────────────
    print("Loading forward panel...")
    fwd_panels = build_fwd_panel(pairs=pairs, start=start, end=end, path=path)

    print(f"Fitting PCA (k={k}) on training data ≤ {_TRAIN_END}...")
    pca_fac = build_pca_factors(fwd_panels, k=k)
    print(f"  Pairs with PCA factors: {len(pca_fac)}")

    pca_res = build_pca_residuals(fwd_panels, pca_fac)

    load_kw = {"path": path} if path else {}
    panels  = build_panels(pairs=pairs, start=start, end=end, **load_kw)
    rx_d    = panels["rx_d"]
    fd_d    = panels["fd_1m"].resample("B").ffill()

    # ── Carry benchmark ────────────────────────────────────────────────────────
    carry_pos = compute_xs_positions(fd_d, n=n_pair)
    common_c  = carry_pos.columns.intersection(rx_d.columns)
    c_pnl     = (carry_pos[common_c].shift(1) * rx_d[common_c]).sum(axis=1)
    c_pnl     = c_pnl.loc[c_pnl != 0]
    c_oos     = c_pnl.loc[oos_start:] if oos_start else c_pnl
    c_sr      = float(c_oos.mean() / c_oos.std() * np.sqrt(252))

    # ── Build signals ──────────────────────────────────────────────────────────
    print("Computing pca_scalar signal...")
    sig_scalar = compute_scalar_signal(pca_res, ewma_hl=ewma_hl, min_periods=min_periods)

    print("Computing pca_null signal...")
    sig_null = compute_null_signal(pca_res, pca_fac, ewma_hl=ewma_hl, min_periods=min_periods)

    print("Computing pca_all signal (pair × tenor)...")
    sig_all  = compute_all_residual_signal(pca_res, ewma_hl=ewma_hl, min_periods=min_periods)

    # ── Positions ──────────────────────────────────────────────────────────────
    pos_scalar = compute_xs_positions(sig_scalar, n=n_pair)
    pos_null   = compute_xs_positions(sig_null,   n=n_pair)
    pos_all    = xs_positions_flat(sig_all, n=n_instr)

    # ── PnL ────────────────────────────────────────────────────────────────────
    def _pnl_rxd(pos: pd.DataFrame) -> pd.Series:
        common = pos.columns.intersection(rx_d.columns)
        pos_a  = pos[common].reindex(rx_d.index).fillna(0.0)
        pnl    = (pos_a.shift(1) * rx_d[common]).sum(axis=1)
        return pnl.loc[pnl != 0]

    pnl_scalar  = _pnl_rxd(pos_scalar)
    pnl_null    = _pnl_rxd(pos_null)
    pnl_all_fd  = pnl_fd_all(pos_all, fwd_panels)
    pnl_all_rxd = pnl_rxd_from_pair_tenor(pos_all, rx_d)

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

    rows = [
        _metrics(pnl_scalar,  f"pca_scalar  (rx_d)  [n_pair={n_pair}]"),
        _metrics(pnl_null,    f"pca_null    (rx_d)  [n_pair={n_pair}]"),
        _metrics(pnl_all_rxd, f"pca_all     (rx_d)  [n_instr={n_instr}]"),
        _metrics(pnl_all_fd,  f"pca_all     (Δfd)   [n_instr={n_instr}]"),
    ]
    df_out = pd.DataFrame(rows).set_index("variant")

    # ── Print ──────────────────────────────────────────────────────────────────
    oos_label = f"OOS {oos_start}→end" if oos_start else "full sample"
    n_inst_total = sig_all.shape[1] if not sig_all.empty else 0
    print(f"\n{oos_label}  |  PCA k={k}  |  {n_inst_total} (pair, tenor) instruments")
    print(f"{'Variant':<50} {'SR':>6} {'Skew':>6} {'MaxDD':>9} {'Corr/carry':>11}")
    print("─" * 86)
    print(f"{'CARRY (benchmark)':<50} {c_sr:>+6.3f}")
    print("─" * 86)
    for _, row in df_out.sort_values("SR", ascending=False).iterrows():
        sr_str = f"{row['SR']:>+6.3f}" if pd.notna(row["SR"]) else "   nan"
        sk_str = f"{row['skew']:>+6.3f}" if pd.notna(row["skew"]) else "   nan"
        dd_str = f"{row['maxDD']:>9.4f}" if pd.notna(row["maxDD"]) else "     nan"
        cr_str = f"{row['corr']:>+11.3f}" if pd.notna(row["corr"]) else "        nan"
        print(f"{row.name:<50} {sr_str} {sk_str} {dd_str} {cr_str}")

    return {
        "df":          df_out,
        "carry_sr":    c_sr,
        "c_pnl":       c_pnl,
        "pca_fac":     pca_fac,
        "pca_res":     pca_res,
        "sig_scalar":  sig_scalar,
        "sig_null":    sig_null,
        "sig_all":     sig_all,
        "pos_scalar":  pos_scalar,
        "pos_null":    pos_null,
        "pos_all":     pos_all,
        "pnl_scalar":  pnl_scalar,
        "pnl_null":    pnl_null,
        "pnl_all_fd":  pnl_all_fd,
        "pnl_all_rxd": pnl_all_rxd,
        "fwd_panels":  fwd_panels,
    }
