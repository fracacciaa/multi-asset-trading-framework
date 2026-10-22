"""
FX Forward Curve Nelson-Siegel Mean Reversion Signal.

Fits a 3-factor Nelson-Siegel model to the annualised log forward discount
term structure (1M / 3M / 6M / 12M / 2Y) for each currency pair on each date.
Residuals from the fit represent deviations from a smooth term structure
and are expected to mean-revert.

Signal formation:
  1. Build per-pair forward discount curves (annualised log fd at 5 tenors).
  2. Fit NS with daily λ grid-search (minimise IS SSE per date).
  3. Z-score each tenor's residual with lagged EWMA vol (hl=63d).
  4. Clip ±3, mean across tenors → scalar per pair per date.
  5. Cross-sectional rank → long top-n / short bottom-n.

Usage:
    from fx.ns_fwd import compare
    compare(start="2005-01-01")
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from fx.data import fetch_spot, fetch_forwards
from fx.production.data import (
    CARRY_PAIRS,
    PRIMARY_START,
    calibrate_divisor,
    _to_fcu_per_usd,
    _fwd_outright_fcu,
    build_panels,
)
from rates.production.ns_core import ns_loadings
from view_formation import ViewFormationTS

# ── Constants ──────────────────────────────────────────────────────────────────

FWD_TENORS = ["1M", "3M", "6M", "12M", "2Y"]
TAU_YRS    = np.array([1 / 12, 3 / 12, 6 / 12, 12 / 12, 24 / 12])

# λ grid in 1/year; curvature loading peaks at τ* = log(2)/λ
# Range 0.2–3.0 covers τ* from ~3.5Y (λ=0.2) down to ~3M (λ=2.77)
LAM_GRID = np.linspace(0.2, 3.0, 40)


# ── Data loading ───────────────────────────────────────────────────────────────

def build_fwd_panel(
    pairs: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    path: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """
    For each pair, build a (date × tenor) DataFrame of annualised log forward discounts.

    Returns
    -------
    dict: pair → DataFrame(date × ["1M","3M","6M","12M","2Y"])
        fd_ann_τ = (log F_τ − log S) / τ_years, both in FCU/USD
    """
    if pairs is None:
        pairs = CARRY_PAIRS
    loader_kw: dict = {} if path is None else {"path": path}

    panels: dict[str, pd.DataFrame] = {}
    for pair in pairs:
        try:
            spot_raw = fetch_spot(pair, start=start, end=end, **loader_kw)
            spot_raw = spot_raw[spot_raw > 0].dropna()
            if len(spot_raw) < 60:
                continue

            fwd_raw = fetch_forwards(pair, start=start, end=end, **loader_kw)
            if fwd_raw.empty or "1M" not in fwd_raw.columns:
                continue

            dv = calibrate_divisor(spot_raw, fwd_raw["1M"])
            if np.isnan(dv):
                warnings.warn(f"{pair}: divisor calibration failed, skipping")
                continue

            spot_fcu = _to_fcu_per_usd(pair, spot_raw)
            log_s = np.log(spot_fcu)

            fd_cols: dict[str, pd.Series] = {}
            for tenor, tau_y in zip(FWD_TENORS, TAU_YRS):
                if tenor not in fwd_raw.columns:
                    continue
                F_fcu = _fwd_outright_fcu(pair, spot_raw, fwd_raw[tenor], dv)
                log_f = np.log(F_fcu.where(F_fcu > 0))
                fd_ann = (log_f - log_s) / tau_y
                fd_cols[tenor] = fd_ann

            if len(fd_cols) < 3:
                continue

            idx = spot_fcu.index
            df = pd.DataFrame(fd_cols)
            df = df[df.index.isin(idx)].dropna()
            panels[pair] = df

        except Exception as exc:
            warnings.warn(f"{pair}: {exc}")

    return panels


# ── NS fitting (vectorised over λ and dates) ───────────────────────────────────

def _hat_matrices(tau: np.ndarray, lam_grid: np.ndarray) -> np.ndarray:
    """
    Precompute projection matrices H_λ = X(X'X)^{-1}X' for every λ.

    Returns: (L, n, n)  L = len(lam_grid), n = len(tau)
    """
    n = len(tau)
    L = len(lam_grid)
    H = np.zeros((L, n, n))
    for i, lam in enumerate(lam_grid):
        X = ns_loadings(tau, lam)               # (n, 3)
        try:
            G = np.linalg.solve(X.T @ X, X.T)  # (3, n)
        except np.linalg.LinAlgError:
            G = np.linalg.lstsq(X, np.eye(n), rcond=None)[0].T
        H[i] = X @ G                            # (n, n)
    return H


def _fit_residuals(fd_arr: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    In-sample NS residuals for a single pair, choosing λ per date via min-SSE.

    Parameters
    ----------
    fd_arr  (T, n) annualised forward discounts, no NaN rows
    H       (L, n, n) precomputed hat matrices

    Returns
    -------
    (T, n) residuals from the best-λ fit for each date
    """
    # fitted values for all λ and all dates: (T, L, n)
    fitted    = np.einsum("lij,tj->tli", H, fd_arr)
    resid_all = fd_arr[:, None, :] - fitted          # (T, L, n)
    sse       = (resid_all ** 2).sum(axis=2)         # (T, L)
    best      = np.argmin(sse, axis=1)               # (T,)
    return resid_all[np.arange(len(fd_arr)), best, :]


# ── Signal computation ─────────────────────────────────────────────────────────

def compute_signal(
    pairs: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    lam_grid: np.ndarray = LAM_GRID,
    path: Optional[str] = None,
    ewma_hl: int = 63,
    min_periods: int = 30,
) -> pd.DataFrame:
    """
    NS forward-curve mean reversion signal.

    Returns
    -------
    DataFrame(date × pair): scalar z-score (mean of per-tenor z-scores, clipped ±3)
    """
    fwd_panels = build_fwd_panel(pairs=pairs, start=start, end=end, path=path)
    default_H  = _hat_matrices(TAU_YRS, lam_grid)

    signals: dict[str, pd.Series] = {}
    for pair, df in fwd_panels.items():
        tenors = [t for t in FWD_TENORS if t in df.columns]
        if len(tenors) < 3:
            continue

        if tenors == FWD_TENORS:
            H = default_H
        else:
            tau = np.array([TAU_YRS[FWD_TENORS.index(t)] for t in tenors])
            H   = _hat_matrices(tau, lam_grid)

        fd_arr   = df[tenors].values
        resid_df = pd.DataFrame(
            _fit_residuals(fd_arr, H), index=df.index, columns=tenors
        )

        vol = resid_df.shift(1).ewm(halflife=ewma_hl, min_periods=min_periods).std()
        z   = (resid_df / vol.clip(lower=1e-8)).clip(-3.0, 3.0)
        # Positive residual = FD above smooth curve = excess carry that will revert → short
        signals[pair] = -z.mean(axis=1)

    if not signals:
        return pd.DataFrame()

    return pd.DataFrame(signals).sort_index()


# ── Positioning ────────────────────────────────────────────────────────────────

def compute_xs_positions(signal: pd.DataFrame, n: int = 4) -> pd.DataFrame:
    """
    Cross-sectional rank: long top-n / short bottom-n at equal weight 1/n.
    """
    def _rank_row(row: pd.Series) -> pd.Series:
        valid = row.dropna()
        w = pd.Series(0.0, index=row.index)
        if len(valid) < 2 * n:
            return w
        ranked = valid.rank()
        w.loc[ranked[ranked >= len(valid) - n + 1].index] =  1.0 / n
        w.loc[ranked[ranked <= n].index]                  = -1.0 / n
        return w

    return signal.apply(_rank_row, axis=1)


# ── PnL calculation ────────────────────────────────────────────────────────────

def compute_pnl(
    pairs: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    n: int = 4,
    path: Optional[str] = None,
) -> dict:
    """
    Compute positions and daily PnL for the NS forward-curve signal.

    Returns
    -------
    dict: signal, positions, pnl (Series), sr (float)
    """
    if pairs is None:
        pairs = CARRY_PAIRS

    sig = compute_signal(pairs=pairs, start=start, end=end, path=path)
    pos = compute_xs_positions(sig, n=n)

    panels      = build_panels(pairs=pairs, start=start, end=end,
                                **({"path": path} if path else {}))
    rx_d        = panels["rx_d"]
    common      = pos.columns.intersection(rx_d.columns)
    pos_aligned = pos[common].reindex(rx_d.index).fillna(0.0)

    pnl = (pos_aligned.shift(1) * rx_d[common]).sum(axis=1)
    pnl = pnl.loc[pnl != 0]

    sr = float(pnl.mean() / pnl.std() * np.sqrt(252)) if len(pnl) > 63 else np.nan
    return {"signal": sig, "positions": pos, "pnl": pnl, "sr": sr}


# ── Comparison harness ─────────────────────────────────────────────────────────

def compare(
    pairs: Optional[list[str]] = None,
    start: Optional[str] = PRIMARY_START,
    end: Optional[str] = None,
    n: int = 4,
    path: Optional[str] = None,
) -> dict:
    """
    Backtest NS forward-curve signal and compare with simple XS carry.

    Prints Sharpe, skew, max drawdown, and signal correlation.
    Returns dict with both result sets for further analysis.
    """
    if pairs is None:
        pairs = CARRY_PAIRS

    print("Computing NS forward-curve signal...")
    ns_res = compute_pnl(pairs=pairs, start=start, end=end, n=n, path=path)
    ns_pnl = ns_res["pnl"]

    print("Computing XS carry benchmark...")
    panels    = build_panels(pairs=pairs, start=start, end=end,
                              **({"path": path} if path else {}))
    rx_d      = panels["rx_d"]
    fd_d      = panels["fd_1m"].resample("B").ffill()
    carry_pos = compute_xs_positions(fd_d, n=n)
    common    = carry_pos.columns.intersection(rx_d.columns)
    c_pnl     = (carry_pos[common].shift(1) * rx_d[common]).sum(axis=1)
    c_pnl     = c_pnl.loc[c_pnl != 0]
    c_sr      = float(c_pnl.mean() / c_pnl.std() * np.sqrt(252))

    idx  = ns_pnl.index.intersection(c_pnl.index)
    corr = float(np.corrcoef(ns_pnl.reindex(idx), c_pnl.reindex(idx))[0, 1])

    def _maxdd(s: pd.Series) -> float:
        cum = s.cumsum()
        return float((cum - cum.cummax()).min())

    def _skew(s: pd.Series) -> float:
        m, v = s.mean(), s.std()
        return float(((s - m) ** 3).mean() / v ** 3) if v > 0 else 0.0

    print()
    print(f"{'Metric':<22} {'NS fwd-curve':>14} {'XS carry':>12}")
    print("-" * 50)
    print(f"{'Sharpe (ann)':<22} {ns_res['sr']:>14.3f} {c_sr:>12.3f}")
    print(f"{'Skew':<22} {_skew(ns_pnl):>14.3f} {_skew(c_pnl):>12.3f}")
    print(f"{'Max drawdown':<22} {_maxdd(ns_pnl):>14.5f} {_maxdd(c_pnl):>12.5f}")
    print(f"{'N obs (days)':<22} {len(ns_pnl):>14d} {len(c_pnl):>12d}")
    print(f"{'Correlation':<22} {corr:>14.3f}")
    print()

    return {
        "ns":       ns_res,
        "carry_sr": c_sr,
        "c_pnl":    c_pnl,
        "corr":     corr,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Full variant comparison — mirrors rates/research/mean_reversion/ sweep
# ══════════════════════════════════════════════════════════════════════════════

_TRAIN_END  = "2018-12-31"
_HYST_ENTRY = 1.0


def _hat_matrices_k(
    tau: np.ndarray, lam_grid: np.ndarray, k: int = 3
) -> np.ndarray:
    """(L, n, n) projection matrices for k-factor NS (k=2: level+slope, k=3: +curvature)."""
    n, L = len(tau), len(lam_grid)
    H = np.zeros((L, n, n))
    for i, lam in enumerate(lam_grid):
        X = ns_loadings(tau, lam)[:, :k]
        try:
            G = np.linalg.solve(X.T @ X, X.T)
        except np.linalg.LinAlgError:
            G = np.linalg.lstsq(X, np.eye(n), rcond=None)[0].T
        H[i] = X @ G
    return H


def _fit_residuals_loo(fd_arr: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    LOO-corrected residuals: e_loo_i = e_i / (1 - h_ii).
    Uses the best-λ hat matrix per date.  h_ii clipped at 1e-6 to avoid blowup.
    """
    T, n   = fd_arr.shape
    fitted    = np.einsum("lij,tj->tli", H, fd_arr)
    resid_all = fd_arr[:, None, :] - fitted
    best      = np.argmin((resid_all ** 2).sum(axis=2), axis=1)   # (T,)
    is_resid  = resid_all[np.arange(T), best, :]                  # (T, n)
    h_diag    = np.diagonal(H[best], axis1=1, axis2=2).copy()     # (T, n)
    return is_resid / np.clip(1.0 - h_diag, 1e-6, None)


def _calibrate_lam(
    fwd_panels: dict,
    lam_grid: np.ndarray,
    k: int = 3,
    train_end: str = _TRAIN_END,
) -> dict:
    """
    Per-pair fixed λ = median of daily best-λ over the training window.
    Mirrors rates research: calibrate ≤ TRAIN_END, hold fixed OOS.
    """
    out: dict = {}
    lam_default = float(np.median(lam_grid))
    for pair, df in fwd_panels.items():
        train = df[df.index <= pd.Timestamp(train_end)]
        if len(train) < 60:
            out[pair] = lam_default
            continue
        tenors = [t for t in FWD_TENORS if t in train.columns]
        tau    = np.array([TAU_YRS[FWD_TENORS.index(t)] for t in tenors])
        H      = _hat_matrices_k(tau, lam_grid, k=k)
        fd_arr = train[tenors].values
        fitted = np.einsum("lij,tj->tli", H, fd_arr)
        sse    = ((fd_arr[:, None, :] - fitted) ** 2).sum(axis=2)
        out[pair] = float(np.median(lam_grid[np.argmin(sse, axis=1)]))
    return out


def _hyst(z: pd.Series, entry: float = _HYST_ENTRY) -> pd.Series:
    """Per-series hysteresis: enter at |z| > entry, exit when z crosses zero."""
    arr, cur = np.zeros(len(z)), 0.0
    for i, v in enumerate(z.values):
        if not np.isnan(v):
            if cur == 0.0:
                if   v >  entry: cur =  1.0
                elif v < -entry: cur = -1.0
            elif cur ==  1.0 and v <= 0.0: cur = 0.0
            elif cur == -1.0 and v >= 0.0: cur = 0.0
        arr[i] = cur
    return pd.Series(arr, index=z.index)


def _aggregate(
    resid_df: pd.DataFrame,
    structure: str,
    hl: int = 63,
    mp: int = 30,
) -> pd.Series:
    """
    Compute scalar per-pair signal from per-tenor NS residuals.

    All structures negate so that excess residual (FD above NS fit) → short.
    Structures mirror rates/research slope_fly_compare:
      agg          mean of all per-tenor z-scores  (factor-neutral aggregate)
      fly_1m3m6m   belly 3M vs wings 1M + 6M
      fly_3m6m12m  belly 6M vs wings 3M + 12M
      fly_6m12m2y  belly 12M vs wings 6M + 2Y
      slope_1m2y   long-end (2Y) minus short-end (1M)
    """
    cols = set(resid_df.columns)

    def _z(s: pd.Series) -> pd.Series:
        v = s.shift(1).ewm(halflife=hl, min_periods=mp).std()
        return (s / v.clip(lower=1e-8)).clip(-3.0, 3.0)

    if structure == "agg":
        vol  = resid_df.shift(1).ewm(halflife=hl, min_periods=mp).std()
        z_df = (resid_df / vol.clip(lower=1e-8)).clip(-3.0, 3.0)
        return -z_df.mean(axis=1)

    if structure == "fly_1m3m6m" and {"1M", "3M", "6M"} <= cols:
        return -_z(resid_df["3M"] - 0.5 * resid_df["1M"] - 0.5 * resid_df["6M"])

    if structure == "fly_3m6m12m" and {"3M", "6M", "12M"} <= cols:
        return -_z(resid_df["6M"] - 0.5 * resid_df["3M"] - 0.5 * resid_df["12M"])

    if structure == "fly_6m12m2y" and {"6M", "12M", "2Y"} <= cols:
        return -_z(resid_df["12M"] - 0.5 * resid_df["6M"] - 0.5 * resid_df["2Y"])

    if structure == "slope_1m2y" and {"1M", "2Y"} <= cols:
        return -_z(resid_df["2Y"] - resid_df["1M"])

    return pd.Series(np.nan, index=resid_df.index)


def build_ns_loo_vfts_signal(
    fwd_panels: dict,
    rx_d: pd.DataFrame,
    ewma_hl: int = 63,
    min_periods: int = 30,
    desired_vol: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the ns3·loo·grid·vfts·agg production signal.

    LOO-corrected NS3 residuals (daily λ grid) → per-pair aggregate z-score →
    monthly ViewFormationTS rebalance at 10% vol target.

    Parameters
    ----------
    fwd_panels  : output of build_fwd_panel()
    rx_d        : date × pair daily excess returns (for EWMA cov)

    Returns
    -------
    signal_df   : date × pair, raw aggregate LOO z-score (un-clipped, un-negated)
    positions_df: date × pair, monthly VF TS weights (forward-filled between rebalances)
    """
    H3 = _hat_matrices_k(TAU_YRS, LAM_GRID, k=3)

    raw_sigs: dict[str, pd.Series] = {}
    for pair, df in fwd_panels.items():
        tenors = [t for t in FWD_TENORS if t in df.columns]
        if len(tenors) < 3:
            continue
        tau    = np.array([TAU_YRS[FWD_TENORS.index(t)] for t in tenors])
        fd_arr = df[tenors].values
        H      = H3 if tenors == FWD_TENORS else _hat_matrices_k(tau, LAM_GRID, k=3)
        resid  = pd.DataFrame(_fit_residuals_loo(fd_arr, H), index=df.index, columns=tenors)
        vol    = resid.shift(1).ewm(halflife=ewma_hl, min_periods=min_periods).std()
        z      = resid / vol.clip(lower=1e-8)
        raw_sigs[pair] = -z.mean(axis=1)   # negate: above-NS → short

    if not raw_sigs:
        empty = pd.DataFrame()
        return empty, empty

    signal_df    = pd.DataFrame(raw_sigs).sort_index()
    common_pairs = signal_df.columns.intersection(rx_d.columns)
    if len(common_pairs) < 3:
        return signal_df, pd.DataFrame()

    cov_panel  = rx_d[common_pairs].ewm(halflife=ewma_hl, min_periods=min_periods).cov()
    common_idx = signal_df.index.intersection(rx_d.index)
    rebal      = _month_ends_idx(common_idx)

    wt_rec: dict = {}
    cov_dates = set(cov_panel.index.get_level_values(0))
    for d in rebal:
        if d not in cov_dates:
            continue
        sig_t = signal_df.loc[d].dropna()
        if len(sig_t) < 3:
            continue
        cov_t = cov_panel.loc[d]
        avail = sig_t.index.intersection(cov_t.index)
        if len(avail) < 3:
            continue
        try:
            w = ViewFormationTS(
                sig_t.loc[avail], cov_t.loc[avail, avail], {"desired_vol": desired_vol}
            ).generate_view()
            wt_rec[d] = w
        except Exception:
            pass

    positions_df = pd.DataFrame(wt_rec).T.reindex(common_idx).ffill()
    positions_df = positions_df.reindex(columns=rx_d.columns).fillna(0.0)
    positions_df.index.name = "date"

    return signal_df, positions_df


def _month_ends_idx(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby([s.index.year, s.index.month]).last().values)


# ── Main comparison entry point ────────────────────────────────────────────────

def compare_all(
    pairs: Optional[list[str]] = None,
    start: Optional[str] = "2005-01-01",
    end: Optional[str] = None,
    n: int = 4,
    path: Optional[str] = None,
    ewma_hl: int = 63,
    min_periods: int = 30,
    oos_start: Optional[str] = "2019-01-01",
) -> tuple:
    """
    Run all NS variant combinations and print a ranked comparison table.

    Variants (11 total) mirror rates/research/mean_reversion/:
      nsrv_compare analogues  : IS/LOO × cont/hyst
      rv_sharpe analogues     : NS2 model, fixed-λ
      slope_fly analogues     : fly 1m3m6m, 3m6m12m, 6m12m2y; slope 1m2y

    Signal is computed from `start` (for EWMA warmup); SR is measured only
    over [oos_start, end] so results are genuinely out-of-sample.
    λ calibration always uses data ≤ _TRAIN_END (2018-12-31).

    Returns (results_df, carry_sr).
    """
    if pairs is None:
        pairs = CARRY_PAIRS

    print("Loading forward panel...")
    fwd_panels = build_fwd_panel(pairs=pairs, start=start, end=end, path=path)
    if not fwd_panels:
        raise RuntimeError("No pairs loaded — check data path and pair list.")

    print("Pre-computing hat matrices (NS3 and NS2)...")
    H3 = _hat_matrices_k(TAU_YRS, LAM_GRID, k=3)
    H2 = _hat_matrices_k(TAU_YRS, LAM_GRID, k=2)

    print("Calibrating per-pair fixed λ on training data (≤ 2018)...")
    lam_fixed = _calibrate_lam(fwd_panels, LAM_GRID, k=3)

    load_kw = {"path": path} if path else {}
    panels  = build_panels(pairs=pairs, start=start, end=end, **load_kw)
    rx_d    = panels["rx_d"]
    fd_d    = panels["fd_1m"].resample("B").ffill()

    carry_pos = compute_xs_positions(fd_d, n=n)
    common_c  = carry_pos.columns.intersection(rx_d.columns)
    c_pnl     = (carry_pos[common_c].shift(1) * rx_d[common_c]).sum(axis=1)
    c_pnl     = c_pnl.loc[c_pnl != 0]
    c_oos_eval = c_pnl.loc[oos_start:] if oos_start else c_pnl
    c_sr      = float(c_oos_eval.mean() / c_oos_eval.std() * np.sqrt(252))

    def _pair_resid(pair, df, tenors, fd_arr, model, res_type, lam_mode):
        tau  = np.array([TAU_YRS[FWD_TENORS.index(t)] for t in tenors])
        k    = 3 if model == "ns3" else 2
        full = (tenors == FWD_TENORS)

        if lam_mode == "fixed":
            lam = lam_fixed.get(pair, float(np.median(LAM_GRID)))
            X   = ns_loadings(tau, lam)[:, :k]
            try:
                G = np.linalg.solve(X.T @ X, X.T)
            except np.linalg.LinAlgError:
                G = np.linalg.lstsq(X, np.eye(len(tau)), rcond=None)[0].T
            H1 = X @ G                              # single (n, n) hat matrix
            return fd_arr - (H1 @ fd_arr.T).T

        H = (H3 if k == 3 else H2) if full else _hat_matrices_k(tau, LAM_GRID, k=k)
        if res_type == "loo":
            return _fit_residuals_loo(fd_arr, H)
        return _fit_residuals(fd_arr, H)

    def _run(model, res_type, lam_mode, position, structure):
        sigs: dict[str, pd.Series] = {}
        for pair, df in fwd_panels.items():
            tenors = [t for t in FWD_TENORS if t in df.columns]
            if len(tenors) < 3:
                continue
            resid_arr = _pair_resid(pair, df, tenors, df[tenors].values,
                                    model, res_type, lam_mode)
            resid_df  = pd.DataFrame(resid_arr, index=df.index, columns=tenors)

            if position == "hyst":
                # Mirror rates nsrv_compare: apply hysteresis per-tenor, then aggregate.
                # The aggregate is too smooth to reliably cross ±1; per-tenor z-scores are not.
                vol    = resid_df.shift(1).ewm(halflife=ewma_hl, min_periods=min_periods).std()
                z_df   = (resid_df / vol.clip(lower=1e-8)).clip(-3.0, 3.0)
                hyst_df = z_df.apply(_hyst)         # {-1, 0, +1} per tenor
                sig     = -hyst_df.mean(axis=1)     # negate: positive per-tenor = short
            else:
                sig = _aggregate(resid_df, structure, hl=ewma_hl, mp=min_periods)
            sigs[pair] = sig

        sig_df = pd.DataFrame(sigs).sort_index()
        if sig_df.empty:
            return dict(SR=np.nan, skew=np.nan, maxDD=np.nan, corr=np.nan)

        pos    = compute_xs_positions(sig_df, n=n)
        common = pos.columns.intersection(rx_d.columns)
        pos_a  = pos[common].reindex(rx_d.index).fillna(0.0)
        pnl    = (pos_a.shift(1) * rx_d[common]).sum(axis=1).loc[lambda s: s != 0]

        # Restrict evaluation to OOS window; signal computed from full history for warmup
        oos_pnl = pnl.loc[oos_start:] if oos_start else pnl
        if len(oos_pnl) < 63:
            return dict(SR=np.nan, skew=np.nan, maxDD=np.nan, corr=np.nan)

        sr  = float(oos_pnl.mean() / oos_pnl.std() * np.sqrt(252))
        m, v = oos_pnl.mean(), oos_pnl.std()
        skw  = float(((oos_pnl - m) ** 3).mean() / v ** 3) if v > 0 else 0.0
        cum  = oos_pnl.cumsum()
        mdd  = float((cum - cum.cummax()).min())
        c_oos = c_pnl.loc[oos_start:] if oos_start else c_pnl
        idx   = oos_pnl.index.intersection(c_oos.index)
        corr  = float(np.corrcoef(oos_pnl.reindex(idx), c_oos.reindex(idx))[0, 1]) \
                if len(idx) > 10 else np.nan
        return dict(SR=sr, skew=skw, maxDD=mdd, corr=corr)

    def _run_vfts(model: str, res_type: str, lam_mode: str) -> dict:
        """VF TS variant: monthly rebalance via ViewFormationTS on the agg signal."""
        sigs: dict[str, pd.Series] = {}
        for pair, df in fwd_panels.items():
            tenors = [t for t in FWD_TENORS if t in df.columns]
            if len(tenors) < 3:
                continue
            resid_arr = _pair_resid(pair, df, tenors, df[tenors].values,
                                    model, res_type, lam_mode)
            resid_df  = pd.DataFrame(resid_arr, index=df.index, columns=tenors)
            vol  = resid_df.shift(1).ewm(halflife=ewma_hl, min_periods=min_periods).std()
            z_df = resid_df / vol.clip(lower=1e-8)
            sigs[pair] = -z_df.mean(axis=1)  # negate: above-NS fit → short

        if len(sigs) < 3:
            return dict(SR=np.nan, skew=np.nan, maxDD=np.nan, corr=np.nan)

        sig_df       = pd.DataFrame(sigs).sort_index()
        common_pairs = sig_df.columns.intersection(rx_d.columns)
        if len(common_pairs) < 3:
            return dict(SR=np.nan, skew=np.nan, maxDD=np.nan, corr=np.nan)

        cov_panel  = rx_d[common_pairs].ewm(halflife=ewma_hl, min_periods=min_periods).cov()
        common_idx = sig_df.index.intersection(rx_d.index)
        rebal      = _month_ends_idx(common_idx)

        wt_rec: dict = {}
        cov_dates = set(cov_panel.index.get_level_values(0))
        for d in rebal:
            if d not in cov_dates:
                continue
            sig_t = sig_df.loc[d].dropna()
            if len(sig_t) < 3:
                continue
            cov_t = cov_panel.loc[d]
            avail = sig_t.index.intersection(cov_t.index)
            if len(avail) < 3:
                continue
            try:
                w = ViewFormationTS(
                    sig_t.loc[avail], cov_t.loc[avail, avail], {"desired_vol": 0.10}
                ).generate_view()
                wt_rec[d] = w
            except Exception:
                pass

        if not wt_rec:
            return dict(SR=np.nan, skew=np.nan, maxDD=np.nan, corr=np.nan)

        wts_df = pd.DataFrame(wt_rec).T.reindex(common_idx).ffill()
        wts_df = wts_df.reindex(columns=rx_d.columns).fillna(0.0)
        pnl    = (wts_df.shift(1) * rx_d.reindex(common_idx)).sum(axis=1)
        pnl    = pnl.loc[pnl != 0]

        oos_pnl = pnl.loc[oos_start:] if oos_start else pnl
        if len(oos_pnl) < 63:
            return dict(SR=np.nan, skew=np.nan, maxDD=np.nan, corr=np.nan)

        sr   = float(oos_pnl.mean() / oos_pnl.std() * np.sqrt(252))
        m, v = oos_pnl.mean(), oos_pnl.std()
        skw  = float(((oos_pnl - m) ** 3).mean() / v ** 3) if v > 0 else 0.0
        cum  = oos_pnl.cumsum()
        mdd  = float((cum - cum.cummax()).min())
        c_oos = c_pnl.loc[oos_start:] if oos_start else c_pnl
        idx   = oos_pnl.index.intersection(c_oos.index)
        corr  = float(np.corrcoef(oos_pnl.reindex(idx), c_oos.reindex(idx))[0, 1]) \
                if len(idx) > 10 else np.nan
        return dict(SR=sr, skew=skw, maxDD=mdd, corr=corr)

    # Variant grid — (name, model, residuals, lambda, position, structure)
    VARIANTS = [
        # ── nsrv_compare analogues ──────────────────────────────────────────
        ("ns3·is·grid·cont·agg",        "ns3", "is",  "grid",  "cont", "agg"),
        ("ns3·is·grid·hyst·agg",        "ns3", "is",  "grid",  "hyst", "agg"),
        ("ns3·loo·grid·cont·agg",       "ns3", "loo", "grid",  "cont", "agg"),
        ("ns3·loo·grid·hyst·agg",       "ns3", "loo", "grid",  "hyst", "agg"),
        # ── rv_sharpe analogues (model / lambda variants) ───────────────────
        ("ns2·is·grid·cont·agg",        "ns2", "is",  "grid",  "cont", "agg"),
        ("ns2·loo·grid·cont·agg",       "ns2", "loo", "grid",  "cont", "agg"),
        ("ns3·is·fixed·cont·agg",       "ns3", "is",  "fixed", "cont", "agg"),
        # ── slope_fly analogues (structure variants) ────────────────────────
        ("ns3·is·grid·cont·fly1m3m6m",  "ns3", "is",  "grid",  "cont", "fly_1m3m6m"),
        ("ns3·is·grid·cont·fly3m6m12m", "ns3", "is",  "grid",  "cont", "fly_3m6m12m"),
        ("ns3·is·grid·cont·fly6m12m2y", "ns3", "is",  "grid",  "cont", "fly_6m12m2y"),
        ("ns3·is·grid·cont·slope1m2y",  "ns3", "is",  "grid",  "cont", "slope_1m2y"),
    ]

    print(f"Running {len(VARIANTS)} variants...\n")
    rows = []
    for name, model, res, lam, pos, struct in VARIANTS:
        r = _run(model, res, lam, pos, struct)
        rows.append({"variant": name, **r})
        print(f"  {name:<42}  SR={r['SR']:+.3f}  skew={r['skew']:+.3f}"
              f"  mdd={r['maxDD']:.4f}  corr={r['corr']:+.3f}")

    # VF TS variants — monthly rebalance, time-series view formation
    VFTS_VARIANTS = [
        ("ns3·is·grid·vfts·agg",  "ns3", "is",  "grid"),
        ("ns3·loo·grid·vfts·agg", "ns3", "loo", "grid"),
    ]
    for name, model, res, lam in VFTS_VARIANTS:
        r = _run_vfts(model, res, lam)
        rows.append({"variant": name, **r})
        print(f"  {name:<42}  SR={r['SR']:+.3f}  skew={r['skew']:+.3f}"
              f"  mdd={r['maxDD']:.4f}  corr={r['corr']:+.3f}")

    df_out = pd.DataFrame(rows).set_index("variant")

    print()
    oos_label = f"OOS {oos_start}→end" if oos_start else "full sample"
    print(f"\n{oos_label}")
    print(f"{'Variant':<42} {'SR':>6} {'Skew':>6} {'MaxDD':>9} {'Corr/carry':>11}")
    print("─" * 78)
    print(f"{'CARRY (benchmark)':<42} {c_sr:>+6.3f}")
    print("─" * 78)
    for _, row in df_out.sort_values("SR", ascending=False).iterrows():
        print(f"{row.name:<42} {row['SR']:>+6.3f} {row['skew']:>+6.3f}"
              f" {row['maxDD']:>9.4f} {row['corr']:>+11.3f}")

    return df_out, c_sr
