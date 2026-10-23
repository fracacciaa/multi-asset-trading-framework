"""
Nelson-Siegel fitting and residual panel construction.

Replicates notebook cells 22 (ns_loadings), 25 (estimate_ns_panel),
34 (build_ns_residual_panels).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from commodities.power_ns.config import (
    CURVE_MAX_RANK,
    TRADE_MAX_RANK,
    MIN_OBS_FOR_RANK,
    NS_CURVATURE_PEAK_X,
    LAMBDA_FIT,
)


# ── Core NS function ──────────────────────────────────────────────────────────


def ns_loadings(M, lam) -> tuple[np.ndarray, np.ndarray]:
    """
    Nelson-Siegel loadings.

        Level     = 1
        Slope     = (1 - exp(-lambda*M)) / (lambda*M)
        Curvature = Slope - exp(-lambda*M)

    Parameters
    ----------
    M : array-like, shape (...,)
    lam : scalar or array-like, broadcastable with M

    Returns
    -------
    slope, curvature : ndarray
    """
    M   = np.asarray(M, dtype=float)
    x   = lam * M
    slope     = (1.0 - np.exp(-x)) / x
    curvature = slope - np.exp(-x)
    return slope, curvature


# ── Panel NS estimator ────────────────────────────────────────────────────────


def estimate_ns_panel(
    prices: pd.DataFrame,
    tickers: pd.DataFrame,
    MATURITY: pd.DataFrame,
    universe: list[str],
    market_depth: Optional[pd.Series] = None,
    max_curve_rank: int = CURVE_MAX_RANK,
    lambda_fit=LAMBDA_FIT,
    ns_curvature_peak_x: float = NS_CURVATURE_PEAK_X,
    min_obs_for_rank: int = MIN_OBS_FOR_RANK,
) -> pd.DataFrame:
    """
    Estimate Nelson-Siegel using the deepest valid contiguous front strip
    available on each market/date, with a minimum of four contracts.

    Fallback: M1-M8 → M1-M7 → ... → M1-M4.

    Parameters
    ----------
    prices, tickers, MATURITY : DataFrames indexed by date, columns as {root}{k}
    universe : list of market root strings (e.g. ["ZUB", "ITP", ...])
    market_depth : optional Series {root: max_rank}, limits per-market rank cap
    max_curve_rank : global upper bound on how many ranks to use in the fit
    lambda_fit : None = depth-adaptive lambda; float = frozen lambda
    ns_curvature_peak_x : x* s.t. NS curvature peaks at x* = lambda * M_bar
    min_obs_for_rank : causal warm-up: rank eligible only after this many obs

    Returns
    -------
    DataFrame indexed by (Date, Commodity) with columns
    beta_L, beta_S, beta_C, lambda, R2_full, R2_LS, R2_LC, N_contracts, M1..Mn
    """
    common_dates = prices.index.intersection(tickers.index).sort_values()
    prices_  = prices.loc[common_dates]
    tickers_ = tickers.loc[common_dates]
    MATURITY_  = MATURITY.reindex(common_dates)
    frames = []

    for root in universe:
        root_cap = max_curve_rank
        if market_depth is not None and root in market_depth.index:
            root_cap = min(root_cap, int(market_depth.loc[root]))

        cols = [f"{root}{k}" for k in range(1, root_cap + 1)]
        if root_cap < 4 or not all(
            c in prices_.columns and c in tickers_.columns for c in cols[:4]
        ):
            continue

        # Keep only contiguous columns that actually exist, never below M1-M4.
        existing = []
        for c in cols:
            if c in prices_.columns and c in tickers_.columns:
                existing.append(c)
            else:
                break
        root_cap = len(existing)
        cols     = existing
        if root_cap < 4:
            continue

        y_all  = prices_[cols]
        tk_all = tickers_[cols]

        # Causal rank maturity: eligible only once MIN_OBS_FOR_RANK obs have accumulated.
        _obs    = (y_all.notna() & tk_all.notna())
        _mature = (_obs.cumsum() >= min_obs_for_rank)
        mature_through = _mature.cummin(axis=1).to_numpy()

        M_all = MATURITY_.reindex(common_dates)[cols].to_numpy(float)

        unresolved = np.ones(len(common_dates), dtype=bool)

        def batched_fit(X, yy):
            try:
                pinv_x = np.linalg.pinv(X)
                return np.einsum("nij,nj->ni", pinv_x, yy)
            except np.linalg.LinAlgError:
                out = np.full((len(yy), X.shape[2]), np.nan)
                for ii in range(len(yy)):
                    try:
                        out[ii] = np.linalg.lstsq(X[ii], yy[ii], rcond=None)[0]
                    except np.linalg.LinAlgError:
                        pass
                return out

        for n_points in range(root_cap, 3, -1):
            front_cols = cols[:n_points]
            front_obs  = (
                y_all[front_cols].notna() & tk_all[front_cols].notna()
            ).all(axis=1).to_numpy()

            candidate = unresolved & front_obs & mature_through[:, n_points - 1]
            if not candidate.any():
                continue

            candidate_idx = np.flatnonzero(candidate)
            dates_n = common_dates[candidate_idx]
            y       = y_all.loc[dates_n, front_cols].to_numpy(dtype=float)
            tk      = tk_all.loc[dates_n, front_cols]
            M       = M_all[candidate_idx, :n_points]

            distinct_tk = tk.nunique(axis=1).to_numpy() == n_points
            distinct_M  = np.array([
                len(np.unique(np.round(row, 10))) == n_points
                if np.isfinite(row).all() else False
                for row in M
            ])

            valid = (
                np.isfinite(y).all(axis=1)
                & np.isfinite(M).all(axis=1)
                & (M > 0).all(axis=1)
                & distinct_tk
                & distinct_M
            )
            if not valid.any():
                continue

            dates_v = dates_n[valid]
            y_v     = y[valid]
            M_v     = M[valid]

            M_bar = M_v.mean(axis=1)
            lam   = (
                np.full(len(M_bar), float(lambda_fit))
                if lambda_fit is not None
                else ns_curvature_peak_x / M_bar
            )

            x_lam  = lam[:, None] * M_v
            slope  = (1.0 - np.exp(-x_lam)) / x_lam
            curv   = slope - np.exp(-x_lam)
            ones   = np.ones_like(slope)

            X_full = np.stack([ones, slope, curv], axis=2)
            beta_full  = batched_fit(X_full, y_v)
            fitted_full = np.einsum("nij,nj->ni", X_full, beta_full)

            X_LS      = np.stack([ones, slope], axis=2)
            beta_LS   = batched_fit(X_LS, y_v)
            fitted_LS = np.einsum("nij,nj->ni", X_LS, beta_LS)

            X_LC      = np.stack([ones, curv], axis=2)
            beta_LC   = batched_fit(X_LC, y_v)
            fitted_LC = np.einsum("nij,nj->ni", X_LC, beta_LC)

            fit_ok = np.isfinite(beta_full).all(axis=1)
            if not fit_ok.any():
                continue

            dates_v    = dates_v[fit_ok]
            y_v        = y_v[fit_ok]
            M_v        = M_v[fit_ok]
            lam        = lam[fit_ok]
            beta_full  = beta_full[fit_ok]
            fitted_full = fitted_full[fit_ok]
            fitted_LS  = fitted_LS[fit_ok]
            fitted_LC  = fitted_LC[fit_ok]

            y_mean = y_v.mean(axis=1, keepdims=True)
            sst    = ((y_v - y_mean) ** 2).sum(axis=1)

            def batch_r2(fitted):
                sse = ((y_v - fitted) ** 2).sum(axis=1)
                return np.where(sst > 0, 1.0 - sse / sst, np.nan)

            data = {
                "Date":       dates_v,
                "Commodity":  root,
                "beta_L":     beta_full[:, 0],
                "beta_S":     beta_full[:, 1],
                "beta_C":     beta_full[:, 2],
                "lambda":     lam,
                "R2_full":    batch_r2(fitted_full),
                "R2_LS":      batch_r2(fitted_LS),
                "R2_LC":      batch_r2(fitted_LC),
                "N_contracts": n_points,
            }
            for j in range(n_points):
                data[f"M{j+1}"] = M_v[:, j]

            frame = pd.DataFrame(data).replace([np.inf, -np.inf], np.nan)
            frame = frame.dropna(subset=["beta_L", "beta_S", "beta_C"])

            if frame.empty:
                continue

            frames.append(frame)
            resolved_dates = frame["Date"].to_numpy()
            resolved_pos   = common_dates.get_indexer(resolved_dates)
            unresolved[resolved_pos] = False

    if not frames:
        return pd.DataFrame()

    out = (
        pd.concat(frames, ignore_index=True)
        .set_index(["Date", "Commodity"])
        .sort_index()
    )
    assert not out.index.duplicated().any(), "Fallback fitter produced duplicate market/date NS fits"
    return out


# ── Residual panel builder ────────────────────────────────────────────────────


def build_ns_residual_panels(
    ns_results: pd.DataFrame,
    prices: pd.DataFrame,
    tickers: pd.DataFrame,
    MATURITY: pd.DataFrame,
    universe: list[str],
    max_curve_rank: int = CURVE_MAX_RANK,
) -> tuple[dict, dict, dict]:
    """
    Reconstruct fitted prices and residuals for each rank.

    Returns (actual_panels, fitted_panels, residual_panels) — each a dict
    keyed by rank (1..max_curve_rank) of DataFrames indexed by date with
    columns = universe.
    """
    idx = prices.index

    def _empty():
        return {
            k: pd.DataFrame(np.nan, index=idx, columns=universe, dtype=float)
            for k in range(1, max_curve_rank + 1)
        }

    actual_panels   = _empty()
    fitted_panels   = _empty()
    residual_panels = _empty()

    for root in universe:
        try:
            x = ns_results.xs(root, level="Commodity").sort_index()
        except KeyError:
            continue

        for k in range(1, max_curve_rank + 1):
            col = f"{root}{k}"
            if col not in prices.columns or col not in tickers.columns:
                continue

            dates = x.index[x["N_contracts"] >= k]
            if len(dates) == 0:
                continue

            px  = pd.to_numeric(prices.loc[dates, col], errors="coerce")
            tk  = tickers.loc[dates, col]
            M_col = MATURITY.reindex(dates)[col].to_numpy(float)

            beta_L = x.loc[dates, "beta_L"].to_numpy(float)
            beta_S = x.loc[dates, "beta_S"].to_numpy(float)
            beta_C = x.loc[dates, "beta_C"].to_numpy(float)
            lam    = x.loc[dates, "lambda"].to_numpy(float)

            good = (
                px.notna().to_numpy()
                & np.isfinite(M_col)
                & (M_col > 0)
                & np.isfinite(lam)
            )
            if not good.any():
                continue

            dates_g = dates[good]
            M_g     = M_col[good]
            lam_g   = lam[good]
            slope, curvature = ns_loadings(M_g, lam_g)
            fitted = beta_L[good] + beta_S[good] * slope + beta_C[good] * curvature
            actual = px.to_numpy(float)[good]

            actual_panels[k].loc[dates_g, root]   = actual
            fitted_panels[k].loc[dates_g, root]   = fitted
            residual_panels[k].loc[dates_g, root] = actual - fitted

    return actual_panels, fitted_panels, residual_panels
