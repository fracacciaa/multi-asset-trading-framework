"""
nsrv_compare.py — four-way signal comparison on 11 OIS markets.

  existing_raw   in-sample NS (fit on 70-tenor grid)  + z-scored continuous
  existing_hyst  in-sample NS                          + hysteresis |z|>1, exit z≤0
  loo_raw        LOO-corrected NS (fit on obs tenors)  + z-scored continuous
  loo_hyst       LOO residuals                         + hysteresis

Factor neutralisation: DV01-weighted NS level/slope/curvature projection using
calibrated λ per market.  All residuals in %, z-scored with a rolling 125d vol
(lagged 1 day so date t uses data through t-1).

OOS: 2019-01-01 → today.
Section-15 diagnostic printed for US before the main table.
"""

import sys
import numpy as np
import pandas as pd

from rates.production.ois_data      import fetch_ois_curves, OIS_MARKETS
from rates.production.yc_builder    import build_zero_curve
from rates.production.nelson_siegel import (
    calibrate_lambda, extract_factors, _reconstruct_fitted, _ns_loadings,
)
from rates.production.ns_core import fit_ns, ns_loadings as nsrv_loadings

TRAIN_END = "2018-12-31"
OOS_START = "2019-01-01"
Z_EWM_HL     = 63     # EWMA halflife in days (≈ 125d rolling in memory)
Z_VOL_FLOOR  = 1e-4   # % — prevents division by near-zero
Z_CAP        = 3.0    # continuous signal cap
HYST_ENTRY   = 1.0    # |z| threshold to enter
MIN_PERIODS  = 30
LOO_PRIOR_W  = 0.0    # 0 = OLS; sweep shows market-mean prior hurts overall (prior degrades NZ/Japan signal)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _long_end_tenors(market: str, zc: pd.DataFrame) -> np.ndarray:
    """≥1Y observed tenors for this market, inferred from raw OIS data."""
    raw_cols = fetch_ois_curves(market).columns.values.astype(float)
    obs = np.sort(raw_cols[raw_cols >= 1.0])
    return np.array([t for t in obs if t in zc.columns])


def _dv01_mat(z_arr: np.ndarray, tenors: np.ndarray) -> np.ndarray:
    """DV01 matrix (T, N) from yield array (T, N) in %."""
    return tenors[None, :] * np.exp(-z_arr / 100.0 * tenors[None, :])


def _factor_neutral_splines(
    resid: np.ndarray,   # (T, N) residuals in %
    z_arr: np.ndarray,   # (T, N) zero yields in %
    tenors: np.ndarray,  # (N,)
    ns_load: np.ndarray, # (N, 3) NS loading matrix (fixed)
) -> np.ndarray:
    """
    Project each date's residual onto the complement of the NS factor space.
    Returns s_perp (T, N) — per-tenor factor-neutral residuals.
    """
    dv01 = _dv01_mat(z_arr, tenors)                                  # (T, N)
    D    = dv01[:, :, None] * ns_load[None, :, :]                    # (T, N, 3)
    DtD  = np.einsum('tni,tnj->tij', D, D)                           # (T, 3, 3)
    Dts  = np.einsum('tni,tn->ti',   D, resid)                       # (T, 3)
    coef = np.linalg.solve(DtD, Dts[:, :, None]).squeeze(-1)         # (T, 3)
    return resid - np.einsum('tni,ti->tn', D, coef)                  # (T, N)


def _zscored(
    s_perp: pd.DataFrame,
) -> pd.DataFrame:
    """EWMA z-score: vol estimated using data through t-1 (no lookahead)."""
    vol = s_perp.shift(1).ewm(halflife=Z_EWM_HL, min_periods=MIN_PERIODS).std()
    return s_perp / vol.clip(lower=Z_VOL_FLOOR)


def _pos_raw(z: pd.DataFrame) -> pd.DataFrame:
    """Continuous signal: pos = +clip(z, -cap, cap). Long when residual is positive (cheap tenor)."""
    return z.clip(-Z_CAP, Z_CAP)


def _pos_hysteresis_series(z: pd.Series) -> pd.Series:
    """Per-tenor hysteresis: enter at |z|>1, exit when z crosses zero."""
    out = np.zeros(len(z))
    cur = 0.0
    zv  = z.to_numpy()
    for i in range(len(zv)):
        if np.isnan(zv[i]):
            out[i] = cur
            continue
        if cur == 0.0:
            if   zv[i] >  HYST_ENTRY: cur =  1.0
            elif zv[i] < -HYST_ENTRY: cur = -1.0
        else:
            if cur ==  1.0 and zv[i] <= 0.0: cur = 0.0
            elif cur == -1.0 and zv[i] >= 0.0: cur = 0.0
        out[i] = cur
    return pd.Series(out, index=z.index)


def _pos_hysteresis(z: pd.DataFrame) -> pd.DataFrame:
    return z.apply(_pos_hysteresis_series)


def _pnl_and_metrics(
    pos: pd.DataFrame,            # (T, N) position SET at close of t
    resid_df: pd.DataFrame,       # (T, N) residuals
    oos_start: str = OOS_START,
) -> dict:
    """
    P&L: pos(t-1) × (-Δresid(t))  — position held over t drives return at t.
    Returns Sharpe and turnover over OOS period.
    """
    delta  = resid_df.diff()                      # (T, N) changes in residual
    pnl_df = pos.shift(1) * (-delta)              # (T, N) per-tenor P&L
    pnl    = pnl_df.sum(axis=1)                   # (T,) portfolio P&L

    to_df  = pos.diff().abs()                      # (T, N) |Δpos| per tenor per day
    to_ann = to_df.mean().mean() * 252             # scalar annualised turnover

    oos = pnl.loc[oos_start:].dropna()
    if len(oos) < 60:
        sharpe = np.nan
    else:
        sharpe = float(oos.mean() / oos.std() * np.sqrt(252))

    return {"sharpe": sharpe, "turnover": to_ann}


# ── LOO residual panel ─────────────────────────────────────────────────────────

def _market_prior(
    zc_obs: pd.DataFrame,
    lam_nsrv: float,
) -> np.ndarray:
    """
    OLS-fit NS on each training-period (≤TRAIN_END) date, return mean beta (k=3).
    Used as the market-specific prior centre for LOO fits.
    """
    tenors = zc_obs.columns.values.astype(float)
    train  = zc_obs.loc[:TRAIN_END]
    betas  = []
    for _, row in train.iterrows():
        y_full = row.values.astype(float)
        valid  = np.isfinite(y_full)
        if valid.sum() < 4:
            continue
        t, y = tenors[valid], y_full[valid]
        betas.append(fit_ns(t, y, lam_nsrv).beta)
    if len(betas) < 30:
        return np.zeros(3)
    return np.mean(betas, axis=0)


def _build_loo_panel(
    zc_obs:     pd.DataFrame,   # date × avail_tenors, yields in %
    lam_nsrv:   float,          # lambda in 1/years (= 1 / lam_years)
    prior_mean: np.ndarray,     # (3,) market-specific mean beta (unused when LOO_PRIOR_W=0)
) -> pd.DataFrame:
    """
    Day loop: fit_ns per date on observed tenors only, return resid_loo.
    LOO_PRIOR_W=0 uses OLS; set >0 to enable market-mean ridge prior.
    """
    tenors  = zc_obs.columns.values.astype(float)
    prior_w = np.full(3, LOO_PRIOR_W) if LOO_PRIOR_W > 0 else None
    pm      = prior_mean               if LOO_PRIOR_W > 0 else None
    rows    = {}
    for date, row in zc_obs.iterrows():
        y_full = row.values.astype(float)
        valid  = np.isfinite(y_full)
        if valid.sum() < 4:
            rows[date] = np.full(len(tenors), np.nan)
            continue
        t, y = tenors[valid], y_full[valid]
        fit  = fit_ns(t, y, lam_nsrv, prior_mean=pm, prior_w=prior_w)

        loo = np.full(len(tenors), np.nan)
        idx = np.where(valid)[0]
        loo[idx] = fit.resid_loo
        rows[date] = loo

    return pd.DataFrame(rows, index=tenors).T   # date × tenors


# ── Section-15 diagnostic ──────────────────────────────────────────────────────

def _section15_diagnostic(
    market: str,
    zc_obs: pd.DataFrame,
    lam_nsrv: float,
) -> pd.DataFrame:
    """
    LOO-to-in-sample sd ratio, leverage, half-life by tenor.
    If ratio > 1.5 at any tenor, the in-sample residual was badly understated.
    """
    tenors = zc_obs.columns.values.astype(float)
    resid_in  = np.full((len(zc_obs), len(tenors)), np.nan)
    resid_loo = np.full((len(zc_obs), len(tenors)), np.nan)
    lev_arr   = np.full((len(zc_obs), len(tenors)), np.nan)

    for i, (date, row) in enumerate(zc_obs.iterrows()):
        y_full = row.values.astype(float)
        valid  = np.isfinite(y_full)
        if valid.sum() < 4:
            continue
        t, y = tenors[valid], y_full[valid]
        fit  = fit_ns(t, y, lam_nsrv)
        idx  = np.where(valid)[0]
        resid_in [i, idx] = fit.resid
        resid_loo[i, idx] = fit.resid_loo
        lev_arr  [i, idx] = fit.leverage

    diag = {}
    for j, t in enumerate(tenors):
        ri  = resid_in[:, j][np.isfinite(resid_in[:, j])]
        rl  = resid_loo[:, j][np.isfinite(resid_loo[:, j])]
        lev = lev_arr[:, j][np.isfinite(lev_arr[:, j])]
        if len(ri) < 30:
            continue
        # AR(1) half-life on LOO residual
        hl = np.nan
        if len(rl) > 50:
            b = np.polyfit(rl[:-1], rl[1:], 1)[0]
            if -2.0 < b < 0.0:
                hl = -np.log(2.0) / np.log(1.0 + b)
        diag[t] = {
            "mean_resid_loo":   round(float(np.mean(rl)), 4),
            "sd_resid_in":      round(float(np.std(ri)),  4),
            "sd_resid_loo":     round(float(np.std(rl)),  4),
            "loo_vs_insample":  round(float(np.std(rl) / max(np.std(ri), 1e-9)), 3),
            "mean_leverage":    round(float(np.mean(lev)), 3),
            "halflife_days":    round(float(hl), 1) if np.isfinite(hl) else np.nan,
        }

    return pd.DataFrame(diag).T


# ── Per-market computation ─────────────────────────────────────────────────────

def _market_results(
    market:   str,
    lambdas:  dict,
    run_diag: bool = False,
) -> dict[str, dict]:
    raw = fetch_ois_curves(market)
    zc  = build_zero_curve(raw)

    avail   = _long_end_tenors(market, zc)
    zc_obs  = zc[list(avail)]

    ns_lam  = lambdas["ns"][market]            # years (existing convention)
    lam_nsrv = 1.0 / ns_lam                   # 1/years (nsrv convention)
    ns_load = _ns_loadings(avail, ns_lam)      # (N, 3) fixed projection basis

    if run_diag:
        print(f"\n── Section-15 diagnostic: {market} ──")
        diag = _section15_diagnostic(market, zc_obs, lam_nsrv)
        pd.set_option("display.float_format", "{:.3f}".format)
        print(diag.to_string())
        pd.set_option("display.float_format", "{:.2f}".format)

    # ── 1. Existing in-sample residuals (fit on full 70-tenor grid) ───────────
    fac_cal   = extract_factors(zc, model="ns_calibrated", lam=ns_lam)
    fit_cal   = _reconstruct_fitted(zc, fac_cal, "ns_calibrated", lam=ns_lam)
    common    = zc_obs.index.intersection(fit_cal.index)
    resid_ex  = zc_obs.loc[common].subtract(fit_cal.loc[common, list(avail)])

    # ── 2. LOO residuals (fit on observed tenors only, market-mean prior) ────────
    prior_mean = _market_prior(zc_obs, lam_nsrv)
    resid_loo  = _build_loo_panel(zc_obs, lam_nsrv, prior_mean)
    common_l   = zc_obs.index.intersection(resid_loo.index)
    resid_loo  = resid_loo.loc[common_l]

    results: dict[str, dict] = {}

    for label, resid_df in [("existing", resid_ex), ("loo", resid_loo)]:
        z_arr   = zc_obs.loc[resid_df.index, list(avail)].values.astype(float)
        r_arr   = resid_df.values.astype(float)

        # Factor-neutral per-tenor projection
        has_nan = ~(np.all(np.isfinite(r_arr), axis=1) & np.all(np.isfinite(z_arr), axis=1))
        r_arr[has_nan]  = 0.0   # temporary zero-fill for batched solve
        z_arr[has_nan]  = 3.0   # placeholder yields (DV01 won't matter; we NaN out below)

        s_perp = _factor_neutral_splines(r_arr, z_arr, avail, ns_load)   # (T, N)
        s_perp[has_nan] = np.nan

        s_df   = pd.DataFrame(s_perp, index=resid_df.index, columns=avail)
        z_df   = _zscored(s_df)

        for sig, pos_fn in [("raw", _pos_raw), ("hyst", _pos_hysteresis)]:
            pos = pos_fn(z_df)
            key = f"{label}_{sig}"
            results[key] = _pnl_and_metrics(pos, s_df)

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Calibrating NS lambdas on train (≤2018)...", flush=True)
    lambdas = calibrate_lambda(end=TRAIN_END)
    for mkt in OIS_MARKETS:
        print(f"  {mkt}...", flush=True)

    print("\nRunning markets...", flush=True)

    sharpe_rows: dict[str, dict] = {}
    to_rows:     dict[str, dict] = {}

    for i, mkt in enumerate(OIS_MARKETS):
        print(f"  {mkt}...", flush=True)
        run_diag = (i == 0)   # Section-15 diagnostic on the first market only
        res = _market_results(mkt, lambdas, run_diag=run_diag)
        sharpe_rows[mkt] = {k: v["sharpe"]   for k, v in res.items()}
        to_rows[mkt]     = {k: v["turnover"] for k, v in res.items()}

    cols = ["existing_raw", "existing_hyst", "loo_raw", "loo_hyst"]

    sh_df = pd.DataFrame(sharpe_rows).T[cols]
    to_df = pd.DataFrame(to_rows).T[cols]

    sh_df.columns = ["ex_raw", "ex_hyst", "loo_raw", "loo_hyst"]
    to_df.columns = ["ex_raw", "ex_hyst", "loo_raw", "loo_hyst"]

    sh_df["mean"] = sh_df.mean(axis=1)
    sh_df = sh_df.sort_values("mean", ascending=False)
    to_df = to_df.loc[sh_df.index]

    print(f"\nOOS Sharpe ({OOS_START}→today) — factor-neutral, z-scored, observed tenors\n")
    pd.set_option("display.float_format", "{:.2f}".format)
    pd.set_option("display.width", 120)
    print(sh_df.to_string())

    print(f"\nAnnualised turnover (mean |Δpos| per tenor × 252)\n")
    print(to_df.to_string())


if __name__ == "__main__":
    main()
