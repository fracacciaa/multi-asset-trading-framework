"""
rv_vf_weights.py — Cross-market VF allocation on the rv_raw signal.

Two-layer architecture:
  Layer 1 (within market, unchanged):
    Per-tenor EWMA z-scored factor-neutral residuals → positions → pnl_raw_m(t)
    Identical to nsrv_compare existing_raw.

  Layer 2 (cross-market, new):
    VF looks at the aggregate z-score per market and returns a scalar weight w_m(t).
    Weights are clipped ≥ 0 so we never flip a market's within-market trade direction.
    Final P&L = sum_m w_m(t−1) × pnl_raw_m(t)

Signal for Layer 2:
  sig_m(t) = mean_τ z(t, τ)   (average of per-tenor z-scores)

Covariance for Layer 2:
  EWMA cov of pnl_raw across markets (halflife=63d)

Variants:
  current      equal unit weight (nsrv_compare baseline)
  vfts         VF TS — sig/ann_vol → ortho → vol_scale, clipped ≥ 0
  vfxs_noclip  VF XS — demean → grinold → ortho → zscore → vol_scale, clipped ≥ 0

OOS: 2019-01-01 → today. Block bootstrap (block=63d, n=2000).
"""

import numpy as np
import pandas as pd

from rates.ois_data      import fetch_ois_curves, OIS_MARKETS
from rates.yc_builder    import build_zero_curve
from rates.nelson_siegel import (
    calibrate_lambda, extract_factors, _reconstruct_fitted, _ns_loadings,
)
from view_formation import ViewFormationTS

TRAIN_END   = "2018-12-31"
OOS_START   = "2019-01-01"
Z_EWM_HL    = 63
Z_VOL_FLOOR = 1e-4
Z_CAP       = 3.0
MIN_PERIODS = 30
EWMA_HL     = 63
DESIRED_VOL = 0.10
BLOCK_SIZE  = 63
N_BOOT      = 2000
RNG         = np.random.default_rng(42)


# ── Factor-neutral helpers ──────────────────────────────────────────────────────

def _long_end_tenors(market: str, zc: pd.DataFrame) -> np.ndarray:
    raw_cols = fetch_ois_curves(market).columns.values.astype(float)
    obs = np.sort(raw_cols[raw_cols >= 1.0])
    return np.array([t for t in obs if t in zc.columns])


def _factor_neutral_splines(
    resid:   np.ndarray,
    z_arr:   np.ndarray,
    tenors:  np.ndarray,
    ns_load: np.ndarray,
) -> np.ndarray:
    """Project residuals onto the complement of the NS factor space. Returns s_perp (T, N)."""
    dv01 = tenors[None, :] * np.exp(-z_arr / 100.0 * tenors[None, :])
    D    = dv01[:, :, None] * ns_load[None, :, :]
    DtD  = np.einsum('tni,tnj->tij', D, D)
    Dts  = np.einsum('tni,tn->ti',   D, resid)
    coef = np.linalg.solve(DtD, Dts[:, :, None]).squeeze(-1)
    return resid - np.einsum('tni,ti->tn', D, coef)


def _zscored_df(s: pd.DataFrame) -> pd.DataFrame:
    vol = s.shift(1).ewm(halflife=Z_EWM_HL, min_periods=MIN_PERIODS).std()
    return s / vol.clip(lower=Z_VOL_FLOOR)


# ── VF XS no-clip ──────────────────────────────────────────────────────────────

def _vfxs_noclip(signal: pd.Series, cov: pd.DataFrame) -> pd.Series:
    common = signal.dropna().index.intersection(cov.index)
    if len(common) < 3:
        return pd.Series(dtype=float)
    s  = signal.loc[common].astype(float)
    c  = cov.loc[common, common]
    ca = c.values

    vols = pd.Series(np.sqrt(np.diag(ca)), index=c.index).replace(0, np.nan)
    w_mkt = (1.0 / vols).dropna()
    w_mkt = (w_mkt / w_mkt.sum()).reindex(c.index).fillna(0.0)
    port_var  = float(w_mkt.values @ ca @ w_mkt.values)
    betas     = pd.Series(ca @ w_mkt.values / max(port_var, 1e-12), index=c.index)
    total_var = pd.Series(np.diag(ca), index=c.index)
    idio_vols = np.sqrt((total_var - (betas ** 2) * port_var).clip(lower=total_var * 0.05))

    s = s - s.mean()
    s = (s / idio_vols.reindex(s.index).replace(0, np.nan)).dropna()
    b  = betas.reindex(s.index).fillna(0.0)
    b2 = float(b @ b)
    if b2 > 1e-12:
        s = s - float(s @ b) / b2 * b
    std = s.std()
    if std > 1e-12:
        s = (s - s.mean()) / std
    w_ = s.reindex(c.index).fillna(0.0).values
    pv = float(w_ @ ca @ w_)
    return s * (DESIRED_VOL / np.sqrt(252) / np.sqrt(max(pv, 1e-12)))


# ── Month-end rebalance dates ───────────────────────────────────────────────────

def _month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby([s.index.year, s.index.month]).last().values)


# ── Metrics ─────────────────────────────────────────────────────────────────────

def _sharpe(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 60:
        return np.nan
    sd = s.std()
    return float(s.mean() / sd * np.sqrt(252)) if sd > 1e-12 else np.nan


def _block_bootstrap(arr: np.ndarray) -> np.ndarray:
    T = len(arr)
    n_blocks = int(np.ceil(T / BLOCK_SIZE))
    boots = np.empty(N_BOOT)
    for b in range(N_BOOT):
        starts = RNG.integers(0, max(T - BLOCK_SIZE + 1, 1), size=n_blocks)
        boot = np.concatenate([arr[s:s + BLOCK_SIZE] for s in starts])[:T]
        sd = boot.std()
        boots[b] = float(boot.mean() / sd * np.sqrt(252)) if sd > 1e-12 else np.nan
    return boots


# ── Per-market Layer 1 ─────────────────────────────────────────────────────────

def _build_market_layer1(market: str, lambdas: dict) -> tuple[pd.Series, pd.Series]:
    """
    Layer 1: unchanged rv_raw per-tenor signal.
    Returns (pnl_raw, sig_agg) for the market.

    pnl_raw(t)  = sum_τ pos(t−1, τ) × (−Δs_perp(t, τ))   [Layer 1 P&L]
    sig_agg(t)  = mean_τ z(t, τ)                            [Layer 2 signal]
    """
    raw = fetch_ois_curves(market)
    zc  = build_zero_curve(raw)

    avail   = _long_end_tenors(market, zc)
    zc_obs  = zc[list(avail)]
    ns_lam  = lambdas["ns"][market]
    ns_load = _ns_loadings(avail, ns_lam)

    fac_cal  = extract_factors(zc, model="ns_calibrated", lam=ns_lam)
    fit_cal  = _reconstruct_fitted(zc, fac_cal, "ns_calibrated", lam=ns_lam)
    common   = zc_obs.index.intersection(fit_cal.index)
    resid_df = zc_obs.loc[common].subtract(fit_cal.loc[common, list(avail)])

    z_arr  = zc_obs.loc[common].values.astype(float)
    r_arr  = resid_df.values.astype(float)

    has_nan = ~(np.all(np.isfinite(r_arr), axis=1) & np.all(np.isfinite(z_arr), axis=1))
    r_arr[has_nan] = 0.0
    z_arr[has_nan] = 3.0
    s_perp = _factor_neutral_splines(r_arr, z_arr, avail, ns_load)
    s_perp[has_nan] = np.nan

    s_df = pd.DataFrame(s_perp, index=common, columns=avail)
    z_df = _zscored_df(s_df)
    pos  = z_df.clip(-Z_CAP, Z_CAP)

    pnl_raw = (pos.shift(1) * (-s_df.diff())).sum(axis=1).rename(market)
    sig_agg = z_df.mean(axis=1).rename(market)   # mean of per-tenor z-scores

    return pnl_raw, sig_agg


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Calibrating NS lambdas on train (≤2018)...", flush=True)
    lambdas = calibrate_lambda(end=TRAIN_END)

    pnl_d:  dict[str, pd.Series] = {}
    sig_d:  dict[str, pd.Series] = {}

    for mkt in OIS_MARKETS:
        print(f"  {mkt}...", flush=True)
        pnl_d[mkt], sig_d[mkt] = _build_market_layer1(mkt, lambdas)

    pnl_df = pd.DataFrame(pnl_d)   # date × market  (Layer 1 P&L)
    sig_df = pd.DataFrame(sig_d)   # date × market  (Layer 2 signal)

    # EWMA cov of Layer 1 P&L across markets (covariance of what we're actually trading)
    print("Computing EWMA covariance of pnl_raw...", flush=True)
    cov_panel = pnl_df.ewm(halflife=EWMA_HL, min_periods=MIN_PERIODS).cov()

    common = pnl_df.index.intersection(sig_df.index)
    rebal  = _month_ends(common)

    VARIANTS = ["vfts", "vfxs_noclip"]
    wt_rec: dict[str, dict] = {v: {} for v in VARIANTS}

    print("Running Layer 2 VF backtest...", flush=True)
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
        sig_t = sig_t.loc[avail]
        cov_t = cov_t.loc[avail, avail]

        try:
            w = ViewFormationTS(sig_t, cov_t, {"desired_vol": DESIRED_VOL}).generate_view()
            wt_rec["vfts"][d] = w.clip(lower=0)   # never short a market's RV trade
        except Exception:
            pass

        try:
            w = _vfxs_noclip(sig_t, cov_t)
            wt_rec["vfxs_noclip"][d] = w.clip(lower=0)
        except Exception:
            pass

    # Forward-fill weights
    wts = {}
    for v in VARIANTS:
        df = pd.DataFrame(wt_rec[v]).T.reindex(common).ffill()
        df = df.reindex(columns=OIS_MARKETS).fillna(0.0)
        wts[v] = df

    # Layer 2 P&L = w_m(t−1) × pnl_raw_m(t)
    pnl_aligned = pnl_df.reindex(common)
    pnl_vf = {v: wts[v].shift(1).multiply(pnl_aligned) for v in VARIANTS}

    # Current baseline = equal unit weight (sum across markets)
    pnl_current = pnl_aligned

    pd.set_option("display.float_format", "{:.2f}".format)
    pd.set_option("display.width", 120)

    # Per-market OOS Sharpe
    rows = {}
    for mkt in OIS_MARKETS:
        rows[mkt] = {"current": _sharpe(pnl_current[mkt].loc[OOS_START:])}
        for v in VARIANTS:
            rows[mkt][v] = _sharpe(pnl_vf[v][mkt].loc[OOS_START:]) if mkt in pnl_vf[v] else np.nan

    cols = ["current"] + VARIANTS
    sh = pd.DataFrame(rows).T[cols]
    sh["mean"] = sh.mean(axis=1)
    sh = sh.sort_values("mean", ascending=False)

    print(f"\nOOS per-market Sharpe ({OOS_START} → today)\n")
    print(sh.to_string())

    # Portfolio Sharpe
    print("\nPortfolio Sharpe (sum across markets):\n")
    port_current = pnl_current.sum(axis=1).loc[OOS_START:].dropna()
    print(f"  current: {_sharpe(port_current):.2f}")
    for v in VARIANTS:
        port = pnl_vf[v].sum(axis=1).loc[OOS_START:].dropna()
        print(f"  {v}: {_sharpe(port):.2f}")

    # Block bootstrap
    print(f"\nBlock bootstrap (block={BLOCK_SIZE}d, n={N_BOOT})...", flush=True)
    boot_rows = []
    for label, arr in [
        ("current", port_current.values),
        *[(v, pnl_vf[v].sum(axis=1).loc[OOS_START:].dropna().values) for v in VARIANTS],
    ]:
        boots = _block_bootstrap(arr)
        boot_rows.append({
            "variant": label,
            "sharpe":  round(_sharpe(pd.Series(arr)), 2),
            "p5":      round(float(np.nanpercentile(boots, 5)), 2),
            "p25":     round(float(np.nanpercentile(boots, 25)), 2),
            "p75":     round(float(np.nanpercentile(boots, 75)), 2),
            "p95":     round(float(np.nanpercentile(boots, 95)), 2),
            "pct_pos": round(float(np.nanmean(boots > 0)) * 100, 1),
        })

    boot_df = pd.DataFrame(boot_rows).set_index("variant")
    print("\nBootstrap — portfolio P&L (sum across 11 markets)\n")
    print(boot_df.to_string())


if __name__ == "__main__":
    main()
