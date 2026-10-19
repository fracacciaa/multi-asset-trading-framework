"""
slope_fly_compare.py — duration-neutral slope and butterfly signals on 11 OIS markets.

  slope_2_10   2s10s: long 10Y, short 2Y, DV01-neutral
  fly_2_5_10   2s5s10s: long belly 5Y, short wings 2Y & 10Y, equal-wing-DV01

Hedge ratios are recomputed daily from current DV01s (dynamic hedging).
Signal = NS residual of the duration-neutral spread, EWMA-vol z-scored (hl=63d).
P&L uses the *previous day's* hedge ratio (position known at close of t, marked at t+1).

OOS: 2019-01-01 → today. Block bootstrap (block=63d, n=2000) for stability.
"""

import numpy as np
import pandas as pd

from rates.production.ois_data      import fetch_ois_curves, OIS_MARKETS
from rates.production.yc_builder    import build_zero_curve
from rates.production.nelson_siegel import (
    calibrate_lambda, extract_factors, _reconstruct_fitted,
)

TRAIN_END  = "2018-12-31"
OOS_START  = "2019-01-01"
Z_EWM_HL   = 63
Z_VOL_FLOOR = 1e-4
Z_CAP       = 3.0
HYST_ENTRY  = 1.0
MIN_PERIODS = 30
BLOCK_SIZE  = 63
N_BOOT      = 2000
RNG         = np.random.default_rng(42)

SLOPE_TENORS = (2.0, 10.0)
FLY_TENORS   = (2.0, 5.0, 10.0)


# ── DV01 ───────────────────────────────────────────────────────────────────────

def _dv01_series(zc: pd.DataFrame, tau: float) -> pd.Series:
    """DV01 per unit notional at tenor tau. y in %, tau in years."""
    y = zc[tau].values
    return pd.Series(tau * np.exp(-y / 100.0 * tau), index=zc.index)


# ── Spread construction ────────────────────────────────────────────────────────

def _slope_spread(
    resid: pd.DataFrame,   # date × tenors, NS residuals in %
    zc:    pd.DataFrame,   # date × tenors, zero yields in %
) -> tuple[pd.Series, pd.Series]:
    """
    Duration-neutral 2s10s slope spread and its daily hedge ratio h(t).

      signal(t) = r(10Y,t) − h(t) × r(2Y,t)
      h(t)      = DV01(10Y,t) / DV01(2Y,t)

    P&L of holding position p(t−1):
      pnl(t) = p(t−1) × −[Δr(10Y,t) − h(t−1) × Δr(2Y,t)]
    """
    τ_s, τ_l = SLOPE_TENORS
    dv01_s = _dv01_series(zc, τ_s)
    dv01_l = _dv01_series(zc, τ_l)
    h      = dv01_l / dv01_s
    idx    = resid.index.intersection(h.index)
    signal = resid.loc[idx, τ_l] - h.loc[idx] * resid.loc[idx, τ_s]
    return signal, h.loc[idx]


def _fly_spread(
    resid: pd.DataFrame,
    zc:    pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Duration-neutral 2s5s10s butterfly (equal-wing-DV01) and daily hedge ratios.

      h₂(t)     = DV01(5Y,t) / (2 × DV01(2Y,t))
      h₁₀(t)   = DV01(5Y,t) / (2 × DV01(10Y,t))
      signal(t) = r(5Y,t) − h₂(t)×r(2Y,t) − h₁₀(t)×r(10Y,t)

    Equal-wing constraint: h₂×DV01(2Y) = h₁₀×DV01(10Y) = DV01(5Y)/2,
    so the long belly DV01 equals the sum of short wing DV01s.
    """
    τ1, τ2, τ3 = FLY_TENORS
    dv01_1 = _dv01_series(zc, τ1)
    dv01_2 = _dv01_series(zc, τ2)
    dv01_3 = _dv01_series(zc, τ3)
    h1     = dv01_2 / (2.0 * dv01_1)
    h3     = dv01_2 / (2.0 * dv01_3)
    idx    = resid.index.intersection(h1.index)
    signal = (resid.loc[idx, τ2]
              - h1.loc[idx] * resid.loc[idx, τ1]
              - h3.loc[idx] * resid.loc[idx, τ3])
    return signal, h1.loc[idx], h3.loc[idx]


# ── P&L ───────────────────────────────────────────────────────────────────────

def _pnl_slope(
    pos:   pd.Series,
    resid: pd.DataFrame,
    h:     pd.Series,
) -> pd.Series:
    """pos(t−1) × −[Δr(10Y,t) − h(t−1)×Δr(2Y,t)]"""
    τ_s, τ_l = SLOPE_TENORS
    dr_l = resid[τ_l].diff()
    dr_s = resid[τ_s].diff()
    idx  = pos.index.intersection(dr_l.index).intersection(h.index)
    spread_chg = dr_l.loc[idx] - h.shift(1).loc[idx] * dr_s.loc[idx]
    return pos.shift(1).loc[idx] * (-spread_chg)


def _pnl_fly(
    pos:   pd.Series,
    resid: pd.DataFrame,
    h1:    pd.Series,
    h3:    pd.Series,
) -> pd.Series:
    """pos(t−1) × −[Δr(5Y,t) − h₂(t−1)×Δr(2Y,t) − h₁₀(t−1)×Δr(10Y,t)]"""
    τ1, τ2, τ3 = FLY_TENORS
    dr1 = resid[τ1].diff()
    dr2 = resid[τ2].diff()
    dr3 = resid[τ3].diff()
    idx = pos.index.intersection(dr1.index).intersection(h1.index)
    spread_chg = (dr2.loc[idx]
                  - h1.shift(1).loc[idx] * dr1.loc[idx]
                  - h3.shift(1).loc[idx] * dr3.loc[idx])
    return pos.shift(1).loc[idx] * (-spread_chg)


# ── Signal helpers ─────────────────────────────────────────────────────────────

def _zscored(s: pd.Series) -> pd.Series:
    vol = s.shift(1).ewm(halflife=Z_EWM_HL, min_periods=MIN_PERIODS).std()
    return s / vol.clip(lower=Z_VOL_FLOOR)


def _pos_raw(z: pd.Series) -> pd.Series:
    return z.clip(-Z_CAP, Z_CAP)


def _pos_hyst(z: pd.Series) -> pd.Series:
    out = np.zeros(len(z)); cur = 0.0; zv = z.to_numpy()
    for i in range(len(zv)):
        if np.isnan(zv[i]): out[i] = cur; continue
        if cur == 0.0:
            if   zv[i] >  HYST_ENTRY: cur =  1.0
            elif zv[i] < -HYST_ENTRY: cur = -1.0
        else:
            if cur ==  1.0 and zv[i] <= 0.0: cur = 0.0
            elif cur == -1.0 and zv[i] >= 0.0: cur = 0.0
        out[i] = cur
    return pd.Series(out, index=z.index)


def _metrics(pnl: pd.Series) -> dict:
    oos = pnl.loc[OOS_START:].dropna()
    if len(oos) < 60:
        return {"sharpe": np.nan, "turnover": np.nan}
    sharpe = float(oos.mean() / oos.std() * np.sqrt(252))
    return {"sharpe": sharpe}


# ── Block bootstrap ────────────────────────────────────────────────────────────

def _block_bootstrap(pnl_arr: np.ndarray) -> np.ndarray:
    T = len(pnl_arr)
    n_blocks = int(np.ceil(T / BLOCK_SIZE))
    boots = np.empty(N_BOOT)
    for b in range(N_BOOT):
        starts = RNG.integers(0, T - BLOCK_SIZE + 1, size=n_blocks)
        boot = np.concatenate([pnl_arr[s:s+BLOCK_SIZE] for s in starts])[:T]
        boots[b] = boot.mean() / boot.std() * np.sqrt(252)
    return boots


# ── Per-market results ─────────────────────────────────────────────────────────

def _market_results(market: str, lambdas: dict) -> dict:
    raw = fetch_ois_curves(market)
    zc  = build_zero_curve(raw)

    ns_lam  = lambdas["ns"][market]
    fac_cal = extract_factors(zc, model="ns_calibrated", lam=ns_lam)
    fit_cal = _reconstruct_fitted(zc, fac_cal, "ns_calibrated", lam=ns_lam)
    common  = zc.index.intersection(fit_cal.index)
    resid   = zc.loc[common].subtract(fit_cal.loc[common])   # date × all tenors

    results = {}

    # slope 2s10s
    sig_slope, h_slope = _slope_spread(resid, zc.loc[common])
    z_slope = _zscored(sig_slope)
    for label, pos_fn in [("slope_raw", _pos_raw), ("slope_hyst", _pos_hyst)]:
        pos = pos_fn(z_slope)
        pnl = _pnl_slope(pos, resid.loc[common], h_slope)
        to  = pos.diff().abs().mean() * 252
        m   = _metrics(pnl)
        m["turnover"] = round(float(to), 2)
        results[label] = m

    # fly 2s5s10s
    sig_fly, h1_fly, h3_fly = _fly_spread(resid, zc.loc[common])
    z_fly = _zscored(sig_fly)
    for label, pos_fn in [("fly_raw", _pos_raw), ("fly_hyst", _pos_hyst)]:
        pos = pos_fn(z_fly)
        pnl = _pnl_fly(pos, resid.loc[common], h1_fly, h3_fly)
        to  = pos.diff().abs().mean() * 252
        m   = _metrics(pnl)
        m["turnover"] = round(float(to), 2)
        results[label] = m

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Calibrating NS lambdas on train (≤2018)...", flush=True)
    lambdas = calibrate_lambda(end=TRAIN_END)

    print("Running markets...", flush=True)
    sharpe_rows: dict = {}
    to_rows:     dict = {}
    oos_pnl = {k: [] for k in ["slope_raw", "slope_hyst", "fly_raw", "fly_hyst"]}

    for mkt in OIS_MARKETS:
        print(f"  {mkt}...", flush=True)

        raw = fetch_ois_curves(mkt)
        zc  = build_zero_curve(raw)
        ns_lam  = lambdas["ns"][mkt]
        fac_cal = extract_factors(zc, model="ns_calibrated", lam=ns_lam)
        fit_cal = _reconstruct_fitted(zc, fac_cal, "ns_calibrated", lam=ns_lam)
        common  = zc.index.intersection(fit_cal.index)
        resid   = zc.loc[common].subtract(fit_cal.loc[common])

        sig_slope, h_slope = _slope_spread(resid, zc.loc[common])
        z_slope = _zscored(sig_slope)

        sig_fly, h1_fly, h3_fly = _fly_spread(resid, zc.loc[common])
        z_fly = _zscored(sig_fly)

        res = {}
        for key, z, pos_fn, pnl_fn, pnl_args in [
            ("slope_raw",  z_slope, _pos_raw,  _pnl_slope, (resid.loc[common], h_slope)),
            ("slope_hyst", z_slope, _pos_hyst, _pnl_slope, (resid.loc[common], h_slope)),
            ("fly_raw",    z_fly,   _pos_raw,  _pnl_fly,   (resid.loc[common], h1_fly, h3_fly)),
            ("fly_hyst",   z_fly,   _pos_hyst, _pnl_fly,   (resid.loc[common], h1_fly, h3_fly)),
        ]:
            pos = pos_fn(z)
            pnl = pnl_fn(pos, *pnl_args)
            to  = float(pos.diff().abs().mean() * 252)
            oos = pnl.loc[OOS_START:].dropna()
            sh  = float(oos.mean() / oos.std() * np.sqrt(252)) if len(oos) >= 60 else np.nan
            res[key] = {"sharpe": sh, "turnover": to}
            oos_pnl[key].append(oos)

        sharpe_rows[mkt] = {k: v["sharpe"]   for k, v in res.items()}
        to_rows[mkt]     = {k: v["turnover"] for k, v in res.items()}

    cols = ["slope_raw", "slope_hyst", "fly_raw", "fly_hyst"]
    sh_df = pd.DataFrame(sharpe_rows).T[cols]
    to_df = pd.DataFrame(to_rows).T[cols]
    sh_df["mean"] = sh_df.mean(axis=1)
    sh_df = sh_df.sort_values("mean", ascending=False)
    to_df = to_df.loc[sh_df.index]

    pd.set_option("display.float_format", "{:.2f}".format)
    pd.set_option("display.width", 120)

    print(f"\nOOS Sharpe ({OOS_START}→today) — duration-neutral, EWMA vol hl={Z_EWM_HL}d\n")
    print(sh_df.to_string())
    print(f"\nAnnualised turnover (|Δpos| × 252)\n")
    print(to_df.to_string())

    # ── Block bootstrap ────────────────────────────────────────────────────────
    print(f"\nBlock bootstrap (block={BLOCK_SIZE}d, n={N_BOOT})...", flush=True)
    boot_rows = []
    for key in cols:
        combined = pd.concat(oos_pnl[key], axis=1).sum(axis=1).dropna()
        arr      = combined.values
        actual   = arr.mean() / arr.std() * np.sqrt(252)
        boots    = _block_bootstrap(arr)
        boot_rows.append({
            "signal":  key,
            "sharpe":  round(actual, 2),
            "p5":      round(np.percentile(boots,  5), 2),
            "p25":     round(np.percentile(boots, 25), 2),
            "p75":     round(np.percentile(boots, 75), 2),
            "p95":     round(np.percentile(boots, 95), 2),
            "pct_pos": round((boots > 0).mean() * 100, 1),
        })

    print("\nBootstrap — portfolio P&L (sum across 11 markets)\n")
    boot_df = pd.DataFrame(boot_rows).set_index("signal")
    print(boot_df.to_string())


if __name__ == "__main__":
    main()
