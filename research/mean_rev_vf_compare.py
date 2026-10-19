"""
mean_rev_vf_compare.py — Mean-reversion signal: direct z-score vs VF variants.

Signal  : −(β₁ − rolling_252d_mean(β₁)) using NS-free model
Return  : +Δβ₁ per market (positive = rates rose)

Sign: sig < 0 when rates above mean → pos < 0 (long bonds) → profit from rates falling.

Variants:
  direct      EWMA z-score, clip ±3, daily — same approach as rv_raw_direct
  vfts        VF TS — signal/ann_vol → ortho → vol_scale, monthly rebalance
  vfxs_noclip VF XS — demean → grinold → ortho → zscore → vol_scale, monthly rebalance
               (tanh compression and normalize_leverage removed)

OOS: 2019-01-01 → today. Block bootstrap (block=63d, n=2000).
"""

import numpy as np
import pandas as pd

from rates.nelson_siegel import extract_factors_all_markets
from rates.ois_data import OIS_MARKETS
from view_formation import ViewFormationTS

OOS_START   = "2019-01-01"
MR_LOOKBACK = 252
EWMA_HL     = 63
MIN_PERIODS = 30
DESIRED_VOL = 0.10
BLOCK_SIZE  = 63
N_BOOT      = 2000
RNG         = np.random.default_rng(42)


# ── VF XS no-clip ──────────────────────────────────────────────────────────────

def _vfxs_noclip(signal: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """
    VF XS without tanh / normalize_leverage.
    Steps: demean → grinold(÷idio_vol) → ortho → zscore → vol_scale.
    """
    common = signal.dropna().index.intersection(cov.index)
    if len(common) < 3:
        return pd.Series(dtype=float)
    s  = signal.loc[common].astype(float)
    c  = cov.loc[common, common]
    ca = c.values

    vols = pd.Series(np.sqrt(np.diag(ca)), index=c.index).replace(0, np.nan)
    w_mkt = (1.0 / vols).dropna()
    w_mkt = (w_mkt / w_mkt.sum()).reindex(c.index).fillna(0.0)
    port_var = float(w_mkt.values @ ca @ w_mkt.values)
    betas    = pd.Series(ca @ w_mkt.values / max(port_var, 1e-12), index=c.index)
    total_var = pd.Series(np.diag(ca), index=c.index)
    sys_var   = (betas ** 2) * port_var
    idio_vols = np.sqrt((total_var - sys_var).clip(lower=total_var * 0.05))

    s = s - s.mean()
    s = (s / idio_vols.reindex(s.index).replace(0, np.nan)).dropna()

    b   = betas.reindex(s.index).fillna(0.0)
    b2  = float(b @ b)
    if b2 > 1e-12:
        s = s - float(s @ b) / b2 * b

    std = s.std()
    if std > 1e-12:
        s = (s - s.mean()) / std

    w_  = s.reindex(c.index).fillna(0.0).values
    pv  = float(w_ @ ca @ w_)
    s   = s * (DESIRED_VOL / np.sqrt(252) / np.sqrt(max(pv, 1e-12)))

    return s


# ── Rebalance dates ─────────────────────────────────────────────────────────────

def _month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading date of each calendar month in `idx`."""
    s = pd.Series(idx, index=idx)
    ends = s.groupby([s.index.year, s.index.month]).last()
    return pd.DatetimeIndex(ends.values)


# ── Sharpe ─────────────────────────────────────────────────────────────────────

def _sharpe(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 60:
        return np.nan
    sd = s.std()
    if sd < 1e-12:
        return np.nan
    return float(s.mean() / sd * np.sqrt(252))


# ── Block bootstrap ─────────────────────────────────────────────────────────────

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


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Extracting NS-free factors...", flush=True)
    factors = extract_factors_all_markets(model="ns_free")

    level        = factors.xs("level", axis=1, level="factor")   # date × market
    rolling_mean = level.rolling(MR_LOOKBACK, min_periods=MR_LOOKBACK // 2).mean()
    sig  = -(level - rolling_mean)   # positive → rates below mean → long rates (expect further rise)
    rets = level.diff()              # +Δβ₁: positive when rates rose

    print("Computing EWMA covariance...", flush=True)
    cov_panel = rets.ewm(halflife=EWMA_HL, min_periods=MIN_PERIODS).cov()
    # MultiIndex (date, market) rows × market columns

    common = sig.index.intersection(rets.index)
    rebal  = _month_ends(common)

    # Direct variant: EWMA z-score → clip ±3 → daily positions (rv_raw_direct style)
    Z_VOL_FLOOR = 1e-4
    Z_CAP       = 3.0
    vol_direct  = sig.shift(1).ewm(halflife=EWMA_HL, min_periods=MIN_PERIODS).std()
    z_direct    = sig / vol_direct.clip(lower=Z_VOL_FLOOR)
    pos_direct  = z_direct.clip(-Z_CAP, Z_CAP)
    pnl_direct  = pos_direct.shift(1).multiply(rets)   # daily, no monthly rebalance

    VARIANTS = ["vfts", "vfxs_noclip"]
    wt_rec: dict[str, dict] = {v: {} for v in VARIANTS}

    print("Running walk-forward backtest...", flush=True)
    cov_dates = set(cov_panel.index.get_level_values(0))
    for d in rebal:
        if d not in cov_dates:
            continue
        sig_t = sig.loc[d].dropna()
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
            wt_rec["vfts"][d] = w
        except Exception:
            pass

        try:
            w = _vfxs_noclip(sig_t, cov_t)
            wt_rec["vfxs_noclip"][d] = w
        except Exception:
            pass

    # Forward-fill weights; missing markets → 0
    wts = {}
    for v in VARIANTS:
        df = pd.DataFrame(wt_rec[v]).T.reindex(common).ffill()
        df = df.reindex(columns=OIS_MARKETS).fillna(0.0)
        wts[v] = df

    # Daily P&L = w(t-1) × ret(t)
    rets_aligned = rets.reindex(common)
    pnl = {v: wts[v].shift(1).multiply(rets_aligned) for v in VARIANTS}

    pd.set_option("display.float_format", "{:.2f}".format)
    pd.set_option("display.width", 100)

    # Per-market OOS Sharpe
    rows = {}
    for mkt in OIS_MARKETS:
        rows[mkt] = {"direct": _sharpe(pnl_direct[mkt].loc[OOS_START:]) if mkt in pnl_direct else np.nan}
        for v in VARIANTS:
            rows[mkt][v] = _sharpe(pnl[v][mkt].loc[OOS_START:]) if mkt in pnl[v].columns else np.nan

    all_cols = ["direct"] + VARIANTS
    sh = pd.DataFrame(rows).T[all_cols]
    sh["mean"] = sh.mean(axis=1)
    sh = sh.sort_values("mean", ascending=False)

    print(f"\nOOS Sharpe ({OOS_START} → today) — mean-reversion signal\n")
    print(sh.to_string())

    # Portfolio Sharpe
    print("\nPortfolio P&L Sharpe (sum across markets):\n")
    port_direct = pnl_direct.reindex(common).sum(axis=1).loc[OOS_START:].dropna()
    print(f"  direct: {_sharpe(port_direct):.2f}")
    for v in VARIANTS:
        port = pnl[v].sum(axis=1).loc[OOS_START:].dropna()
        print(f"  {v}: {_sharpe(port):.2f}")

    # Block bootstrap
    print(f"\nBlock bootstrap (block={BLOCK_SIZE}d, n={N_BOOT})...", flush=True)
    boot_rows = []
    for label, arr in [
        ("direct", port_direct.values),
        *[(v, pnl[v].sum(axis=1).loc[OOS_START:].dropna().values) for v in VARIANTS],
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
