"""
Compare CC carry: old 10Y-only anchor vs new multi-tenor average (≥1Y).

Both use the same PnL engine, DV01-neutral positions, lag=1.
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from rates.ois_data import fetch_ois_curves
from rates.ns_core import fit_ns
from rates.carry.experiment import BASELINE
from rates.carry.cross_country import (
    country_carry_at_target, build_cross_country_positions, DM_MARKETS,
)
from rates.carry.returns import cm_total_return_bp, compute_dur_panel, pnl_book

LAM, BASIS = 0.3, "ns"
MIN_TENORS = 4
TAU_CC     = 10.0
TENOR_MIN  = 1.0
MIN_MKTS   = 4


def _get_funding(c):
    col = 0.25 if 0.25 in c.columns else min(c.columns, key=lambda x: abs(x - 0.25))
    return c[col]


def _fit_all(c):
    fits = {}
    for date in c.index:
        row = c.loc[date].dropna()
        if len(row) < MIN_TENORS:
            continue
        try:
            fits[date] = fit_ns(row.index.to_numpy(float), row.to_numpy(float) * 100.0,
                                lam=LAM, basis=BASIS)
        except Exception:
            pass
    return fits


def _stats(s, carry, label):
    s = s.dropna().replace(0, np.nan).dropna()
    if len(s) < 63:
        return {}
    ann = 252
    cum = s.cumsum()
    carry_s = carry.reindex(s.index).fillna(0)

    def _dd(s, y0, y1):
        sub = s.loc[f"{y0}-01-01":f"{y1}-12-31"]
        if not len(sub):
            return float("nan")
        c = sub.cumsum()
        return float((c - c.cummax()).min())

    return {
        "label":        label,
        "sharpe":       s.mean() / s.std(ddof=1) * np.sqrt(ann),
        "max_dd_bp":    float((cum - cum.cummax()).min()),
        "carry_ann_bp": float(carry_s.mean() * ann),
        "hit_rate":     float((s > 0).mean()),
        "dd_2013":      _dd(s, 2013, 2013),
        "dd_2022":      _dd(s, 2022, 2022),
        "n_days":       len(s),
    }


def _run_cc(country_carry_df, curves_dict, funding_dict, fits_dict, label):
    pos_df = build_cross_country_positions(
        country_carry_df, curves_dict,
        tau_vol_ref=TAU_CC, universe="DM",
        vol_lookback=252, min_obs=63,
        fits_dict=fits_dict, global_beta_neutralize=False,
    )
    valid = pos_df.notna().sum(axis=1) >= MIN_MKTS
    pos_df = pos_df.where(valid)
    gross = pos_df.abs().sum(axis=1).replace(0, np.nan)
    pos_df = pos_df.div(gross, axis=0).shift(1)

    tau_arr = np.array([TAU_CC])
    dates = list(pos_df.index)
    pnl_total = pd.Series(0.0, index=pos_df.index)
    pnl_carry = pd.Series(0.0, index=pos_df.index)

    for i in range(1, len(dates)):
        date, prev = dates[i], dates[i - 1]
        pos_row = pos_df.loc[date].dropna()
        if len(pos_row) == 0:
            continue
        total_ret, carry_ret = 0.0, 0.0
        for mkt, p in pos_row.items():
            if abs(p) < 1e-12:
                continue
            if mkt not in fits_dict or prev not in fits_dict[mkt]:
                continue
            cm = curves_dict[mkt]
            if prev not in cm.index or date not in cm.index:
                continue
            fit_prev = fits_dict[mkt][prev]
            r_p = cm.loc[prev].dropna()
            r_n = cm.loc[date].dropna()
            if len(r_p) == 0 or len(r_n) == 0:
                continue
            tp = float(r_p.index.to_numpy(float)[np.argmin(np.abs(r_p.index.to_numpy(float) - TAU_CC))])
            tn = float(r_n.index.to_numpy(float)[np.argmin(np.abs(r_n.index.to_numpy(float) - TAU_CC))])
            y_p_bp = float(r_p.loc[tp]) * 100.0
            y_n_bp = float(r_n.loc[tn]) * 100.0
            rf     = float(funding_dict[mkt].loc[prev]) * 100.0
            dur    = compute_dur_panel(fit_prev, tau_arr)
            rc, ryc = cm_total_return_bp(np.array([y_p_bp]), np.array([y_n_bp]),
                                         fit_prev, tau_arr, rf, dur)
            total_ret += p * float(rc[0] + ryc[0])
            carry_ret += p * float(rc[0])
        pnl_total[date] = total_ret
        pnl_carry[date] = carry_ret

    active = pos_df.notna().any(axis=1).sum()
    print(f"  [{label}] active dates: {active}")
    return pnl_total, pnl_carry


# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading OIS data...")
curves_dict, funding_dict, fits_dict = {}, {}, {}
for mkt in DM_MARKETS:
    try:
        c = fetch_ois_curves(mkt)
        fits = _fit_all(c)
        if len(fits) >= 252:
            curves_dict[mkt]  = c
            funding_dict[mkt] = _get_funding(c)
            fits_dict[mkt]    = fits
            print(f"  {mkt:<15}: {len(c)} dates, {len(c.columns)} tenors")
    except Exception as exc:
        print(f"  {mkt:<15}: failed — {exc}")

# ── Build carry panels ────────────────────────────────────────────────────────

print("\nBuilding carry panels...")

cc_old, cc_new = {}, {}
for mkt in DM_MARKETS:
    if mkt not in fits_dict:
        continue
    rf_bp = funding_dict[mkt] * 100.0

    # OLD: 10Y-only
    ts_old = country_carry_at_target(fits_dict[mkt], curves_dict[mkt], rf_bp, TAU_CC, BASELINE)
    if len(ts_old) > 0:
        cc_old[mkt] = ts_old

    # NEW: average across all liquid tenors ≥ TENOR_MIN
    raw_tenors = np.sort(curves_dict[mkt].columns.to_numpy(float))
    liquid = raw_tenors[raw_tenors >= TENOR_MIN]
    if len(liquid) == 0:
        liquid = np.array([TAU_CC])

    tenor_series = []
    for tau in liquid:
        ts = country_carry_at_target(fits_dict[mkt], curves_dict[mkt], rf_bp, float(tau), BASELINE)
        if len(ts) > 0:
            tenor_series.append(ts)
    if tenor_series:
        cc_new[mkt] = pd.concat(tenor_series, axis=1).mean(axis=1)
        n_tenors = len(liquid)
        print(f"  {mkt:<15}: {n_tenors} tenors averaged ({liquid[0]:.1f}Y – {liquid[-1]:.1f}Y)")

df_old = pd.DataFrame(cc_old).sort_index()
df_new = pd.DataFrame(cc_new).sort_index()

# ── Carry signal correlation (before position construction) ───────────────────
print("\nCarry signal correlation (per market, old vs new):")
for mkt in sorted(set(cc_old) & set(cc_new)):
    s1, s2 = cc_old[mkt].dropna(), cc_new[mkt].dropna()
    common = s1.index.intersection(s2.index)
    if len(common) > 30:
        print(f"  {mkt:<15}: {s1.loc[common].corr(s2.loc[common]):.3f}")

# ── Run backtests ─────────────────────────────────────────────────────────────
print("\nRunning backtests...")
pnl_old, carry_old = _run_cc(df_old, curves_dict, funding_dict, fits_dict, "10Y-only")
pnl_new, carry_new = _run_cc(df_new, curves_dict, funding_dict, fits_dict, "multi-tenor")

# ── Print table ───────────────────────────────────────────────────────────────
rows = [
    _stats(pnl_old, carry_old, "CC_10Y_only (old)"),
    _stats(pnl_new, carry_new, "CC_multi_tenor (new)"),
]

W = 100
hdr = (f"{'Strategy':<30} {'Sharpe':>7} {'MaxDD':>8} "
       f"{'CarryAnn':>9} {'HitRate':>8} {'dd_2013':>8} {'dd_2022':>8} {'N':>6}")
print(f"\n{'='*W}")
print(f"  CC CARRY: 10Y-only vs multi-tenor average  (DV01-neutral, lag=1)")
print(f"{'='*W}")
print(hdr)
print("-" * W)
for r in rows:
    if not r or "sharpe" not in r:
        continue
    dd13 = f"{r['dd_2013']:>8.1f}" if not np.isnan(r.get("dd_2013", float("nan"))) else "     n/a"
    dd22 = f"{r['dd_2022']:>8.1f}" if not np.isnan(r.get("dd_2022", float("nan"))) else "     n/a"
    print(f"{r['label']:<30} {r['sharpe']:>7.3f} {r['max_dd_bp']:>8.1f} "
          f"{r['carry_ann_bp']:>9.1f} {r['hit_rate']:>8.3f} {dd13} {dd22} {r['n_days']:>6}")
print("=" * W)

# PnL correlation
s1 = pnl_old.replace(0, np.nan).dropna()
s2 = pnl_new.replace(0, np.nan).dropna()
common = s1.index.intersection(s2.index)
if len(common) > 30:
    print(f"\n  PnL correlation (old vs new): {s1.loc[common].corr(s2.loc[common]):.3f}")
