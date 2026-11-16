"""
Classic-NS signal: per-country backtest diagnostics.
Order: factor loadings → concentration → per-tenor → carry/MTM → drawdowns → Sharpe.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from rates.production.ois_data        import fetch_ois_curves
from rates.production.ns_classic.bootstrap import bootstrap_curves, _ACT360
from rates.production.ns_classic.ns_fit    import select_lambda, fit_ns_panel
from rates.production.ns_classic.signal    import (
    classic_ns_signal, build_dv01_positions, compute_pnl, vol_scale_k,
)
from rates.production.ns_classic.pnl       import classic_ns_pnl_repriced

TARGET_VOL       = 100.0        # bp / year
BURN_IN          = 252
LAM_TRAIN_END    = "2015-12-31"
MARKETS          = ["US", "EUR", "UK", "Japan", "Canada", "Australia",
                    "New Zealand", "Sweden"]

# ── rough FOMC dates 2010-2026 ────────────────────────────────────────────────
_FOMC = {pd.Timestamp(d) for d in [
    "2010-01-27","2010-03-16","2010-04-28","2010-06-23","2010-08-10",
    "2010-09-21","2010-11-03","2010-12-14",
    "2011-01-26","2011-03-15","2011-04-27","2011-06-22","2011-08-09",
    "2011-09-21","2011-11-02","2011-12-13",
    "2012-01-25","2012-03-13","2012-04-25","2012-06-20","2012-08-01",
    "2012-09-13","2012-10-24","2012-12-12",
    "2013-01-30","2013-03-20","2013-05-01","2013-06-19","2013-07-31",
    "2013-09-18","2013-10-30","2013-12-18",
    "2014-01-29","2014-03-19","2014-04-30","2014-06-18","2014-07-30",
    "2014-09-17","2014-10-29","2014-12-17",
    "2015-01-28","2015-03-18","2015-04-29","2015-06-17","2015-07-29",
    "2015-09-17","2015-10-28","2015-12-16",
    "2016-01-27","2016-03-16","2016-04-27","2016-06-15","2016-07-27",
    "2016-09-21","2016-11-02","2016-12-14",
    "2017-02-01","2017-03-15","2017-05-03","2017-06-14","2017-07-26",
    "2017-09-20","2017-11-01","2017-12-13",
    "2018-01-31","2018-03-21","2018-05-02","2018-06-13","2018-08-01",
    "2018-09-26","2018-11-08","2018-12-19",
    "2019-01-30","2019-03-20","2019-05-01","2019-06-19","2019-07-31",
    "2019-09-18","2019-10-30","2019-12-11",
    "2020-01-29","2020-03-03","2020-03-15","2020-04-29","2020-06-10",
    "2020-07-29","2020-09-16","2020-11-05","2020-12-16",
    "2021-01-27","2021-03-17","2021-04-28","2021-06-16","2021-07-28",
    "2021-09-22","2021-11-03","2021-12-15",
    "2022-01-26","2022-03-16","2022-05-04","2022-06-15","2022-07-27",
    "2022-09-21","2022-11-02","2022-12-14",
    "2023-02-01","2023-03-22","2023-05-03","2023-06-14","2023-07-26",
    "2023-09-20","2023-11-01","2023-12-13",
    "2024-01-31","2024-03-20","2024-05-01","2024-06-12","2024-07-31",
    "2024-09-18","2024-11-07","2024-12-18",
    "2025-01-29","2025-03-19","2025-05-07","2025-06-18","2025-07-30",
    "2025-09-17",
]}


# ── helpers ───────────────────────────────────────────────────────────────────

def _sr(pnl):
    v = pnl.dropna()
    return float(np.sqrt(252) * v.mean() / v.std()) if len(v) > 30 else float("nan")


def _factor_loadings(pnl, betas):
    db = betas.diff() * 100.0        # % → bp/day
    idx = pnl.dropna().index.intersection(db.dropna().index)
    if len(idx) < 100:
        return None
    y  = pnl.loc[idx].to_numpy(float)
    Xr = db.loc[idx].to_numpy(float)
    Xc = np.column_stack([np.ones(len(y)), Xr])
    b, _, _, _ = np.linalg.lstsq(Xc, y, rcond=None)
    res = y - Xc @ b
    n, k = Xc.shape
    s2  = np.dot(res, res) / (n - k)
    se  = np.sqrt(np.maximum(np.diag(s2 * np.linalg.pinv(Xc.T @ Xc)), 0))
    t   = b / np.where(se > 0, se, np.nan)
    ss_tot = np.dot(y - y.mean(), y - y.mean())
    r2  = 1 - np.dot(res, res) / ss_tot if ss_tot > 0 else float("nan")
    return pd.DataFrame(
        {"coef": b[1:], "t-stat": t[1:]},
        index=["Δβ0 (level)", "Δβ1 (slope)", "Δβ2 (curve)"]
    ), r2


def _concentration(pnl):
    v = pnl.dropna()
    if len(v) < 10:
        return float("nan"), []
    top10 = v.abs().nlargest(10)
    conc  = top10.sum() / v.abs().sum()
    return float(conc), top10


def _per_tenor(z_scores, zeros):
    dv01, _ = build_dv01_positions(z_scores, zeros, k=1.0)
    tenors  = dv01.columns
    zc  = zeros.reindex(columns=tenors)
    dz  = zc.diff().clip(-0.25, 0.25)
    pos = dv01.shift(1)
    out = {}
    for col in tenors:
        out[float(col)] = float((pos[col] * dz[col]).dropna().sum() * 100.0)
    return out


def _carry_mtm(pnl_repriced, dv01_scaled, zeros):
    pnl_mtm   = compute_pnl(dv01_scaled, zeros, include_carry=False)
    pnl_full_a = compute_pnl(dv01_scaled, zeros, include_carry=True)
    carry     = pnl_full_a - pnl_mtm
    idx       = pnl_repriced.dropna().index
    s_total   = float(pnl_repriced.loc[idx].sum())
    s_mtm     = float(pnl_mtm.reindex(idx).sum())
    s_carry   = float(carry.reindex(idx).sum())
    s_other   = s_total - s_mtm - s_carry    # roll-down + convexity
    return s_mtm, s_carry, s_other, s_total


def _cal_flag(d):
    flags = []
    if d in _FOMC:
        flags.append("FOMC")
    if d.year == 2020 and d.month == 3:
        flags.append("COVID-Mar20")
    if d.month in (3, 6, 9, 12) and d.day >= 25:
        flags.append("QE")
    return ", ".join(flags)


def _drawdowns(pnl, n=10):
    v   = pnl.dropna()
    cum = v.cumsum()
    dd  = cum - cum.cummax()
    return float(dd.min()), dd.idxmin(), v.nsmallest(n)


# ── main runner ───────────────────────────────────────────────────────────────

def run(market):
    sep = "=" * 68
    print(f"\n{sep}\n  {market}\n{sep}")

    try:
        par  = fetch_ois_curves(market)
    except Exception as e:
        print(f"  DATA ERROR: {e}"); return

    print(f"  {par.index[0].date()} → {par.index[-1].date()} | "
          f"{len(par.columns)} tenors  "
          f"[{par.columns[0]:.3f}Y – {par.columns[-1]:.0f}Y]")

    try:
        zeros = bootstrap_curves(par)
    except Exception as e:
        print(f"  BOOTSTRAP ERROR: {e}"); return

    # λ selection on training data
    try:
        lr   = select_lambda(zeros, end=LAM_TRAIN_END)
        lam  = lr["best_lambda"]
        tbl  = lr["table"].set_index("lambda")
        print(f"\n  λ grid (train ≤ {LAM_TRAIN_END}):  "
              + "  ".join(f"λ={l:.1f} hl={tbl.loc[l,'mean_half_life_days']:.0f}d"
                           for l in tbl.index.tolist()))
        print(f"  → selected λ = {lam}  "
              f"(curvature peak ≈ {1.793*lam:.1f}Y)")
    except Exception as e:
        lam = 2.5
        print(f"  λ selection failed ({e}), fallback λ=2.5")

    # NS fit + z-scores
    try:
        z_scores, betas, resids = classic_ns_signal(zeros, lam, burn_in=BURN_IN)
    except Exception as e:
        print(f"  SIGNAL ERROR: {e}"); return

    # Build vol-scaled DV01 positions (for carry/MTM split)
    k_arr     = vol_scale_k(z_scores, resids, TARGET_VOL)
    dv01_base, _ = build_dv01_positions(z_scores, zeros, k=1.0)
    dv01_scaled  = dv01_base.mul(k_arr, axis=0)

    # Full repricing P&L
    try:
        pnl, _, _ = classic_ns_pnl_repriced(
            par, lam, target_ann_vol=TARGET_VOL, burn_in=BURN_IN
        )
    except Exception as e:
        print(f"  REPRICING ERROR: {e}"); return

    valid = pnl.dropna()
    if len(valid) < 100:
        print("  Insufficient post-burn-in observations."); return

    # ── 1. Factor loadings ────────────────────────────────────────────────────
    print("\n  ── 1. FACTOR LOADINGS ─────────────────────────────────────────")
    fl = _factor_loadings(pnl, betas)
    if fl is not None:
        df_fl, r2 = fl
        print(f"  R² = {r2:.4f}")
        for name, row in df_fl.iterrows():
            flag = "  ← LEAK" if abs(row["t-stat"]) > 2.0 else ""
            print(f"    {name:<18}  coef={row['coef']:+.4f}  t={row['t-stat']:+.2f}{flag}")
        if df_fl["t-stat"].abs().max() < 2.0:
            print("  All |t| < 2 → factor-neutral ✓")
    else:
        print("  insufficient data")

    # ── 2. P&L concentration ─────────────────────────────────────────────────
    print("\n  ── 2. P&L CONCENTRATION ───────────────────────────────────────")
    conc, top10 = _concentration(pnl)
    verdict = "event-driven ⚠" if conc > 0.30 else "reversion ✓"
    print(f"  Top-10 days / |total|: {conc*100:.1f}%  ({verdict})")
    print(f"  Top 10 largest |P&L| days:")
    for d, v in pnl.dropna().abs().nlargest(10).items():
        sign = "+" if pnl.loc[d] > 0 else "-"
        print(f"    {d.date()}  {sign}{v:6.1f} bp")

    # ── 3. Per-tenor attribution ──────────────────────────────────────────────
    print("\n  ── 3. PER-TENOR ATTRIBUTION ───────────────────────────────────")
    tp = _per_tenor(z_scores, zeros)
    total_tp = sum(tp.values()) or 1.0
    for t in sorted(tp):
        v   = tp[t]
        pct = 100 * v / abs(total_tp)
        bar = "█" * max(0, int(abs(pct) / 2))
        print(f"    {t:5.1f}Y : {v:+8.0f} bp  ({pct:+6.1f}%)  {bar}")

    # ── 4. Carry / MTM split ──────────────────────────────────────────────────
    print("\n  ── 4. CARRY vs CURVE-MOVE DECOMPOSITION ───────────────────────")
    s_mtm, s_carry, s_other, s_total = _carry_mtm(pnl, dv01_scaled, zeros)
    if abs(s_total) > 0:
        print(f"    Curve-move (MTM)   : {s_mtm:+8.0f} bp  ({100*s_mtm/abs(s_total):+5.1f}%)")
        print(f"    Carry              : {s_carry:+8.0f} bp  ({100*s_carry/abs(s_total):+5.1f}%)")
        print(f"    Roll-down + other  : {s_other:+8.0f} bp  ({100*s_other/abs(s_total):+5.1f}%)")
        print(f"    ─────────────────────────────────────────")
        print(f"    Total (repricing)  : {s_total:+8.0f} bp")
        if abs(s_carry) > 0.5 * abs(s_total):
            print("    ⚠  Carry > 50% of total — predominantly carry strategy")
    else:
        print("  zero total P&L")

    # ── 5. Drawdowns vs calendar ──────────────────────────────────────────────
    print("\n  ── 5. DRAWDOWNS vs CALENDAR ───────────────────────────────────")
    max_dd, trough, worst10 = _drawdowns(pnl)
    print(f"  Max drawdown: {max_dd:+.1f} bp  (trough {trough.date()})")
    print(f"  Worst 10 single days:")
    fomc_hits = 0
    for d, v in worst10.items():
        flag = _cal_flag(d)
        if "FOMC" in flag:
            fomc_hits += 1
        print(f"    {d.date()}  {v:+7.1f} bp   {flag}")
    if fomc_hits >= 3:
        print(f"  ⚠  {fomc_hits}/10 worst days on FOMC — front-end mis-specification likely")

    # ── Sharpe (last) ─────────────────────────────────────────────────────────
    print("\n  ── SHARPE ─────────────────────────────────────────────────────")
    sr    = _sr(pnl)
    n_yr  = len(valid) / 252
    a_ret = valid.mean() * 252
    a_vol = valid.std() * np.sqrt(252)
    print(f"  Annualised Sharpe : {sr:.2f}")
    print(f"  Return / Vol      : {a_ret:.1f} / {a_vol:.1f} bp/year")
    print(f"  History           : {n_yr:.1f} years  ({len(valid)} trading days)")


if __name__ == "__main__":
    for m in MARKETS:
        run(m)
    print("\n\nAll markets done.")
