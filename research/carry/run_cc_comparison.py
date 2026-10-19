"""
Three-way comparison:
  A  — within-curve bootstrap carry (Pipeline A, average over DM markets)
  B  — within-curve NS carry         (Pipeline B baseline, average over DM markets)
  CC — cross-country NS carry        (DM universe, all available markets)
  B+CC — equal-weight combined

All use same PnL engine (cm_total_return_bp), same lag=1, monthly rebalance for
within-curve and daily signal update (lag=1) for cross-country.

Cross-country position sizing: inverse-vol, zero-net, gross DV01 normalised to 1.0.
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from rates.ois_data import fetch_ois_curves, OIS_MARKETS
from rates.carry_ois import ois_carry_panel
from rates.ns_core import fit_ns
from rates.carry.experiment import BASELINE, _compute_carry_panel, _compute_pnl_series
from rates.carry.signal import xs_score, dv01_positions, _rebalance_dates, build_positions
from rates.carry.returns import cm_total_return_bp, compute_dur_panel, pnl_book
from rates.carry.cross_country import (
    country_carry_at_target, build_cross_country_positions, DM_MARKETS,
)
from rates.carry.config import CarryConfig

LAM   = 0.3
BASIS = "ns"
MIN_TENORS = 4
TAU_CC = 10.0          # 10Y anchor for cross-country carry
MIN_CC_MARKETS = 4     # require at least this many markets for a valid CS score


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_funding(curves: pd.DataFrame) -> pd.Series:
    t = 0.25
    col = t if t in curves.columns else min(curves.columns, key=lambda x: abs(x - t))
    return curves[col]


def _fit_all(curves: pd.DataFrame) -> dict:
    """Return {date: NSFit} for all dates with enough data."""
    fits = {}
    for date in curves.index:
        row = curves.loc[date].dropna()
        if len(row) < MIN_TENORS:
            continue
        tau = row.index.to_numpy(float)
        y_bp = row.to_numpy(float) * 100.0
        try:
            fits[date] = fit_ns(tau, y_bp, lam=LAM, basis=BASIS)
        except Exception:
            pass
    return fits


def _stats(total: pd.Series, carry: pd.Series, label: str) -> dict:
    s = total.dropna().replace(0, np.nan).dropna()
    if len(s) < 63:
        return {"label": label, "n_days": len(s)}
    ann = 252
    sharpe = s.mean() / s.std(ddof=1) * np.sqrt(ann)
    cum = s.cumsum()
    max_dd = float((cum - cum.cummax()).min())

    def _dd(s, y0, y1):
        sub = s.loc[f"{y0}-01-01":f"{y1}-12-31"]
        if len(sub) == 0:
            return np.nan
        c = sub.cumsum()
        return float((c - c.cummax()).min())

    carry_s = carry.reindex(s.index).fillna(0)
    return {
        "label":        label,
        "sharpe":       sharpe,
        "max_dd_bp":    max_dd,
        "hit_rate":     float((s > 0).mean()),
        "carry_ann_bp": float(carry_s.mean() * ann),
        "yldchg_ann_bp":float((s - carry_s).mean() * ann),
        "dd_2013":      _dd(s, 2013, 2013),
        "dd_2022":      _dd(s, 2022, 2022),
        "n_days":       len(s),
    }


def _print_table(rows: list[dict]) -> None:
    W = 110
    hdr = (
        f"{'Strategy':<28} {'Sharpe':>7} {'MaxDD':>8} "
        f"{'CarryAnn':>9} {'YldChgAnn':>10} {'HitRate':>8} "
        f"{'dd2013':>8} {'dd2022':>8} {'N':>6}"
    )
    print(f"\n{'='*W}")
    print(f"  CARRY SIGNAL COMPARISON  (same PnL engine, lag=1, DV01-neutral where applicable)")
    print(f"{'='*W}")
    print(hdr)
    print("-" * W)
    prev_group = None
    for r in rows:
        if not r or "sharpe" not in r:
            continue
        group = r["label"].split("_")[0]
        if prev_group and group != prev_group:
            print()
        dd13 = f"{r['dd_2013']:>8.1f}" if not np.isnan(r.get("dd_2013", float("nan"))) else "     n/a"
        dd22 = f"{r['dd_2022']:>8.1f}" if not np.isnan(r.get("dd_2022", float("nan"))) else "     n/a"
        print(
            f"{r['label']:<28} {r['sharpe']:>7.3f} {r['max_dd_bp']:>8.1f} "
            f"{r['carry_ann_bp']:>9.1f} {r['yldchg_ann_bp']:>10.1f} {r['hit_rate']:>8.3f} "
            f"{dd13} {dd22} {r['n_days']:>6}"
        )
        prev_group = group


# ── Pipeline A position builder ───────────────────────────────────────────────

def _pipeline_a_positions(
    carry_wide: pd.DataFrame,
    dur_wide: pd.DataFrame,
    rebalance: str = "monthly",
    lag: int = 1,
) -> pd.DataFrame:
    tenors = carry_wide.columns
    dates = carry_wide.index
    reb_dates = _rebalance_dates(dates, rebalance)
    pos_rows, held = {}, None
    for date in dates:
        if date in reb_dates:
            c, d = carry_wide.loc[date], dur_wide.loc[date]
            valid = c.notna() & d.notna() & (d > 1e-8)
            if valid.sum() >= 2:
                score = xs_score(c[valid], mode="z")
                held = dv01_positions(score, d[valid], gross=1.0).reindex(tenors)
        pos_rows[date] = held if held is not None else pd.Series(np.nan, index=tenors)
    return pd.DataFrame(pos_rows).T.shift(lag)


def _pipeline_a_pnl(pos_df, curves, funding, tenors, fits):
    dates = list(pos_df.index)
    pc = pd.Series(0.0, index=pos_df.index)
    pyc = pd.Series(0.0, index=pos_df.index)
    for i in range(1, len(dates)):
        date, prev = dates[i], dates[i-1]
        pos = pos_df.loc[date].reindex(tenors).to_numpy(float)
        if np.isnan(pos).any():
            continue
        if prev not in fits or prev not in curves.index or date not in curves.index:
            continue
        if date not in funding.index or prev not in funding.index:
            continue
        r_prev = curves.loc[prev].reindex(tenors)
        r_now  = curves.loc[date].reindex(tenors)
        if r_prev.isna().any() or r_now.isna().any():
            continue
        y_p = r_prev.to_numpy(float) * 100.0
        y_n = r_now.to_numpy(float) * 100.0
        rf  = float(funding.loc[prev]) * 100.0
        dur = compute_dur_panel(fits[prev], tenors)
        rc, ryc = cm_total_return_bp(y_p, y_n, fits[prev], tenors, rf, dur)
        p_c, p_yc = pnl_book(pos, rc, ryc)
        pc[date], pyc[date] = p_c, p_yc
    return pc, pyc


# ── Cross-country carry ───────────────────────────────────────────────────────

def _build_cc_carry_df(curves_dict: dict, funding_dict: dict, fits_dict: dict) -> pd.DataFrame:
    """Compute country_carry_df (date × market) once; reuse across hedge variants."""
    cc_series = {}
    for mkt in DM_MARKETS:
        if mkt not in fits_dict or mkt not in curves_dict:
            continue
        rf_bp = funding_dict[mkt] * 100.0
        carry_ts = country_carry_at_target(
            fits_dict[mkt], curves_dict[mkt], rf_bp, TAU_CC, BASELINE,
        )
        if len(carry_ts) > 0:
            cc_series[mkt] = carry_ts
            print(f"    {mkt}: {len(carry_ts)} carry observations")
    if len(cc_series) < 2:
        raise ValueError("Not enough markets for cross-country carry")
    return pd.DataFrame(cc_series).sort_index()


def run_cross_country(
    country_carry_df: pd.DataFrame,
    curves_dict: dict,
    funding_dict: dict,
    fits_dict: dict,
    *,
    dv01_neutralize: bool = True,
    global_beta_neutralize: bool = False,
    label: str = "CC",
) -> tuple[pd.Series, pd.Series]:
    """
    Build cross-country DM carry book and compute PnL.

    Returns (pnl_total, pnl_carry) in bp.
    """
    # ── Step 1: build positions ───────────────────────────────────────────────
    pos_df = build_cross_country_positions(
        country_carry_df,
        curves_dict,
        tau_vol_ref=TAU_CC,
        universe="DM",
        vol_lookback=252,
        min_obs=63,
        fits_dict=fits_dict if dv01_neutralize else None,
        global_beta_neutralize=global_beta_neutralize,
        pc1_lookback=252,
    )

    # Filter to dates with enough markets, normalise gross to 1.0, apply lag=1
    valid_rows = pos_df.notna().sum(axis=1) >= MIN_CC_MARKETS
    pos_df = pos_df.where(valid_rows)
    gross = pos_df.abs().sum(axis=1).replace(0, np.nan)
    pos_df = pos_df.div(gross, axis=0).shift(1)

    print(f"    [{label}] active dates: {pos_df.notna().any(axis=1).sum()}")

    # ── Step 3: PnL — position × 10Y return per market ───────────────────────
    dates = list(pos_df.index)
    pnl_total = pd.Series(0.0, index=pos_df.index)
    pnl_carry = pd.Series(0.0, index=pos_df.index)
    tau_arr = np.array([TAU_CC])

    for i in range(1, len(dates)):
        date, prev = dates[i], dates[i-1]
        pos_row = pos_df.loc[date].dropna()
        if len(pos_row) == 0:
            continue

        total_ret, carry_ret = 0.0, 0.0

        for mkt, p in pos_row.items():
            if abs(p) < 1e-12:
                continue
            if mkt not in fits_dict or prev not in fits_dict[mkt]:
                continue
            curves_m = curves_dict[mkt]
            if prev not in curves_m.index or date not in curves_m.index:
                continue
            if prev not in funding_dict[mkt].index:
                continue

            fit_prev = fits_dict[mkt][prev]
            row_prev = curves_m.loc[prev].dropna()
            row_now  = curves_m.loc[date].dropna()
            if len(row_prev) == 0 or len(row_now) == 0:
                continue

            # Closest tenor to TAU_CC
            tens_p = row_prev.index.to_numpy(float)
            tens_n = row_now.index.to_numpy(float)
            t_p = float(tens_p[np.argmin(np.abs(tens_p - TAU_CC))])
            t_n = float(tens_n[np.argmin(np.abs(tens_n - TAU_CC))])

            y_prev_bp = float(row_prev.loc[t_p]) * 100.0
            y_now_bp  = float(row_now.loc[t_n])  * 100.0
            rf_prev   = float(funding_dict[mkt].loc[prev]) * 100.0
            dur_prev  = compute_dur_panel(fit_prev, tau_arr)

            rc, ryc = cm_total_return_bp(
                np.array([y_prev_bp]), np.array([y_now_bp]),
                fit_prev, tau_arr, rf_prev, dur_prev,
            )
            total_ret += p * float(rc[0] + ryc[0])
            carry_ret += p * float(rc[0])

        pnl_total[date] = total_ret
        pnl_carry[date] = carry_ret

    return pnl_total, pnl_carry


# ── Main ─────────────────────────────────────────────────────────────────────

# Load all DM markets
print("Loading OIS data for all DM markets...")
curves_dict, funding_dict, fits_dict = {}, {}, {}
available_markets = []

for mkt in DM_MARKETS:
    try:
        c = fetch_ois_curves(mkt)
        f = _get_funding(c)
        fits = _fit_all(c)
        if len(fits) >= 252:
            curves_dict[mkt] = c
            funding_dict[mkt] = f
            fits_dict[mkt] = fits
            available_markets.append(mkt)
            print(f"  {mkt:<15}: {len(c)} dates, {len(c.columns)} tenors, {len(fits)} NS fits")
        else:
            print(f"  {mkt:<15}: skipped (only {len(fits)} fits)")
    except Exception as exc:
        print(f"  {mkt:<15}: failed — {exc}")

print(f"\nAvailable markets: {available_markets}")

# ── Per-market Pipeline A and B within-curve ────────────────────────────────
print("\n" + "─"*60)
print("  Within-curve carry (per market)")
print("─"*60)

pnl_a_per_mkt = {}
pnl_b_per_mkt = {}

for mkt in available_markets:
    curves  = curves_dict[mkt]
    funding = funding_dict[mkt]
    fits    = fits_dict[mkt]
    print(f"\n  {mkt}")

    # Pipeline B
    try:
        carry_df_b, dur_df_b, _ = _compute_carry_panel(curves, funding, BASELINE, LAM, BASIS)
        tenors_b = carry_df_b.columns.to_numpy(float)
        pos_b = build_positions(
            carry_df_b, dur_df_b,
            rebalance=BASELINE.rebalance, lag=BASELINE.lag,
            xs_mode=BASELINE.xs_mode, neutralize=BASELINE.neutralize,
            gross=BASELINE.gross_dv01,
        )
        pc_b, pyc_b = _compute_pnl_series(curves, funding, pos_b, fits, tenors_b)
        pnl_b_per_mkt[mkt] = (pc_b + pyc_b, pc_b)
        print(f"    B Sharpe: {_stats(pc_b+pyc_b, pc_b, '').get('sharpe', float('nan')):.3f}")
    except Exception as exc:
        print(f"    B failed: {exc}")

    # Pipeline A
    try:
        panel_a = ois_carry_panel(mkt)
        if not panel_a.empty:
            sub = panel_a.xs("3M", level="horizon")["carry_dv01_bp"].unstack("tenor")
            sub.index = pd.to_datetime(sub.index)
            sub = sub.reindex(curves.index).dropna(how="all")
            tenors_a = sub.columns.to_numpy(float)

            dur_rows = {}
            for date in curves.index:
                row_c = curves.loc[date].reindex(tenors_a).dropna()
                if len(row_c) < MIN_TENORS:
                    continue
                tau = row_c.index.to_numpy(float)
                try:
                    fit = fit_ns(tau, row_c.to_numpy(float)*100, lam=LAM, basis=BASIS)
                    dur_rows[date] = pd.Series(compute_dur_panel(fit, tau), index=tau).reindex(tenors_a)
                except Exception:
                    pass
            dur_df_a = pd.DataFrame(dur_rows).T.reindex(sub.index)

            pos_a = _pipeline_a_positions(sub, dur_df_a)
            pc_a, pyc_a = _pipeline_a_pnl(pos_a, curves, funding, tenors_a, fits)
            pnl_a_per_mkt[mkt] = (pc_a + pyc_a, pc_a)
            print(f"    A Sharpe: {_stats(pc_a+pyc_a, pc_a, '').get('sharpe', float('nan')):.3f}")
    except Exception as exc:
        print(f"    A failed: {exc}")

# ── Cross-country — three hedge variants ─────────────────────────────────────
print("\n" + "─"*60)
print("  Cross-country carry (DM universe) — building carry panel...")
print("─"*60)
country_carry_df = _build_cc_carry_df(curves_dict, funding_dict, fits_dict)

print("\n  Variant 1: no hedges")
pnl_cc_raw,  carry_cc_raw  = run_cross_country(country_carry_df, curves_dict, funding_dict, fits_dict,
                                                dv01_neutralize=False, global_beta_neutralize=False,
                                                label="CC_raw")
print("  Variant 2: DV01 only")
pnl_cc_dv01, carry_cc_dv01 = run_cross_country(country_carry_df, curves_dict, funding_dict, fits_dict,
                                                dv01_neutralize=True,  global_beta_neutralize=False,
                                                label="CC_dv01")
print("  Variant 3: DV01 + global yield beta")
pnl_cc_full, carry_cc_full = run_cross_country(country_carry_df, curves_dict, funding_dict, fits_dict,
                                                dv01_neutralize=True,  global_beta_neutralize=True,
                                                label="CC_full")

# ── Aggregate within-curve (equal-weight DM average) ─────────────────────────
def _ew_avg(pnl_dict: dict) -> tuple[pd.Series, pd.Series]:
    """Equal-weight average of (total, carry) PnL across markets."""
    all_dates = sorted(set().union(*[s[0].index for s in pnl_dict.values()]))
    idx = pd.DatetimeIndex(all_dates)
    total_sum  = pd.Series(0.0, index=idx)
    carry_sum  = pd.Series(0.0, index=idx)
    count      = pd.Series(0.0, index=idx)
    for mkt, (tot, car) in pnl_dict.items():
        tot = tot.reindex(idx, fill_value=0)
        car = car.reindex(idx, fill_value=0)
        active = (tot != 0) | (car != 0)
        total_sum += tot
        carry_sum += car
        count += active.astype(float)
    count = count.replace(0, np.nan)
    return total_sum / count, carry_sum / count

# ── Results table ─────────────────────────────────────────────────────────────
rows = []

# Per-market rows
for mkt in available_markets:
    if mkt in pnl_b_per_mkt:
        tot, car = pnl_b_per_mkt[mkt]
        rows.append(_stats(tot, car, f"B_within_{mkt}"))
    if mkt in pnl_a_per_mkt:
        tot, car = pnl_a_per_mkt[mkt]
        rows.append(_stats(tot, car, f"A_within_{mkt}"))

# DM averages
if pnl_b_per_mkt:
    tot_b_avg, car_b_avg = _ew_avg(pnl_b_per_mkt)
    rows.append(_stats(tot_b_avg, car_b_avg, "B_within_DM_avg"))
if pnl_a_per_mkt:
    tot_a_avg, car_a_avg = _ew_avg(pnl_a_per_mkt)
    rows.append(_stats(tot_a_avg, car_a_avg, "A_within_DM_avg"))

# Cross-country variants
rows.append(_stats(pnl_cc_raw,  carry_cc_raw,  "CC_no_hedge"))
rows.append(_stats(pnl_cc_dv01, carry_cc_dv01, "CC_dv01_only"))
rows.append(_stats(pnl_cc_full, carry_cc_full, "CC_dv01+globalbeta"))

# Combined B within + CC full hedge (50/50, common dates)
if pnl_b_per_mkt:
    common = tot_b_avg.index.intersection(pnl_cc_full.index)
    combined_tot = 0.5 * tot_b_avg.reindex(common) + 0.5 * pnl_cc_full.reindex(common)
    combined_car = 0.5 * car_b_avg.reindex(common) + 0.5 * carry_cc_full.reindex(common)
    rows.append(_stats(combined_tot, combined_car, "B+CC_full_hedge_50_50"))

    # Also show combined with unhedged CC for comparison
    common2 = tot_b_avg.index.intersection(pnl_cc_raw.index)
    combined_tot2 = 0.5 * tot_b_avg.reindex(common2) + 0.5 * pnl_cc_raw.reindex(common2)
    combined_car2 = 0.5 * car_b_avg.reindex(common2) + 0.5 * carry_cc_raw.reindex(common2)
    rows.append(_stats(combined_tot2, combined_car2, "B+CC_no_hedge_50_50"))

_print_table(rows)

# ── Correlation matrix ────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  PnL CORRELATIONS  (daily, common dates)")
print(f"{'='*70}")

series_for_corr = {}
if pnl_b_per_mkt:
    series_for_corr["B_within_DM"] = tot_b_avg
series_for_corr["CC_no_hedge"]   = pnl_cc_raw
series_for_corr["CC_dv01"]       = pnl_cc_dv01
series_for_corr["CC_full"]       = pnl_cc_full

names = list(series_for_corr.keys())
if len(names) >= 2:
    print(f"  {'':22}", end="")
    for n in names:
        print(f"  {n:>16}", end="")
    print()
    for n1 in names:
        print(f"  {n1:<22}", end="")
        for n2 in names:
            s1 = series_for_corr[n1].replace(0, np.nan).dropna()
            s2 = series_for_corr[n2].replace(0, np.nan).dropna()
            common = s1.index.intersection(s2.index)
            if len(common) > 30:
                c = float(s1.loc[common].corr(s2.loc[common]))
                print(f"  {c:>16.3f}", end="")
            else:
                print(f"  {'n/a':>16}", end="")
        print()
