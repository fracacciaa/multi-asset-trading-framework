"""
Per-tenor annualised Sharpe ratio for the NS classic signal — each country/tenor pair.

Position = -z_score (k=1, unscaled) lagged one day.
Daily PnL (bp) = pos(t-1) × Δz(t) × 100, with 25bp hard cap on Δz.
Sharpe = sqrt(252) × mean(pnl) / std(pnl).

Also records: annualised return (bp/yr), annualised vol (bp/yr), n_obs, hit_rate.
Lambda and burn-in consistent with ns_classic_run.py.

Saves: rates/signals/ns_sharpe_per_tenor.parquet
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from rates.production.ois_data import fetch_ois_curves
from rates.production.ns_classic.bootstrap import bootstrap_curves
from rates.production.ns_classic.ns_fit import select_lambda, fit_ns_panel
from rates.production.ns_classic.signal import compute_zscores, build_dv01_positions

MARKETS       = ["US", "EUR", "UK", "Japan", "Canada", "Australia",
                 "New Zealand", "Sweden", "Switzerland", "India", "China"]
LAM_TRAIN_END = "2015-12-31"
BURN_IN       = 252
DZ_CAP        = 0.25   # 25bp hard cap on Δz (% units)
OUT_PATH      = "rates/signals/ns_sharpe_per_tenor.parquet"


def _sharpe(pnl: pd.Series) -> float:
    v = pnl.dropna()
    if len(v) < 30 or v.std() == 0:
        return float("nan")
    return float(np.sqrt(252) * v.mean() / v.std())


def _per_tenor_pnl(z_scores: pd.DataFrame, zeros: pd.DataFrame) -> pd.DataFrame:
    """Return date × tenor DataFrame of daily PnL in bp (k=1, MTM only)."""
    dv01, _ = build_dv01_positions(z_scores, zeros, k=1.0)
    tenors   = dv01.columns
    zc  = zeros.reindex(columns=tenors)
    dz  = zc.diff().clip(-DZ_CAP, DZ_CAP)
    pos = dv01.shift(1)
    return (pos * dz * 100.0).rename(columns=float)


rows = []

for market in MARKETS:
    print(f"\n{'='*64}")
    print(f"  {market}")
    print(f"{'='*64}")

    try:
        par = fetch_ois_curves(market)
    except Exception as e:
        print(f"  DATA ERROR: {e}"); continue

    try:
        zeros = bootstrap_curves(par)
    except Exception as e:
        print(f"  BOOTSTRAP ERROR: {e}"); continue

    try:
        lr  = select_lambda(zeros, end=LAM_TRAIN_END)
        lam = lr["best_lambda"]
        print(f"  λ = {lam}")
    except Exception as e:
        lam = 2.5
        print(f"  λ fallback 2.5 ({e})")

    try:
        _, resids = fit_ns_panel(zeros, lam)
        z_scores  = compute_zscores(resids, burn_in=BURN_IN)
    except Exception as e:
        print(f"  FIT/SIGNAL ERROR: {e}"); continue

    pnl_df = _per_tenor_pnl(z_scores, zeros)

    print(f"\n  {'Tenor':>6}  {'N':>5}  {'Sharpe':>7}  "
          f"{'Ret bp/yr':>10}  {'Vol bp/yr':>10}  {'Hit%':>6}")
    print(f"  {'-'*58}")

    for tenor in sorted(pnl_df.columns):
        s = pnl_df[tenor].dropna()
        n = len(s)
        if n < 30:
            sr, ann_ret, ann_vol, hit = float("nan"), float("nan"), float("nan"), float("nan")
        else:
            sr      = _sharpe(s)
            ann_ret = float(s.mean() * 252)
            ann_vol = float(s.std() * np.sqrt(252))
            hit     = float((s > 0).mean() * 100)

        print(f"  {tenor:>6.2f}Y  {n:>5}  {sr:>7.3f}  "
              f"{ann_ret:>10.1f}  {ann_vol:>10.1f}  {hit:>6.1f}")

        rows.append(dict(
            country=market,
            tenor_y=float(tenor),
            lambda_used=lam,
            sharpe=round(sr, 4) if not np.isnan(sr) else float("nan"),
            ann_ret_bp=round(ann_ret, 2) if not np.isnan(ann_ret) else float("nan"),
            ann_vol_bp=round(ann_vol, 2) if not np.isnan(ann_vol) else float("nan"),
            hit_rate_pct=round(hit, 2) if not np.isnan(hit) else float("nan"),
            n_obs=int(n),
        ))

results = pd.DataFrame(rows)
results = results.sort_values(["country", "tenor_y"]).reset_index(drop=True)
results.to_parquet(OUT_PATH, index=False)
print(f"\n\nSaved → {OUT_PATH}  ({len(results)} rows)")

# ── Summary: top Sharpe pairs across all markets ───────────────────────────────
print("\n── Top 20 tenor/country pairs by Sharpe ────────────────────────────────")
top = (results.dropna(subset=["sharpe"])
              .sort_values("sharpe", ascending=False)
              .head(20)
              [["country", "tenor_y", "sharpe", "ann_ret_bp", "ann_vol_bp", "n_obs"]])
print(top.to_string(index=False))

print("\n── Bottom 10 (worst Sharpe, tradable tenors) ────────────────────────────")
bot = (results.dropna(subset=["sharpe"])
              .query("n_obs >= 500")
              .sort_values("sharpe")
              .head(10)
              [["country", "tenor_y", "sharpe", "ann_ret_bp", "ann_vol_bp", "n_obs"]])
print(bot.to_string(index=False))

print("\n── Mean Sharpe by country (tradable tenors, n≥500) ─────────────────────")
mkt = (results.query("n_obs >= 500")
              .groupby("country")["sharpe"]
              .agg(["mean", "count", "min", "max"])
              .round(3)
              .sort_values("mean", ascending=False))
mkt.columns = ["mean_sharpe", "n_tenors", "min_sharpe", "max_sharpe"]
print(mkt.to_string())
