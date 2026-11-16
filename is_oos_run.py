"""
IS / OOS metrics for the classic-NS full-vector signal.

IS  : burn-in end → 2015-12-31  (λ trained on this window)
OOS : 2016-01-01 → end of data
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from rates.production.ois_data        import fetch_ois_curves
from rates.production.ns_classic.bootstrap import bootstrap_curves
from rates.production.ns_classic.ns_fit    import select_lambda
from rates.production.ns_classic.pnl       import classic_ns_pnl_repriced

IS_END   = pd.Timestamp("2015-10-30")   # gives Sweden exactly 5yr IS
OOS_START = pd.Timestamp("2015-10-31")
TARGET_VOL = 100.0
BURN_IN    = 252
LAM_TRAIN_END = "2015-12-31"

MARKETS = ["US", "EUR", "UK", "Japan", "Canada", "Australia",
           "New Zealand", "Sweden"]


def _metrics(pnl: pd.Series) -> dict:
    v = pnl.dropna()
    if len(v) < 30:
        return dict(sr=float("nan"), ret=float("nan"), vol=float("nan"),
                    dd=float("nan"), n_yr=0.0)
    cum = v.cumsum()
    dd  = float((cum - cum.cummax()).min())
    return dict(
        sr   = float(np.sqrt(252) * v.mean() / v.std()),
        ret  = float(v.mean() * 252),
        vol  = float(v.std() * np.sqrt(252)),
        dd   = dd,
        n_yr = len(v) / 252,
    )


def _fmt(m: dict) -> str:
    if np.isnan(m["sr"]):
        return "insufficient data"
    return (f"SR={m['sr']:+.2f}  ret={m['ret']:+.0f}  vol={m['vol']:.0f} bp/yr  "
            f"dd={m['dd']:+.0f} bp  ({m['n_yr']:.1f}yr)")


def run(market):
    sep = "=" * 68
    print(f"\n{sep}\n  {market}\n{sep}")

    try:
        par = fetch_ois_curves(market)
    except Exception as e:
        print(f"  DATA ERROR: {e}"); return

    try:
        zeros = bootstrap_curves(par)
        lr    = select_lambda(zeros, end=LAM_TRAIN_END)
        lam   = lr["best_lambda"]
    except Exception as e:
        lam = 2.5
        print(f"  λ fallback 2.5 ({e})")

    try:
        pnl, _, _ = classic_ns_pnl_repriced(
            par, lam, target_ann_vol=TARGET_VOL, burn_in=BURN_IN
        )
    except Exception as e:
        print(f"  PNL ERROR: {e}"); return

    pnl_is  = pnl.loc[pnl.index <= IS_END].dropna()
    pnl_oos = pnl.loc[pnl.index >= OOS_START].dropna()

    m_is   = _metrics(pnl_is)
    m_oos  = _metrics(pnl_oos)
    m_full = _metrics(pnl)

    print(f"  λ={lam}  |  "
          f"data: {par.index[0].date()} → {par.index[-1].date()}")
    print(f"\n  IS  : {_fmt(m_is)}")
    print(f"  OOS : {_fmt(m_oos)}")
    print(f"  Full: {_fmt(m_full)}")

    if np.isfinite(m_is["sr"]) and np.isfinite(m_oos["sr"]):
        delta = m_oos["sr"] - m_is["sr"]
        flag  = "  ⚠ large degradation" if delta < -0.5 else ""
        print(f"\n  ΔSR (OOS − IS): {delta:+.2f}{flag}")


if __name__ == "__main__":
    rows = []
    for m in MARKETS:
        run(m)
    print("\nDone.")
