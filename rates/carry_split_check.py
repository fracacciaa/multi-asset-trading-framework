"""Cross-market carry Sharpe under different IS/OOS splits, same full-repricing parquet."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from rates.blend import _mkt_blend_static_ridge, _CROSS_MARKET_MIN_MKT_DAYS, _SIGNALS_DIR, SIGNAL_FILES

pnl = pd.read_parquet(_SIGNALS_DIR / SIGNAL_FILES["carry"]).sort_index()


def sr(s):
    s = s.dropna()
    return s.mean() / s.std() * np.sqrt(252) if len(s) > 30 and s.std() > 0 else float("nan")


for is_end in ["2015-10-30", "2018-12-31"]:
    is_raw = pnl.loc[:is_end]
    elig = [m for m, n in is_raw.notna().sum().items() if n >= _CROSS_MARKET_MIN_MKT_DAYS]
    blend = _mkt_blend_static_ridge(pnl, is_end)
    plain = pnl[elig].sum(axis=1, min_count=len(elig))
    print(f"\nIS_END={is_end}  ridge markets={elig}")
    for name, s in [("static ridge", blend), ("plain sum (eligible)", plain)]:
        print(f"  {name:<22} IS={sr(s.loc[:is_end]):+.2f}  OOS={sr(s.loc[pd.Timestamp(is_end) + pd.Timedelta(days=1):]):+.2f}")

print("\nStatic ridge (IS_END=2018-12-31) SR by calendar year:")
b = _mkt_blend_static_ridge(pnl, "2018-12-31")
for y, g in b.groupby(b.index.year):
    print(f"  {y}: {sr(g):+.2f}")

print("\nPer-market carry SR, 2015-11 to 2018-12 vs 2019+:")
for m in pnl.columns:
    a = sr(pnl[m].loc["2015-11-01":"2018-12-31"])
    c = sr(pnl[m].loc["2019-01-01":])
    print(f"  {m:<12} 2015-11..2018 = {a:+.2f}   2019+ = {c:+.2f}")
