"""Compare continuous-z daily MR vs hysteresis (enter |z|>1, exit at 0) MR: gross SR and TC."""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from rates.production.ns_classic.bootstrap import bootstrap_curves
from rates.production.ns_classic.ns_fit import select_lambda
from rates.production.ns_classic.pnl import classic_ns_pnl_repriced
from rates.production.ois_data import fetch_ois_curves

IS_END    = "2018-12-31"
OOS_START = "2019-01-01"
MARKETS   = ["US", "EUR", "UK", "Japan", "Canada", "Australia"]
H_BEST = {"US": 0.25, "EUR": 0.50, "UK": 0.50, "Japan": 0.50, "Canada": 0.50, "Australia": 1.00}
H_BASE = {"US": 0.75, "EUR": 1.00, "UK": 1.50, "Japan": 2.00, "Canada": 1.25, "Australia": 2.00}

VARIANTS = {"continuous z": None, "hysteresis 1.0": 1.0}


def sr(v):
    v = v.dropna()
    return v.mean() / v.std() * np.sqrt(252) if len(v) > 30 and v.std() > 0 else float("nan")


rows = []
for mkt in MARKETS:
    par = fetch_ois_curves(mkt)
    lam = select_lambda(bootstrap_curves(par), end="2015-12-31")["best_lambda"]
    for label, entry in VARIANTS.items():
        pnl, _, _, held = classic_ns_pnl_repriced(
            par, lam, target_ann_vol=100.0, burn_in=252,
            rebalance_freq="daily", entry_z=entry, _return_held_dv01=True,
        )
        chg = held.diff().abs().sum(axis=1)
        for period, sl in [("IS", slice(None, IS_END)), ("OOS", slice(OOS_START, None))]:
            p = pnl.loc[sl].dropna()
            if len(p) < 30:
                continue
            c = chg.reindex(p.index).fillna(0.0)
            vol = p.std() * np.sqrt(252)
            rows.append(dict(
                mkt=mkt, variant=label, period=period,
                gross_sr=sr(p),
                gross_mean=p.mean() * 252,
                vol=vol,
                tc_best=(c * H_BEST[mkt]).mean() * 252,
                tc_base=(c * H_BASE[mkt]).mean() * 252,
                dv01_per_day=c.mean(),
            ))
        print(f"done {mkt} / {label}", flush=True)

df = pd.DataFrame(rows)
df["net_sr_best"] = (df.gross_mean - df.tc_best) / df.vol
df["net_sr_base"] = (df.gross_mean - df.tc_base) / df.vol
pd.set_option("display.width", 200)
print(df.round(2).to_string(index=False))

print("\n=== Equal-vol average across markets ===")
agg = (df.groupby(["variant", "period"])[["gross_sr", "net_sr_best", "net_sr_base", "dv01_per_day"]]
         .mean().round(2))
print(agg)
