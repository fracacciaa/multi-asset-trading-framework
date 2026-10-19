"""Print carry signal summary tables (mean, std, Sharpe) for OIS and govt bond markets."""
import time
import numpy as np
import pandas as pd
from rates.production.carry.carry_ois import ois_carry_all_markets
from rates.production.carry.carry_bonds import bond_carry_all_markets


def _summary(df: pd.DataFrame, col: str) -> pd.DataFrame:
    s = df[col].groupby(level=["market", "tenor", "horizon"])
    mean_ = s.mean()
    std_  = s.std()
    sharpe = mean_ / std_.clip(lower=1e-12) * np.sqrt(252)
    pct_pos = s.apply(lambda x: (x > 0).mean() * 100)
    result = pd.DataFrame({
        "Mean":   mean_.round(1),
        "Std":    std_.round(1),
        "Sharpe": sharpe.round(3),
        "Pct+":   pct_pos.round(1),
        "Min":    s.min().round(1),
        "Max":    s.max().round(1),
        "N":      s.count(),
    })
    return result


def _print_table(title: str, stats: pd.DataFrame) -> None:
    W = 80
    print(f"\n{'='*W}")
    print(f"  {title}")
    print(f"{'='*W}")
    hdr = f"{'Market':<12} {'Tenor':>6} {'Horiz':>6} {'Mean':>8} {'Std':>8} {'Sharpe':>8} {'Pct+':>7} {'Min':>8} {'Max':>8} {'N':>7}"
    print(hdr)
    print("-" * W)
    prev_mkt = None
    for (mkt, tenor, horiz), row in stats.iterrows():
        if prev_mkt and mkt != prev_mkt:
            print()
        print(
            f"{mkt:<12} {tenor:>6.1f} {horiz:>6} "
            f"{row.Mean:>8.1f} {row.Std:>8.1f} {row.Sharpe:>8.3f} "
            f"{row['Pct+']:>7.1f} {row.Min:>8.1f} {row.Max:>8.1f} {int(row.N):>7}"
        )
        prev_mkt = mkt


# ── Load ──────────────────────────────────────────────────────────────────────

print("Loading OIS carry for all markets...")
t0 = time.time()
ois = ois_carry_all_markets()
print(f"  done in {time.time()-t0:.0f}s — {len(ois):,} rows")

print("Loading bond carry for all markets...")
t0 = time.time()
bonds = bond_carry_all_markets()
print(f"  done in {time.time()-t0:.0f}s — {len(bonds):,} rows")

# ── Print ─────────────────────────────────────────────────────────────────────

_print_table(
    "OIS STATIC CARRY (bp, receiver=fixed; unchanged-curve scenario)",
    _summary(ois, "static_carry_bp"),
)

_print_table(
    "OIS CARRY per DV01 (bp per bp-DV01; risk-normalised)",
    _summary(ois, "carry_dv01_bp"),
)

_print_table(
    "GOVT BOND TOTAL CARRY §6.2 (bp; income + roll-down)",
    _summary(bonds, "total_carry_bp"),
)

_print_table(
    "GOVT BOND CARRY per D_mod (bp per unit duration)",
    _summary(bonds, "carry_dmod_bp"),
)
