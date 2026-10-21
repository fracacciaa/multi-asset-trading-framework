"""
OOS Sharpe comparison: Signal A (BC LOO fly) vs Signal B (NS null-space projection).

Signal A  commodities/mean_reversion/signal.py
  BC 4-factor | fit=8 | LOO residuals | manual F1/F4/F8 fly | calibrated-once λ

Signal B  commodities/mean_reversion/ns_proj.py
  NS 3-factor | fit=8 | IS residuals | null-space projection on F1–F4 | λ_trade=1.7937/M̄

Both use EWMA z-score (hl=63) + hysteresis (|z|>1 entry, z=0 exit), lag-1.
OOS period: 2019-01-01 onwards.
"""

import warnings
import numpy as np
import pandas as pd

from commodities.mean_reversion.research.signal import compute_pnl as pnl_a
from commodities.mean_reversion.ns_proj        import compute_pnl as pnl_b

OOS_START = "2019-01-01"


def _sharpe(s: pd.Series) -> float:
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 60:
        return np.nan
    std = s.std()
    return float(s.mean() / std * np.sqrt(252)) if std > 1e-12 else np.nan


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pd.set_option("display.float_format", "{:.3f}".format)

    print("Signal A: BC LOO fly...")
    a = pnl_a(start=OOS_START)

    print("Signal B: NS null-space projection...")
    b = pnl_b(start=OOS_START)

    # ── Per-commodity Sharpe ──────────────────────────────────────────────────
    coms = sorted(set(a.columns) | set(b.columns))
    rows = []
    for c in coms:
        rows.append({
            "commodity": c,
            "A_BC_fly":  _sharpe(a[c]) if c in a.columns else np.nan,
            "B_NS_proj": _sharpe(b[c]) if c in b.columns else np.nan,
        })
    tbl = pd.DataFrame(rows).set_index("commodity")
    tbl["diff (B-A)"] = tbl["B_NS_proj"] - tbl["A_BC_fly"]

    print("\n=== Per-commodity OOS Sharpe (2019–present) ===")
    print(tbl.to_string())

    print("\n=== Mean across commodities ===")
    print(tbl.mean(numeric_only=True).to_string())

    # ── Equal-weight portfolio Sharpe ─────────────────────────────────────────
    a_port = a.mean(axis=1)
    b_port = b.mean(axis=1)
    print(f"\n=== Equal-weight portfolio Sharpe ===")
    print(f"A (BC fly):   {_sharpe(a_port):.3f}")
    print(f"B (NS proj):  {_sharpe(b_port):.3f}")

    # ── Correlation ───────────────────────────────────────────────────────────
    idx = a_port.dropna().index.intersection(b_port.dropna().index)
    if len(idx) > 60:
        corr = a_port.loc[idx].corr(b_port.loc[idx])
        print(f"\nPortfolio return correlation A vs B: {corr:.3f}")

    # ── Combined portfolio ────────────────────────────────────────────────────
    combo = (a_port + b_port) / 2
    print(f"50/50 combined portfolio Sharpe:     {_sharpe(combo.loc[OOS_START:]):.3f}")
