"""
mom_compare.py — Rates momentum: four signal specifications.

All signals use NS-calibrated β₁ (level factor) and are evaluated on the same
return series: Δβ₁ per market (positive = rates rose, bond fell).
Positions vol-scaled to 10 % ann. target using EWMA(hl=63) realized vol.

Variants
--------
  level_12_1   TSMOM on β₁, 12m lookback, 1m skip  [current production]
  level_12_0   TSMOM on β₁, 12m lookback, no skip
  level_multi  Equal-weight blend of 1m+3m+6m+12m z-scored TSMOM signals (no skip)
  csmom_12_1   Cross-sectional: z-score markets by 12m-1m level change at each date

Sign convention: positive signal → rates expected to keep rising → short bonds.
PnL = pos.shift(1) × Δβ₁  (positive when signal correctly bets on rates rising).

OOS: 2019-01-01 → today. Block bootstrap (block=63d, n=2000).
"""

import numpy as np
import pandas as pd

from rates.production.nelson_siegel import extract_factors_all_markets
from rates.production.ois_data import OIS_MARKETS

OOS_START  = "2019-01-01"
EWMA_HL    = 63
MIN_PERIODS = 21
VOL_FLOOR  = 1e-4
BLOCK_SIZE = 63
N_BOOT     = 2000
RNG        = np.random.default_rng(42)


# ── Signal builders ────────────────────────────────────────────────────────────

def _tsmom(level: pd.DataFrame, lookback: int, skip: int = 0) -> pd.DataFrame:
    return level.shift(1 + skip) - level.shift(lookback + 1 + skip)


def _vol_scale(signal: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
    """Divide signal by EWMA vol of rets; result has units of signal/vol."""
    vol = rets.shift(1).ewm(halflife=EWMA_HL, min_periods=MIN_PERIODS).std()
    return (signal / vol.clip(lower=VOL_FLOOR)).clip(-3.0, 3.0)


def _level_multi(level: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
    """
    Equal-weight blend of 1m, 3m, 6m, 12m TSMOM signals.
    Each lookback is independently z-scored (rolling 252d std) before blending
    so that horizons with larger raw swings don't dominate.
    """
    components = []
    for lb in [21, 63, 126, 252]:
        raw = _tsmom(level, lookback=lb, skip=0)
        std = raw.rolling(252, min_periods=63).std().replace(0, np.nan)
        components.append(raw / std)
    blended = sum(components) / len(components)
    vol = rets.shift(1).ewm(halflife=EWMA_HL, min_periods=MIN_PERIODS).std()
    return (blended / vol.clip(lower=VOL_FLOOR)).clip(-3.0, 3.0)


def _csmom(level: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional momentum: at each date, z-score markets by their
    12m-1m level change.  Long markets where rates rose most, short least.
    Zero net by construction.
    """
    raw = _tsmom(level, lookback=252, skip=21)
    xs_mean = raw.mean(axis=1)
    xs_std  = raw.std(axis=1).replace(0, np.nan)
    xs_z    = raw.sub(xs_mean, axis=0).div(xs_std, axis=0)
    vol = rets.shift(1).ewm(halflife=EWMA_HL, min_periods=MIN_PERIODS).std()
    return (xs_z / vol.clip(lower=VOL_FLOOR)).clip(-3.0, 3.0)


# ── Sharpe ─────────────────────────────────────────────────────────────────────

def _sharpe(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 60:
        return np.nan
    sd = s.std()
    if sd < 1e-12:
        return np.nan
    return float(s.mean() / sd * np.sqrt(252))


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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Extracting NS-calibrated factors...", flush=True)
    factors = extract_factors_all_markets(model="ns_calibrated")
    level = factors.xs("level", axis=1, level="factor")
    rets  = level.diff()   # Δβ₁: positive = rates rose

    print("Building signals...", flush=True)
    positions = {
        "level_12_1":  _vol_scale(_tsmom(level, 252, skip=21), rets),
        "level_12_0":  _vol_scale(_tsmom(level, 252, skip=0),  rets),
        "level_multi": _level_multi(level, rets),
        "csmom_12_1":  _csmom(level, rets),
    }

    pnls = {name: pos.shift(1).multiply(rets) for name, pos in positions.items()}

    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.width", 120)

    # ── Per-market OOS Sharpe ──────────────────────────────────────────────────
    rows = {}
    for mkt in OIS_MARKETS:
        row = {}
        for name, pnl in pnls.items():
            row[name] = _sharpe(pnl[mkt].loc[OOS_START:]) if mkt in pnl.columns else np.nan
        row["mean"] = np.nanmean(list(row.values()))
        rows[mkt] = row
    sh = pd.DataFrame(rows).T.sort_values("mean", ascending=False)
    print(f"\nPer-market OOS Sharpe ({OOS_START} → today)\n")
    print(sh.to_string())

    # ── Portfolio OOS Sharpe ───────────────────────────────────────────────────
    print("\nPortfolio Sharpe (sum across markets):\n")
    port_pnls = {}
    for name, pnl in pnls.items():
        port = pnl.sum(axis=1).loc[OOS_START:].dropna()
        port_pnls[name] = port
        print(f"  {name:<16}: {_sharpe(port):.3f}")

    # ── Pairwise PnL correlations ─────────────────────────────────────────────
    print("\nPairwise portfolio PnL correlations (OOS):\n")
    names = list(port_pnls.keys())
    port_df = pd.DataFrame({n: port_pnls[n] for n in names}).loc[OOS_START:].dropna()
    print(port_df.corr().to_string())

    # ── Block bootstrap ────────────────────────────────────────────────────────
    print(f"\nBlock bootstrap (block={BLOCK_SIZE}d, n={N_BOOT})...", flush=True)
    boot_rows = []
    for name in names:
        arr = port_pnls[name].values
        boots = _block_bootstrap(arr)
        boot_rows.append({
            "variant":  name,
            "sharpe":   round(_sharpe(pd.Series(arr)), 3),
            "p5":       round(float(np.nanpercentile(boots, 5)),  3),
            "p25":      round(float(np.nanpercentile(boots, 25)), 3),
            "p75":      round(float(np.nanpercentile(boots, 75)), 3),
            "p95":      round(float(np.nanpercentile(boots, 95)), 3),
            "pct_pos":  round(float(np.nanmean(boots > 0)) * 100, 1),
        })

    boot_df = pd.DataFrame(boot_rows).set_index("variant")
    print("\nBootstrap — portfolio PnL (sum across markets)\n")
    print(boot_df.to_string())


if __name__ == "__main__":
    main()
