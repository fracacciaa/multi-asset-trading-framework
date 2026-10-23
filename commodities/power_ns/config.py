"""
Global constants for the Power NS residual-reversal strategy.

All parameters match POWER_NS_PROD_FINAL.ipynb exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

# ── Futures month codes ────────────────────────────────────────────────────────

MONTH_CODES: dict[str, int] = {
    "F": 1, "G": 2, "H": 3, "J": 4,
    "K": 5, "M": 6, "N": 7, "Q": 8,
    "U": 9, "V": 10, "X": 11, "Z": 12,
}

# ── Trading calendar ───────────────────────────────────────────────────────────

TRADING_DAYS = 252

# ── Curve depth ────────────────────────────────────────────────────────────────

CURVE_MAX_RANK  = 8   # use up to M8 for NS fitting
TRADE_MAX_RANK  = 4   # execute views only in M1–M4
MAX_RANK        = TRADE_MAX_RANK  # backward-compatible alias

DYNAMIC_MIN_DEPTH = 4
DYNAMIC_MAX_DEPTH = 5

MAX_AVAILABLE_RANK = 8
MIN_OBS_FOR_RANK   = 250   # rank eligible only after this many obs

# ── Execution ──────────────────────────────────────────────────────────────────

LAG_BASE          = 1     # signal at t-1 earns t-1 → t return
ROLL_OUT_BUFFER   = 2     # business days before month-end to flatten butterfly
COVERAGE_FLOOR    = 1.00  # require fully measurable held gross

# ── Lambda ─────────────────────────────────────────────────────────────────────

LAMBDA_GRID: list[float] = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.10, 1.40, 1.80, 2.40]
LAMBDA_FIT = None   # None = depth-adaptive; set float to freeze fit lambda

# ── IS/OOS split (frozen from 2026-08-07 vintage, 70% of trading calendar) ───

TRAIN_END  = pd.Timestamp("2021-06-04")
TEST_START = pd.Timestamp("2021-06-07")
TRAIN_FRACTION = 0.70   # kept for cross-check only

# ── Risk model ─────────────────────────────────────────────────────────────────

COV_HALFLIFE    = 60
COV_SHRINKAGE   = 0.40
COV_MIN_OBS     = 250
COV_MIN_PAIR_OBS = 60
COV_EXTRA_LAG   = 1
COV_RIDGE       = 1e-8

# ── Diagnostics ────────────────────────────────────────────────────────────────

ADF_LAGS       = 5
DF_CRIT        = {"10%": -2.57, "5%": -2.86, "1%": -3.43}
DEFAULT_NW_LAGS = 21
NW_LAG_GRID    = (5, 10, 21, 42)

# ── Power universe ─────────────────────────────────────────────────────────────

POWER_MARKETS_ALL: dict[str, str] = {
    "ZUB": "ZUB",
    "ITP": "ITP",
    "DET": "DET",
    "FAC": "FAC",
    "DFP": "DFP",
    "SXD": "SXD",
    "UTD": "UTD",
    "MSP": "MSP",
    "PHB": "PHB",
    "PCB": "PCB",
}

EXCLUDED_POWER_MARKETS: dict[str, str] = {}

POWER_MARKETS: dict[str, str] = {
    root: label
    for root, label in POWER_MARKETS_ALL.items()
    if root not in EXCLUDED_POWER_MARKETS
}

# ── NS curvature peak (computed once at import) ───────────────────────────────

def _ns_curvature_at_x(x: float) -> float:
    if x <= 0:
        return np.inf
    slope = (1.0 - np.exp(-x)) / x
    curv  = slope - np.exp(-x)
    return -float(curv)


_opt = minimize_scalar(_ns_curvature_at_x, bounds=(1e-6, 20.0), method="bounded")
NS_CURVATURE_PEAK_X: float = float(_opt.x)
