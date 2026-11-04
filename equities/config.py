"""
Equities backtesting configuration.

Canonical IS/OOS split and data paths used across signals, blending, and research.
"""

# ── IS / OOS split ──────────────────────────────────────────────────────────────
IS_END  = "2020-12-31"   # last date of in-sample window (inclusive)
OOS_START = "2021-01-01" # first date of out-of-sample window

# ── Managed-return cache ────────────────────────────────────────────────────────
# Pre-computed F = build_managed_returns(build_vf_positions(...), returns)
# Regenerate with: python -m equities.cache_managed_returns
MANAGED_RETURNS_PATH = "equities/cache/managed_returns.parquet"

# ── build_vf_positions defaults ─────────────────────────────────────────────────
VF_WARMUP       = 504   # trading days before first rebalance (~2 years)
VF_REBAL_FREQ   = 21    # rebalance every N days (~monthly)
VF_COV_LOOKBACK = 504
VF_VOL_HALFLIFE = 21
VF_COR_HALFLIFE = 126
