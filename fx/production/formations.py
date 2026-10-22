"""
View formations: convert a continuous score into long/short portfolio weights.

xs_weights  — cross-sectional rank: long top portfolio, short bottom (Menkhoff style)
ts_weights  — time-series sign: each currency independently +1/N or -1/N
ts_vol_weights — time-series sign scaled by (target / sigma), equal gross exposure

These are agnostic to signal family and frequency; pass any date × pair score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fx.production.signal_a import _assign_portfolios, menkhoff_weights


def xs_weights(
    score: pd.DataFrame,
    n_port: int = 6,
    min_size: int = 2,
) -> pd.DataFrame:
    """
    Cross-sectional rank formation.

    Rank currencies each period into n_port buckets; go long the top bucket
    and short the bottom bucket, equal-weighted within each leg.
    Identical logic to Signal A applied to any continuous score.
    """
    port = score.apply(_assign_portfolios, axis=1, n_port=n_port)
    return menkhoff_weights(port, n_port=n_port, min_size=min_size)


def ts_weights(score: pd.DataFrame) -> pd.DataFrame:
    """
    Time-series sign formation.

    Each currency is independently long (+) or short (−) based on the sign of
    its score. Gross exposure normalised to 2 (1-dollar long + 1-dollar short)
    at each date; equal weight within each leg.
    """
    sgn   = np.sign(score)
    denom = sgn.abs().sum(axis=1).replace(0, np.nan)
    return sgn.div(denom, axis=0)


def ts_vol_weights(
    score: pd.DataFrame,
    sigma: pd.DataFrame,
    target: float = 0.40,
) -> pd.DataFrame:
    """
    Vol-scaled time-series formation.

    Same sign logic as ts_weights but each currency position is sized by
    (target / sigma) rather than equal weight — MOP equation (5) style.
    """
    sgn = np.sign(score)
    return sgn * (target / sigma.replace(0, np.nan))
