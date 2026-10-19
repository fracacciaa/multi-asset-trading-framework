"""
Base SpecResult dataclass shared by carry and residual-signal grids.

Deflated Sharpe follows Bailey & Lopez de Prado (2014). The `n_trials` field
lets the caller pass the total number of specs tested across both grids so
that multiple-testing correction is applied consistently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def _deflated_sharpe(sr: float, n_trials: int, t: int, skew: float = 0.0, excess_kurt: float = 0.0) -> float:
    """
    Bailey & Lopez de Prado (2014) deflated Sharpe ratio.

    sr         : annualized Sharpe (sqrt-T scaled)
    n_trials   : number of strategies in the selection set
    t          : number of return observations used to estimate sr
    skew, excess_kurt: return distribution moments (0 → normal)
    """
    if n_trials <= 1 or t < 5:
        return sr
    # expected maximum SR under iid normality over n_trials trials
    euler_mascheroni = 0.5772156649
    z_max = (1 - euler_mascheroni) * math.sqrt(math.log(n_trials)) + euler_mascheroni / math.sqrt(math.log(n_trials)) if n_trials > 1 else 0.0
    sr_adj = sr * math.sqrt(1.0 - skew * sr + (excess_kurt / 4) * sr ** 2) / math.sqrt(t)
    from scipy.special import ndtr
    p_val = ndtr(sr_adj - z_max)
    return float(p_val)


@dataclass
class SpecResult:
    spec_id: str
    sharpe: float
    max_dd: float
    turnover: float

    sharpe_deflated: Optional[float] = None
    sharpe_oos: Optional[float] = None
    ir_turnover_adj: Optional[float] = None
    half_life: Optional[float] = None
    half_life_stability: Optional[float] = None
    hit_rate: Optional[float] = None
    n_curves_positive: int = 0
    corr_to_level: Optional[float] = None
    corr_to_vol: Optional[float] = None
