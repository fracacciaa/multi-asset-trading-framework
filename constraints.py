"""
Portfolio constraints.
Defines and validates all hard and soft constraints passed to the optimizer.
"""

import pandas as pd


def build_constraints(
    universe: list[str],
    gross_leverage: float = 1.0,
    net_exposure: tuple[float, float] = (-0.1, 0.1),
    position_limit: float = 0.10,
    asset_class_limits: dict[str, tuple[float, float]] | None = None,
    turnover_limit: float | None = None,
    current_weights: pd.Series | None = None,
) -> dict:
    """
    Build the constraints dictionary consumed by optimization.optimize().

    gross_leverage:      max sum of |weights|
    net_exposure:        (min, max) net portfolio exposure
    position_limit:      max |weight| per instrument
    asset_class_limits:  dict mapping asset_class -> (min_weight, max_weight)
    turnover_limit:      max one-way turnover vs current_weights (requires current_weights)
    current_weights:     current portfolio weights for turnover constraint

    Returns a dict that optimization.optimize() unpacks into solver constraints.
    """
    if turnover_limit is not None and current_weights is None:
        raise ValueError("current_weights required when turnover_limit is set")

    return {
        "gross_leverage": gross_leverage,
        "net_exposure": net_exposure,
        "position_limit": position_limit,
        "asset_class_limits": asset_class_limits or {},
        "turnover_limit": turnover_limit,
        "current_weights": current_weights,
    }

def validate_weights(weights: pd.Series, constraints: dict) -> bool:
    """
    Check that a weights vector satisfies all constraints.
    Returns True if valid, raises ValueError with a descriptive message otherwise.
    """
    raise NotImplementedError
