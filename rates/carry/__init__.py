"""
Rates curve carry signal.

Signal construction follows:
  Martens, Beekhuizen, Duyvesteyn, Zomerdijk (2019) FAJ 75(4)
  Koijen, Moskowitz, Pedersen, Vrugt (2018) JFE 127(2)

Build order (carry_signal_spec.md §12):
  1. data.py      – funding loader + validation
  2. measures.py  – carry definitions
  3. returns.py   – constant-maturity total-return decomposition
  4. signal.py    – duration adjustment, cross-section, DV01 positions
  5. cross_country.py – country-level cross-sectional book
  6. experiment.py    – grid runner and CarrySpecResult
"""

from .config import CarryConfig
from .measures import (
    approx_dv01,
    carry_pickup,
    carry_rolldown_dur,
    carry_reprice,
    carry_forward,
    carry_rolldown_raw,
    CARRY_MEASURES,
)
from .returns import cm_total_return_bp
from .signal import carry_per_duration, xs_score, dv01_positions, build_positions

__all__ = [
    "CarryConfig",
    "approx_dv01",
    "carry_pickup",
    "carry_rolldown_dur",
    "carry_reprice",
    "carry_forward",
    "carry_rolldown_raw",
    "CARRY_MEASURES",
    "cm_total_return_bp",
    "carry_per_duration",
    "xs_score",
    "dv01_positions",
    "build_positions",
]
