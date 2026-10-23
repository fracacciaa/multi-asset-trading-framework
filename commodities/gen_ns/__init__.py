"""General commodity NS residual-reversal — data layer."""

from commodities.gen_ns.config import GEN_MARKETS, GEN_MAX_RANK
from commodities.gen_ns.data   import load_gen_data

__all__ = ["GEN_MARKETS", "GEN_MAX_RANK", "load_gen_data"]
