"""
Combined NS universe: 4 most-liquid power markets + 13 general commodities.

Power selection is by trailing average OI (M1-M4):
    DET > FAC > ITP > MSP  (ZUB excluded; UTD, SXD, DFP, PCB, PHB dropped).
"""

from commodities.gen_ns.config import GEN_MARKETS

# 4 most liquid power markets (by average M1-M4 OI in the full sample)
POWER_4_MARKETS: dict[str, str] = {
    "DET": "Germany baseload",
    "FAC": "France baseload",
    "ITP": "Italy baseload",
    "MSP": "Nordic mid-Sweden",
}

# Combined universe: power first, then general commodities
COMBINED_MARKETS: dict[str, str] = {**POWER_4_MARKETS, **GEN_MARKETS}

COMBINED_TRADE_MAX_RANK: int = 4   # M1-M4 only; no dynamic M5 in combined portfolio
