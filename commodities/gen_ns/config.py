"""
General commodity futures universe and constants.
"""

GEN_MARKETS: dict[str, str] = {
    "CL": "WTI Crude Oil",
    "XB": "RBOB Gasoline",
    "HO": "Heating Oil",
    "C":  "Corn",
    "O":  "Oats",
    "RR": "Rough Rice",
    "W":  "Wheat",
    "CT": "Cotton",
    "LB": "Lumber",
    "HG": "Copper",
    "GC": "Gold",
    "SI": "Silver",
    "KC": "Coffee",
}

GEN_MAX_RANK: int = 4  # all gen markets are M1–M4 only
