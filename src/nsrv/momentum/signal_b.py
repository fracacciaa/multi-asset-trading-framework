"""
Signal B: Sihvonen (2022) yield curve momentum.

Four variants (spec §3):

  B1 own      own-maturity momentum: past return predicts next return
  B2 factor   common factor (equal-weight average across maturities)
              predicts each maturity independently
  B3 yieldchg yield-change component predicts itself (85-93% of total
              predictability per Table 4)
  B4 pc1      first PC change of zero-curve levels predicts all maturities
              (negative loading: PC1 up = yields up = bond returns down)

Key differences from Signal A (spec §8):
  - Lookback h = 1 month (not 12)
  - No volatility scaling (explicitly stated in the paper)
  - Uses log excess returns on zero-coupon bonds (exact bootstrap), not
    constant-maturity approximation

Maturities: N = {12, 24, …, 120} months = {1, 2, …, 10} years.
These are the rows of rx_m, c_m, yc_m panels from returns.py.

PC1 loadings are fitted on the first train_frac of dates and frozen,
to avoid any lookahead through the covariance matrix.

Spec: ois_momentum_signals_spec.md §3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


# ── B1: own-maturity momentum ─────────────────────────────────────────────────

def sihvonen_own(rx_m_bp: pd.DataFrame, h: int = 1) -> dict[str, pd.DataFrame]:
    """
    B1: predict rx^n_{t+1} from its h-month lagged value.

    Returns both forms:
      linear  rx^n_{t-h,t}      — the regression predictor
      sign    sign(rx^n_{t-h,t}) — the subsample-cut predictor
    """
    lag = rx_m_bp.rolling(h).sum()
    return {"linear": lag, "sign": np.sign(lag)}


# ── B2: factor momentum ───────────────────────────────────────────────────────

def sihvonen_factor(
    rx_m_bp: pd.DataFrame,
    maturities: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0),
) -> pd.Series:
    """
    B2: rxbar_t = equal-weight average of rx^n_t across all N maturities.

    This is a single series per curve; it serves as the predictor for every
    maturity (with maturity-specific loadings estimated downstream).
    """
    cols = [m for m in maturities if m in rx_m_bp.columns]
    return rx_m_bp[cols].mean(axis=1)


# ── B3: yield-change momentum ─────────────────────────────────────────────────

def sihvonen_yieldchange(yc_m_bp: pd.DataFrame, h: int = 1) -> pd.DataFrame:
    """
    B3: signal for the yield-change component yc^n_{t+1}.

    Sign-flipped so that a positive signal is a positive expected RETURN:
    rx = c - yc, so a large positive past yc forecasts a positive future yc,
    which forecasts a *lower* return contribution.  Flip to align with the
    return-signal convention.

    Spec eq: signal = -yc^n_{t-h,t}
    """
    return -yc_m_bp.rolling(h).sum()


# ── B4: PC1 change momentum ───────────────────────────────────────────────────

def pc1_change(
    zero_m_pct: pd.DataFrame,
    n_pc: int = 3,
    train_frac: float = 0.40,
) -> pd.DataFrame:
    """
    B4: Sihvonen eq (11) — first difference of PC1 score from zero-curve levels.

    PCA is fitted on zero yields in bp (z × 100) on the FIRST train_frac of
    dates, then loadings are frozen for the rest of the sample.  This avoids
    lookahead through the covariance matrix.

    Sign: -pc1_diff so that a positive signal = expect positive return.
    (PC1 up = yields up = bond prices down = negative return.)

    Parameters
    ----------
    zero_m_pct  : monthly zero-curve panel, annual cc %, date × tenor
    n_pc        : number of PCs to extract (default 3; only PC1 is used)
    train_frac  : fraction of dates for PCA fit (default 0.40)

    Returns
    -------
    DataFrame with columns pc1 … pc{n_pc}, index = dates.
    Use the 'pc1' column.  Expected beta is negative (Table 6).
    """
    z_bp = zero_m_pct * 100.0    # % → bp
    X    = z_bp.dropna()

    if len(X) < 12:
        return pd.DataFrame()

    n_train = max(12, int(len(X) * train_frac))
    X_train = X.iloc[:n_train]
    Xc_train = X_train - X_train.mean()

    # SVD on training block; loadings (Vt rows) are frozen
    _, _, Vt = np.linalg.svd(Xc_train.values, full_matrices=False)
    loadings = Vt[:n_pc]                           # (n_pc, n_tenors)

    # Apply frozen loadings to full sample (demeaned with training mean)
    Xc_full = X - X_train.mean()
    scores   = Xc_full.values @ loadings.T         # (T, n_pc)

    pc = pd.DataFrame(
        scores,
        index=X.index,
        columns=[f"pc{i + 1}" for i in range(n_pc)],
    )
    # Signal: negative first difference (PC1 up = yields up = return down)
    return -pc.diff()


# ── Panel builder ─────────────────────────────────────────────────────────────

def build_signal_b(
    panels_by_market: dict[str, dict[str, pd.DataFrame]],
    h: int = 1,
    train_frac: float = 0.40,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Build all four Signal B variants for every market.

    Parameters
    ----------
    panels_by_market : output of build_all_return_panels()
    h                : lookback in months (Sihvonen h=1 is the headline result)
    train_frac       : PC1 training fraction

    Returns
    -------
    {market: {"own_linear": df, "own_sign": df, "factor": series,
              "yieldchg": df, "pc1": df}}

    All DataFrames have month-end dates as index.
    """
    result: dict = {}

    for market, panels in panels_by_market.items():
        rx_m  = panels["rx_m"]
        yc_m  = panels["yc_m"]
        z_m   = panels["z_m"]

        if rx_m.empty:
            continue

        own_d    = sihvonen_own(rx_m, h=h)
        factor_s = sihvonen_factor(rx_m)
        yc_sig   = sihvonen_yieldchange(yc_m, h=h)
        pc_sig   = pc1_change(z_m, train_frac=train_frac)

        result[market] = {
            "own_linear": own_d["linear"],
            "own_sign":   own_d["sign"],
            "factor":     factor_s,
            "yieldchg":   yc_sig,
            "pc1":        pc_sig,
        }

    return result
