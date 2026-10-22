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


def build_signal_b_pnl(
    panels_by_market: dict[str, dict[str, pd.DataFrame]],
    h: int = 1,
    train_frac: float = 0.40,
) -> dict[str, pd.DataFrame]:
    """
    Per-market monthly PnL for each Signal B variant.

    For B1 / B3 (per-tenor signals):
      pnl = sum_tenor( sign(signal(t-1)) × rx_m(t) )

    For B2 / B4 (scalar signals applied uniformly across tenors):
      pnl = sign(signal(t-1)) × sum_tenor( rx_m(t) )

    Returns
    -------
    {
      "b1_own":      date × market DataFrame, monthly PnL in bp
      "b2_factor":   date × market DataFrame, monthly PnL in bp
      "b3_yieldchg": date × market DataFrame, monthly PnL in bp
      "b4_pc1":      date × market DataFrame, monthly PnL in bp
    }
    Annualise Sharpe with √12.
    """
    sigs = build_signal_b(panels_by_market, h=h, train_frac=train_frac)

    b1: dict[str, pd.Series] = {}
    b2: dict[str, pd.Series] = {}
    b3: dict[str, pd.Series] = {}
    b4: dict[str, pd.Series] = {}

    for market, d in sigs.items():
        rx = panels_by_market[market]["rx_m"]

        # ── B1: own-maturity sign momentum ───────────────────────────────────
        own_sign = d["own_sign"]
        cols = own_sign.columns.intersection(rx.columns)
        idx  = own_sign.index.intersection(rx.index)
        if len(idx) >= 12 and len(cols) > 0:
            b1[market] = (
                own_sign.loc[idx, cols].shift(1) * rx.loc[idx, cols]
            ).sum(axis=1)

        # ── B3: yield-change sign momentum ───────────────────────────────────
        yldchg = d["yieldchg"]     # already sign-flipped: positive → expect +return
        cols3 = yldchg.columns.intersection(rx.columns)
        idx3  = yldchg.index.intersection(rx.index)
        if len(idx3) >= 12 and len(cols3) > 0:
            b3[market] = (
                np.sign(yldchg.loc[idx3, cols3].shift(1)) * rx.loc[idx3, cols3]
            ).sum(axis=1)

        # ── B2: factor momentum (scalar → uniform across tenors) ──────────────
        factor = d["factor"]
        idx2 = factor.index.intersection(rx.index)
        cols2 = rx.columns.tolist()
        if len(idx2) >= 12 and len(cols2) > 0:
            b2[market] = (
                np.sign(factor.loc[idx2].shift(1)) * rx.loc[idx2, cols2].sum(axis=1)
            )

        # ── B4: PC1 scalar momentum (uniform across tenors) ───────────────────
        pc_sig = d["pc1"]
        if not pc_sig.empty and "pc1" in pc_sig.columns:
            idx4 = pc_sig.index.intersection(rx.index)
            cols4 = rx.columns.tolist()
            if len(idx4) >= 12 and len(cols4) > 0:
                b4[market] = (
                    np.sign(pc_sig.loc[idx4, "pc1"].shift(1))
                    * rx.loc[idx4, cols4].sum(axis=1)
                )

    def _to_df(d: dict) -> pd.DataFrame:
        if not d:
            return pd.DataFrame()
        df = pd.concat(d.values(), axis=1, keys=d.keys())
        df.index.name = "date"
        return df

    return {
        "b1_own":      _to_df(b1),
        "b2_factor":   _to_df(b2),
        "b3_yieldchg": _to_df(b3),
        "b4_pc1":      _to_df(b4),
    }
