"""
Signal B: Zhang (2022, JFE) factor decomposition and factor momentum.

Decomposes each currency's monthly excess return into:
  systematic  = b_carry * carry_factor + b_dollar * dollar_factor
  idiosyncratic = rx - systematic

Betas estimated each month from that month's daily data (time-varying).

Factor momentum: condition carry and dollar factor positions on the sign of
the past-h-month average factor return. Expected result: strong following
positive factor months, near-zero following negative (asymmetric).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from fx.production.signal_a import build_signal_a, build_signal_a_grid
from fx.production.signal_c import build_signal_c, build_signal_c_grid


# ── Factors ───────────────────────────────────────────────────────────────────

def dollar_factor(rx: pd.DataFrame) -> pd.Series:
    """Zhang equation (2): equal-weighted average excess return."""
    return rx.mean(axis=1)


def carry_factor(
    rx: pd.DataFrame,
    fd: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Zhang equations (3)-(4): rank-weighted, dollar-neutral carry factor.

    Returns (factor_return, weights) where weights are the carry positions.
    Assert: weights.sum(axis=1) == 0 and weights.abs().sum(axis=1) == 2.
    """
    r  = fd.rank(axis=1)
    w  = r.sub(r.mean(axis=1), axis=0)
    gross = w.abs().sum(axis=1).replace(0, np.nan)
    w  = w.div(gross / 2.0, axis=0)   # scale to 1 long, 1 short
    ret = (w * rx).sum(axis=1)
    return ret, w


# ── Monthly betas from daily data ─────────────────────────────────────────────

def _add_const(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X])


def monthly_betas(
    rx_d: pd.DataFrame,
    carry_d: pd.Series,
    dollar_d: pd.Series,
    min_obs: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Estimate per-currency carry and dollar betas each month from daily returns.

    Returns (b_carry, b_dollar), both month-end × pair DataFrames.
    Uses OLS without statsmodels: np.linalg.lstsq on [1, carry, dollar].
    """
    X_full = pd.DataFrame({"carry": carry_d, "dollar": dollar_d}).dropna()
    out_c: dict = {}
    out_d: dict = {}

    for month, grp in rx_d.groupby(pd.Grouper(freq="ME")):
        idx = grp.index
        xm  = X_full.reindex(idx).dropna()
        if len(xm) < min_obs:
            out_c[month] = {p: np.nan for p in rx_d.columns}
            out_d[month] = {p: np.nan for p in rx_d.columns}
            continue

        bc, bd = {}, {}
        for pair in rx_d.columns:
            y = grp[pair].reindex(xm.index).dropna()
            if len(y) < min_obs:
                bc[pair] = bd[pair] = np.nan
                continue
            A_y = _add_const(xm.loc[y.index].to_numpy())
            beta, *_ = np.linalg.lstsq(A_y, y.to_numpy(), rcond=None)
            bc[pair] = beta[1]   # carry beta (intercept is beta[0])
            bd[pair] = beta[2]   # dollar beta
        out_c[month] = bc
        out_d[month] = bd

    return pd.DataFrame(out_c).T, pd.DataFrame(out_d).T


# ── Decomposition ─────────────────────────────────────────────────────────────

def decompose(
    rx_m: pd.DataFrame,
    b_carry: pd.DataFrame,
    b_dollar: pd.DataFrame,
    carry_m: pd.Series,
    dollar_m: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Zhang equation (5): rx = b_carry*carry + b_dollar*dollar + (a + e).

    Returns (systematic, idiosyncratic), both month-end × pair DataFrames.
    The month-t betas and month-t factor returns are both known at end of t,
    so systematic_t may be used to form a position for t+1.
    """
    sys = b_carry.mul(carry_m, axis=0) + b_dollar.mul(dollar_m, axis=0)
    idio = rx_m - sys
    return sys, idio


# ── Factor momentum ───────────────────────────────────────────────────────────

def factor_momentum_sign(factor_m: pd.Series, h: int = 3) -> pd.Series:
    """
    +1 when the past-h-month average factor return is positive, else -1.

    Already lagged by shift(1) so the signal is known before the holding month.
    """
    past = factor_m.rolling(h).mean().shift(1)
    return np.sign(past)


def factor_momentum_weights(
    carry_m: pd.Series,
    dollar_m: pd.Series,
    carry_w: pd.DataFrame,
    n_pairs: int,
    hs: tuple = (1, 3, 12),
) -> dict[int, dict[str, pd.DataFrame]]:
    """
    Zhang §4.4: condition carry/dollar factor positions on sign of past returns.

    Returns dict h → {'carry': date×pair, 'dollar': date×pair}.
    """
    N = n_pairs
    results: dict[int, dict[str, pd.DataFrame]] = {}
    for h in hs:
        sign_carry  = factor_momentum_sign(carry_m,  h)
        sign_dollar = factor_momentum_sign(dollar_m, h)
        # carry position: apply sign to existing carry weights
        w_carry = carry_w.mul(sign_carry, axis=0)
        # dollar position: equal 1/N across all pairs, direction from sign
        base = pd.DataFrame(1.0 / N, index=carry_w.index, columns=carry_w.columns)
        w_dollar = base.mul(sign_dollar.reindex(carry_w.index), axis=0).fillna(0.0)
        results[h] = {"carry": w_carry, "dollar": w_dollar}
    return results


# ── Systematic / idiosyncratic momentum ──────────────────────────────────────

def build_sys_idio_momentum(
    systematic: pd.DataFrame,
    idiosyncratic: pd.DataFrame,
    fs: tuple = (1, 3, 6, 9, 12),
    hs: tuple = (1, 3, 6, 9, 12),
) -> dict:
    """
    Run Signal A (cross-sectional) and Signal C (TSMOM equal-weight)
    on systematic and idiosyncratic return components.

    Zhang's diagnostic: systematic spreads should be ~45/48/30bp at 1/3/12m
    and significant; idiosyncratic spreads near zero and insignificant.
    """
    results: dict = {}
    for tag, rx in [("sys", systematic), ("idio", idiosyncratic)]:
        results[f"{tag}_csmom"] = build_signal_a_grid(rx, fs=fs, hs=hs)
        results[f"{tag}_tsmom"] = {}
        for k in fs:
            sgn  = np.sign(rx.rolling(k).sum())
            denom = sgn.abs().sum(axis=1).replace(0, np.nan)
            results[f"{tag}_tsmom"][k] = sgn.div(denom, axis=0)
    return results


# ── Full signal builder ───────────────────────────────────────────────────────

def build_signal_b(
    rx_m: pd.DataFrame,
    rx_d: pd.DataFrame,
    fd_m: pd.DataFrame,
    fd_d: Optional[pd.DataFrame] = None,
    min_obs: int = 15,
    beta_hs: tuple = (1, 3, 12),
) -> dict:
    """
    Build all Signal B components.

    Returns
    -------
    dict with keys:
      'carry_m'           — carry factor monthly returns
      'dollar_m'          — dollar factor monthly returns
      'carry_d'           — carry factor daily returns
      'dollar_d'          — dollar factor daily returns
      'carry_weights'     — rank-weighted carry weights (month-end × pair)
      'b_carry'           — carry betas (month-end × pair)
      'b_dollar'          — dollar betas (month-end × pair)
      'systematic'        — systematic return (month-end × pair)
      'idiosyncratic'     — idiosyncratic return (month-end × pair)
      'factor_mom_{h}'    — dict {'carry': w, 'dollar': w} for each h in beta_hs
      'sys_idio_momentum' — nested dict from build_sys_idio_momentum
      'conditional_means' — dict with E[rx_m | sign(factor)] tables (diagnostic)
    """
    # ── Daily factors ──────────────────────────────────────────────────────────
    carry_d, carry_w_d = carry_factor(rx_d, fd_m.reindex(rx_d.index, method="ffill"))
    dollar_d = dollar_factor(rx_d)

    # ── Monthly factors ────────────────────────────────────────────────────────
    carry_m, carry_w_m = carry_factor(rx_m, fd_m)
    dollar_m = dollar_factor(rx_m)

    # ── Monthly betas from daily data ──────────────────────────────────────────
    b_carry, b_dollar = monthly_betas(rx_d, carry_d, dollar_d, min_obs=min_obs)

    # ── Decomposition ──────────────────────────────────────────────────────────
    rx_m_aligned = rx_m.reindex(b_carry.index)
    cm_aligned   = carry_m.reindex(b_carry.index)
    dm_aligned   = dollar_m.reindex(b_carry.index)
    systematic, idiosyncratic = decompose(
        rx_m_aligned, b_carry, b_dollar, cm_aligned, dm_aligned,
    )

    # ── Factor momentum weights ────────────────────────────────────────────────
    N = rx_m.shape[1]
    fmom = factor_momentum_weights(
        carry_m, dollar_m, carry_w_m, N, hs=beta_hs,
    )

    # ── Conditional means (Zhang Table 3 / 4 diagnostic) ──────────────────────
    cond_means: dict = {}
    for h in beta_hs:
        sc = factor_momentum_sign(carry_m,  h)
        sd = factor_momentum_sign(dollar_m, h)
        for fname, fsign in [("carry", sc), ("dollar", sd)]:
            pos_mask = fsign > 0
            neg_mask = fsign < 0
            # what does the factor earn next month after positive/negative sign?
            factor_nx = carry_m.shift(-1) if fname == "carry" else dollar_m.shift(-1)
            cond_means[f"{fname}_h{h}_pos"] = float(factor_nx[pos_mask].mean())
            cond_means[f"{fname}_h{h}_neg"] = float(factor_nx[neg_mask].mean())

    return {
        "carry_m":          carry_m,
        "dollar_m":         dollar_m,
        "carry_d":          carry_d,
        "dollar_d":         dollar_d,
        "carry_weights":    carry_w_m,
        "b_carry":          b_carry,
        "b_dollar":         b_dollar,
        "systematic":       systematic,
        "idiosyncratic":    idiosyncratic,
        **{f"factor_mom_h{h}": fmom[h] for h in beta_hs},
        "sys_idio_momentum": build_sys_idio_momentum(systematic, idiosyncratic),
        "conditional_means": cond_means,
    }
