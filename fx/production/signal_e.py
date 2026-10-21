"""
Signal E: Brunnermeier, Nagel & Pedersen (2008) crash-risk conditioned carry.

The base carry portfolio (from signal_d.carry_weights) is modulated three ways:

  E1 — screen: drop the most crash-exposed long positions
  E2 — adjusted carry: rank on fd / cost-of-protection rather than fd alone
  E3 — aggregate de-risking: scale down when global vol high / skew negative

Option-moment extension (§6.4, Chen 2017 described): carry interacted with
option-implied vol, skew and tail. Flagged as speculative.

Cross-sectional diagnostic (§6.2) must be run first: mean RR should be
negative for high-carry currencies (AUD, NZD, ZAR, TRY, MXN) and positive
for funding currencies (JPY, CHF).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fx.production.signal_d import carry_weights


# ── Crash-risk measures ───────────────────────────────────────────────────────

def realised_skew(s_log: pd.DataFrame, freq: str = "QE") -> pd.DataFrame:
    """
    BNP physical crash measure: skewness of foreign-currency appreciation (-Δs).
    """
    ds = -s_log.diff()
    return ds.groupby(pd.Grouper(freq=freq)).skew()


def cross_sectional_carry_skew_corr(
    fd_m: pd.DataFrame,
    rr25_m: pd.DataFrame,
) -> float:
    """
    §6.2 diagnostic: cross-sectional correlation of mean fd and mean RR.

    Expect clearly negative. If positive, the RR orientation is wrong
    and Signal E will be inverted.
    """
    mean_fd  = fd_m.mean()
    mean_rr  = rr25_m.mean()
    pairs    = mean_fd.index.intersection(mean_rr.index)
    if len(pairs) < 4:
        return np.nan
    return float(np.corrcoef(mean_fd[pairs], mean_rr[pairs])[0, 1])


# ── E1: crash-risk screen ─────────────────────────────────────────────────────

def e1_screen(
    w_carry: pd.DataFrame,
    rr: pd.DataFrame,
    pct: float = 0.2,
) -> pd.DataFrame:
    """
    Drop long-leg currencies whose foreign-oriented RR is in the most negative
    pct of the cross-section (market is most expensive to insure their crash).

    Re-normalises remaining weights to maintain 1-dollar-long, 1-dollar-short
    gross exposure.
    """
    thresh = rr.quantile(pct, axis=1)
    drop   = rr.lt(thresh, axis=0) & (w_carry > 0)
    w      = w_carry.mask(drop, 0.0)
    gross  = w.abs().sum(axis=1).replace(0, np.nan)
    return w.div(gross / 2.0, axis=0)


# ── E2: carry per unit of implied crash risk ──────────────────────────────────

def e2_adjusted_carry(
    fd: pd.DataFrame,
    rr: pd.DataFrame,
    bf: pd.DataFrame,
    kind: str = "rr",
) -> pd.DataFrame:
    """
    Rank on fd / protection_cost rather than fd alone.

    kind in {'rr', 'bf', 'rr_bf'}.
      rr      : cost = -rr25 (left-tail skew cost)
      bf      : cost = bf25 (both tails cost)
      rr_bf   : cost = -rr25 + bf25 (combined)

    Protection cost floored at 0.1 vol points to avoid division by near-zero.
    """
    denom_map = {
        "rr":    -rr,
        "bf":    bf,
        "rr_bf": (-rr + bf),
    }
    if kind not in denom_map:
        raise ValueError(f"kind must be one of {list(denom_map)}")
    denom = denom_map[kind].clip(lower=0.1)
    return fd / denom


# ── E3: aggregate de-risking ──────────────────────────────────────────────────

def e3_global_state(
    atm: pd.DataFrame,
    rr: pd.DataFrame,
    min_periods: int = 36,
) -> pd.Series:
    """
    Global FX vol and risk-appetite state, z-scored expanding.

    Returns a scalar in [-2, 2]: negative when global vol is high and/or
    global skew is negative (crisis signal → scale carry down).

    Clipped at ±2 to limit leverage.
    """
    gvol  = atm.mean(axis=1)
    gskew = rr.mean(axis=1)

    def _z(x: pd.Series) -> pd.Series:
        mu  = x.expanding(min_periods=min_periods).mean()
        std = x.expanding(min_periods=min_periods).std().replace(0, np.nan)
        return (x - mu) / std

    return (-_z(gvol) - _z(gskew)).clip(-2.0, 2.0)


# ── Full signal builder ───────────────────────────────────────────────────────

def build_signal_e(
    fd_m: pd.DataFrame,
    rr25_m: pd.DataFrame,
    rr10_m: pd.DataFrame,
    bf25_m: pd.DataFrame,
    bf10_m: pd.DataFrame,
    atm_m: pd.DataFrame,
    s_m: pd.DataFrame,
    n: int = 4,
    e1_pct: float = 0.2,
    e3_min_periods: int = 36,
) -> dict:
    """
    Build all E1/E2/E3 signal variants.

    Returns
    -------
    dict with keys:
      'base_weights'           — plain carry weights (date × pair)
      'cs_corr_fd_rr'          — §6.2 cross-sectional diagnostic (scalar)
      'realised_skew'          — quarterly realised skew (date × pair)
      'e1_weights'             — E1 screened weights
      'e2_weights_{kind}'      — E2 adjusted for kind in {rr, bf, rr_bf}
      'e3_state'               — E3 global state time series
      'e3_weights'             — E3 scaled weights
      'e2_e3_corr'             — correlation of E3 scaler vs Dupuy 1/V (diagnostic)
    """
    # align all panels to common dates
    common = fd_m.index
    for df in [rr25_m, bf25_m, atm_m]:
        if not df.empty:
            common = common.intersection(df.index)

    fd   = fd_m.loc[common]
    rr25 = rr25_m.reindex(common) if not rr25_m.empty else pd.DataFrame(index=common)
    rr10 = rr10_m.reindex(common) if not rr10_m.empty else pd.DataFrame(index=common)
    bf25 = bf25_m.reindex(common) if not bf25_m.empty else pd.DataFrame(index=common)
    bf10 = bf10_m.reindex(common) if not bf10_m.empty else pd.DataFrame(index=common)
    atm  = atm_m.reindex(common)  if not atm_m.empty  else pd.DataFrame(index=common)

    w_base = carry_weights(fd, n=n)

    results: dict = {
        "base_weights": w_base,
        "cs_corr_fd_rr": cross_sectional_carry_skew_corr(fd, rr25),
    }

    if not s_m.empty:
        results["realised_skew"] = realised_skew(s_m.reindex(common, method="ffill"))

    # E1
    if not rr25.empty:
        results["e1_weights"] = e1_screen(w_base, rr25, pct=e1_pct)

    # E2 (use 25Δ RR and BF)
    for kind in ("rr", "bf", "rr_bf"):
        try:
            adj = e2_adjusted_carry(fd, rr25, bf25, kind=kind)
            w_adj = carry_weights(adj, n=n)
            results[f"e2_weights_{kind}"] = w_adj
        except Exception:
            pass

    # E3
    if not atm.empty and not rr25.empty:
        state = e3_global_state(atm, rr25, min_periods=e3_min_periods)
        results["e3_state"] = state
        state_aligned = state.shift(1).reindex(w_base.index, method="ffill")
        results["e3_weights"] = w_base.mul(state_aligned, axis=0)

    return results


# ── Option-moment extension (§6.4, speculative) ───────────────────────────────

def build_option_moment_signals(
    fd_m: pd.DataFrame,
    atm_m: pd.DataFrame,
    rr25_m: pd.DataFrame,
    bf25_m: pd.DataFrame,
    bf10_m: pd.DataFrame,
    n: int = 4,
    min_periods: int = 36,
) -> dict:
    """
    Chen (2017) interaction signals: carry ranked on option-implied moments.

    Speculative — three predictors on a 23-year option sample needs deflated
    Sharpe at selection time before any trading conclusion.

    Returns level-form and change-form signal weights.
    """
    def _z_cross(x: pd.DataFrame) -> pd.DataFrame:
        mu  = x.expanding(min_periods=min_periods).mean()
        std = x.expanding(min_periods=min_periods).std().replace(0, np.nan)
        return (x - mu) / std

    results: dict = {}
    try:
        z_vol  = _z_cross(atm_m)
        z_skew = _z_cross(-rr25_m)   # more negative = worse skew
        z_tail = _z_cross(bf25_m)

        for name, z in [("vol", z_vol), ("skew", z_skew), ("tail", z_tail)]:
            adj = fd_m * z
            results[f"level_{name}_weights"] = carry_weights(adj, n=n)

        d_skew = rr25_m.diff(1)
        results["change_skew_weights"] = -np.sign(d_skew) * carry_weights(fd_m, n=n)
    except Exception:
        pass

    return results
