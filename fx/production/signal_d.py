"""
Signal D: Dupuy (2021, JBF) risk-adjusted managed carry trade.

Carry portfolio with leverage timed by E/V where:
  E = expected profit (average forward discount of attractive currencies)
  V = realised market variance (FSS with AR(1) correction)

The extra one-month lag on the signal is explicit in the paper and must not
be removed: the vol leg of V is measured over month t and overlaps t if used
immediately. The signal uses data through t, positions held from t+1 to t+2.

Missing bid/ask: the attractive-currency filter degenerates to a simple
long/short on sign of forward discount (all currencies attractive on one side).
See expected_profit_mid and the cost_bp sweep for a partial fix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Base carry portfolio ──────────────────────────────────────────────────────

def carry_weights(fd: pd.DataFrame, n: int = 4) -> pd.DataFrame:
    """
    Dupuy equation (6): rank by forward discount, long top-n, short bottom-n.

    Equal weight within each leg (1/n). Weights are zero for pairs not in
    top-n or bottom-n, and zero for NaN fd values.
    """
    r    = fd.rank(axis=1, ascending=False)
    k    = fd.notna().sum(axis=1)
    lo_cut = (k.values - n)[:, None]

    w = np.where(r.values <= n,         1.0 / n,
        np.where(r.values > lo_cut,    -1.0 / n, 0.0))
    w = pd.DataFrame(w, index=fd.index, columns=fd.columns)
    return w.where(fd.notna(), 0.0)


# ── Expected profit (E) ───────────────────────────────────────────────────────

def expected_profit_mid(
    fd: pd.DataFrame,
    cost_bp: float | None = None,
) -> pd.Series:
    """
    Dupuy equation (2), degraded for mid quotes (no bid/ask dead zone).

    With mid quotes every currency is attractive on exactly one side, so the
    attractive-currency filter is vacuous. cost_bp restores a synthetic dead
    zone of that half-spread width.

    Baseline: cost_bp=None.  Sweep: cost_bp in {2, 5, 10}.
    """
    if cost_bp is not None:
        c    = cost_bp / 1e4
        buy  = fd.where(fd >  c)
        sell = fd.where(fd < -c)
    else:
        buy  = fd.where(fd > 0)
        sell = fd.where(fd < 0)

    return buy.mean(axis=1).fillna(0.0) - sell.mean(axis=1).fillna(0.0)


# ── Market variance (V) ───────────────────────────────────────────────────────

def market_variance(rx_daily: pd.DataFrame) -> pd.Series:
    """
    Dupuy equation (3): French-Schwert-Stambaugh realised variance with
    first-order autocovariance correction on the equal-weighted FX market portfolio.

    MV can be negative (strong negative autocorrelation); floored at 1e-8.
    """
    rM = rx_daily.mean(axis=1)

    def _mv(x: pd.Series) -> float:
        D = len(x)
        if D < 15:
            return np.nan
        sq    = float((x ** 2).sum()) / D
        xcov  = float((x * x.shift(1)).sum()) / D
        return max(sq + 2.0 * xcov, 1e-8)

    return rM.groupby(pd.Grouper(freq="ME")).apply(_mv)


# ── Signal ────────────────────────────────────────────────────────────────────

def dupuy_signal(
    E: pd.Series,
    MV: pd.Series,
    mode: str = "E_over_V",
    min_periods: int = 36,
) -> pd.Series:
    """
    Dupuy equation (4): normalised E/V overlay with the explicit extra lag.

    mode in {'E_over_V', 'E_only', 'V_only'}.

    Normalisation: expanding mean so the average exposure equals 1.
    Dupuy normalises on the full sample (requires lookahead); the expanding
    version is the implementable approximation he acknowledges.

    The .shift(1) is the extra month lag documented in §5.4. Do not remove.
    """
    raw_map = {
        "E_over_V": E / MV,
        "E_only":   E,
        "V_only":   1.0 / MV,
    }
    if mode not in raw_map:
        raise ValueError(f"mode must be one of {list(raw_map)}")

    # align E and MV to the same monthly index
    e, mv = E.align(MV, join="inner")
    raw = raw_map[mode].reindex(e.index) if mode != "V_only" else (1.0 / mv)
    if mode == "E_over_V":
        raw = e / mv
    elif mode == "E_only":
        raw = e
    else:
        raw = 1.0 / mv

    norm = raw / raw.expanding(min_periods=min_periods).mean()
    return norm.shift(1)   # the extra month lag from the paper


# ── Full signal builder ───────────────────────────────────────────────────────

def build_signal_d(
    fd_m: pd.DataFrame,
    rx_d: pd.DataFrame,
    n: int = 4,
    cost_bp_sweep: tuple = (None, 2, 5, 10),
    modes: tuple = ("E_over_V", "E_only", "V_only"),
) -> dict:
    """
    Build Signal D for all modes and cost_bp values.

    Returns
    -------
    dict with keys:
      'base_weights'        — unlevered carry weights (date × pair)
      'E'                   — expected profit time series
      'MV'                  — market variance time series
      'signal_{mode}'       — normalised E/V signal (month-end series)
      'weights_{mode}'      — base_weights * signal_{mode} (date × pair)
    Also 'MV_floor_count' and 'MV_floor_pct' diagnostics.
    """
    w_base = carry_weights(fd_m, n=n)
    MV     = market_variance(rx_d)

    mv_floor_binds = (MV == 1e-8).sum()
    mv_floor_pct   = float(mv_floor_binds / MV.notna().sum()) if MV.notna().sum() > 0 else 0.0

    # align fd to month-end (should already be, but ensure)
    fd_me = fd_m.resample("BME").last() if not fd_m.index.freq else fd_m

    results: dict = {
        "base_weights":    w_base,
        "MV":              MV,
        "MV_floor_count":  int(mv_floor_binds),
        "MV_floor_pct":    mv_floor_pct,
    }

    for cost_bp in cost_bp_sweep:
        key = "E" if cost_bp is None else f"E_cost{cost_bp}bp"
        E = expected_profit_mid(fd_me, cost_bp=cost_bp)
        results[key] = E

        for mode in modes:
            sig = dupuy_signal(E, MV, mode=mode)
            sig_key = f"signal_{mode}" + ("" if cost_bp is None else f"_cost{cost_bp}bp")
            w_key   = f"weights_{mode}" + ("" if cost_bp is None else f"_cost{cost_bp}bp")
            results[sig_key] = sig
            # broadcast signal (month-end series) onto base_weights (date × pair)
            sig_aligned = sig.reindex(w_base.index, method="ffill")
            results[w_key] = w_base.mul(sig_aligned, axis=0)

    # ensure "E" key always present (no-cost baseline)
    if "E" not in results:
        results["E"] = expected_profit_mid(fd_me)

    return results
