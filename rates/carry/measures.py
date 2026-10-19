"""
Carry definitions: four measures + a raw-yield contamination demo.

All functions:
  - Take a fitted NSFit (zero curve in bp), tenor array, funding rate in bp,
    and horizon h in years.
  - Return carry in bp of total return over the horizon h.

Rule 1 (non-negotiable): roll-down is computed from fit.predict(), never
from raw observed yields, so the carry signal is orthogonal to the NS
residual signal by construction.  The raw variant (carry_rolldown_raw) is
provided only to demonstrate the contamination; it must not be used in
production or in any spec other than axis-9 of the grid.

Units throughout:
  yields in bp        (fit.predict returns bp; rf_bp in bp)
  tenor in years      (float)
  horizon in years    (float)
  carry returned in bp  (return over the full horizon h)

approx_dv01(tau, y_dec) → modified duration (years) for a par bond.
  tau   : time to maturity (years)
  y_dec : yield as a decimal (0.04 = 4%)
  result: tau × exp(-y_dec × tau)   [same formula as signals_rv._dv01]

Duration times yield-change-in-bp gives bp of price return:
  rolldn_bp = dur_years × Δy_bp
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rates.ns_core import NSFit


# ── DV01 / duration helper ─────────────────────────────────────────────────────

def approx_dv01(tau: float, y_dec: float) -> float:
    """
    Modified duration of a par ZCB.

    tau   : time to maturity, years
    y_dec : yield as decimal (0.04 = 4%)
    Units : years  (multiply by Δy_bp → bp of price return)
    """
    return float(np.maximum(tau, 1e-8) * np.exp(-y_dec * np.maximum(tau, 1e-8)))


# ── Measure 1: yield pickup (income only) ─────────────────────────────────────

def carry_pickup(fit: "NSFit", tau: np.ndarray, rf_bp: float, h: float) -> np.ndarray:
    """
    Income-only carry: no roll-down.  The simplest measure and the control.

    Returns (y(tau) - rf) × h for each tenor.
    Positive = receiver earns over the horizon if the curve does not move.
    """
    tau = np.asarray(tau, float)
    return (fit.predict(tau) - rf_bp) * h


# ── Measure 2: carry + roll-down, duration approximation ───────────────────────

def carry_rolldown_dur(
    fit: "NSFit",
    tau: np.ndarray,
    rf_bp: float,
    h: float,
    *,
    roll_floor: float,
) -> np.ndarray:
    """
    Carry + roll-down using a first-order duration approximation.

    roll_floor: shortest tenor on the grid.  Tenors that age below roll_floor
    have no lower bucket to roll into; they roll to the funding rate (Martens
    et al convention).  Flag these in attribution if pct_pnl_from_shortest > 0.4.

    Returns income + rolldn in bp over horizon h.
    """
    tau = np.asarray(tau, float)
    y_now = fit.predict(tau)                  # bp, shape (n,)
    tau_f = tau - h                           # aged tenors

    # Where tau_f > roll_floor, interpolate fitted yield; else roll to rf
    tau_f_safe = np.maximum(tau_f, 1e-8)
    y_fwd = np.where(
        tau_f > roll_floor,
        fit.predict(tau_f_safe),
        rf_bp,
    )

    # Modified duration at the forward point (years)
    dur = np.array([
        approx_dv01(max(float(t), 1e-8), float(y) / 1e4)
        for t, y in zip(tau_f_safe, y_fwd)
    ])

    income = (y_now - rf_bp) * h       # bp × years → bp over h
    rolldn = dur * (y_now - y_fwd)     # years × bp → bp
    return income + rolldn


# ── Measure 3: full repricing (preferred) ────────────────────────────────────

def carry_reprice(
    fit: "NSFit",
    tau: np.ndarray,
    rf_bp: float,
    h: float,
    *,
    freq: int = 2,
) -> np.ndarray:
    """
    Total return in bp over h under the static-curve scenario.

    Assumes the zero curve (as a function of TTM) is unchanged at t+h.
    The bond is at par today, so coupon = par yield = fit.predict(tau).

    Handles convexity automatically via full discounting; use this measure
    as the baseline (not carry_rolldown_dur) for the grid.

    cf. carry_signal_spec.md §4.3
    """
    tau = np.asarray(tau, float)
    out = np.empty(len(tau))

    for i, n in enumerate(tau):
        # Full payment schedule from t=0
        k_all = np.arange(1, int(round(n * freq)) + 1) / freq   # all payment times

        # Derive the par yield from the zero curve so the bond is exactly at par today.
        # fit.predict may return par OIS rates (treated as approximate zeros), so computing
        # the par yield from those ensures internal consistency: P(t=0) = 1.0 exactly.
        z_all = fit.predict(k_all) / 1e4
        df_all = np.exp(-z_all * k_all)
        annuity_all = df_all.sum() / freq
        if annuity_all < 1e-10:
            out[i] = 0.0
            continue
        c_dec = (1.0 - df_all[-1]) / annuity_all   # par yield as decimal

        k_rem = k_all[k_all > h] - h               # remaining maturities at t+h

        if len(k_rem) == 0:
            out[i] = (c_dec * 1e4 - rf_bp) * h
            continue

        # Static-curve discount factors at t+h: zero curve as function of TTM unchanged
        z_rem = fit.predict(k_rem) / 1e4
        df = np.exp(-z_rem * k_rem)

        p_fwd = (c_dec / freq) * df.sum() + df[-1]
        coupons_paid = (c_dec / freq) * float(np.sum(k_all <= h))

        out[i] = ((p_fwd - 1.0) + coupons_paid) * 1e4 - rf_bp * h

    return out


# ── Measure 4: forward-based carry (yield-space equivalent) ──────────────────

def carry_forward(
    fit: "NSFit",
    tau: np.ndarray,
    rf_bp: float,
    h: float,
    *,
    freq: int = 2,
) -> np.ndarray:
    """
    Carry = annuity_rem × (K_now - K_fwd) - rf × h

    Yield-space equivalent of carry_reprice under the static-curve scenario.
    K_fwd is the par yield of the bond's remaining cashflows evaluated on the
    current zero curve (same static-curve DFs as carry_reprice).  Agrees with
    carry_reprice to within floating-point precision at all tenors and horizons;
    diverges from carry_reprice only due to convexity when h is large.

    The formula is exact (not linearised): the annuity factor acts as both the
    effective duration and the discount-factor sum, so no linearisation error.

    cf. carry_signal_spec.md §4.4
    """
    tau = np.asarray(tau, float)
    out = np.empty(len(tau))

    for i, n in enumerate(tau):
        # ── Full payment schedule ─────────────────────────────────────────────
        k_all = np.arange(1, int(round(n * freq)) + 1) / freq
        z_all = fit.predict(k_all) / 1e4
        df_all = np.exp(-z_all * k_all)

        annuity_all = df_all.sum() / freq
        if annuity_all < 1e-10:
            out[i] = 0.0
            continue
        K_now_bp = (1.0 - df_all[-1]) / annuity_all * 1e4

        tau_rem = n - h
        if tau_rem <= 1e-6:
            out[i] = (K_now_bp - rf_bp) * h
            continue

        # ── Remaining cashflow TTMs at h (static-curve, same as carry_reprice) ──
        mask = k_all > h
        if not mask.any():
            out[i] = (K_now_bp - rf_bp) * h
            continue
        k_rem = k_all[mask] - h                        # TTMs from h
        z_rem = fit.predict(k_rem) / 1e4
        df_rem = np.exp(-z_rem * k_rem)
        annuity_rem = df_rem.sum() / freq
        if annuity_rem < 1e-10:
            out[i] = (K_now_bp - rf_bp) * h
            continue
        K_fwd_bp = (1.0 - df_rem[-1]) / annuity_rem * 1e4

        # ── coupons paid before h (in bp) ─────────────────────────────────────
        n_paid = int(np.sum(~mask))
        coupons_bp = K_now_bp * n_paid / freq

        out[i] = annuity_rem * (K_now_bp - K_fwd_bp) + coupons_bp - rf_bp * h

    return out


# ── Contamination demo: raw-yield roll-down ────────────────────────────────────

def carry_rolldown_raw(
    wide_row: "pd.Series",
    tau: np.ndarray,
    rf_bp: float,
    h: float,
    *,
    roll_floor: float,
    interp: str = "linear",
) -> np.ndarray:
    """
    Roll-down interpolated from RAW observed yields (not the NS fit).

    This is the contamination demo.  Because raw yields contain the NS
    residual (rich/cheap) component, this measure's roll-down is correlated
    with the NS residual signal.  The correlation is the metric reported in
    test_orthogonality.py.

    DO NOT use outside of axis-9 of the experiment grid.
    """
    import pandas as pd  # local import to avoid top-level dependency

    tau = np.asarray(tau, float)
    s = wide_row.dropna()
    raw_tenors = s.index.to_numpy(float)
    raw_yields = s.to_numpy(float)  # bp

    tau_f = tau - h
    tau_f_safe = np.maximum(tau_f, raw_tenors[0])

    y_now = np.interp(tau, raw_tenors, raw_yields)
    y_fwd = np.where(
        tau_f > roll_floor,
        np.interp(tau_f_safe, raw_tenors, raw_yields),
        rf_bp,
    )

    dur = np.array([
        approx_dv01(max(float(t), 1e-8), float(y) / 1e4)
        for t, y in zip(np.maximum(tau_f, 1e-8), y_fwd)
    ])
    income = (y_now - rf_bp) * h
    rolldn = dur * (y_now - y_fwd)
    return income + rolldn


# ── Registry ───────────────────────────────────────────────────────────────────

CARRY_MEASURES: dict[str, object] = {
    "pickup":   carry_pickup,
    "roll_dur": carry_rolldown_dur,
    "reprice":  carry_reprice,
    "forward":  carry_forward,
}
