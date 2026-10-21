"""
Family V: PPP half-life value signals (Ca' Zorzi & Rubaszek 2018, ECB WP 2151).

Convention note (§1.1): s = log(FCU/USD) throughout. An increase in s means
the foreign currency depreciated. The real exchange rate is:

    q = s + p_us - p_f

where q RISING means the foreign currency is CHEAP in real terms → should
appreciate → go LONG. So the value signal is +(q - q_bar): long real-cheap.

The binding data constraint (§0): CPI is not in the Bloomberg workbook.
  - With rpi (p_f - p_us) provided: full Ca' Zorzi & Rubaszek HL and 2HL models.
  - Without rpi (rpi=None): Route 3 nominal-only proxy for G10 ONLY.
    A hard assertion prevents EM pairs in nominal-only mode.

Family V module is gated: build_signal_v returns empty results if CPI unavailable
and nominal_only=False.

Value + carry combinations (§5):
  C1 — independent books, summed at a chosen risk split
  C2 — value as a carry filter (drop overvalued longs / undervalued shorts)
  C3 — rank on combined expected return h*fd + (1-ρ^h)(q-q_bar)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


# ── §1.3 Half-life calibration ────────────────────────────────────────────────

def rho_from_halflife(years: float = 3.0, periods_per_year: int = 12) -> float:
    """
    Monthly AR(1) coefficient implied by a given half-life in years.

    Verification (spec napkin, h in months):
      h=6  → 1 - ρ^6  ≈ 0.109
      h=12 → 1 - ρ^12 ≈ 0.206
      h=24 → 1 - ρ^24 ≈ 0.370
      h=36 → 1 - ρ^36 = 0.500
    """
    return 0.5 ** (1.0 / (years * periods_per_year))


RHO_3Y   = rho_from_halflife(3.0)
RHO_PI   = 0.5 ** (1.0 / 6.0)    # 6-month half-life for inflation differential

# G10 universe restriction for the nominal-only proxy (§6)
G10_ONLY: frozenset[str] = frozenset({
    "AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCAD",
    "USDCHF", "USDJPY", "USDNOK", "USDSEK", "USDDKK",
})


# ── §1.4 Recursive mean (no lookahead) ────────────────────────────────────────

def recursive_mean(q: pd.DataFrame | pd.Series, min_periods: int = 60) -> pd.DataFrame | pd.Series:
    """
    Expanding mean lagged by 1 period: q_bar at date t uses only data up to t-1.

    min_periods=60 = 5 years; first 5 years of each pair are NaN.
    """
    return q.expanding(min_periods=min_periods).mean().shift(1)


# ── §4.2 HL value signal ─────────────────────────────────────────────────────

def hl_value_signal(
    q: pd.DataFrame,
    q_bar: pd.DataFrame,
    halflife_years: float = 3.0,
    h: int = 12,
) -> pd.DataFrame:
    """
    Expected h-month log appreciation of the foreign currency.

    Positive = foreign currency cheap in real terms = go long foreign.
    The (1-ρ^h) factor preserves cross-sectional ranking and puts the
    signal in units of expected log return over h months.
    """
    rho = rho_from_halflife(halflife_years)
    return (1.0 - rho ** h) * (q - q_bar)


# ── §4.3 2HL value signal ─────────────────────────────────────────────────────

def two_hl_value_signal(
    q: pd.DataFrame,
    q_bar: pd.DataFrame,
    dpi: pd.DataFrame,
    halflife_years: float = 3.0,
    h: int = 12,
) -> pd.DataFrame:
    """
    Two-half-life model: RER mean reversion + inflation-differential mean reversion.

    dpi = monthly change in (p_f - p_us), the inflation differential.
    Expected to marginally lose to HL (§4.3 diagnostic).
    """
    rho = rho_from_halflife(halflife_years)
    rer_part = (1.0 - rho ** h) * (q - q_bar)
    pi_part  = dpi * RHO_PI * (1.0 - RHO_PI ** h) / (1.0 - RHO_PI)
    return rer_part + pi_part


# ── §4.4 Value portfolio ──────────────────────────────────────────────────────

def value_weights(sig: pd.DataFrame, n: int = 4) -> pd.DataFrame:
    """
    Long top-n (most real-cheap) and short bottom-n (most real-expensive).

    Same shape as the carry book (4 long, 4 short) for direct comparability.
    """
    r = sig.rank(axis=1, ascending=False)
    k = sig.notna().sum(axis=1)
    w = np.where(
        r.values <= n,                        1.0 / n,
        np.where(r.values > (k.values[:, None] - n), -1.0 / n, 0.0)
    )
    w = pd.DataFrame(w, index=sig.index, columns=sig.columns)
    return w.where(sig.notna(), 0.0)


# ── §6 Nominal-only proxy (Route 3, G10 only) ────────────────────────────────

def nominal_value_signal(
    s_m: pd.DataFrame,
    halflife_years: float = 3.0,
    h: int = 12,
    min_periods: int = 60,
    nominal_only: bool = True,
) -> pd.DataFrame:
    """
    Degrade the PPP signal to nominal rate mean reversion.

    Restricted to G10: EM pairs have trending nominal rates driven by inflation
    differentials; mean reversion to a nominal recursive mean is structurally
    wrong and will permanently lose on high-inflation pairs (TRY, BRL, MXN, etc.).

    Raises if nominal_only=True and columns outside G10_ONLY are present.
    """
    if nominal_only:
        non_g10 = set(s_m.columns) - G10_ONLY
        if non_g10:
            raise ValueError(
                f"nominal_only mode requires G10 pairs only; found EM pairs: {sorted(non_g10)}. "
                "Either remove EM pairs or provide CPI data for the full PPP model."
            )
    rho = rho_from_halflife(halflife_years)
    q_bar = recursive_mean(s_m, min_periods=min_periods)
    return (1.0 - rho ** h) * (s_m - q_bar)


# ── §4.5 In-sample regression diagnostics ────────────────────────────────────

def ppp_regressions(
    q: pd.DataFrame,
    q_bar: pd.DataFrame,
    s_m: pd.DataFrame,
    rpi: pd.DataFrame | None,
    hs: tuple = (6, 12, 24, 60),
) -> dict:
    """
    Equations (2a)–(2c) from the paper, pooled across pairs.

    (2a): Δq_{t,h} = α0 + α1 * q_{t-h}   expect α1 → -1 as h grows
    (2b): Δs_{t,h} = β0 + β1 * Δq_{t,h}  expect β1 ≈ 1
    (2c): Δrpi_{t,h} = γ0 + γ1 * Δq_{t,h} expect γ1 ≈ 0 (if rpi provided)

    Validates the whole q construction before building portfolios.
    """
    results: dict = {}
    misalign = q - q_bar    # NaN in the first 60 months per pair

    for h in hs:
        dq = q - q.shift(h)
        q_lag = q.shift(h)

        # stack to pooled panel (pair as extra observations)
        pairs_q   = q_lag.stack(future_stack=True).dropna()
        pairs_dq  = dq.reindex_like(q_lag).stack(future_stack=True).reindex(pairs_q.index)
        idx = pairs_q.index.intersection(pairs_dq.index)
        y_a = pairs_dq.loc[idx].values
        x_a = np.column_stack([np.ones(len(idx)), pairs_q.loc[idx].values])
        if len(x_a) >= 2:
            beta_a, *_ = np.linalg.lstsq(x_a, y_a, rcond=None)
            results[f"alpha1_h{h}"] = float(beta_a[1])

        # (2b): Δs vs Δq
        ds = s_m - s_m.shift(h)
        idx2 = dq.stack(future_stack=True).dropna().index.intersection(
               ds.stack(future_stack=True).dropna().index)
        if len(idx2) >= 2:
            y_b = ds.stack(future_stack=True).reindex(idx2).values
            x_b = np.column_stack([np.ones(len(idx2)), dq.stack(future_stack=True).reindex(idx2).values])
            beta_b, *_ = np.linalg.lstsq(x_b, y_b, rcond=None)
            results[f"beta1_h{h}"] = float(beta_b[1])

        # (2c): Δrpi vs Δq (only if rpi provided)
        if rpi is not None:
            drpi = rpi - rpi.shift(h)
            idx3 = dq.stack(future_stack=True).dropna().index.intersection(
                   drpi.stack(future_stack=True).dropna().index)
            if len(idx3) >= 2:
                y_c = drpi.stack(future_stack=True).reindex(idx3).values
                x_c = np.column_stack([np.ones(len(idx3)), dq.stack(future_stack=True).reindex(idx3).values])
                beta_c, *_ = np.linalg.lstsq(x_c, y_c, rcond=None)
                results[f"gamma1_h{h}"] = float(beta_c[1])

    return results


# ── §5 Value + carry combinations ────────────────────────────────────────────

def c1_independent_books(
    w_carry: pd.DataFrame,
    w_value: pd.DataFrame,
    risk_split: float = 0.5,
) -> pd.DataFrame:
    """
    C1: sum carry and value weights at a 50/50 risk split (adjustable).

    Both books should already be at unit gross exposure of 2 before calling.
    risk_split = share allocated to carry (1-risk_split to value).
    """
    # align to common dates
    c, v = w_carry.align(w_value, join="outer", fill_value=0.0)
    return risk_split * c + (1.0 - risk_split) * v


def c2_carry_filter(
    w_carry: pd.DataFrame,
    value_sig: pd.DataFrame,
    pct: float = 1.0 / 3.0,
) -> pd.DataFrame:
    """
    C2: drop overvalued longs and undervalued shorts from the carry portfolio.

    Removes from long leg any pair in the most-expensive value tercile, and
    from short leg any pair in the cheapest value tercile.
    """
    c, vs = w_carry.align(value_sig, join="left")
    rank_pct = vs.rank(axis=1, pct=True)
    # cheapest: high value_sig (real-cheap foreign) → rank near 1 → keep in long
    # most expensive: low rank → might be in long leg → drop
    most_expensive = rank_pct < pct          # low value rank = overvalued
    cheapest       = rank_pct > (1.0 - pct)  # high value rank = undervalued

    w = c.copy()
    w = w.mask((w > 0) & most_expensive, 0.0)   # drop overvalued longs
    w = w.mask((w < 0) & cheapest,       0.0)   # drop undervalued shorts

    row_gross = w.abs().sum(axis=1)
    out = w.div(row_gross / 2.0, axis=0)
    out.loc[row_gross == 0] = np.nan
    return out


def c3_combined_rank(
    fd: pd.DataFrame,
    q: pd.DataFrame,
    q_bar: pd.DataFrame,
    halflife_years: float = 3.0,
    h: int = 12,
    n: int = 4,
) -> pd.DataFrame:
    """
    C3: rank on combined expected return  h*fd + (1-ρ^h)(q - q_bar).

    Uses carry level (not rank), so is sensitive to the forward-point divisor
    calibration. Run the divisor sanity check before trusting this.
    """
    rho = rho_from_halflife(halflife_years)
    exp_ret = h * fd + (1.0 - rho ** h) * (q - q_bar)
    return value_weights(exp_ret, n=n)


# ── Full signal builder ───────────────────────────────────────────────────────

def build_signal_v(
    s_m: pd.DataFrame,
    fd_m: pd.DataFrame,
    rpi: pd.DataFrame | None = None,
    carry_w_m: pd.DataFrame | None = None,
    halflife_grid: tuple = (2.0, 3.0, 4.0, 5.0),
    h_primary: int = 12,
    n: int = 4,
    min_periods: int = 60,
    cpi_lag_months: int = 2,
) -> dict:
    """
    Build all Family-V value signals.

    Parameters
    ----------
    s_m           log spot FCU/USD, month-end date × pair
    fd_m          log 1M forward discount, month-end date × pair
    rpi           log(p_f/p_us), monthly date × pair; None triggers nominal-only
    carry_w_m     carry portfolio weights (optional; for C2/C3 combinations)
    halflife_grid half-life years to run as robustness grid
    h_primary     forecast horizon in months for the primary signal
    n             long/short legs per portfolio
    min_periods   minimum history for recursive mean (60 = 5 years)
    cpi_lag_months CPI publication lag applied before computing q (default 2)

    Returns
    -------
    dict with keys:
      'nominal_only'     — True if rpi was not provided
      'q'                — real (or nominal if nominal_only) exchange rate
      'q_bar'            — recursive mean (lagged)
      'misalignment'     — q - q_bar
      'hl_{hl}y_h{h}'    — HL value signal at (halflife, horizon)
      '2hl_{hl}y_h{h}'   — 2HL signal (if rpi provided)
      'weights_hl_{hl}y' — 4×4 portfolio from HL at h=h_primary
      'c1', 'c2', 'c3'   — value+carry combos (if carry_w_m and q available)
      'regressions'      — α1, β1, γ1 at h in {6,12,24,60}
    """
    nominal_only = rpi is None
    result: dict = {"nominal_only": nominal_only}

    if nominal_only:
        g10_pairs = [p for p in s_m.columns if p in G10_ONLY]
        if not g10_pairs:
            warnings.warn("No G10 pairs found; Family V signals empty.")
            return result
        s_g10 = s_m[g10_pairs]
        try:
            sig_nom = nominal_value_signal(
                s_g10, halflife_years=3.0, h=h_primary,
                min_periods=min_periods, nominal_only=True,
            )
        except ValueError as e:
            warnings.warn(str(e))
            return result
        # for nominal-only, q = s and q_bar = recursive mean of s
        q = s_g10.copy()
        q_bar = recursive_mean(s_g10, min_periods=min_periods)
        result.update({"q": q, "q_bar": q_bar, "misalignment": q - q_bar})
        result["nominal_value_signal"] = sig_nom
        result["weights_nominal_hl_3y"] = value_weights(sig_nom, n=n)

        # value + carry combos (nominal-only)
        if carry_w_m is not None:
            c_g10, v_g10 = carry_w_m[g10_pairs].align(value_weights(sig_nom, n=n), join="inner")
            result["c1_nominal"] = c1_independent_books(c_g10, v_g10)
            result["c2_nominal"] = c2_carry_filter(c_g10, sig_nom)
            result["c3_nominal"] = c3_combined_rank(
                fd_m[g10_pairs].align(sig_nom, join="inner")[0],
                q, q_bar, halflife_years=3.0, h=h_primary, n=n
            )
        return result

    # ── Full PPP model (CPI available) ────────────────────────────────────────
    # apply CPI publication lag
    rpi_lagged = rpi.shift(cpi_lag_months)

    # align everything
    common_pairs = s_m.columns.intersection(rpi_lagged.columns)
    s  = s_m[common_pairs]
    fd = fd_m[common_pairs] if fd_m is not None else None
    rp = rpi_lagged[common_pairs]

    q = s + (-rp)         # q = s + p_us - p_f = s - (p_f - p_us) = s - rpi
    q_bar = recursive_mean(q, min_periods=min_periods)
    misalign = q - q_bar
    dpi = rp.diff()

    result.update({"q": q, "q_bar": q_bar, "misalignment": misalign})

    for hl in halflife_grid:
        sig_hl = hl_value_signal(q, q_bar, halflife_years=hl, h=h_primary)
        result[f"hl_{hl:.0f}y_h{h_primary}"] = sig_hl
        result[f"weights_hl_{hl:.0f}y"] = value_weights(sig_hl, n=n)

        sig_2hl = two_hl_value_signal(q, q_bar, dpi, halflife_years=hl, h=h_primary)
        result[f"2hl_{hl:.0f}y_h{h_primary}"] = sig_2hl
        result[f"weights_2hl_{hl:.0f}y"] = value_weights(sig_2hl, n=n)

    # primary signal (3y half-life)
    sig_primary = hl_value_signal(q, q_bar, halflife_years=3.0, h=h_primary)
    result["hl_primary"] = sig_primary

    # in-sample regressions (§4.5)
    result["regressions"] = ppp_regressions(q, q_bar, s, rp, hs=(6, 12, 24, 60))

    # value + carry combinations (§5)
    if carry_w_m is not None:
        c_pairs = carry_w_m.columns.intersection(common_pairs)
        w_c = carry_w_m[c_pairs]
        w_v = value_weights(sig_primary[c_pairs] if c_pairs is not None else sig_primary, n=n)

        result["c1"] = c1_independent_books(w_c, w_v)
        result["c2"] = c2_carry_filter(w_c, sig_primary[c_pairs] if c_pairs.size > 0 else sig_primary)
        if fd is not None:
            result["c3"] = c3_combined_rank(
                fd[c_pairs], q[c_pairs], q_bar[c_pairs],
                halflife_years=3.0, h=h_primary, n=n,
            )

    return result
