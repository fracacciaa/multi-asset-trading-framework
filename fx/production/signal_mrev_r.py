"""
Family R: short-term mean-reversion signals.

R1 (§2.2): cross-sectional reversal — rank past losers vs winners, go contrarian.
R2 (§2.3): per-pair time-series reversal, vol-normalised.

Both built at lookbacks k in {5, 10, 20} business days, holding h in {5, 20} days.

Three ranking modes (§2.1):
  'spot'           — pure spot return; filters carry contamination (primary)
  'total'          — carry-inclusive return; control; expected ≈ short carry
  'carry_demeaned' — total minus cross-sectional mean

Overlapping holding via rolling mean (Jegadeesh-Titman, §2.4).

Conditional variants (§3):
  large_move — only act when |z_{k-day}| > thresh
  vol_regime — tercile of the expanding aggregate-vol percentile

Economic significance gate (§8): edge_bp, turnover, breakeven_bp computed before
any Sharpe comparison. Pairs whose breakeven exceeds their typical bid/ask spread
cannot sustain the signal; flag them rather than including them in a Sharpe.

Note on vol convention: all vol here is in DAILY units (std of daily log returns,
not annualised). Callers that need annualised vol should multiply by sqrt(261).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── §2.1 Past return in three modes ──────────────────────────────────────────

def past_return(
    s_log: pd.DataFrame,
    fd_1m_d: pd.DataFrame,
    k: int,
    mode: str = "spot",
) -> pd.DataFrame:
    """
    k-business-day past return, positive = foreign currency appreciated.

    Parameters
    ----------
    s_log    log spot FCU/USD, daily date × pair
    fd_1m_d  log 1M forward discount reindexed to daily (month-end value held)
    k        lookback in business days
    mode     'spot', 'total', or 'carry_demeaned'
    """
    # s falls when foreign appreciates; negate so positive = foreign gained
    spot_ret = -(s_log - s_log.shift(k))
    if mode == "spot":
        return spot_ret
    # carry accrual: hold the previous month-end discount fixed through the month
    carry_acc = fd_1m_d.shift(1).rolling(k).sum() / 21.0
    tot = spot_ret + carry_acc
    if mode == "total":
        return tot
    if mode == "carry_demeaned":
        return tot.sub(tot.mean(axis=1), axis=0)
    raise ValueError(f"mode must be 'spot', 'total', or 'carry_demeaned'; got {mode!r}")


# ── Trailing realised vol (daily, not annualised) ─────────────────────────────

def trailing_vol(s_log: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Trailing std of daily log spot returns over `window` days (not annualised)."""
    return s_log.diff().rolling(window).std()


# ── §2.2 R1: cross-sectional reversal ─────────────────────────────────────────

def xs_reversal(past: pd.DataFrame) -> pd.DataFrame:
    """
    Rank-linear contrarian weights. Long past losers, short past winners.

    Weights sum to zero at each date. Gross exposure = 2.

    rank(pct=True) gives values in {1/n, ..., n/n} whose mean is (n+1)/(2n),
    not 0.5. Subtract the row mean explicitly to ensure exact dollar neutrality
    for any universe size.
    """
    r = past.rank(axis=1, pct=True)
    w = -(r.sub(r.mean(axis=1), axis=0))   # demean so sum = 0 exactly
    row_gross = w.abs().sum(axis=1)
    valid = row_gross > 1e-12
    out = w.div(row_gross / 2.0, axis=0)
    out.loc[~valid] = np.nan
    return out


# ── §2.3 R2: time-series reversal ─────────────────────────────────────────────

def ts_reversal(past: pd.DataFrame, vol_d: pd.DataFrame, cap: float = 2.0) -> pd.DataFrame:
    """
    Per-pair contrarian position = -z(past k-day move), capped at `cap`.

    vol_d is daily trailing vol (lagged 1 day inside this function to avoid
    using today's vol to normalise today's move).

    R2 is not dollar-neutral; the net dollar exposure is a diagnostic.
    """
    z = past / vol_d.shift(1).clip(lower=1e-8)
    return (-z).clip(-cap, cap)


# ── §2.4 Overlapping holding periods ─────────────────────────────────────────

def apply_holding(w: pd.DataFrame, h: int) -> pd.DataFrame:
    """Jegadeesh-Titman rolling mean: average h daily vintages of weights."""
    if h <= 1:
        return w
    return w.rolling(h).mean()


# ── §3.1 Large-move conditioning mask ────────────────────────────────────────

def large_move_mask(
    past: pd.DataFrame,
    vol_d: pd.DataFrame,
    k: int,
    thresh: float = 2.0,
) -> pd.DataFrame:
    """
    True where the k-day move exceeds `thresh` k-day standard deviations.

    Scales daily vol to the k-day horizon by sqrt(k), then tests |past| / vol_k.
    """
    vol_k = vol_d.shift(1).clip(lower=1e-8) * np.sqrt(k)
    return (past.abs() / vol_k) > thresh


def _apply_mask_and_renorm(w: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Zero positions where mask is False, renormalise gross exposure to 2."""
    w_m = w.where(mask, 0.0)
    row_gross = w_m.abs().sum(axis=1)
    out = w_m.div(row_gross / 2.0, axis=0)
    out.loc[row_gross == 0] = np.nan
    return out


# ── §3.2 Volatility regime (expanding percentile) ────────────────────────────

def vol_regime_percentile(
    realised_vol_d: pd.DataFrame,
    atm_1m_d: pd.DataFrame | None = None,
    mode: str = "realised",
) -> pd.Series:
    """
    Cross-sectional average vol state as an expanding empirical CDF value [0, 1].

    mode='realised' uses trailing daily spot vol (available 1990-01).
    mode='implied'  uses ATM 1M vol reindexed to daily (available 2003-10).

    Returns NaN until min_periods=250 observations are available.
    """
    if mode == "implied":
        if atm_1m_d is None or atm_1m_d.empty:
            raise ValueError("atm_1m_d required for implied vol regime")
        v = atm_1m_d.mean(axis=1)
    else:
        v = realised_vol_d.mean(axis=1)

    # expanding empirical CDF: fraction of past values strictly below current
    return v.expanding(min_periods=250).apply(
        lambda x: (x.iloc[-1] > x).mean(), raw=False
    )


# ── §8 Economic significance gate ────────────────────────────────────────────

def economic_significance(
    w: pd.DataFrame,
    rx: pd.DataFrame,
    tc_bp: dict[str, float] | None = None,
) -> dict:
    """
    Compute edge_bp, turnover, breakeven_bp per spec §8.

    All three numbers must be visible before any Sharpe comparison. A signal
    whose breakeven is below its typical bid/ask cannot survive in that pair.

    Parameters
    ----------
    w      weights (daily date × pair); may contain NaN
    rx     daily excess returns (same shape as w)
    tc_bp  optional per-pair round-trip cost in bp

    Returns
    -------
    dict with 'edge_bp', 'turnover', 'breakeven_bp', and (if tc_bp) per-pair splits
    """
    common = w.index.intersection(rx.index)
    w2  = w.reindex(common).fillna(0.0)
    rx2 = rx.reindex(common)

    gross_exp = w2.abs().sum(axis=1).clip(lower=1e-12)
    port_ret  = (w2 * rx2).sum(axis=1)
    edge_bp   = float((port_ret / gross_exp).mean() * 1e4)

    turnover = float(w2.diff().abs().sum(axis=1).mean())
    breakeven_bp = float(edge_bp / turnover) if turnover > 1e-12 else np.nan

    result: dict = {
        "edge_bp":      edge_bp,
        "turnover":     turnover,
        "breakeven_bp": breakeven_bp,
    }

    if tc_bp is not None:
        pair_edge: dict[str, float] = {}
        pair_be:   dict[str, float] = {}
        for col in w2.columns:
            ge = w2[col].abs().clip(lower=1e-12)
            pr = w2[col] * rx2[col] if col in rx2.columns else pd.Series(dtype=float)
            if pr.empty:
                pair_edge[col] = np.nan
                pair_be[col]   = np.nan
                continue
            e  = float((pr / ge).mean() * 1e4)
            tv = float(w2[col].diff().abs().mean())
            pair_edge[col] = e
            pair_be[col]   = float(e / tv) if tv > 1e-12 else np.nan
        result["pair_edge_bp"]      = pair_edge
        result["pair_breakeven_bp"] = pair_be

    return result


# ── Full signal builder ───────────────────────────────────────────────────────

def build_signal_r(
    s_d: pd.DataFrame,
    fd_1m_m: pd.DataFrame,
    rx_d: pd.DataFrame | None = None,
    atm_1m_m: pd.DataFrame | None = None,
    rr25_1m_m: pd.DataFrame | None = None,
    carry_w_m: pd.DataFrame | None = None,
    ks: tuple = (5, 10, 20),
    hs: tuple = (5, 20),
    modes: tuple = ("spot", "total", "carry_demeaned"),
    thresh_list: tuple = (1.5, 2.0, 2.5),
    vol_window: int = 60,
    tc_bp: dict | None = None,
) -> dict:
    """
    Build all Family-R mean-reversion signals.

    Parameters
    ----------
    s_d         log spot FCU/USD, daily date × pair
    fd_1m_m     log 1M forward discount, month-end date × pair (reindexed internally)
    rx_d        daily excess returns (optional; required for §8 gate)
    atm_1m_m    ATM 1M vol, month-end date × pair (optional; for implied vol regime)
    rr25_1m_m   foreign-oriented 25Δ RR, month-end × pair (optional; risk appetite)
    carry_w_m   carry portfolio weights, month-end × pair (optional; contamination corr)
    ks          lookback periods in business days
    hs          holding periods in business days
    modes       ranking modes for R1
    thresh_list large-move thresholds
    vol_window  rolling window for trailing realised vol
    tc_bp       per-pair round-trip costs in bp (for §8 gate)

    Returns
    -------
    dict with keys:
      'r1_{mode}_k{k}_h{h}'                     — R1 weights (daily × pair)
      'r2_k{k}_h{h}'                            — R2 vol-scaled weights
      'r2_unscaled_k{k}_h{h}'                   — R2 unnormalised (comparison)
      'r1_{mode}_cond_lm{thresh:.1f}_k{k}_h{h}' — R1 large-move conditioned
      'vol_regime_realised'                      — expanding vol percentile (daily)
      'vol_regime_implied'                       — same from ATM (if atm_1m_m given)
      'rr_risk_appetite'                         — cross-sectional mean RR (if given)
      'r2_net_dollar_k{k}_h{h}'                 — R2 net long exposure diagnostic
      'carry_contamination_corr'                 — corr(total-R1, spot-R1) at k=10
      'diagnostics'                              — §8 gate per signal (if rx_d given)
      'mask_hit_rate'                            — fraction of pair-days passing each mask
    """
    # reindex fd_1m to daily (hold month-end value constant through the month)
    fd_d = fd_1m_m.reindex(s_d.index, method="ffill")

    vol_d = trailing_vol(s_d, window=vol_window)

    # ATM and RR reindexed to daily if available
    atm_d  = atm_1m_m.reindex(s_d.index, method="ffill") if atm_1m_m is not None and not atm_1m_m.empty else None
    rr_d   = rr25_1m_m.reindex(s_d.index, method="ffill") if rr25_1m_m is not None and not rr25_1m_m.empty else None
    c_d    = carry_w_m.reindex(s_d.index, method="ffill") if carry_w_m is not None and not carry_w_m.empty else None

    out: dict = {}

    # ── §2.1 Carry contamination check ───────────────────────────────────────
    try:
        pr_spot_10  = past_return(s_d, fd_d, k=10, mode="spot")
        pr_total_10 = past_return(s_d, fd_d, k=10, mode="total")
        w_spot_10   = xs_reversal(pr_spot_10)
        w_total_10  = xs_reversal(pr_total_10)

        ds = -s_d.diff()
        r_spot  = (w_spot_10  * ds).sum(axis=1).dropna()
        r_total = (w_total_10 * ds).sum(axis=1).dropna()
        common  = r_spot.index.intersection(r_total.index)
        if len(common) > 20:
            out["carry_contamination_corr"] = float(
                np.corrcoef(r_spot.reindex(common), r_total.reindex(common))[0, 1]
            )

        # also correlate total-ranked R1 against the monthly carry signal if given
        if c_d is not None:
            c_ret = (c_d * ds).sum(axis=1).dropna()
            t_ret = (w_total_10 * ds).sum(axis=1).dropna()
            com2  = c_ret.index.intersection(t_ret.index)
            if len(com2) > 20:
                out["total_vs_carry_corr"] = float(
                    np.corrcoef(c_ret.reindex(com2), t_ret.reindex(com2))[0, 1]
                )
    except Exception:
        pass

    mask_hit: dict[str, float] = {}

    # ── R1 and R2 across all k / h / mode ────────────────────────────────────
    for k in ks:
        pr_spot = past_return(s_d, fd_d, k=k, mode="spot")

        # ── R2 vol-scaled and unscaled ────────────────────────────────────
        w_r2 = ts_reversal(pr_spot, vol_d)
        # unscaled: just negate and cap to uniform ±1 per pair, then scale by N
        ds_k = -(s_d - s_d.shift(k))
        w_r2_unscaled = (-np.sign(ds_k)).clip(-1.0, 1.0)
        # net dollar exposure diagnostic
        for h in hs:
            key_vs = f"r2_k{k}_h{h}"
            key_us = f"r2_unscaled_k{k}_h{h}"
            out[key_vs] = apply_holding(w_r2, h)
            out[key_us] = apply_holding(w_r2_unscaled, h)
            out[f"r2_net_dollar_k{k}_h{h}"] = out[key_vs].sum(axis=1)

        # ── R1 per mode ───────────────────────────────────────────────────
        for mode in modes:
            pr = past_return(s_d, fd_d, k=k, mode=mode) if mode != "spot" else pr_spot
            w_base = xs_reversal(pr)

            for h in hs:
                out[f"r1_{mode}_k{k}_h{h}"] = apply_holding(w_base, h)

                # large-move conditioning
                for thresh in thresh_list:
                    mask = large_move_mask(pr, vol_d, k=k, thresh=thresh)
                    hit  = float(mask.stack().mean())
                    mask_hit[f"lm{thresh:.1f}_k{k}"] = hit

                    w_cond = _apply_mask_and_renorm(w_base, mask)
                    key = f"r1_{mode}_cond_lm{thresh:.1f}_k{k}_h{h}"
                    out[key] = apply_holding(w_cond, h)

    # ── §3.2 Vol regime percentile ────────────────────────────────────────────
    out["vol_regime_realised"] = vol_regime_percentile(vol_d, mode="realised")
    if atm_d is not None:
        out["vol_regime_implied"] = vol_regime_percentile(
            vol_d, atm_1m_d=atm_d, mode="implied"
        )

    # risk-appetite state (cross-sectional mean foreign-oriented RR)
    if rr_d is not None:
        out["rr_risk_appetite"] = rr_d.mean(axis=1)

    out["mask_hit_rate"] = mask_hit

    # ── §8 Economic significance gate ────────────────────────────────────────
    if rx_d is not None:
        diag: dict[str, dict] = {}
        for key, w in out.items():
            if not isinstance(w, pd.DataFrame):
                continue
            try:
                diag[key] = economic_significance(w, rx_d, tc_bp=tc_bp)
            except Exception:
                pass
        out["diagnostics"] = diag

    return out
