"""
Rates production signal evaluation.

Computes Sharpe, annualised turnover, and max drawdown in IS and OOS for
each of the three production signals (momentum, carry, mean_reversion).

IS  : start → 2018-12-31
OOS : 2019-01-01 → latest

Usage:
    from rates.evaluate_signals import run_rates_signals

    df = run_rates_signals()                          # production, prints table
    df = run_rates_signals(vol_scale_carry_mr=True)   # test: vol-scale carry & MR
    df = run_rates_signals(dv01_neutral_mom=True)     # test: DV01-neutralise momentum
    df = run_rates_signals(print_results=False)       # silent, returns DataFrame

Vol-scaling carry and MR (vol_scale_carry_mr=True):
    Applies a trailing EWMA vol normalisation to carry and mean_reversion.
    Momentum is left unchanged (already vol-scaled at construction).
    Scale is capped at 5× to prevent extreme leverage in low-vol periods.

DV01 neutralisation for momentum (dv01_neutral_mom=True):
    For each market on each date, projects out the DV01 component from the
    position vector across tenors {2Y, 5Y, 10Y}, then rescales to preserve
    gross exposure.  DV01 approximated as tenor (years).  PnL is recomputed
    exactly from neutralised positions × ret_d (momentum is the only signal
    where this is possible without re-running the full pipeline).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

IS_END    = "2018-12-31"
OOS_START = "2019-01-01"
MIN_OBS   = 30
VOL_SCALE_CAP = 5.0

_SIGNALS_DIR = Path(__file__).parent / "signals"

_PNL_FILES = {
    "momentum":       "momentum_a_daily_pnl.parquet",
    "carry":          "carry_ois_par_pnl.parquet",
    "mean_reversion": "mean_reversion_pnl.parquet",
}

_POS_FILES = {
    "momentum":       "momentum_a_daily.parquet",
    "carry":          "carry_ois_par.parquet",
    "mean_reversion": "mean_reversion.parquet",
}


# ── Metric helpers ────────────────────────────────────────────────────────────

def _sharpe(pnl: pd.Series) -> float:
    s = pnl.dropna()
    if len(s) < MIN_OBS or s.std() < 1e-12:
        return float("nan")
    return float(s.mean() / s.std() * np.sqrt(252))


def _max_drawdown(pnl: pd.Series) -> float:
    """Max drawdown in vol units (MDD / ann_vol) — comparable across signals."""
    s = pnl.dropna()
    if len(s) < 2:
        return float("nan")
    ann_vol = s.std() * np.sqrt(252)
    if ann_vol < 1e-12:
        return float("nan")
    cum = s.cumsum()
    return float((cum - cum.cummax()).min() / ann_vol)


def _annualised_turnover(pos: pd.DataFrame, period: slice) -> float:
    """Annualised portfolio turns: sum(|Δpos|) / avg_gross × 252 (dimensionless)."""
    p = pos.loc[period].dropna(how="all")
    if len(p) < 2:
        return float("nan")
    daily_to  = p.diff().abs().sum(axis=1)
    avg_gross = p.abs().sum(axis=1).replace(0, np.nan).mean()
    if avg_gross < 1e-12 or np.isnan(avg_gross):
        return float("nan")
    return float(daily_to.mean() / avg_gross * 252)


# ── Vol-scaling ───────────────────────────────────────────────────────────────

def _vol_scale(
    pnl_df: pd.DataFrame,
    pos_df: pd.DataFrame,
    halflife: int = 21,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply EWMA vol-normalisation to a signal's PnL and positions.

    scale_t = target_vol / ewma_vol_{t-1}   (lagged 1 day, capped at VOL_SCALE_CAP).
    Target is anchored to the full-sample EWMA vol so the overall level is preserved.

    Because the scale is a uniform scalar across all positions on a given day,
    scaled_pnl_t = pnl_t × scale_{t-1} is exact.
    """
    port = pnl_df.sum(axis=1)

    ewma_vol   = port.ewm(halflife=halflife).std().shift(1)
    target_vol = port.ewm(halflife=halflife).std().iloc[-1]   # full-sample anchor
    scale      = (target_vol / ewma_vol).clip(upper=VOL_SCALE_CAP)

    scaled_pnl = pnl_df.multiply(scale, axis=0)
    scaled_pos = pos_df.multiply(scale, axis=0)

    return scaled_pnl, scaled_pos


# ── DV01 neutralisation ───────────────────────────────────────────────────────

def _dv01_neutralise_momentum(
    pos_df: pd.DataFrame,
    sig_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Project out the within-country DV01 component from momentum positions.

    For each market, on each date, the position vector across tenors is
    projected onto the complement of the DV01 vector:
        pos_neutral = pos - (pos·dv01 / dv01·dv01) × dv01
    then rescaled to preserve the original gross exposure.

    DV01 ≈ tenor (years) — standard first-order approximation.

    PnL is recomputed exactly: pos_neutral.shift(1) × ret_d, summed per market.
    Returns (neutralised_pos_df, neutralised_pnl_df).
    """
    ret_d = pd.read_parquet(sig_dir / "ret_d.parquet")
    markets = pos_df.columns.get_level_values("market").unique()

    new_pos = pos_df.copy().astype(float)

    for mkt in markets:
        mkt_pos = pos_df[mkt].to_numpy(float)        # (T, 3)
        tenors  = pos_df[mkt].columns.to_numpy(float) # [2., 5., 10.]
        dv01    = tenors                               # DV01 ≈ tenor

        dd = np.dot(dv01, dv01)
        # Projection: remove the DV01 component row-by-row (vectorised)
        proj_coef         = mkt_pos @ dv01 / dd        # (T,)
        pos_neutral       = mkt_pos - np.outer(proj_coef, dv01)

        # Rescale each row to preserve original gross exposure
        gross_orig = np.abs(mkt_pos).sum(axis=1)        # (T,)
        gross_new  = np.abs(pos_neutral).sum(axis=1)    # (T,)
        # avoid divide-by-zero on rows where neutralisation kills all exposure
        scale = np.where(gross_new > 1e-12, gross_orig / gross_new, 0.0)
        pos_neutral *= scale[:, None]

        # Restore NaN rows (dates where original pos was all-NaN)
        nan_rows = ~np.isfinite(mkt_pos).all(axis=1)
        pos_neutral[nan_rows] = np.nan

        new_pos[mkt] = pos_neutral

    # Recompute PnL exactly: pos_neutral.shift(1) × ret_d, sum across tenors per market
    common_cols = new_pos.columns.intersection(ret_d.columns)
    common_idx  = new_pos.index.intersection(ret_d.index)
    pnl_per_tenor = (
        new_pos.loc[common_idx, common_cols].shift(1)
        * ret_d.loc[common_idx, common_cols]
    )
    pnl_df = pnl_per_tenor.T.groupby(level="market").sum().T
    pnl_df.index.name = "date"

    return new_pos, pnl_df


# ── Main evaluation ───────────────────────────────────────────────────────────

def run_rates_signals(
    signals_dir: str | Path | None = None,
    vol_scale_carry_mr: bool = False,
    vol_halflife: int = 21,
    dv01_neutral_mom: bool = False,
    print_results: bool = True,
) -> pd.DataFrame:
    """
    Evaluate each production rates signal in IS and OOS.

    Parameters
    ----------
    vol_scale_carry_mr
        If True, apply EWMA vol-normalisation to carry and mean_reversion.
        Momentum is unchanged (already vol-scaled at construction).
    vol_halflife
        Half-life in days for the EWMA vol estimator (default 21).
    dv01_neutral_mom
        If True, project out the within-country DV01 component from momentum
        positions before computing metrics.  PnL is recomputed from neutralised
        positions × ret_d.

    Returns
    -------
    DataFrame indexed by signal with columns:
        IS_sharpe, IS_turnover, IS_maxdd, OOS_sharpe, OOS_turnover, OOS_maxdd
    """
    sig_dir   = Path(signals_dir) if signals_dir else _SIGNALS_DIR
    is_slice  = slice(None, IS_END)
    oos_slice = slice(OOS_START, None)

    rows: dict[str, dict] = {}
    for name in _PNL_FILES:
        pnl_df = pd.read_parquet(sig_dir / _PNL_FILES[name])
        pos_df = pd.read_parquet(sig_dir / _POS_FILES[name])

        if vol_scale_carry_mr and name in ("carry", "mean_reversion"):
            pnl_df, pos_df = _vol_scale(pnl_df, pos_df, halflife=vol_halflife)

        if dv01_neutral_mom and name == "momentum":
            pos_df, pnl_df = _dv01_neutralise_momentum(pos_df, sig_dir)

        port = pnl_df.sum(axis=1)

        rows[name] = {
            "IS_sharpe":    _sharpe(port.loc[is_slice]),
            "IS_turnover":  _annualised_turnover(pos_df, is_slice),
            "IS_maxdd":     _max_drawdown(port.loc[is_slice]),
            "OOS_sharpe":   _sharpe(port.loc[oos_slice]),
            "OOS_turnover": _annualised_turnover(pos_df, oos_slice),
            "OOS_maxdd":    _max_drawdown(port.loc[oos_slice]),
        }

    result = pd.DataFrame(rows).T
    result.index.name = "signal"

    if print_results:
        _print_table(result, vol_scale_carry_mr, vol_halflife, dv01_neutral_mom)

    return result


# ── Printing ──────────────────────────────────────────────────────────────────

def _print_table(
    df: pd.DataFrame,
    vol_scaled: bool = False,
    halflife: int = 21,
    dv01_neutral_mom: bool = False,
) -> None:
    parts = []
    if dv01_neutral_mom:
        parts.append("DV01-neutral mom")
    if vol_scaled:
        parts.append(f"vol-scaled carry & MR (hl={halflife}d)")
    label = " + ".join(parts) if parts else "PRODUCTION"
    print("\n" + "=" * 72)
    print(f"  RATES PRODUCTION SIGNALS — {label}")
    print(f"  IS: start → {IS_END}   |   OOS: {OOS_START} → latest")
    print("=" * 72)

    header = f"  {'Signal':<18}  {'IS SR':>7}  {'IS TO':>8}  {'IS MDD':>9}  {'OOS SR':>7}  {'OOS TO':>8}  {'OOS MDD':>9}"
    print(header)
    print("  " + "-" * 68)

    def _fmt_sr(v): return f"{v:+.2f}" if not np.isnan(v) else "   n/a"
    def _fmt_to(v): return f"{v:.2f}"  if not np.isnan(v) else "   n/a"
    def _fmt_dd(v): return f"{v:+.2f}" if not np.isnan(v) else "   n/a"

    for sig, row in df.iterrows():
        print(
            f"  {sig:<18}  {_fmt_sr(row.IS_sharpe):>7}  {_fmt_to(row.IS_turnover):>8}  "
            f"{_fmt_dd(row.IS_maxdd):>9}  {_fmt_sr(row.OOS_sharpe):>7}  "
            f"{_fmt_to(row.OOS_turnover):>8}  {_fmt_dd(row.OOS_maxdd):>9}"
        )

    print()
    print("  Sharpe: annualised (sqrt252). Turnover: portfolio turns/year (sum|Dpos|/avg_gross x 252).")
    print("  Max drawdown: vol units (MDD / ann_vol) — dimensionless.")
    print("=" * 72)


if __name__ == "__main__":
    run_rates_signals()
    print()
    run_rates_signals(dv01_neutral_mom=True)
