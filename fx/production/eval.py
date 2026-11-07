"""
run_fx_signals — evaluate the production FX signals.

Metrics per split (IS: ≤ IS_END, OOS: > IS_END):
  Sharpe     annualised
  Turnover   annual one-way  (mean |Δw| summed across pairs × ann_factor)
  MDD        maximum drawdown of cumulative PnL (return units)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SIGNALS_DIR = Path(__file__).parent.parent / "signals"
IS_END = "2020-12-31"

_SIGNALS: dict[str, dict] = {
    "Momentum XS (f=9m, h=1)": {
        "file": "signal_a_f9_h1.parquet",
        "returns": "rx_m",
        "freq": "monthly",
        "shift": True,   # weights at month-end t → hold month t+1
    },
    "MR NS-LOO (raw XS, monthly)": {
        "file": None,    # built on-the-fly from mr_ns_loo_signal.parquet
        "returns": "rx_d",
        "freq": "daily",
        "shift": False,
    },
    "MR NS-LOO (monthly VF-TS)": {
        "file": "mr_ns_loo_positions.parquet",
        "returns": "rx_d",
        "freq": "daily",
        "shift": False,  # VF positions at t → earn rx_d[t] (lag baked in)
    },
    "Carry Dupuy E-only (daily)": {
        "file": "signal_d_E_only_weights_daily.parquet",
        "returns": "rx_d",
        "freq": "daily",
        "shift": True,   # weights at close of t → hold day t+1
    },
    "Carry XS (monthly)": {
        "file": "view_xs_carry_m.parquet",
        "returns": "rx_m",
        "freq": "monthly",
        "shift": True,   # weights at month-end t → hold month t+1
    },
}


def _xs_positions(signal: pd.DataFrame, n: int = 4) -> pd.DataFrame:
    """Long top-n / short bottom-n equal-weight cross-sectional formation."""
    def _rank_row(row: pd.Series) -> pd.Series:
        valid = row.dropna()
        w = pd.Series(0.0, index=row.index)
        if len(valid) < 2 * n:
            return w
        ranked = valid.rank()
        w.loc[ranked[ranked >= len(valid) - n + 1].index] =  1.0 / n
        w.loc[ranked[ranked <= n].index]                  = -1.0 / n
        return w
    return signal.apply(_rank_row, axis=1)


def _build_raw_xs_monthly(sdir: Path, rx_d: pd.DataFrame) -> pd.DataFrame:
    """
    NS-LOO signal before VF-TS: raw z-score → simple XS n=4 at month-end,
    forward-filled to daily.
    """
    sig = pd.read_parquet(sdir / "mr_ns_loo_signal.parquet")
    sig_m = sig.resample("BME").last()
    pos_m = _xs_positions(sig_m, n=4)
    return pos_m.reindex(rx_d.index, method="ffill").fillna(0.0)


def _sharpe(pnl: pd.Series, freq: str) -> float:
    ann = 261 if freq == "daily" else 12
    r = pnl.dropna()
    if len(r) < 10 or r.std() == 0:
        return np.nan
    return float(r.mean() / r.std() * np.sqrt(ann))


def _turnover(w: pd.DataFrame, freq: str) -> float:
    """Annual one-way turnover in units of gross portfolio."""
    ann = 261 if freq == "daily" else 12
    return float(w.diff().abs().sum(axis=1).mean() * ann)


def _mdd(pnl: pd.Series) -> float:
    """Maximum drawdown of cumulative PnL (return units, negative number)."""
    cum = pnl.dropna().cumsum()
    if cum.empty:
        return np.nan
    return float((cum - cum.cummax()).min())


def run_fx_signals(
    is_end: str = IS_END,
    signals_dir: Path | None = None,
    print_table: bool = True,
) -> pd.DataFrame:
    """
    Evaluate the production FX signals.

    Returns a DataFrame with columns:
      IS_sharpe, IS_mdd, IS_turnover, OOS_sharpe, OOS_mdd, OOS_turnover
    """
    sdir = Path(signals_dir) if signals_dir else SIGNALS_DIR

    rx_m = pd.read_parquet(sdir / "returns.parquet")["rx_m"]
    rx_d = pd.read_parquet(sdir / "rx_d.parquet")

    rows = []
    for name, cfg in _SIGNALS.items():
        if cfg["file"] is None:
            w = _build_raw_xs_monthly(sdir, rx_d)
        else:
            w = pd.read_parquet(sdir / cfg["file"])

        rx   = rx_m if cfg["returns"] == "rx_m" else rx_d
        freq = cfg["freq"]

        w_applied = w.shift(1) if cfg["shift"] else w
        pnl = (w_applied * rx.reindex(columns=w.columns)).sum(axis=1)

        is_w    = w[w.index   <= is_end]
        oos_w   = w[w.index   >  is_end]
        is_pnl  = pnl[pnl.index <= is_end]
        oos_pnl = pnl[pnl.index >  is_end]

        rows.append({
            "signal":       name,
            "IS_sharpe":    _sharpe(is_pnl,  freq),
            "IS_mdd":       _mdd(is_pnl),
            "IS_turnover":  _turnover(is_w,  freq),
            "OOS_sharpe":   _sharpe(oos_pnl, freq),
            "OOS_mdd":      _mdd(oos_pnl),
            "OOS_turnover": _turnover(oos_w, freq),
        })

    df = pd.DataFrame(rows).set_index("signal")

    if print_table:
        _print(df, is_end)

    return df


def _print(df: pd.DataFrame, is_end: str) -> None:
    hdr = f"{'Signal':<36}  {'IS':>22}   {'OOS':>22}"
    sub = f"{'':36}  {'Sharpe':>6}  {'MDD':>7}  {'TO':>5}   {'Sharpe':>6}  {'MDD':>7}  {'TO':>5}"
    sep = "-" * len(sub)
    print(f"\n{hdr}")
    print(sub)
    print(sep)
    for name, row in df.iterrows():
        def _fmt_sr(v):  return f"{v:+.2f}" if not np.isnan(v) else "  N/A"
        def _fmt_mdd(v): return f"{v*100:+.1f}%" if not np.isnan(v) else "   N/A"
        def _fmt_to(v):  return f"{v:.1f}x" if not np.isnan(v) else " N/A"
        print(
            f"{name:<36}  "
            f"{_fmt_sr(row.IS_sharpe):>6}  {_fmt_mdd(row.IS_mdd):>7}  {_fmt_to(row.IS_turnover):>5}   "
            f"{_fmt_sr(row.OOS_sharpe):>6}  {_fmt_mdd(row.OOS_mdd):>7}  {_fmt_to(row.OOS_turnover):>5}"
        )
    print(sep)
    print(f"  IS ≤ {is_end}   OOS > {is_end}   MDD in return units   TO = annual one-way\n")
