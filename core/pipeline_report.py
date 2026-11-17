"""
Shared IS / OOS reporting helpers for the asset-class pipeline runners
(rates/pipeline.py, fx/pipeline.py, commodities/pipeline.py).

Conventions
  Sharpe    mean/std × √ann
  MDD       series pre-scaled to unit IS σ; compounded at `vol_target` (10% vol)
  Turnover  single position panels: annualised Σ|Δpos| / mean gross position
            composites: mean over days of Σ|Δ(scaled pos)| / Σ|scaled pos|, annualised
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

COLS = ["IS_SR", "IS_MDD", "IS_TO", "OOS_SR", "OOS_MDD", "OOS_TO"]
WIDTH = 92


@dataclass(frozen=True)
class ReportCfg:
    ann: int              # periods per year (252 daily, 12 monthly)
    is_end: str
    oos_start: str
    vol_target: float = 0.10
    min_obs: int = 30     # min observations for a Sharpe / for using IS σ

    @property
    def sub(self) -> str:
        return (f"IS ≤ {self.is_end}  |  OOS ≥ {self.oos_start}  |  "
                f"MDD at {self.vol_target*100:.0f}% vol")


# ── metrics ───────────────────────────────────────────────────────────────────

def sharpe(r: pd.Series, cfg: ReportCfg) -> float:
    r = r.dropna()
    if len(r) < cfg.min_obs or r.std() < 1e-12:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(cfg.ann))


def mdd(r_norm: pd.Series, cfg: ReportCfg) -> float:
    r = r_norm.dropna()
    if len(r) < 2 or r.std() < 1e-12:
        return float("nan")
    cum = (1 + r * cfg.vol_target).cumprod()
    return float((cum / cum.cummax() - 1).min() * 100)


def ann_sigma(r: pd.Series, cfg: ReportCfg) -> float:
    """IS σ (annualised); falls back to full-sample σ if IS has < cfg.min_obs obs."""
    r = r.dropna()
    is_r = r.loc[r.index <= cfg.is_end]
    return float((is_r if len(is_r) >= cfg.min_obs else r).std() * np.sqrt(cfg.ann))


def pos_turnover(pos: Optional[pd.DataFrame], idx: pd.Index, cfg: ReportCfg) -> float:
    if pos is None or len(idx) == 0:
        return float("nan")
    p = pos.loc[idx[0]:idx[-1]].dropna(how="all")
    if len(p) < 2:
        return float("nan")
    gross = p.abs().sum(axis=1).replace(0.0, np.nan).mean()
    if np.isnan(gross) or gross < 1e-12:
        return float("nan")
    return float(p.diff().dropna(how="all").abs().sum(axis=1).mean() * cfg.ann / gross)


def weight_positions(pos: pd.DataFrame, w: pd.Series) -> pd.DataFrame:
    """Scale each column by its market weight (level 0 of the columns); drop unweighted markets."""
    mk = pos.columns.get_level_values(0)
    out = pos.loc[:, mk.isin(w.index)]
    return out * w.reindex(out.columns.get_level_values(0)).values


def composite_turnover(
    w_df: pd.DataFrame,
    posw: dict[str, pd.DataFrame],
    sigma: pd.Series,
    idx: pd.Index,
    cfg: ReportCfg,
) -> float:
    """Turnover of Σ_k w_k(t)/σ_k · pos_k(t) over idx, with w_df the (T × K) blend weights."""
    idx = idx.intersection(w_df.dropna(how="any").index)
    if len(idx) == 0:
        return float("nan")
    w = w_df.loc[idx]
    w = w.div(w.abs().sum(axis=1).replace(0.0, np.nan), axis=0)
    delta = np.zeros(len(idx))
    gross = np.zeros(len(idx))
    for k in w.columns:
        if k not in posw or not np.isfinite(sigma.get(k, np.nan)):
            continue
        pk = posw[k].reindex(index=idx).fillna(0.0).values
        scaled = pk * w[k].values[:, None] / float(sigma[k])
        gross += np.abs(scaled).sum(axis=1)
        delta[1:] += np.abs(np.diff(scaled, axis=0)).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        to = np.where(gross > 1e-12, delta / gross, np.nan)
    to = to[~np.isnan(to)]
    return float(to.mean() * cfg.ann) if len(to) else float("nan")


def metrics(
    r_norm: pd.Series,
    to_fn: Callable[[pd.Index], float],
    cfg: ReportCfg,
) -> dict:
    """IS/OOS Sharpe, MDD, turnover for a unit-σ-normalised return series."""
    out = {}
    for tag, rp in (
        ("IS",  r_norm.loc[r_norm.index <= cfg.is_end]),
        ("OOS", r_norm.loc[r_norm.index >= cfg.oos_start]),
    ):
        rp = rp.dropna()
        out[f"{tag}_SR"]  = sharpe(rp, cfg)
        out[f"{tag}_MDD"] = mdd(rp, cfg)
        out[f"{tag}_TO"]  = to_fn(rp.index)
    return out


# ── printing ──────────────────────────────────────────────────────────────────

def header(title: str, label_w: int, sub: str = "") -> None:
    print("\n" + "=" * WIDTH)
    print(f"  {title}")
    if sub:
        print(f"  {sub}")
    print("=" * WIDTH)
    print(f"  {'':<{label_w}}  {'IS SR':>7}  {'IS MDD':>8}  {'IS TO':>7}"
          f"  {'OOS SR':>7}  {'OOS MDD':>8}  {'OOS TO':>8}")
    print("  " + "─" * (WIDTH - 2))


def print_row(label: str, row: dict, label_w: int) -> None:
    sr  = lambda v: f"{v:+.2f}" if np.isfinite(v) else "n/a"
    dd  = lambda v: f"{v:.1f}%" if np.isfinite(v) else "n/a"
    to  = lambda v: f"{v:.1f}x" if np.isfinite(v) else "n/a"
    print(f"  {label:<{label_w}}"
          f"  {sr(row['IS_SR']):>7}  {dd(row['IS_MDD']):>8}  {to(row['IS_TO']):>7}"
          f"  {sr(row['OOS_SR']):>7}  {dd(row['OOS_MDD']):>8}  {to(row['OOS_TO']):>8}")


def print_tc(tc_df: pd.DataFrame, title: str, cfg: ReportCfg, label_w: int = 30) -> None:
    """Print a tc_sweep_* metrics frame ([Method, Scenario, Period, Sharpe, MDD_pct, TO])."""
    header(title, label_w, cfg.sub)
    for method in ("Fixed weights", "Adaptive Ridge"):
        print(f"  [{method}]")
        for name in dict.fromkeys(tc_df["Scenario"]):
            m = tc_df[(tc_df["Method"] == method) & (tc_df["Scenario"] == name)]
            if m.empty:
                continue
            row = {}
            for tag in ("IS", "OOS"):
                x = m[m["Period"] == tag]
                row[f"{tag}_SR"]  = float(x["Sharpe"].iloc[0])  if len(x) else float("nan")
                row[f"{tag}_MDD"] = float(x["MDD_pct"].iloc[0]) if len(x) else float("nan")
                row[f"{tag}_TO"]  = float(x["TO"].iloc[0])      if len(x) else float("nan")
            print_row(f"  {name}", row, label_w)
    print("=" * WIDTH)


def to_frame(d: dict, names: list[str]) -> pd.DataFrame:
    out = pd.DataFrame.from_dict(d, orient="index")[COLS]
    if len(out) and isinstance(out.index[0], tuple):
        out.index = pd.MultiIndex.from_tuples(out.index, names=names)
    else:
        out.index.name = names[0]
    return out
