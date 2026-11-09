"""
Commodities signal blend layer.

Loads pre-computed vol-scaled managed-return parquets, and exposes the same
blend API as equities/blend.py (all algorithms are re-exported).

Pipeline position:
    commodities/signals/*_pnl.parquet  →  load_managed_returns()  →  F (T × 4)
    F  →  fit_static / MLMSRR / LinearDynamicBlend / …  →  blend weights
    blend weights  →  combined portfolio return

Signals (production, from commodities/signals/):
    tsmom_252        tsmom_252_pnl.parquet        TS momentum 252-day (sign)
    xs_carry         xs_carry_pnl.parquet          XS tenor-normalised F1/F2 carry
    xs_carry_enh     xs_carry_enh_pnl.parquet      XS carry + seasonal/sector/roll
    curvature        curvature_pnl.parquet         NS butterfly (F1+F4-2*F2)

Managed-return convention: for each signal k and commodity i,
    w_k[i,t] = signal_k[i,t] × target_vol / sigma_hat[i,t]   (vol-targeted)
    F_k[t]   = Σ_i  w_k[i,t-1] × r[i,t]                     (weekly W-FRI rebalance)

The NS null-space butterfly (Signal B) is excluded: its daily vol is ~10× lower
than directional signals; the vol-scaling step would apply ~10× leverage, making
its managed return non-comparable and its blend weight trivially dominant.

Parquets are pre-computed by build_signal_parquets(); loading them takes ~1 ms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Re-export every blend algorithm from the equities layer ──────────────────
from equities.blend import (                                        # noqa: F401
    BlendWeights,
    equal_weight,
    inv_vol_blend,
    erc_blend,
    hrp_blend,
    sharpe_heuristic,
    msrr_ridge,
    msrr_lasso,
    combine_signals,
    build_managed_returns,
    build_state_features,
    enet_blend,
    c_enet_blend,
    e_enet_blend,
    inv_msfe_blend,
    pcr_blend,
    pls_blend,
    winsorize_signals,
    pool_blend,
    fit_static,
    MLMSRR,
    LinearDynamicBlend,
    GBRTBlend,
    RandomForestBlend,
    DMABlend,
    STMOM,
)

# ── Commodities-specific constants ────────────────────────────────────────────

IS_END = "2020-12-31"

_SIGNALS_DIR = Path(__file__).parent / "signals"

SIGNAL_FILES: dict[str, str] = {
    "tsmom_252":    "tsmom_252_pnl.parquet",
    "xs_carry_enh": "xs_carry_enh_pnl.parquet",
    "curvature":    "curvature_pnl.parquet",
}

# Production pipeline parameters (must match those used in build_signal_parquets).
# Subset of backtest.PARAMS relevant to the managed-return computation.
BLEND_PARAMS = dict(
    asset_vol = 0.10,
    vol_com   = 60,
    rebal     = "W-FRI",
    lag       = 1,
    mode      = "sign",
    clip      = 2.0,
)


# ── One-time parquet builder ──────────────────────────────────────────────────

def build_signal_parquets(
    output_dir: str | Path | None = None,
    params: dict | None = None,
) -> None:
    """
    Compute vol-scaled managed returns for all four production signals and save
    them to parquet.  Call once; subsequent load_managed_returns() calls read
    from disk in ~1 ms.

    Parameters
    ----------
    output_dir  Directory for parquet output.  Default: commodities/signals/.
    params      Pipeline parameters.  Default: BLEND_PARAMS.
    """
    from commodities.data import fetch_prices, fetch_term_structure, fetch_tickers
    from commodities.backtest import (
        specific_contract_returns, vol_scale, rebalance, _build_signals,
    )

    p = params or BLEND_PARAMS
    out_dir = Path(output_dir) if output_dir else _SIGNALS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Bloomberg data …")
    prices  = fetch_prices()
    ts      = fetch_term_structure()
    tk      = fetch_tickers()

    returns = specific_contract_returns(ts, tk).reindex(columns=prices.columns)
    raw_signals = _build_signals(prices, ts, tk, p)

    # Map from _build_signals keys → SIGNAL_FILES keys
    _key_map = {
        "tsmom_252":         "tsmom_252",
        "xs_carry_enh_f1f2": "xs_carry_enh",
        "curvature":         "curvature",
    }

    print("Computing managed returns …")
    for raw_key, blend_key in _key_map.items():
        if raw_key not in raw_signals:
            print(f"  Skipping {raw_key} (not available)")
            continue
        raw_sig = raw_signals[raw_key]
        sig = raw_sig.reindex(index=returns.index, columns=returns.columns)
        w = vol_scale(sig, returns, target_vol=p["asset_vol"], com=p["vol_com"])
        w = rebalance(w, rule=p["rebal"])
        # Managed return: w[t-1] × r[t], zero rows dropped
        F_k = (w.shift(p["lag"]) * returns).sum(axis=1)
        F_k = F_k.replace(0.0, np.nan).dropna()
        F_k.index.name = "date"
        F_k.name = blend_key
        path = out_dir / SIGNAL_FILES[blend_key]
        F_k.to_frame().to_parquet(path)
        sr = F_k.mean() / F_k.std() * np.sqrt(252)
        print(f"  {blend_key:15s}  SR={sr:+.3f}  rows={len(F_k)}  → {path.name}")

    print("Done.")


# ── Data loading ─────────────────────────────────────────────────────────────

def load_managed_returns(
    signals: dict[str, str] | None = None,
    signals_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load commodities managed-return parquets and return F (T × K).

    Parameters
    ----------
    signals     Map of signal_name → parquet filename.  Default: SIGNAL_FILES.
    signals_dir Directory containing the parquet files.  Default: commodities/signals/.

    Returns
    -------
    pd.DataFrame, shape (T, K), index=date, columns=signal names.
    Only dates where all signals are available are returned (inner join).

    Raises
    ------
    FileNotFoundError if parquets are missing.  Run build_signal_parquets() first.
    """
    if signals is None:
        signals = SIGNAL_FILES
    sig_dir = Path(signals_dir) if signals_dir else _SIGNALS_DIR

    series: dict[str, pd.Series] = {}
    for name, fname in signals.items():
        path = sig_dir / fname
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found.  Run commodities.blend.build_signal_parquets() first."
            )
        df = pd.read_parquet(path)
        series[name] = df.iloc[:, 0]

    F = pd.DataFrame(series)
    F.index.name = "date"
    F = F.sort_index().dropna(how="any")
    return F


# ── Portfolio-level combination ───────────────────────────────────────────────

def combined_return(
    F: pd.DataFrame,
    weights: "BlendWeights",
) -> pd.Series:
    """
    Apply static blend weights to the managed-return matrix.

    R_t = Σ_k  w_k · F_k[t]

    Returns a pd.Series of daily portfolio returns.
    """
    w = weights.weights
    r = pd.Series(0.0, index=F.index)
    for k in w.index:
        if k in F.columns:
            r = r + w[k] * F[k]
    r.name = "combined"
    return r


# ── Rolling OLS blend ─────────────────────────────────────────────────────────

def _expanding_ols_weights(F: pd.DataFrame, min_obs: int = 252) -> pd.DataFrame:
    """
    Expanding-window MSRR OLS weights (causal).

    At row t, the weight is fitted on F.iloc[:t] (all rows strictly before t),
    so applying it directly to F.iloc[t] is look-ahead free.
    Rows 0..min_obs-1 are NaN (insufficient history).
    """
    vals = F.values.astype(float)
    T, K = vals.shape
    out = np.full((T, K), np.nan)
    ones = np.ones(min_obs)
    for t in range(min_obs, T):
        Ft = vals[:t]
        ones_t = np.ones(t)
        b, _, _, _ = np.linalg.lstsq(Ft, ones_t, rcond=None)
        out[t] = b
    return pd.DataFrame(out, index=F.index, columns=F.columns)


def run_comm_blend(
    signals_dir: str | Path | None = None,
    is_end: str = IS_END,
    min_obs: int = 252,
) -> dict:
    """
    Canonical commodities blend pipeline: daily managed returns → rolling OLS.

    Vol-normalises to unit annual vol using IS σ only (no look-ahead), then fits
    an expanding-window MSRR OLS blend where weights at day t are estimated on
    all data strictly before t.

    Parameters
    ----------
    signals_dir   Directory with parquet files.  Default: commodities/signals/.
    is_end        Last IS date.  Default: IS_END ("2020-12-31").
    min_obs       Minimum observations before computing blend weights.  Default: 252.

    Returns
    -------
    dict:
        F_is        vol-normalised IS managed returns  (T_is × 3)
        F_oos       vol-normalised OOS managed returns (T_oos × 3)
        weights_is  pd.DataFrame (T_is × 3) expanding OLS weights, IS slice
        weights_oos pd.DataFrame (T_oos × 3) expanding OLS weights, OOS slice
        r_is        combined IS return series
        r_oos       combined OOS return series
    """
    F = load_managed_returns(signals_dir=signals_dir)
    F_is_raw  = F.loc[F.index <= is_end]
    F_oos_raw = F.loc[F.index >  is_end]

    sigma = F_is_raw.std() * np.sqrt(252)
    sigma = sigma.replace(0.0, np.nan)
    F_is  = F_is_raw.div(sigma)
    F_oos = F_oos_raw.div(sigma)

    # Expanding OLS over the full history (IS then OOS) — fully causal
    F_full = pd.concat([F_is, F_oos])
    w_all  = _expanding_ols_weights(F_full, min_obs=min_obs)

    # L1-normalise each row so total gross = 1 (matches _dyn_blend_ret convention)
    norm  = w_all.abs().sum(axis=1).replace(0.0, np.nan)
    w_all = w_all.div(norm, axis=0)

    w_is  = w_all.loc[F_is.index].dropna()
    w_oos = w_all.loc[F_oos.index].dropna()

    r_is  = (F_is.loc[w_is.index]   * w_is).sum(axis=1)
    r_oos = (F_oos.loc[w_oos.index] * w_oos).sum(axis=1)
    r_is.name  = "comm_rolling_ols"
    r_oos.name = "comm_rolling_ols"

    return {
        "F_is":        F_is,
        "F_oos":       F_oos,
        "weights_is":  w_is,
        "weights_oos": w_oos,
        "r_is":        r_is,
        "r_oos":       r_oos,
    }


def run_commodities_blend(
    signals_dir: str | Path | None = None,
    is_end: str = IS_END,
    min_obs: int = 252,
    rebuild_signals: bool = False,
) -> dict:
    """
    Full commodities pipeline: build signals → rolling OLS blend.

    Step 1 — Signal pipeline (commodities/blend.build_signal_parquets):
        Computes vol-scaled managed returns for tsmom_252, xs_carry_enh, and
        curvature and writes them to commodities/signals/.  Skipped when
        rebuild_signals=False and all three parquets already exist.

    Step 2 — Blend (run_comm_blend):
        Loads the three parquets, vol-normalises on IS σ, fits an
        expanding-window MSRR OLS blend (causal, daily rebalance).

    Parameters
    ----------
    signals_dir       Parquet directory.  Default: commodities/signals/.
    is_end            Last IS date.  Default: IS_END ("2020-12-31").
    min_obs           Minimum obs before blend weights are non-NaN.  Default: 252.
    rebuild_signals   Force re-run of the signal pipeline.  Default: False.

    Returns
    -------
    Same dict as run_comm_blend().
    """
    sig_dir   = Path(signals_dir) if signals_dir else _SIGNALS_DIR
    pnl_paths = [sig_dir / f for f in SIGNAL_FILES.values()]
    needs_build = rebuild_signals or not all(p.exists() for p in pnl_paths)

    if needs_build:
        missing = [p.name for p in pnl_paths if not p.exists()]
        if missing:
            print(f"Missing parquets: {missing} — running signal pipeline...")
        else:
            print("rebuild_signals=True — re-running signal pipeline...")
        build_signal_parquets(output_dir=sig_dir)
    else:
        print("Signal parquets found — skipping rebuild. Pass rebuild_signals=True to refresh.")

    return run_comm_blend(signals_dir=sig_dir, is_end=is_end, min_obs=min_obs)


# ── Production entry point ────────────────────────────────────────────────────

def run_comm_prod(
    is_end: str = IS_END,
    params: dict | None = None,
    min_obs: int = 252,
    rebuild_signals: bool = False,
    vol_target: float = 0.10,
) -> pd.Series:
    """
    Production commodities pipeline: 3 signals → rolling OLS blend.

    Loads (or rebuilds) signal parquets, fits an expanding-window MSRR OLS
    blend, prints IS / OOS Sharpe, MDD, and annual turnover, and returns the
    full daily return series (IS concatenated with OOS).

    Parameters
    ----------
    is_end            Last IS date.  Default: IS_END ("2020-12-31").
    params            Pipeline parameters.  Default: commodities/backtest.PARAMS.
    min_obs           Warmup before blend weights are non-NaN.  Default: 252.
    rebuild_signals   Force rebuild of signal parquets.  Default: False.

    Returns
    -------
    pd.Series — daily return series named "commodities", IS then OOS.
    """
    from commodities.backtest import (
        PARAMS as _PARAMS, specific_contract_returns, _build_signals,
        vol_scale, rebalance, book_scale,
    )
    from commodities.data import fetch_prices, fetch_term_structure, fetch_tickers

    TD = 252
    p  = params or _PARAMS

    # ── 1. Blend ──────────────────────────────────────────────────────────────
    res   = run_commodities_blend(is_end=is_end, min_obs=min_obs,
                                  rebuild_signals=rebuild_signals)
    r_is  = res["r_is"]
    r_oos = res["r_oos"]
    w_is  = res["weights_is"]   # T_is × 3  (L1-normalised blend weights)
    w_oos = res["weights_oos"]  # T_oos × 3

    # ── 2. Instrument-level weights for turnover ──────────────────────────────
    prices   = fetch_prices()
    ts_data  = fetch_term_structure()
    tk       = fetch_tickers()
    returns  = specific_contract_returns(ts_data, tk).reindex(columns=prices.columns)
    raw_sigs = _build_signals(prices, ts_data, tk, p)

    _name_map = {
        "tsmom_252":         "tsmom_252",
        "xs_carry_enh_f1f2": "xs_carry_enh",
        "curvature":         "curvature",
    }
    w_k: dict[str, pd.DataFrame] = {}
    for raw_key, blend_key in _name_map.items():
        if raw_key not in raw_sigs:
            continue
        sig = raw_sigs[raw_key].reindex(index=returns.index, columns=returns.columns)
        w   = vol_scale(sig, returns, target_vol=p["asset_vol"], com=p["vol_com"])
        w   = rebalance(w, rule=p["rebal"])
        w   = book_scale(w, returns, target_vol=p["book_vol"],
                         cov_hl=p["cov_hl"], k_max=p["k_max"])
        w_k[blend_key] = w

    # ── 3. Blended instrument weights ─────────────────────────────────────────
    names    = list(w_is.columns)
    all_idx  = w_k[names[0]].index
    all_cols = w_k[names[0]].columns
    ts_cut   = pd.Timestamp(is_end)

    alpha_full = pd.concat([w_is, w_oos]).sort_index()
    a_df       = (alpha_full
                  .reindex(all_idx).ffill().fillna(0.0))
    norms      = a_df.abs().sum(axis=1).replace(0.0, np.nan)
    a_norm     = a_df.div(norms, axis=0).fillna(0.0)

    W_blend = pd.DataFrame(0.0, index=all_idx, columns=all_cols)
    for k in names:
        if k in w_k:
            W_blend += (w_k[k]
                        .reindex(index=all_idx, columns=all_cols)
                        .fillna(0.0)
                        .mul(a_norm[k], axis=0))

    to_is  = float(W_blend.loc[W_blend.index <= ts_cut].diff().abs().sum(axis=1).mean() * TD)
    to_oos = float(W_blend.loc[W_blend.index >  ts_cut].diff().abs().sum(axis=1).mean() * TD)

    # ── 4. Performance metrics ────────────────────────────────────────────────
    def _sr(r: pd.Series) -> float:
        r = r.dropna()
        return float(r.mean() / r.std() * np.sqrt(TD)) if len(r) > 60 else float("nan")

    def _mdd(r: pd.Series) -> float:
        r = r.dropna()
        if r.empty:
            return float("nan")
        r_scaled = r * vol_target
        cum = (1 + r_scaled).cumprod()
        return float((cum / cum.cummax() - 1).min())

    # ── 5. Print ──────────────────────────────────────────────────────────────
    W = 72
    print("=" * W)
    print(f"  COMMODITIES PRODUCTION  —  Rolling OLS blend  (vol target={vol_target*100:.0f}%)")
    print(f"  IS ≤ {is_end}   |   OOS > {is_end}")
    print("=" * W)
    print(f"  {'':20}  {'SR':>7}  {'MDD':>8}  {'TO (ann)':>9}")
    print(f"  {'─'*20}  {'─'*7}  {'─'*8}  {'─'*9}")
    for label, r, to in [("IS", r_is, to_is), ("OOS", r_oos, to_oos)]:
        sr  = _sr(r)
        mdd = _mdd(r)
        print(f"  {label:<20}  {sr:>+7.2f}  {mdd*100:>+7.1f}%  {to:>8.1f}x")
    print("=" * W)
    print()
    print("  Avg blend weights (IS):")
    for col in names:
        print(f"    {col:<20}  {w_is[col].mean():>+.4f}")
    print()

    # ── 6. Return series ──────────────────────────────────────────────────────
    r_full = pd.concat([r_is, r_oos]).sort_index()
    r_full.name = "commodities"
    return r_full


from equities.blend import RollingMSRR as _RollingMSRR   # noqa: E402


def compare_blend_methods(
    signals_dir: str | Path | None = None,
    is_end: str = IS_END,
    lookback: int = 252,
    min_obs: int = 252,
    ridge_lam: float = 0.01,
    vol_target: float = 0.10,
) -> pd.DataFrame:
    """
    Compare three commodities blend methods side-by-side: Sharpe, MDD, TO.

    Methods
    -------
    Rolling OLS MSRR   fixed lookback=252d rolling window (note: production uses expanding OLS)
    Rolling Ridge MSRR same rolling window with L2 penalty (lam=ridge_lam)
    Static Ridge MSRR  IS-fit once with ridge, constant weights applied to full period

    Daily frequency; Sharpe annualised with √252.
    MDD at vol_target wealth scaling.
    TO = annualised one-way turnover of blend weights for rolling methods; n/a for Static
         Ridge (blend-weight TO is 0 by construction; instrument-level TO not available here).

    Returns
    -------
    pd.DataFrame with columns [Method, Period, Sharpe, MDD, TO].
    """
    F = load_managed_returns(signals_dir=signals_dir)
    F_is_raw  = F.loc[F.index <= is_end]
    F_oos_raw = F.loc[F.index >  is_end]

    sigma = F_is_raw.std() * np.sqrt(252)
    sigma = sigma.replace(0.0, np.nan)
    F_is  = F_is_raw.div(sigma)
    F_oos = F_oos_raw.div(sigma)
    F_full = pd.concat([F_is, F_oos]).sort_index()

    def _sharpe(r: pd.Series) -> float:
        r = r.dropna()
        if len(r) < 60 or r.std() < 1e-12:
            return float("nan")
        return float(r.mean() / r.std() * np.sqrt(252))

    def _mdd(r: pd.Series) -> float:
        r = r.dropna()
        if r.empty:
            return float("nan")
        cum = (1 + r * vol_target).cumprod()
        return float((cum / cum.cummax() - 1).min() * 100)

    def _blend_weight_to(w_df: pd.DataFrame) -> float:
        """Annual one-way turnover of L1-normalised blend weights."""
        w = w_df.dropna(how="any")
        norm = w.abs().sum(axis=1).replace(0.0, np.nan)
        w_norm = w.div(norm, axis=0)
        return float(w_norm.diff().dropna(how="all").abs().sum(axis=1).mean() * 252)

    def _normalise_w(w_raw: pd.DataFrame) -> pd.DataFrame:
        norm = w_raw.abs().sum(axis=1).replace(0.0, np.nan)
        return w_raw.div(norm, axis=0)

    # ── Method 1: Rolling OLS ────────────────────────────────────────────────
    w_ols = _normalise_w(
        _RollingMSRR(penalty="ols", lookback=lookback, min_obs=min_obs)._roll(F_full)
    )

    # ── Method 2: Rolling Ridge ──────────────────────────────────────────────
    w_roll_ridge = _normalise_w(
        _RollingMSRR(penalty="ridge", lam=ridge_lam, lookback=lookback, min_obs=min_obs)
        ._roll(F_full)
    )

    # ── Method 3: Static Ridge (IS-fit, constant weights) ───────────────────
    static_bw  = msrr_ridge(F_is, lam=ridge_lam)
    w_static_s = static_bw.weights / static_bw.weights.abs().sum()
    w_static = pd.DataFrame(
        np.tile(w_static_s.reindex(F_full.columns).values, (len(F_full), 1)),
        index=F_full.index,
        columns=F_full.columns,
    )

    methods = {
        "Rolling OLS":   w_ols,
        "Rolling Ridge": w_roll_ridge,
        "Static Ridge":  w_static,
    }

    # ── Collect metrics ──────────────────────────────────────────────────────
    records = []
    for method_name, w_full in methods.items():
        w_is_  = w_full.reindex(F_is.index)
        w_oos_ = w_full.reindex(F_oos.index)
        r_is_  = (F_is.loc[w_is_.dropna().index]   * w_is_.dropna()).sum(axis=1)
        r_oos_ = (F_oos.loc[w_oos_.dropna().index] * w_oos_.dropna()).sum(axis=1)
        for period, r, w in [
            ("IS",  r_is_,  w_is_),
            ("OOS", r_oos_, w_oos_),
        ]:
            records.append({
                "Method": method_name,
                "Period": period,
                "Sharpe": _sharpe(r),
                "MDD":    _mdd(r),
                "TO":     float("nan") if method_name == "Static Ridge" else _blend_weight_to(w),
            })

    df = pd.DataFrame(records)

    # ── Print table ──────────────────────────────────────────────────────────
    W = 72
    print("=" * W)
    print(f"  COMMODITIES BLEND COMPARISON  (vol target={vol_target*100:.0f}%, λ_ridge={ridge_lam})")
    print(f"  note: Rolling OLS uses fixed lookback={lookback}d; production uses expanding OLS")
    print(f"  note: TO = blend-weight turnover (ann.); Static Ridge = n/a (0 by construction, instrument-level TO unavailable)")
    print("=" * W)
    print(f"  {'Method':<20}  {'Period':<6}  {'Sharpe':>7}  {'MDD':>8}  {'TO':>7}")
    print("  " + "─" * (W - 2))
    for _, row in df.iterrows():
        sr  = f"{row['Sharpe']:+.2f}" if not np.isnan(row["Sharpe"]) else "   n/a"
        mdd = f"{row['MDD']:.1f}%"    if not np.isnan(row["MDD"])    else "    n/a"
        to  = f"{row['TO']:.2f}x"     if not np.isnan(row["TO"])     else "   n/a"
        print(f"  {row['Method']:<20}  {row['Period']:<6}  {sr:>7}  {mdd:>8}  {to:>7}")
        if row["Period"] == "OOS":
            print("  " + "─" * (W - 2))
    print("=" * W)

    # ── Static Ridge IS weights ───────────────────────────────────────────────
    names  = list(F_is.columns)
    col_w  = 16
    print(f"\n  Avg blend weights  (Static Ridge IS-fit, λ={ridge_lam})")
    print("  " + "─" * (22 + col_w * len(names)))
    print(f"  {'Signal':<20}" + "".join(f"  {s:>{col_w}}" for s in names))
    print("  " + "─" * (22 + col_w * len(names)))
    print(f"  {'IS+OOS':<20}" + "".join(f"  {w_static_s.get(s, 0.0):>+{col_w}.1%}" for s in names))
    print("  " + "─" * (22 + col_w * len(names)))

    return df
