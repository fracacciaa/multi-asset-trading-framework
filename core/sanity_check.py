"""
core/sanity_check.py — End-to-end pipeline sanity check.

Verifies data integrity, numerical conventions, and cross-layer consistency
without rebuilding signals. Reads from committed parquets only.

Sections
  A. Parquet integrity       — files exist, shape, sort, NaN, date coverage
  B. IS/OOS split            — all production modules use the same cutoff
  C. Vol normalization       — IS vol ≈ 10% for every signal sleeve
  D. Backtest engine math    — zero-TC NAV, drift-weight sum, TC formula
  E. TC logic                — half-spreads >0, net≤gross formula, key alignment
  F. Turnover sanity         — cache validity, IS/OOS ratio, plausible ranges
  G. Signal statistics       — IS Sharpe, OOS vol ratio, pairwise correlation, kurtosis
  H. Blend weight checks     — no look-ahead, L1 norm, static cross-asset weights
  I. Cross-layer consistency — monthly resampling formula, return consistency

Usage:
    python -m core.sanity_check
"""
from __future__ import annotations

import datetime
import sys
import io
import contextlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
RATES_SIG  = ROOT / "rates"       / "signals"
EQ_SIG     = ROOT / "equities"    / "signals"
FX_SIG     = ROOT / "fx"          / "signals"
COMM_SIG   = ROOT / "commodities" / "signals"
CORE_CACHE = ROOT / "core"        / "cache"

_IS_END    = "2020-12-31"
_OOS_START = "2021-01-01"

# ── Check result ───────────────────────────────────────────────────────────────

@dataclass
class Check:
    status:  str   # PASS | WARN | FAIL
    section: str
    label:   str
    value:   str = ""
    note:    str = ""


def _ok(section, label, value="", note="")   -> Check:
    return Check("PASS", section, label, value, note)


def _warn(section, label, value="", note="") -> Check:
    return Check("WARN", section, label, value, note)


def _fail(section, label, value="", note="") -> Check:
    return Check("FAIL", section, label, value, note)


def _safe(section: str, label: str, fn) -> Check:
    """Run fn() and surface any exception as FAIL."""
    try:
        return fn()
    except Exception as exc:
        return _fail(section, label, note=f"{type(exc).__name__}: {exc}")


def _quiet(fn):
    """Call fn() suppressing all stdout."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()


# ═══════════════════════════════════════════════════════════════════════════════
# A. Parquet Integrity
# ═══════════════════════════════════════════════════════════════════════════════

_EXPECTED_PARQUETS: dict[str, list[str]] = {
    "rates": [
        "momentum_a_daily_pnl.parquet",
        "carry_ois_par_pnl.parquet",
        "mean_reversion_fly_pnl.parquet",
        "momentum_a_daily.parquet",
        "carry_ois_par.parquet",
        "mean_reversion_fly_dv01_pos.parquet",
        # classic whole-curve MR kept for research
        "mean_reversion_pnl.parquet",
        "mean_reversion_dv01_pos.parquet",
    ],
    "equities": [
        "returns.parquet",
        "momentum_positions.parquet",
        "mean_reversion_positions.parquet",
        "carry_positions.parquet",
        "combined_positions.parquet",
        "rebal_dates.parquet",
    ],
    "fx": [
        "signal_a_f9_h1.parquet",
        "mr_ns_loo_positions.parquet",
        "view_xs_carry_m.parquet",
        "rx_d.parquet",
        "returns.parquet",
    ],
    "commodities": [
        "momentum_pnl.parquet",
        "carry_pnl.parquet",
        "mean_reversion_pnl.parquet",
        "momentum_positions.parquet",
        "carry_positions.parquet",
        "mean_reversion_positions.parquet",
        "combined_positions.parquet",
    ],
}

_SIG_DIRS = {
    "rates":       RATES_SIG,
    "equities":    EQ_SIG,
    "fx":          FX_SIG,
    "commodities": COMM_SIG,
}


def check_parquet_integrity() -> list[Check]:
    sec = "A.Parquets"
    out = []

    # A.1 — All expected files exist
    missing = []
    for ac, files in _EXPECTED_PARQUETS.items():
        d = _SIG_DIRS[ac]
        for f in files:
            if not (d / f).exists():
                missing.append(f"{ac}/{f}")
    total = sum(len(v) for v in _EXPECTED_PARQUETS.values())
    if missing:
        out.append(_fail(sec, "A.1 file existence",
                         note="Missing: " + ", ".join(missing[:5])
                              + (" …" if len(missing) > 5 else "")))
    else:
        out.append(_ok(sec, "A.1 file existence",
                       value=f"{total} files found"))

    # A.2–A.5: load each file and inspect
    max_nan  = 0.0
    max_nan_name = ""
    bad_sort  = []
    bad_shape = []
    no_is     = []
    no_oos    = []

    for ac, files in _EXPECTED_PARQUETS.items():
        d = _SIG_DIRS[ac]
        for f in files:
            p = d / f
            if not p.exists():
                continue
            try:
                df = pd.read_parquet(p)
            except Exception:
                bad_shape.append(f"{ac}/{f} (unreadable)")
                continue

            # A.2 shape ≥ 1000 rows
            if len(df) < 1000:
                bad_shape.append(f"{ac}/{f} ({len(df)} rows)")

            # A.3 sorted ascending
            if not df.index.is_monotonic_increasing:
                bad_sort.append(f"{ac}/{f}")

            if not pd.api.types.is_datetime64_any_dtype(df.index):
                continue

            # A.4 NaN coverage
            # Skip intentionally sparse position/DV01 files: a 3-leg fly or
            # multi-tenor position frame has most columns NaN by construction.
            _sparse_pos = any(tok in f for tok in ("dv01_pos", "_positions", "_pos."))
            if isinstance(df, pd.DataFrame) and df.shape[1] > 0 and not _sparse_pos:
                # Use overall (all-cell) NaN rate; per-column max is misleading
                # for wide position files where many instruments have different
                # history start dates (structural absence ≠ data quality issue)
                overall_nan = float(df.isna().values.mean())
                if overall_nan > max_nan:
                    max_nan      = overall_nan
                    max_nan_name = f"{ac}/{f}"

            # A.5 date coverage
            if (df.index <= _IS_END).sum() < 10:
                no_is.append(f"{ac}/{f}")
            if (df.index >= _OOS_START).sum() < 10:
                no_oos.append(f"{ac}/{f}")

    out.append(
        _ok(sec, "A.2 shape", value="all ≥ 1000 rows") if not bad_shape
        else _warn(sec, "A.2 shape",
                   note="< 1000 rows: " + ", ".join(bad_shape[:3]))
    )
    out.append(
        _ok(sec, "A.3 index sort", value="all sorted ascending") if not bad_sort
        else _fail(sec, "A.3 index sort",
                   note="Not sorted: " + ", ".join(bad_sort[:3]))
    )

    if max_nan > 0.80:
        out.append(_fail(sec, "A.4 NaN coverage",
                         value=f"max {max_nan:.0%} in {max_nan_name}",
                         note=">80% overall NaN"))
    elif max_nan > 0.50:
        out.append(_warn(sec, "A.4 NaN coverage",
                         value=f"max {max_nan:.0%} in {max_nan_name}",
                         note=">50% overall NaN (wide position file with sparse history)"))
    else:
        out.append(_ok(sec, "A.4 NaN coverage", value=f"max {max_nan:.1%}"))

    all_missing_cov = no_is + no_oos
    out.append(
        _ok(sec, "A.5 date coverage", value="IS and OOS both covered") if not all_missing_cov
        else _warn(sec, "A.5 date coverage",
                   note="Missing IS or OOS coverage: " + ", ".join(all_missing_cov[:3]))
    )

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# B. IS/OOS Split Consistency
# ═══════════════════════════════════════════════════════════════════════════════

def check_is_oos_split() -> list[Check]:
    sec = "B.IS/OOS split"

    def _run():
        from rates.blend                  import IS_END as ra_is, OOS_START as ra_oos
        from fx.blend                     import IS_END as fx_is, OOS_START as fx_oos
        from commodities.blend            import IS_END as cm_is, OOS_START as cm_oos
        from core.asset_blend             import IS_END as ab_is, OOS_START as ab_oos
        from core.signal_metrics          import IS_END as sm_is, OOS_START as sm_oos
        from equities.production.pipeline import IS_END as ep_is, OOS_START as ep_oos

        mismatches = []
        for name, is_e, oos_s in [
            ("rates.blend",             ra_is, ra_oos),
            ("fx.blend",                fx_is, fx_oos),
            ("commodities.blend",       cm_is, cm_oos),
            ("core.asset_blend",        ab_is, ab_oos),
            ("core.signal_metrics",     sm_is, sm_oos),
            ("equities.prod.pipeline",  ep_is, ep_oos),
        ]:
            if is_e != _IS_END or oos_s != _OOS_START:
                mismatches.append(f"{name}: IS_END={is_e!r} OOS_START={oos_s!r}")

        if mismatches:
            return _fail(sec, "B.1 IS/OOS split", note="; ".join(mismatches))
        return _ok(sec, "B.1 IS/OOS split",
                   value=f"IS_END={_IS_END!r}  OOS_START={_OOS_START!r}",
                   note="all 6 production modules agree")

    return [_safe(sec, "B.1 IS/OOS split", _run)]


# ═══════════════════════════════════════════════════════════════════════════════
# C. Vol Normalization
# ═══════════════════════════════════════════════════════════════════════════════

def _check_vol(sec: str, label: str, F_is_norm: pd.DataFrame, freq: int) -> Check:
    is_vols = F_is_norm.std() * np.sqrt(freq)
    max_err  = (is_vols - 0.10).abs().max()
    val_str  = "  ".join(f"{k}={v:.4f}" for k, v in is_vols.items())
    if max_err > 0.005:
        return _fail(sec, label, value=val_str,
                     note=f"max deviation from 10%: {max_err:.4%}")
    if max_err > 0.001:
        return _warn(sec, label, value=val_str,
                     note=f"max deviation: {max_err:.4%}")
    return _ok(sec, label, value=val_str)


def check_vol_normalization() -> list[Check]:
    sec = "C.Vol norm"
    out = []

    # C.1 Rates (daily, √252)
    def _rates():
        from rates.blend import load_managed_returns, vol_normalise, IS_END, OOS_START
        F = _quiet(load_managed_returns)
        F_is, _ = vol_normalise(F.loc[F.index <= IS_END], F.loc[F.index >= OOS_START])
        return _check_vol(sec, "C.1 rates IS vol (√252)", F_is, 252)

    out.append(_safe(sec, "C.1 rates IS vol", _rates))

    # C.2 FX (monthly, √12)
    def _fx():
        from fx.blend import load_monthly_managed_returns, IS_END, SIGNAL_VOL_TARGET
        F = _quiet(load_monthly_managed_returns)
        F_is  = F.loc[F.index <= IS_END]
        sigma = (F_is.std() * np.sqrt(12) / SIGNAL_VOL_TARGET).replace(0.0, np.nan)
        F_is_n = F_is.div(sigma)
        return _check_vol(sec, "C.2 FX IS vol (√12)", F_is_n, 12)

    out.append(_safe(sec, "C.2 FX IS vol", _fx))

    # C.3 Commodities (daily, √252)
    def _comm():
        from commodities.blend import load_managed_returns as cm_load, IS_END, OOS_START
        from rates.blend import vol_normalise
        F = _quiet(cm_load)
        F_is, _ = vol_normalise(F.loc[F.index <= IS_END], F.loc[F.index >= OOS_START])
        return _check_vol(sec, "C.3 commodities IS vol (√252)", F_is, 252)

    out.append(_safe(sec, "C.3 commodities IS vol", _comm))

    # C.4 Equities managed returns (daily, √252)
    def _equities():
        from equities.blend import (
            load_signal_positions, load_eq_returns_panel, build_managed_returns,
            signal_vol, normalise_to_target,
        )
        IS_EQ = "2020-12-31"
        positions = load_signal_positions()
        returns   = load_eq_returns_panel()
        pos_is    = {k: v.loc[v.index <= IS_EQ] for k, v in positions.items()}
        F_is_raw  = build_managed_returns(pos_is, returns.loc[returns.index <= IS_EQ])
        sigma     = signal_vol(F_is_raw)
        F_is_n    = normalise_to_target(F_is_raw, sigma)
        return _check_vol(sec, "C.4 equities IS vol (√252)", F_is_n, 252)

    out.append(_safe(sec, "C.4 equities IS vol", _equities))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# D. Backtest Engine Math
# ═══════════════════════════════════════════════════════════════════════════════

def check_backtest_math() -> list[Check]:
    sec = "D.Backtest math"
    out = []

    # D.1 Zero-TC: daily rebalance to 50/50 → returns match exactly
    def _zero_tc():
        from core.backtest import run_backtest
        rng   = np.random.default_rng(42)
        dates = pd.date_range("2020-01-01", periods=20, freq="B")
        r_A   = pd.Series(rng.normal(0.001, 0.01, 20), index=dates)
        r_B   = pd.Series(rng.normal(-0.001, 0.015, 20), index=dates)
        returns = pd.DataFrame({"A": r_A, "B": r_B})
        bt = run_backtest(
            returns,
            lambda d: pd.Series({"A": 0.5, "B": 0.5}),
            rebalance_freq="D",
            transaction_cost_bps=0.0,
        )
        # Day 0: initial weights=0 → return=0 (before first rebalance takes effect)
        # Day 1+: weights=[0.5,0.5] from prior day's rebalance → return = 0.5*(rA+rB)
        expected = 0.5 * r_A + 0.5 * r_B
        err = (bt["daily_return"].iloc[1:] - expected.iloc[1:]).abs().max()
        if err > 1e-9:
            return _fail(sec, "D.1 zero-TC NAV",
                         value=f"max err={err:.2e}",
                         note="daily_return deviates from expected portfolio return")
        return _ok(sec, "D.1 zero-TC NAV", value=f"max err={err:.2e}")

    out.append(_safe(sec, "D.1 zero-TC NAV", _zero_tc))

    # D.2 Drift weights: Σ w_i·(1+r_i)/(1+Σw_j·r_j) = 1 always when Σw_i = 1
    def _drift_sum():
        rng = np.random.default_rng(42)
        failures = 0
        max_err  = 0.0
        for _ in range(2_000):
            K = rng.integers(2, 8)
            w = rng.dirichlet(np.ones(K))          # positive simplex, sum=1
            r = rng.normal(0.001, 0.02, K)
            port_ret = float(w @ r)
            if abs(1 + port_ret) < 1e-12:
                continue
            w_drift  = w * (1 + r) / (1 + port_ret)
            err      = abs(w_drift.sum() - 1.0)
            max_err  = max(max_err, err)
            if err > 1e-10:
                failures += 1
        if failures:
            return _fail(sec, "D.2 drift weights sum",
                         note=f"{failures}/2000 violated sum=1 (max err={max_err:.2e})")
        return _ok(sec, "D.2 drift weights sum",
                   value=f"max err={max_err:.2e} over 2000 simulations")

    out.append(_safe(sec, "D.2 drift weights sum", _drift_sum))

    # D.3 TC formula: (|Δw|/2) × bps/10_000
    def _tc_formula():
        from core.backtest import apply_transaction_costs
        old      = pd.Series({"A": 0.5, "B": 0.5})
        new      = pd.Series({"A": 0.6, "B": 0.4})
        bps      = 7.0
        expected = abs(new - old).sum() / 2 * bps / 10_000  # sum|Δw|/2 × bps/10_000
        actual   = apply_transaction_costs(new, old, bps)
        err      = abs(actual - expected)
        if err > 1e-15:
            return _fail(sec, "D.3 TC formula",
                         value=f"expected={expected:.6e} got={actual:.6e}")
        return _ok(sec, "D.3 TC formula",
                   value=f"{actual:.6e}  (7 bps on 5% one-way TO)")

    out.append(_safe(sec, "D.3 TC formula", _tc_formula))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# E. TC Logic
# ═══════════════════════════════════════════════════════════════════════════════

def check_tc_logic() -> list[Check]:
    sec = "E.TC logic"
    out = []

    # E.1 All TC half-spreads > 0 in every scenario dict
    def _positivity():
        from rates.blend       import TC_SCENARIOS, _TC_DEFAULT_BP
        from fx.blend          import TC_SCENARIOS_FX
        from commodities.blend import TC_SCENARIOS_COMM

        negatives  = []
        total_vals = 0
        for name, scenarios in [
            ("rates",       TC_SCENARIOS),
            ("fx",          TC_SCENARIOS_FX),
            ("commodities", TC_SCENARIOS_COMM),
        ]:
            for sc, d in scenarios.items():
                for mkt, h in d.items():
                    total_vals += 1
                    if h <= 0:
                        negatives.append(f"{name}.{sc}.{mkt}={h}")

        defaults = [
            ("rates._TC_DEFAULT_BP", _TC_DEFAULT_BP),
        ]
        for name, val in defaults:
            if val <= 0:
                negatives.append(f"{name}={val}")

        if negatives:
            return _fail(sec, "E.1 TC half-spreads >0",
                         note="Non-positive: " + ", ".join(negatives))
        return _ok(sec, "E.1 TC half-spreads >0",
                   value=f"{total_vals} market-scenario pairs all >0")

    out.append(_safe(sec, "E.1 TC half-spreads >0", _positivity))

    # E.2 Net return formula: net = G·w − C·|w| ≤ G·w = gross whenever C ≥ 0
    def _net_le_gross():
        rng = np.random.default_rng(42)
        failures = 0
        for _ in range(10_000):
            K = 3
            w = rng.uniform(-1, 1, K)
            norm = np.abs(w).sum()
            if norm < 1e-9:
                continue
            w = w / norm
            G = rng.normal(0.001, 0.01, K)
            C = np.abs(rng.normal(0, 0.0001, K))   # costs always ≥ 0
            gross = float(G @ w)
            net   = gross - float(C @ np.abs(w))
            if net > gross + 1e-14:
                failures += 1
        if failures:
            return _fail(sec, "E.2 net ≤ gross formula",
                         note=f"{failures}/10000 cases violated net ≤ gross")
        return _ok(sec, "E.2 net ≤ gross formula",
                   value="10000 random cases",
                   note="G·w − C·|w| ≤ G·w always holds when C ≥ 0")

    out.append(_safe(sec, "E.2 net ≤ gross formula", _net_le_gross))

    # E.3 Cross-asset equities TC keys: TC_SCENARIOS_ASSETS must match tc_sweep_eq default
    def _eq_key_alignment():
        import inspect
        from core.asset_blend  import TC_SCENARIOS_ASSETS
        from equities.blend    import tc_sweep_eq
        asset_vals   = sorted(float(s["equities"]) for s in TC_SCENARIOS_ASSETS.values())
        default_bps  = sorted(float(x) for x in
                              inspect.signature(tc_sweep_eq)
                              .parameters["tc_bps_list"].default)
        if asset_vals != default_bps:
            return _fail(sec, "E.3 equities TC key alignment",
                         value=f"TC_SCENARIOS_ASSETS={asset_vals}  "
                               f"tc_sweep_eq default={default_bps}",
                         note="Mismatch — run_assets_tc will fail to find equities TC returns")
        return _ok(sec, "E.3 equities TC key alignment",
                   value=str(asset_vals))

    out.append(_safe(sec, "E.3 equities TC key alignment", _eq_key_alignment))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# F. Turnover Sanity
# ═══════════════════════════════════════════════════════════════════════════════

_TO_RANGES = {
    "equities":    (2.0,  50.0),
    "rates":       (5.0, 200.0),
    "fx":          (0.5, 100.0),
    "commodities": (0.5, 100.0),
}


def check_turnover_sanity() -> list[Check]:
    sec = "F.Turnover"
    out = []

    to_path = CORE_CACHE / "position_turnover.parquet"

    # F.1 Cache exists, non-negative, finite; staleness warning
    def _cache_basic():
        if not to_path.exists():
            return _fail(sec, "F.1 turnover cache",
                         note="position_turnover.parquet not found — "
                              "run core.asset_blend.run_assets_tc() to generate it")
        to_df = pd.read_parquet(to_path)
        mtime = datetime.datetime.fromtimestamp(to_path.stat().st_mtime)
        age_d = (datetime.datetime.now() - mtime).days
        vals  = to_df.dropna(how="all").stack()
        if (vals < 0).any():
            return _fail(sec, "F.1 turnover cache",
                         note="Negative turnover values found")
        if not np.isfinite(vals).all():
            return _fail(sec, "F.1 turnover cache",
                         note="Non-finite turnover values found")
        status = "WARN" if age_d > 30 else "PASS"
        return Check(status, sec, "F.1 turnover cache",
                     value=f"{len(to_df)} months  age={age_d}d",
                     note="file is stale (>30d)" if age_d > 30 else "")

    out.append(_safe(sec, "F.1 turnover cache", _cache_basic))

    # F.2 IS/OOS median ratio within 0.1–10×
    def _is_oos_ratio():
        if not to_path.exists():
            return _fail(sec, "F.2 IS/OOS TO ratio", note="cache missing")
        to_df = pd.read_parquet(to_path)
        problems = []
        summary  = {}
        for ac in ["equities", "rates", "fx", "commodities"]:
            if ac not in to_df.columns:
                problems.append(f"{ac}: column missing")
                continue
            s = to_df[ac].dropna()
            is_med  = s.loc[s.index <= _IS_END].median()
            oos_med = s.loc[s.index >= _OOS_START].median()
            if np.isnan(is_med) or np.isnan(oos_med) or is_med < 1e-6:
                continue
            ratio = oos_med / is_med
            summary[ac] = f"{ratio:.2f}×"
            if not (0.1 < ratio < 10.0):
                problems.append(f"{ac}: ratio={ratio:.2f} "
                                 f"(IS={is_med:.1f}x OOS={oos_med:.1f}x)")
        val_str = "  ".join(f"{k}={v}" for k, v in summary.items())
        if problems:
            return _fail(sec, "F.2 IS/OOS TO ratio", value=val_str,
                         note="; ".join(problems))
        return _ok(sec, "F.2 IS/OOS TO ratio", value=val_str)

    out.append(_safe(sec, "F.2 IS/OOS TO ratio", _is_oos_ratio))

    # F.3 Plausible ranges per asset class
    def _ranges():
        if not to_path.exists():
            return _fail(sec, "F.3 TO plausible range", note="cache missing")
        to_df = pd.read_parquet(to_path)
        problems = []
        meds = {}
        for ac, (lo, hi) in _TO_RANGES.items():
            if ac not in to_df.columns:
                continue
            med = to_df[ac].dropna().median()
            meds[ac] = f"{med:.1f}x"
            if not (lo < med < hi):
                problems.append(f"{ac}: {med:.1f}x (expected {lo}–{hi}×)")
        val_str = "  ".join(f"{k}={v}" for k, v in meds.items())
        if problems:
            return _warn(sec, "F.3 TO plausible range", value=val_str,
                         note="; ".join(problems))
        return _ok(sec, "F.3 TO plausible range", value=val_str)

    out.append(_safe(sec, "F.3 TO plausible range", _ranges))

    # F.4 Drift-corrected TO > raw TO for a constant signal with trending returns
    def _drift_gt_raw():
        from core.pipeline_report import pos_turnover, ReportCfg
        rng   = np.random.default_rng(42)
        T     = 252
        dates = pd.date_range("2010-01-01", periods=T, freq="B")
        # Constant 50/50 position (signal never changes)
        pos   = pd.DataFrame({"A": 0.5, "B": 0.5}, index=dates)
        # Strongly trending returns so portfolio drifts far from target
        ret   = pd.DataFrame({
            "A": np.full(T, 0.003),   # 0.3% per day upward
            "B": np.full(T, -0.001),  # 0.1% per day downward
        }, index=dates)
        cfg   = ReportCfg(ann=252, is_end="2015-12-31", oos_start="2016-01-01")
        to_raw   = pos_turnover(pos, dates, cfg, ret=None)
        to_drift = pos_turnover(pos, dates, cfg, ret=ret)
        if any(np.isnan(x) for x in [to_raw, to_drift]):
            return _warn(sec, "F.4 drift TO > raw TO",
                         note="could not compute turnover")
        if to_drift <= to_raw:
            return _warn(sec, "F.4 drift TO > raw TO",
                         value=f"drift={to_drift:.2f}x  raw={to_raw:.2f}x",
                         note="drift-corrected should exceed raw for constant positions with trending returns")
        return _ok(sec, "F.4 drift TO > raw TO",
                   value=f"drift={to_drift:.4f}x > raw={to_raw:.4f}x")

    out.append(_safe(sec, "F.4 drift TO > raw TO", _drift_gt_raw))

    # F.5 pos_turnover formula correctness: matches manual computation
    def _formula_check():
        from core.pipeline_report import pos_turnover, ReportCfg
        rng   = np.random.default_rng(99)
        T     = 252
        dates = pd.date_range("2015-01-01", periods=T, freq="B")
        # Random positions
        pos_A = rng.uniform(0.3, 0.7, T)
        pos   = pd.DataFrame({"A": pos_A, "B": 1.0 - pos_A}, index=dates)
        cfg   = ReportCfg(ann=252, is_end="2016-12-31", oos_start="2017-01-01")
        # Manual: mean(|Δpos|.sum()) / gross_mean * 252
        delta    = pos.diff().dropna(how="all").abs().sum(axis=1)
        gross    = pos.abs().sum(axis=1).replace(0.0, np.nan).mean()
        expected = float(delta.mean() * 252 / gross)
        actual   = pos_turnover(pos, dates, cfg, ret=None)
        if np.isnan(actual):
            return _warn(sec, "F.5 TO formula", note="pos_turnover returned NaN")
        err = abs(actual - expected)
        if err > 1e-9:
            return _fail(sec, "F.5 TO formula",
                         value=f"pos_turnover={actual:.6f}  manual={expected:.6f}",
                         note=f"err={err:.2e}")
        return _ok(sec, "F.5 TO formula",
                   value=f"{actual:.2f}x  (matches manual)")

    out.append(_safe(sec, "F.5 TO formula", _formula_check))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# G. Signal Statistics
# ═══════════════════════════════════════════════════════════════════════════════

def _excess_kurtosis(r: pd.Series) -> float:
    r = r.dropna() - r.dropna().mean()
    s2 = float(np.mean(r**2))
    if s2 < 1e-30:
        return 0.0
    return float(np.mean(r**4) / s2**2 - 3.0)


def check_signal_statistics() -> list[Check]:
    sec = "G.Signal stats"
    out = []

    # Load all asset-class F matrices (vol-normalised, IS/OOS split)
    # Each entry: (F_is, F_oos, ann_factor)
    asset_data: dict[str, tuple] = {}

    try:
        from rates.blend import load_managed_returns, vol_normalise, IS_END, OOS_START
        F = _quiet(load_managed_returns)
        F_is, F_oos = vol_normalise(F.loc[F.index <= IS_END],
                                    F.loc[F.index >= OOS_START])
        asset_data["rates"] = (F_is, F_oos, 252)
    except Exception as exc:
        out.append(_fail(sec, "G.x rates load", note=str(exc)))

    try:
        from fx.blend import (load_monthly_managed_returns,
                              IS_END as FX_IS, OOS_START as FX_OOS, SIGNAL_VOL_TARGET)
        F = _quiet(load_monthly_managed_returns)
        F_is_raw  = F.loc[F.index <= FX_IS]
        F_oos_raw = F.loc[F.index >= FX_OOS]
        sigma     = (F_is_raw.std() * np.sqrt(12) / SIGNAL_VOL_TARGET).replace(0.0, np.nan)
        asset_data["fx"] = (F_is_raw.div(sigma), F_oos_raw.div(sigma), 12)
    except Exception as exc:
        out.append(_fail(sec, "G.x fx load", note=str(exc)))

    try:
        from commodities.blend import load_managed_returns as cm_load, IS_END as CM_IS, OOS_START as CM_OOS
        from rates.blend import vol_normalise as vn
        F = _quiet(cm_load)
        F_is, F_oos = vn(F.loc[F.index <= CM_IS], F.loc[F.index >= CM_OOS])
        asset_data["commodities"] = (F_is, F_oos, 252)
    except Exception as exc:
        out.append(_fail(sec, "G.x commodities load", note=str(exc)))

    if not asset_data:
        out.append(_fail(sec, "G.1–G.4", note="no asset data loaded"))
        return out

    # G.1 IS Sharpe per signal within [-1, 4] PASS, [-2, 5] WARN
    def _sharpe():
        fail_cases = []
        warn_cases = []
        vals = {}
        for ac, (F_is, _, ann) in asset_data.items():
            for sig in F_is.columns:
                r = F_is[sig].dropna()
                if len(r) < 30 or r.std() < 1e-12:
                    continue
                sr = float(r.mean() / r.std() * np.sqrt(ann))
                key = f"{ac}.{sig}"
                vals[key] = sr
                if not (-2 <= sr <= 7):
                    fail_cases.append(f"{key}={sr:.2f}")
                elif not (-1 <= sr <= 5):
                    warn_cases.append(f"{key}={sr:.2f}")
        val_str = "  ".join(f"{k}={v:.2f}" for k, v in vals.items())
        if fail_cases:
            return _fail(sec, "G.1 IS Sharpe", value=val_str,
                         note="outside [-2,7]: " + ", ".join(fail_cases))
        if warn_cases:
            return _warn(sec, "G.1 IS Sharpe", value=val_str,
                         note="outside [-1,5]: " + ", ".join(warn_cases))
        return _ok(sec, "G.1 IS Sharpe", value=val_str)

    out.append(_safe(sec, "G.1 IS Sharpe", _sharpe))

    # G.2 OOS/IS vol ratio ≤ 2× PASS, ≤ 3× WARN, >3× FAIL
    def _vol_ratio():
        fail_cases = []
        warn_cases = []
        vals = {}
        for ac, (F_is, F_oos, ann) in asset_data.items():
            for sig in F_is.columns:
                is_vol  = F_is[sig].dropna().std() * np.sqrt(ann)
                oos_vol = F_oos[sig].dropna().std() * np.sqrt(ann)
                if is_vol < 1e-10:
                    continue
                ratio = oos_vol / is_vol
                key   = f"{ac}.{sig}"
                vals[key] = ratio
                if ratio > 3:
                    fail_cases.append(f"{key}={ratio:.2f}×")
                elif ratio > 2:
                    warn_cases.append(f"{key}={ratio:.2f}×")
        val_str = "  ".join(f"{k}={v:.2f}×" for k, v in vals.items())
        if fail_cases:
            return _fail(sec, "G.2 OOS/IS vol ratio", value=val_str,
                         note=">3× (possible regime change): " + ", ".join(fail_cases))
        if warn_cases:
            return _warn(sec, "G.2 OOS/IS vol ratio", value=val_str,
                         note=">2×: " + ", ".join(warn_cases))
        return _ok(sec, "G.2 OOS/IS vol ratio", value=val_str)

    out.append(_safe(sec, "G.2 OOS/IS vol ratio", _vol_ratio))

    # G.3 Pairwise IS signal correlation — no pair ≥ 0.90 PASS, ≥ 0.95 FAIL
    def _correlation():
        fail_pairs = []
        warn_pairs = []
        all_max    = []
        for ac, (F_is, _, _) in asset_data.items():
            df  = F_is.dropna()
            if df.shape[1] < 2:
                continue
            corr = df.corr()
            cols = list(corr.columns)
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    c    = abs(corr.iloc[i, j])
                    pair = f"{ac}:({cols[i]},{cols[j]})={c:.3f}"
                    all_max.append((c, pair))
                    if c >= 0.95:
                        fail_pairs.append(pair)
                    elif c >= 0.90:
                        warn_pairs.append(pair)
        if all_max:
            top = max(all_max, key=lambda x: x[0])
            val_str = f"max|corr|={top[1]}"
        else:
            val_str = "n/a"
        if fail_pairs:
            return _fail(sec, "G.3 pairwise IS correlation", value=val_str,
                         note="≥0.95: " + ", ".join(fail_pairs))
        if warn_pairs:
            return _warn(sec, "G.3 pairwise IS correlation", value=val_str,
                         note="≥0.90: " + ", ".join(warn_pairs))
        return _ok(sec, "G.3 pairwise IS correlation", value=val_str)

    out.append(_safe(sec, "G.3 pairwise IS correlation", _correlation))

    # G.4 Excess kurtosis ≤ 10 PASS, ≤ 20 WARN, >20 FAIL
    def _kurtosis():
        fail_cases = []
        warn_cases = []
        vals = {}
        for ac, (F_is, _, _) in asset_data.items():
            for sig in F_is.columns:
                k = _excess_kurtosis(F_is[sig])
                key = f"{ac}.{sig}"
                vals[key] = k
                if k > 50:
                    fail_cases.append(f"{key}={k:.1f}")
                elif k > 20:
                    warn_cases.append(f"{key}={k:.1f}")
        val_str = "  ".join(f"{k}={v:.1f}" for k, v in vals.items())
        if fail_cases:
            return _fail(sec, "G.4 excess kurtosis", value=val_str,
                         note=">50 (severe outliers): " + ", ".join(fail_cases))
        if warn_cases:
            return _warn(sec, "G.4 excess kurtosis", value=val_str,
                         note=">20 (fat tails — check for outlier days): "
                              + ", ".join(warn_cases))
        return _ok(sec, "G.4 excess kurtosis", value=val_str)

    out.append(_safe(sec, "G.4 excess kurtosis", _kurtosis))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# H. Blend Weight Checks
# ═══════════════════════════════════════════════════════════════════════════════

def check_blend_weights() -> list[Check]:
    sec = "H.Blend weights"
    out = []

    # H.1 RollingMSRR no look-ahead: perturbing row k should not change weights at steps ≤ k
    def _no_lookahead():
        from equities.blend import RollingMSRR
        rng   = np.random.default_rng(42)
        T     = 60
        dates = pd.date_range("2010-01-01", periods=T, freq="B")
        F     = pd.DataFrame(rng.normal(0, 0.01, (T, 3)),
                              index=dates, columns=["A", "B", "C"])
        m     = RollingMSRR(penalty="ridge", lookback=20, min_obs=10)
        # Perturb row at position 30 with extreme values
        F_pert         = F.copy()
        F_pert.iloc[30] = 1_000_000.0
        w_orig = m._roll(F)
        w_pert = m._roll(F_pert)
        # Steps 10–30: computed from rows 0..(i-1), never see row 30 itself
        # → weights should be identical to original
        early = dates[:31]   # positions 0–30 inclusive
        early_w1 = w_orig.reindex(early).dropna(how="all")
        early_w2 = w_pert.reindex(early).dropna(how="all")
        common   = early_w1.index.intersection(early_w2.index)
        if common.empty:
            return _warn(sec, "H.1 no look-ahead", note="no overlapping weights to compare")
        err_early = (early_w1.loc[common] - early_w2.loc[common]).abs().max().max()
        # Steps 31+: window includes row 30 → weights MUST change
        late = dates[31:]
        late_w1 = w_orig.reindex(late).dropna(how="all")
        late_w2 = w_pert.reindex(late).dropna(how="all")
        late_common = late_w1.index.intersection(late_w2.index)
        err_late = (
            (late_w1.loc[late_common] - late_w2.loc[late_common]).abs().max().max()
            if not late_common.empty else np.nan
        )
        if err_early > 1e-10:
            return _fail(sec, "H.1 no look-ahead",
                         value=f"err_early={err_early:.2e}  err_late={err_late:.2e}",
                         note="Weights at steps ≤ perturbation point changed — LOOK-AHEAD DETECTED")
        if not np.isnan(err_late) and err_late < 1e-3:
            return _warn(sec, "H.1 no look-ahead",
                         value=f"err_early={err_early:.2e}  err_late={err_late:.2e}",
                         note="Late weights did not change after extreme perturbation — unexpected")
        return _ok(sec, "H.1 no look-ahead",
                   value=f"err_early={err_early:.2e}  err_late={err_late:.2e}",
                   note="early weights unchanged; late weights affected as expected")

    out.append(_safe(sec, "H.1 no look-ahead", _no_lookahead))

    # H.2 L1-normalized blend weights: each non-NaN row sums to 1.0
    def _l1_norm():
        from rates.blend import run_blend
        res  = _quiet(run_blend)
        w    = res["weights"].dropna(how="any")
        norms = w.abs().sum(axis=1)
        max_err = float((norms - 1.0).abs().max())
        n       = len(norms)
        if max_err > 1e-6:
            return _fail(sec, "H.2 L1 weight norm",
                         value=f"max |norm−1| = {max_err:.2e} over {n} rows")
        return _ok(sec, "H.2 L1 weight norm",
                   value=f"max |norm−1| = {max_err:.2e} over {n} rows")

    out.append(_safe(sec, "H.2 L1 weight norm", _l1_norm))

    # H.3 Cross-asset inv_vol weights: sum=1, all positive, proportional to 1/std
    def _inv_vol_weights():
        from core.asset_blend import _inv_vol_weights as _ivw
        rng   = np.random.default_rng(42)
        dates = pd.date_range("2010-01-01", periods=252, freq="B")
        stds  = {"equities": 0.010, "rates": 0.005,
                 "fx": 0.015, "commodities": 0.008}
        F_is  = pd.DataFrame(
            {k: rng.normal(0.001, v, 252) for k, v in stds.items()},
            index=dates,
        )
        w = _ivw(F_is)
        total = float(w.abs().sum())
        if abs(total - 1.0) > 1e-10:
            return _fail(sec, "H.3 cross-asset inv_vol weights",
                         value=f"sum|w|={total:.6f}",
                         note="weights don't sum to 1")
        if (w <= 0).any():
            return _fail(sec, "H.3 cross-asset inv_vol weights",
                         value=w.round(4).to_dict(),
                         note="some weight ≤ 0")
        # Verify proportionality: w_k ∝ 1/std_k (using empirical stds, not nominal)
        sigma_emp  = F_is.std()
        expected_w = (1.0 / sigma_emp) / (1.0 / sigma_emp).sum()
        err = float((w.reindex(expected_w.index) - expected_w).abs().max())
        if err > 1e-6:
            return _warn(sec, "H.3 cross-asset inv_vol weights",
                         value=w.round(4).to_dict(),
                         note=f"not proportional to 1/std (err={err:.2e})")
        return _ok(sec, "H.3 cross-asset inv_vol weights",
                   value="  ".join(f"{k}={v:.3f}" for k, v in w.items()),
                   note="sum=1, all>0, proportional to 1/std")

    out.append(_safe(sec, "H.3 cross-asset inv_vol weights", _inv_vol_weights))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# I. Cross-Layer Consistency
# ═══════════════════════════════════════════════════════════════════════════════

def check_cross_layer_consistency() -> list[Check]:
    sec = "I.Cross-layer"
    out = []

    # I.1 Monthly resampling: _to_monthly uses .resample("BME").sum()
    def _monthly_sum():
        from core.asset_blend import _to_monthly
        dates    = pd.date_range("2020-01-01", periods=63, freq="B")
        r        = pd.Series(np.ones(63) * 0.001, index=dates)
        monthly  = _to_monthly(r, "test")
        expected = r.resample("BME").sum().rename("test")
        err      = (monthly - expected).abs().max()
        if err > 1e-15:
            return _fail(sec, "I.1 monthly resampling",
                         value=f"max err={err:.2e}",
                         note="_to_monthly does not equal .resample('BME').sum()")
        return _ok(sec, "I.1 monthly resampling", value=f"max err={err:.2e}")

    out.append(_safe(sec, "I.1 monthly resampling", _monthly_sum))

    # I.2 Cross-asset return formula: manual inv_vol blend == _apply_static
    def _cross_asset_formula():
        from core.asset_blend import _inv_vol_weights, _apply_static, IS_END
        from equities.blend   import BlendWeights
        from rates.blend      import load_managed_returns, vol_normalise
        from rates.blend      import IS_END as R_IS, OOS_START as R_OOS
        # Use rates managed returns as a lightweight 3-sleeve proxy
        F     = _quiet(load_managed_returns)
        F_is  = F.loc[F.index <= R_IS]
        w     = _inv_vol_weights(F_is)
        # Manual blend
        r_manual   = (F_is * w).sum(axis=1)
        # Via _apply_static
        bw         = BlendWeights(w, "inv_vol")
        r_function = _apply_static(F_is, bw)
        err = float((r_manual - r_function).abs().max())
        if err > 1e-13:
            return _fail(sec, "I.2 cross-asset return formula",
                         value=f"max err={err:.2e}")
        return _ok(sec, "I.2 cross-asset return formula",
                   value=f"max err={err:.2e}",
                   note="manual w·F == _apply_static(F, BlendWeights(w))")

    out.append(_safe(sec, "I.2 cross-asset return formula", _cross_asset_formula))

    # I.3 Equities cached parquet vs freshly computed (tc_bps=0)
    def _eq_consistency():
        cached_path = EQ_SIG / "eq_prod_returns.parquet"
        if not cached_path.exists():
            return _warn(sec, "I.3 equities return consistency",
                         note="eq_prod_returns.parquet not found")
        cached = pd.read_parquet(cached_path)["eq_prod"]
        from equities.blend import build_eq_composite_return
        fresh  = build_eq_composite_return(tc_bps=0.0)
        common = cached.index.intersection(fresh.index)
        if common.empty:
            return _fail(sec, "I.3 equities return consistency",
                         note="no overlapping dates")
        # Compare only on days with non-trivial values in both
        both = pd.DataFrame({"c": cached.loc[common],
                              "f": fresh.loc[common]}).dropna()
        both = both[(both.abs() > 1e-8).any(axis=1)]
        if both.empty:
            return _warn(sec, "I.3 equities return consistency",
                         note="no non-trivial overlapping dates")
        mae = float((both["c"] - both["f"]).abs().mean())
        max_err = float((both["c"] - both["f"]).abs().max())
        if mae > 1e-3:
            return _fail(sec, "I.3 equities return consistency",
                         value=f"MAE={mae:.2e}  max={max_err:.2e}",
                         note="Cached parquet may be stale or generated with different blend params")
        if mae > 1e-4:
            return _warn(sec, "I.3 equities return consistency",
                         value=f"MAE={mae:.2e}  max={max_err:.2e}",
                         note="Small deviation — consider refreshing the parquet")
        return _ok(sec, "I.3 equities return consistency",
                   value=f"MAE={mae:.2e}  max={max_err:.2e}",
                   note="cached matches fresh computation")

    out.append(_safe(sec, "I.3 equities return consistency", _eq_consistency))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestration
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_checks() -> list[Check]:
    checks: list[Check] = []
    checks += check_parquet_integrity()
    checks += check_is_oos_split()
    checks += check_vol_normalization()
    checks += check_backtest_math()
    checks += check_tc_logic()
    checks += check_turnover_sanity()
    checks += check_signal_statistics()
    checks += check_blend_weights()
    checks += check_cross_layer_consistency()
    return checks


def print_report(checks: list[Check]) -> None:
    W = 108
    _BADGE = {"PASS": " OK ", "WARN": "WARN", "FAIL": "FAIL"}
    print()
    print("=" * W)
    print(f"  PIPELINE SANITY CHECK  —  {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M')}")
    print("=" * W)

    current_section = None
    for c in checks:
        if c.section != current_section:
            print(f"\n  ── {c.section} {'─' * max(0, W - len(c.section) - 6)}")
            current_section = c.section
        badge = _BADGE.get(c.status, "????")
        line  = f"  [{badge}] {c.label}"
        if c.value:
            line += f"  ({c.value})"
        if c.note:
            line += f"  — {c.note}"
        print(line)

    n_pass = sum(1 for c in checks if c.status == "PASS")
    n_warn = sum(1 for c in checks if c.status == "WARN")
    n_fail = sum(1 for c in checks if c.status == "FAIL")

    print()
    print("=" * W)
    overall = "PASS" if n_fail == 0 and n_warn == 0 else ("WARN" if n_fail == 0 else "FAIL")
    print(f"  RESULT  {n_pass} PASS  {n_warn} WARN  {n_fail} FAIL  —  overall: {overall}")
    print("=" * W)
    print()


if __name__ == "__main__":
    import time
    t0     = time.time()
    checks = run_all_checks()
    t1     = time.time()
    print_report(checks)
    print(f"  (completed in {t1 - t0:.1f}s)\n")
    sys.exit(0 if all(c.status != "FAIL" for c in checks) else 1)
