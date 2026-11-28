"""
Refresh all core/cache report files from current signal parquets.

Usage:  python -m core.refresh_cache
"""
from __future__ import annotations

import contextlib
import io
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CACHE = Path(__file__).parent / "cache"
HEADER = (
    f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d')} UTC\n"
    f"IS <= 2020-12-31  |  OOS >= 2021-01-01  |  10% vol target\n"
)


def _capture(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return result, buf.getvalue()


def refresh_rates_blend():
    print("[1/4] rates blend comparison …")
    from rates.blend import compare_blend_methods
    df, txt = _capture(compare_blend_methods)
    (CACHE / "rates_blend_comparison.csv").write_text(df.to_csv(index=False))
    (CACHE / "rates_blend_comparison.txt").write_text(HEADER + "\n" + txt)
    print("      done.")


def refresh_rates_tc():
    print("[2/4] rates TC sweep …")
    from rates.blend import run_rates_tc, tc_sweep_rates, TC_SCENARIOS
    result, txt = _capture(tc_sweep_rates, scenarios=TC_SCENARIOS)
    if isinstance(result, tuple):
        df = result[0]
    else:
        df = result
    (CACHE / "rates_tc_metrics.csv").write_text(df.to_csv(index=False))
    (CACHE / "rates_tc_report.txt").write_text(HEADER + "\n" + txt)
    print("      done.")


def refresh_asset_blend():
    print("[3/4] cross-asset blend comparison …")
    from core.asset_blend import compare_blend_methods
    df, txt = _capture(compare_blend_methods)
    (CACHE / "asset_blend_comparison.csv").write_text(df.to_csv(index=False))
    (CACHE / "asset_blend_report.txt").write_text(HEADER + "\n" + txt)
    print("      done.")


def refresh_asset_tc():
    print("[4/4] cross-asset TC sweep …")
    from core.asset_blend import run_assets_tc, TC_SCENARIOS_ASSETS
    result, txt = _capture(run_assets_tc, scenarios=TC_SCENARIOS_ASSETS)
    df = result["metrics"]
    (CACHE / "asset_tc_metrics.csv").write_text(df.to_csv(index=False))
    (CACHE / "asset_tc_report.txt").write_text(HEADER + "\n" + txt)
    print("      done.")


if __name__ == "__main__":
    refresh_rates_blend()
    refresh_rates_tc()
    refresh_asset_blend()
    refresh_asset_tc()
    print("\nAll cache files refreshed.")
