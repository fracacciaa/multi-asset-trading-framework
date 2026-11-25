"""
Regenerate all signal parquets from raw Bloomberg data.

Usage
-----
    python -m core.export_all

Runs the production signal pipeline for all four asset-class sleeves in order,
writing parquets to */signals/.  Requires Bloomberg export files in MATF_DATA_DIR
(default: ~/Downloads).  After this, python -m core.main runs with no raw data.
"""

from __future__ import annotations


def main(verbose: bool = True) -> None:
    if verbose:
        print("=" * 60)
        print("  export_all: regenerating all signal parquets")
        print("=" * 60)

    if verbose:
        print("\n[1/4] Rates …")
    from rates.production.run import export_signals as rates_export
    rates_export(verbose=verbose)

    if verbose:
        print("\n[2/4] FX …")
    from fx.production.run import export_signals as fx_export
    fx_export(verbose=verbose)

    if verbose:
        print("\n[3/4] Commodities …")
    from commodities.blend import export_signals as comm_export
    comm_export()

    if verbose:
        print("\n[4/4] Equities …")
    from equities.production.pipeline import export_signals as eq_export
    eq_export()

    if verbose:
        print("\n" + "=" * 60)
        print("  export_all: done.  Run python -m core.main to verify.")
        print("=" * 60)


if __name__ == "__main__":
    main()
