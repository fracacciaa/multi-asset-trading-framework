#!/usr/bin/env python3
"""
Research signal evaluation: 4 production signals + 7 new variants.

Usage:
    python equities_research_eval.py           # use cache if available
    python equities_research_eval.py --rebuild  # force recompute signals + VF
"""

from __future__ import annotations
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true",
                        help="Force recompute research VF data (reuses production covariances)")
    args = parser.parse_args()

    from equities.pipeline import run_eq_research_signals
    run_eq_research_signals(rebuild=args.rebuild)


if __name__ == "__main__":
    main()
