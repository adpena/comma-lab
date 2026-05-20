#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Operator-facing CLI for exploit #7 - information-theoretic floor estimator.

Per RESPAWN-MG-7-BUNDLE 2026-05-20 + CLAUDE.md "Meta-Lagrangian/Pareto solver".

Usage::

    .venv/bin/python tools/estimate_information_theoretic_floor.py \\
        --m-contest <path.npy> [--mode cramer_rao|fisher_trace|shannon_lower] \\
        [--current-best-empirical-score X] [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Information-theoretic floor per exploit #7",
    )
    parser.add_argument("--m-contest", required=True, type=Path)
    parser.add_argument("--mode", choices=["cramer_rao", "fisher_trace", "shannon_lower"],
                        default="cramer_rao")
    parser.add_argument("--current-best-empirical-score", type=float, default=None,
                        help="Optional current best contest score for gap computation")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        import numpy as np
    except ImportError:
        print("FATAL: numpy required", file=sys.stderr)
        return 2

    from tac.cathedral_consumers.information_theoretic_floor_consumer import (
        estimate_cramer_rao_lower_bound,
    )

    if not args.m_contest.is_file():
        print(f"FATAL: M_contest not found: {args.m_contest}", file=sys.stderr)
        return 2

    m_contest = np.load(args.m_contest)
    floor = estimate_cramer_rao_lower_bound(m_contest, mode=args.mode)

    gap = None
    if args.current_best_empirical_score is not None:
        gap = args.current_best_empirical_score - floor

    payload = {
        "schema": "estimate_information_theoretic_floor_v1",
        "exploit_id": 7,
        "exploit_name": "information_theoretic_floor",
        "mode": args.mode,
        "floor_estimate": floor,
        "current_best_empirical_score": args.current_best_empirical_score,
        "gap_to_floor": gap,
        "n_pairs": int(m_contest.shape[0]),
        "canonical_citations": [
            "Cramer-Rao bound (Cover & Thomas 2006 Ch 4)",
            "Shannon R(D) (Cover & Thomas 2006 Ch 10)",
            "Blahut 1972 IEEE Trans Info Theory",
            "Arimoto 1972 IEEE Trans Info Theory",
        ],
        "axis_tag": "[predicted]",
        "promotable": False,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Information-theoretic floor estimate (mode={args.mode}): {floor:.6e}")
        print(f"n_pairs = {payload['n_pairs']}")
        if gap is not None:
            print(f"current_best_empirical_score = {args.current_best_empirical_score}")
            print(f"gap_to_floor = {gap:.6e}")
            if gap > 0:
                print(f"  -> SATURATION_REMAINING (positive gap; further optimization possible)")
            else:
                print(f"  -> NEAR_FLOOR (zero/negative gap; consider substrate-class shift)")
        print(f"\naxis_tag: {payload['axis_tag']} (theoretical guidance; non-promotable per Catalog #341)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
