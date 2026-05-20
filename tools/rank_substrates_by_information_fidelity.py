#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Operator-facing CLI for exploit #6 - substrate fit ranking.

Per RESPAWN-MG-7-BUNDLE 2026-05-20.

Usage::

    .venv/bin/python tools/rank_substrates_by_information_fidelity.py \\
        --m-contest <path.npy> \\
        --m-inflated-per-substrate <substrate_name>:<path.npy> [<substrate_name>:<path.npy> ...] \\
        [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Substrate fit ranking per exploit #6",
    )
    parser.add_argument("--m-contest", required=True, type=Path,
                        help="Per-pixel M_contest tensor (N_pairs, 3, H, W)")
    parser.add_argument("--m-inflated-per-substrate", nargs="+", required=True,
                        help="One or more <substrate_name>:<path.npy> pairs")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        import numpy as np
    except ImportError:
        print("FATAL: numpy required", file=sys.stderr)
        return 2

    from tac.cathedral_consumers.substrate_fit_diagnostic_consumer import (
        compute_substrate_fit_score,
    )

    if not args.m_contest.is_file():
        print(f"FATAL: M_contest not found: {args.m_contest}", file=sys.stderr)
        return 2

    m_contest = np.load(args.m_contest)

    m_inflated_dict: dict[str, "object"] = {}
    for entry in args.m_inflated_per_substrate:
        if ":" not in entry:
            print(f"FATAL: invalid --m-inflated-per-substrate entry: {entry}",
                  file=sys.stderr)
            return 2
        name, path_str = entry.split(":", 1)
        path = Path(path_str)
        if not path.is_file():
            print(f"FATAL: M_inflated file not found for {name}: {path}", file=sys.stderr)
            return 2
        m_inflated_dict[name] = np.load(path)

    fit_scores = compute_substrate_fit_score(m_contest, m_inflated_dict)

    # Rank substrates by fit score descending.
    ranked = sorted(fit_scores.items(), key=lambda kv: -kv[1])

    payload = {
        "schema": "rank_substrates_by_information_fidelity_v1",
        "exploit_id": 6,
        "exploit_name": "substrate_fit_diagnostic",
        "n_substrates": len(fit_scores),
        "substrate_fit_scores": fit_scores,
        "ranked_substrates": [
            {"substrate_name": name, "fit_score": score} for name, score in ranked
        ],
        "axis_tag": "[predicted]",
        "promotable": False,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Substrate fit ranking (n={len(fit_scores)}):")
        for i, (name, score) in enumerate(ranked):
            print(f"  rank {i+1:3d}: {name:40s} fit_score={score:.6f}")
        print(f"\naxis_tag: {payload['axis_tag']}  (advisory; non-promotable per Catalog #341)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
