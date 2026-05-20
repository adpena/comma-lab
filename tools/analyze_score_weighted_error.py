#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Operator-facing CLI for exploit #2 - score-weighted reconstruction error analysis.

Per RESPAWN-MG-7-BUNDLE 2026-05-20. Runs the canonical
``compute_score_weighted_reconstruction_loss`` helper on a substrate
candidate's inflated video + the contest video + the per-pixel M_contest
tensor and emits a per-pair ranking.

Usage::

    .venv/bin/python tools/analyze_score_weighted_error.py \\
        --inflated-video <path.npy> --contest-video <path.npy> \\
        --m-contest <path.npy> [--json] [--top-k N]

Per CLAUDE.md "Apples-to-apples evidence discipline": the output is
[predicted] axis tag; non-promotable by construction.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score-weighted reconstruction error per exploit #2",
    )
    parser.add_argument("--inflated-video", required=True, type=Path)
    parser.add_argument("--contest-video", required=True, type=Path)
    parser.add_argument("--m-contest", required=True, type=Path,
                        help="Per-pixel M_contest tensor (N_pairs, 3, H, W) from "
                             "tac.master_gradient_comparison.multi_granularity.extract_M_contest")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Top-K worst pairs to surface (by score-weighted error)")
    parser.add_argument("--json", action="store_true", help="Emit JSON to stdout")
    args = parser.parse_args(argv)

    try:
        import numpy as np
    except ImportError:
        print("FATAL: numpy required", file=sys.stderr)
        return 2

    from tac.cathedral_consumers.score_weighted_reconstruction_error_consumer import (
        compute_score_weighted_reconstruction_loss,
    )

    for path in (args.inflated_video, args.contest_video, args.m_contest):
        if not path.is_file():
            print(f"FATAL: input file not found: {path}", file=sys.stderr)
            return 2

    inflated = np.load(args.inflated_video)
    contest = np.load(args.contest_video)
    m_contest = np.load(args.m_contest)

    total_loss = compute_score_weighted_reconstruction_loss(inflated, contest, m_contest)

    # Per-pair ranking.
    diff = (inflated - contest).astype(np.float64)
    err_per_pixel = np.sum(np.square(diff), axis=1)
    m_l2_sq = np.sum(np.square(m_contest.astype(np.float64)), axis=1)
    weighted = err_per_pixel * m_l2_sq
    per_pair = weighted.mean(axis=(1, 2))
    order = np.argsort(-per_pair, kind="stable")
    top_k_pairs = [
        {"pair_idx": int(i), "score_weighted_error": float(per_pair[i])}
        for i in order[: args.top_k]
    ]

    payload = {
        "schema": "analyze_score_weighted_error_v1",
        "exploit_id": 2,
        "exploit_name": "score_weighted_reconstruction_error",
        "n_pairs": int(inflated.shape[0]),
        "score_weighted_error_total": total_loss,
        "score_weighted_error_per_pair_mean": float(per_pair.mean()),
        "score_weighted_error_per_pair_max": float(per_pair.max()),
        "top_k_worst_pairs": top_k_pairs,
        "axis_tag": "[predicted]",
        "promotable": False,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"score_weighted_error_total = {total_loss:.6e}")
        print(f"n_pairs = {payload['n_pairs']}")
        print(f"per-pair mean = {payload['score_weighted_error_per_pair_mean']:.6e}")
        print(f"per-pair max  = {payload['score_weighted_error_per_pair_max']:.6e}")
        print(f"\nTop-{args.top_k} worst pairs (by score-weighted error):")
        for entry in top_k_pairs:
            print(f"  pair {entry['pair_idx']:4d}: {entry['score_weighted_error']:.6e}")
        print(f"\naxis_tag: {payload['axis_tag']}  (non-promotable per Catalog #341)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
