#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Operator-facing CLI for exploit #5 - per-class chroma priority.

Per RESPAWN-MG-7-BUNDLE 2026-05-20 + NSCS06 v6 -> v7 anchor.

Usage::

    .venv/bin/python tools/list_per_class_chroma_priority.py \\
        --m-contest-per-class <cls_idx>:<path.npy> [<cls_idx>:<path.npy> ...] \\
        [--json]

Each ``--m-contest-per-class`` entry is ``<class_idx>:<path.npy>`` where the
.npy is the (N_pairs, 3, H, W) class-conditioned gradient tensor from
``tac.master_gradient_comparison.multi_granularity.decompose_M_contest_per_segnet_class``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Per-SegNet-class chroma priority per exploit #5",
    )
    parser.add_argument("--m-contest-per-class", nargs="+", required=True,
                        help="One or more <class_idx>:<path.npy> pairs")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        import numpy as np
    except ImportError:
        print("FATAL: numpy required", file=sys.stderr)
        return 2

    from tac.cathedral_consumers.per_segnet_class_chroma_consumer import (
        compute_per_class_chroma_priority,
    )

    m_dict: dict[int, "object"] = {}
    for entry in args.m_contest_per_class:
        if ":" not in entry:
            print(f"FATAL: invalid entry: {entry}", file=sys.stderr)
            return 2
        cls_str, path_str = entry.split(":", 1)
        try:
            cls_idx = int(cls_str)
        except ValueError:
            print(f"FATAL: invalid class index: {cls_str}", file=sys.stderr)
            return 2
        path = Path(path_str)
        if not path.is_file():
            print(f"FATAL: class tensor not found: {path}", file=sys.stderr)
            return 2
        m_dict[cls_idx] = np.load(path)

    priority = compute_per_class_chroma_priority(m_dict)

    # Rank classes by priority descending.
    ranked = sorted(priority.items(), key=lambda kv: -kv[1])

    payload = {
        "schema": "list_per_class_chroma_priority_v1",
        "exploit_id": 5,
        "exploit_name": "per_segnet_class_chroma_priority",
        "n_classes": len(priority),
        "per_class_chroma_priority": priority,
        "ranked_classes": [
            {"class_idx": cls, "priority": p} for cls, p in ranked
        ],
        "empirical_anchor_cite": "NSCS06_v6_to_v7_chroma_optical_flow_redesign_20260516",
        "axis_tag": "[predicted]",
        "promotable": False,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Per-SegNet-class chroma priority (n={len(priority)}):")
        for i, (cls, p) in enumerate(ranked):
            print(f"  rank {i+1:3d}: class_idx={cls:3d}  priority={p:.6f}")
        print(f"\naxis_tag: {payload['axis_tag']} (chroma-allocation guidance; non-promotable per Catalog #341)")
        print(f"empirical anchor: {payload['empirical_anchor_cite']} (design-time citation; not a score claim)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
