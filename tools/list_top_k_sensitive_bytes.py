#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Operator-facing CLI for exploit #3 - top-K most-sensitive archive bytes.

Per RESPAWN-MG-7-BUNDLE 2026-05-20 + Catalog #318 chain-rule discipline.

Usage::

    .venv/bin/python tools/list_top_k_sensitive_bytes.py \\
        --m-archive <path.npy> --k-top N [--axis 0|1|2] [--json]

``--m-archive`` MUST be a (N_bytes, 3) tensor derived via the canonical chain
rule (``tac.master_gradient_comparison.multi_granularity.extract_M_archive_via_chain_rule``);
raw bit-flip FD is FORBIDDEN per Catalog #318.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Top-K sensitive archive bytes per exploit #3",
    )
    parser.add_argument("--m-archive", required=True, type=Path,
                        help="Per-byte M_archive tensor (N_bytes, 3) - chain-rule-derived per Catalog #318")
    parser.add_argument("--k-top", type=int, required=True)
    parser.add_argument("--axis", type=int, choices=[0, 1, 2], default=None,
                        help="Axis to rank by (0=seg, 1=pose, 2=rate); default aggregate L2")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        import numpy as np
    except ImportError:
        print("FATAL: numpy required", file=sys.stderr)
        return 2

    from tac.cathedral_consumers.top_k_byte_sensitivity_consumer import (
        rank_archive_bytes_by_sensitivity,
    )

    if not args.m_archive.is_file():
        print(f"FATAL: M_archive file not found: {args.m_archive}", file=sys.stderr)
        return 2

    m_archive = np.load(args.m_archive)
    top_k = rank_archive_bytes_by_sensitivity(m_archive, args.k_top, axis=args.axis)

    payload = {
        "schema": "list_top_k_sensitive_bytes_v1",
        "exploit_id": 3,
        "exploit_name": "top_k_byte_sensitivity",
        "n_bytes_total": int(m_archive.shape[0]),
        "k_top": args.k_top,
        "axis": args.axis,
        "axis_label": (
            ["seg", "pose", "rate"][args.axis] if args.axis is not None else "aggregate_L2"
        ),
        "top_k_byte_indices": top_k,
        "catalog_318_chain_rule_cited": True,
        "axis_tag": "[predicted]",
        "promotable": False,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Top-{args.k_top} most-sensitive bytes (axis={payload['axis_label']}):")
        for i, byte_idx in enumerate(top_k):
            magnitude = (
                float(np.linalg.norm(m_archive[byte_idx]))
                if args.axis is None
                else float(abs(m_archive[byte_idx, args.axis]))
            )
            print(f"  rank {i+1:4d}: byte_idx={byte_idx:8d} sensitivity={magnitude:.6e}")
        print(f"\naxis_tag: {payload['axis_tag']} (non-promotable per Catalog #341)")
        print("Catalog #318 chain-rule discipline cited.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
