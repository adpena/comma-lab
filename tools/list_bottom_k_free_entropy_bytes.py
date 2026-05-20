#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Operator-facing CLI for exploit #4 - bottom-K free-entropy archive bytes.

Per RESPAWN-MG-7-BUNDLE 2026-05-20 + Catalog #318 + Catalog #220.

Usage::

    .venv/bin/python tools/list_bottom_k_free_entropy_bytes.py \\
        --m-archive <path.npy> --k-bottom N \\
        [--sensitivity-threshold 1e-6] [--axis 0|1|2] [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bottom-K free-entropy archive bytes per exploit #4",
    )
    parser.add_argument("--m-archive", required=True, type=Path)
    parser.add_argument("--k-bottom", type=int, required=True)
    parser.add_argument("--sensitivity-threshold", type=float, default=1e-6)
    parser.add_argument("--axis", type=int, choices=[0, 1, 2], default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        import numpy as np
    except ImportError:
        print("FATAL: numpy required", file=sys.stderr)
        return 2

    from tac.cathedral_consumers.bottom_k_free_entropy_byte_consumer import (
        rank_archive_bytes_by_low_sensitivity,
    )

    if not args.m_archive.is_file():
        print(f"FATAL: M_archive file not found: {args.m_archive}", file=sys.stderr)
        return 2

    m_archive = np.load(args.m_archive)
    bottom_k = rank_archive_bytes_by_low_sensitivity(
        m_archive, args.k_bottom, args.sensitivity_threshold, axis=args.axis,
    )

    payload = {
        "schema": "list_bottom_k_free_entropy_bytes_v1",
        "exploit_id": 4,
        "exploit_name": "bottom_k_free_entropy_bytes",
        "n_bytes_total": int(m_archive.shape[0]),
        "k_bottom_requested": args.k_bottom,
        "k_bottom_returned": len(bottom_k),
        "sensitivity_threshold": args.sensitivity_threshold,
        "axis": args.axis,
        "axis_label": (
            ["seg", "pose", "rate"][args.axis] if args.axis is not None else "aggregate_L2"
        ),
        "bottom_k_byte_indices": bottom_k,
        "catalog_318_chain_rule_cited": True,
        "catalog_220_operational_mechanism_cited": True,
        "axis_tag": "[predicted]",
        "promotable": False,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Bottom-{args.k_bottom} free-entropy bytes (threshold={args.sensitivity_threshold}):")
        print(f"  Returned {len(bottom_k)} bytes below threshold.")
        for i, byte_idx in enumerate(bottom_k[:20]):
            magnitude = (
                float(np.linalg.norm(m_archive[byte_idx]))
                if args.axis is None
                else float(abs(m_archive[byte_idx, args.axis]))
            )
            print(f"  rank {i+1:4d}: byte_idx={byte_idx:8d} sensitivity={magnitude:.6e}")
        if len(bottom_k) > 20:
            print(f"  ... ({len(bottom_k) - 20} more)")
        print(f"\naxis_tag: {payload['axis_tag']} (non-promotable per Catalog #341)")
        print("Catalog #220 operational-mechanism: bytes must be verified safe to coarsen before adoption.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
