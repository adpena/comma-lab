#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Operator-facing CLI for exploit #8 - top-K score-critical bits.

Per RESPAWN-MG-7-BUNDLE 2026-05-20 + Catalog #318.

Usage::

    .venv/bin/python tools/list_score_critical_bits.py \\
        --m-archive <path.npy> --k-top-bits N \\
        [--axis 0|1|2] [--msb-aligned] [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Top-K score-critical bits per exploit #8",
    )
    parser.add_argument("--m-archive", required=True, type=Path,
                        help="Per-byte M_archive (N_bytes, 3) - chain-rule per Catalog #318")
    parser.add_argument("--k-top-bits", type=int, required=True)
    parser.add_argument("--axis", type=int, choices=[0, 1, 2], default=None)
    parser.add_argument("--msb-aligned", action="store_true", default=True,
                        help="Use MSB-aligned bit weighting (default True)")
    parser.add_argument("--lsb-aligned", action="store_true",
                        help="Override to LSB-aligned weighting")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        import numpy as np
    except ImportError:
        print("FATAL: numpy required", file=sys.stderr)
        return 2

    from tac.cathedral_consumers.bit_level_score_critical_bits_consumer import (
        extract_M_archive_bit_level,
        rank_score_critical_bits,
    )

    if not args.m_archive.is_file():
        print(f"FATAL: M_archive not found: {args.m_archive}", file=sys.stderr)
        return 2

    msb_aligned = not args.lsb_aligned

    m_archive = np.load(args.m_archive)
    m_bit_level = extract_M_archive_bit_level(m_archive, msb_aligned=msb_aligned)
    top_k_bits = rank_score_critical_bits(m_bit_level, args.k_top_bits, axis=args.axis)

    # Decompose bit indices into (byte_idx, bit_pos) for readability.
    bit_locations = []
    for bit_idx in top_k_bits:
        byte_idx = bit_idx // 8
        bit_pos = bit_idx % 8
        bit_locations.append(
            {"bit_idx": bit_idx, "byte_idx": byte_idx, "bit_pos_in_byte": bit_pos}
        )

    payload = {
        "schema": "list_score_critical_bits_v1",
        "exploit_id": 8,
        "exploit_name": "bit_level_score_critical_bits",
        "n_bytes_total": int(m_archive.shape[0]),
        "n_bits_total": int(m_archive.shape[0] * 8),
        "k_top_bits": args.k_top_bits,
        "axis": args.axis,
        "msb_aligned": msb_aligned,
        "axis_label": (
            ["seg", "pose", "rate"][args.axis] if args.axis is not None else "aggregate_L2"
        ),
        "score_critical_bit_indices": top_k_bits,
        "bit_locations": bit_locations,
        "catalog_318_chain_rule_cited": True,
        "axis_tag": "[predicted]",
        "promotable": False,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Top-{args.k_top_bits} score-critical bits "
              f"(axis={payload['axis_label']}, msb_aligned={msb_aligned}):")
        for i, loc in enumerate(bit_locations[:20]):
            magnitude = (
                float(np.linalg.norm(m_bit_level[loc['bit_idx']]))
                if args.axis is None
                else float(abs(m_bit_level[loc['bit_idx'], args.axis]))
            )
            print(f"  rank {i+1:4d}: bit_idx={loc['bit_idx']:9d} "
                  f"(byte={loc['byte_idx']:8d}, bit_pos={loc['bit_pos_in_byte']}) "
                  f"sensitivity={magnitude:.6e}")
        if len(bit_locations) > 20:
            print(f"  ... ({len(bit_locations) - 20} more)")
        print(f"\naxis_tag: {payload['axis_tag']} (non-promotable per Catalog #341)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
