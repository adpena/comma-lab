#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a fail-closed MPS research-signal manifest.

This is the sanctioned path for using local Apple MPS as a free discovery
device. The output is useful for curve fitting and candidate generation, but it
is deliberately non-promotable: ``score_claim=false``,
``promotion_eligible=false``, and ``ready_for_exact_eval_dispatch=false``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.optimization.meta_lagrangian_allocator import build_atom_ledger  # noqa: E402
from tac.optimization.mps_research_signal import (  # noqa: E402
    build_mps_research_signal_manifest,
    json_text,
    load_observations,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True, help="JSON or JSONL MPS observation rows")
    parser.add_argument("--output", type=Path, required=True, help="Manifest JSON output")
    parser.add_argument("--run-id", required=True, help="Stable run id for provenance")
    parser.add_argument("--source", help="Source label/path for the observation set")
    parser.add_argument("--anchor-d-seg", type=float)
    parser.add_argument("--anchor-d-pose", type=float)
    parser.add_argument("--anchor-archive-bytes", type=int)
    parser.add_argument(
        "--atom-ledger-output",
        type=Path,
        help="Optional meta-Lagrangian ledger output built from proxy atoms",
    )
    parser.add_argument(
        "--base-pose-dist",
        type=float,
        default=0.01,
        help="Base pose distance for optional atom-ledger planning deltas",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    observations = load_observations(args.observations)
    manifest = build_mps_research_signal_manifest(
        observations,
        source=args.source or args.observations.as_posix(),
        run_id=args.run_id,
        anchor_d_seg=args.anchor_d_seg,
        anchor_d_pose=args.anchor_d_pose,
        anchor_archive_bytes=args.anchor_archive_bytes,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json_text(manifest), encoding="utf-8")

    if args.atom_ledger_output:
        ledger = build_atom_ledger(
            manifest["meta_lagrangian_atoms"],
            base_pose_dist=args.base_pose_dist,
            source=args.output.as_posix(),
        )
        args.atom_ledger_output.parent.mkdir(parents=True, exist_ok=True)
        args.atom_ledger_output.write_text(json_text(ledger), encoding="utf-8")

    print(f"wrote {args.output}")
    if args.atom_ledger_output:
        print(f"wrote {args.atom_ledger_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
