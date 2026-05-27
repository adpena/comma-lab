#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Operator-facing CLI for the BUILD-1 5D canvas populator.

Canonical entry point per BUILD-1 routing directive 2026-05-26:

    $ .venv/bin/python tools/populate_5d_canvas_cli.py --list-archives
    $ .venv/bin/python tools/populate_5d_canvas_cli.py \\
        --archive-sha256 6bae0201fb08...  --write-sidecar
    $ .venv/bin/python tools/populate_5d_canvas_cli.py \\
        --latest --write-sidecar --summary

The CLI is observability-only per Catalog #341 + Tier A scaffold; the
output manifest carries canonical non-promotable markers per Catalog
#341 (predicted_delta_adjustment=0.0 / promotable=False /
axis_tag="[predicted]" / score_claim=False / etc.). Operator-facing
audit surface for the 5D canvas empirical population.

Exit codes:
    0 - successful population
    1 - PopulatorError (no anchors, ledger corrupt, etc.)
    2 - CLI argument error

Per CLAUDE.md "Beauty, simplicity, and developer experience": thin
delegation to canonical
`tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas_populator`.

Per CLAUDE.md "Operator gates must be wired and used": this CLI is the
operator-facing wiring of the canonical populator.

Sister of:
- `tools/list_canonical_equations.py` (canonical CLI pattern)
- `tools/refresh_canonical_frontier.py` (canonical CLI pattern)
- `tools/audit_provenance_compliance.py` (operator-facing audit pattern)

Lane: `lane_build_1_populate_5d_canvas_20260526` L1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src/ on sys.path for direct invocation
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="populate_5d_canvas_cli",
        description=(
            "BUILD-1 5D canvas empirical populator CLI. Reads "
            ".omx/state/master_gradient_anchors.jsonl + writes canonical "
            "sidecar at .omx/state/pair_frame_scorer_geometry_lattice/."
        ),
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--list-archives",
        action="store_true",
        help="List distinct archive sha256 values in the master gradient ledger",
    )
    mode_group.add_argument(
        "--archive-sha256",
        type=str,
        default=None,
        help="Full archive sha256 to populate the canvas for",
    )
    mode_group.add_argument(
        "--latest",
        action="store_true",
        help=(
            "Populate the canvas for the LATEST archive in the ledger "
            "(by measurement_utc)"
        ),
    )

    parser.add_argument(
        "--ledger-path",
        type=Path,
        default=None,
        help=(
            "Override ledger path (default: "
            "<repo_root>/.omx/state/master_gradient_anchors.jsonl)"
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Override sidecar output path (default: "
            "<repo_root>/.omx/state/pair_frame_scorer_geometry_lattice/"
            "<sha[:12]>_<utc>.json)"
        ),
    )
    parser.add_argument(
        "--no-sidecar",
        action="store_true",
        help="Do not persist sidecar JSON (smoke / dry-run mode)",
    )
    parser.add_argument(
        "--include-non-authoritative",
        action="store_true",
        help=(
            "Include non-authoritative advisory axes "
            "([macOS-CPU advisory] / [MPS-PROXY] / etc.) in the canvas. "
            "DEFAULT: skip per CLAUDE.md 'MPS auth eval is NOISE' + "
            "'Submission auth eval - BOTH CPU AND CUDA' non-negotiables."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON manifest summary",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Emit human-readable summary (default if --json not set)",
    )

    return parser


def _emit_list_archives(archives: list[str], json_mode: bool) -> int:
    if json_mode:
        print(json.dumps({"archives": archives}, indent=2, sort_keys=True))
    else:
        print(f"Distinct archive sha256 values in ledger ({len(archives)}):")
        for sha in archives:
            print(f"  {sha[:12]}...  ({sha})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Lazy import so --help is fast.
    try:
        from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas_populator import (
            PopulatorError,
            list_distinct_archives_in_ledger,
            populate_5d_canvas_from_master_gradient_anchors,
        )
    except ImportError as exc:
        print(
            f"[populate_5d_canvas_cli] ERROR: failed to import canonical "
            f"populator: {exc}",
            file=sys.stderr,
        )
        return 1

    if args.list_archives:
        try:
            archives = list_distinct_archives_in_ledger(
                ledger_path=args.ledger_path
            )
        except PopulatorError as exc:
            print(
                f"[populate_5d_canvas_cli] PopulatorError: {exc}",
                file=sys.stderr,
            )
            return 1
        return _emit_list_archives(archives, json_mode=args.json)

    # Population mode: --archive-sha256 or --latest.
    target_sha = args.archive_sha256 if not args.latest else None

    try:
        manifest = populate_5d_canvas_from_master_gradient_anchors(
            archive_sha256=target_sha,
            ledger_path=args.ledger_path,
            output_path=args.output_path,
            write_sidecar=not args.no_sidecar,
            skip_non_authoritative=not args.include_non_authoritative,
        )
    except PopulatorError as exc:
        print(
            f"[populate_5d_canvas_cli] PopulatorError: {exc}",
            file=sys.stderr,
        )
        return 1

    if args.json:
        print(json.dumps(manifest.as_dict(), indent=2, sort_keys=True))
    elif args.summary or not args.json:
        # Default: human-readable summary.
        print("=" * 72)
        print("5D canvas populator manifest (Tier A observability-only)")
        print("=" * 72)
        print(f"archive_sha256: {manifest.archive_sha256}")
        print(f"cells_populated: {manifest.cells_populated}")
        print(f"anchors_consumed: {manifest.anchors_consumed}")
        print(
            f"anchors_skipped_non_authoritative: "
            f"{manifest.anchors_skipped_non_authoritative}"
        )
        print(f"output_path: {manifest.output_path}")
        print()
        print("Canonical non-promotable markers per Catalog #341:")
        print("  predicted_delta_adjustment: 0.0")
        print("  promotable: False")
        print("  axis_tag: '[predicted]'")
        print()
        print(
            "BUILD-1 canonical empirical populator. Per Catalog #357: "
            "Tier A observability-only at scaffold landing. BUILD-4 "
            "sister subagent op-routable promotes to Tier B with "
            "canonical-routing markers per Catalog #341."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
