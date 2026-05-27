# SPDX-License-Identifier: MIT
"""Operator-facing CLI: apply canvas operation generator → candidate manifest.

Per PAIR-FRAME-SCORER-GEOMETRY-LATTICE BUILD-2 + BUILD-3 sister subagent
op-routable per
``.omx/research/drop_many_replace_composition_apparatus_state_audit_20260526.md``
PRIORITY 2.

This CLI is the canonical operator entrypoint for the 4 canonical operations
on the 5D canvas (full-drop / repair / masked / feathered). It consumes a
canvas populated by BUILD-1 (currently scaffold-only; this CLI accepts a
canonical JSON canvas file emitted by BUILD-1's reader) and emits per-
operation ExecutableCandidate rows ranked by predicted ΔS.

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA": this CLI emits
PREDICTED candidates per Catalog #287 axis_tag="[predicted]" + Catalog #341
canonical-routing-markers (Tier A non-promotable); PAID DISPATCH per FIRE
phase audits/operator-routable #4 of the audit memo's PRIORITY 4.

Usage examples
--------------

Apply FULL_DROP operation to a canvas JSON file::

    .venv/bin/python tools/apply_operation_to_5d_canvas_cli.py \\
        --canvas-input experiments/results/canvas_a1b2c3.json \\
        --operation full-drop \\
        --output-archive .omx/state/pair_frame_scorer_geometry_lattice_candidates/full_drop.json \\
        --top-n 32

Emit JSON output for machine consumption::

    .venv/bin/python tools/apply_operation_to_5d_canvas_cli.py \\
        --canvas-input ... --operation repair --json

Exit codes
----------

- 0: CLEAN — operation completed; candidates emitted to ``--output-archive``
- 1: OPERATION-FAILED — operation raised at runtime (e.g. canvas empty)
- 2: CANVAS-INVALID — canvas JSON failed parsing or schema validation
- 3: CLI error (argparse / IO / unrecognized flag)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas import (
    CANVAS_SCHEMA,
    CanonicalOperation,
    CpuCudaAxis,
    ExecutableCandidate,
    PairFrameScorerGeometryCell,
    PairFrameScorerGeometryLattice,
    ReceiverRuntime,
    ScorerAxis,
)

_EXIT_CLEAN = 0
_EXIT_OPERATION_FAILED = 1
_EXIT_CANVAS_INVALID = 2
_EXIT_CLI_ERROR = 3

_POPULATED_CANVAS_SCHEMA = "pair_frame_scorer_geometry_lattice_5d_canvas_populated_v1"


_OPERATION_CLI_MAP: dict[str, CanonicalOperation] = {
    "full-drop": CanonicalOperation.FULL_DROP,
    "full_drop": CanonicalOperation.FULL_DROP,
    "repair": CanonicalOperation.REPAIR,
    "masked": CanonicalOperation.MASKED,
    "feathered": CanonicalOperation.FEATHERED,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="apply_operation_to_5d_canvas_cli",
        description=(
            "Apply canonical operation generator (full-drop / repair / "
            "masked / feathered) to a populated 5D canvas; emit ranked "
            "ExecutableCandidate manifest. Per BUILD-2 + BUILD-3 + audit "
            "memo PRIORITY 2."
        ),
    )
    parser.add_argument(
        "--canvas-input",
        type=Path,
        required=True,
        help=(
            "Path to populated canvas JSON (emitted by BUILD-1 reader; "
            "schema: pair_frame_scorer_geometry_lattice_5d_canvas.v0_scaffold)"
        ),
    )
    parser.add_argument(
        "--operation",
        choices=sorted(_OPERATION_CLI_MAP.keys()),
        required=True,
        help="Canonical operation to apply (full-drop / repair / masked / feathered)",
    )
    parser.add_argument(
        "--output-archive",
        type=Path,
        help=(
            "Path to write the canonical candidate manifest JSON (parent "
            "dir created if missing; sorted-keys deterministic output)"
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=32,
        help="Maximum candidates to emit (default: 32)",
    )
    parser.add_argument(
        "--receiver-runtime",
        choices=[m.value for m in ReceiverRuntime],
        default=None,
        help=(
            "Override receiver runtime (default: per-operation canonical "
            "mode per design memo §DELIVERABLE 1)"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON to stdout (in addition to file write)",
    )
    return parser


def _load_canvas_from_json(path: Path) -> PairFrameScorerGeometryLattice:
    """Load canvas from canonical JSON emitted by BUILD-1 reader.

    Schema (per design memo + canvas SCAFFOLD):

        {
          "schema": CANVAS_SCHEMA,
          "archive_sha256": "<sha>",
          "cells": [<cell_dict>, ...]
        }

    Each cell follows ``PairFrameScorerGeometryCell.as_dict()`` per
    CELL_SCHEMA.
    """
    if not path.exists():
        raise FileNotFoundError(f"canvas JSON not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"canvas JSON malformed at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"canvas JSON root must be object, got {type(payload).__name__}")
    schema = payload.get("schema")
    if schema not in {
        CANVAS_SCHEMA,
        "pair_frame_scorer_geometry_lattice_5d_canvas.v0_scaffold",
        _POPULATED_CANVAS_SCHEMA,
    }:
        raise ValueError(
            f"canvas JSON schema mismatch: got {schema!r}, "
            f"expected {CANVAS_SCHEMA!r}"
        )
    archive_sha256 = payload.get("archive_sha256")
    if not isinstance(archive_sha256, str) or not archive_sha256:
        raise ValueError("canvas JSON must carry non-empty archive_sha256")
    cells_payload = payload.get("cells", [])
    if not isinstance(cells_payload, list):
        raise ValueError(
            f"canvas JSON cells must be list, got {type(cells_payload).__name__}"
        )
    canvas = PairFrameScorerGeometryLattice(archive_sha256)
    for idx, cell_dict in enumerate(cells_payload):
        if not isinstance(cell_dict, dict):
            raise ValueError(f"canvas JSON cells[{idx}] must be object")
        try:
            cell = PairFrameScorerGeometryCell(
                pair_idx=int(cell_dict["pair_idx"]),
                frame_idx=int(cell_dict["frame_idx"]),
                scorer_axis=ScorerAxis(cell_dict["scorer_axis"]),
                receiver_runtime=ReceiverRuntime(cell_dict["receiver_runtime"]),
                cpu_cuda_axis=CpuCudaAxis(cell_dict["cpu_cuda_axis"]),
                predicted_delta_score=float(cell_dict["predicted_delta_score"]),
                predicted_byte_cost=int(cell_dict["predicted_byte_cost"]),
                receiver_feasibility=bool(cell_dict["receiver_feasibility"]),
                catalog_323_provenance=cell_dict.get("catalog_323_provenance", {}),
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"canvas JSON cells[{idx}] failed validation: {exc}"
            ) from exc
        canvas._cells[cell.coordinate] = cell
    return canvas


def _invoke_operation(
    canvas: PairFrameScorerGeometryLattice,
    operation: CanonicalOperation,
    top_n: int,
    receiver_runtime_override: ReceiverRuntime | None,
) -> list[ExecutableCandidate]:
    """Dispatch to the canonical operation generator method on the canvas."""
    if operation is CanonicalOperation.FULL_DROP:
        if receiver_runtime_override is not None:
            return canvas.generate_full_drop_starts(
                top_n=top_n, receiver_runtime=receiver_runtime_override
            )
        return canvas.generate_full_drop_starts(top_n=top_n)
    if operation is CanonicalOperation.REPAIR:
        if receiver_runtime_override is not None:
            return canvas.generate_repair_starts(
                top_n=top_n, receiver_runtime=receiver_runtime_override
            )
        return canvas.generate_repair_starts(top_n=top_n)
    if operation is CanonicalOperation.MASKED:
        if receiver_runtime_override is not None:
            return canvas.generate_masked_starts(
                top_n=top_n, receiver_runtime=receiver_runtime_override
            )
        return canvas.generate_masked_starts(top_n=top_n)
    if operation is CanonicalOperation.FEATHERED:
        if receiver_runtime_override is not None:
            return canvas.generate_feathered_starts(
                top_n=top_n, receiver_runtime=receiver_runtime_override
            )
        return canvas.generate_feathered_starts(top_n=top_n)
    raise ValueError(f"unrecognized operation: {operation}")


def _build_manifest_payload(
    canvas: PairFrameScorerGeometryLattice,
    operation: CanonicalOperation,
    receiver_runtime_override: ReceiverRuntime | None,
    candidates: list[ExecutableCandidate],
) -> dict[str, Any]:
    """Canonical candidate manifest JSON payload (sorted-keys deterministic)."""
    return {
        "schema": "pair_frame_scorer_geometry_lattice_candidate_manifest.v0",
        "archive_sha256": canvas.archive_sha256,
        "operation": operation.value,
        "receiver_runtime_override": (
            receiver_runtime_override.value if receiver_runtime_override else None
        ),
        "candidate_count": len(candidates),
        "candidates": [c.as_dict() for c in candidates],
        "canvas_cell_count": canvas.cell_count(),
        "produced_by": "tools/apply_operation_to_5d_canvas_cli.py",
        "design_memo": (
            ".omx/research/pair_frame_scorer_geometry_lattice_design_memo_20260525.md"
        ),
        "audit_memo": (
            ".omx/research/drop_many_replace_composition_apparatus_state_audit_20260526.md"
        ),
        "build_memo": (
            ".omx/research/build_2_3_operation_generators_landed_20260526.md"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        return _EXIT_CLI_ERROR

    try:
        canvas = _load_canvas_from_json(args.canvas_input)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[apply_operation_to_5d_canvas_cli] CANVAS-INVALID: {exc}", file=sys.stderr)
        return _EXIT_CANVAS_INVALID

    operation = _OPERATION_CLI_MAP[args.operation]
    receiver_runtime_override = (
        ReceiverRuntime(args.receiver_runtime) if args.receiver_runtime else None
    )
    try:
        candidates = _invoke_operation(
            canvas, operation, args.top_n, receiver_runtime_override
        )
    except (ValueError, RuntimeError) as exc:
        print(
            f"[apply_operation_to_5d_canvas_cli] OPERATION-FAILED: "
            f"operation={operation.value} reason={exc}",
            file=sys.stderr,
        )
        return _EXIT_OPERATION_FAILED

    payload = _build_manifest_payload(
        canvas, operation, receiver_runtime_override, candidates
    )

    if args.output_archive:
        args.output_archive.parent.mkdir(parents=True, exist_ok=True)
        args.output_archive.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"[apply_operation_to_5d_canvas_cli] CLEAN: "
            f"operation={operation.value} archive_sha256={canvas.archive_sha256[:12]}... "
            f"candidates={len(candidates)} canvas_cells={canvas.cell_count()}",
            file=sys.stderr,
        )

    return _EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
