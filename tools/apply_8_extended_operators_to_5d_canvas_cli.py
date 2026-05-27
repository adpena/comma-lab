#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Apply 8 NOT-BUILT extended operators to the 5D canvas — operator-facing CLI.

Sister of ``tools/apply_operation_to_5d_canvas_cli.py`` (sister BUILD-2+3 CLI
covering the canonical 4 operations FULL_DROP / REPAIR / MASKED / FEATHERED).
This CLI covers the 8 NOT-BUILT operators enumerated in the audit memo
``.omx/research/drop_many_replace_composition_apparatus_state_audit_20260526.md``
§"Phase 1 catalog enumeration":

    1. replace-one         (DISTORTION-axis primitive)
    2. replace-many        (beam-search per Catalog #356)
    3. merge-pair          (rate+distortion joint optimization)
    4. reorder-pair        (entropy-coder context optimization)
    5. drop-frame          (per-frame drop; finer-grained than pair-level)
    6. synthesize-frame    (per-frame synthesis per Atick-Redlich)
    7. motion-conditional  (per-pair conditional per Rao-Ballard)
    8. temporal-coherence  (cross-pair joint per Wyner-Ziv)

The CLI consumes a populated 5D canvas (from BUILD-1 sister subagent) and
emits ranked ``ExecutableCandidate`` JSON manifests for downstream FIRE-phase
paired CPU+CUDA dispatch per CLAUDE.md "Submission auth eval — BOTH CPU AND
CUDA".

Per CLAUDE.md "Operator gates must be wired and used" + Catalog #335 +
Catalog #357: the CLI is non-promotable (Tier A observability-only); every
emitted candidate carries canonical-routing markers per Catalog #341
(``predicted_delta_adjustment=0.0`` + ``promotable=False`` + ``axis_tag="[predicted]"``).

USAGE:

    # List all 8 operators
    python tools/apply_8_extended_operators_to_5d_canvas_cli.py --list

    # Replace one specific pair's selector
    python tools/apply_8_extended_operators_to_5d_canvas_cli.py \
        --canvas-path .omx/state/pair_frame_scorer_geometry_lattice/<archive_sha[:12]>_<utc>.json \
        --operator replace_one \
        --target-pair-idx 371 \
        --alternative-selector-id 5 \
        --top-n 16

    # Replace many with beam search
    python tools/apply_8_extended_operators_to_5d_canvas_cli.py \
        --canvas-path <path> \
        --operator replace_many \
        --beam-width 8 --beam-depth 4 \
        --top-n 16

    # All other operators per --help
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas import (
    CANVAS_SCHEMA,
    CpuCudaAxis,
    PairFrameScorerGeometryCell,
    PairFrameScorerGeometryLattice,
    ReceiverRuntime,
    ScorerAxis,
)
from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas_extended_operators import (
    CONSUMER_NAME,
    CONSUMER_VERSION,
    DEFAULT_BEAM_DEPTH,
    DEFAULT_BEAM_WIDTH,
    DEFAULT_MERGE_MAX_CANDIDATES,
    DEFAULT_MOTION_THRESHOLD_PERCENTILE,
    DEFAULT_TEMPORAL_COHERENCE_WINDOW,
    DEFAULT_TOP_N,
    EXTENDED_OPERATION_CANONICAL_EQUATION_IDS,
    EXTENDED_OPERATOR_REGISTRY,
    DropFrameParameters,
    ExtendedOperation,
    MergePairParameters,
    MotionConditionalParameters,
    ReorderPairParameters,
    ReplaceManyParameters,
    ReplaceOneParameters,
    SynthesizeFrameParameters,
    TemporalCoherenceParameters,
)

_POPULATED_CANVAS_SCHEMA = "pair_frame_scorer_geometry_lattice_5d_canvas_populated_v1"
_ALL_OPERATORS_VALUE = "all"


def _load_canvas_from_json(canvas_path: Path) -> PairFrameScorerGeometryLattice:
    """Load a populated 5D canvas from a JSON manifest.

    The expected schema mirrors the canvas's ``as_dict()`` output:
    ``{"archive_sha256": "...", "cells": [<cell.as_dict()>, ...]}``.
    Per BUILD-1 sister subagent: the canvas is populated empirically
    from ``.omx/state/master_gradient_anchors.jsonl`` + sister Cable D
    master-gradient consumer rows.
    """
    if not canvas_path.exists():
        raise SystemExit(
            f"[apply_8_extended_operators_to_5d_canvas_cli] FATAL: "
            f"canvas-path does not exist: {canvas_path}"
        )
    with canvas_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    schema = payload.get("schema")
    if schema not in (CANVAS_SCHEMA, _POPULATED_CANVAS_SCHEMA):
        raise SystemExit(
            "[apply_8_extended_operators_to_5d_canvas_cli] FATAL: "
            f"canvas JSON schema must be {CANVAS_SCHEMA!r} or "
            f"{_POPULATED_CANVAS_SCHEMA!r}; got {schema!r}"
        )
    archive_sha256 = payload.get("archive_sha256")
    if not isinstance(archive_sha256, str) or len(archive_sha256) != 64:
        raise SystemExit(
            "[apply_8_extended_operators_to_5d_canvas_cli] FATAL: "
            "canvas JSON missing valid archive_sha256 (sha256 hex 64 chars)"
        )
    cells_payload = payload.get("cells", [])
    if not isinstance(cells_payload, list):
        raise SystemExit(
            "[apply_8_extended_operators_to_5d_canvas_cli] FATAL: "
            "canvas JSON 'cells' must be a list"
        )
    cells_map: dict[tuple, PairFrameScorerGeometryCell] = {}
    for cell_dict in cells_payload:
        if not isinstance(cell_dict, dict):
            raise SystemExit(
                "[apply_8_extended_operators_to_5d_canvas_cli] FATAL: "
                "canvas JSON cells must be objects"
            )
        cell = PairFrameScorerGeometryCell(
            pair_idx=int(cell_dict["pair_idx"]),
            frame_idx=int(cell_dict["frame_idx"]),
            scorer_axis=ScorerAxis(cell_dict["scorer_axis"]),
            receiver_runtime=ReceiverRuntime(cell_dict["receiver_runtime"]),
            cpu_cuda_axis=CpuCudaAxis(cell_dict["cpu_cuda_axis"]),
            predicted_delta_score=float(cell_dict["predicted_delta_score"]),
            predicted_byte_cost=int(cell_dict["predicted_byte_cost"]),
            receiver_feasibility=bool(cell_dict["receiver_feasibility"]),
            catalog_323_provenance=dict(
                cell_dict.get("catalog_323_provenance", {})
            ),
        )
        cells_map[cell.coordinate] = cell
    return PairFrameScorerGeometryLattice(
        archive_sha256=archive_sha256, cells=cells_map
    )


def _write_text_atomic(path: Path, text: str) -> None:
    """Write a text file atomically in the destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="apply_8_extended_operators_to_5d_canvas_cli",
        description=(
            "Apply 8 NOT-BUILT extended operators (replace-one / "
            "replace-many / merge-pair / reorder-pair / drop-frame / "
            "synthesize-frame / motion-conditional / temporal-coherence) "
            "to a populated 5D pair-frame scorer-geometry canvas. Emits "
            "ranked ExecutableCandidate JSON for the downstream FIRE-phase "
            "paired CPU+CUDA dispatch."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the 8 operators + canonical equation IDs; exit 0.",
    )
    parser.add_argument(
        "--canvas-path",
        type=Path,
        default=None,
        help=(
            "Path to populated 5D canvas JSON (from BUILD-1 sister "
            "subagent). Required unless --list is passed."
        ),
    )
    parser.add_argument(
        "--operator",
        type=str,
        choices=[*(op.value for op in ExtendedOperation), _ALL_OPERATORS_VALUE],
        default=None,
        help="Operator to apply, or 'all' to batch all 8 operators.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Max candidates to emit per operator (default {DEFAULT_TOP_N}).",
    )
    parser.add_argument(
        "--global-top-n",
        type=int,
        default=0,
        help=(
            "[operator=all] Optional global cap after per-operator ranking; "
            "0 keeps every per-operator candidate."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Path to write candidates JSON manifest (default stdout). "
            "Per Catalog #131: writes via fcntl-locked transactional "
            "atomic-replace pattern downstream."
        ),
    )
    parser.add_argument(
        "--receiver-runtime",
        type=str,
        choices=[rt.value for rt in ReceiverRuntime],
        default=ReceiverRuntime.RAW_RESIDUAL.value,
        help="Receiver-runtime mode (default raw_residual).",
    )
    # replace-one parameters
    parser.add_argument(
        "--target-pair-idx",
        type=int,
        default=None,
        help="[replace-one] Target pair_idx in [0, 600).",
    )
    parser.add_argument(
        "--alternative-selector-id",
        type=int,
        default=None,
        help="[replace-one] Opaque substitute selector ID (non-negative).",
    )
    # replace-many parameters
    parser.add_argument(
        "--beam-width",
        type=int,
        default=DEFAULT_BEAM_WIDTH,
        help=(
            f"[replace-many] Beam-search width (default {DEFAULT_BEAM_WIDTH}, "
            "max 1024)."
        ),
    )
    parser.add_argument(
        "--beam-depth",
        type=int,
        default=DEFAULT_BEAM_DEPTH,
        help=(
            f"[replace-many] Beam-search depth (default {DEFAULT_BEAM_DEPTH}, "
            "max 16)."
        ),
    )
    parser.add_argument(
        "--target-pair-indices",
        type=str,
        default="",
        help=(
            "[replace-many] Comma-separated list of pair indices to "
            "consider for replacement (defaults to all feasible)."
        ),
    )
    # merge-pair parameters
    parser.add_argument(
        "--max-merge-candidates",
        type=int,
        default=DEFAULT_MERGE_MAX_CANDIDATES,
        help=f"[merge-pair] Max O(N^2) candidates (default {DEFAULT_MERGE_MAX_CANDIDATES}).",
    )
    # reorder-pair parameters
    parser.add_argument(
        "--reorder-block-size",
        type=int,
        default=8,
        help="[reorder-pair] Permutation block size (default 8, max 32).",
    )
    # drop-frame + synthesize-frame parameters
    parser.add_argument(
        "--which-frame",
        type=str,
        choices=["first", "last", "both"],
        default="last",
        help="[drop-frame|synthesize-frame] Which frame to target.",
    )
    parser.add_argument(
        "--synthesis-seed",
        type=int,
        default=0,
        help="[synthesize-frame] Deterministic synthesizer seed.",
    )
    # motion-conditional parameters
    parser.add_argument(
        "--motion-threshold-percentile",
        type=float,
        default=DEFAULT_MOTION_THRESHOLD_PERCENTILE,
        help=(
            "[motion-conditional] Percentile cutoff in [0, 1] "
            f"(default {DEFAULT_MOTION_THRESHOLD_PERCENTILE})."
        ),
    )
    parser.add_argument(
        "--high-motion-operator",
        type=str,
        choices=[op.value for op in ExtendedOperation],
        default=ExtendedOperation.SYNTHESIZE_FRAME.value,
        help="[motion-conditional] Operator for high-motion pairs.",
    )
    parser.add_argument(
        "--low-motion-operator",
        type=str,
        choices=[op.value for op in ExtendedOperation],
        default=ExtendedOperation.REPLACE_ONE.value,
        help="[motion-conditional] Operator for low-motion pairs.",
    )
    # temporal-coherence parameters
    parser.add_argument(
        "--temporal-window",
        type=int,
        default=DEFAULT_TEMPORAL_COHERENCE_WINDOW,
        help=(
            f"[temporal-coherence] Window size (default {DEFAULT_TEMPORAL_COHERENCE_WINDOW}, "
            "max 64)."
        ),
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.7,
        help=(
            "[temporal-coherence] Cosine-similarity threshold in [-1, 1] "
            "(default 0.7)."
        ),
    )
    return parser


def _build_parameters(args: argparse.Namespace) -> Any:
    """Build the canonical Parameters dataclass for the requested operator."""
    op = ExtendedOperation(args.operator)
    rt = ReceiverRuntime(args.receiver_runtime)
    if op is ExtendedOperation.REPLACE_ONE:
        if args.target_pair_idx is None or args.alternative_selector_id is None:
            raise SystemExit(
                "[apply_8_extended_operators_to_5d_canvas_cli] FATAL: "
                "--target-pair-idx and --alternative-selector-id required "
                "for replace_one"
            )
        return ReplaceOneParameters(
            target_pair_idx=int(args.target_pair_idx),
            alternative_selector_id=int(args.alternative_selector_id),
            receiver_runtime=rt,
        )
    if op is ExtendedOperation.REPLACE_MANY:
        target_indices = tuple(
            int(x.strip())
            for x in args.target_pair_indices.split(",")
            if x.strip()
        )
        return ReplaceManyParameters(
            beam_width=int(args.beam_width),
            beam_depth=int(args.beam_depth),
            receiver_runtime=rt,
            target_pair_indices=target_indices,
        )
    if op is ExtendedOperation.MERGE_PAIR:
        return MergePairParameters(
            receiver_runtime=rt,
            max_candidates=int(args.max_merge_candidates),
        )
    if op is ExtendedOperation.REORDER_PAIR:
        return ReorderPairParameters(
            receiver_runtime=rt,
            block_size=int(args.reorder_block_size),
        )
    if op is ExtendedOperation.DROP_FRAME:
        return DropFrameParameters(
            receiver_runtime=rt,
            which_frame=str(args.which_frame),
        )
    if op is ExtendedOperation.SYNTHESIZE_FRAME:
        if args.which_frame == "both":
            raise SystemExit(
                "[apply_8_extended_operators_to_5d_canvas_cli] FATAL: "
                "synthesize_frame does not accept --which-frame=both"
            )
        return SynthesizeFrameParameters(
            receiver_runtime=rt,
            which_frame=str(args.which_frame),
            synthesis_seed=int(args.synthesis_seed),
        )
    if op is ExtendedOperation.MOTION_CONDITIONAL:
        return MotionConditionalParameters(
            receiver_runtime=rt,
            motion_threshold_percentile=float(args.motion_threshold_percentile),
            high_motion_operator=ExtendedOperation(args.high_motion_operator),
            low_motion_operator=ExtendedOperation(args.low_motion_operator),
        )
    if op is ExtendedOperation.TEMPORAL_COHERENCE:
        return TemporalCoherenceParameters(
            receiver_runtime=rt,
            temporal_window=int(args.temporal_window),
            similarity_threshold=float(args.similarity_threshold),
        )
    raise SystemExit(
        f"[apply_8_extended_operators_to_5d_canvas_cli] FATAL: unknown "
        f"operator {op.value}"
    )


def _validate_numeric_caps(args: argparse.Namespace) -> None:
    if not isinstance(args.top_n, int) or args.top_n < 1:
        raise SystemExit(
            "[apply_8_extended_operators_to_5d_canvas_cli] FATAL: "
            f"--top-n must be a positive integer, got {args.top_n!r}"
        )
    if not isinstance(args.global_top_n, int) or args.global_top_n < 0:
        raise SystemExit(
            "[apply_8_extended_operators_to_5d_canvas_cli] FATAL: "
            f"--global-top-n must be a non-negative integer, got {args.global_top_n!r}"
        )


def _default_params_for_batch(
    args: argparse.Namespace,
    op: ExtendedOperation,
    receiver_runtime: ReceiverRuntime,
) -> Any:
    """Build default parameters for the all-operator batch mode."""
    if op is ExtendedOperation.REPLACE_MANY:
        target_indices = tuple(
            int(x.strip())
            for x in args.target_pair_indices.split(",")
            if x.strip()
        )
        return ReplaceManyParameters(
            beam_width=int(args.beam_width),
            beam_depth=int(args.beam_depth),
            receiver_runtime=receiver_runtime,
            target_pair_indices=target_indices,
        )
    if op is ExtendedOperation.MERGE_PAIR:
        return MergePairParameters(
            receiver_runtime=receiver_runtime,
            max_candidates=int(args.max_merge_candidates),
        )
    if op is ExtendedOperation.REORDER_PAIR:
        return ReorderPairParameters(
            receiver_runtime=receiver_runtime,
            block_size=int(args.reorder_block_size),
        )
    if op is ExtendedOperation.DROP_FRAME:
        return DropFrameParameters(
            receiver_runtime=receiver_runtime,
            which_frame=str(args.which_frame),
        )
    if op is ExtendedOperation.SYNTHESIZE_FRAME:
        return SynthesizeFrameParameters(
            receiver_runtime=receiver_runtime,
            which_frame="last" if args.which_frame == "both" else str(args.which_frame),
            synthesis_seed=int(args.synthesis_seed),
        )
    if op is ExtendedOperation.MOTION_CONDITIONAL:
        return MotionConditionalParameters(
            receiver_runtime=receiver_runtime,
            motion_threshold_percentile=float(args.motion_threshold_percentile),
            high_motion_operator=ExtendedOperation(args.high_motion_operator),
            low_motion_operator=ExtendedOperation(args.low_motion_operator),
        )
    if op is ExtendedOperation.TEMPORAL_COHERENCE:
        return TemporalCoherenceParameters(
            receiver_runtime=receiver_runtime,
            temporal_window=int(args.temporal_window),
            similarity_threshold=float(args.similarity_threshold),
        )
    raise AssertionError(f"batch params unsupported for {op.value}")


def _feasible_pair_indices(
    lattice: PairFrameScorerGeometryLattice,
    receiver_runtime: ReceiverRuntime,
) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                cell.pair_idx
                for cell in lattice._cells.values()
                if cell.receiver_runtime is receiver_runtime
                and cell.receiver_feasibility
            }
        )
    )


def _zero_candidate_blockers(
    lattice: PairFrameScorerGeometryLattice,
    receiver_runtime: ReceiverRuntime,
) -> list[str]:
    if lattice.cell_count() == 0:
        return ["empty_canvas"]
    runtime_cells = [
        cell
        for cell in lattice._cells.values()
        if cell.receiver_runtime is receiver_runtime
    ]
    if not runtime_cells:
        return [f"no_cells_for_receiver_runtime:{receiver_runtime.value}"]
    feasible_cells = [cell for cell in runtime_cells if cell.receiver_feasibility]
    if not feasible_cells:
        return [
            f"no_receiver_feasible_cells_for_runtime:{receiver_runtime.value}"
        ]
    if all(float(cell.predicted_delta_score) >= 0.0 for cell in feasible_cells):
        return ["no_negative_predicted_delta_cells"]
    return ["operator_specific_filters_eliminated_all_candidates"]


def _candidate_rank_key(candidate: Any) -> tuple[float, int, str]:
    hint = candidate.canonical_dispatch_recipe_hint
    return (
        float(candidate.predicted_delta_score),
        int(candidate.predicted_byte_cost),
        json.dumps(hint, sort_keys=True),
    )


def _run_all_operators(
    lattice: PairFrameScorerGeometryLattice,
    args: argparse.Namespace,
) -> dict[str, Any]:
    receiver_runtime = ReceiverRuntime(args.receiver_runtime)
    per_operator: dict[str, dict[str, Any]] = {}
    flattened: list[Any] = []
    feasible_pairs = _feasible_pair_indices(lattice, receiver_runtime)

    for op in ExtendedOperation:
        if op is ExtendedOperation.REPLACE_ONE:
            pair_indices = (
                (int(args.target_pair_idx),)
                if args.target_pair_idx is not None
                else _feasible_pair_indices(lattice, receiver_runtime)
            )
            alternative_selector_id = (
                int(args.alternative_selector_id)
                if args.alternative_selector_id is not None
                else 0
            )
            candidates = []
            for pair_idx in pair_indices:
                params = ReplaceOneParameters(
                    target_pair_idx=pair_idx,
                    alternative_selector_id=alternative_selector_id,
                    receiver_runtime=receiver_runtime,
                )
                candidates.extend(
                    EXTENDED_OPERATOR_REGISTRY[op](
                        lattice, params, top_n=int(args.top_n)
                    )
                )
            candidates.sort(key=_candidate_rank_key)
            candidates = candidates[: int(args.top_n)]
        else:
            params = _default_params_for_batch(args, op, receiver_runtime)
            candidates = EXTENDED_OPERATOR_REGISTRY[op](
                lattice, params, top_n=int(args.top_n)
            )

        flattened.extend(candidates)
        per_operator[op.value] = {
            "canonical_equation_id_pending": (
                EXTENDED_OPERATION_CANONICAL_EQUATION_IDS[op]
            ),
            "candidates_emitted": len(candidates),
            "blockers": []
            if candidates
            else _zero_candidate_blockers(lattice, receiver_runtime),
            "candidates": [candidate.as_dict() for candidate in candidates],
        }

    flattened.sort(key=_candidate_rank_key)
    if args.global_top_n:
        flattened = flattened[: int(args.global_top_n)]

    return {
        "schema": "extended_operator_batch_candidates.v0",
        "consumer_name": CONSUMER_NAME,
        "consumer_version": CONSUMER_VERSION,
        "operator": _ALL_OPERATORS_VALUE,
        "canvas_archive_sha256": lattice.archive_sha256,
        "canvas_cells": lattice.cell_count(),
        "receiver_runtime": receiver_runtime.value,
        "feasible_pair_count": len(feasible_pairs),
        "top_n_requested_per_operator": int(args.top_n),
        "global_top_n_requested": int(args.global_top_n),
        "operator_count": len(ExtendedOperation),
        "total_candidates_emitted": sum(
            result["candidates_emitted"] for result in per_operator.values()
        ),
        "flattened_candidates_emitted": len(flattened),
        "operator_results": per_operator,
        "flattened_candidates": [
            candidate.as_dict() for candidate in flattened
        ],
        "audit_memo_reference": (
            ".omx/research/drop_many_replace_composition_apparatus_state_audit_20260526.md"
        ),
    }


def _emit_list() -> int:
    """Emit the 8 operators + their canonical equation IDs (FORMALIZATION_PENDING)."""
    payload = {
        "schema": "extended_operator_registry_listing.v0",
        "consumer_name": CONSUMER_NAME,
        "consumer_version": CONSUMER_VERSION,
        "operator_count": len(ExtendedOperation),
        "operators": [
            {
                "operator": op.value,
                "canonical_equation_id_pending": (
                    EXTENDED_OPERATION_CANONICAL_EQUATION_IDS[op]
                ),
            }
            for op in ExtendedOperation
        ],
        "audit_memo_reference": (
            ".omx/research/drop_many_replace_composition_apparatus_state_audit_20260526.md"
        ),
        "v14_v2_memo_reference": (
            ".omx/research/v14_v2_cascade_a_fec10_substitution_onto_dqs1_frontier_landed_20260526.md"
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.list:
        return _emit_list()
    _validate_numeric_caps(args)
    if args.canvas_path is None or args.operator is None:
        parser.error("--canvas-path and --operator are required (or use --list)")
        return 2  # unreachable per argparse error

    lattice = _load_canvas_from_json(args.canvas_path)
    if args.operator == _ALL_OPERATORS_VALUE:
        output_payload = _run_all_operators(lattice, args)
    else:
        parameters = _build_parameters(args)
        generator = EXTENDED_OPERATOR_REGISTRY[ExtendedOperation(args.operator)]
        candidates = generator(lattice, parameters, top_n=int(args.top_n))
        receiver_runtime = ReceiverRuntime(args.receiver_runtime)
        output_payload = {
            "schema": "extended_operator_candidates.v0",
            "consumer_name": CONSUMER_NAME,
            "consumer_version": CONSUMER_VERSION,
            "operator": args.operator,
            "canonical_equation_id_pending": (
                EXTENDED_OPERATION_CANONICAL_EQUATION_IDS[
                    ExtendedOperation(args.operator)
                ]
            ),
            "canvas_archive_sha256": lattice.archive_sha256,
            "canvas_cells": lattice.cell_count(),
            "receiver_runtime": receiver_runtime.value,
            "feasible_pair_count": len(
                _feasible_pair_indices(lattice, receiver_runtime)
            ),
            "top_n_requested": int(args.top_n),
            "candidates_emitted": len(candidates),
            "blockers": []
            if candidates
            else _zero_candidate_blockers(lattice, receiver_runtime),
            "candidates": [c.as_dict() for c in candidates],
            "audit_memo_reference": (
                ".omx/research/drop_many_replace_composition_apparatus_state_audit_20260526.md"
            ),
        }
    payload_json = json.dumps(output_payload, indent=2, sort_keys=True)
    if args.output is None:
        print(payload_json)
    else:
        _write_text_atomic(args.output, payload_json + "\n")
        print(
            f"[apply_8_extended_operators_to_5d_canvas_cli] OK: "
            f"wrote {output_payload.get('total_candidates_emitted', output_payload.get('candidates_emitted', 0))} "
            f"candidates to {args.output}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
