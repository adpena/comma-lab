# SPDX-License-Identifier: MIT
"""Queue builder for 5D-canvas extended operator sweeps."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas_coverage import (
    COVERAGE_AUDIT_SCHEMA,
    CanvasCoverageAuditError,
    audit_5d_canvas_coverage,
    load_5d_canvas_json,
)
from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas_extended_operators import (
    EXTENDED_OPERATION_CANONICAL_EQUATION_IDS,
    ExtendedOperation,
)
from tac.optimization.proxy_candidate_contract import require_no_truthy_authority_fields

from .experiment_queue import QUEUE_SCHEMA, ExperimentQueueError, normalize_queue_definition

PAIR_FRAME_5D_EXTENDED_OPERATOR_QUEUE_SCHEMA = (
    "pair_frame_5d_extended_operator_queue.v1"
)
FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "score_claim_eligible": False,
    "promotion_eligible": False,
    "promotable": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "dispatch_packet_ready": False,
    "reproduction_claim": False,
    "reproduction_equivalence": False,
}


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_path(value: str | Path, *, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_") or "operator"


def _operator_args(operation: ExtendedOperation) -> list[str]:
    if operation is ExtendedOperation.REPLACE_ONE:
        return ["--target-pair-idx", "0", "--alternative-selector-id", "1"]
    if operation is ExtendedOperation.REPLACE_MANY:
        return ["--beam-width", "8", "--beam-depth", "4"]
    if operation is ExtendedOperation.MERGE_PAIR:
        return ["--max-merge-candidates", "256"]
    if operation is ExtendedOperation.REORDER_PAIR:
        return ["--reorder-block-size", "8"]
    if operation is ExtendedOperation.DROP_FRAME:
        return ["--which-frame", "both"]
    if operation is ExtendedOperation.SYNTHESIZE_FRAME:
        return [
            "--receiver-runtime",
            "smoothed_residual",
            "--which-frame",
            "last",
            "--synthesis-seed",
            "0",
        ]
    if operation is ExtendedOperation.MOTION_CONDITIONAL:
        return ["--motion-threshold-percentile", "0.75"]
    if operation is ExtendedOperation.TEMPORAL_COHERENCE:
        return ["--temporal-window", "4", "--similarity-threshold", "0.7"]
    raise ExperimentQueueError(f"unsupported extended operator {operation!r}")


def _canvas_coverage_audit(canvas: Path) -> dict[str, Any]:
    if not canvas.exists():
        return {
            "schema": COVERAGE_AUDIT_SCHEMA,
            "verdict": "densification_required",
            "blockers": ["canvas_path_missing_for_coverage_audit"],
            "work_orders": [],
            "work_order_count": 0,
            "allowed_use": "local_planning_and_experiment_queue_acquisition_only",
            "forbidden_use": "score_claim_or_promotion_or_rank_kill_authority",
            **FALSE_AUTHORITY,
        }
    try:
        return audit_5d_canvas_coverage(load_5d_canvas_json(canvas))
    except CanvasCoverageAuditError as exc:
        return {
            "schema": COVERAGE_AUDIT_SCHEMA,
            "verdict": "densification_required",
            "blockers": [f"coverage_audit_failed:{exc}"],
            "work_orders": [],
            "work_order_count": 0,
            "allowed_use": "local_planning_and_experiment_queue_acquisition_only",
            "forbidden_use": "score_claim_or_promotion_or_rank_kill_authority",
            **FALSE_AUTHORITY,
        }


def build_pair_frame_5d_extended_operator_queue(
    *,
    repo_root: str | Path,
    canvas_path: str | Path,
    output_root: str | Path,
    queue_id: str = "pair_frame_5d_extended_operator_queue",
    top_n: int = 32,
    local_cpu_concurrency: int = 4,
    status: str = "queued",
) -> dict[str, Any]:
    """Build a local queue that fires all eight extended 5D operators.

    The queue is planning-only and false-authority. It transforms a populated
    5D canvas into per-operator candidate manifests; exact score, promotion,
    and rank authority remain outside this queue.
    """
    if top_n < 1:
        raise ExperimentQueueError("top_n must be >= 1")
    if (
        isinstance(local_cpu_concurrency, bool)
        or not isinstance(local_cpu_concurrency, int)
        or local_cpu_concurrency < 1
    ):
        raise ExperimentQueueError("local_cpu_concurrency must be >= 1")
    if status not in {"queued", "frozen", "disabled"}:
        raise ExperimentQueueError("status must be queued, frozen, or disabled")

    repo = Path(repo_root)
    canvas = _resolve_path(canvas_path, repo_root=repo)
    out_root = _resolve_path(output_root, repo_root=repo)
    canvas_ref = _repo_rel(canvas, repo)
    out_ref_root = _repo_rel(out_root, repo)
    coverage_audit = _canvas_coverage_audit(canvas)
    experiments: list[dict[str, Any]] = []
    for priority, operation in enumerate(ExtendedOperation, start=1):
        op_slug = _slug(operation.value)
        output_path = out_root / f"{op_slug}_extended_candidates.json"
        output_ref = _repo_rel(output_path, repo)
        metadata = {
            "schema": PAIR_FRAME_5D_EXTENDED_OPERATOR_QUEUE_SCHEMA,
            "operation": operation.value,
            "canonical_equation_id_pending": (
                EXTENDED_OPERATION_CANONICAL_EQUATION_IDS[operation]
            ),
            "canvas_path": canvas_ref,
            "canvas_coverage_audit": coverage_audit,
            "output_path": output_ref,
            "output_root": out_ref_root,
            "allowed_use": "local_encoder_side_5d_extended_operator_planning_only",
            "forbidden_use": "score_claim_or_promotion_or_rank_kill_authority",
            **FALSE_AUTHORITY,
        }
        require_no_truthy_authority_fields(
            metadata,
            context=f"pair_frame_5d_extended_operator_queue:{operation.value}",
        )
        experiments.append(
            {
                "id": f"pair_frame_5d_extended_operator_{op_slug}",
                "priority": priority,
                "status": status,
                "tags": [
                    "pair-frame-5d-canvas",
                    "extended-operator",
                    "encoder-side-planning",
                    "no-score-authority",
                    operation.value,
                ],
                "metadata": metadata,
                "steps": [
                    {
                        "id": f"emit_{op_slug}_candidates",
                        "kind": "command",
                        "command": [
                            ".venv/bin/python",
                            "tools/apply_8_extended_operators_to_5d_canvas_cli.py",
                            "--canvas-path",
                            canvas_ref,
                            "--operator",
                            operation.value,
                            "--top-n",
                            str(top_n),
                            "--output",
                            output_ref,
                            *_operator_args(operation),
                        ],
                        "resources": {"kind": "local_cpu"},
                        "timeout_seconds": 120,
                        "postconditions": [
                            {
                                "type": "json_equals",
                                "path": output_ref,
                                "key": "schema",
                                "equals": "extended_operator_candidates.v0",
                            },
                            {
                                "type": "json_false_authority",
                                "path": output_ref,
                                "required_false": [],
                                "false_or_missing": sorted(FALSE_AUTHORITY),
                            },
                        ],
                        "telemetry": {
                            "artifact_paths": [output_ref],
                            "input_artifact_paths": [canvas_ref],
                            "include_postcondition_paths": True,
                        },
                    }
                ],
            }
        )
    queue = normalize_queue_definition(
        {
            "schema": QUEUE_SCHEMA,
            "queue_id": queue_id,
            "controls": {
                "mode": "running",
                "local_first": True,
                "max_concurrency": {"local_cpu": local_cpu_concurrency},
            },
            "metadata": {
                "schema": PAIR_FRAME_5D_EXTENDED_OPERATOR_QUEUE_SCHEMA,
                "canvas_path": canvas_ref,
                "operator_count": len(experiments),
                "canvas_coverage_audit": coverage_audit,
                "allowed_use": "local_encoder_side_5d_extended_operator_planning_only",
                **FALSE_AUTHORITY,
            },
            "experiments": experiments,
        }
    )
    require_no_truthy_authority_fields(
        queue.get("metadata") if isinstance(queue.get("metadata"), Mapping) else {},
        context="pair_frame_5d_extended_operator_queue:metadata",
    )
    return queue


__all__ = [
    "PAIR_FRAME_5D_EXTENDED_OPERATOR_QUEUE_SCHEMA",
    "build_pair_frame_5d_extended_operator_queue",
]
