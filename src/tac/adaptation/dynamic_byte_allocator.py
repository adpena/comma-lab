# SPDX-License-Identifier: MIT
"""Byte-atom allocation contracts for DVAR1 dynamic adaptation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from tac.adaptation.video_telemetry import (
    DynamicVideoTelemetryError,
    telemetry_to_hard_pair_indices,
)
from tac.optimization.meta_lagrangian_allocator import build_atom_ledger

SCHEMA = "dynamic_byte_atom_ledger_v1"
FAMILY = "dynamic_video_adaptive_receiver"
DEFAULT_OPERATIONS = (
    "ibps1_byte_patch",
    "latent_residual_sidecar",
    "postdecode_selector",
    "film_grain_selector",
)


class DynamicByteAllocatorError(ValueError):
    """Raised when DVAR1 atom planning would be unsafe or ambiguous."""


def select_hard_pairs(telemetry: Mapping[str, Any], *, top_k: int) -> list[int]:
    """Expose hard-pair selection under the allocator namespace."""

    try:
        return telemetry_to_hard_pair_indices(telemetry, top_k=top_k)
    except DynamicVideoTelemetryError as exc:
        raise DynamicByteAllocatorError(str(exc)) from exc


def build_dynamic_byte_atom_ledger(
    telemetry: Mapping[str, Any],
    *,
    top_k: int,
    section: str,
    operation: str,
    byte_delta_per_pair: int,
    predicted_score_delta_per_pair: float,
    confidence: float,
    evidence_source_path: str,
    allowed_operations: Sequence[str] = DEFAULT_OPERATIONS,
) -> dict[str, Any]:
    """Build planning atoms from hard-pair telemetry.

    The output is a planner artifact only. It does not mutate an archive and
    cannot be promoted until a deterministic packet consumes the atoms and an
    exact eval validates the packet.
    """

    if operation not in allowed_operations:
        raise DynamicByteAllocatorError(
            f"operation {operation!r} not in allowed operations {tuple(allowed_operations)!r}"
        )
    if not section:
        raise DynamicByteAllocatorError("section must be non-empty")
    if byte_delta_per_pair < 0:
        raise DynamicByteAllocatorError("byte_delta_per_pair must be non-negative")
    if not (0.0 <= confidence <= 1.0):
        raise DynamicByteAllocatorError("confidence must be in [0, 1]")

    pair_indices = select_hard_pairs(telemetry, top_k=top_k)
    atoms: list[dict[str, Any]] = []
    for rank, pair_idx in enumerate(pair_indices, start=1):
        atoms.append(
            {
                "atom_id": f"dvar1:{operation}:{section}:pair{pair_idx}",
                "family": FAMILY,
                "family_group": FAMILY,
                "pareto_scope": "dynamic_video_adaptation",
                "pair_idx": int(pair_idx),
                "hard_pair_rank": rank,
                "operation": operation,
                "section": section,
                "byte_delta": int(byte_delta_per_pair),
                "expected_score_delta_direct": float(predicted_score_delta_per_pair),
                # meta_lagrangian_allocator consumes component deltas. DVAR1
                # stage-1 atoms may only have a scalar local-score proxy, so
                # represent it as an equivalent SegNet-axis delta and keep the
                # original scalar beside it for later exact decomposition.
                "expected_pose_dist_delta": 0.0,
                "expected_seg_dist_delta": float(predicted_score_delta_per_pair) / 100.0,
                "confidence": float(confidence),
                "evidence_grade": str(telemetry.get("axis_label", "[proxy]")),
                "proxy_row": True,
                "rankable": False,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_blockers": [
                    "planning_atom_only",
                    "requires_byte_closed_packet",
                    "requires_runtime_consumption_proof",
                    "requires_exact_eval_before_promotion",
                ],
                "evidence_source_path": evidence_source_path,
                "source_archive_sha256": str(telemetry.get("source_archive_sha256") or ""),
                "video_sha256": str(telemetry.get("video_sha256") or ""),
                "interaction_assumptions": [
                    "compress_time_dynamic_adaptation_only",
                    "scorer_free_inflate_runtime",
                    "selector_or_patch_bytes_charged",
                ],
            }
        )

    base_pose = 1.0
    pair_rows = telemetry.get("pair_rows")
    if isinstance(pair_rows, list) and pair_rows:
        pose_values = [
            float(row.get("pose_dist", 0.0))
            for row in pair_rows
            if isinstance(row, Mapping)
        ]
        if pose_values:
            base_pose = max(sum(pose_values) / len(pose_values), 1e-12)

    return {
        "schema": SCHEMA,
        "family": FAMILY,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "telemetry_schema": str(telemetry.get("schema", "")),
        "axis_label": str(telemetry.get("axis_label", "")),
        "operation": operation,
        "section": section,
        "pair_count": len(pair_indices),
        "atoms": atoms,
        "atom_ledger": build_atom_ledger(
            atoms,
            base_pose_dist=base_pose,
            source=evidence_source_path or "dynamic_video_telemetry",
        ),
    }


__all__ = [
    "DEFAULT_OPERATIONS",
    "DynamicByteAllocatorError",
    "FAMILY",
    "SCHEMA",
    "build_dynamic_byte_atom_ledger",
    "select_hard_pairs",
]
