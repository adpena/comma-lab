# SPDX-License-Identifier: MIT
"""Fail-closed headline contract for the DDM inverse-solve campaign.

The campaign's decision number is not a residual-code diagnostic in isolation.
It is the own-lineage stored problem plus solve-mandated exceptions, measured
after deterministic expansion through uint8, the real resize, and both frozen
scorers.  Donor-conditioned rows are structurally inadmissible.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

HEADLINE_SCHEMA = "ddm_min_description_headline.v1"
SOLVE_TYPING_SCHEMA = "ddm_recursive_solve_typing.v1"
TYPED_STREAM_SCHEMA = "ddm_typed_stream_tag.v1"


class MinimumDescriptionContractError(ValueError):
    """A malformed byte, lineage, or realized-acceptance declaration."""


class StreamType(StrEnum):
    """Disjoint description-stream roles induced by scorer recursion."""

    SKELETON = "SKELETON"
    CONNECTION = "CONNECTION"
    FIBER = "FIBER"
    GAUGE = "GAUGE"
    RESIDUAL = "RESIDUAL"


class LayerHome(StrEnum):
    """Earliest recursion layer that owns one stream's counted information."""

    L1_PROGRAM = "L1_program"
    L2_CHART = "L2_chart"
    L3_RASTER = "L3_raster"
    L4_SCORER_FEATURE = "L4_scorer_feature"
    L5_VERDICT = "L5_verdict"


@dataclass(frozen=True)
class TypedStreamTag:
    """One byte-home declaration for a description or archive section.

    ``free_receiver_code`` says the generic decoder/operator implementation is
    free.  It does not make video-derived parameters free: those remain in
    ``counted_bytes``.  Exact receiver-null GAUGE coordinates are the only type
    required to carry zero counted bytes.
    """

    type: StreamType
    layer_home: LayerHome
    evaluate_py_recursion_level_cited: str
    counted_bytes: int
    free_receiver_code: bool

    def __post_init__(self) -> None:
        if not isinstance(self.type, StreamType):
            raise MinimumDescriptionContractError("typed stream type must be StreamType")
        if not isinstance(self.layer_home, LayerHome):
            raise MinimumDescriptionContractError("layer_home must be LayerHome")
        citation = self.evaluate_py_recursion_level_cited
        if not isinstance(citation, str) or not citation.strip():
            raise MinimumDescriptionContractError(
                "evaluate.py recursion citation must be a nonempty string"
            )
        if (
            isinstance(self.counted_bytes, bool)
            or not isinstance(self.counted_bytes, int)
            or self.counted_bytes < 0
        ):
            raise MinimumDescriptionContractError(
                "typed stream counted_bytes must be a nonnegative exact integer"
            )
        if not isinstance(self.free_receiver_code, bool):
            raise MinimumDescriptionContractError(
                "free_receiver_code must be an exact boolean"
            )
        if self.type is StreamType.GAUGE and self.counted_bytes != 0:
            raise MinimumDescriptionContractError(
                "GAUGE streams must carry zero counted bytes"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TYPED_STREAM_SCHEMA,
            "type": self.type.value,
            "layer_home": self.layer_home.value,
            "evaluate_py_recursion_level_cited": (
                self.evaluate_py_recursion_level_cited
            ),
            "counted_bytes": self.counted_bytes,
            "free_receiver_code": self.free_receiver_code,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TypedStreamTag:
        expected = {
            "schema",
            "type",
            "layer_home",
            "evaluate_py_recursion_level_cited",
            "counted_bytes",
            "free_receiver_code",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise MinimumDescriptionContractError(
                "typed stream tag keys differ from the sealed schema"
            )
        if value["schema"] != TYPED_STREAM_SCHEMA:
            raise MinimumDescriptionContractError("typed stream tag schema differs")
        try:
            stream_type = StreamType(value["type"])
            layer_home = LayerHome(value["layer_home"])
        except (TypeError, ValueError) as exc:
            raise MinimumDescriptionContractError(
                "typed stream type or layer_home is outside the sealed vocabulary"
            ) from exc
        return cls(
            type=stream_type,
            layer_home=layer_home,
            evaluate_py_recursion_level_cited=value[
                "evaluate_py_recursion_level_cited"
            ],
            counted_bytes=value["counted_bytes"],
            free_receiver_code=value["free_receiver_code"],
        )


def _typed_stream_tag(value: TypedStreamTag | Mapping[str, Any]) -> TypedStreamTag:
    if isinstance(value, TypedStreamTag):
        return value
    if isinstance(value, Mapping):
        return TypedStreamTag.from_dict(value)
    raise MinimumDescriptionContractError(
        "typed stream tags must be TypedStreamTag instances or sealed mappings"
    )


def _optional_bytes(value: int | None, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MinimumDescriptionContractError(
            f"{field} must be a nonnegative exact integer or null"
        )
    return value


def _optional_sha256(value: str | None, field: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MinimumDescriptionContractError(
            f"{field} must be a lowercase SHA-256 or null"
        )
    return value


def _distortion(value: float, field: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise MinimumDescriptionContractError(
            f"{field} must be finite and nonnegative"
        )
    return result


def build_recursive_solve_typing_contract(
    *,
    quotient_coordinates_only: bool,
    scorer_metric_active: bool,
    alternating_typed_subproblems: bool,
    typed_blocks_active: bool,
    per_dimension_quanta_active: bool,
) -> dict[str, Any]:
    """Declare the five non-interchangeable dimensions of the inverse solve."""

    declarations = {
        "quotient_coordinates_only": quotient_coordinates_only,
        "scorer_metric_active": scorer_metric_active,
        "alternating_typed_subproblems": alternating_typed_subproblems,
        "typed_blocks_active": typed_blocks_active,
        "per_dimension_quanta_active": per_dimension_quanta_active,
    }
    if any(not isinstance(value, bool) for value in declarations.values()):
        raise MinimumDescriptionContractError(
            "recursive solve-typing declarations must be exact booleans"
        )
    blockers = [
        blocker
        for field, blocker in (
            ("quotient_coordinates_only", "GAUGE_COORDINATES_NOT_DROPPED"),
            ("scorer_metric_active", "SCORER_METRIC_NOT_ACTIVE"),
            (
                "alternating_typed_subproblems",
                "TYPED_SUBPROBLEM_ALTERNATION_NOT_ACTIVE",
            ),
            ("typed_blocks_active", "TYPED_BLOCK_ATLAS_NOT_ACTIVE"),
            (
                "per_dimension_quanta_active",
                "PER_DIMENSION_EFFECTIVE_QUANTA_NOT_ACTIVE",
            ),
        )
        if not declarations[field]
    ]
    return {
        "schema": SOLVE_TYPING_SCHEMA,
        "headline_ready": not blockers,
        "declarations": declarations,
        "required_geometry": {
            "variables": (
                "range(A)/quotient coordinates only; ker(A) is gauge and is "
                "realized by the preimage compiler plus deterministic free fill"
            ),
            "metric": (
                "Seg rank-4 head plus margin-Fisher blocks and a <=6-dimensional "
                "Pose quadratic"
            ),
            "subproblems": (
                "alternate argmax-cell selection, within-cell continuous lattice "
                "solve, and real-coder pricing"
            ),
            "blocks": "stratum x scorer-visibility x g4 temporal class",
            "quanta": "uint8 step x per-dimension scorer sensitivity",
        },
        "blockers": blockers,
    }


def build_minimum_description_headline(
    *,
    stored_problem_bytes: int | None,
    stored_problem_sha256: str | None,
    exception_bytes: int | None,
    exception_sha256: str | None,
    realized_d_seg: float,
    realized_d_pose: float,
    stored_problem_own_lineage: bool,
    donor_conditioned: bool,
    expansion_receiver_closed: bool,
    pose_tube_active: bool,
    realized_uint8_r_frozen_scorers: bool,
    quotient_coordinates_only: bool,
    scorer_metric_active: bool,
    alternating_typed_subproblems: bool,
    typed_blocks_active: bool,
    per_dimension_quanta_active: bool,
    typed_stream_tags: Sequence[TypedStreamTag | Mapping[str, Any]] | None = None,
    untagged_stream_waiver: str | None = None,
    strict_typed_stream_tags: bool = False,
    metric_custody_bundle_path: str | Path | None = None,
    metric_custody_repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build the only row eligible to headline minimum-description progress.

    A diagnostic row is still returned when custody is incomplete so a failed
    rung remains useful system intelligence.  Its decision triple is withheld,
    and blockers state exactly which authority edge is absent.
    """

    problem_bytes = _optional_bytes(stored_problem_bytes, "stored_problem_bytes")
    exceptions = _optional_bytes(exception_bytes, "exception_bytes")
    problem_sha = _optional_sha256(stored_problem_sha256, "stored_problem_sha256")
    exceptions_sha = _optional_sha256(exception_sha256, "exception_sha256")
    d_seg = _distortion(realized_d_seg, "realized_d_seg")
    d_pose = _distortion(realized_d_pose, "realized_d_pose")
    metric_bundle: dict[str, Any] | None = None
    metric_bundle_complete = True
    metric_pose_tube_complete = True
    if metric_custody_bundle_path is not None:
        if metric_custody_repository_root is None:
            raise MinimumDescriptionContractError(
                "metric_custody_repository_root is required with a bundle path"
            )
        try:
            from tac.optimization.ddm_metric_custody_bundle import (
                load_metric_custody_bundle,
            )

            loaded_bundle = load_metric_custody_bundle(
                metric_custody_bundle_path,
                repository_root=metric_custody_repository_root,
            )
        except (OSError, ValueError) as exc:
            raise MinimumDescriptionContractError(
                "metric custody bundle failed freshness or schema validation"
            ) from exc
        metric_bundle_complete = loaded_bundle.complete
        metric_pose_tube_complete = loaded_bundle.headline_flags()[
            "pose_tube_active"
        ]
        metric_bundle = {
            "path": str(loaded_bundle.path),
            "bundle_id": loaded_bundle.bundle_id,
            "status": loaded_bundle.status.value,
            "complete": loaded_bundle.complete,
            "blockers": list(loaded_bundle.blockers),
        }
    elif metric_custody_repository_root is not None:
        raise MinimumDescriptionContractError(
            "metric_custody_repository_root cannot be supplied without a bundle path"
        )

    effective_scorer_metric_active = (
        scorer_metric_active and metric_bundle_complete
    )
    effective_pose_tube_active = pose_tube_active and metric_pose_tube_complete
    solve_typing = build_recursive_solve_typing_contract(
        quotient_coordinates_only=quotient_coordinates_only,
        scorer_metric_active=effective_scorer_metric_active,
        alternating_typed_subproblems=alternating_typed_subproblems,
        typed_blocks_active=typed_blocks_active,
        per_dimension_quanta_active=per_dimension_quanta_active,
    )
    declarations = (
        stored_problem_own_lineage,
        donor_conditioned,
        expansion_receiver_closed,
        pose_tube_active,
        realized_uint8_r_frozen_scorers,
    )
    if any(not isinstance(value, bool) for value in declarations):
        raise MinimumDescriptionContractError(
            "lineage and acceptance declarations must be exact booleans"
        )
    if not isinstance(strict_typed_stream_tags, bool):
        raise MinimumDescriptionContractError(
            "strict_typed_stream_tags must be an exact boolean"
        )
    if untagged_stream_waiver is not None and (
        not isinstance(untagged_stream_waiver, str)
        or len(untagged_stream_waiver.strip()) < 16
    ):
        raise MinimumDescriptionContractError(
            "untagged_stream_waiver must be null or a substantive rationale"
        )
    tags = (
        None
        if typed_stream_tags is None
        else tuple(_typed_stream_tag(value) for value in typed_stream_tags)
    )
    if tags is not None and not tags:
        raise MinimumDescriptionContractError(
            "typed_stream_tags must be null or a nonempty sequence"
        )
    if strict_typed_stream_tags and tags is None:
        raise MinimumDescriptionContractError(
            "strict typed-stream custody refuses an untagged headline"
        )

    blockers: list[str] = []
    if donor_conditioned:
        blockers.append("DONOR_CONDITIONING_INADMISSIBLE")
    if not stored_problem_own_lineage:
        blockers.append("OWN_LINEAGE_STORED_PROBLEM_NOT_PROVEN")
    if problem_bytes is None or problem_sha is None:
        blockers.append("STORED_PROBLEM_BYTE_CUSTODY_MISSING")
    if exceptions is None or exceptions_sha is None:
        blockers.append("SOLVE_EXCEPTION_BYTE_CUSTODY_MISSING")
    if not expansion_receiver_closed:
        blockers.append("STORED_PROBLEM_EXPANSION_NOT_RECEIVER_CLOSED")
    if not effective_pose_tube_active:
        blockers.append("POSE_TUBE_NOT_ACTIVE_IN_SOLVE")
    if not realized_uint8_r_frozen_scorers:
        blockers.append("REALIZED_UINT8_R_FROZEN_SCORER_ACCEPTANCE_MISSING")
    blockers.extend(solve_typing["blockers"])
    if metric_bundle is not None and not metric_bundle_complete:
        blockers.append("METRIC_CUSTODY_BUNDLE_INCOMPLETE")
    if tags is None:
        blockers.append("TYPED_STREAM_TAG_CUSTODY_MISSING_WARN_ONLY")
        if untagged_stream_waiver is not None:
            blockers.append("UNTAGGED_STREAM_WAIVER_NONAUTHORIZING")

    expected_typed_bytes = (
        None
        if problem_bytes is None or exceptions is None
        else problem_bytes + exceptions
    )
    typed_bytes = None if tags is None else sum(tag.counted_bytes for tag in tags)
    if (
        tags is not None
        and expected_typed_bytes is not None
        and typed_bytes != expected_typed_bytes
    ):
        blockers.append("TYPED_STREAM_COUNTED_BYTES_DO_NOT_RECONCILE")

    eligible = not blockers
    total_bytes = (
        int(problem_bytes + exceptions)
        if eligible and problem_bytes is not None and exceptions is not None
        else None
    )
    return {
        "schema": HEADLINE_SCHEMA,
        "campaign": "inverse_solve_minimum_description_witness",
        "status": (
            "HEADLINE_ELIGIBLE"
            if eligible
            else (
                "INADMISSIBLE_DONOR_CONDITIONING"
                if donor_conditioned
                else "HEADLINE_BLOCKED"
            )
        ),
        "headline_eligible": eligible,
        "stored_problem": {
            "bytes": problem_bytes,
            "sha256": problem_sha,
            "own_lineage": stored_problem_own_lineage,
            "receiver_expansion_closed": expansion_receiver_closed,
        },
        "solve_mandated_exceptions": {
            "bytes": exceptions,
            "sha256": exceptions_sha,
            "conditional_coding_role": (
                "exceptions conditioned only on deterministic expansion of the "
                "counted own-lineage stored problem"
            ),
        },
        "joint_constraints": {
            "pose_tube_active": effective_pose_tube_active,
            "realized_uint8_r_frozen_scorers": realized_uint8_r_frozen_scorers,
        },
        "recursive_solve_typing": solve_typing,
        "metric_custody_bundle": metric_bundle,
        "typed_stream_custody": {
            "schema": TYPED_STREAM_SCHEMA,
            "mode": (
                "STRICT"
                if strict_typed_stream_tags
                else (
                    "WARN_ONLY_WITH_HEADLINE_WITHHELD"
                    if tags is None
                    else "WARN_PHASE_TAGGED"
                )
            ),
            "tags": None if tags is None else [tag.to_dict() for tag in tags],
            "typed_counted_bytes": typed_bytes,
            "expected_counted_bytes": expected_typed_bytes,
            "reconciled": (
                tags is not None
                and expected_typed_bytes is not None
                and typed_bytes == expected_typed_bytes
            ),
            "waiver": untagged_stream_waiver,
            "waiver_authorizes_headline": False,
        },
        "donor_conditioned": donor_conditioned,
        "decision_triple": {
            "total_counted_bytes": total_bytes,
            "realized_d_seg": d_seg if eligible else None,
            "realized_d_pose": d_pose if eligible else None,
        },
        "diagnostic_distortions": {
            "realized_d_seg": d_seg,
            "realized_d_pose": d_pose,
        },
        "blockers": blockers,
        "score_claim": False,
        "promotion_eligible": False,
    }


__all__ = [
    "HEADLINE_SCHEMA",
    "SOLVE_TYPING_SCHEMA",
    "TYPED_STREAM_SCHEMA",
    "LayerHome",
    "MinimumDescriptionContractError",
    "StreamType",
    "TypedStreamTag",
    "build_minimum_description_headline",
    "build_recursive_solve_typing_contract",
]
