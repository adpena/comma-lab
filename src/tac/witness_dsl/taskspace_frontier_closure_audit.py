# SPDX-License-Identifier: MIT
"""Dynamic frontier-closure audit for the selected-solution codec campaign.

This module deliberately does not invent a scalar progress percentage.  The
campaign currently has evidence at different points of the coupled
``(d_seg, d_pose, archive_bytes)`` surface.  A low-rate state without an exact
full-population score and a low-distortion teacher without a candidate archive
are not endpoints of one measured transition.

The audit reopens the exact evidence, derives the current competitive target
from the canonical pointer's constituent rows, and reports the missing bridge.
It is pure/read-only: no scorer, archive mutation, dispatch, or pointer write.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.canonical_frontier_pointer import (
    CanonicalFrontierPointer,
    effective_frontier_score,
    recompute_effective_frontier,
)
from tac.score_geometry import contest_score, target_byte_budget_for_score


class TaskspaceFrontierClosureError(RuntimeError):
    """Raised when an evidence object cannot support the closure audit."""


@dataclass(frozen=True)
class ClosureEvidencePaths:
    pointer: Path
    ms1: Path
    ms2r: Path
    g20: Path


@dataclass(frozen=True)
class EvidenceObject:
    path: str
    bytes: int
    sha256: str
    value: Mapping[str, Any]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise TaskspaceFrontierClosureError(f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def _read_json_object(path: Path) -> EvidenceObject:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise TaskspaceFrontierClosureError(f"cannot read evidence {path}: {exc}") from exc
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise TaskspaceFrontierClosureError(f"invalid JSON evidence {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TaskspaceFrontierClosureError(f"evidence root must be an object: {path}")
    return EvidenceObject(
        path=str(path),
        bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
        value=value,
    )


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TaskspaceFrontierClosureError(f"{field} must be an object")
    return value


def _finite_nonnegative(value: Any, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise TaskspaceFrontierClosureError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise TaskspaceFrontierClosureError(f"{field} must be finite and nonnegative")
    return parsed


def _nonnegative_int(value: Any, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise TaskspaceFrontierClosureError(f"{field} must be a nonnegative integer")
    return value


def _evidence_identity(obj: EvidenceObject) -> dict[str, Any]:
    return {"path": obj.path, "bytes": obj.bytes, "sha256": obj.sha256}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _target_row(pointer_obj: EvidenceObject) -> tuple[float, Mapping[str, Any]]:
    try:
        pointer = CanonicalFrontierPointer.from_dict(pointer_obj.value)
        target = effective_frontier_score(pointer)
        row = recompute_effective_frontier(pointer)
    except (KeyError, TypeError, ValueError) as exc:
        raise TaskspaceFrontierClosureError(f"canonical pointer is invalid: {exc}") from exc
    if target is None or row is None:
        raise TaskspaceFrontierClosureError("canonical pointer has no competitive target")

    cached = pointer_obj.value.get("effective_frontier")
    if isinstance(cached, Mapping):
        try:
            cached_score = float(cached["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TaskspaceFrontierClosureError("cached effective frontier is malformed") from exc
        if not math.isclose(cached_score, target, rel_tol=0.0, abs_tol=1e-15):
            raise TaskspaceFrontierClosureError(
                "cached effective frontier disagrees with recomputed constituent minimum"
            )
    return target, row


def _ms1_anchor(obj: EvidenceObject, target: float) -> dict[str, Any]:
    root = obj.value
    diag = _mapping(root.get("realized_frozen_scorer_diagnostic"), field="MS1 scorer diagnostic")
    coder = _mapping(root.get("diagnostic_conditioning_coder"), field="MS1 diagnostic coder")
    d_seg = _finite_nonnegative(diag.get("realized_d_seg"), field="MS1 d_seg")
    d_pose = _finite_nonnegative(diag.get("realized_d_pose"), field="MS1 d_pose")
    diagnostic_bytes = _nonnegative_int(coder.get("previous_frame_exception_bytes"), field="MS1 diagnostic bytes")
    budget = target_byte_budget_for_score(
        target_score=target,
        d_seg_floor=d_seg,
        d_pose_floor=d_pose,
        current_archive_bytes=diagnostic_bytes,
    )
    if budget.max_archive_bytes is None:
        compression_ratio = None
    else:
        compression_ratio = diagnostic_bytes / max(1, budget.max_archive_bytes)
    return {
        "role": "low_distortion_encoder_teacher_not_candidate_archive",
        "authority": root.get("authority"),
        "d_seg": d_seg,
        "d_pose": d_pose,
        "distortion_term": budget.distortion_floor_score,
        "diagnostic_description_bytes": diagnostic_bytes,
        "diagnostic_composite_if_bytes_were_charged": contest_score(d_seg, d_pose, diagnostic_bytes),
        "conditional_frontier_max_archive_bytes": budget.max_archive_bytes,
        "conditional_required_byte_reduction_from_diagnostic": budget.required_savings_bytes,
        "diagnostic_to_conditional_budget_ratio": compression_ratio,
        "feasible_distortions_at_zero_rate": budget.feasible_under_floors,
        "candidate_claim": False,
        "evidence": _evidence_identity(obj),
    }


def _ms2r_anchor(obj: EvidenceObject, target: float) -> dict[str, Any]:
    root = obj.value
    headline = _mapping(root.get("headline"), field="MS2R headline")
    stored = _mapping(headline.get("stored_problem"), field="MS2R stored problem")
    diag = _mapping(headline.get("diagnostic_distortions"), field="MS2R distortions")
    archive_bytes = _nonnegative_int(stored.get("bytes"), field="MS2R archive bytes")
    d_seg = _finite_nonnegative(diag.get("realized_d_seg"), field="MS2R d_seg")
    d_pose = _finite_nonnegative(diag.get("realized_d_pose"), field="MS2R d_pose")
    budget = target_byte_budget_for_score(
        target_score=target,
        d_seg_floor=d_seg,
        d_pose_floor=d_pose,
        current_archive_bytes=archive_bytes,
    )
    return {
        "role": "receiver_closed_dense_selected_solution_teacher_not_candidate",
        "authority": root.get("authority"),
        "archive_bytes": archive_bytes,
        "archive_sha256": stored.get("sha256"),
        "d_seg": d_seg,
        "d_pose": d_pose,
        "distortion_term": budget.distortion_floor_score,
        "composite": contest_score(d_seg, d_pose, archive_bytes),
        "conditional_frontier_max_archive_bytes": budget.max_archive_bytes,
        "feasible_distortions_at_zero_rate": budget.feasible_under_floors,
        "candidate_claim": False,
        "evidence": _evidence_identity(obj),
    }


def _g20_anchor(obj: EvidenceObject) -> dict[str, Any]:
    root = obj.value
    selected = _mapping(root.get("selected"), field="G20 selected row")
    proof = _mapping(root.get("proof"), field="G20 equality proof")
    truth = _mapping(root.get("truth"), field="G20 truth")
    archive_bytes = _nonnegative_int(selected.get("archive_bytes"), field="G20 archive bytes")
    state_equal = proof.get("full_quantized_state_equal") is True
    if not state_equal:
        raise TaskspaceFrontierClosureError("G20 does not prove full quantized-state equality")
    return {
        "role": "low_rate_same_state_spelling_not_scored_candidate",
        "archive_bytes": archive_bytes,
        "archive_sha256": selected.get("archive_sha256"),
        "full_quantized_state_equal": True,
        "d_seg": None,
        "d_pose": None,
        "full_n600_replay_owed": truth.get("full_n600_receiver_replay_owed") is True,
        "contest_cpu_cuda_owed": truth.get("contest_cpu_cuda_same_bytes_owed") is True,
        "candidate_claim": truth.get("candidate_claim") is True,
        "score_claim": truth.get("score_claim") is True,
        "evidence": _evidence_identity(obj),
    }


def build_taskspace_frontier_closure_audit(
    paths: ClosureEvidencePaths,
    *,
    observed_gates: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the dynamic, dependency-scoped closure audit.

    ``observed_gates`` is deliberately supplied by the caller.  This keeps live
    process/checkpoint observation separate from immutable scientific evidence.
    """

    pointer_obj = _read_json_object(paths.pointer)
    ms1_obj = _read_json_object(paths.ms1)
    ms2r_obj = _read_json_object(paths.ms2r)
    g20_obj = _read_json_object(paths.g20)
    target, target_row = _target_row(pointer_obj)

    anchors = {
        "low_rate": _g20_anchor(g20_obj),
        "low_distortion": _ms1_anchor(ms1_obj, target),
        "receiver_closed_dense_control": _ms2r_anchor(ms2r_obj, target),
    }
    complete_triples = [
        row
        for row in anchors.values()
        if row.get("candidate_claim") is True
        and row.get("d_seg") is not None
        and row.get("d_pose") is not None
        and row.get("archive_bytes") is not None
    ]
    bridge_measured = bool(complete_triples)

    core: dict[str, Any] = {
        "schema": "tac.taskspace_frontier_closure_audit.v1",
        "authority": {
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "pointer_moved": False,
        },
        "competitive_target": {
            **dict(target_row),
            "score": target,
            "pointer_evidence": _evidence_identity(pointer_obj),
            "cached_row_rederived": True,
        },
        "objective": {
            "formula": "100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37545489",
            "admission": "one complete exact triple against the current target sublevel",
            "independent_axis_thresholds_forbidden": True,
        },
        "anchors": anchors,
        "measured_bridge": {
            "exists": bridge_measured,
            "complete_own_lineage_candidate_triples": len(complete_triples),
            "scalar_distance_to_frontier": None,
            "reason": (
                "a complete own-lineage candidate triple exists"
                if bridge_measured
                else "low-rate and low-distortion evidence belong to different objects; no exact finite transition joins them"
            ),
        },
        "proof_domains": {
            "artifact_identity": "archive/runtime/output equality and custody",
            "semantic_control_identity": "current competitive target and admission rule",
            "proof_dependency_set": "only observed external reads invalidate a proof",
            "decode_equality_depends_on_frontier_pointer": False,
            "competitive_admission_depends_on_frontier_pointer": True,
        },
        "principal_cruxes": [
            {
                "code": "R10_RECEIVER_EXECUTABLE_FEATURE_RELAY_OWED",
                "meaning": "carry amplitude/frequency/phase/contrast/channel-energy/texture and chronological Pose continuity through generic decoder operations with counted operands",
            },
            {
                "code": "SELECTED_SOLUTION_COMPILER_LIFECYCLE_OWED",
                "meaning": "close SourceTruth->IR->RealizedPair->Archive->Decode->axis-score with real callbacks, physical ownership, and authority sealing",
            },
            {
                "code": "CURRENT_FULL_POPULATION_BRIDGE_ROW_OWED",
                "meaning": "one current own-lineage n600 object must jointly expose exact output, exact final ZIP bytes, d_seg, and d_pose",
            },
            {
                "code": "SUBSTITUTIVE_HOMOTOPY_OWED",
                "meaning": "replace dense teacher/base/code homes rather than add sidecars; retain whole-object interactions and Pareto branches",
            },
        ],
        "dependency_order": [
            "close exact full-population equality and whole-object recode proofs",
            "close selected-solution compiler authority and physical ownership",
            "materialize terminal G14 feedback/costates without extrapolating n2",
            "implement and price the decoder-executable R10 relay against pixel/tensor controls",
            "run selected-solution homotopy from low-rate and low-distortion anchors to a complete n600 bridge row",
            "standalone double-decode then frozen-scorer advisory full-n600 row",
            "governed same-byte contest CPU and CUDA evaluation",
        ],
        "observed_gates": {str(key): dict(value) for key, value in sorted((observed_gates or {}).items())},
    }
    core["closure_identity_sha256"] = hashlib.sha256(_canonical_json(core)).hexdigest()
    return core


__all__ = [
    "ClosureEvidencePaths",
    "TaskspaceFrontierClosureError",
    "build_taskspace_frontier_closure_audit",
]
