"""Fail-closed readiness contract for geometry-feedback lanes.

Geometry signals such as LA-POSE-style atoms, telescopic foveation fields,
RAFT flow, and openpilot priors are optimizer feedback until archive bytes
prove that the scored inflate runtime consumes them. This module gives roadmap
and readiness tools one shared contract for that boundary.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

SCHEMA_VERSION = 1
CONTRACT_NAME = "charged_geometry_feedback_runtime_consumer_v1"

GEOMETRY_FEEDBACK_ROADMAP_KEYS = (
    "lapose_motion_atom_allocator",
    "telescopic_foveation_field",
    "raft_radial_openpilot_pose",
)

UNCHARGED_GEOMETRY_FEEDBACK_BLOCKER = "geometry_feedback_not_charged_runtime_consumed"
CANDIDATE_MANIFEST_BLOCKER = "candidate_specific_archive_manifest_required"
GEOMETRY_COMPONENT_GATE_BLOCKER = "geometry_component_gates_required"
EXACT_CUDA_BLOCKER = "exact_cuda_auth_eval_required_before_score_claim"

GEOMETRY_FEEDBACK_ALWAYS_BLOCKERS = (
    CANDIDATE_MANIFEST_BLOCKER,
    GEOMETRY_COMPONENT_GATE_BLOCKER,
    EXACT_CUDA_BLOCKER,
)
GEOMETRY_FEEDBACK_REQUIRED_DISPATCH_BLOCKERS = (
    UNCHARGED_GEOMETRY_FEEDBACK_BLOCKER,
    *GEOMETRY_FEEDBACK_ALWAYS_BLOCKERS,
)


def build_geometry_feedback_runtime_contract(
    *,
    lane_key: str,
    paradigms: Iterable[str],
    role: str,
    charged_artifacts: Iterable[Mapping[str, Any] | str] = (),
    runtime_consumers: Iterable[Mapping[str, Any] | str] = (),
    evidence_grade: str = "planning_feedback_only",
) -> dict[str, Any]:
    """Build a deterministic non-dispatching contract for geometry feedback.

    ``charged_artifacts`` and ``runtime_consumers`` are optional proof records.
    Supplying both can narrow the uncharged-feedback blocker, but this generic
    contract still does not unlock dispatch; candidate-specific archive and CUDA
    gates remain required.
    """

    if not lane_key:
        raise ValueError("lane_key is required")
    normalized_artifacts = tuple(_normalize_artifact(item) for item in charged_artifacts)
    normalized_consumers = tuple(_normalize_consumer(item) for item in runtime_consumers)
    proven_artifacts = tuple(item for item in normalized_artifacts if _artifact_is_proven(item))
    artifact_ids = {
        value
        for artifact in proven_artifacts
        for value in (artifact["path"], artifact["member"])
        if value
    }
    matching_consumers = tuple(
        consumer
        for consumer in normalized_consumers
        if bool(consumer["consumes_charged_artifact"])
        and bool(set(consumer["consumed_artifacts"]) & artifact_ids)
    )
    charged_runtime_consumed = bool(proven_artifacts and matching_consumers)

    blockers: list[str] = []
    if not proven_artifacts:
        blockers.append("geometry_feedback_no_charged_artifact_proof")
    if not matching_consumers:
        blockers.append("geometry_feedback_no_runtime_consumer_proof")
    if not charged_runtime_consumed:
        blockers.append(UNCHARGED_GEOMETRY_FEEDBACK_BLOCKER)
    blockers.extend(GEOMETRY_FEEDBACK_ALWAYS_BLOCKERS)

    return {
        "schema_version": SCHEMA_VERSION,
        "contract_name": CONTRACT_NAME,
        "lane_key": lane_key,
        "paradigms": _stable_strings(paradigms),
        "role": str(role),
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "evidence_grade": evidence_grade,
        "charged_runtime_consumed": charged_runtime_consumed,
        "charged_artifact_count": len(proven_artifacts),
        "runtime_consumer_count": len(matching_consumers),
        "charged_artifacts": list(normalized_artifacts),
        "runtime_consumers": list(normalized_consumers),
        "requirements": {
            "charged_archive_artifact_required": True,
            "runtime_consumer_required": True,
            "candidate_specific_archive_manifest_required": True,
            "geometry_component_gates_required": True,
            "exact_cuda_auth_eval_required": True,
        },
        "dispatch_blockers": _unique(blockers),
    }


def geometry_feedback_contract_failures(contract: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Return fail-closed validation failures for a geometry-feedback contract."""

    if not isinstance(contract, Mapping):
        return ("geometry_feedback_contract_missing",)
    failures: list[str] = []
    if contract.get("schema_version") != SCHEMA_VERSION:
        failures.append("geometry_feedback_contract_schema_version")
    if contract.get("contract_name") != CONTRACT_NAME:
        failures.append("geometry_feedback_contract_name")
    for field_name in ("score_claim", "dispatch_attempted", "ready_for_exact_eval_dispatch", "promotion_eligible"):
        if contract.get(field_name) is not False:
            failures.append(f"geometry_feedback_contract_{field_name}_false")

    blockers = set(_stable_strings(contract.get("dispatch_blockers") or ()))
    if contract.get("charged_runtime_consumed") is not True:
        if UNCHARGED_GEOMETRY_FEEDBACK_BLOCKER not in blockers:
            failures.append("geometry_feedback_contract_uncharged_blocker_present")
    for blocker in GEOMETRY_FEEDBACK_ALWAYS_BLOCKERS:
        if blocker not in blockers:
            failures.append(f"geometry_feedback_contract_missing_{blocker}")
    return tuple(failures)


def _normalize_artifact(item: Mapping[str, Any] | str) -> dict[str, Any]:
    if isinstance(item, str):
        return {
            "path": item,
            "member": "",
            "bytes": None,
            "sha256": "",
            "charged": bool(item),
        }
    path = str(item.get("path") or "")
    member = str(item.get("member") or "")
    byte_count = item.get("bytes")
    if not isinstance(byte_count, int) or byte_count < 0:
        byte_count = None
    charged = item.get("charged", True)
    return {
        "path": path,
        "member": member,
        "bytes": byte_count,
        "sha256": str(item.get("sha256") or ""),
        "charged": charged is True,
    }


def _normalize_consumer(item: Mapping[str, Any] | str) -> dict[str, Any]:
    if isinstance(item, str):
        return {
            "path": item,
            "consumes_charged_artifact": False,
            "consumed_artifacts": [],
        }
    consumed = item.get("consumed_artifacts") or item.get("charged_artifacts") or ()
    if isinstance(consumed, str):
        consumed = (consumed,)
    consumes_flag = item.get("consumes_charged_artifact", bool(consumed))
    return {
        "path": str(item.get("path") or ""),
        "consumes_charged_artifact": consumes_flag is True,
        "consumed_artifacts": _stable_strings(consumed),
    }


def _artifact_is_proven(item: Mapping[str, Any]) -> bool:
    return (
        item.get("charged") is True
        and bool(item.get("path") or item.get("member"))
        and isinstance(item.get("bytes"), int)
        and int(item["bytes"]) > 0
        and _is_sha256(str(item.get("sha256") or ""))
    )


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


def _stable_strings(values: Iterable[Any]) -> list[str]:
    if isinstance(values, str):
        values = (values,)
    return sorted({str(value) for value in values if str(value)})


def _unique(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


__all__ = [
    "CANDIDATE_MANIFEST_BLOCKER",
    "CONTRACT_NAME",
    "EXACT_CUDA_BLOCKER",
    "GEOMETRY_COMPONENT_GATE_BLOCKER",
    "GEOMETRY_FEEDBACK_ALWAYS_BLOCKERS",
    "GEOMETRY_FEEDBACK_REQUIRED_DISPATCH_BLOCKERS",
    "GEOMETRY_FEEDBACK_ROADMAP_KEYS",
    "SCHEMA_VERSION",
    "UNCHARGED_GEOMETRY_FEEDBACK_BLOCKER",
    "build_geometry_feedback_runtime_contract",
    "geometry_feedback_contract_failures",
]
