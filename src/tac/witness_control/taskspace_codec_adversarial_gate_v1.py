"""Retrospective-only G57 diagnostic linter for scorer-native codecs.

This v1 implementation predates the chained live G59 gate.  It may preserve a
diagnostic verdict for regression comparison, but it is not enforcement:
every receipt is explicitly ``RETROSPECTIVE_ONLY``, has
``candidate_admission=false``, and cannot admit a next stage.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

REQUEST_SCHEMA: Final = "tac.taskspace_codec_adversarial_review_request.v1"
RECEIPT_SCHEMA: Final = "tac.taskspace_codec_adversarial_review_receipt.v1"
RETROSPECTIVE_ONLY: Final = "RETROSPECTIVE_ONLY"

PRE_ENCODE: Final = "PRE_ENCODE"
PRE_PUBLIC_CLOSURE: Final = "PRE_PUBLIC_CLOSURE"
PRE_PROMOTION: Final = "PRE_PROMOTION"
POST_EVAL: Final = "POST_EVAL"
BOUNDARIES: Final = frozenset({PRE_ENCODE, PRE_PUBLIC_CLOSURE, PRE_PROMOTION, POST_EVAL})

DIRECT_CONTROL: Final = "DIRECT_TASK_LAYERED_CONTROL"
PROGRAM_RESIDUAL: Final = "PROGRAM_RESIDUAL_LAYERED"
REPRESENTATIONS: Final = frozenset({DIRECT_CONTROL, PROGRAM_RESIDUAL})

VERDICT_ADMIT: Final = "ADMIT"
VERDICT_CONTROL_ONLY: Final = "ADMIT_CONTROL_ONLY"
VERDICT_REFUSE: Final = "REFUSE"

PAIR_COUNT: Final = 600
EXPECTED_RAW_BYTES: Final = 3_662_409_600
EXPECTED_FRAME_COUNT: Final = 1200
ORIGINAL_ARCHIVE_BYTES: Final = 37_545_489
CONTEST_AXES: Final = frozenset({"[contest-CPU]", "[contest-CUDA]"})

TOP_FIELDS: Final = frozenset(
    {
        "schema",
        "review_id",
        "boundary",
        "requested_representation",
        "frontier",
        "evidence",
    }
)
FRONTIER_FIELDS: Final = frozenset({"target_score", "selection_rule", "pointer_sha256"})
PRE_ENCODE_FIELDS: Final = frozenset(
    {
        "actual_representation",
        "pair_count",
        "scorer_batch_size",
        "provider_kind",
        "source_plane_definition",
        "semantic_archive_bytes",
        "semantic_archive_sha256",
        "semantic_archive_counted",
        "semantic_archive_reopened",
        "program_packet_bytes",
        "program_packet_sha256",
        "factor_count",
        "behavior_changing_factor_count",
        "target_payload_embedded",
        "historical_payload_reused",
    }
)
PRE_PUBLIC_FIELDS: Final = frozenset(
    {
        "pair_count",
        "archive_bytes",
        "archive_sha256",
        "raw_sha256",
        "exact_components_available",
        "exact_component_source",
        "realized_through_R",
        "d_seg",
        "d_pose",
    }
)
PRE_PROMOTION_FIELDS: Final = frozenset(
    {
        "pair_count",
        "archive_bytes",
        "archive_sha256",
        "runtime_tree_sha256",
        "upstream_recursive_closure",
        "two_distinct_clean_roots",
        "fresh_decode_count_a",
        "fresh_decode_count_b",
        "resume_count_a",
        "resume_count_b",
        "raw_sha256_a",
        "raw_sha256_b",
        "raw_bytes_a",
        "raw_bytes_b",
        "frame_count_a",
        "frame_count_b",
        "axis",
    }
)
POST_EVAL_FIELDS: Final = frozenset(
    {
        "pair_count",
        "archive_bytes",
        "archive_sha256",
        "d_seg",
        "d_pose",
        "axis",
        "verdict_scope",
        "not_killed",
        "evidence_receipt_sha256",
        "integration_hooks",
        "integration_blocker",
    }
)
INTEGRATION_HOOK_FIELDS: Final = frozenset(
    {
        "sensitivity_map",
        "pareto_allocator",
        "bit_allocator",
        "autopilot",
        "continual_posterior",
        "probe_ledger",
    }
)


class CodecAdversarialGateError(ValueError):
    """Malformed request, receipt, or unsafe write."""


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: object, fields: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != fields:
        raise CodecAdversarialGateError(f"{label} must have exactly fields {sorted(fields)!r}")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CodecAdversarialGateError(f"{label} must be a nonempty string")
    return value


def _boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise CodecAdversarialGateError(f"{label} must be an exact boolean")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CodecAdversarialGateError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: object, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CodecAdversarialGateError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise CodecAdversarialGateError(f"{label} must be finite and >= {minimum}")
    return result


def _optional_number(value: object, label: str) -> float | None:
    if value is None:
        return None
    return _number(value, label)


def _sha256(value: object, label: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    text = _text(value, label).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise CodecAdversarialGateError(f"{label} must be lowercase SHA-256")
    return text


def _frontier(value: object) -> tuple[Mapping[str, Any], float]:
    frontier = _mapping(value, FRONTIER_FIELDS, "frontier")
    target = _number(frontier["target_score"], "frontier.target_score")
    _text(frontier["selection_rule"], "frontier.selection_rule")
    _sha256(frontier["pointer_sha256"], "frontier.pointer_sha256")
    return frontier, target


def _score(d_seg: float, d_pose: float, archive_bytes: int) -> dict[str, float]:
    seg_term = 100.0 * d_seg
    pose_term = math.sqrt(10.0 * d_pose)
    rate_term = 25.0 * archive_bytes / ORIGINAL_ARCHIVE_BYTES
    return {
        "seg_term": seg_term,
        "pose_term": pose_term,
        "rate_term": rate_term,
        "score": seg_term + pose_term + rate_term,
    }


def _review_pre_encode(
    evidence_value: object,
    *,
    requested: str,
) -> tuple[str, bool, bool, list[str], dict[str, Any]]:
    evidence = _mapping(evidence_value, PRE_ENCODE_FIELDS, "PRE_ENCODE evidence")
    actual = _text(evidence["actual_representation"], "actual_representation")
    pair_count = _integer(evidence["pair_count"], "pair_count", minimum=1)
    batch_size = _integer(evidence["scorer_batch_size"], "scorer_batch_size", minimum=1)
    provider_kind = _text(evidence["provider_kind"], "provider_kind")
    source_definition = _text(evidence["source_plane_definition"], "source_plane_definition")
    semantic_bytes = _integer(evidence["semantic_archive_bytes"], "semantic_archive_bytes")
    semantic_sha = _sha256(
        evidence["semantic_archive_sha256"],
        "semantic_archive_sha256",
        optional=True,
    )
    semantic_counted = _boolean(evidence["semantic_archive_counted"], "semantic_archive_counted")
    semantic_reopened = _boolean(evidence["semantic_archive_reopened"], "semantic_archive_reopened")
    packet_bytes = _integer(evidence["program_packet_bytes"], "program_packet_bytes")
    packet_sha = _sha256(evidence["program_packet_sha256"], "program_packet_sha256", optional=True)
    factor_count = _integer(evidence["factor_count"], "factor_count")
    behavior_count = _integer(
        evidence["behavior_changing_factor_count"],
        "behavior_changing_factor_count",
    )
    target_embedded = _boolean(evidence["target_payload_embedded"], "target_payload_embedded")
    historical = _boolean(evidence["historical_payload_reused"], "historical_payload_reused")
    failures: list[str] = []
    if pair_count != PAIR_COUNT:
        failures.append("PAIR_COUNT_NOT_FULL_N600")
    if batch_size != 16:
        failures.append("SCORER_BATCH_NOT_UPSTREAM_DEFAULT_16")
    if target_embedded:
        failures.append("ENCODER_ONLY_TARGET_PAYLOAD_EMBEDDED")
    if historical:
        failures.append("HISTORICAL_PAYLOAD_REUSED")

    if requested == DIRECT_CONTROL:
        if actual != "DIRECT_TASK_LAYERED":
            failures.append("ACTUAL_REPRESENTATION_NOT_DIRECT_TASK_LAYERED")
        if failures:
            return VERDICT_REFUSE, False, False, failures, {}
        return (
            VERDICT_CONTROL_ONLY,
            True,
            False,
            [],
            {
                "provider_kind": provider_kind,
                "source_plane_definition": source_definition,
                "selected_preimage_claim_allowed": False,
            },
        )

    if actual != "PROGRAM_RESIDUAL_LAYERED":
        failures.append("REQUESTED_PROGRAM_ACTUAL_REPRESENTATION_MISMATCH")
    if provider_kind != "G49_SELECTED_PREIMAGE_PROGRAM":
        failures.append("PROVIDER_IS_NOT_G49_SELECTED_PREIMAGE_PROGRAM")
    if source_definition != "G49_DECODE_SELECTED_PREIMAGE_PAIR":
        failures.append("SOURCE_PLANES_NOT_G49_DECODE_OUTPUT")
    if semantic_bytes <= 0 or semantic_sha is None:
        failures.append("FRESH_SEMANTIC_ARCHIVE_IDENTITY_ABSENT")
    if not semantic_counted:
        failures.append("FRESH_SEMANTIC_ARCHIVE_NOT_COUNTED")
    if not semantic_reopened:
        failures.append("FRESH_SEMANTIC_ARCHIVE_NOT_REOPENED")
    if packet_bytes <= 0 or packet_sha is None:
        failures.append("G49_PROGRAM_PACKET_IDENTITY_ABSENT")
    if factor_count <= 0:
        failures.append("COUNTED_FACTOR_SET_EMPTY")
    if behavior_count <= 0:
        failures.append("NO_BEHAVIOR_CHANGING_FACTOR")
    if behavior_count > factor_count:
        failures.append("BEHAVIOR_FACTOR_COUNT_EXCEEDS_FACTOR_COUNT")
    return (
        VERDICT_REFUSE if failures else VERDICT_ADMIT,
        not failures,
        not failures,
        failures,
        {
            "semantic_archive_bytes": semantic_bytes,
            "program_packet_bytes": packet_bytes,
            "factor_count": factor_count,
            "behavior_changing_factor_count": behavior_count,
            "selected_preimage_claim_allowed": not failures,
        },
    )


def _review_pre_public(
    evidence_value: object,
    *,
    target_score: float,
) -> tuple[str, bool, bool, list[str], dict[str, Any]]:
    evidence = _mapping(evidence_value, PRE_PUBLIC_FIELDS, "PRE_PUBLIC_CLOSURE evidence")
    pair_count = _integer(evidence["pair_count"], "pair_count", minimum=1)
    archive_bytes = _integer(evidence["archive_bytes"], "archive_bytes", minimum=1)
    _sha256(evidence["archive_sha256"], "archive_sha256")
    raw_sha = _sha256(evidence["raw_sha256"], "raw_sha256", optional=True)
    exact_available = _boolean(evidence["exact_components_available"], "exact_components_available")
    component_source = _text(evidence["exact_component_source"], "exact_component_source")
    through_r = _boolean(evidence["realized_through_R"], "realized_through_R")
    d_seg = _optional_number(evidence["d_seg"], "d_seg")
    d_pose = _optional_number(evidence["d_pose"], "d_pose")
    failures: list[str] = []
    if pair_count != PAIR_COUNT:
        failures.append("PAIR_COUNT_NOT_FULL_N600")
    if not exact_available or d_seg is None or d_pose is None:
        failures.append("RATE_ONLY_ADMISSION_FORBIDDEN")
    if component_source != "FULL_N600_FROZEN_SCORER_ON_EXACT_DECODED_PLANES":
        failures.append("COMPONENT_SOURCE_NOT_EXACT_FULL_N600_DECODED_PLANES")
    if not through_r or raw_sha is None:
        failures.append("REALIZED_THROUGH_R_RAW_CUSTODY_ABSENT")
    if failures:
        return (
            VERDICT_REFUSE,
            False,
            False,
            failures,
            {
                "rate_term": 25.0 * archive_bytes / ORIGINAL_ARCHIVE_BYTES,
                "score": None,
            },
        )
    assert d_seg is not None and d_pose is not None
    terms = _score(d_seg, d_pose, archive_bytes)
    competitive = terms["score"] < target_score
    if not competitive:
        failures.append("COUPLED_SCORE_NOT_STRICTLY_BELOW_DYNAMIC_FRONTIER")
    return (
        VERDICT_ADMIT if competitive else VERDICT_REFUSE,
        competitive,
        competitive,
        failures,
        terms,
    )


def _review_pre_promotion(
    evidence_value: object,
) -> tuple[str, bool, bool, list[str], dict[str, Any]]:
    evidence = _mapping(evidence_value, PRE_PROMOTION_FIELDS, "PRE_PROMOTION evidence")
    pair_count = _integer(evidence["pair_count"], "pair_count", minimum=1)
    archive_bytes = _integer(evidence["archive_bytes"], "archive_bytes", minimum=1)
    archive_sha = _sha256(evidence["archive_sha256"], "archive_sha256")
    runtime_sha = _sha256(evidence["runtime_tree_sha256"], "runtime_tree_sha256")
    recursive = _boolean(evidence["upstream_recursive_closure"], "upstream_recursive_closure")
    clean_roots = _boolean(evidence["two_distinct_clean_roots"], "two_distinct_clean_roots")
    fresh_a = _integer(evidence["fresh_decode_count_a"], "fresh_decode_count_a")
    fresh_b = _integer(evidence["fresh_decode_count_b"], "fresh_decode_count_b")
    resume_a = _integer(evidence["resume_count_a"], "resume_count_a")
    resume_b = _integer(evidence["resume_count_b"], "resume_count_b")
    raw_a = _sha256(evidence["raw_sha256_a"], "raw_sha256_a")
    raw_b = _sha256(evidence["raw_sha256_b"], "raw_sha256_b")
    raw_bytes_a = _integer(evidence["raw_bytes_a"], "raw_bytes_a", minimum=1)
    raw_bytes_b = _integer(evidence["raw_bytes_b"], "raw_bytes_b", minimum=1)
    frames_a = _integer(evidence["frame_count_a"], "frame_count_a", minimum=1)
    frames_b = _integer(evidence["frame_count_b"], "frame_count_b", minimum=1)
    axis = _text(evidence["axis"], "axis")
    failures: list[str] = []
    if pair_count != PAIR_COUNT:
        failures.append("PAIR_COUNT_NOT_FULL_N600")
    if not recursive:
        failures.append("UPSTREAM_RECURSIVE_CLOSURE_ABSENT")
    if not clean_roots or (fresh_a, fresh_b, resume_a, resume_b) != (1, 1, 0, 0):
        failures.append("TWO_DISTINCT_CLEAN_ROOT_FRESH_DECODES_ABSENT")
    if raw_a != raw_b:
        failures.append("DOUBLE_DECODE_RAW_SHA_MISMATCH")
    if (raw_bytes_a, raw_bytes_b) != (EXPECTED_RAW_BYTES, EXPECTED_RAW_BYTES):
        failures.append("DOUBLE_DECODE_RAW_BYTE_COUNT_MISMATCH")
    if (frames_a, frames_b) != (EXPECTED_FRAME_COUNT, EXPECTED_FRAME_COUNT):
        failures.append("DOUBLE_DECODE_FRAME_COUNT_MISMATCH")
    if axis not in CONTEST_AXES:
        failures.append("PROMOTION_AXIS_NOT_CONTEST_CPU_OR_CUDA")
    return (
        VERDICT_REFUSE if failures else VERDICT_ADMIT,
        not failures,
        not failures,
        failures,
        {
            "archive_bytes": archive_bytes,
            "archive_sha256": archive_sha,
            "runtime_tree_sha256": runtime_sha,
            "raw_sha256": raw_a if raw_a == raw_b else None,
        },
    )


def _review_post_eval(
    evidence_value: object,
    *,
    target_score: float,
) -> tuple[str, bool, bool, list[str], dict[str, Any]]:
    evidence = _mapping(evidence_value, POST_EVAL_FIELDS, "POST_EVAL evidence")
    pair_count = _integer(evidence["pair_count"], "pair_count", minimum=1)
    archive_bytes = _integer(evidence["archive_bytes"], "archive_bytes", minimum=1)
    _sha256(evidence["archive_sha256"], "archive_sha256")
    d_seg = _number(evidence["d_seg"], "d_seg")
    d_pose = _number(evidence["d_pose"], "d_pose")
    axis = _text(evidence["axis"], "axis")
    scope = _text(evidence["verdict_scope"], "verdict_scope")
    _sha256(evidence["evidence_receipt_sha256"], "evidence_receipt_sha256")
    not_killed_value = evidence["not_killed"]
    if not isinstance(not_killed_value, list) or not all(
        isinstance(item, str) and item.strip() for item in not_killed_value
    ):
        raise CodecAdversarialGateError("not_killed must be a nonempty list of strings")
    hooks = _mapping(
        evidence["integration_hooks"],
        INTEGRATION_HOOK_FIELDS,
        "integration_hooks",
    )
    hook_values = {key: _boolean(value, f"integration_hooks.{key}") for key, value in hooks.items()}
    blocker = evidence["integration_blocker"]
    if blocker is not None:
        _text(blocker, "integration_blocker")
    failures: list[str] = []
    if pair_count != PAIR_COUNT:
        failures.append("PAIR_COUNT_NOT_FULL_N600")
    if not scope.startswith("FORMULATION:"):
        failures.append("VERDICT_SCOPE_NOT_FORMULATION_PREFIXED")
    missing_hooks = sorted(key for key, value in hook_values.items() if not value)
    if missing_hooks and blocker is None:
        failures.append("RESULT_SIGNAL_ORPHANED_WITHOUT_TYPED_BLOCKER")
    terms = _score(d_seg, d_pose, archive_bytes)
    terms.update(
        {
            "competitive": terms["score"] < target_score,
            "axis": axis,
            "verdict_scope": scope,
            "missing_integration_hooks": missing_hooks,
        }
    )
    return (
        VERDICT_REFUSE if failures else VERDICT_ADMIT,
        not failures,
        False,
        failures,
        terms,
    )


def review_request(value: object) -> dict[str, Any]:
    """Adversarially review one boundary and return a sealed canonical receipt."""

    request = _mapping(value, TOP_FIELDS, "request")
    if request["schema"] != REQUEST_SCHEMA:
        raise CodecAdversarialGateError("request schema drift")
    review_id = _text(request["review_id"], "review_id")
    boundary = _text(request["boundary"], "boundary")
    if boundary not in BOUNDARIES:
        raise CodecAdversarialGateError(f"unsupported boundary: {boundary}")
    requested = _text(request["requested_representation"], "requested_representation")
    if requested not in REPRESENTATIONS:
        raise CodecAdversarialGateError(f"unsupported requested representation: {requested}")
    frontier, target_score = _frontier(request["frontier"])
    if boundary == PRE_ENCODE:
        result = _review_pre_encode(request["evidence"], requested=requested)
    elif boundary == PRE_PUBLIC_CLOSURE:
        result = _review_pre_public(request["evidence"], target_score=target_score)
    elif boundary == PRE_PROMOTION:
        result = _review_pre_promotion(request["evidence"])
    else:
        result = _review_post_eval(request["evidence"], target_score=target_score)
    verdict, _diagnostic_admit_next, _diagnostic_candidate_admission, failures, computed = result
    body = {
        "schema": RECEIPT_SCHEMA,
        "review_id": review_id,
        "boundary": boundary,
        "requested_representation": requested,
        "request_sha256": sha256_bytes(canonical_json(request)),
        "frontier": dict(frontier),
        "verdict": verdict,
        "authority_mode": RETROSPECTIVE_ONLY,
        "admit_next_stage": False,
        "candidate_admission": False,
        "failures": failures,
        "computed": computed,
        "recurrent_review_required": True,
        "next_boundary": {
            PRE_ENCODE: PRE_PUBLIC_CLOSURE,
            PRE_PUBLIC_CLOSURE: PRE_PROMOTION,
            PRE_PROMOTION: POST_EVAL,
            POST_EVAL: None,
        }[boundary],
    }
    return {**body, "body_sha256": sha256_bytes(canonical_json(body))}


def verify_receipt(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CodecAdversarialGateError("receipt must be an object")
    receipt = dict(value)
    body_sha = _sha256(receipt.pop("body_sha256", None), "body_sha256")
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise CodecAdversarialGateError("receipt schema drift")
    if sha256_bytes(canonical_json(receipt)) != body_sha:
        raise CodecAdversarialGateError("receipt body SHA mismatch")
    return dict(value)


def write_once_receipt(path: Path | str, receipt: Mapping[str, Any]) -> Path:
    """Atomically create a receipt, or accept an exact existing receipt."""

    target = Path(path)
    payload = canonical_json(verify_receipt(receipt))
    if target.exists():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != payload:
            raise CodecAdversarialGateError("existing receipt differs or is not a regular file")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            if target.is_symlink() or not target.is_file() or target.read_bytes() != payload:
                raise CodecAdversarialGateError("receipt publication raced with different content") from exc
        return target
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "DIRECT_CONTROL",
    "POST_EVAL",
    "PRE_ENCODE",
    "PRE_PROMOTION",
    "PRE_PUBLIC_CLOSURE",
    "PROGRAM_RESIDUAL",
    "RECEIPT_SCHEMA",
    "REQUEST_SCHEMA",
    "RETROSPECTIVE_ONLY",
    "CodecAdversarialGateError",
    "canonical_json",
    "review_request",
    "verify_receipt",
    "write_once_receipt",
]
