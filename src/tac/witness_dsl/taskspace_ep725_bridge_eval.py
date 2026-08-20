# SPDX-License-Identifier: MIT
"""Strict same-object bridge validation for the ep725 selected archive.

G20/G25 change only the physical spelling of one decoded state.  G22 proves
that the selected spelling reaches the same full-population uint8 witness as
the source spelling.  This module validates that proof and derives the coupled
frontier envelope for a scorer row measured on *that exact object*.

The scorer axis remains explicit.  A macOS CPU row closes the previously
missing ``(d_seg, d_pose, archive_bytes)`` bridge for scientific steering, but
does not become a contest-CPU/CUDA score or mutate the frontier pointer.
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

PAIR_COUNT = 600
FRAME_COUNT = 1_200
CAMERA_H = 874
CAMERA_W = 1_164
CHANNELS = 3
RAW_BYTES = FRAME_COUNT * CAMERA_H * CAMERA_W * CHANNELS
SOURCE_BYTES = 37_545_489


class TaskspaceBridgeEvalError(RuntimeError):
    """Raised when bridge custody or score geometry is incomplete."""


@dataclass(frozen=True)
class JsonEvidence:
    path: str
    bytes: int
    sha256: str
    value: Mapping[str, Any]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise TaskspaceBridgeEvalError(f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def read_json_evidence(path: Path) -> JsonEvidence:
    """Reopen one JSON object with duplicate-key and byte custody."""

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise TaskspaceBridgeEvalError(f"cannot read evidence {path}: {exc}") from exc
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise TaskspaceBridgeEvalError(f"invalid JSON evidence {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise TaskspaceBridgeEvalError(f"evidence root must be an object: {path}")
    return JsonEvidence(
        path=str(path.resolve()),
        bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
        value=value,
    )


def file_identity(path: Path) -> dict[str, Any]:
    """Hash one regular, non-symlink file."""

    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise TaskspaceBridgeEvalError(f"expected regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TaskspaceBridgeEvalError(f"{field} must be an object")
    return value


def _finite_nonnegative(value: Any, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise TaskspaceBridgeEvalError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise TaskspaceBridgeEvalError(f"{field} must be finite and nonnegative")
    return parsed


def _require_file_row(row: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    try:
        path = Path(str(row["path"]))
        expected_bytes = int(row["bytes"])
        expected_sha256 = str(row["sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TaskspaceBridgeEvalError(f"{label} file row is malformed") from exc
    actual = file_identity(path)
    if actual["bytes"] != expected_bytes or actual["sha256"] != expected_sha256:
        raise TaskspaceBridgeEvalError(f"{label} file custody mismatch")
    return actual


def validate_terminal_g22(receipt: JsonEvidence) -> dict[str, Any]:
    """Validate a terminal full-n600 G22 receiver-equality receipt."""

    root = receipt.value
    if root.get("schema") != "tac.g22_ep725_xcodec_decode_receipt.v1":
        raise TaskspaceBridgeEvalError("G22 final receipt schema mismatch")

    archive = _mapping(root.get("archive_artifact"), "G22 archive_artifact")
    selected = _mapping(archive.get("selected"), "G22 selected archive")
    runtime = _mapping(root.get("generic_decoder_runtime"), "G22 runtime")
    decode = _mapping(root.get("decode_receipt"), "G22 decode receipt")
    output = _mapping(root.get("output_witness"), "G22 output witness")
    cleanup = _mapping(root.get("cleanup"), "G22 cleanup")
    authority = _mapping(root.get("authority_pointer_status"), "G22 authority")

    pair_ids = decode.get("pair_ids")
    expected_pair_ids = list(range(PAIR_COUNT))
    if pair_ids != expected_pair_ids:
        raise TaskspaceBridgeEvalError("G22 pair population is not exact ordered 0..599")
    if (
        decode.get("pair_population_exact_ordered_0_through_599") is not True
        or decode.get("all_chunks_immutable_and_validated") is not True
        or not isinstance(decode.get("chunk_checkpoints"), list)
        or len(decode["chunk_checkpoints"]) != decode.get("chunk_count")
    ):
        raise TaskspaceBridgeEvalError("G22 immutable full-population decode proof is incomplete")

    required_output = {
        "pairs_compared": PAIR_COUNT,
        "frames_compared": FRAME_COUNT,
        "raw_bytes_each": RAW_BYTES,
        "camera_h": CAMERA_H,
        "camera_w": CAMERA_W,
        "dtype": "uint8",
        "all_bytes_directly_compared": True,
        "uint8_exact_equal": True,
    }
    mismatches = {
        key: {"expected": expected, "actual": output.get(key)}
        for key, expected in required_output.items()
        if output.get(key) != expected
    }
    if output.get("source_raw_sha256") != output.get("selected_raw_sha256"):
        mismatches["raw_sha256"] = "source and selected differ"
    if mismatches:
        raise TaskspaceBridgeEvalError(f"G22 output witness mismatch: {mismatches}")

    if (
        cleanup.get("completed") is not True
        or cleanup.get("scratch_absent") is not True
        or cleanup.get("per_chunk_checkpoints_preserved") is not True
        or authority.get("full_n600_receiver_replay_owed") is not False
        or authority.get("decode_equality_invalidated") is not False
        or authority.get("exact_eval_invoked") is not False
    ):
        raise TaskspaceBridgeEvalError("G22 terminal cleanup/authority closure is incomplete")

    selected_file = _require_file_row(selected, label="G22 selected archive")
    runtime_file = _require_file_row(runtime, label="G22 generic decoder runtime")
    member = _mapping(selected.get("member"), "G22 selected member")
    if selected.get("counted_rate_bytes") != selected_file["bytes"]:
        raise TaskspaceBridgeEvalError("G22 counted rate differs from selected ZIP bytes")
    if selected.get("classification") != "not_a_candidate":
        raise TaskspaceBridgeEvalError("G22 selected archive classification drift")
    if runtime.get("contest_rate_cost_bytes") != 0 or runtime.get("free_under_rule_118") is not True:
        raise TaskspaceBridgeEvalError("G22 generic runtime source-boundary typing drift")

    raw_sha256 = str(output.get("selected_raw_sha256"))
    if len(raw_sha256) != 64:
        raise TaskspaceBridgeEvalError("G22 selected raw SHA-256 is malformed")
    return {
        "receipt": {
            "path": receipt.path,
            "bytes": receipt.bytes,
            "sha256": receipt.sha256,
        },
        "selected_archive": selected_file,
        "selected_member": {
            "name": member.get("name"),
            "bytes": member.get("bytes"),
            "sha256": member.get("sha256"),
        },
        "decoded_quantized_state_sha256": selected.get("decoded_quantized_state_sha256"),
        "runtime": runtime_file,
        "selected_raw_sha256": raw_sha256,
        "pair_count": PAIR_COUNT,
        "frame_count": FRAME_COUNT,
        "raw_bytes": RAW_BYTES,
        "decode_equality_independent_of_pointer_change": True,
    }


def current_competitive_target(pointer: JsonEvidence) -> tuple[float, dict[str, Any]]:
    """Recompute the target from pointer constituents, never a cached literal."""

    try:
        parsed = CanonicalFrontierPointer.from_dict(dict(pointer.value))
        score = effective_frontier_score(parsed)
        row = recompute_effective_frontier(parsed)
    except (KeyError, TypeError, ValueError) as exc:
        raise TaskspaceBridgeEvalError(f"canonical pointer is invalid: {exc}") from exc
    if score is None or row is None:
        raise TaskspaceBridgeEvalError("canonical pointer has no competitive target")
    cached = pointer.value.get("effective_frontier")
    if isinstance(cached, Mapping):
        cached_score = _finite_nonnegative(cached.get("score"), "cached pointer score")
        if not math.isclose(cached_score, score, rel_tol=0.0, abs_tol=1e-15):
            raise TaskspaceBridgeEvalError("cached pointer target disagrees with constituent minimum")
    return score, {
        **dict(row),
        "score": score,
        "pointer_evidence": {
            "path": pointer.path,
            "bytes": pointer.bytes,
            "sha256": pointer.sha256,
        },
    }


def coupled_score_envelope(
    *,
    d_seg: float,
    d_pose: float,
    archive_bytes: int,
    target_score: float,
) -> dict[str, Any]:
    """Derive coupled, conditional budgets for one measured object."""

    d_seg = _finite_nonnegative(d_seg, "d_seg")
    d_pose = _finite_nonnegative(d_pose, "d_pose")
    target_score = _finite_nonnegative(target_score, "target_score")
    if type(archive_bytes) is not int or archive_bytes <= 0:
        raise TaskspaceBridgeEvalError("archive_bytes must be a positive integer")

    seg_component = 100.0 * d_seg
    pose_component = math.sqrt(10.0 * d_pose)
    rate_component = 25.0 * archive_bytes / SOURCE_BYTES
    score = contest_score(d_seg, d_pose, archive_bytes)
    budget = target_byte_budget_for_score(
        target_score=target_score,
        d_seg_floor=d_seg,
        d_pose_floor=d_pose,
        current_archive_bytes=archive_bytes,
    )

    pose_residual = target_score - seg_component - rate_component
    max_pose_same_seg_rate = (pose_residual * pose_residual / 10.0) if pose_residual >= 0.0 else None
    seg_residual = target_score - pose_component - rate_component
    max_seg_same_pose_rate = seg_residual / 100.0 if seg_residual >= 0.0 else None
    rate_residual = target_score - seg_component - pose_component

    return {
        "formula": "100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37545489",
        "d_seg": d_seg,
        "d_pose": d_pose,
        "archive_bytes": archive_bytes,
        "seg_component": seg_component,
        "pose_component": pose_component,
        "rate_component": rate_component,
        "score": score,
        "target_score": target_score,
        "score_minus_target": score - target_score,
        "inside_target_sublevel": score < target_score,
        "conditional_boundaries": {
            "max_d_pose_at_same_d_seg_and_bytes": max_pose_same_seg_rate,
            "max_d_seg_at_same_d_pose_and_bytes": max_seg_same_pose_rate,
            "max_archive_bytes_at_same_distortions": budget.max_archive_bytes,
            "rate_score_headroom_at_same_distortions": rate_residual,
        },
        "independent_axis_thresholds_forbidden": True,
    }


def build_bridge_receipt(
    *,
    g22: JsonEvidence,
    pointer: JsonEvidence,
    scorer_row: Mapping[str, Any],
    scorer_evidence: JsonEvidence,
    run_custody: Mapping[str, Any],
    scored_archive: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose a full-population advisory bridge receipt."""

    g22_closure = validate_terminal_g22(g22)
    target, target_row = current_competitive_target(pointer)
    if scorer_row.get("pair_count") != PAIR_COUNT or scorer_row.get("device") != "cpu":
        raise TaskspaceBridgeEvalError("scorer row must be exact n600 CPU")
    if scorer_row.get("deterministic_algorithms") is not True:
        raise TaskspaceBridgeEvalError("scorer row lacks deterministic-algorithm custody")
    raw_sha256 = str(scorer_row.get("raw_sha256"))
    if raw_sha256 != g22_closure["selected_raw_sha256"]:
        raise TaskspaceBridgeEvalError("scored raw bytes differ from terminal G22 witness")
    d_seg = _finite_nonnegative(scorer_row.get("d_seg"), "scorer d_seg")
    d_pose = _finite_nonnegative(scorer_row.get("d_pose"), "scorer d_pose")
    if scored_archive is None:
        scored_archive_row = dict(g22_closure["selected_archive"])
    else:
        scored_archive_row = _require_file_row(scored_archive, label="scored archive")
    archive_bytes = int(scored_archive_row["bytes"])
    envelope = coupled_score_envelope(
        d_seg=d_seg,
        d_pose=d_pose,
        archive_bytes=archive_bytes,
        target_score=target,
    )
    core: dict[str, Any] = {
        "schema": "tac.taskspace_ep725_same_object_bridge_eval.v1",
        "authority": {
            "axis": "[macOS-CPU frozen-scorer advisory]",
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "pointer_moved": False,
            "contest_cpu_cuda_same_bytes_owed": True,
        },
        "verdict": (
            "MEASURED_FULL_N600_SAME_OBJECT_BRIDGE_INSIDE_ADVISORY_TARGET"
            if envelope["inside_target_sublevel"]
            else "MEASURED_FULL_N600_SAME_OBJECT_BRIDGE_OUTSIDE_TARGET"
        ),
        "verdict_scope": (
            "exact selected archive/runtime/output lineage and deterministic full-n600 macOS CPU "
            "frozen-scorer row; contest CPU/CUDA and pointer admission remain separately owed"
        ),
        "artifact_identity": {
            **g22_closure,
            "scored_archive": scored_archive_row,
            "scored_archive_matches_g22_selected": scored_archive_row
            == g22_closure["selected_archive"],
            "scored_output_matches_g22_reference": True,
        },
        "semantic_control_identity": target_row,
        "proof_dependency_set": {
            "decode_equality": ["selected archive", "selected member", "generic runtime", "G22 receipt"],
            "score_row": ["selected raw SHA-256", "frozen scorer", "source video", "scorer arithmetic"],
            "competitive_admission": ["current canonical frontier pointer"],
            "pointer_change_invalidates_decode_equality": False,
        },
        "score_row": {
            **dict(scorer_row),
            **envelope,
            "scorer_evidence": {
                "path": scorer_evidence.path,
                "bytes": scorer_evidence.bytes,
                "sha256": scorer_evidence.sha256,
            },
        },
        "run_custody": dict(run_custody),
        "system_wire_in": {
            "sensitivity_map": "component debt and conditional target boundaries on one exact object",
            "pareto_constraint": "retain whole-object archive/runtime/output identity while lowering coupled action",
            "bit_allocator_hook": "price R10 and semantic substitutions against exact component and byte headroom",
            "cathedral_autopilot_hook": "route next homotopy rung from measured dominant component debt",
            "continual_learning_update": "same-object bridge row supersedes cross-object scalar-distance refusal",
            "probe_disambiguator": "R10 relay-only, semantic-only, and joint controls on the same archive lineage",
        },
    }
    core["receipt_identity_sha256"] = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return core


__all__ = [
    "CAMERA_H",
    "CAMERA_W",
    "FRAME_COUNT",
    "PAIR_COUNT",
    "RAW_BYTES",
    "SOURCE_BYTES",
    "JsonEvidence",
    "TaskspaceBridgeEvalError",
    "build_bridge_receipt",
    "coupled_score_envelope",
    "current_competitive_target",
    "file_identity",
    "read_json_evidence",
    "validate_terminal_g22",
]
