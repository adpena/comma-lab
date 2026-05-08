"""Fail-closed readiness planner for PR101 frame-conditional A5 packets.

The A5 byte anchor proves a typed side-info wire contract and a local byte
proxy. It is not a packet-local runtime patch, runtime-consumption proof, or
score artifact. This module turns that boundary into a deterministic readiness
manifest that names the exact missing artifacts before any exact-eval spend.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

from tac.codec.frame_conditional_bit_budget import (
    FRAME_CONDITIONAL_LATENT_WIRE_SCHEMA,
)
from tac.repo_io import repo_relative, sha256_file

SCHEMA = "pr101_frame_conditional_packet_readiness.v1"
A5_ANCHOR_SCHEMA = "pr101_frame_conditional_bit_anchor.v1"

CANDIDATE_ARCHIVE_MANIFEST = "candidate_archive_manifest"
PACKET_RUNTIME_PATCH_MANIFEST = "packet_local_runtime_patch_manifest"
RUNTIME_CONSUMPTION_PROOF = "frame_conditional_runtime_consumption_proof"
PER_PAIR_SCORE_MARGINAL_MANIFEST = "per_pair_score_marginal_manifest"
STRICT_PRE_SUBMISSION_COMPLIANCE_JSON = "strict_pre_submission_compliance_json"
FALSE_AUTHORITY_KEYS = frozenset(
    {
        "score_claim",
        "promotion_eligible",
        "rank_or_kill_eligible",
        "score_promotion_allowed",
        "ready_for_exact_eval_dispatch",
        "dispatch_attempted",
    }
)

REQUIRED_ARTIFACTS: tuple[tuple[str, str, str], ...] = (
    (
        CANDIDATE_ARCHIVE_MANIFEST,
        "A5 candidate archive manifest with changed charged bytes",
        "missing_candidate_archive_manifest",
    ),
    (
        PACKET_RUNTIME_PATCH_MANIFEST,
        "packet-local PR101 inflate/runtime patch manifest",
        "missing_packet_local_runtime_patch_manifest",
    ),
    (
        RUNTIME_CONSUMPTION_PROOF,
        "runtime proof binding side-info and variable-width latent payload SHAs",
        "missing_frame_conditional_runtime_consumption_proof",
    ),
    (
        PER_PAIR_SCORE_MARGINAL_MANIFEST,
        "per-pair score-marginal evidence for the A5 q-bit redistribution",
        "missing_per_pair_score_marginal_manifest",
    ),
    (
        STRICT_PRE_SUBMISSION_COMPLIANCE_JSON,
        "strict pre-submission compliance JSON for the exact packet",
        "missing_strict_pre_submission_compliance_json",
    ),
)

ARTIFACT_ORDER = tuple(row[0] for row in REQUIRED_ARTIFACTS)


class FrameConditionalPacketReadinessError(ValueError):
    """Raised when a readiness input cannot be interpreted."""


def build_packet_readiness(
    *,
    a5_manifest_path: Path,
    artifact_paths: Mapping[str, Path | None] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build a fail-closed A5 packet-readiness manifest.

    The returned manifest never promotes score and never marks dispatch-ready
    in this local planner. When every local packet prerequisite is supplied and
    valid, it sets ``ready_for_exact_eval_after_lane_claim`` so the operator can
    see that the remaining gate is the Level-2 claim plus exact eval itself.
    """

    root = Path.cwd() if repo_root is None else Path(repo_root)
    manifest = _load_json_object(a5_manifest_path)
    artifact_paths = dict(artifact_paths or {})

    a5_summary, a5_blockers = _summarize_a5_manifest(
        manifest,
        manifest_path=a5_manifest_path,
        repo_root=root,
    )
    sideinfo_sha = a5_summary.get("q_bits_sideinfo_sha256")
    latent_sha = a5_summary.get("latent_wire_payload_sha256")
    expected_n_pairs = _as_int(a5_summary.get("n_pairs"))
    source_archive_sha = _as_str(a5_summary.get("input_archive_sha256"))
    source_archive_bytes = _as_int(a5_summary.get("input_archive_bytes"))

    artifact_records: list[dict[str, Any]] = []
    missing_artifacts: list[str] = []
    invalid_artifacts: list[str] = []
    local_blockers = list(a5_blockers)

    for requirement_id, description, missing_blocker in REQUIRED_ARTIFACTS:
        record = _build_artifact_record(
            requirement_id=requirement_id,
            description=description,
            path=artifact_paths.get(requirement_id),
            missing_blocker=missing_blocker,
            repo_root=root,
            context={
                "sideinfo_sha": sideinfo_sha,
                "latent_sha": latent_sha,
                "expected_n_pairs": expected_n_pairs,
                "source_archive_sha": source_archive_sha,
                "source_archive_bytes": source_archive_bytes,
            },
        )
        artifact_records.append(record)
        if record["available"] is not True:
            missing_artifacts.append(requirement_id)
            local_blockers.append(missing_blocker)
        if record.get("validation_blockers"):
            invalid_artifacts.append(requirement_id)
            local_blockers.extend(record["validation_blockers"])

    local_blockers = _dedupe(local_blockers)
    ready_after_lane_claim = not local_blockers
    dispatch_blockers = list(local_blockers)
    dispatch_blockers.append("requires_level2_dispatch_claim_before_exact_eval")
    dispatch_blockers.append("requires_exact_cuda_auth_eval_before_score_promotion")
    dispatch_blockers = _dedupe(dispatch_blockers)

    return {
        "schema": SCHEMA,
        "tool": "tac.pr101_frame_conditional_packet_readiness",
        "a5_manifest": a5_summary,
        "packet_artifacts": artifact_records,
        "missing_artifacts": missing_artifacts,
        "invalid_artifacts": invalid_artifacts,
        "readiness_blockers": local_blockers,
        "dispatch_blockers": dispatch_blockers,
        "operator_missing_work": _operator_missing_work(
            missing_artifacts=missing_artifacts,
            invalid_artifacts=invalid_artifacts,
            a5_summary=a5_summary,
        ),
        "exact_eval_command_status": {
            "local_packet_prerequisites_ready": ready_after_lane_claim,
            "command_emitted": False,
            "reason": (
                "packet prerequisites satisfied; Level-2 dispatch claim and "
                "operator-approved exact eval still required"
                if ready_after_lane_claim
                else "packet prerequisites missing or invalid"
            ),
            "remote_or_gpu_eval_started": False,
        },
        "ready_for_local_packet_review": ready_after_lane_claim,
        "ready_for_archive_preflight": ready_after_lane_claim,
        "ready_for_exact_eval_after_lane_claim": ready_after_lane_claim,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "dispatch_attempted": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "score_promotion_allowed": False,
        "evidence_semantics": "fail_closed_a5_packet_readiness_no_score",
    }


def existing_artifact_input_paths(
    a5_manifest_path: Path,
    artifact_paths: Mapping[str, Path | None] | None = None,
) -> list[Path]:
    """Return existing JSON inputs for a tool-run manifest."""

    paths = [a5_manifest_path]
    for requirement_id in ARTIFACT_ORDER:
        path = (artifact_paths or {}).get(requirement_id)
        if path is not None and Path(path).is_file():
            paths.append(Path(path))
    return paths


def _summarize_a5_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    if manifest.get("schema") != A5_ANCHOR_SCHEMA:
        blockers.append("a5_manifest_schema_not_pr101_frame_conditional_bit_anchor_v1")
    if manifest.get("score_claim") is not False:
        blockers.append("a5_manifest_must_not_claim_score")
    if manifest.get("byte_proxy_only") is not True:
        blockers.append("a5_manifest_must_be_byte_proxy_only")
    if manifest.get("ready_for_exact_eval_dispatch") is True:
        blockers.append("a5_manifest_unexpectedly_claims_exact_eval_readiness")
    for path in _true_authority_flag_paths(manifest):
        blockers.append(f"a5_manifest_authority_flag_true:{path}")

    rows = manifest.get("rows")
    best_row: Mapping[str, Any] | None = None
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        row_dicts = [row for row in rows if isinstance(row, Mapping)]
        if row_dicts:
            best_row = min(
                row_dicts,
                key=lambda row: float(row.get("archive_delta_bytes", float("inf"))),
            )
        else:
            blockers.append("a5_manifest_rows_have_no_objects")
    else:
        blockers.append("a5_manifest_rows_missing")
    if best_row is not None:
        best_delta = best_row.get("archive_delta_bytes")
        if not _is_finite_number(best_delta):
            blockers.append("best_a5_row_archive_delta_bytes_not_finite")
        elif float(best_delta) >= 0.0:
            blockers.append("best_a5_row_archive_delta_bytes_not_negative")

    wire_contract: Mapping[str, Any] = {}
    if best_row is not None and isinstance(best_row.get("frame_conditional_wire_contract"), Mapping):
        wire_contract = best_row["frame_conditional_wire_contract"]  # type: ignore[assignment]
    else:
        blockers.append("best_a5_row_missing_frame_conditional_wire_contract")

    status = manifest.get("frame_conditional_wire_contract_status")
    if not isinstance(status, Mapping):
        status = {}
        blockers.append("a5_manifest_wire_contract_status_missing")

    if wire_contract.get("schema") != FRAME_CONDITIONAL_LATENT_WIRE_SCHEMA:
        blockers.append("a5_wire_contract_schema_invalid")
    if status.get("typed_sideinfo_wire_contract_landed") is not True:
        blockers.append("typed_sideinfo_wire_contract_not_landed")
    if wire_contract.get("decoder_helper_consumes_sideinfo_bytes") is not True:
        blockers.append("decoder_helper_consumes_sideinfo_bytes_not_true")
    if _nested_bool(wire_contract, "q_bits_roundtrip", "passed") is not True:
        blockers.append("q_bits_sideinfo_roundtrip_not_proven")
    if _nested_bool(wire_contract, "latent_decode_roundtrip", "passed") is not True:
        blockers.append("latent_wire_decode_roundtrip_not_proven")

    sideinfo = _nested_mapping(wire_contract, "q_bits_sideinfo")
    latent_payload = _nested_mapping(wire_contract, "latent_wire_payload")
    sideinfo_sha = _as_str(sideinfo.get("sha256"))
    latent_sha = _as_str(latent_payload.get("sha256"))
    if not _is_sha256(sideinfo_sha):
        blockers.append("q_bits_sideinfo_sha256_missing")
    if not _is_sha256(latent_sha):
        blockers.append("latent_wire_payload_sha256_missing")
    if latent_payload.get("score_affecting_payload_changed") is not True:
        blockers.append("latent_wire_payload_changed_not_true")

    return (
        {
            "path": repo_relative(manifest_path, repo_root),
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
            "schema": manifest.get("schema"),
            "input_archive": manifest.get("input_archive"),
            "input_archive_bytes": manifest.get("input_archive_bytes"),
            "input_archive_sha256": manifest.get("input_archive_sha256"),
            "n_pairs": manifest.get("n_pairs"),
            "latent_dim": manifest.get("latent_dim"),
            "best_eta": manifest.get("best_eta"),
            "best_archive_delta_bytes": manifest.get("best_archive_delta_bytes"),
            "score_claim": False,
            "byte_proxy_only": manifest.get("byte_proxy_only") is True,
            "ready_for_exact_eval_dispatch": False,
            "wire_schema": wire_contract.get("schema"),
            "q_bits_sideinfo_bytes": sideinfo.get("bytes"),
            "q_bits_sideinfo_sha256": sideinfo_sha,
            "latent_wire_payload_bytes": latent_payload.get("bytes"),
            "latent_wire_payload_sha256": latent_sha,
            "latent_wire_payload_changed": latent_payload.get(
                "score_affecting_payload_changed"
            ),
        },
        _dedupe(blockers),
    )


def _build_artifact_record(
    *,
    requirement_id: str,
    description: str,
    path: Path | None,
    missing_blocker: str,
    repo_root: Path,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": requirement_id,
        "description": description,
        "required_before_exact_eval": True,
        "available": False,
        "missing_blocker": missing_blocker,
        "score_claim": False,
    }
    if path is None:
        record["missing_reason"] = "path_not_supplied"
        return record
    path = Path(path)
    record["path"] = repo_relative(path, repo_root)
    if not path.is_file():
        record["missing_reason"] = "path_not_found"
        return record
    record["bytes"] = path.stat().st_size
    record["sha256"] = sha256_file(path)

    try:
        payload = _load_json_object(path)
    except (OSError, json.JSONDecodeError, FrameConditionalPacketReadinessError) as exc:
        record["missing_reason"] = "json_unreadable"
        record["validation_blockers"] = [f"{requirement_id}:json_unreadable:{exc}"]
        return record

    validators = {
        CANDIDATE_ARCHIVE_MANIFEST: _validate_candidate_archive_manifest,
        PACKET_RUNTIME_PATCH_MANIFEST: _validate_runtime_patch_manifest,
        RUNTIME_CONSUMPTION_PROOF: _validate_runtime_consumption_proof,
        PER_PAIR_SCORE_MARGINAL_MANIFEST: _validate_score_marginal_manifest,
        STRICT_PRE_SUBMISSION_COMPLIANCE_JSON: _validate_strict_compliance_json,
    }
    blockers = validators[requirement_id](payload, context)
    if blockers:
        record["missing_reason"] = "validation_blockers"
        record["validation_blockers"] = blockers
        return record
    record["available"] = True
    record["missing_reason"] = ""
    record["validation_blockers"] = []
    return record


def _validate_candidate_archive_manifest(
    payload: Mapping[str, Any], context: Mapping[str, Any]
) -> list[str]:
    blockers: list[str] = []
    if payload.get("score_claim") is True:
        blockers.append(f"{CANDIDATE_ARCHIVE_MANIFEST}:score_claim_must_be_false")
    archive = _first_mapping(payload, "candidate_archive", "archive")
    archive_sha = _as_str(archive.get("sha256"))
    archive_bytes = _as_int(archive.get("bytes"))
    if not _is_sha256(archive_sha):
        blockers.append(f"{CANDIDATE_ARCHIVE_MANIFEST}:candidate_archive_sha256_missing")
    if archive_bytes is None or archive_bytes <= 0:
        blockers.append(f"{CANDIDATE_ARCHIVE_MANIFEST}:candidate_archive_bytes_missing")
    source_sha = _as_str(context.get("source_archive_sha"))
    source_bytes = _as_int(context.get("source_archive_bytes"))
    if _is_sha256(source_sha) and archive_sha == source_sha:
        blockers.append(
            f"{CANDIDATE_ARCHIVE_MANIFEST}:candidate_archive_sha256_matches_source"
        )
    if source_bytes is not None and archive_bytes is not None and archive_bytes >= source_bytes:
        blockers.append(
            f"{CANDIDATE_ARCHIVE_MANIFEST}:candidate_archive_bytes_not_below_source"
        )
    if payload.get("score_affecting_payload_changed") is not True and payload.get(
        "charged_bits_changed"
    ) is not True:
        blockers.append(f"{CANDIDATE_ARCHIVE_MANIFEST}:charged_bits_changed_not_true")
    return blockers


def _validate_runtime_patch_manifest(
    payload: Mapping[str, Any], context: Mapping[str, Any]
) -> list[str]:
    del context
    blockers: list[str] = []
    if payload.get("score_claim") is True:
        blockers.append(f"{PACKET_RUNTIME_PATCH_MANIFEST}:score_claim_must_be_false")
    patch = _first_mapping(payload, "packet_local_runtime_patch", "runtime_patch", "patch")
    schema = _as_str(
        patch.get("consumes_schema")
        or patch.get("wire_schema")
        or payload.get("consumes_schema")
        or payload.get("wire_schema")
    )
    if schema != FRAME_CONDITIONAL_LATENT_WIRE_SCHEMA:
        blockers.append(f"{PACKET_RUNTIME_PATCH_MANIFEST}:wire_schema_not_a5_v1")
    if not _truthy_any(
        patch,
        payload,
        "parse_archive_consumes_q_bits_sideinfo",
        "runtime_consumes_q_bits_sideinfo",
    ):
        blockers.append(
            f"{PACKET_RUNTIME_PATCH_MANIFEST}:parse_archive_consumes_q_bits_sideinfo_not_true"
        )
    if not _truthy_any(
        patch,
        payload,
        "decode_latents_consumes_variable_width_payload",
        "runtime_consumes_variable_width_latent_payload",
    ):
        blockers.append(
            f"{PACKET_RUNTIME_PATCH_MANIFEST}:variable_width_latent_payload_not_consumed"
        )
    return blockers


def _validate_runtime_consumption_proof(
    payload: Mapping[str, Any], context: Mapping[str, Any]
) -> list[str]:
    blockers: list[str] = []
    if payload.get("score_claim") is True:
        blockers.append(f"{RUNTIME_CONSUMPTION_PROOF}:score_claim_must_be_false")
    if payload.get("ready_for_exact_eval_runtime") is not True and payload.get(
        "runtime_consumption_proven"
    ) is not True:
        blockers.append(f"{RUNTIME_CONSUMPTION_PROOF}:runtime_not_proven_ready")
    for label, key in (
        ("q_bits_sideinfo_sha256", "sideinfo_sha"),
        ("latent_wire_payload_sha256", "latent_sha"),
    ):
        expected = _as_str(context.get(key))
        if _is_sha256(expected) and not _json_contains_string(payload, expected):
            blockers.append(f"{RUNTIME_CONSUMPTION_PROOF}:missing_{label}")
    return blockers


def _validate_score_marginal_manifest(
    payload: Mapping[str, Any], context: Mapping[str, Any]
) -> list[str]:
    blockers: list[str] = []
    if payload.get("score_claim") is True:
        blockers.append(f"{PER_PAIR_SCORE_MARGINAL_MANIFEST}:score_claim_must_be_false")
    for path in _true_authority_flag_paths(payload):
        blockers.append(f"{PER_PAIR_SCORE_MARGINAL_MANIFEST}:authority_flag_true:{path}")
    expected_n_pairs = _as_int(context.get("expected_n_pairs"))
    manifest_n_pairs = _as_int(payload.get("n_pairs"))
    if expected_n_pairs is not None and manifest_n_pairs != expected_n_pairs:
        blockers.append(f"{PER_PAIR_SCORE_MARGINAL_MANIFEST}:n_pairs_mismatch")
    marginals = payload.get("per_pair_score_marginals")
    ready = payload.get("per_pair_score_marginals_ready") is True or payload.get(
        "marginal_evidence_available"
    ) is True
    if isinstance(marginals, Sequence) and not isinstance(marginals, (str, bytes)):
        if expected_n_pairs is not None and len(marginals) != expected_n_pairs:
            blockers.append(f"{PER_PAIR_SCORE_MARGINAL_MANIFEST}:marginal_count_mismatch")
        if not _all_finite_numbers(marginals):
            blockers.append(f"{PER_PAIR_SCORE_MARGINAL_MANIFEST}:marginals_not_finite_numeric")
        else:
            ready = ready or (expected_n_pairs is None or len(marginals) == expected_n_pairs)
    if not ready:
        blockers.append(
            f"{PER_PAIR_SCORE_MARGINAL_MANIFEST}:per_pair_score_marginals_not_ready"
        )
    return blockers


def _validate_strict_compliance_json(
    payload: Mapping[str, Any], context: Mapping[str, Any]
) -> list[str]:
    del context
    blockers: list[str] = []
    if payload.get("score_claim") is True:
        blockers.append(f"{STRICT_PRE_SUBMISSION_COMPLIANCE_JSON}:score_claim_must_be_false")
    status = _as_str(payload.get("status")).lower()
    passed = (
        payload.get("ok") is True
        or payload.get("passed") is True
        or status in {"ok", "pass", "passed", "success"}
    )
    if not passed:
        blockers.append(f"{STRICT_PRE_SUBMISSION_COMPLIANCE_JSON}:strict_check_not_passed")
    return blockers


def _operator_missing_work(
    *,
    missing_artifacts: Sequence[str],
    invalid_artifacts: Sequence[str],
    a5_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    missing = set(missing_artifacts) | set(invalid_artifacts)
    work: list[dict[str, Any]] = []
    if CANDIDATE_ARCHIVE_MANIFEST in missing:
        work.append(
            {
                "id": CANDIDATE_ARCHIVE_MANIFEST,
                "next_action": (
                    "materialize an A5 archive.zip that replaces PR101 latent "
                    "bytes with the variable-width payload and includes the "
                    "q-bit side-info consumed by the runtime"
                ),
                "acceptance": [
                    "candidate archive SHA-256 differs from source",
                    "candidate archive bytes are below source archive bytes",
                    "charged_bits_changed=true",
                ],
            }
        )
    if PACKET_RUNTIME_PATCH_MANIFEST in missing:
        work.append(
            {
                "id": PACKET_RUNTIME_PATCH_MANIFEST,
                "next_action": (
                    "patch packet-local PR101 inflate/runtime parsing for "
                    f"{FRAME_CONDITIONAL_LATENT_WIRE_SCHEMA}"
                ),
                "acceptance": [
                    "parse_archive consumes q-bit side-info bytes",
                    "latent decoder consumes variable-width payload bytes",
                    "inflate path does not load scorer code",
                ],
            }
        )
    if RUNTIME_CONSUMPTION_PROOF in missing:
        work.append(
            {
                "id": RUNTIME_CONSUMPTION_PROOF,
                "next_action": (
                    "capture local runtime-consumption proof binding the "
                    "candidate archive to A5 side-info and latent payload SHAs"
                ),
                "acceptance": [
                    f"q_bits_sideinfo_sha256={a5_summary.get('q_bits_sideinfo_sha256')}",
                    f"latent_wire_payload_sha256={a5_summary.get('latent_wire_payload_sha256')}",
                    "ready_for_exact_eval_runtime=true",
                ],
            }
        )
    if PER_PAIR_SCORE_MARGINAL_MANIFEST in missing:
        work.append(
            {
                "id": PER_PAIR_SCORE_MARGINAL_MANIFEST,
                "next_action": (
                    "produce per-pair score-marginal evidence for the selected "
                    "A5 q-bit schedule before treating the byte proxy as useful"
                ),
                "acceptance": [
                    f"n_pairs={a5_summary.get('n_pairs')}",
                    "marginal_evidence_available=true",
                    "score_claim=false until exact CUDA auth eval",
                ],
            }
        )
    if STRICT_PRE_SUBMISSION_COMPLIANCE_JSON in missing:
        work.append(
            {
                "id": STRICT_PRE_SUBMISSION_COMPLIANCE_JSON,
                "next_action": "run strict pre-submission compliance on the exact packet",
                "acceptance": [
                    "strict compliance JSON has ok=true or passed=true",
                    "same archive/runtime bytes as the runtime proof",
                ],
            }
        )
    return work


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FrameConditionalPacketReadinessError(f"{path} must contain a JSON object")
    return payload


def _nested_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, Mapping) else {}


def _nested_bool(payload: Mapping[str, Any], key: str, subkey: str) -> bool | None:
    nested = _nested_mapping(payload, key)
    value = nested.get(subkey)
    return value if isinstance(value, bool) else None


def _first_mapping(payload: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _truthy_any(*payloads_and_keys: Any) -> bool:
    payloads = payloads_and_keys[:-2]
    keys = payloads_and_keys[-2:]
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        for key in keys:
            if payload.get(key) is True:
                return True
    return False


def _json_contains_string(value: Any, target: str) -> bool:
    if isinstance(value, str):
        return value == target or target in value
    if isinstance(value, Mapping):
        return any(_json_contains_string(item, target) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_json_contains_string(item, target) for item in value)
    return False


def _as_str(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(ch in "0123456789abcdefABCDEF" for ch in value)


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(float(value))


def _all_finite_numbers(values: Sequence[Any]) -> bool:
    return all(_is_finite_number(value) for value in values)


def _true_authority_flag_paths(value: Any, *, path: str = "$") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key) in FALSE_AUTHORITY_KEYS and child is True:
                paths.append(child_path)
            paths.extend(_true_authority_flag_paths(child, path=child_path))
        return paths
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            paths.extend(_true_authority_flag_paths(child, path=f"{path}[{index}]"))
    return paths


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


__all__ = [
    "ARTIFACT_ORDER",
    "CANDIDATE_ARCHIVE_MANIFEST",
    "PER_PAIR_SCORE_MARGINAL_MANIFEST",
    "PACKET_RUNTIME_PATCH_MANIFEST",
    "RUNTIME_CONSUMPTION_PROOF",
    "SCHEMA",
    "STRICT_PRE_SUBMISSION_COMPLIANCE_JSON",
    "FrameConditionalPacketReadinessError",
    "build_packet_readiness",
    "existing_artifact_input_paths",
]
