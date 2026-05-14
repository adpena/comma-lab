#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan the PR86 HPAC contract port onto the PR85 QMA9 mask stream.

This is a deterministic, build-only preflight. It reads already-captured PR86
anatomy and PR85 bundle profile artifacts, computes the gross mask-byte
opportunity, and emits fail-closed parity gates. It does not build archives,
run scorers, claim scores, touch dispatch state, or dispatch remote jobs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from tac.repo_io import json_text, read_json, sha256_bytes, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = "experiments/plan_pr86_hpac_pr85_contract_port.py"
SCHEMA = "pr86_hpac_pr85_contract_port_plan_v1"
CANDIDATE_SPEC_SCHEMA = "pr86_hpac_pr85_candidate_spec_gate_v1"
BLOCKER_ID = "pr86_hpac_pr85_mask_contract_port"
EVIDENCE_GRADE = "planning_only_from_static_artifacts"
EXPECTED_PR86_ARCHIVE_BYTES = 207_579
EXPECTED_PR86_ARCHIVE_SHA256 = (
    "e67b7c22240dbe33853c19d049b0044a5df16ce5f751ba8f1021cab8ceb03cef"
)
EXPECTED_PR86_MEMBER_BYTES = {
    "master.pt.gz": 31_144,
    "slave.pt.gz": 32_287,
    "hpac.pt.ppmd": 28_243,
    "tokens.bin": 113_900,
    "meta.pt": 1_499,
}
EXPECTED_PR86_TOKEN_WORDS = 28_475
EXPECTED_PR85_ARCHIVE_BYTES = 236_328
EXPECTED_PR85_ARCHIVE_SHA256 = (
    "eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e"
)
EXPECTED_PR85_MASK_BYTES = 159_011
EXPECTED_GROSS_MASK_OPPORTUNITY_BYTES = 15_369
PR85_T4_BEST_SCORE = 0.25806611029397786

DEFAULT_PR86_DIR = REPO_ROOT / "experiments/results/public_pr86_intake_20260504_codex"
DEFAULT_PR85_DIR = REPO_ROOT / "experiments/results/public_pr85_intake_20260503_codex"
DEFAULT_PR86_ANATOMY_JSON = DEFAULT_PR86_DIR / "pr86_hpac_token_anatomy_forensics.json"
LEGACY_PR86_PARITY_JSON = DEFAULT_PR86_DIR / "pr86_hpac_replay_parity_diagnostic.json"
FULL_PR86_DECODE_REENCODE_GATE_JSON = (
    DEFAULT_PR86_DIR / "pr86_hpac_full_decode_reencode_gate_20260504_codex.json"
)
DEFAULT_PR86_PARITY_JSON = (
    FULL_PR86_DECODE_REENCODE_GATE_JSON
    if FULL_PR86_DECODE_REENCODE_GATE_JSON.exists()
    else LEGACY_PR86_PARITY_JSON
)
DEFAULT_PR85_PROFILE_JSON = DEFAULT_PR85_DIR / "profile_pr85_bundle.json"
DEFAULT_PR85_ARCHIVE = DEFAULT_PR85_DIR / "archive.zip"
DEFAULT_PR85_TOKEN_SOURCE_JSON = (
    DEFAULT_PR85_DIR / "qma9_token_source/pr85_qma9_token_source_profile.json"
)
DEFAULT_PR85_HPAC_PARITY_JSON = DEFAULT_PR86_DIR / "pr86_hpac_pr85_qma9_parity_probe.json"
DEFAULT_PR86_PROBABILITY_MATRIX_JSON = (
    REPO_ROOT
    / "experiments/results/pr86_hpac_probability_contract_20260504_worker"
    / "pr86_hpac_probability_contract_variants.json"
)
DEFAULT_JSON_OUT = DEFAULT_PR86_DIR / "pr86_hpac_pr85_contract_port_plan.json"

LOCAL_GATE_ORDER = (
    "pr86_full_decode_reencode_token_parity",
    "pr85_baseline_token_extraction",
    "pr85_hpac_token_parity",
    "pr85_runtime_output_parity",
    "candidate_archive_byte_closure",
)
DISPATCH_GATE_ORDER = (
    "pr86_exact_score_evidence",
    "pr86_full_decode_reencode_token_parity",
    "pr85_baseline_token_extraction",
    "pr85_hpac_token_parity",
    "pr85_runtime_output_parity",
    "candidate_archive_byte_closure",
)


class ContractPortError(ValueError):
    """Raised when a contract-port planning input is missing or malformed."""


def _rel(path: Path | str | None) -> str | None:
    if path is None:
        return None
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise ContractPortError(f"required JSON input is missing: {_rel(path)}")
        return {}
    try:
        payload = read_json(path)
    except json.JSONDecodeError as exc:
        raise ContractPortError(f"invalid JSON input: {_rel(path)}") from exc
    if not isinstance(payload, dict):
        raise ContractPortError(f"JSON input must be an object: {_rel(path)}")
    return payload


def _stable_digest(payload: dict[str, Any]) -> str:
    stable = {
        key: value
        for key, value in payload.items()
        if key not in {"recorded_at_utc", "stable_plan_digest_sha256"}
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return sha256_bytes(encoded)


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _segment_rows(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    segments = profile.get("segments", {})
    if isinstance(segments, dict):
        return {
            str(name): row
            for name, row in segments.items()
            if isinstance(row, dict)
        }
    if isinstance(segments, list):
        return {
            str(row.get("name")): row
            for row in segments
            if isinstance(row, dict) and row.get("name")
        }
    return {}


def _member_rows(anatomy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    contract = anatomy.get("archive_member_contract", {})
    members = contract.get("members", []) if isinstance(contract, dict) else []
    return {
        str(row.get("name")): row
        for row in members
        if isinstance(row, dict) and row.get("name")
    }


def _payload_layer_rows(anatomy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    layers = anatomy.get("member_payload_layers", {})
    members = layers.get("members", []) if isinstance(layers, dict) else []
    return {
        str(row.get("name")): row
        for row in members
        if isinstance(row, dict) and row.get("name")
    }


def _required_gate(parity: dict[str, Any], gate_id: str) -> dict[str, Any]:
    gates = parity.get("required_gates_before_hpac_transfer_to_pr85", [])
    if not isinstance(gates, list):
        return {}
    for row in gates:
        if isinstance(row, dict) and row.get("gate") == gate_id:
            return row
    return {}


def _artifact_identity(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "status": "not_checked"}
    if not path.exists():
        return {"path": _rel(path), "exists": False, "status": "missing_optional"}
    return {
        "path": _rel(path),
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _score_archive_bytes(replay: dict[str, Any]) -> int | None:
    direct = _int_or_none(replay.get("archive_size_bytes"))
    if direct is not None:
        return direct
    identity = replay.get("archive_identity", {})
    if isinstance(identity, dict):
        return _int_or_none(identity.get("archive_size_bytes"))
    return None


def _score_gate_status(replay: dict[str, Any], *, pr86_archive_bytes: int | None) -> tuple[bool, str]:
    if replay.get("status") != "score_json_present":
        return False, "blocked_missing_pr86_exact_score_json"
    if _int_or_none(replay.get("n_samples")) != 600:
        return False, "blocked_pr86_exact_score_not_full_600_samples"
    score_bytes = _score_archive_bytes(replay)
    if score_bytes != pr86_archive_bytes:
        return False, "blocked_pr86_exact_score_archive_identity_mismatch"
    return True, "passed"


def _full_decode_passed(parity: dict[str, Any]) -> bool:
    probes = parity.get("decode_probes", [])
    if isinstance(probes, list):
        for row in probes:
            if (
                isinstance(row, dict)
                and row.get("status") == "passed"
                and row.get("full_archive_decode") is True
            ):
                return True
    conclusions = parity.get("conclusions", {})
    if isinstance(conclusions, dict):
        return conclusions.get("own_stream_decode_status") in {
            "full_archive_decodes",
            "full_archive_decode_passed",
            "full_decode_reencode_parity_passed",
        }
    return False


def _token_parity_gate_status(parity: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    byte_gate = _required_gate(parity, "byte_exact_reencode")
    byte_status = byte_gate.get("status")
    full_decode = _full_decode_passed(parity)
    if byte_status in {"passed", "byte_exact_passed"} and full_decode:
        return True, "passed", {
            "byte_exact_reencode_status": byte_status,
            "full_archive_decode": True,
        }
    if byte_status in {"passed", "byte_exact_passed"}:
        return False, "blocked_full_archive_decode_not_proven", {
            "byte_exact_reencode_status": byte_status,
            "full_archive_decode": False,
        }
    if not parity:
        return False, "not_run_missing_pr86_parity_artifact", {}
    return False, str(byte_status or "not_run"), {
        "byte_exact_reencode_status": byte_status or "not_run",
        "full_archive_decode": full_decode,
    }


def _pr85_transfer_parity_status(parity: dict[str, Any]) -> tuple[bool, str]:
    gate = _required_gate(parity, "pr85_transfer_parity")
    status = str(gate.get("status") or "not_run")
    if status in {"passed", "pr85_transfer_parity_passed"}:
        return True, "passed"
    return False, status


def _probability_matrix_summary(matrix: dict[str, Any]) -> dict[str, Any]:
    if not matrix:
        return {"status": "not_present"}
    results = matrix.get("variant_results", [])
    variants: list[dict[str, Any]] = []
    if isinstance(results, list):
        for row in results:
            if not isinstance(row, dict):
                continue
            variant = row.get("probability_variant")
            if isinstance(variant, dict):
                name = variant.get("name")
                source_contract = variant.get("source_contract")
            else:
                name = variant
                source_contract = None
            failure_context = row.get("failure_context")
            if not isinstance(failure_context, dict):
                failure_context = {}
            variants.append(
                {
                    "name": name,
                    "source_contract": source_contract,
                    "status": row.get("status"),
                    "failure_stage": row.get("failure_stage"),
                    "failure_reason": row.get("failure_reason"),
                    "failed_at": failure_context.get("failed_at"),
                    "decoded_symbol_count_before_failure": failure_context.get(
                        "decoded_symbol_count_before_failure"
                    ),
                    "byte_parity_achieved": row.get("byte_parity_achieved"),
                    "dispatch_unlocked": row.get("dispatch_unlocked"),
                }
            )
    return {
        "status": matrix.get("status"),
        "failure_reason": matrix.get("failure_reason"),
        "dispatch_unlocked": matrix.get("dispatch_unlocked"),
        "byte_parity_variants": matrix.get("byte_parity_variants", []),
        "source_contract_byte_parity_variants": matrix.get(
            "source_contract_byte_parity_variants", []
        ),
        "variants": variants,
    }


def _pr85_token_source_gate_status(profile: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    if not profile:
        return False, "not_run_missing_pr85_token_source_profile", {}
    token_source = profile.get("token_source")
    mask_identity = profile.get("mask_segment_identity")
    exactness = profile.get("exactness")
    if not isinstance(token_source, dict) or not isinstance(mask_identity, dict):
        return False, "failed_closed_malformed_pr85_token_source_profile", {}
    if not isinstance(exactness, dict):
        exactness = {}
    shape = token_source.get("shape")
    observed_range = token_source.get("observed_range")
    invalid = token_source.get("invalid_symbol_values")
    mask_sha = mask_identity.get("sha256")
    token_sha = token_source.get("sha256")
    evidence = {
        "profile_json_status": "present",
        "token_extracted": token_source.get("extracted"),
        "token_dtype": token_source.get("dtype"),
        "token_shape": shape,
        "token_bytes": token_source.get("bytes"),
        "token_sha256": token_sha,
        "observed_range": observed_range,
        "invalid_symbol_values": invalid,
        "mask_segment_bytes": mask_identity.get("bytes"),
        "mask_segment_sha256": mask_sha,
        "raw_tensor_exact": exactness.get("raw_tensor_exact"),
    }
    passed = (
        token_source.get("extracted") is True
        and token_source.get("dtype") == "uint8"
        and shape == [600, 512, 384]
        and token_source.get("bytes") == 600 * 512 * 384
        and isinstance(token_sha, str)
        and len(token_sha) == 64
        and isinstance(observed_range, dict)
        and observed_range.get("min") == 0
        and observed_range.get("max") == 4
        and invalid == []
        and mask_identity.get("bytes") == EXPECTED_PR85_MASK_BYTES
        and mask_sha == "4b9d93fedb37a9d6fd435054cc33e216d703818b3ac94f4616c89969a4e0d179"
        and exactness.get("raw_tensor_exact") is True
    )
    return passed, "passed_token_source_profiled" if passed else "failed_closed_pr85_token_source_profile", evidence


def _pr85_hpac_candidate_parity_status(
    probe: dict[str, Any],
    *,
    token_source_evidence: dict[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    baseline_token_sha = token_source_evidence.get("token_sha256")
    baseline_token_shape = token_source_evidence.get("token_shape")
    baseline_mask_sha = token_source_evidence.get("mask_segment_sha256")
    evidence: dict[str, Any] = {
        "baseline_token_sha256": baseline_token_sha,
        "baseline_token_shape": baseline_token_shape,
        "baseline_mask_sha256": baseline_mask_sha,
    }
    if not probe:
        return False, "not_run_missing_pr85_hpac_parity_probe", evidence

    candidate = probe.get("candidate")
    if not isinstance(candidate, dict):
        candidate = {}
    candidate_status = str(probe.get("status") or candidate.get("status") or "unknown")
    candidate_token_sha = (
        candidate.get("decoded_token_sha256")
        or probe.get("candidate_decoded_token_sha256")
        or probe.get("decoded_token_sha256")
    )
    candidate_shape = (
        candidate.get("decoded_token_shape")
        or probe.get("candidate_decoded_token_shape")
        or probe.get("decoded_token_shape")
    )
    candidate_mask_sha = (
        candidate.get("mask_segment_sha256")
        or probe.get("candidate_mask_segment_sha256")
        or probe.get("mask_segment_sha256")
    )
    no_op = candidate.get("noop")
    if no_op is None:
        no_op = probe.get("noop")
    byte_exact_token_parity = (
        candidate.get("byte_exact_token_parity")
        if "byte_exact_token_parity" in candidate
        else probe.get("byte_exact_token_parity")
    )
    runtime_output_parity = (
        candidate.get("runtime_output_parity")
        if "runtime_output_parity" in candidate
        else probe.get("runtime_output_parity")
    )
    archive_byte_closed = (
        candidate.get("archive_byte_closed")
        if "archive_byte_closed" in candidate
        else probe.get("archive_byte_closed")
    )
    replacement_kind = (
        candidate.get("mask_replacement_kind")
        or probe.get("mask_replacement_kind")
        or probe.get("candidate_kind")
    )
    candidate_paths = candidate.get("paths") or probe.get("candidate_paths") or []
    candidate_digests = candidate.get("digests") or probe.get("candidate_digests") or {}
    evidence.update(
        {
            "probe_status": candidate_status,
            "probe_failure_class": probe.get("failure_class"),
            "probe_observed_error": probe.get("observed_error"),
            "probe_pr85_decoded_sha256": probe.get("pr85_decoded_sha256"),
            "probe_pr85_decoded_shape": probe.get("pr85_decoded_shape"),
            "candidate_mask_replacement_kind": replacement_kind,
            "candidate_decoded_token_sha256": candidate_token_sha,
            "candidate_decoded_token_shape": candidate_shape,
            "candidate_mask_segment_sha256": candidate_mask_sha,
            "candidate_paths": candidate_paths,
            "candidate_digests": candidate_digests,
            "byte_exact_token_parity": byte_exact_token_parity,
            "runtime_output_parity": runtime_output_parity,
            "archive_byte_closed": archive_byte_closed,
            "noop": no_op,
            "score_claim": probe.get("score_claim"),
            "dispatch_performed": probe.get("dispatch_performed"),
        }
    )

    status_passed = candidate_status in {
        "passed",
        "pr85_transfer_parity_passed",
        "byte_exact_token_parity_passed",
    }
    passed = (
        status_passed
        and probe.get("score_claim") is False
        and probe.get("dispatch_performed") is False
        and replacement_kind == "hpac_pr85_mask_replacement"
        and byte_exact_token_parity is True
        and runtime_output_parity is True
        and archive_byte_closed is True
        and no_op is False
        and isinstance(candidate_mask_sha, str)
        and candidate_mask_sha != baseline_mask_sha
        and candidate_token_sha == baseline_token_sha
        and candidate_shape == baseline_token_shape
    )
    if passed:
        return True, "passed_pr85_hpac_byte_closed_non_noop_parity", evidence
    if candidate_status.startswith("blocked") or candidate_status.startswith("failed"):
        return False, candidate_status, evidence
    if status_passed:
        return False, "failed_closed_pr85_hpac_candidate_missing_non_noop_parity_proof", evidence
    return False, "not_run_pr85_hpac_candidate_parity", evidence


def _gate(
    *,
    gate_id: str,
    phase: str,
    status: str,
    passed: bool,
    requirement: str,
    next_action: str,
    dispatch_blocker: bool = True,
    local_build_gate: bool = False,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": gate_id,
        "phase": phase,
        "required": True,
        "status": status,
        "passed": bool(passed),
        "dispatch_blocker": bool(dispatch_blocker),
        "local_build_gate": bool(local_build_gate),
        "requirement": requirement,
        "next_action": next_action,
        "evidence": evidence or {},
    }


def build_plan(
    *,
    pr86_anatomy_json: Path = DEFAULT_PR86_ANATOMY_JSON,
    pr85_profile_json: Path = DEFAULT_PR85_PROFILE_JSON,
    pr86_parity_json: Path | None = DEFAULT_PR86_PARITY_JSON,
    pr85_token_source_json: Path | None = DEFAULT_PR85_TOKEN_SOURCE_JSON,
    pr85_hpac_parity_json: Path | None = DEFAULT_PR85_HPAC_PARITY_JSON,
    pr86_probability_matrix_json: Path | None = DEFAULT_PR86_PROBABILITY_MATRIX_JSON,
    pr85_archive: Path | None = DEFAULT_PR85_ARCHIVE,
    request_dispatchable: bool = False,
    request_score_claim: bool = False,
) -> dict[str, Any]:
    anatomy = _read_json(pr86_anatomy_json)
    pr85_profile = _read_json(pr85_profile_json)
    parity = _read_json(pr86_parity_json, required=False) if pr86_parity_json else {}
    pr85_token_source = (
        _read_json(pr85_token_source_json, required=False) if pr85_token_source_json else {}
    )
    pr85_hpac_parity = (
        _read_json(pr85_hpac_parity_json, required=False) if pr85_hpac_parity_json else {}
    )
    pr86_probability_matrix = (
        _read_json(pr86_probability_matrix_json, required=False)
        if pr86_probability_matrix_json
        else {}
    )

    source = anatomy.get("source_archive", {})
    if not isinstance(source, dict):
        raise ContractPortError("PR86 anatomy JSON is missing source_archive object")
    replay = anatomy.get("current_exact_replay_status", {})
    if not isinstance(replay, dict):
        replay = {}
    token_contract = anatomy.get("token_hpac_decode_contract", {})
    if not isinstance(token_contract, dict):
        token_contract = {}
    members = _member_rows(anatomy)
    layers = _payload_layer_rows(anatomy)
    pr85_segments = _segment_rows(pr85_profile)

    pr86_archive_bytes = _int_or_none(source.get("bytes"))
    pr86_archive_sha = source.get("sha256")
    member_bytes = {
        name: _int_or_none(members.get(name, {}).get("file_size"))
        for name in EXPECTED_PR86_MEMBER_BYTES
    }
    hpac_stack_bytes = (
        member_bytes["hpac.pt.ppmd"]
        and member_bytes["tokens.bin"]
        and member_bytes["meta.pt"]
        and (
            member_bytes["hpac.pt.ppmd"]
            + member_bytes["tokens.bin"]
            + member_bytes["meta.pt"]
        )
    )
    token_layer = layers.get("tokens.bin", {})
    token_words = _int_or_none(token_layer.get("uint32_word_count"))
    if token_words is None and member_bytes["tokens.bin"] is not None:
        token_words = member_bytes["tokens.bin"] // 4
    token_aligned = token_layer.get("uint32_aligned")
    if token_aligned is None and member_bytes["tokens.bin"] is not None:
        token_aligned = member_bytes["tokens.bin"] % 4 == 0

    pr85_archive_info = pr85_profile.get("archive", {})
    if not isinstance(pr85_archive_info, dict):
        pr85_archive_info = {}
    pr85_archive_bytes = _int_or_none(pr85_archive_info.get("archive_size_bytes"))
    pr85_archive_sha = pr85_archive_info.get("archive_sha256")
    pr85_mask_bytes = _int_or_none(pr85_segments.get("mask", {}).get("bytes"))
    gross_opportunity = (
        pr85_mask_bytes - hpac_stack_bytes
        if isinstance(pr85_mask_bytes, int) and isinstance(hpac_stack_bytes, int)
        else None
    )
    pr85_archive_identity = _artifact_identity(pr85_archive)
    pr85_archive_file_matches_profile = (
        True
        if not pr85_archive_identity.get("exists")
        else (
            pr85_archive_identity.get("bytes") == pr85_archive_bytes
            and pr85_archive_identity.get("sha256") == pr85_archive_sha
        )
    )

    expected_member_mismatches = {
        name: {"expected": expected, "actual": member_bytes.get(name)}
        for name, expected in EXPECTED_PR86_MEMBER_BYTES.items()
        if member_bytes.get(name) != expected
    }
    pr86_archive_identity_passed = (
        pr86_archive_bytes == EXPECTED_PR86_ARCHIVE_BYTES
        and pr86_archive_sha == EXPECTED_PR86_ARCHIVE_SHA256
        and not expected_member_mismatches
    )
    pr86_member_contract = anatomy.get("archive_member_contract", {})
    member_contract_passed = bool(
        isinstance(pr86_member_contract, dict)
        and pr86_member_contract.get("promotable_member_contract") is True
    )
    token_contract_passed = (
        token_contract.get("submitted_archive_token_encoding") == "raw_tokens"
        and token_aligned is True
        and token_words == EXPECTED_PR86_TOKEN_WORDS
    )
    exact_score_passed, exact_score_status = _score_gate_status(
        replay, pr86_archive_bytes=pr86_archive_bytes
    )
    token_parity_passed, token_parity_status, token_parity_evidence = _token_parity_gate_status(
        parity
    )
    pr85_token_source_passed, pr85_token_source_status, pr85_token_source_evidence = (
        _pr85_token_source_gate_status(pr85_token_source)
    )
    pr85_transfer_gate_passed, pr85_transfer_gate_status = _pr85_transfer_parity_status(parity)
    pr85_candidate_passed, pr85_candidate_status, pr85_candidate_evidence = (
        _pr85_hpac_candidate_parity_status(
            pr85_hpac_parity,
            token_source_evidence=pr85_token_source_evidence,
        )
    )
    pr85_transfer_passed = pr85_transfer_gate_passed and pr85_candidate_passed
    pr85_transfer_status = (
        "passed"
        if pr85_transfer_passed
        else (
            pr85_candidate_status
            if pr85_candidate_status != "not_run_missing_pr85_hpac_parity_probe"
            else pr85_transfer_gate_status
        )
    )
    probability_matrix_summary = _probability_matrix_summary(pr86_probability_matrix)
    pr85_profile_passed = (
        pr85_archive_bytes == EXPECTED_PR85_ARCHIVE_BYTES
        and pr85_archive_sha == EXPECTED_PR85_ARCHIVE_SHA256
        and pr85_mask_bytes == EXPECTED_PR85_MASK_BYTES
        and pr85_archive_file_matches_profile
    )
    gross_math_passed = gross_opportunity == EXPECTED_GROSS_MASK_OPPORTUNITY_BYTES

    gates = [
        _gate(
            gate_id="pr86_archive_identity",
            phase="source_pr86",
            status="passed" if pr86_archive_identity_passed else "failed_closed_identity_mismatch",
            passed=pr86_archive_identity_passed,
            dispatch_blocker=True,
            requirement="PR86 anatomy must match the known archive SHA-256, bytes, and five member byte sizes.",
            next_action="Regenerate the PR86 anatomy artifact from the exact archive before planning a port.",
            evidence={
                "archive_bytes": pr86_archive_bytes,
                "archive_sha256": pr86_archive_sha,
                "expected_member_mismatches": expected_member_mismatches,
            },
        ),
        _gate(
            gate_id="pr86_member_contract",
            phase="source_pr86",
            status="passed" if member_contract_passed else "failed_closed_member_contract",
            passed=member_contract_passed,
            dispatch_blocker=True,
            requirement="PR86 anatomy must show exactly the expected members with no sidecars or unsafe names.",
            next_action="Fix the PR86 archive custody/member anatomy before using HPAC bytes as a design prior.",
            evidence={
                "promotable_member_contract": (
                    pr86_member_contract.get("promotable_member_contract")
                    if isinstance(pr86_member_contract, dict)
                    else None
                ),
            },
        ),
        _gate(
            gate_id="pr86_raw_uint32_token_stream",
            phase="source_pr86",
            status="passed" if token_contract_passed else "failed_closed_token_contract",
            passed=token_contract_passed,
            dispatch_blocker=True,
            requirement="Submitted PR86 tokens.bin must be raw-token uint32 queue words, not residual-token or ambiguous semantics.",
            next_action="Resolve token semantics against PR86 inflate/training code before port planning.",
            evidence={
                "submitted_archive_token_encoding": token_contract.get(
                    "submitted_archive_token_encoding"
                ),
                "queue_word_dtype": token_layer.get("queue_word_dtype", "uint32"),
                "uint32_aligned": token_aligned,
                "uint32_word_count": token_words,
                "tokens_bin_bytes": member_bytes.get("tokens.bin"),
            },
        ),
        _gate(
            gate_id="pr85_bundle_profile",
            phase="target_pr85",
            status="passed" if pr85_profile_passed else "failed_closed_pr85_profile_mismatch",
            passed=pr85_profile_passed,
            dispatch_blocker=True,
            requirement="PR85 profile must match the known T4-best archive and QMA9 mask segment byte count.",
            next_action="Refresh the PR85 bundle profile from the exact PR85 archive before computing byte economics.",
            evidence={
                "archive_bytes": pr85_archive_bytes,
                "archive_sha256": pr85_archive_sha,
                "mask_segment_bytes": pr85_mask_bytes,
                "pr85_archive_file_identity": pr85_archive_identity,
            },
        ),
        _gate(
            gate_id="gross_mask_byte_math",
            phase="byte_economics",
            status="passed" if gross_math_passed else "failed_closed_gross_math_mismatch",
            passed=gross_math_passed,
            dispatch_blocker=True,
            requirement="Gross PR85 mask opportunity must be recomputed from profile bytes before any port decision.",
            next_action="Recompute from source artifacts; do not carry forward stale byte arithmetic.",
            evidence={
                "pr85_mask_segment_bytes": pr85_mask_bytes,
                "pr86_hpac_tokens_meta_bytes": hpac_stack_bytes,
                "gross_mask_byte_opportunity": gross_opportunity,
            },
        ),
        _gate(
            gate_id="pr86_exact_score_evidence",
            phase="exact_evidence",
            status=exact_score_status,
            passed=exact_score_passed,
            dispatch_blocker=True,
            requirement="A PR86 exact CUDA contest_auth_eval.json with 600 samples and matching archive identity must exist before PR86 can justify a dispatchable contract port.",
            next_action="Classify the current PR86 replay blocker; do not claim score from external PR text or partial Lightning artifacts.",
            evidence={
                "current_replay_status": replay.get("status"),
                "evidence_grade": replay.get("evidence_grade"),
                "score_recomputed_from_components": replay.get("score_recomputed_from_components"),
                "n_samples": replay.get("n_samples"),
                "archive_size_bytes": _score_archive_bytes(replay),
                "score_delta_vs_pr85": replay.get("score_delta_vs_pr85"),
            },
        ),
        _gate(
            gate_id="pr86_full_decode_reencode_token_parity",
            phase="local_parity",
            status=token_parity_status,
            passed=token_parity_passed,
            local_build_gate=True,
            dispatch_blocker=True,
            requirement="PR86 tokens.bin must fully decode, then byte-exactly re-encode through the pinned HPAC/constriction contract.",
            next_action="Run the local PR86 full-stream decode and decode->encode tokens.bin byte-parity probe.",
            evidence={
                **token_parity_evidence,
                "probability_variant_matrix": probability_matrix_summary,
            },
        ),
        _gate(
            gate_id="pr85_baseline_token_extraction",
            phase="local_parity",
            status=pr85_token_source_status,
            passed=pr85_token_source_passed,
            local_build_gate=True,
            dispatch_blocker=True,
            requirement="Decode PR85 QMA9 mask bytes into the exact raw token source that HPAC would consume.",
            next_action="Extract PR85 QMA9 raw tokens with shape/range/SHA proof before fitting HPAC.",
            evidence={
                "source_mask_magic": pr85_segments.get("mask", {}).get("magic_ascii"),
                "source_mask_sha256": pr85_segments.get("mask", {}).get("sha256"),
                **pr85_token_source_evidence,
            },
        ),
        _gate(
            gate_id="pr85_hpac_token_parity",
            phase="local_parity",
            status=pr85_transfer_status,
            passed=pr85_transfer_passed,
            local_build_gate=True,
            dispatch_blocker=True,
            requirement="A PR85 HPAC-coded candidate must decode to the same PR85 raw tokens before archive replacement is allowed.",
            next_action="After PR85 HPAC fitting, require decoded-token SHA and byte-closed runtime parity against the PR85 baseline.",
            evidence={
                "parity_artifact_gate": _required_gate(parity, "pr85_transfer_parity"),
                "candidate_parity_probe": pr85_candidate_evidence,
            },
        ),
        _gate(
            gate_id="pr85_runtime_output_parity",
            phase="local_parity",
            status="not_run",
            passed=False,
            local_build_gate=True,
            dispatch_blocker=True,
            requirement="The PR85 inflate/runtime output must be byte- or tensor-parity equivalent before any exact eval dispatch.",
            next_action="Run PR85 runtime output parity after token parity passes.",
        ),
        _gate(
            gate_id="candidate_archive_byte_closure",
            phase="candidate_archive",
            status="not_run",
            passed=False,
            local_build_gate=True,
            dispatch_blocker=True,
            requirement="A candidate archive must be deterministic, sidecar-free, manifest-complete, and byte-closed.",
            next_action="Build only after PR86 and PR85 token parity gates pass.",
        ),
    ]

    gate_by_id = {gate["id"]: gate for gate in gates}
    fail_closed_reasons = [
        {
            "gate": gate["id"],
            "status": gate["status"],
            "reason": gate["requirement"],
            "next_action": gate["next_action"],
        }
        for gate in gates
        if gate["dispatch_blocker"] and not gate["passed"]
    ]
    hard_blockers = [
        gate_by_id[gate_id]
        for gate_id in DISPATCH_GATE_ORDER
        if gate_id in gate_by_id and not gate_by_id[gate_id]["passed"]
    ]
    dispatchable = not hard_blockers
    dispatch_status = (
        "eligible_for_exact_eval_after_lane_claim"
        if dispatchable
        else (
            "refused_fail_closed"
            if request_dispatchable
            else "non_dispatchable_until_required_gates_pass"
        )
    )
    next_local = next(
        (gate_by_id[gate_id] for gate_id in LOCAL_GATE_ORDER if not gate_by_id[gate_id]["passed"]),
        None,
    )
    next_dispatch = next(
        (gate_by_id[gate_id] for gate_id in DISPATCH_GATE_ORDER if not gate_by_id[gate_id]["passed"]),
        None,
    )
    local_build_blockers = [
        gate_by_id[gate_id]
        for gate_id in LOCAL_GATE_ORDER
        if gate_id in gate_by_id and not gate_by_id[gate_id]["passed"]
    ]
    candidate_paths = pr85_candidate_evidence.get("candidate_paths") or []
    candidate_digests = pr85_candidate_evidence.get("candidate_digests") or {}
    candidate_spec = {
        "schema": CANDIDATE_SPEC_SCHEMA,
        "status": (
            "ready_for_byte_closed_candidate_build"
            if not local_build_blockers
            else "not_emitted_fail_closed"
        ),
        "candidate_paths": candidate_paths if not local_build_blockers else [],
        "candidate_digests": candidate_digests if not local_build_blockers else {},
        "score_claim": False,
        "dispatch_performed": False,
        "non_noop_guard": {
            "required": True,
            "baseline_mask_sha256": pr85_token_source_evidence.get("mask_segment_sha256"),
            "candidate_mask_sha256": pr85_candidate_evidence.get("candidate_mask_segment_sha256"),
            "candidate_mask_must_differ_from_baseline": True,
            "noop": pr85_candidate_evidence.get("noop"),
            "passed": pr85_candidate_evidence.get("noop") is False
            and pr85_candidate_evidence.get("candidate_mask_segment_sha256")
            != pr85_token_source_evidence.get("mask_segment_sha256"),
        },
        "token_parity_guard": {
            "required": True,
            "baseline_token_sha256": pr85_token_source_evidence.get("token_sha256"),
            "candidate_decoded_token_sha256": pr85_candidate_evidence.get(
                "candidate_decoded_token_sha256"
            ),
            "byte_exact_token_parity": pr85_candidate_evidence.get("byte_exact_token_parity"),
            "passed": pr85_candidate_passed,
        },
        "runtime_output_parity_required": True,
        "archive_byte_closure_required": True,
        "blocked_by": [
            {
                "gate": gate["id"],
                "status": gate["status"],
                "next_action": gate["next_action"],
            }
            for gate in local_build_blockers
        ],
    }
    blocker = {
        "id": BLOCKER_ID,
        "status": "removed" if not local_build_blockers else "blocked_fail_closed",
        "primary_local_blocker": (
            {
                "gate": local_build_blockers[0]["id"],
                "status": local_build_blockers[0]["status"],
                "next_action": local_build_blockers[0]["next_action"],
            }
            if local_build_blockers
            else None
        ),
        "removal_criteria": [
            "PR86 full submitted tokens.bin decode succeeds under a named probability contract.",
            "PR86 decode->reencode emits byte-identical tokens.bin.",
            "PR85 HPAC candidate decodes to the PR85 baseline token SHA and shape.",
            "Candidate mask replacement is byte-closed, runtime-output parity checked, and non-no-op.",
        ],
        "precise_failure": {
            "pr86_probability_matrix": probability_matrix_summary,
            "pr85_hpac_parity_probe": pr85_candidate_evidence,
        },
    }

    plan: dict[str, Any] = {
        "schema_version": 1,
        "schema": SCHEMA,
        "tool": TOOL,
        "blocker": blocker,
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "score_claim": False,
        "score_claim_refusal": {
            "status": "refused",
            "requested_score_claim": bool(request_score_claim),
            "reasons": [
                "This planner is build-only and never emits a score claim.",
                "External PR text and partial replay artifacts are carried as context only.",
                "PR86 exact score evidence is a required gate, not a score claimed by this plan.",
            ],
        },
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "planning_only": True,
        "build_only": True,
        "evidence_grade": EVIDENCE_GRADE,
        "dispatchable": dispatchable,
        "ready_for_exact_eval_dispatch": dispatchable,
        "dispatch_gate": (
            "eligible_for_cuda_auth_eval_after_lane_claim"
            if dispatchable
            else "blocked/fail_closed_no_remote_dispatch"
        ),
        "source_artifacts": {
            "pr86_anatomy_json": _rel(pr86_anatomy_json),
            "pr86_parity_json": _rel(pr86_parity_json),
            "pr86_probability_matrix_json": _rel(pr86_probability_matrix_json),
            "pr85_profile_json": _rel(pr85_profile_json),
            "pr85_token_source_json": _rel(pr85_token_source_json),
            "pr85_hpac_parity_json": _rel(pr85_hpac_parity_json),
            "pr85_archive": _rel(pr85_archive),
        },
        "pr86_source_archive": {
            "bytes": pr86_archive_bytes,
            "sha256": pr86_archive_sha,
            "expected_bytes": EXPECTED_PR86_ARCHIVE_BYTES,
            "expected_sha256": EXPECTED_PR86_ARCHIVE_SHA256,
            "member_bytes": member_bytes,
            "current_exact_replay_status": replay.get("status", "unknown"),
            "score_claim_from_this_plan": False,
        },
        "pr85_target_reference": {
            "archive_bytes": pr85_archive_bytes,
            "archive_sha256": pr85_archive_sha,
            "expected_archive_bytes": EXPECTED_PR85_ARCHIVE_BYTES,
            "expected_archive_sha256": EXPECTED_PR85_ARCHIVE_SHA256,
            "t4_best_score_reference": PR85_T4_BEST_SCORE,
            "score_claim_from_this_plan": False,
            "mask_segment": {
                "bytes": pr85_mask_bytes,
                "sha256": pr85_segments.get("mask", {}).get("sha256"),
                "magic_ascii": pr85_segments.get("mask", {}).get("magic_ascii"),
            },
        },
        "token_stream_contract": {
            "drop_in_status": "not_drop_in",
            "submitted_archive_token_encoding": token_contract.get(
                "submitted_archive_token_encoding"
            ),
            "training_objective": token_contract.get("training_objective"),
            "stream_dtype": "uint32",
            "uint32_word_count": token_words,
            "tokens_bin_bytes": member_bytes.get("tokens.bin"),
            "raw_uint32_token_stream": token_contract_passed,
            "why_not_drop_in": (
                "PR86 tokens.bin is raw uint32 queue-coded HPAC tokens tied to "
                "PR86 HPAC probabilities. PR85 uses a QMA9 mask segment inside "
                "a different bundle/runtime contract."
            ),
        },
        "gross_byte_math": {
            "pr85_archive_bytes": pr85_archive_bytes,
            "pr85_mask_segment_bytes": pr85_mask_bytes,
            "pr85_non_mask_archive_bytes_estimate": (
                pr85_archive_bytes - pr85_mask_bytes
                if isinstance(pr85_archive_bytes, int) and isinstance(pr85_mask_bytes, int)
                else None
            ),
            "pr86_archive_bytes": pr86_archive_bytes,
            "pr86_hpac_model_bytes": member_bytes.get("hpac.pt.ppmd"),
            "pr86_tokens_bytes": member_bytes.get("tokens.bin"),
            "pr86_meta_bytes": member_bytes.get("meta.pt"),
            "pr86_hpac_tokens_meta_bytes": hpac_stack_bytes,
            "gross_mask_byte_opportunity": gross_opportunity,
            "gross_saved_bytes_if_same_contract": gross_opportunity,
            "score_claim": False,
            "interpretation": (
                "Gross byte opportunity only. It is not a score delta and is "
                "non-dispatchable until the parity gates pass."
            ),
        },
        "required_parity_gates": gates,
        "candidate_spec": candidate_spec,
        "fail_closed": bool(fail_closed_reasons),
        "fail_closed_reasons": fail_closed_reasons,
        "dispatchability": {
            "requested_dispatchable": bool(request_dispatchable),
            "status": dispatch_status,
            "dispatchable": dispatchable,
            "refusal_reasons": [
                {
                    "gate": gate["id"],
                    "status": gate["status"],
                    "reason": gate["requirement"],
                }
                for gate in hard_blockers
            ],
        },
        "next_gate": (
            {
                "id": next_local["id"],
                "status": next_local["status"],
                "kind": "local_build_parity",
                "next_action": next_local["next_action"],
            }
            if next_local
            else None
        ),
        "next_dispatch_gate": (
            {
                "id": next_dispatch["id"],
                "status": next_dispatch["status"],
                "next_action": next_dispatch["next_action"],
            }
            if next_dispatch
            else None
        ),
    }
    plan["stable_plan_digest_sha256"] = _stable_digest(plan)
    return plan


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr86-anatomy-json", type=Path, default=DEFAULT_PR86_ANATOMY_JSON)
    parser.add_argument("--pr86-parity-json", type=Path, default=DEFAULT_PR86_PARITY_JSON)
    parser.add_argument(
        "--pr86-probability-matrix-json",
        type=Path,
        default=DEFAULT_PR86_PROBABILITY_MATRIX_JSON,
    )
    parser.add_argument("--pr85-profile-json", type=Path, default=DEFAULT_PR85_PROFILE_JSON)
    parser.add_argument("--pr85-token-source-json", type=Path, default=DEFAULT_PR85_TOKEN_SOURCE_JSON)
    parser.add_argument("--pr85-hpac-parity-json", type=Path, default=DEFAULT_PR85_HPAC_PARITY_JSON)
    parser.add_argument("--pr85-archive", type=Path, default=DEFAULT_PR85_ARCHIVE)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument(
        "--request-dispatchable",
        action="store_true",
        help="Ask the planner to evaluate dispatchability; it still fails closed until gates pass.",
    )
    parser.add_argument(
        "--request-score-claim",
        action="store_true",
        help="Ask the planner to evaluate a score-claim request; it always refuses.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    plan = build_plan(
        pr86_anatomy_json=args.pr86_anatomy_json.resolve(),
        pr86_parity_json=args.pr86_parity_json.resolve() if args.pr86_parity_json else None,
        pr86_probability_matrix_json=(
            args.pr86_probability_matrix_json.resolve()
            if args.pr86_probability_matrix_json
            else None
        ),
        pr85_profile_json=args.pr85_profile_json.resolve(),
        pr85_token_source_json=(
            args.pr85_token_source_json.resolve() if args.pr85_token_source_json else None
        ),
        pr85_hpac_parity_json=(
            args.pr85_hpac_parity_json.resolve() if args.pr85_hpac_parity_json else None
        ),
        pr85_archive=args.pr85_archive.resolve() if args.pr85_archive else None,
        request_dispatchable=args.request_dispatchable,
        request_score_claim=args.request_score_claim,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json_text(plan), encoding="utf-8")
    print(json_text(plan), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
