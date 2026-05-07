"""Readiness checks for byte-closed categorical compression candidates."""

from __future__ import annotations

import json
import struct
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from tac.categorical_candidate_plan import audit_categorical_charged_label_plan
from tac.categorical_compression_contract import build_categorical_compression_contract
from tac.categorical_label_prior_payload_manifest import (
    LABEL_PRIOR_PAYLOAD_MANIFEST_CONTRACT,
    LABEL_PRIOR_PAYLOAD_MANIFEST_KIND,
    LABEL_PRIOR_PAYLOAD_MANIFEST_MEMBER,
    LABEL_PRIOR_PAYLOAD_MANIFEST_ROLE,
    RUNTIME_LABEL_CONTRACT,
    canonical_categorical_label_prior_class_rows,
)
from tac.categorical_openpilot_mask_prior_contract import (
    audit_categorical_openpilot_mask_priors,
)
from tac.repo_io import json_text, repo_relative, sha256_bytes, sha256_file
from tac.semantic_label_contract import CONTEST_SEGNET_CLASS_NAME_TUPLE, SELFCOMP_CLASS_TO_GRAY

SCHEMA_VERSION = 1
CANDIDATE_MANIFEST_CONTRACT = "categorical_byte_closed_candidate_manifest_v1"
ARCHIVE_MEMBER_MANIFEST_CONTRACT = "categorical_archive_member_manifest_v1"
RUNTIME_LOADER_PARITY_CONTRACT = "categorical_runtime_loader_parity_v1"
DECODE_REENCODE_PARITY_CONTRACT = "categorical_decode_reencode_parity_v1"
DECODE_REENCODE_INDEPENDENT_PROOF_KIND = "categorical_decode_reencode_independent_proof"
RUNTIME_EXECUTION_PROOF_KIND = "categorical_runtime_execution_independent_proof"
EXACT_EVAL_DISPATCH_REQUIREMENTS_CONTRACT = "categorical_exact_eval_dispatch_requirements_v1"
HPM1_STRUCTURAL_DECODE_INVENTORY_CONTRACT = "hpm1_payload_structural_decode_inventory_v1"
HPM1_SEMANTIC_PARITY_FAIL_CLOSED_CONTRACT = "hpm1_semantic_parity_fail_closed_v1"
CANDIDATE_MANIFEST_KINDS = (
    "categorical_candidate_manifest",
    "categorical_candidate_fixture_manifest",
    "categorical_qma9_clade_spade_openpilot_candidate_manifest",
)
ROW_COLLECTION_FIELDS = ("candidate_rows", "evidence_rows", "source_rows")
PROXY_ROW_KEYS = (
    "proxy",
    "proxy_score",
    "proxy_pose",
    "proxy_seg",
    "distortion_proxy",
)
PROXY_ROW_STRING_FIELDS = (
    "evidence_grade",
    "evidence_semantics",
    "ranking_basis",
    "selection_basis",
    "source",
    "status",
    "tag",
)
SIDECAR_ROW_KEYS = (
    "sidecar",
    "sidecars_allowed",
    "external_sidecar",
    "sidecar_path",
)
REQUIRED_CONTROL_NAMES = (
    "decode_reencode_identity_control",
    "label_permutation_fail_closed_control",
    "charged_member_presence_control",
    "runtime_consumes_conditioning_control",
)
REQUIRED_MEMBER_ROLES = ("categorical_payload", "decoder_or_runtime_consumer")
CONTEST_ARCHIVE_CONTRACT = "contest_archive_zip"
CONTEST_INFLATE_MEMBER = "inflate.sh"
EXACT_EVAL_ENTRYPOINT = "experiments/contest_auth_eval.py"
EXACT_EVAL_PATH = "archive.zip -> inflate.sh -> upstream/evaluate.py"
DISPATCH_CLAIMS_PATH = ".omx/state/active_lane_dispatch_claims.md"
TERMINAL_DISPATCH_STATUS_PREFIXES = (
    "completed",
    "failed",
    "stopped",
    "refused_dispatch",
    "stale_superseded",
)
LOCAL_FILE_HEADER_SIGNATURE = 0x04034B50
CENTRAL_DIRECTORY_SIGNATURE = 0x02014B50
END_OF_CENTRAL_DIRECTORY_SIGNATURE = 0x06054B50
DETERMINISTIC_ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)
DETERMINISTIC_ZIP_FILE_MODE = 0o644
DETERMINISTIC_ZIP_INFLATE_MODE = 0o755
DETERMINISTIC_ZIP_CREATE_SYSTEM = 3
DETERMINISTIC_ZIP_ALLOWED_COMPRESS_TYPES = (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
ZIP_DATA_DESCRIPTOR_FLAG = 0x0008


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(char in "0123456789abcdef" for char in value.lower())


def _safe_member_name(name: Any) -> bool:
    if not isinstance(name, str) or not name:
        return False
    if "\x00" in name or "\\" in name:
        return False
    path = PurePosixPath(name)
    parts = path.parts
    return (
        not path.is_absolute()
        and ".." not in parts
        and all(part not in {"", ".", "__MACOSX"} for part in parts)
        and not any(part.startswith(".") for part in parts)
    )


def _archive_member_manifest_kind_valid(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("categorical_") and value.endswith("archive_member_manifest")


def _resolve_path(path_value: Any, *, repo_root: Path, manifest_dir: Path | None) -> Path | None:
    if not isinstance(path_value, str) or not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    if manifest_dir is not None and (manifest_dir / path).exists():
        return manifest_dir / path
    return repo_root / path


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _control_passed(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, dict):
        return value.get("passed") is True
    return False


def _artifact_proof_report(
    record: Any,
    *,
    repo_root: Path,
    manifest_dir: Path | None,
    required_kind: str,
    prefix: str,
) -> tuple[dict[str, Any], list[str], dict[str, Any] | None]:
    blockers: list[str] = []
    summary: dict[str, Any] = {
        "declared": isinstance(record, dict),
        "accepted": False,
        "path": "",
        "exists": False,
        "bytes": None,
        "sha256": "",
        "kind": "",
        "independent_proof": False,
        "producer_tool": "",
        "proof_scope": "",
        "blockers": [],
    }
    if not isinstance(record, dict):
        blockers.append(f"{prefix}_proof_artifact_missing")
        summary["blockers"] = blockers
        return summary, blockers, None

    path = _resolve_path(
        record.get("path"),
        repo_root=repo_root,
        manifest_dir=manifest_dir,
    )
    if path is None or not path.exists():
        blockers.append(f"{prefix}_proof_artifact_path_missing")
    elif not path.is_file():
        blockers.append(f"{prefix}_proof_artifact_path_not_file")
    else:
        raw = path.read_bytes()
        actual_sha = sha256_bytes(raw)
        summary.update(
            {
                "path": repo_relative(path, repo_root),
                "exists": True,
                "bytes": len(raw),
                "sha256": actual_sha,
            }
        )
        if record.get("bytes") != len(raw):
            blockers.append(f"{prefix}_proof_artifact_bytes_mismatch")
        if record.get("sha256") != actual_sha:
            blockers.append(f"{prefix}_proof_artifact_sha256_mismatch")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            blockers.append(f"{prefix}_proof_artifact_json_invalid")
            payload = None
        if isinstance(payload, dict):
            kind = payload.get("kind")
            producer_tool = payload.get("producer_tool", payload.get("tool", ""))
            proof_scope = payload.get("proof_scope", "")
            summary.update(
                {
                    "kind": kind if isinstance(kind, str) else "",
                    "independent_proof": payload.get("independent_proof") is True,
                    "producer_tool": producer_tool if isinstance(producer_tool, str) else "",
                    "proof_scope": proof_scope if isinstance(proof_scope, str) else "",
                }
            )
            if payload.get("schema_version") != SCHEMA_VERSION:
                blockers.append(f"{prefix}_proof_schema_version_missing_or_invalid")
            if kind != required_kind:
                blockers.append(f"{prefix}_proof_kind_missing_or_invalid")
            if payload.get("independent_proof") is not True:
                blockers.append(f"{prefix}_proof_independent_flag_missing")
            if payload.get("score_claim") is not False:
                blockers.append(f"{prefix}_proof_score_claim_must_be_false")
            if payload.get("dispatch_attempted") is not False:
                blockers.append(f"{prefix}_proof_dispatch_attempted_must_be_false")
            if (
                payload.get("manifest_declared") is True
                or payload.get("no_op_control") is True
                or payload.get("proof_source") in {"candidate_manifest", "no_op_control"}
            ):
                blockers.append(f"{prefix}_proof_must_not_be_manifest_declared_no_op")
        else:
            payload = None

    summary["accepted"] = not blockers
    summary["blockers"] = blockers
    return summary, blockers, payload


def _runtime_loader_parity_report(
    report: Any,
    *,
    runtime_path: Path | None,
    member_records: list[dict[str, Any]],
    archive_members: dict[str, bytes],
    repo_root: Path,
    manifest_dir: Path | None,
    candidate_archive_sha256: str,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    expected_runtime_path = repo_relative(runtime_path, repo_root) if runtime_path else ""
    runtime_exists = bool(runtime_path is not None and runtime_path.is_file())
    runtime_sha = sha256_file(runtime_path) if runtime_exists else ""
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": RUNTIME_LOADER_PARITY_CONTRACT,
        "declared": isinstance(report, dict),
        "accepted": False,
        "runtime_consumer_path": expected_runtime_path,
        "runtime_consumer_sha256": runtime_sha,
        "loader_member": "",
        "loader_member_sha256": "",
        "byte_identical_to_runtime_consumer": False,
        "sidecar_free": False,
        "fallback_used": None,
        "loaded_charged_members": [],
        "runtime_execution_proof": {
            "declared": False,
            "accepted": False,
            "blockers": ["runtime_execution_proof_artifact_missing"],
        },
        "blockers": [],
    }
    if not isinstance(report, dict):
        blockers.append("runtime_loader_parity_missing")
        summary["blockers"] = blockers
        return summary, blockers

    if report.get("schema_version") != SCHEMA_VERSION:
        blockers.append("runtime_loader_parity_schema_version_missing_or_invalid")
    if report.get("runtime_loader_parity_contract") != RUNTIME_LOADER_PARITY_CONTRACT:
        blockers.append("runtime_loader_parity_contract_missing_or_invalid")
    if report.get("passed") is not True:
        blockers.append("runtime_loader_parity_not_passed")
    if report.get("score_claim") is not False:
        blockers.append("runtime_loader_parity_score_claim_must_be_false")
    if report.get("dispatch_attempted") is not False:
        blockers.append("runtime_loader_parity_dispatch_attempted_must_be_false")

    loader_member = report.get("loader_member")
    loader_member_str = loader_member if isinstance(loader_member, str) else ""
    summary["loader_member"] = loader_member_str
    if not _safe_member_name(loader_member_str):
        blockers.append("runtime_loader_parity_loader_member_missing_or_unsafe")

    charged_by_name = {record["name"]: record for record in member_records if record.get("name")}
    charged_record = charged_by_name.get(loader_member_str)
    if loader_member_str and charged_record is None:
        blockers.append("runtime_loader_parity_loader_member_not_charged")
    elif charged_record is not None and charged_record.get("role") != "decoder_or_runtime_consumer":
        blockers.append("runtime_loader_parity_loader_role_not_decoder_or_runtime_consumer")

    loader_raw = archive_members.get(loader_member_str)
    loader_sha = sha256_bytes(loader_raw) if loader_raw is not None else ""
    summary["loader_member_sha256"] = loader_sha
    if loader_member_str and loader_raw is None:
        blockers.append("runtime_loader_parity_loader_member_missing_from_archive")
    if charged_record is not None and loader_raw is not None:
        if charged_record.get("bytes") != len(loader_raw):
            blockers.append("runtime_loader_parity_loader_member_bytes_mismatch")
        if charged_record.get("sha256") != loader_sha:
            blockers.append("runtime_loader_parity_loader_member_charged_sha256_mismatch")
    if report.get("loader_member_sha256") != loader_sha:
        blockers.append("runtime_loader_parity_loader_member_sha256_mismatch")

    if report.get("runtime_consumer_path") != expected_runtime_path:
        blockers.append("runtime_loader_parity_runtime_consumer_path_mismatch")
    if report.get("runtime_consumer_sha256") != runtime_sha:
        blockers.append("runtime_loader_parity_runtime_consumer_sha256_mismatch")
    if report.get("byte_identical_to_runtime_consumer") is not True:
        blockers.append("runtime_loader_parity_not_byte_identical")
    if runtime_sha and loader_sha and runtime_sha != loader_sha:
        blockers.append("runtime_loader_parity_source_loader_sha256_mismatch")

    sidecar_free = report.get("sidecar_free") is True
    fallback_used = report.get("fallback_used")
    summary["byte_identical_to_runtime_consumer"] = report.get("byte_identical_to_runtime_consumer") is True
    summary["sidecar_free"] = sidecar_free
    summary["fallback_used"] = fallback_used
    if not sidecar_free:
        blockers.append("runtime_loader_parity_sidecar_free_not_proven")
    if fallback_used is not False:
        blockers.append("runtime_loader_parity_fallback_used")

    loaded_members = report.get("loaded_charged_members")
    if not isinstance(loaded_members, list) or not loaded_members:
        blockers.append("runtime_loader_parity_loaded_charged_members_missing")
        loaded_member_names: list[str] = []
    else:
        loaded_member_names = []
        for item in loaded_members:
            if not _safe_member_name(item):
                blockers.append("runtime_loader_parity_loaded_charged_member_unsafe")
                continue
            loaded_member_names.append(item)
            if item not in charged_by_name:
                blockers.append(f"runtime_loader_parity_loaded_charged_member_not_declared:{item}")
    summary["loaded_charged_members"] = loaded_member_names

    proof_summary, proof_blockers, proof_payload = _artifact_proof_report(
        report.get("runtime_execution_proof", report.get("proof_artifact")),
        repo_root=repo_root,
        manifest_dir=manifest_dir,
        required_kind=RUNTIME_EXECUTION_PROOF_KIND,
        prefix="runtime_execution",
    )
    if proof_payload is not None:
        if proof_payload.get("candidate_archive_sha256") != candidate_archive_sha256:
            proof_blockers.append("runtime_execution_proof_candidate_archive_sha256_mismatch")
        if proof_payload.get("runtime_consumer_sha256") != runtime_sha:
            proof_blockers.append("runtime_execution_proof_runtime_consumer_sha256_mismatch")
        if proof_payload.get("loader_member") != loader_member_str:
            proof_blockers.append("runtime_execution_proof_loader_member_mismatch")
        if proof_payload.get("loader_member_sha256") != loader_sha:
            proof_blockers.append("runtime_execution_proof_loader_member_sha256_mismatch")
        if proof_payload.get("runtime_executed") is not True:
            proof_blockers.append("runtime_execution_proof_runtime_not_executed")
        if proof_payload.get("executed_archive_inflate") is not True:
            proof_blockers.append("runtime_execution_proof_inflate_not_executed")
        if proof_payload.get("sidecar_free") is not True:
            proof_blockers.append("runtime_execution_proof_sidecar_free_not_proven")
        if proof_payload.get("fallback_used") is not False:
            proof_blockers.append("runtime_execution_proof_fallback_used")
        consumed_members = proof_payload.get("consumed_charged_members")
        if not isinstance(consumed_members, list) or not consumed_members:
            proof_blockers.append("runtime_execution_proof_consumed_charged_members_missing")
            proof_summary["consumed_charged_members"] = []
        else:
            consumed_member_names = [item for item in consumed_members if isinstance(item, str)]
            proof_summary["consumed_charged_members"] = consumed_member_names
            if len(consumed_member_names) != len(consumed_members):
                proof_blockers.append("runtime_execution_proof_consumed_charged_members_invalid")
            for member in loaded_member_names:
                if member not in consumed_member_names:
                    proof_blockers.append(f"runtime_execution_proof_missing_loaded_member:{member}")
        output_sha = proof_payload.get(
            "runtime_output_sha256",
            proof_payload.get("decoded_output_sha256"),
        )
        proof_summary["runtime_output_sha256"] = output_sha if isinstance(output_sha, str) else ""
        if not _is_sha256(output_sha):
            proof_blockers.append("runtime_execution_proof_output_sha256_missing_or_invalid")
    proof_blockers = list(dict.fromkeys(proof_blockers))
    proof_summary["accepted"] = not proof_blockers
    proof_summary["blockers"] = proof_blockers
    summary["runtime_execution_proof"] = proof_summary
    blockers.extend(proof_blockers)

    accepted = not blockers
    summary["accepted"] = accepted
    summary["blockers"] = blockers
    return summary, blockers


def _decode_reencode_parity_report(
    report: Any,
    *,
    member_records: list[dict[str, Any]],
    archive_members: dict[str, bytes],
    repo_root: Path,
    manifest_dir: Path | None,
    candidate_archive_sha256: str,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": DECODE_REENCODE_PARITY_CONTRACT,
        "declared": isinstance(report, dict),
        "accepted": False,
        "payload_member": "",
        "payload_member_sha256": "",
        "full_decode_proven": False,
        "decoded_frame_count": None,
        "decoded_masks_sha256": "",
        "byte_exact_reencode": False,
        "reencoded_payload_sha256": "",
        "sidecar_free": False,
        "independent_proof": {
            "declared": False,
            "accepted": False,
            "blockers": ["decode_reencode_independent_proof_artifact_missing"],
        },
        "blockers": [],
    }
    if not isinstance(report, dict):
        blockers.append("decode_reencode_parity_missing")
        summary["blockers"] = blockers
        return summary, blockers

    if report.get("schema_version") != SCHEMA_VERSION:
        blockers.append("decode_reencode_schema_version_missing_or_invalid")
    if report.get("decode_reencode_parity_contract") != DECODE_REENCODE_PARITY_CONTRACT:
        blockers.append("decode_reencode_contract_missing_or_invalid")
    if report.get("passed") is not True:
        blockers.append("decode_reencode_parity_not_passed")
    if report.get("score_claim") is not False:
        blockers.append("decode_reencode_score_claim_must_be_false")
    if report.get("dispatch_attempted") is not False:
        blockers.append("decode_reencode_dispatch_attempted_must_be_false")

    payload_member = report.get("payload_member")
    payload_member_str = payload_member if isinstance(payload_member, str) else ""
    summary["payload_member"] = payload_member_str
    if not _safe_member_name(payload_member_str):
        blockers.append("decode_reencode_payload_member_missing_or_unsafe")

    charged_by_name = {record["name"]: record for record in member_records if record.get("name")}
    charged_record = charged_by_name.get(payload_member_str)
    if payload_member_str and charged_record is None:
        blockers.append("decode_reencode_payload_member_not_charged")
    elif charged_record is not None and charged_record.get("role") != "categorical_payload":
        blockers.append("decode_reencode_payload_member_role_not_categorical_payload")

    payload_raw = archive_members.get(payload_member_str)
    payload_sha = sha256_bytes(payload_raw) if payload_raw is not None else ""
    summary["payload_member_sha256"] = payload_sha
    if payload_member_str and payload_raw is None:
        blockers.append("decode_reencode_payload_member_missing_from_archive")
    if charged_record is not None and payload_raw is not None:
        if charged_record.get("bytes") != len(payload_raw):
            blockers.append("decode_reencode_payload_member_bytes_mismatch")
        if charged_record.get("sha256") != payload_sha:
            blockers.append("decode_reencode_payload_member_charged_sha256_mismatch")
    if report.get("payload_member_sha256") != payload_sha:
        blockers.append("decode_reencode_payload_member_sha256_mismatch")

    full_decode = report.get("full_decode")
    full_decode_proven = isinstance(full_decode, dict) and full_decode.get("passed") is True
    summary["full_decode_proven"] = full_decode_proven
    if not full_decode_proven:
        blockers.append("decode_reencode_full_decode_not_proven")
    else:
        frame_count = full_decode.get("frame_count")
        summary["decoded_frame_count"] = frame_count if isinstance(frame_count, int) else None
        if not isinstance(frame_count, int) or frame_count <= 0:
            blockers.append("decode_reencode_full_decode_frame_count_invalid")
        decoded_sha = full_decode.get("decoded_masks_sha256", full_decode.get("decoded_payload_sha256"))
        summary["decoded_masks_sha256"] = decoded_sha if isinstance(decoded_sha, str) else ""
        if not _is_sha256(decoded_sha):
            blockers.append("decode_reencode_full_decode_sha256_missing_or_invalid")

    reencode = report.get("byte_exact_reencode")
    reencode_proven = isinstance(reencode, dict) and reencode.get("passed") is True
    summary["byte_exact_reencode"] = reencode_proven
    if not reencode_proven:
        blockers.append("decode_reencode_byte_exact_reencode_not_proven")
    else:
        reencoded_sha = reencode.get("reencoded_payload_sha256", reencode.get("reencoded_member_sha256"))
        summary["reencoded_payload_sha256"] = reencoded_sha if isinstance(reencoded_sha, str) else ""
        if reencode.get("byte_exact") is not True:
            blockers.append("decode_reencode_byte_exact_flag_not_true")
        if reencoded_sha != payload_sha:
            blockers.append("decode_reencode_reencoded_payload_sha256_mismatch")

    sidecar_free = report.get("sidecar_free") is True
    summary["sidecar_free"] = sidecar_free
    if not sidecar_free:
        blockers.append("decode_reencode_sidecar_free_not_proven")

    proof_summary, proof_blockers, proof_payload = _artifact_proof_report(
        report.get("independent_proof_artifact", report.get("proof_artifact")),
        repo_root=repo_root,
        manifest_dir=manifest_dir,
        required_kind=DECODE_REENCODE_INDEPENDENT_PROOF_KIND,
        prefix="decode_reencode_independent",
    )
    if proof_payload is not None:
        if proof_payload.get("candidate_archive_sha256") != candidate_archive_sha256:
            proof_blockers.append("decode_reencode_independent_proof_candidate_archive_sha256_mismatch")
        if proof_payload.get("payload_member") != payload_member_str:
            proof_blockers.append("decode_reencode_independent_proof_payload_member_mismatch")
        if proof_payload.get("payload_member_sha256") != payload_sha:
            proof_blockers.append("decode_reencode_independent_proof_payload_member_sha256_mismatch")
        if proof_payload.get("proof_scope") != "full_decode_reencode":
            proof_blockers.append("decode_reencode_independent_proof_scope_invalid")
        proof_full_decode = proof_payload.get("full_decode")
        proof_full_ok = isinstance(proof_full_decode, dict) and proof_full_decode.get("passed") is True
        if not proof_full_ok:
            proof_blockers.append("decode_reencode_independent_proof_full_decode_not_proven")
        else:
            proof_frame_count = proof_full_decode.get("frame_count")
            if not isinstance(proof_frame_count, int) or proof_frame_count <= 0:
                proof_blockers.append("decode_reencode_independent_proof_full_decode_frame_count_invalid")
            proof_decoded_sha = proof_full_decode.get(
                "decoded_masks_sha256",
                proof_full_decode.get("decoded_payload_sha256"),
            )
            if not _is_sha256(proof_decoded_sha):
                proof_blockers.append("decode_reencode_independent_proof_full_decode_sha256_missing_or_invalid")
        proof_reencode = proof_payload.get("byte_exact_reencode")
        proof_reencode_ok = isinstance(proof_reencode, dict) and proof_reencode.get("passed") is True
        if not proof_reencode_ok:
            proof_blockers.append("decode_reencode_independent_proof_byte_exact_reencode_not_proven")
        else:
            proof_reencoded_sha = proof_reencode.get(
                "reencoded_payload_sha256",
                proof_reencode.get("reencoded_member_sha256"),
            )
            if proof_reencode.get("byte_exact") is not True:
                proof_blockers.append("decode_reencode_independent_proof_byte_exact_flag_not_true")
            if proof_reencoded_sha != payload_sha:
                proof_blockers.append("decode_reencode_independent_proof_reencoded_payload_sha256_mismatch")
        if proof_payload.get("sidecar_free") is not True:
            proof_blockers.append("decode_reencode_independent_proof_sidecar_free_not_proven")
    proof_blockers = list(dict.fromkeys(proof_blockers))
    proof_summary["accepted"] = not proof_blockers
    proof_summary["blockers"] = proof_blockers
    summary["independent_proof"] = proof_summary
    blockers.extend(proof_blockers)

    summary["accepted"] = not blockers
    summary["blockers"] = blockers
    return summary, blockers


def _label_prior_payload_manifest_report(
    record: Any,
    *,
    member_records: list[dict[str, Any]],
    archive_members: dict[str, bytes],
    candidate_source_archive_sha256: Any,
    candidate_conditioning_priors: Any,
    expected_class_order: list[str],
    expected_gray_codebook: list[int],
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": LABEL_PRIOR_PAYLOAD_MANIFEST_CONTRACT,
        "declared": isinstance(record, dict),
        "accepted": False,
        "member": "",
        "bytes": None,
        "sha256": "",
        "label_contract": "",
        "conditioning_prior_count": 0,
        "class_rows_match": False,
        "conditioning_prior_contract_matches_recomputed": False,
        "blockers": [],
    }
    if not isinstance(record, dict):
        blockers.append("label_prior_payload_manifest_record_missing")
        summary["blockers"] = blockers
        return summary, blockers

    member = record.get("member", record.get("name"))
    member_str = member if isinstance(member, str) else ""
    summary["member"] = member_str
    if not _safe_member_name(member_str):
        blockers.append("label_prior_payload_manifest_member_missing_or_unsafe")

    if record.get("contract") != LABEL_PRIOR_PAYLOAD_MANIFEST_CONTRACT:
        blockers.append("label_prior_payload_manifest_contract_record_missing_or_invalid")
    if not isinstance(record.get("bytes"), int) or record.get("bytes") <= 0:
        blockers.append("label_prior_payload_manifest_record_bytes_invalid")
    if not _is_sha256(record.get("sha256")):
        blockers.append("label_prior_payload_manifest_record_sha256_invalid")

    charged_by_name = {item["name"]: item for item in member_records if item.get("name")}
    charged_record = charged_by_name.get(member_str)
    if member_str and charged_record is None:
        blockers.append("label_prior_payload_manifest_member_not_charged")
    elif charged_record is not None:
        if charged_record.get("role") != LABEL_PRIOR_PAYLOAD_MANIFEST_ROLE:
            blockers.append("label_prior_payload_manifest_charged_role_invalid")
        if charged_record.get("bytes") != record.get("bytes"):
            blockers.append("label_prior_payload_manifest_charged_bytes_mismatch")
        if charged_record.get("sha256") != record.get("sha256"):
            blockers.append("label_prior_payload_manifest_charged_sha256_mismatch")

    raw = archive_members.get(member_str)
    if member_str and raw is None:
        blockers.append("label_prior_payload_manifest_member_missing_from_archive")
        summary["blockers"] = blockers
        return summary, blockers

    manifest_payload: dict[str, Any] | None = None
    if raw is not None:
        actual_sha = sha256_bytes(raw)
        summary["bytes"] = len(raw)
        summary["sha256"] = actual_sha
        if record.get("bytes") != len(raw):
            blockers.append("label_prior_payload_manifest_member_bytes_mismatch")
        if record.get("sha256") != actual_sha:
            blockers.append("label_prior_payload_manifest_member_sha256_mismatch")
        try:
            loaded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            blockers.append("label_prior_payload_manifest_json_invalid")
            loaded = None
        if isinstance(loaded, dict):
            manifest_payload = loaded
        else:
            blockers.append("label_prior_payload_manifest_not_object")

    if manifest_payload is not None:
        if manifest_payload.get("schema_version") != SCHEMA_VERSION:
            blockers.append("label_prior_payload_manifest_schema_version_missing_or_invalid")
        if manifest_payload.get("kind") != LABEL_PRIOR_PAYLOAD_MANIFEST_KIND:
            blockers.append("label_prior_payload_manifest_kind_missing_or_invalid")
        if manifest_payload.get("label_prior_payload_manifest_contract") != LABEL_PRIOR_PAYLOAD_MANIFEST_CONTRACT:
            blockers.append("label_prior_payload_manifest_contract_missing_or_invalid")
        if manifest_payload.get("score_claim") is not False:
            blockers.append("label_prior_payload_manifest_score_claim_must_be_false")
        if manifest_payload.get("dispatch_attempted") is not False:
            blockers.append("label_prior_payload_manifest_dispatch_attempted_must_be_false")
        if manifest_payload.get("ready_for_exact_eval_dispatch") is not False:
            blockers.append("label_prior_payload_manifest_ready_for_exact_eval_must_be_false")
        if manifest_payload.get("source_archive_sha256") != candidate_source_archive_sha256:
            blockers.append("label_prior_payload_manifest_source_archive_sha256_mismatch")

        label_contract = manifest_payload.get("label_contract")
        summary["label_contract"] = label_contract if isinstance(label_contract, str) else ""
        if label_contract != RUNTIME_LABEL_CONTRACT:
            blockers.append("label_prior_payload_manifest_label_contract_mismatch")
        if manifest_payload.get("semantic_class_order") != expected_class_order:
            blockers.append("label_prior_payload_manifest_semantic_class_order_mismatch")
        if manifest_payload.get("selfcomp_gray_codebook") != expected_gray_codebook:
            blockers.append("label_prior_payload_manifest_selfcomp_gray_codebook_mismatch")
        expected_class_rows = canonical_categorical_label_prior_class_rows()
        summary["class_rows_match"] = manifest_payload.get("class_rows") == expected_class_rows
        if not summary["class_rows_match"]:
            blockers.append("label_prior_payload_manifest_class_rows_mismatch")
        if manifest_payload.get("conditioning_priors") != candidate_conditioning_priors:
            blockers.append("label_prior_payload_manifest_conditioning_priors_mismatch")
        priors = manifest_payload.get("conditioning_priors")
        summary["conditioning_prior_count"] = len(priors) if isinstance(priors, list) else 0

        prior_contract = audit_categorical_openpilot_mask_priors(
            priors,
            charged_member_names=sorted(charged_by_name),
            charged_members=member_records,
        )
        summary["conditioning_prior_contract"] = prior_contract
        summary["conditioning_prior_contract_matches_recomputed"] = (
            manifest_payload.get("conditioning_prior_contract") == prior_contract
        )
        if not summary["conditioning_prior_contract_matches_recomputed"]:
            blockers.append("label_prior_payload_manifest_conditioning_prior_contract_mismatch")
        for blocker in prior_contract["dispatch_blockers"]:
            blockers.append(f"label_prior_payload_manifest_{blocker}")

        charged_member_links = manifest_payload.get("charged_member_links")
        if not isinstance(charged_member_links, list) or not charged_member_links:
            blockers.append("label_prior_payload_manifest_charged_member_links_missing")
        else:
            for index, link in enumerate(charged_member_links):
                if not isinstance(link, dict):
                    blockers.append(f"label_prior_payload_manifest_charged_member_link_{index}_not_object")
                    continue
                name = link.get("name")
                digest = link.get("sha256")
                if not _safe_member_name(name):
                    blockers.append(f"label_prior_payload_manifest_charged_member_link_{index}_name_invalid")
                    continue
                if name not in charged_by_name:
                    blockers.append(f"label_prior_payload_manifest_charged_member_link_not_charged:{name}")
                    continue
                if digest != charged_by_name[name].get("sha256"):
                    blockers.append(f"label_prior_payload_manifest_charged_member_link_sha256_mismatch:{name}")

        required_controls = manifest_payload.get("required_no_op_controls")
        if not isinstance(required_controls, list):
            blockers.append("label_prior_payload_manifest_required_controls_missing")
        else:
            for control_name in REQUIRED_CONTROL_NAMES:
                if control_name not in required_controls:
                    blockers.append(f"label_prior_payload_manifest_required_control_missing:{control_name}")

    blockers = list(dict.fromkeys(blockers))
    summary["accepted"] = not blockers
    summary["blockers"] = blockers
    return summary, blockers


def _status_is_terminal(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    return value.startswith(TERMINAL_DISPATCH_STATUS_PREFIXES)


def _exact_eval_dispatch_requirements_report(
    report: Any,
    *,
    candidate_archive_sha256: str,
    candidate_archive_bytes: int | None,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": EXACT_EVAL_DISPATCH_REQUIREMENTS_CONTRACT,
        "declared": isinstance(report, dict),
        "accepted": False,
        "ready_for_exact_eval_dispatch_claim": False,
        "lane_claim": {
            "declared": False,
            "active": False,
            "lane_id": "",
            "instance_job_id": "",
            "claims_path": "",
            "claim_status": "",
        },
        "exact_cuda": {
            "declared": False,
            "device": "",
            "entrypoint": "",
            "eval_path": "",
            "archive_sha256": "",
            "archive_bytes": None,
            "full_sample_count": False,
        },
        "blockers": [],
    }
    if not isinstance(report, dict):
        blockers.append("exact_eval_dispatch_requirements_missing")
        summary["blockers"] = blockers
        return summary, blockers

    if report.get("schema_version") != SCHEMA_VERSION:
        blockers.append("exact_eval_dispatch_schema_version_missing_or_invalid")
    if report.get("exact_eval_dispatch_requirements_contract") != EXACT_EVAL_DISPATCH_REQUIREMENTS_CONTRACT:
        blockers.append("exact_eval_dispatch_contract_missing_or_invalid")
    ready_claim = report.get("ready_for_exact_eval_dispatch_claim") is True
    summary["ready_for_exact_eval_dispatch_claim"] = ready_claim
    if not ready_claim:
        blockers.append("exact_eval_dispatch_ready_claim_missing")
    if report.get("score_claim") is not False:
        blockers.append("exact_eval_dispatch_score_claim_must_be_false")
    if report.get("dispatch_attempted") is not False:
        blockers.append("exact_eval_dispatch_dispatch_attempted_must_be_false")

    lane_claim = report.get("lane_claim", report.get("claim"))
    if not isinstance(lane_claim, dict):
        blockers.append("exact_eval_dispatch_lane_claim_missing")
    else:
        lane_id = lane_claim.get("lane_id")
        instance_job_id = lane_claim.get(
            "instance_job_id",
            lane_claim.get("job_name"),
        )
        claims_path = lane_claim.get("claims_path")
        claim_status = lane_claim.get("claim_status", lane_claim.get("status"))
        claimed_with = lane_claim.get("claimed_with", "")
        lane_summary = {
            "declared": True,
            "active": lane_claim.get("active") is True,
            "lane_id": lane_id if isinstance(lane_id, str) else "",
            "instance_job_id": instance_job_id if isinstance(instance_job_id, str) else "",
            "claims_path": claims_path if isinstance(claims_path, str) else "",
            "claim_status": claim_status if isinstance(claim_status, str) else "",
        }
        summary["lane_claim"] = lane_summary
        if not isinstance(lane_id, str) or not lane_id:
            blockers.append("exact_eval_dispatch_lane_id_missing")
        if not isinstance(instance_job_id, str) or not instance_job_id:
            blockers.append("exact_eval_dispatch_instance_job_id_missing")
        if claims_path != DISPATCH_CLAIMS_PATH:
            blockers.append("exact_eval_dispatch_claims_path_not_canonical")
        if lane_claim.get("active") is not True:
            blockers.append("exact_eval_dispatch_lane_claim_not_active")
        if not isinstance(claim_status, str) or not claim_status:
            blockers.append("exact_eval_dispatch_lane_claim_status_missing")
        elif _status_is_terminal(claim_status):
            blockers.append("exact_eval_dispatch_lane_claim_status_terminal")
        if (
            not isinstance(claimed_with, str)
            or "tools/claim_lane_dispatch.py" not in claimed_with
            or " claim" not in f" {claimed_with} "
        ):
            blockers.append("exact_eval_dispatch_lane_claim_helper_not_represented")

    exact_cuda = report.get("exact_cuda")
    if not isinstance(exact_cuda, dict):
        blockers.append("exact_eval_dispatch_cuda_requirements_missing")
    else:
        archive_bytes = exact_cuda.get("archive_bytes")
        cuda_summary = {
            "declared": True,
            "device": exact_cuda.get("device") if isinstance(exact_cuda.get("device"), str) else "",
            "entrypoint": exact_cuda.get("entrypoint") if isinstance(exact_cuda.get("entrypoint"), str) else "",
            "eval_path": exact_cuda.get("eval_path") if isinstance(exact_cuda.get("eval_path"), str) else "",
            "archive_sha256": exact_cuda.get("archive_sha256")
            if isinstance(exact_cuda.get("archive_sha256"), str)
            else "",
            "archive_bytes": archive_bytes if isinstance(archive_bytes, int) else None,
            "full_sample_count": exact_cuda.get("full_sample_count") is True,
        }
        summary["exact_cuda"] = cuda_summary
        if exact_cuda.get("required") is not True:
            blockers.append("exact_eval_dispatch_cuda_required_not_true")
        if exact_cuda.get("device") != "cuda":
            blockers.append("exact_eval_dispatch_device_not_cuda")
        if exact_cuda.get("entrypoint") != EXACT_EVAL_ENTRYPOINT:
            blockers.append("exact_eval_dispatch_entrypoint_not_canonical")
        if exact_cuda.get("eval_path") != EXACT_EVAL_PATH:
            blockers.append("exact_eval_dispatch_eval_path_not_canonical")
        exact_archive_sha = exact_cuda.get("archive_sha256")
        if not _is_sha256(exact_archive_sha):
            blockers.append("exact_eval_dispatch_archive_sha256_missing_or_invalid")
        elif exact_archive_sha != candidate_archive_sha256:
            blockers.append("exact_eval_dispatch_archive_sha256_mismatch")
        if not isinstance(archive_bytes, int) or archive_bytes <= 0:
            blockers.append("exact_eval_dispatch_archive_bytes_missing_or_invalid")
        elif archive_bytes != candidate_archive_bytes:
            blockers.append("exact_eval_dispatch_archive_bytes_mismatch")
        if exact_cuda.get("full_sample_count") is not True:
            blockers.append("exact_eval_dispatch_full_sample_count_not_required")
        if exact_cuda.get("contest_auth_eval_json_required") is not True:
            blockers.append("exact_eval_dispatch_contest_auth_eval_json_not_required")
        if exact_cuda.get("score_claim_after_eval_only") is not True:
            blockers.append("exact_eval_dispatch_score_claim_after_eval_only_missing")

    blockers = list(dict.fromkeys(blockers))
    summary["accepted"] = not blockers
    summary["blockers"] = blockers
    return summary, blockers


def _hpm1_structural_inventory_report(
    report: Any,
    *,
    payload_member: str,
    payload_member_sha256: str,
    repo_root: Path,
    manifest_dir: Path | None,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": HPM1_STRUCTURAL_DECODE_INVENTORY_CONTRACT,
        "declared": isinstance(report, dict),
        "accepted": False,
        "path": "",
        "bytes": None,
        "sha256": "",
        "payload_member": payload_member,
        "payload_member_sha256": payload_member_sha256,
        "structural_reencode_matches_source": False,
        "full_decode_proven": False,
        "byte_exact_semantic_reencode_proven": False,
        "unsupported_wire_constructs": [],
        "blockers": [],
    }
    if report is None:
        return summary, blockers
    if not isinstance(report, dict):
        blockers.append("hpm1_structural_inventory_record_not_object")
        summary["blockers"] = blockers
        return summary, blockers

    if report.get("contract") != HPM1_STRUCTURAL_DECODE_INVENTORY_CONTRACT:
        blockers.append("hpm1_structural_inventory_contract_mismatch")
    if report.get("payload_member") != payload_member:
        blockers.append("hpm1_structural_inventory_payload_member_mismatch")
    if report.get("payload_member_sha256") != payload_member_sha256:
        blockers.append("hpm1_structural_inventory_payload_sha256_mismatch")
    if report.get("full_decode_proven") is not False:
        blockers.append("hpm1_structural_inventory_must_not_claim_full_decode")
    if report.get("byte_exact_semantic_reencode_proven") is not False:
        blockers.append("hpm1_structural_inventory_must_not_claim_semantic_reencode")

    raw_path = report.get("path")
    path_is_safe = _safe_member_name(raw_path)
    if not path_is_safe:
        blockers.append("hpm1_structural_inventory_path_unsafe")
    path = (
        _resolve_path(
            raw_path,
            repo_root=repo_root,
            manifest_dir=manifest_dir,
        )
        if path_is_safe
        else None
    )
    if path is None:
        blockers.append("hpm1_structural_inventory_path_missing")
    elif not path.exists():
        blockers.append("hpm1_structural_inventory_path_missing")
        summary["path"] = repo_relative(path, repo_root)
    elif not path.is_file():
        blockers.append("hpm1_structural_inventory_path_not_file")
        summary["path"] = repo_relative(path, repo_root)
    else:
        raw = path.read_bytes()
        actual_sha = sha256_bytes(raw)
        summary.update(
            {
                "path": repo_relative(path, repo_root),
                "bytes": len(raw),
                "sha256": actual_sha,
            }
        )
        if report.get("bytes") != len(raw):
            blockers.append("hpm1_structural_inventory_bytes_mismatch")
        if report.get("sha256") != actual_sha:
            blockers.append("hpm1_structural_inventory_sha256_mismatch")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = None
            blockers.append("hpm1_structural_inventory_json_invalid")
        if isinstance(payload, dict):
            if payload.get("schema_version") != SCHEMA_VERSION:
                blockers.append("hpm1_structural_inventory_schema_version_invalid")
            if payload.get("kind") != "hpm1_payload_structural_decode_inventory":
                blockers.append("hpm1_structural_inventory_kind_invalid")
            if payload.get("hpm1_structural_decode_inventory_contract") != HPM1_STRUCTURAL_DECODE_INVENTORY_CONTRACT:
                blockers.append("hpm1_structural_inventory_payload_contract_invalid")
            if payload.get("score_claim") is not False:
                blockers.append("hpm1_structural_inventory_score_claim_must_be_false")
            if payload.get("dispatch_attempted") is not False:
                blockers.append("hpm1_structural_inventory_dispatch_attempted_must_be_false")
            if payload.get("ready_for_exact_eval_dispatch") is not False:
                blockers.append("hpm1_structural_inventory_ready_for_exact_eval_must_be_false")
            segment = payload.get("segment")
            if not isinstance(segment, dict):
                blockers.append("hpm1_structural_inventory_segment_missing")
            elif segment.get("sha256") != payload_member_sha256:
                blockers.append("hpm1_structural_inventory_segment_sha256_mismatch")
            structural = payload.get("structural_reencode")
            structural_matches = (
                isinstance(structural, dict)
                and structural.get("matches_source_segment") is True
                and structural.get("reencoded_segment_sha256") == payload_member_sha256
                and structural.get("not_semantic_decode_reencode_parity") is True
            )
            summary["structural_reencode_matches_source"] = structural_matches
            if not structural_matches:
                blockers.append("hpm1_structural_inventory_structural_reencode_not_proven")
            full_decode = payload.get("full_decode")
            full_decode_proven = isinstance(full_decode, dict) and full_decode.get("passed") is True
            summary["full_decode_proven"] = full_decode_proven
            if full_decode_proven:
                blockers.append("hpm1_structural_inventory_claims_full_decode")
            reencode = payload.get("byte_exact_semantic_reencode")
            semantic_reencode_proven = isinstance(reencode, dict) and reencode.get("passed") is True
            summary["byte_exact_semantic_reencode_proven"] = semantic_reencode_proven
            if semantic_reencode_proven:
                blockers.append("hpm1_structural_inventory_claims_semantic_reencode")
            unsupported = payload.get("unsupported_wire_constructs")
            if not isinstance(unsupported, list) or not unsupported:
                blockers.append("hpm1_structural_inventory_unsupported_constructs_missing")
                unsupported_names: list[str] = []
            else:
                unsupported_names = [
                    row.get("name", "")
                    for row in unsupported
                    if isinstance(row, dict) and isinstance(row.get("name"), str)
                ]
                if len(unsupported_names) != len(unsupported):
                    blockers.append("hpm1_structural_inventory_unsupported_constructs_invalid")
            summary["unsupported_wire_constructs"] = unsupported_names

    summary["accepted"] = not blockers
    summary["blockers"] = blockers
    return summary, blockers


def _hpm1_semantic_parity_fail_closed_report(
    report: Any,
    *,
    payload_member: str,
    payload_member_sha256: str,
    candidate_archive_sha256: str,
    repo_root: Path,
    manifest_dir: Path | None,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": HPM1_SEMANTIC_PARITY_FAIL_CLOSED_CONTRACT,
        "declared": isinstance(report, dict),
        "accepted": False,
        "path": "",
        "bytes": None,
        "sha256": "",
        "payload_member": payload_member,
        "payload_member_sha256": payload_member_sha256,
        "candidate_archive_sha256": candidate_archive_sha256,
        "divergence_caught_before_exact_eval": False,
        "hpac_model_loaded": False,
        "probability_rows_inventoried": False,
        "prefix_decode_passed": None,
        "failure_reason": "",
        "failure_context": {},
        "full_decode_proven": False,
        "byte_exact_semantic_reencode_proven": False,
        "blockers": [],
    }
    if report is None:
        return summary, blockers
    if not isinstance(report, dict):
        blockers.append("hpm1_semantic_parity_record_not_object")
        summary["blockers"] = blockers
        return summary, blockers

    if report.get("contract") != HPM1_SEMANTIC_PARITY_FAIL_CLOSED_CONTRACT:
        blockers.append("hpm1_semantic_parity_contract_mismatch")
    if report.get("payload_member") != payload_member:
        blockers.append("hpm1_semantic_parity_payload_member_mismatch")
    if report.get("payload_member_sha256") != payload_member_sha256:
        blockers.append("hpm1_semantic_parity_payload_sha256_mismatch")
    if report.get("candidate_archive_sha256") != candidate_archive_sha256:
        blockers.append("hpm1_semantic_parity_candidate_archive_sha256_mismatch")

    raw_path = report.get("path")
    path_is_safe = _safe_member_name(raw_path)
    if not path_is_safe:
        blockers.append("hpm1_semantic_parity_path_unsafe")
    path = (
        _resolve_path(raw_path, repo_root=repo_root, manifest_dir=manifest_dir)
        if path_is_safe
        else None
    )
    if path is None:
        blockers.append("hpm1_semantic_parity_path_missing")
    elif not path.exists():
        blockers.append("hpm1_semantic_parity_path_missing")
        summary["path"] = repo_relative(path, repo_root)
    elif not path.is_file():
        blockers.append("hpm1_semantic_parity_path_not_file")
        summary["path"] = repo_relative(path, repo_root)
    else:
        raw = path.read_bytes()
        actual_sha = sha256_bytes(raw)
        summary.update({"path": repo_relative(path, repo_root), "bytes": len(raw), "sha256": actual_sha})
        if report.get("bytes") != len(raw):
            blockers.append("hpm1_semantic_parity_bytes_mismatch")
        if report.get("sha256") != actual_sha:
            blockers.append("hpm1_semantic_parity_sha256_mismatch")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = None
            blockers.append("hpm1_semantic_parity_json_invalid")
        if isinstance(payload, dict):
            if payload.get("schema_version") != SCHEMA_VERSION:
                blockers.append("hpm1_semantic_parity_schema_version_invalid")
            if payload.get("kind") != "hpm1_semantic_parity_fail_closed":
                blockers.append("hpm1_semantic_parity_kind_invalid")
            if payload.get("hpm1_semantic_parity_contract") != HPM1_SEMANTIC_PARITY_FAIL_CLOSED_CONTRACT:
                blockers.append("hpm1_semantic_parity_payload_contract_invalid")
            if payload.get("score_claim") is not False:
                blockers.append("hpm1_semantic_parity_score_claim_must_be_false")
            if payload.get("dispatch_attempted") is not False:
                blockers.append("hpm1_semantic_parity_dispatch_attempted_must_be_false")
            if payload.get("ready_for_exact_eval_dispatch") is not False:
                blockers.append("hpm1_semantic_parity_ready_for_exact_eval_must_be_false")
            if payload.get("candidate_archive_sha256") != candidate_archive_sha256:
                blockers.append("hpm1_semantic_parity_payload_candidate_archive_sha256_mismatch")
            if payload.get("payload_member") != payload_member:
                blockers.append("hpm1_semantic_parity_payload_member_record_mismatch")
            if payload.get("payload_member_sha256") != payload_member_sha256:
                blockers.append("hpm1_semantic_parity_payload_sha256_record_mismatch")

            hpac_model = payload.get("hpac_model_load")
            hpac_loaded = isinstance(hpac_model, dict) and hpac_model.get("loaded") is True
            summary["hpac_model_loaded"] = hpac_loaded
            if not hpac_loaded:
                blockers.append("hpm1_semantic_parity_hpac_model_not_loaded")

            probability_rows = payload.get("probability_row_probe")
            probability_rows_ok = isinstance(probability_rows, dict) and probability_rows.get("passed") is True
            summary["probability_rows_inventoried"] = probability_rows_ok
            if not probability_rows_ok:
                blockers.append("hpm1_semantic_parity_probability_rows_not_inventoried")

            prefix_decode = payload.get("prefix_decode")
            prefix_failed = isinstance(prefix_decode, dict) and prefix_decode.get("passed") is False
            summary["prefix_decode_passed"] = (
                prefix_decode.get("passed") if isinstance(prefix_decode, dict) else None
            )
            failure_reason = prefix_decode.get("failure_reason") if isinstance(prefix_decode, dict) else ""
            failure_context = prefix_decode.get("failure_context") if isinstance(prefix_decode, dict) else {}
            summary["failure_reason"] = failure_reason if isinstance(failure_reason, str) else ""
            summary["failure_context"] = failure_context if isinstance(failure_context, dict) else {}
            if not prefix_failed:
                blockers.append("hpm1_semantic_parity_prefix_divergence_not_caught")
            elif not isinstance(failure_reason, str) or not failure_reason:
                blockers.append("hpm1_semantic_parity_failure_reason_missing")
            elif not isinstance(failure_context, dict):
                blockers.append("hpm1_semantic_parity_failure_context_missing")
            else:
                required_context = (
                    "frame",
                    "group",
                    "symbol_in_group",
                    "decoded_symbol_count_before_failure",
                    "probability_variant",
                )
                for key in required_context:
                    if key not in failure_context:
                        blockers.append(f"hpm1_semantic_parity_failure_context_missing:{key}")

            divergence = payload.get("divergence_caught_before_exact_eval") is True
            summary["divergence_caught_before_exact_eval"] = divergence
            if not divergence:
                blockers.append("hpm1_semantic_parity_divergence_gate_missing")

            full_decode = payload.get("full_decode")
            full_decode_proven = isinstance(full_decode, dict) and full_decode.get("passed") is True
            summary["full_decode_proven"] = full_decode_proven
            if full_decode_proven:
                blockers.append("hpm1_semantic_parity_must_not_claim_full_decode")
            reencode = payload.get("byte_exact_semantic_reencode")
            semantic_reencode_proven = isinstance(reencode, dict) and reencode.get("passed") is True
            summary["byte_exact_semantic_reencode_proven"] = semantic_reencode_proven
            if semantic_reencode_proven:
                blockers.append("hpm1_semantic_parity_must_not_claim_semantic_reencode")

    summary["accepted"] = not blockers
    summary["blockers"] = blockers
    return summary, blockers


def _contains_marker(value: Any, marker: str) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.lower().replace("-", "_")
    if marker == "proxy" and ("non_proxy" in normalized or "no_proxy" in normalized or "proxy_free" in normalized):
        return False
    return marker in normalized


def _declared_marker_value(value: Any) -> bool:
    return value not in (False, None, "")


def _audit_candidate_rows(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    row_summaries: dict[str, Any] = {}
    blockers: list[str] = []
    for field in ROW_COLLECTION_FIELDS:
        rows = payload.get(field)
        if rows is None:
            row_summaries[field] = {"declared": False, "count": 0, "blocked_rows": []}
            continue
        if not isinstance(rows, list):
            blockers.append(f"{field}_not_list")
            row_summaries[field] = {"declared": True, "count": 0, "blocked_rows": []}
            continue
        blocked_rows: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                blockers.append(f"{field}_{index}_not_object")
                blocked_rows.append({"index": index, "reasons": ["not_object"]})
                continue
            reasons: list[str] = []
            if row.get("score_claim") is True:
                reasons.append("score_claim_true")
            if row.get("dispatch_attempted") is True or row.get("dispatch_performed") is True:
                reasons.append("dispatch_attempted_true")
            if any(key in row and _declared_marker_value(row.get(key)) for key in PROXY_ROW_KEYS):
                reasons.append("proxy_marker")
            if any(_contains_marker(row.get(key), "proxy") for key in PROXY_ROW_STRING_FIELDS):
                reasons.append("proxy_marker")
            if any(key in row and _declared_marker_value(row.get(key)) for key in SIDECAR_ROW_KEYS):
                reasons.append("sidecar_marker")
            if any(_contains_marker(row.get(key), "sidecar") for key in ("role", "source", "tag", "name")):
                reasons.append("sidecar_marker")
            if reasons:
                reason_set = sorted(set(reasons))
                blocked_rows.append({"index": index, "reasons": reason_set})
                for reason in reason_set:
                    blockers.append(f"{field}_{reason}:{index}")
        row_summaries[field] = {
            "declared": True,
            "count": len(rows),
            "blocked_rows": blocked_rows,
        }
    return row_summaries, blockers


def _scan_local_headers(raw: bytes) -> list[dict[str, Any]]:
    headers: list[dict[str, Any]] = []
    offset = 0
    while offset + 4 <= len(raw):
        signature = struct.unpack_from("<I", raw, offset)[0]
        if signature in {
            CENTRAL_DIRECTORY_SIGNATURE,
            END_OF_CENTRAL_DIRECTORY_SIGNATURE,
        }:
            break
        if signature != LOCAL_FILE_HEADER_SIGNATURE:
            break
        if offset + 30 > len(raw):
            break
        (
            _version,
            flag_bits,
            compress_type,
            _mtime,
            _mdate,
            _crc32,
            compress_size,
            file_size,
            name_len,
            extra_len,
        ) = struct.unpack_from("<HHHHHIIIHH", raw, offset + 4)
        name_start = offset + 30
        name_end = name_start + name_len
        data_start = name_end + extra_len
        data_end = data_start + compress_size
        if data_end > len(raw):
            break
        try:
            filename = raw[name_start:name_end].decode("utf-8")
        except UnicodeDecodeError:
            filename = ""
        headers.append(
            {
                "filename": filename,
                "header_offset": offset,
                "file_size": file_size,
                "compress_size": compress_size,
                "compress_type": compress_type,
                "flag_bits": flag_bits,
                "extra_len": extra_len,
            }
        )
        offset = data_end
    return headers


def _expected_zip_mode(name: str) -> int:
    return DETERMINISTIC_ZIP_INFLATE_MODE if name == CONTEST_INFLATE_MEMBER else DETERMINISTIC_ZIP_FILE_MODE


def _zip_determinism_contract(
    infos: list[zipfile.ZipInfo],
    local_headers: list[dict[str, Any]],
) -> dict[str, Any]:
    central_names = [info.filename for info in infos]
    local_names = [row["filename"] for row in local_headers]
    local_by_offset = {int(row["header_offset"]): row for row in local_headers}
    bad_timestamps: list[str] = []
    bad_external_attr_modes: list[dict[str, Any]] = []
    bad_create_systems: list[dict[str, Any]] = []
    bad_compress_types: list[dict[str, Any]] = []
    data_descriptor_members: list[str] = []
    extra_field_members: list[str] = []
    for info in infos:
        if tuple(info.date_time) != DETERMINISTIC_ZIP_DATE_TIME:
            bad_timestamps.append(info.filename)
        mode = info.external_attr >> 16
        expected_mode = _expected_zip_mode(info.filename)
        if mode != expected_mode:
            bad_external_attr_modes.append(
                {
                    "filename": info.filename,
                    "mode": mode,
                    "expected_mode": expected_mode,
                }
            )
        if info.create_system != DETERMINISTIC_ZIP_CREATE_SYSTEM:
            bad_create_systems.append(
                {
                    "filename": info.filename,
                    "create_system": info.create_system,
                    "expected_create_system": DETERMINISTIC_ZIP_CREATE_SYSTEM,
                }
            )
        if info.compress_type not in DETERMINISTIC_ZIP_ALLOWED_COMPRESS_TYPES:
            bad_compress_types.append(
                {
                    "filename": info.filename,
                    "compress_type": info.compress_type,
                    "allowed": list(DETERMINISTIC_ZIP_ALLOWED_COMPRESS_TYPES),
                }
            )
        if info.flag_bits & ZIP_DATA_DESCRIPTOR_FLAG:
            data_descriptor_members.append(info.filename)
        local = local_by_offset.get(int(info.header_offset), {})
        if int(local.get("extra_len", 0) or 0) != 0 or bool(info.extra):
            extra_field_members.append(info.filename)

    central_local_order_matches = local_names == central_names
    passed = (
        central_local_order_matches
        and not bad_timestamps
        and not bad_external_attr_modes
        and not bad_create_systems
        and not bad_compress_types
        and not data_descriptor_members
        and not extra_field_members
    )
    return {
        "schema_version": 1,
        "passed": passed,
        "required_date_time": list(DETERMINISTIC_ZIP_DATE_TIME),
        "required_file_mode": DETERMINISTIC_ZIP_FILE_MODE,
        "required_inflate_mode": DETERMINISTIC_ZIP_INFLATE_MODE,
        "required_create_system": DETERMINISTIC_ZIP_CREATE_SYSTEM,
        "allowed_compress_types": list(DETERMINISTIC_ZIP_ALLOWED_COMPRESS_TYPES),
        "central_local_order_matches": central_local_order_matches,
        "bad_timestamps": bad_timestamps,
        "bad_external_attr_modes": bad_external_attr_modes,
        "bad_create_systems": bad_create_systems,
        "bad_compress_types": bad_compress_types,
        "data_descriptor_members": data_descriptor_members,
        "extra_field_members": extra_field_members,
    }


def _zip_wire_contract(archive_path: Path, infos: list[zipfile.ZipInfo]) -> dict[str, Any]:
    raw = archive_path.read_bytes()
    local_headers = _scan_local_headers(raw)
    mismatches = []
    central_records = []
    for info in infos:
        local_name = ""
        local_error = ""
        try:
            if info.header_offset + 30 > len(raw):
                raise ValueError("local header extends beyond archive")
            signature = struct.unpack_from("<I", raw, info.header_offset)[0]
            if signature != LOCAL_FILE_HEADER_SIGNATURE:
                raise ValueError(f"bad local header signature 0x{signature:08x}")
            name_len = struct.unpack_from("<H", raw, info.header_offset + 26)[0]
            extra_len = struct.unpack_from("<H", raw, info.header_offset + 28)[0]
            name_start = info.header_offset + 30
            name_end = name_start + name_len
            if name_end + extra_len > len(raw):
                raise ValueError("local header name/extra extends beyond archive")
            local_name = raw[name_start:name_end].decode("utf-8")
        except (UnicodeDecodeError, ValueError, struct.error) as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        if local_error or local_name != info.filename:
            mismatches.append(
                {
                    "central_name": info.filename,
                    "local_name": local_name,
                    "header_offset": info.header_offset,
                    "error": local_error,
                }
            )
        central_records.append(
            {
                "filename": info.filename,
                "header_offset": info.header_offset,
                "file_size": info.file_size,
                "compress_size": info.compress_size,
                "compress_type": info.compress_type,
                "flag_bits": info.flag_bits,
                "external_attr_mode": info.external_attr >> 16,
                "create_system": info.create_system,
                "extra_len": len(info.extra),
                "date_time": list(info.date_time),
            }
        )
    local_names = [row["filename"] for row in local_headers]
    central_names = [info.filename for info in infos]
    duplicate_local_names = sorted({name for name in local_names if local_names.count(name) > 1})
    unsafe_names = sorted(name for name in [*local_names, *central_names] if not _safe_member_name(name))
    passed = (
        len(local_headers) == len(infos)
        and not duplicate_local_names
        and not unsafe_names
        and not mismatches
        and all(row["filename"] for row in local_headers)
    )
    determinism_contract = _zip_determinism_contract(infos, local_headers)
    return {
        "schema_version": 1,
        "passed": passed,
        "central_directory_names": central_names,
        "local_header_names": local_names,
        "local_header_count": len(local_headers),
        "central_directory_count": len(infos),
        "duplicate_local_names": duplicate_local_names,
        "unsafe_names": unsafe_names,
        "central_local_name_mismatches": mismatches,
        "local_headers": local_headers,
        "central_records": central_records,
        "determinism_contract": determinism_contract,
    }


def _read_archive_members(
    archive_path: Path,
) -> tuple[dict[str, bytes], list[str], str | None, dict[str, Any]]:
    wire_contract: dict[str, Any] = {
        "schema_version": 1,
        "passed": False,
        "error": "archive_not_read",
    }
    try:
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            infos = archive.infolist()
            duplicates = sorted({name for name in names if names.count(name) > 1})
            wire_contract = _zip_wire_contract(archive_path, infos)
            members = {name: archive.read(name) for name in names}
            return members, duplicates, None, wire_contract
    except Exception as exc:
        return {}, [], f"{type(exc).__name__}: {exc}", wire_contract


def _byte_closed_archive_parity_report(
    *,
    candidate_archive_contract: Any,
    archive_path: Path | None,
    candidate_archive_bytes: int | None,
    candidate_archive_sha256: str,
    archive_record: Any,
    archive_error: str | None,
    archive_wire_contract: Mapping[str, Any],
    archive_untracked_members: list[str],
    archive_member_order_matches_manifest: bool,
    manifest_path: Path | None,
    manifest_payload: dict[str, Any] | None,
    manifest_error: str,
    manifest_record: Any,
    archive_member_manifest_sha256: Any,
    charged_members: Any,
    member_names: list[str],
    member_records: list[dict[str, Any]],
    archive_members: dict[str, bytes],
    repo_root: Path,
) -> dict[str, Any]:
    """Summarize archive/member-manifest byte closure without score claims."""

    determinism_contract = archive_wire_contract.get("determinism_contract")
    manifest_members = manifest_payload.get("members") if isinstance(manifest_payload, dict) else None
    manifest_member_order = manifest_payload.get("member_order") if isinstance(manifest_payload, dict) else None
    manifest_member_count = manifest_payload.get("member_count") if isinstance(manifest_payload, dict) else None
    archive_member_names = sorted(archive_members)
    charged_name_set = {record["name"] for record in member_records if record.get("name")}
    missing_members = sorted(charged_name_set - set(archive_members))
    extra_members = sorted(set(archive_members) - charged_name_set)
    member_hash_mismatches: list[str] = []
    member_byte_mismatches: list[str] = []
    for record in member_records:
        name = record["name"]
        if not name or name not in archive_members:
            continue
        raw = archive_members[name]
        if record.get("bytes") != len(raw):
            member_byte_mismatches.append(name)
        if record.get("sha256") != sha256_bytes(raw):
            member_hash_mismatches.append(name)

    blockers: list[str] = []
    if candidate_archive_contract != CONTEST_ARCHIVE_CONTRACT:
        blockers.append("byte_closed_candidate_archive_contract_not_contest_zip")
    if not isinstance(archive_record, dict):
        blockers.append("byte_closed_candidate_archive_record_missing")
    if archive_path is None or not archive_path.is_file():
        blockers.append("byte_closed_candidate_archive_path_missing")
    if not isinstance(candidate_archive_bytes, int) or candidate_archive_bytes <= 0:
        blockers.append("byte_closed_candidate_archive_bytes_missing")
    if not _is_sha256(candidate_archive_sha256):
        blockers.append("byte_closed_candidate_archive_sha256_missing_or_invalid")
    if isinstance(archive_record, dict):
        if archive_record.get("bytes") != candidate_archive_bytes:
            blockers.append("byte_closed_candidate_archive_record_bytes_mismatch")
        if archive_record.get("sha256") != candidate_archive_sha256:
            blockers.append("byte_closed_candidate_archive_record_sha256_mismatch")
    if archive_error:
        blockers.append("byte_closed_candidate_archive_not_readable_zip")
    if archive_wire_contract.get("passed") is not True:
        blockers.append("byte_closed_zip_wire_contract_failed")
    if not isinstance(determinism_contract, dict) or determinism_contract.get("passed") is not True:
        blockers.append("byte_closed_zip_determinism_contract_failed")
    if CONTEST_INFLATE_MEMBER not in archive_members:
        blockers.append("byte_closed_inflate_member_missing")
    if archive_untracked_members or extra_members:
        blockers.append("byte_closed_archive_has_untracked_members")
    if missing_members:
        blockers.append("byte_closed_archive_missing_charged_members")
    if member_byte_mismatches:
        blockers.append("byte_closed_archive_member_bytes_mismatch")
    if member_hash_mismatches:
        blockers.append("byte_closed_archive_member_sha256_mismatch")
    if not archive_member_order_matches_manifest:
        blockers.append("byte_closed_archive_member_order_mismatch")
    if not isinstance(manifest_record, dict):
        blockers.append("byte_closed_archive_member_manifest_record_missing")
    if manifest_path is None or not manifest_path.is_file():
        blockers.append("byte_closed_archive_member_manifest_path_missing")
    if manifest_error:
        blockers.append("byte_closed_archive_member_manifest_json_invalid")
    if manifest_payload is None:
        blockers.append("byte_closed_archive_member_manifest_payload_missing")
    else:
        if manifest_payload.get("schema_version") != SCHEMA_VERSION:
            blockers.append("byte_closed_archive_member_manifest_schema_version_invalid")
        if manifest_payload.get("archive_member_manifest_contract") != ARCHIVE_MEMBER_MANIFEST_CONTRACT:
            blockers.append("byte_closed_archive_member_manifest_contract_invalid")
        if not _archive_member_manifest_kind_valid(manifest_payload.get("kind")):
            blockers.append("byte_closed_archive_member_manifest_kind_invalid")
        if manifest_members != charged_members:
            blockers.append("byte_closed_archive_member_manifest_members_mismatch")
        if manifest_member_order != member_names:
            blockers.append("byte_closed_archive_member_manifest_order_mismatch")
        if manifest_member_count != len(member_names):
            blockers.append("byte_closed_archive_member_manifest_count_mismatch")
    if manifest_path is not None and manifest_path.is_file():
        actual_manifest_sha = sha256_file(manifest_path)
        if archive_member_manifest_sha256 != actual_manifest_sha:
            blockers.append("byte_closed_archive_member_manifest_sha256_mismatch")
        if isinstance(manifest_record, dict):
            if manifest_record.get("bytes") != manifest_path.stat().st_size:
                blockers.append("byte_closed_archive_member_manifest_record_bytes_mismatch")
            if manifest_record.get("sha256") != actual_manifest_sha:
                blockers.append("byte_closed_archive_member_manifest_record_sha256_mismatch")

    blockers = list(dict.fromkeys(blockers))
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": "categorical_byte_closed_archive_parity_v1",
        "proven": not blockers,
        "score_claim": False,
        "dispatch_attempted": False,
        "candidate_archive": {
            "path": repo_relative(archive_path, repo_root) if archive_path is not None else "",
            "bytes": candidate_archive_bytes,
            "sha256": candidate_archive_sha256,
        },
        "archive_member_manifest": {
            "path": repo_relative(manifest_path, repo_root) if manifest_path is not None else "",
            "sha256": sha256_file(manifest_path) if manifest_path is not None and manifest_path.is_file() else "",
            "schema_valid": (
                isinstance(manifest_payload, dict)
                and manifest_payload.get("schema_version") == SCHEMA_VERSION
                and manifest_payload.get("archive_member_manifest_contract") == ARCHIVE_MEMBER_MANIFEST_CONTRACT
                and _archive_member_manifest_kind_valid(manifest_payload.get("kind"))
            ),
        },
        "member_count": len(archive_member_names),
        "charged_member_count": len(member_records),
        "member_order_matches_manifest": archive_member_order_matches_manifest,
        "members_match_charged_members": not missing_members and not extra_members,
        "member_bytes_match": not member_byte_mismatches,
        "member_sha256s_match": not member_hash_mismatches,
        "zip_wire_contract_passed": archive_wire_contract.get("passed") is True,
        "zip_determinism_contract_passed": (
            isinstance(determinism_contract, dict)
            and determinism_contract.get("passed") is True
        ),
        "contains_inflate_sh": CONTEST_INFLATE_MEMBER in archive_members,
        "missing_charged_members": missing_members,
        "untracked_members": sorted({*archive_untracked_members, *extra_members}),
        "member_byte_mismatches": member_byte_mismatches,
        "member_sha256_mismatches": member_hash_mismatches,
        "blockers": blockers,
    }


def audit_categorical_candidate_manifest(
    payload: dict[str, Any],
    *,
    repo_root: str | Path,
    manifest_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Audit a categorical candidate manifest before exact-eval dispatch."""

    root = Path(repo_root)
    manifest_base = Path(manifest_dir) if manifest_dir is not None else None
    contract = build_categorical_compression_contract()
    blockers: list[str] = []
    warnings: list[str] = []

    candidate_schema_valid = payload.get("schema_version") == SCHEMA_VERSION
    candidate_kind = payload.get("kind")
    candidate_kind_valid = candidate_kind in CANDIDATE_MANIFEST_KINDS
    if not candidate_schema_valid:
        blockers.append("candidate_manifest_schema_version_missing_or_invalid")
    if not candidate_kind_valid:
        blockers.append("candidate_manifest_kind_missing_or_invalid")

    if payload.get("candidate_manifest_contract") != CANDIDATE_MANIFEST_CONTRACT:
        blockers.append("candidate_manifest_contract_missing_or_invalid")
    if payload.get("score_claim") is not False:
        blockers.append("candidate_manifest_score_claim_must_be_false")
    if payload.get("dispatch_attempted") is not False:
        blockers.append("candidate_manifest_dispatch_attempted_must_be_false")

    if not _is_sha256(payload.get("source_archive_sha256")):
        blockers.append("source_archive_sha256_missing_or_invalid")

    if payload.get("fixture_only") is True:
        blockers.append("fixture_only_candidate_not_dispatchable")

    expected_names = list(CONTEST_SEGNET_CLASS_NAME_TUPLE)
    if payload.get("semantic_class_order") != expected_names:
        blockers.append("semantic_class_order_mismatch")

    expected_gray = [SELFCOMP_CLASS_TO_GRAY[index] for index in range(len(SELFCOMP_CLASS_TO_GRAY))]
    if payload.get("selfcomp_gray_codebook") != expected_gray:
        blockers.append("selfcomp_gray_codebook_mismatch")

    if not _is_sha256(payload.get("archive_member_manifest_sha256")):
        blockers.append("archive_member_manifest_sha256_missing_or_invalid")
    manifest_payload: dict[str, Any] | None = None
    manifest_record = payload.get("archive_member_manifest")
    manifest_path: Path | None = None
    manifest_error = ""
    if not isinstance(manifest_record, dict):
        blockers.append("archive_member_manifest_record_missing")
    else:
        manifest_path = _resolve_path(
            manifest_record.get("path"),
            repo_root=root,
            manifest_dir=manifest_base,
        )
        if manifest_path is None or not manifest_path.exists():
            blockers.append("archive_member_manifest_path_missing")
        elif not manifest_path.is_file():
            blockers.append("archive_member_manifest_path_not_file")
        else:
            manifest_bytes = manifest_path.read_bytes()
            actual_sha = sha256_bytes(manifest_bytes)
            if manifest_record.get("bytes") != len(manifest_bytes):
                blockers.append("archive_member_manifest_bytes_mismatch")
            if manifest_record.get("sha256") != actual_sha:
                blockers.append("archive_member_manifest_record_sha256_mismatch")
            if payload.get("archive_member_manifest_sha256") != actual_sha:
                blockers.append("archive_member_manifest_sha256_mismatch")
            try:
                loaded = json.loads(manifest_bytes.decode("utf-8"))
                if not isinstance(loaded, dict):
                    blockers.append("archive_member_manifest_not_object")
                else:
                    if loaded.get("schema_version") != SCHEMA_VERSION:
                        blockers.append("archive_member_manifest_schema_version_missing_or_invalid")
                    if loaded.get("archive_member_manifest_contract") != ARCHIVE_MEMBER_MANIFEST_CONTRACT:
                        blockers.append("archive_member_manifest_contract_missing_or_invalid")
                    if not _archive_member_manifest_kind_valid(loaded.get("kind")):
                        blockers.append("archive_member_manifest_kind_missing_or_invalid")
                    manifest_payload = loaded
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                manifest_error = f"{type(exc).__name__}: {exc}"
                blockers.append("archive_member_manifest_json_invalid")

    if payload.get("candidate_archive_contract") != CONTEST_ARCHIVE_CONTRACT:
        blockers.append("candidate_archive_contract_not_contest_archive_zip")

    runtime_consumer = payload.get("runtime_consumer")
    runtime_path: Path | None = None
    if not isinstance(runtime_consumer, dict):
        blockers.append("runtime_consumer_missing")
    else:
        runtime_path = _resolve_path(
            runtime_consumer.get("path"),
            repo_root=root,
            manifest_dir=manifest_base,
        )
        if runtime_path is None or not runtime_path.exists():
            blockers.append("runtime_consumer_path_missing")
        elif not runtime_path.is_file():
            blockers.append("runtime_consumer_path_not_file")
        elif not _is_relative_to(runtime_path, root):
            blockers.append("runtime_consumer_path_outside_repo")
        if runtime_consumer.get("consumes_charged_members") is not True:
            blockers.append("runtime_consumer_does_not_declare_charged_member_use")

    controls = payload.get("no_op_controls")
    if not isinstance(controls, dict):
        blockers.append("no_op_controls_missing")
        control_summary = dict.fromkeys(REQUIRED_CONTROL_NAMES, False)
    else:
        control_summary = {name: _control_passed(controls.get(name)) for name in REQUIRED_CONTROL_NAMES}
        for name, passed in control_summary.items():
            if not passed:
                blockers.append(f"no_op_control_not_passed:{name}")

    charged_members = payload.get("charged_members")
    member_records: list[dict[str, Any]] = []
    role_counts: dict[str, int] = {}
    member_names: list[str] = []
    if not isinstance(charged_members, list) or not charged_members:
        blockers.append("charged_members_missing")
    else:
        for index, record in enumerate(charged_members):
            if not isinstance(record, dict):
                blockers.append(f"charged_member_{index}_not_object")
                continue
            name = record.get("name")
            role = record.get("role")
            byte_count = record.get("bytes")
            digest = record.get("sha256")
            if not _safe_member_name(name):
                blockers.append(f"charged_member_{index}_unsafe_name")
            else:
                member_names.append(name)
            if not isinstance(role, str) or not role:
                blockers.append(f"charged_member_{index}_role_missing")
            elif "sidecar" in role.lower():
                blockers.append(f"charged_member_{index}_sidecar_role_forbidden")
            else:
                role_counts[role] = role_counts.get(role, 0) + 1
            if isinstance(name, str) and "sidecar" in name.lower():
                blockers.append(f"charged_member_{index}_sidecar_name_forbidden")
            if record.get("sidecar") is True or record.get("external_sidecar") is True:
                blockers.append(f"charged_member_{index}_sidecar_flag_forbidden")
            if not isinstance(byte_count, int) or byte_count <= 0:
                blockers.append(f"charged_member_{index}_bytes_invalid")
            if not _is_sha256(digest):
                blockers.append(f"charged_member_{index}_sha256_invalid")
            member_records.append(
                {
                    "name": name if isinstance(name, str) else "",
                    "role": role if isinstance(role, str) else "",
                    "bytes": byte_count if isinstance(byte_count, int) else None,
                    "sha256": digest if isinstance(digest, str) else "",
                }
            )
        duplicates = sorted({name for name in member_names if member_names.count(name) > 1})
        if duplicates:
            blockers.append("charged_member_duplicate_names")
            warnings.append(f"duplicate charged member names: {duplicates}")

    conditioning_prior_contract = audit_categorical_openpilot_mask_priors(
        payload.get("conditioning_priors"),
        charged_member_names=member_names,
        charged_members=member_records,
    )
    blockers.extend(conditioning_prior_contract["dispatch_blockers"])
    warnings.extend(conditioning_prior_contract["warnings"])

    if manifest_payload is not None:
        manifest_members = manifest_payload.get("members")
        manifest_member_order = manifest_payload.get("member_order")
        manifest_member_count = manifest_payload.get("member_count")
        if not isinstance(manifest_members, list):
            blockers.append("archive_member_manifest_members_missing")
        elif manifest_members != charged_members:
            blockers.append("archive_member_manifest_members_mismatch")
        else:
            expected_member_order = [record.get("name", "") for record in manifest_members if isinstance(record, dict)]
            if manifest_member_order != expected_member_order:
                blockers.append("archive_member_manifest_member_order_mismatch")
            if manifest_member_count != len(manifest_members):
                blockers.append("archive_member_manifest_member_count_mismatch")

    for role in REQUIRED_MEMBER_ROLES:
        if role_counts.get(role, 0) < 1:
            blockers.append(f"required_charged_member_role_missing:{role}")

    archive_record = payload.get("candidate_archive")
    archive_path: Path | None = None
    archive_members: dict[str, bytes] = {}
    archive_error: str | None = None
    archive_untracked_members: list[str] = []
    archive_member_order_matches_manifest = False
    candidate_archive_sha256 = ""
    candidate_archive_bytes: int | None = None
    archive_wire_contract: dict[str, Any] = {
        "schema_version": 1,
        "passed": False,
        "error": "candidate_archive_missing",
    }
    if not isinstance(archive_record, dict):
        blockers.append("candidate_archive_missing")
    else:
        archive_path = _resolve_path(
            archive_record.get("path"),
            repo_root=root,
            manifest_dir=manifest_base,
        )
        if archive_path is None or not archive_path.exists():
            blockers.append("candidate_archive_path_missing")
        elif not archive_path.is_file():
            blockers.append("candidate_archive_path_not_file")
        else:
            actual_bytes = archive_path.stat().st_size
            actual_sha = sha256_file(archive_path)
            candidate_archive_bytes = actual_bytes
            candidate_archive_sha256 = actual_sha
            if archive_record.get("bytes") != actual_bytes:
                blockers.append("candidate_archive_bytes_mismatch")
            if archive_record.get("sha256") != actual_sha:
                blockers.append("candidate_archive_sha256_mismatch")
            (
                archive_members,
                archive_duplicates,
                archive_error,
                archive_wire_contract,
            ) = _read_archive_members(archive_path)
            if archive_error is not None:
                blockers.append("candidate_archive_not_readable_zip")
            if archive_duplicates:
                blockers.append("candidate_archive_duplicate_member_names")
            if archive_wire_contract.get("passed") is not True:
                blockers.append("candidate_archive_zip_wire_contract_failed")
            determinism_contract = archive_wire_contract.get("determinism_contract")
            if not isinstance(determinism_contract, dict) or determinism_contract.get("passed") is not True:
                blockers.append("candidate_archive_zip_determinism_contract_failed")
        unsafe_archive_names = sorted(name for name in archive_members if not _safe_member_name(name))
        if unsafe_archive_names:
            blockers.append("candidate_archive_unsafe_member_names")
        if CONTEST_INFLATE_MEMBER not in archive_members:
            blockers.append("candidate_archive_missing_inflate_sh")

    if archive_members:
        archive_name_set = set(archive_members)
        charged_name_set = {record["name"] for record in member_records if record["name"]}
        archive_order = archive_wire_contract.get("central_directory_names")
        archive_member_order_matches_manifest = archive_order == member_names
        if not archive_member_order_matches_manifest:
            blockers.append("candidate_archive_member_order_mismatch")
        for record in member_records:
            name = record["name"]
            if not name:
                continue
            raw = archive_members.get(name)
            if raw is None:
                blockers.append(f"charged_member_missing_from_archive:{name}")
                continue
            if record["bytes"] != len(raw):
                blockers.append(f"charged_member_archive_bytes_mismatch:{name}")
            if record["sha256"] != sha256_bytes(raw):
                blockers.append(f"charged_member_archive_sha256_mismatch:{name}")
        archive_untracked_members = sorted(archive_name_set - charged_name_set)
        if archive_untracked_members:
            blockers.append("candidate_archive_untracked_members")
            warnings.append(f"archive contains untracked members: {archive_untracked_members}")

    byte_closed_archive_parity = _byte_closed_archive_parity_report(
        candidate_archive_contract=payload.get("candidate_archive_contract", ""),
        archive_path=archive_path,
        candidate_archive_bytes=candidate_archive_bytes,
        candidate_archive_sha256=candidate_archive_sha256,
        archive_record=archive_record,
        archive_error=archive_error,
        archive_wire_contract=archive_wire_contract,
        archive_untracked_members=archive_untracked_members,
        archive_member_order_matches_manifest=archive_member_order_matches_manifest,
        manifest_path=manifest_path,
        manifest_payload=manifest_payload,
        manifest_error=manifest_error,
        manifest_record=manifest_record,
        archive_member_manifest_sha256=payload.get("archive_member_manifest_sha256"),
        charged_members=charged_members,
        member_names=member_names,
        member_records=member_records,
        archive_members=archive_members,
        repo_root=root,
    )

    label_prior_payload_manifest, label_prior_payload_manifest_blockers = (
        _label_prior_payload_manifest_report(
            payload.get(
                "label_prior_payload_manifest",
                payload.get("label_prior_payload_manifest_member"),
            ),
            member_records=member_records,
            archive_members=archive_members,
            candidate_source_archive_sha256=payload.get("source_archive_sha256"),
            candidate_conditioning_priors=payload.get("conditioning_priors"),
            expected_class_order=expected_names,
            expected_gray_codebook=expected_gray,
        )
    )
    blockers.extend(label_prior_payload_manifest_blockers)

    runtime_loader_parity, runtime_loader_blockers = _runtime_loader_parity_report(
        payload.get("runtime_loader_parity"),
        runtime_path=runtime_path,
        member_records=member_records,
        archive_members=archive_members,
        repo_root=root,
        manifest_dir=manifest_base,
        candidate_archive_sha256=candidate_archive_sha256,
    )
    blockers.extend(runtime_loader_blockers)

    decode_reencode_parity, decode_reencode_blockers = _decode_reencode_parity_report(
        payload.get("decode_reencode_parity"),
        member_records=member_records,
        archive_members=archive_members,
        repo_root=root,
        manifest_dir=manifest_base,
        candidate_archive_sha256=candidate_archive_sha256,
    )
    blockers.extend(decode_reencode_blockers)

    hpm1_structural_inventory, hpm1_structural_blockers = _hpm1_structural_inventory_report(
        payload.get("hpm1_structural_decode_inventory"),
        payload_member=decode_reencode_parity["payload_member"],
        payload_member_sha256=decode_reencode_parity["payload_member_sha256"],
        repo_root=root,
        manifest_dir=manifest_base,
    )
    blockers.extend(hpm1_structural_blockers)

    hpm1_semantic_parity, hpm1_semantic_parity_blockers = _hpm1_semantic_parity_fail_closed_report(
        payload.get("hpm1_semantic_parity_fail_closed"),
        payload_member=decode_reencode_parity["payload_member"],
        payload_member_sha256=decode_reencode_parity["payload_member_sha256"],
        candidate_archive_sha256=candidate_archive_sha256,
        repo_root=root,
        manifest_dir=manifest_base,
    )
    blockers.extend(hpm1_semantic_parity_blockers)

    construction_plan = audit_categorical_charged_label_plan(
        payload.get("candidate_construction_plan"),
        charged_member_names=member_names,
        charged_members=member_records,
    )
    if construction_plan["declared"] and construction_plan["accepted"] is not True:
        blockers.extend(
            f"candidate_construction_plan_{blocker}" for blocker in construction_plan["validation_blockers"]
        )

    candidate_rows, candidate_row_blockers = _audit_candidate_rows(payload)
    blockers.extend(candidate_row_blockers)

    exact_eval_dispatch_requirements, exact_eval_dispatch_blockers = _exact_eval_dispatch_requirements_report(
        payload.get(
            "exact_eval_dispatch_requirements",
            payload.get("exact_eval_dispatch", payload.get("dispatch_readiness")),
        ),
        candidate_archive_sha256=candidate_archive_sha256,
        candidate_archive_bytes=candidate_archive_bytes,
    )
    blockers.extend(exact_eval_dispatch_blockers)

    ready = len(blockers) == 0
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "categorical_candidate_readiness",
        "candidate_manifest": {
            "schema_version": payload.get("schema_version"),
            "kind": payload.get("kind", ""),
            "contract": payload.get("candidate_manifest_contract", ""),
            "required_contract": CANDIDATE_MANIFEST_CONTRACT,
            "allowed_kinds": list(CANDIDATE_MANIFEST_KINDS),
            "schema_valid": (
                candidate_schema_valid
                and candidate_kind_valid
                and payload.get("candidate_manifest_contract") == CANDIDATE_MANIFEST_CONTRACT
                and payload.get("score_claim") is False
                and payload.get("dispatch_attempted") is False
            ),
        },
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": ready,
        "promotion_eligible": False,
        "evidence_grade": "archive_readiness_audit" if ready else "planning_manifest_audit",
        "contract_sha256": sha256_bytes(json_text(contract).encode("utf-8")),
        "source_archive_sha256": payload.get("source_archive_sha256", ""),
        "fixture_only": payload.get("fixture_only") is True,
        "candidate_archive": {
            "contract": payload.get("candidate_archive_contract", ""),
            "path": repo_relative(archive_path, root) if archive_path is not None else "",
            "bytes": archive_path.stat().st_size if archive_path is not None and archive_path.exists() else None,
            "sha256": sha256_file(archive_path) if archive_path is not None and archive_path.exists() else "",
            "zip_read_error": archive_error or "",
            "zip_wire_contract": archive_wire_contract,
            "zip_determinism_contract": archive_wire_contract.get("determinism_contract", {}),
            "contains_inflate_sh": CONTEST_INFLATE_MEMBER in archive_members,
            "untracked_members": archive_untracked_members,
            "member_order_matches_manifest": archive_member_order_matches_manifest,
        },
        "archive_member_manifest": {
            "path": repo_relative(manifest_path, root) if manifest_path is not None else "",
            "exists": bool(manifest_path is not None and manifest_path.exists()),
            "schema_version": (manifest_payload.get("schema_version") if isinstance(manifest_payload, dict) else None),
            "kind": manifest_payload.get("kind", "") if isinstance(manifest_payload, dict) else "",
            "contract": (
                manifest_payload.get("archive_member_manifest_contract", "")
                if isinstance(manifest_payload, dict)
                else ""
            ),
            "required_contract": ARCHIVE_MEMBER_MANIFEST_CONTRACT,
            "schema_valid": (
                isinstance(manifest_payload, dict)
                and manifest_payload.get("schema_version") == SCHEMA_VERSION
                and manifest_payload.get("archive_member_manifest_contract") == ARCHIVE_MEMBER_MANIFEST_CONTRACT
                and _archive_member_manifest_kind_valid(manifest_payload.get("kind"))
            ),
            "bytes": manifest_path.stat().st_size if manifest_path is not None and manifest_path.exists() else None,
            "sha256": sha256_file(manifest_path) if manifest_path is not None and manifest_path.exists() else "",
            "json_read_error": manifest_error,
            "members_match_charged_members": (
                manifest_payload is not None
                and isinstance(manifest_payload.get("members"), list)
                and manifest_payload.get("members") == charged_members
            ),
            "member_order_matches_charged_members": (
                manifest_payload is not None and manifest_payload.get("member_order") == member_names
            ),
            "member_count_matches_charged_members": (
                manifest_payload is not None and manifest_payload.get("member_count") == len(member_names)
            ),
        },
        "byte_closed_archive_parity": byte_closed_archive_parity,
        "semantic_contract": {
            "class_order": expected_names,
            "selfcomp_gray_codebook": expected_gray,
            "matches_candidate": (
                payload.get("semantic_class_order") == expected_names
                and payload.get("selfcomp_gray_codebook") == expected_gray
            ),
        },
        "runtime_consumer": {
            "path": repo_relative(runtime_path, root) if runtime_path is not None else "",
            "exists": bool(runtime_path is not None and runtime_path.exists()),
            "sha256": sha256_file(runtime_path) if runtime_path is not None and runtime_path.is_file() else "",
            "consumes_charged_members": (
                isinstance(runtime_consumer, dict) and runtime_consumer.get("consumes_charged_members") is True
            ),
        },
        "label_prior_payload_manifest": label_prior_payload_manifest,
        "runtime_loader_parity": runtime_loader_parity,
        "decode_reencode_parity": decode_reencode_parity,
        "exact_eval_dispatch_requirements": exact_eval_dispatch_requirements,
        "hpm1_structural_decode_inventory": hpm1_structural_inventory,
        "hpm1_semantic_parity_fail_closed": hpm1_semantic_parity,
        "candidate_construction_plan": construction_plan,
        "conditioning_prior_contract": conditioning_prior_contract,
        "charged_member_summary": {
            "count": len(member_records),
            "roles": dict(sorted(role_counts.items())),
            "required_roles": list(REQUIRED_MEMBER_ROLES),
            "records": sorted(member_records, key=lambda item: item["name"]),
        },
        "no_op_controls": control_summary,
        "candidate_rows": candidate_rows,
        "dispatch_blockers": blockers,
        "warnings": warnings,
    }


__all__ = [
    "ARCHIVE_MEMBER_MANIFEST_CONTRACT",
    "CANDIDATE_MANIFEST_CONTRACT",
    "CANDIDATE_MANIFEST_KINDS",
    "CONTEST_ARCHIVE_CONTRACT",
    "CONTEST_INFLATE_MEMBER",
    "DECODE_REENCODE_INDEPENDENT_PROOF_KIND",
    "DECODE_REENCODE_PARITY_CONTRACT",
    "DETERMINISTIC_ZIP_ALLOWED_COMPRESS_TYPES",
    "DETERMINISTIC_ZIP_CREATE_SYSTEM",
    "DETERMINISTIC_ZIP_DATE_TIME",
    "DETERMINISTIC_ZIP_FILE_MODE",
    "DETERMINISTIC_ZIP_INFLATE_MODE",
    "DISPATCH_CLAIMS_PATH",
    "EXACT_EVAL_DISPATCH_REQUIREMENTS_CONTRACT",
    "EXACT_EVAL_ENTRYPOINT",
    "EXACT_EVAL_PATH",
    "HPM1_SEMANTIC_PARITY_FAIL_CLOSED_CONTRACT",
    "HPM1_STRUCTURAL_DECODE_INVENTORY_CONTRACT",
    "LABEL_PRIOR_PAYLOAD_MANIFEST_CONTRACT",
    "LABEL_PRIOR_PAYLOAD_MANIFEST_MEMBER",
    "LABEL_PRIOR_PAYLOAD_MANIFEST_ROLE",
    "REQUIRED_CONTROL_NAMES",
    "REQUIRED_MEMBER_ROLES",
    "RUNTIME_EXECUTION_PROOF_KIND",
    "RUNTIME_LOADER_PARITY_CONTRACT",
    "SCHEMA_VERSION",
    "audit_categorical_candidate_manifest",
]
