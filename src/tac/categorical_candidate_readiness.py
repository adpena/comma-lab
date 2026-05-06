"""Readiness checks for byte-closed categorical compression candidates."""

from __future__ import annotations

import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from tac.categorical_compression_contract import build_categorical_compression_contract
from tac.repo_io import json_text, repo_relative, sha256_bytes, sha256_file
from tac.semantic_label_contract import CONTEST_SEGNET_CLASS_NAME_TUPLE, SELFCOMP_CLASS_TO_GRAY

SCHEMA_VERSION = 1
REQUIRED_CONTROL_NAMES = (
    "decode_reencode_identity_control",
    "label_permutation_fail_closed_control",
    "charged_member_presence_control",
    "runtime_consumes_conditioning_control",
)
REQUIRED_MEMBER_ROLES = ("categorical_payload", "decoder_or_runtime_consumer")
CONTEST_ARCHIVE_CONTRACT = "contest_archive_zip"
CONTEST_INFLATE_MEMBER = "inflate.sh"


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(char in "0123456789abcdef" for char in value.lower())


def _safe_member_name(name: Any) -> bool:
    if not isinstance(name, str) or not name:
        return False
    path = PurePosixPath(name)
    parts = path.parts
    return (
        not path.is_absolute()
        and ".." not in parts
        and all(part not in {"", ".", "__MACOSX"} for part in parts)
        and not any(part.startswith(".") for part in parts)
    )


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


def _read_archive_members(archive_path: Path) -> tuple[dict[str, bytes], list[str], str | None]:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            duplicates = sorted({name for name in names if names.count(name) > 1})
            members = {name: archive.read(name) for name in names}
            return members, duplicates, None
    except Exception as exc:
        return {}, [], f"{type(exc).__name__}: {exc}"


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

    if not _is_sha256(payload.get("source_archive_sha256")):
        blockers.append("source_archive_sha256_missing_or_invalid")

    expected_names = list(CONTEST_SEGNET_CLASS_NAME_TUPLE)
    if payload.get("semantic_class_order") != expected_names:
        blockers.append("semantic_class_order_mismatch")

    expected_gray = [SELFCOMP_CLASS_TO_GRAY[index] for index in range(len(SELFCOMP_CLASS_TO_GRAY))]
    if payload.get("selfcomp_gray_codebook") != expected_gray:
        blockers.append("selfcomp_gray_codebook_mismatch")

    if not _is_sha256(payload.get("archive_member_manifest_sha256")):
        blockers.append("archive_member_manifest_sha256_missing_or_invalid")

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
        control_summary = {name: False for name in REQUIRED_CONTROL_NAMES}
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
            else:
                role_counts[role] = role_counts.get(role, 0) + 1
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

    for role in REQUIRED_MEMBER_ROLES:
        if role_counts.get(role, 0) < 1:
            blockers.append(f"required_charged_member_role_missing:{role}")

    archive_record = payload.get("candidate_archive")
    archive_path: Path | None = None
    archive_members: dict[str, bytes] = {}
    archive_error: str | None = None
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
            if archive_record.get("bytes") != actual_bytes:
                blockers.append("candidate_archive_bytes_mismatch")
            if archive_record.get("sha256") != actual_sha:
                blockers.append("candidate_archive_sha256_mismatch")
            archive_members, archive_duplicates, archive_error = _read_archive_members(archive_path)
            if archive_error is not None:
                blockers.append("candidate_archive_not_readable_zip")
            if archive_duplicates:
                blockers.append("candidate_archive_duplicate_member_names")
        unsafe_archive_names = sorted(name for name in archive_members if not _safe_member_name(name))
        if unsafe_archive_names:
            blockers.append("candidate_archive_unsafe_member_names")
        if CONTEST_INFLATE_MEMBER not in archive_members:
            blockers.append("candidate_archive_missing_inflate_sh")

    if archive_members:
        archive_name_set = set(archive_members)
        charged_name_set = {record["name"] for record in member_records if record["name"]}
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
        untracked = sorted(archive_name_set - charged_name_set)
        if untracked:
            warnings.append(f"archive contains untracked members: {untracked}")

    ready = len(blockers) == 0
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "categorical_candidate_readiness",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": ready,
        "promotion_eligible": False,
        "evidence_grade": "archive_readiness_audit" if ready else "planning_manifest_audit",
        "contract_sha256": sha256_bytes(json_text(contract).encode("utf-8")),
        "source_archive_sha256": payload.get("source_archive_sha256", ""),
        "candidate_archive": {
            "contract": payload.get("candidate_archive_contract", ""),
            "path": repo_relative(archive_path, root) if archive_path is not None else "",
            "bytes": archive_path.stat().st_size if archive_path is not None and archive_path.exists() else None,
            "sha256": sha256_file(archive_path) if archive_path is not None and archive_path.exists() else "",
            "zip_read_error": archive_error or "",
            "contains_inflate_sh": CONTEST_INFLATE_MEMBER in archive_members,
        },
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
            "consumes_charged_members": (
                isinstance(runtime_consumer, dict)
                and runtime_consumer.get("consumes_charged_members") is True
            ),
        },
        "charged_member_summary": {
            "count": len(member_records),
            "roles": dict(sorted(role_counts.items())),
            "required_roles": list(REQUIRED_MEMBER_ROLES),
            "records": sorted(member_records, key=lambda item: item["name"]),
        },
        "no_op_controls": control_summary,
        "dispatch_blockers": blockers,
        "warnings": warnings,
    }


__all__ = [
    "CONTEST_ARCHIVE_CONTRACT",
    "CONTEST_INFLATE_MEMBER",
    "REQUIRED_CONTROL_NAMES",
    "REQUIRED_MEMBER_ROLES",
    "SCHEMA_VERSION",
    "audit_categorical_candidate_manifest",
]
