"""Readiness checks for byte-closed categorical compression candidates."""

from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from tac.categorical_compression_contract import build_categorical_compression_contract
from tac.categorical_openpilot_mask_prior_contract import (
    audit_categorical_openpilot_mask_priors,
)
from tac.repo_io import json_text, repo_relative, sha256_bytes, sha256_file
from tac.semantic_label_contract import CONTEST_SEGNET_CLASS_NAME_TUPLE, SELFCOMP_CLASS_TO_GRAY

SCHEMA_VERSION = 1
CANDIDATE_MANIFEST_KINDS = (
    "categorical_candidate_manifest",
    "categorical_candidate_fixture_manifest",
    "categorical_qma9_clade_spade_openpilot_candidate_manifest",
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
    return (
        isinstance(value, str)
        and value.startswith("categorical_")
        and value.endswith("archive_member_manifest")
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
    duplicate_local_names = sorted(
        {name for name in local_names if local_names.count(name) > 1}
    )
    unsafe_names = sorted(
        name
        for name in [*local_names, *central_names]
        if not _safe_member_name(name)
    )
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
                        blockers.append(
                            "archive_member_manifest_schema_version_missing_or_invalid"
                        )
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

    conditioning_prior_contract = audit_categorical_openpilot_mask_priors(
        payload.get("conditioning_priors"),
        charged_member_names=member_names,
    )
    blockers.extend(conditioning_prior_contract["dispatch_blockers"])
    warnings.extend(conditioning_prior_contract["warnings"])

    if manifest_payload is not None:
        manifest_members = manifest_payload.get("members")
        if not isinstance(manifest_members, list):
            blockers.append("archive_member_manifest_members_missing")
        elif manifest_members != charged_members:
            blockers.append("archive_member_manifest_members_mismatch")

    for role in REQUIRED_MEMBER_ROLES:
        if role_counts.get(role, 0) < 1:
            blockers.append(f"required_charged_member_role_missing:{role}")

    archive_record = payload.get("candidate_archive")
    archive_path: Path | None = None
    archive_members: dict[str, bytes] = {}
    archive_error: str | None = None
    archive_untracked_members: list[str] = []
    archive_member_order_matches_manifest = False
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

    ready = len(blockers) == 0
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "categorical_candidate_readiness",
        "candidate_manifest": {
            "schema_version": payload.get("schema_version"),
            "kind": payload.get("kind", ""),
            "allowed_kinds": list(CANDIDATE_MANIFEST_KINDS),
            "schema_valid": candidate_schema_valid and candidate_kind_valid,
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
            "schema_version": (
                manifest_payload.get("schema_version") if isinstance(manifest_payload, dict) else None
            ),
            "kind": manifest_payload.get("kind", "") if isinstance(manifest_payload, dict) else "",
            "schema_valid": (
                isinstance(manifest_payload, dict)
                and manifest_payload.get("schema_version") == SCHEMA_VERSION
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
        "conditioning_prior_contract": conditioning_prior_contract,
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
    "CANDIDATE_MANIFEST_KINDS",
    "CONTEST_ARCHIVE_CONTRACT",
    "CONTEST_INFLATE_MEMBER",
    "DETERMINISTIC_ZIP_ALLOWED_COMPRESS_TYPES",
    "DETERMINISTIC_ZIP_CREATE_SYSTEM",
    "DETERMINISTIC_ZIP_DATE_TIME",
    "DETERMINISTIC_ZIP_FILE_MODE",
    "DETERMINISTIC_ZIP_INFLATE_MODE",
    "REQUIRED_CONTROL_NAMES",
    "REQUIRED_MEMBER_ROLES",
    "SCHEMA_VERSION",
    "audit_categorical_candidate_manifest",
]
