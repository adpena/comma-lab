# SPDX-License-Identifier: MIT
"""Local byte-closed archive candidates for LFV1 foveation tuple payloads.

The builder packages an existing deterministic ``LFV1`` tuple payload with a
charged runtime-consumer skeleton. The emitted archive is intentionally
fail-closed: it proves local member bytes/SHA-256 custody, but remains blocked
on runtime output parity, no-op controls, and exact CUDA auth eval.
"""

from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from tac.analysis.lapose_foveation_payload import (
    PAYLOAD_MEMBER,
    decode_lapose_foveation_tuple_payload,
)
from tac.lapose_foveation_runtime_skeleton import (
    FOVEATION_PARAMS_MEMBER,
    LFV1_FOVEATION_PARAMS_BRIDGE_CONTRACT,
    PROOF_MEMBER,
    RUNTIME_EFFECT_CONTROLS_CONTRACT,
    RUNTIME_PROOF_SKELETON_CONTRACT,
    RUNTIME_SCORER_VISIBLE_BRIDGE_CONTRACT,
    build_lfv1_foveation_params_bridge_report,
    build_runtime_effect_control_report,
    build_scorer_visible_bridge_report,
    lower_lfv1_to_foveation_params,
)
from tac.repo_io import json_text, repo_relative, sha256_bytes, sha256_file, write_json

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
BUILD_KIND = "lapose_foveation_byte_closed_local_candidate_build"
CANDIDATE_KIND = "lapose_foveation_tuple_payload_candidate_manifest"
ARCHIVE_MEMBER_MANIFEST_KIND = "lapose_foveation_local_candidate_archive_member_manifest"
READINESS_KIND = "lapose_foveation_tuple_payload_candidate_readiness"
RUNTIME_PROOF_SKELETON_KIND = "lapose_foveation_runtime_consumer_proof_skeleton"
CANDIDATE_MANIFEST_CONTRACT = "lapose_foveation_byte_closed_candidate_manifest_v1"
ARCHIVE_MEMBER_MANIFEST_CONTRACT = "lapose_foveation_archive_member_manifest_v1"
RUNTIME_LOADER_PARITY_CONTRACT = "lapose_foveation_runtime_loader_parity_v1"
LFV1_PAYLOAD_DECODE_CONTRACT = "lapose_foveation_lfv1_payload_decode_v1"
RUNTIME_CONSUMER_REPO_PATH = "src/tac/lapose_foveation_runtime_skeleton.py"
CONTEST_INFLATE_MEMBER = "inflate.sh"
RUNTIME_CONSUMER_MEMBER = "runtime_consumer.py"
MEMBER_ROLES = {
    CONTEST_INFLATE_MEMBER: "inflate_entrypoint_fail_closed",
    FOVEATION_PARAMS_MEMBER: "derived_hfv1_foveation_geometry",
    PAYLOAD_MEMBER: "lapose_foveation_tuple_payload",
    RUNTIME_CONSUMER_MEMBER: "decoder_or_runtime_consumer",
    PROOF_MEMBER: "runtime_consumer_proof",
}
MEMBER_ORDER = tuple(sorted(MEMBER_ROLES))
DETERMINISTIC_ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)
DETERMINISTIC_ZIP_FILE_MODE = 0o644
DETERMINISTIC_ZIP_INFLATE_MODE = 0o755
DETERMINISTIC_ZIP_CREATE_SYSTEM = 3
ZIP_DATA_DESCRIPTOR_FLAG = 0x0008
LOCAL_FILE_HEADER_SIGNATURE = 0x04034B50
CENTRAL_DIRECTORY_SIGNATURE = 0x02014B50
END_OF_CENTRAL_DIRECTORY_SIGNATURE = 0x06054B50
REQUIRED_NO_OP_CONTROLS = (
    "lfv1_identity_decode_control",
    "lfv1_tuple_mutation_runtime_output_control",
    "charged_member_presence_control",
    "runtime_consumes_foveation_tuple_control",
    "scorer_visible_frame_warp_control",
    "scorer_visible_byte_output_control",
    "inflate_adapter_byte_output_control",
)


class LaposeFoveationPayloadCandidateError(RuntimeError):
    """Raised when an LFV1 local archive candidate cannot be built or audited."""


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


def _resolve_path(path_value: Any, *, repo_root: Path, manifest_dir: Path | None) -> Path | None:
    if not isinstance(path_value, str) or not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    if manifest_dir is not None and (manifest_dir / path).exists():
        return manifest_dir / path
    return repo_root / path


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=DETERMINISTIC_ZIP_DATE_TIME)
    info.compress_type = zipfile.ZIP_STORED
    mode = DETERMINISTIC_ZIP_INFLATE_MODE if name == CONTEST_INFLATE_MEMBER else DETERMINISTIC_ZIP_FILE_MODE
    info.external_attr = mode << 16
    info.create_system = DETERMINISTIC_ZIP_CREATE_SYSTEM
    return info


def _inflate_script() -> bytes:
    return (
        b"#!/usr/bin/env bash\n"
        b"set -euo pipefail\n"
        b"HERE=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\n"
        b"PYTHON_BIN=\"${PACT_PYTHON_BIN:-python3}\"\n"
        b"if [ \"$#\" -eq 3 ] && [ -n \"${LFV1_BASE_RAW_DIR:-}\" ]; then\n"
        b"  \"$PYTHON_BIN\" \"$HERE/runtime_consumer.py\" --archive-root \"$1\" "
        b"--official-output-dir \"$2\" --file-list \"$3\" "
        b"--base-raw-dir \"$LFV1_BASE_RAW_DIR\" "
        b"--chunk-frames \"${LFV1_CHUNK_FRAMES:-16}\" >/dev/stderr\n"
        b"  exit 0\n"
        b"fi\n"
        b"\"$PYTHON_BIN\" \"$HERE/runtime_consumer.py\" --archive-root \"$HERE\" >/dev/stderr || true\n"
        b"echo 'LFV1 lapose_foveation payload candidate is fail-closed: "
        b"runtime output parity, no-op controls, and exact CUDA auth eval are missing' >&2\n"
        b"exit 2\n"
    )


def _write_archive(path: Path, member_payloads: dict[str, bytes]) -> list[dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in MEMBER_ORDER:
            raw = member_payloads[name]
            archive.writestr(_zip_info(name), raw, compress_type=zipfile.ZIP_STORED)
            records.append(
                {
                    "name": name,
                    "role": MEMBER_ROLES[name],
                    "bytes": len(raw),
                    "sha256": sha256_bytes(raw),
                }
            )
    return records


def _runtime_proof_skeleton(
    *,
    payload_source: dict[str, Any],
    payload_sha256: str,
    payload_bytes: int,
    runtime_consumer_sha256: str,
    runtime_consumer_bytes: int,
    decoded_payload: dict[str, Any],
    runtime_effect_controls: dict[str, Any],
    lfv1_foveation_params_bridge: dict[str, Any],
    scorer_visible_bridge: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": RUNTIME_PROOF_SKELETON_KIND,
        "runtime_consumer_proof_skeleton_contract": RUNTIME_PROOF_SKELETON_CONTRACT,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "payload_source": payload_source,
        "charged_member_names": list(MEMBER_ORDER),
        "charged_member_sha256": {
            PAYLOAD_MEMBER: payload_sha256,
            FOVEATION_PARAMS_MEMBER: lfv1_foveation_params_bridge["output_sha256"],
            RUNTIME_CONSUMER_MEMBER: runtime_consumer_sha256,
        },
        "charged_member_bytes": {
            PAYLOAD_MEMBER: payload_bytes,
            FOVEATION_PARAMS_MEMBER: lfv1_foveation_params_bridge["output_bytes"],
            RUNTIME_CONSUMER_MEMBER: runtime_consumer_bytes,
        },
        "lfv1_payload_decode": {
            "contract": LFV1_PAYLOAD_DECODE_CONTRACT,
            "passed": True,
            "row_count": decoded_payload["row_count"],
            "frame_width": decoded_payload["frame_width"],
            "frame_height": decoded_payload["frame_height"],
        },
        "runtime_effect_controls": runtime_effect_controls,
        "lfv1_foveation_params_bridge": lfv1_foveation_params_bridge,
        "proof_status": {
            "archive_contains_payload_and_runtime": True,
            "lfv1_structure_decode": True,
            "structural_runtime_consumption": runtime_effect_controls[
                "structural_runtime_consumption"
            ]["passed"],
            "scorer_visible_frame_warp_control": runtime_effect_controls[
                "scorer_visible_frame_warp_control"
            ]["passed"],
            "scorer_visible_byte_output_control": runtime_effect_controls[
                "scorer_visible_byte_output_control"
            ]["passed"],
            "local_rgb24_inflate_adapter_control": runtime_effect_controls[
                "inflate_adapter_byte_output_control"
            ]["passed"],
            "lfv1_to_foveation_params_bridge": lfv1_foveation_params_bridge["passed"],
            "scorer_visible_output_bridge": scorer_visible_bridge["bridge_path_present"],
            "scored_runtime_output_parity": False,
            "runtime_output_parity": False,
            "noop_controls": runtime_effect_controls["passed"],
            "exact_cuda_auth_eval": False,
        },
        "dispatch_blockers": [
            *scorer_visible_bridge["blockers"],
            "lapose_foveation_scored_runtime_output_parity_not_proven",
            "exact_cuda_auth_eval_missing",
        ],
    }


def _archive_member_manifest(member_records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": ARCHIVE_MEMBER_MANIFEST_KIND,
        "archive_member_manifest_contract": ARCHIVE_MEMBER_MANIFEST_CONTRACT,
        "fixture_only": False,
        "member_count": len(member_records),
        "member_order": [record["name"] for record in member_records],
        "payload_member": PAYLOAD_MEMBER,
        "members": member_records,
    }


def build_lapose_foveation_payload_archive_candidate(
    *,
    out_dir: str | Path,
    lfv1_payload: bytes,
    payload_source: dict[str, Any],
    repo_root: str | Path,
    source_archive_sha256: str = "",
) -> dict[str, Any]:
    """Build a deterministic fail-closed local archive around LFV1 bytes."""

    decoded_payload = decode_lapose_foveation_tuple_payload(lfv1_payload)
    runtime_effect_controls = build_runtime_effect_control_report(lfv1_payload)
    root = Path(repo_root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    runtime_source_path = root / RUNTIME_CONSUMER_REPO_PATH
    if not runtime_source_path.is_file():
        raise LaposeFoveationPayloadCandidateError(f"runtime skeleton missing: {runtime_source_path}")
    runtime_consumer = runtime_source_path.read_bytes()

    payload_sha = sha256_bytes(lfv1_payload)
    foveation_params, _bridge_preview = lower_lfv1_to_foveation_params(decoded_payload)
    lfv1_foveation_params_bridge = build_lfv1_foveation_params_bridge_report(
        lfv1_payload,
        foveation_params,
    )
    runtime_consumer_sha = sha256_bytes(runtime_consumer)
    runtime_consumer_source = runtime_consumer.decode("utf-8", errors="replace")
    scorer_visible_bridge = build_scorer_visible_bridge_report(
        [*MEMBER_ORDER],
        runtime_consumer_source=runtime_consumer_source,
    )
    proof_skeleton = json_text(
        _runtime_proof_skeleton(
            payload_source=payload_source,
            payload_sha256=payload_sha,
            payload_bytes=len(lfv1_payload),
            runtime_consumer_sha256=runtime_consumer_sha,
            runtime_consumer_bytes=len(runtime_consumer),
            decoded_payload=decoded_payload,
            runtime_effect_controls=runtime_effect_controls,
            lfv1_foveation_params_bridge=lfv1_foveation_params_bridge,
            scorer_visible_bridge=scorer_visible_bridge,
        )
    ).encode("utf-8")
    member_payloads = {
        CONTEST_INFLATE_MEMBER: _inflate_script(),
        FOVEATION_PARAMS_MEMBER: foveation_params,
        PAYLOAD_MEMBER: lfv1_payload,
        RUNTIME_CONSUMER_MEMBER: runtime_consumer,
        PROOF_MEMBER: proof_skeleton,
    }

    archive_path = out / "archive.zip"
    member_records = _write_archive(archive_path, member_payloads)
    archive_sha = sha256_file(archive_path)

    archive_member_manifest = _archive_member_manifest(member_records)
    archive_member_manifest_path = out / "archive_member_manifest.json"
    write_json(archive_member_manifest_path, archive_member_manifest)
    archive_member_manifest_sha = sha256_file(archive_member_manifest_path)

    candidate = {
        "schema_version": SCHEMA_VERSION,
        "kind": CANDIDATE_KIND,
        "candidate_manifest_contract": CANDIDATE_MANIFEST_CONTRACT,
        "fixture_only": False,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "source_archive_sha256": source_archive_sha256,
        "archive_member_manifest_sha256": archive_member_manifest_sha,
        "archive_member_manifest": {
            "path": "archive_member_manifest.json",
            "bytes": archive_member_manifest_path.stat().st_size,
            "sha256": archive_member_manifest_sha,
        },
        "candidate_archive_contract": "contest_archive_zip_fail_closed_local",
        "candidate_archive": {
            "path": "archive.zip",
            "bytes": archive_path.stat().st_size,
            "sha256": archive_sha,
        },
        "lfv1_payload": {
            "member": PAYLOAD_MEMBER,
            "bytes": len(lfv1_payload),
            "sha256": payload_sha,
            "decode_contract": LFV1_PAYLOAD_DECODE_CONTRACT,
            "decoded": decoded_payload,
        },
        "payload_source": payload_source,
        "runtime_consumer": {
            "path": RUNTIME_CONSUMER_REPO_PATH,
            "packaged_member": RUNTIME_CONSUMER_MEMBER,
            "bytes": len(runtime_consumer),
            "sha256": runtime_consumer_sha,
            "consumes_charged_members": True,
        },
        "runtime_loader_parity": {
            "schema_version": SCHEMA_VERSION,
            "runtime_loader_parity_contract": RUNTIME_LOADER_PARITY_CONTRACT,
            "passed": False,
            "score_claim": False,
            "dispatch_attempted": False,
            "runtime_consumer_path": RUNTIME_CONSUMER_REPO_PATH,
            "runtime_consumer_sha256": runtime_consumer_sha,
            "loader_member": RUNTIME_CONSUMER_MEMBER,
            "loader_member_sha256": runtime_consumer_sha,
            "byte_identical_to_runtime_consumer": True,
            "sidecar_free": True,
            "fallback_used": False,
            "loaded_charged_members": [PAYLOAD_MEMBER, FOVEATION_PARAMS_MEMBER, PROOF_MEMBER],
            "structural_runtime_consumption": runtime_effect_controls[
                "structural_runtime_consumption"
            ],
            "scored_runtime_output_parity": runtime_effect_controls[
                "scored_runtime_output_parity"
            ],
            "blocker": "scored_runtime_output_parity_not_proven",
        },
        "scorer_visible_bridge": scorer_visible_bridge,
        "lfv1_foveation_params_bridge": lfv1_foveation_params_bridge,
        "payload_decode": {
            "schema_version": SCHEMA_VERSION,
            "payload_decode_contract": LFV1_PAYLOAD_DECODE_CONTRACT,
            "passed": True,
            "score_claim": False,
            "dispatch_attempted": False,
            "payload_member": PAYLOAD_MEMBER,
            "payload_member_sha256": payload_sha,
            "decoded_row_count": decoded_payload["row_count"],
            "sidecar_free": True,
        },
        "runtime_effect_controls": runtime_effect_controls,
        "no_op_controls": {
            "lfv1_identity_decode_control": runtime_effect_controls[
                "lfv1_identity_decode_control"
            ],
            "lfv1_tuple_mutation_runtime_output_control": runtime_effect_controls[
                "lfv1_tuple_mutation_runtime_output_control"
            ],
            "charged_member_presence_control": {
                "passed": True,
                "scope": "archive_member_manifest_and_zip_member_sha256",
            },
            "runtime_consumes_foveation_tuple_control": runtime_effect_controls[
                "runtime_consumes_foveation_tuple_control"
            ],
            "scorer_visible_frame_warp_control": runtime_effect_controls[
                "scorer_visible_frame_warp_control"
            ],
            "scorer_visible_byte_output_control": runtime_effect_controls[
                "scorer_visible_byte_output_control"
            ],
            "inflate_adapter_byte_output_control": runtime_effect_controls[
                "inflate_adapter_byte_output_control"
            ],
        },
        "charged_members": member_records,
        "runtime_consumer_proof_skeleton_member": {
            "name": PROOF_MEMBER,
            "bytes": len(proof_skeleton),
            "sha256": sha256_bytes(proof_skeleton),
            "contract": RUNTIME_PROOF_SKELETON_CONTRACT,
        },
        "candidate_rows": [
            {
                "row_id": "local_lapose_foveation_lfv1_payload_custody",
                "score_claim": False,
                "dispatch_attempted": False,
                "ready_for_exact_eval_dispatch": False,
                "evidence_grade": "local_payload_archive_custody",
                "payload_member": PAYLOAD_MEMBER,
                "payload_sha256": payload_sha,
                "status": "byte_closed_local_candidate_blocked_on_scored_output_cuda",
            }
        ],
    }
    candidate_path = out / "candidate.json"
    readiness_path = out / "readiness.json"
    write_json(candidate_path, candidate)
    readiness = audit_lapose_foveation_payload_candidate(
        candidate,
        repo_root=root,
        manifest_dir=out,
    )
    write_json(readiness_path, readiness)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": BUILD_KIND,
        "fixture_only": False,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "payload_source": payload_source,
        "paths": {
            "archive": repo_relative(archive_path, root),
            "archive_member_manifest": repo_relative(archive_member_manifest_path, root),
            "candidate": repo_relative(candidate_path, root),
            "readiness": repo_relative(readiness_path, root),
        },
        "archive_sha256": archive_sha,
        "archive_bytes": archive_path.stat().st_size,
        "charged_members": member_records,
        "readiness_blockers": readiness["dispatch_blockers"],
    }
    write_json(out / "summary.json", summary)
    return {
        "archive_member_manifest": archive_member_manifest,
        "candidate_manifest": candidate,
        "readiness": readiness,
        "summary": summary,
    }


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
        if info.compress_type != zipfile.ZIP_STORED:
            bad_compress_types.append({"filename": info.filename, "compress_type": info.compress_type})
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
        "schema_version": SCHEMA_VERSION,
        "passed": passed,
        "required_date_time": list(DETERMINISTIC_ZIP_DATE_TIME),
        "required_file_mode": DETERMINISTIC_ZIP_FILE_MODE,
        "required_inflate_mode": DETERMINISTIC_ZIP_INFLATE_MODE,
        "required_create_system": DETERMINISTIC_ZIP_CREATE_SYSTEM,
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
    local_names = [row["filename"] for row in local_headers]
    central_names = [info.filename for info in infos]
    duplicate_local_names = sorted({name for name in local_names if local_names.count(name) > 1})
    unsafe_names = sorted(name for name in [*local_names, *central_names] if not _safe_member_name(name))
    central_local_name_mismatches = [
        {
            "central_name": info.filename,
            "local_name": local_headers[index]["filename"] if index < len(local_headers) else "",
            "header_offset": info.header_offset,
        }
        for index, info in enumerate(infos)
        if index >= len(local_headers) or local_headers[index]["filename"] != info.filename
    ]
    passed = (
        len(local_headers) == len(infos)
        and not duplicate_local_names
        and not unsafe_names
        and not central_local_name_mismatches
        and all(row["filename"] for row in local_headers)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "passed": passed,
        "central_directory_names": central_names,
        "local_header_names": local_names,
        "local_header_count": len(local_headers),
        "central_directory_count": len(infos),
        "duplicate_local_names": duplicate_local_names,
        "unsafe_names": unsafe_names,
        "central_local_name_mismatches": central_local_name_mismatches,
        "determinism_contract": _zip_determinism_contract(infos, local_headers),
    }


def _read_archive_members(
    archive_path: Path,
) -> tuple[dict[str, bytes], list[str], str | None, dict[str, Any]]:
    wire_contract: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
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


def _runtime_loader_parity_report(
    report: Any,
    *,
    runtime_path: Path | None,
    member_records: list[dict[str, Any]],
    archive_members: dict[str, bytes],
    repo_root: Path,
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
    for required in (PAYLOAD_MEMBER, FOVEATION_PARAMS_MEMBER, PROOF_MEMBER):
        if required not in loaded_member_names:
            blockers.append(f"runtime_loader_parity_required_member_not_loaded:{required}")

    summary["accepted"] = not blockers
    summary["blockers"] = blockers
    return summary, blockers


def _control_passed(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, dict):
        return value.get("passed") is True
    return False


def _runtime_effect_controls_report(
    report: Any,
    *,
    raw_payload: bytes | None,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": RUNTIME_EFFECT_CONTROLS_CONTRACT,
        "declared": isinstance(report, dict),
        "accepted": False,
        "structural_runtime_consumption": {"passed": False},
        "scorer_visible_frame_warp_control": {"passed": False},
        "scorer_visible_byte_output_control": {"passed": False},
        "inflate_adapter_byte_output_control": {"passed": False},
        "scored_runtime_output_parity": {"passed": False},
        "identity_decode_control_passed": False,
        "tuple_mutation_control_passed": False,
        "blockers": [],
    }
    if not isinstance(report, dict):
        blockers.append("runtime_effect_controls_missing")
        summary["blockers"] = blockers
        return summary, blockers

    summary["structural_runtime_consumption"] = report.get(
        "structural_runtime_consumption", {"passed": False}
    )
    summary["scorer_visible_frame_warp_control"] = report.get(
        "scorer_visible_frame_warp_control", {"passed": False}
    )
    summary["scorer_visible_byte_output_control"] = report.get(
        "scorer_visible_byte_output_control", {"passed": False}
    )
    summary["inflate_adapter_byte_output_control"] = report.get(
        "inflate_adapter_byte_output_control", {"passed": False}
    )
    summary["scored_runtime_output_parity"] = report.get(
        "scored_runtime_output_parity", {"passed": False}
    )
    summary["identity_decode_control_passed"] = _control_passed(
        report.get("lfv1_identity_decode_control")
    )
    summary["tuple_mutation_control_passed"] = _control_passed(
        report.get("lfv1_tuple_mutation_runtime_output_control")
    )

    if report.get("schema_version") != SCHEMA_VERSION:
        blockers.append("runtime_effect_controls_schema_version_missing_or_invalid")
    if report.get("runtime_effect_controls_contract") != RUNTIME_EFFECT_CONTROLS_CONTRACT:
        blockers.append("runtime_effect_controls_contract_missing_or_invalid")
    if report.get("score_claim") is not False:
        blockers.append("runtime_effect_controls_score_claim_must_be_false")
    if report.get("dispatch_attempted") is not False:
        blockers.append("runtime_effect_controls_dispatch_attempted_must_be_false")
    if report.get("ready_for_exact_eval_dispatch") is not False:
        blockers.append("runtime_effect_controls_ready_for_exact_eval_dispatch_must_be_false")
    if report.get("passed") is not True:
        blockers.append("runtime_effect_controls_not_passed")
    if not _control_passed(report.get("lfv1_identity_decode_control")):
        blockers.append("runtime_effect_controls_identity_decode_not_passed")
    if not _control_passed(report.get("lfv1_tuple_mutation_runtime_output_control")):
        blockers.append("runtime_effect_controls_tuple_mutation_not_passed")
    if not _control_passed(report.get("runtime_consumes_foveation_tuple_control")):
        blockers.append("runtime_effect_controls_runtime_consumption_not_passed")
    if not _control_passed(report.get("scorer_visible_frame_warp_control")):
        blockers.append("runtime_effect_controls_scorer_visible_frame_warp_not_passed")
    if not _control_passed(report.get("scorer_visible_byte_output_control")):
        blockers.append("runtime_effect_controls_scorer_visible_byte_output_not_passed")
    if not _control_passed(report.get("inflate_adapter_byte_output_control")):
        blockers.append("runtime_effect_controls_inflate_adapter_byte_output_not_passed")

    structural = report.get("structural_runtime_consumption")
    if not isinstance(structural, dict) or structural.get("passed") is not True:
        blockers.append("runtime_effect_controls_structural_consumption_not_passed")
    frame_warp = report.get("scorer_visible_frame_warp_control")
    if not isinstance(frame_warp, dict) or frame_warp.get("passed") is not True:
        blockers.append("runtime_effect_controls_frame_warp_control_not_passed")
    byte_output = report.get("scorer_visible_byte_output_control")
    if not isinstance(byte_output, dict) or byte_output.get("passed") is not True:
        blockers.append("runtime_effect_controls_byte_output_control_not_passed")
    inflate_adapter = report.get("inflate_adapter_byte_output_control")
    if not isinstance(inflate_adapter, dict) or inflate_adapter.get("passed") is not True:
        blockers.append("runtime_effect_controls_inflate_adapter_control_not_passed")
    scored_output = report.get("scored_runtime_output_parity")
    if not isinstance(scored_output, dict):
        blockers.append("runtime_effect_controls_scored_output_parity_missing")
    elif scored_output.get("passed") is not False:
        blockers.append("runtime_effect_controls_scored_output_parity_must_be_false")

    if raw_payload is None:
        blockers.append("runtime_effect_controls_payload_member_missing")
    else:
        try:
            expected = build_runtime_effect_control_report(raw_payload)
        except Exception as exc:
            blockers.append(f"runtime_effect_controls_recompute_failed:{type(exc).__name__}")
        else:
            if report != expected:
                blockers.append("runtime_effect_controls_report_mismatch")

    summary["accepted"] = not blockers
    summary["blockers"] = blockers
    return summary, blockers


def _scorer_visible_bridge_report(
    report: Any,
    *,
    archive_members: dict[str, bytes],
) -> tuple[dict[str, Any], list[str], list[str]]:
    validation_blockers: list[str] = []
    dispatch_blockers: list[str] = []
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": RUNTIME_SCORER_VISIBLE_BRIDGE_CONTRACT,
        "declared": isinstance(report, dict),
        "accepted": False,
        "passed": False,
        "bridge_path_present": False,
        "archive_member_groups": {},
        "runtime_source_references": {},
        "declared_blockers": [],
        "validation_blockers": [],
    }
    if not isinstance(report, dict):
        validation_blockers.append("scorer_visible_bridge_report_missing")
        summary["validation_blockers"] = validation_blockers
        return summary, validation_blockers, dispatch_blockers

    if report.get("schema_version") != SCHEMA_VERSION:
        validation_blockers.append("scorer_visible_bridge_schema_version_missing_or_invalid")
    if report.get("contract") != RUNTIME_SCORER_VISIBLE_BRIDGE_CONTRACT:
        validation_blockers.append("scorer_visible_bridge_contract_missing_or_invalid")
    if report.get("score_claim") is not False:
        validation_blockers.append("scorer_visible_bridge_score_claim_must_be_false")
    if report.get("dispatch_attempted") is not False:
        validation_blockers.append("scorer_visible_bridge_dispatch_attempted_must_be_false")
    if report.get("ready_for_exact_eval_dispatch") is not False:
        validation_blockers.append("scorer_visible_bridge_ready_for_exact_eval_dispatch_must_be_false")
    if report.get("passed") is not False:
        validation_blockers.append("scorer_visible_bridge_must_remain_fail_closed")

    runtime_raw = archive_members.get(RUNTIME_CONSUMER_MEMBER)
    runtime_source = ""
    if runtime_raw is None:
        validation_blockers.append("scorer_visible_bridge_runtime_member_missing")
    else:
        try:
            runtime_source = runtime_raw.decode("utf-8")
        except UnicodeDecodeError:
            validation_blockers.append("scorer_visible_bridge_runtime_member_not_text")

    expected = build_scorer_visible_bridge_report(
        sorted(archive_members),
        runtime_consumer_source=runtime_source,
    )
    if report != expected:
        validation_blockers.append("scorer_visible_bridge_report_mismatch")

    declared = report.get("blockers")
    if isinstance(declared, list) and all(isinstance(item, str) for item in declared):
        dispatch_blockers.extend(declared)
    else:
        validation_blockers.append("scorer_visible_bridge_blockers_missing_or_invalid")

    summary.update(
        {
            "accepted": not validation_blockers,
            "passed": report.get("passed") is True,
            "bridge_path_present": report.get("bridge_path_present") is True,
            "archive_member_groups": report.get("archive_member_groups", {}),
            "runtime_source_references": report.get("runtime_source_references", {}),
            "declared_blockers": dispatch_blockers,
            "validation_blockers": validation_blockers,
        }
    )
    return summary, validation_blockers, dispatch_blockers


def _lfv1_foveation_params_bridge_report(
    report: Any,
    *,
    archive_members: dict[str, bytes],
) -> tuple[dict[str, Any], list[str], list[str]]:
    validation_blockers: list[str] = []
    dispatch_blockers: list[str] = []
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": LFV1_FOVEATION_PARAMS_BRIDGE_CONTRACT,
        "declared": isinstance(report, dict),
        "accepted": False,
        "passed": False,
        "source_member": PAYLOAD_MEMBER,
        "target_member": FOVEATION_PARAMS_MEMBER,
        "derived_bytes_match": False,
        "validation_blockers": [],
        "declared_blockers": [],
    }
    if not isinstance(report, dict):
        validation_blockers.append("lfv1_foveation_params_bridge_report_missing")
        summary["validation_blockers"] = validation_blockers
        return summary, validation_blockers, dispatch_blockers

    if report.get("schema_version") != SCHEMA_VERSION:
        validation_blockers.append("lfv1_foveation_params_bridge_schema_version_missing_or_invalid")
    if report.get("contract") != LFV1_FOVEATION_PARAMS_BRIDGE_CONTRACT:
        validation_blockers.append("lfv1_foveation_params_bridge_contract_missing_or_invalid")
    if report.get("score_claim") is not False:
        validation_blockers.append("lfv1_foveation_params_bridge_score_claim_must_be_false")
    if report.get("dispatch_attempted") is not False:
        validation_blockers.append("lfv1_foveation_params_bridge_dispatch_attempted_must_be_false")
    if report.get("ready_for_exact_eval_dispatch") is not False:
        validation_blockers.append(
            "lfv1_foveation_params_bridge_ready_for_exact_eval_dispatch_must_be_false"
        )

    raw_payload = archive_members.get(PAYLOAD_MEMBER)
    foveation_raw = archive_members.get(FOVEATION_PARAMS_MEMBER)
    if raw_payload is None:
        validation_blockers.append("lfv1_foveation_params_bridge_source_member_missing")
    if foveation_raw is None:
        validation_blockers.append("lfv1_foveation_params_bridge_target_member_missing")
    if raw_payload is not None and foveation_raw is not None:
        try:
            expected = build_lfv1_foveation_params_bridge_report(raw_payload, foveation_raw)
        except Exception as exc:
            validation_blockers.append(
                f"lfv1_foveation_params_bridge_recompute_failed:{type(exc).__name__}"
            )
        else:
            if report != expected:
                validation_blockers.append("lfv1_foveation_params_bridge_report_mismatch")

    declared = report.get("blockers")
    if isinstance(declared, list) and all(isinstance(item, str) for item in declared):
        dispatch_blockers.extend(declared)
    else:
        validation_blockers.append("lfv1_foveation_params_bridge_blockers_missing_or_invalid")

    summary.update(
        {
            "accepted": not validation_blockers,
            "passed": report.get("passed") is True,
            "source_member": report.get("source_member", PAYLOAD_MEMBER),
            "target_member": report.get("target_member", FOVEATION_PARAMS_MEMBER),
            "target_frame_count": report.get("target_frame_count"),
            "target_member_sha256": report.get("target_member_sha256", ""),
            "derived_bytes_match": report.get("derived_bytes_match") is True,
            "declared_blockers": dispatch_blockers,
            "validation_blockers": validation_blockers,
        }
    )
    return summary, validation_blockers, dispatch_blockers


def audit_lapose_foveation_payload_candidate(
    payload: dict[str, Any],
    *,
    repo_root: str | Path,
    manifest_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Audit an LFV1 local archive candidate before any exact-eval use."""

    root = Path(repo_root)
    manifest_base = Path(manifest_dir) if manifest_dir is not None else None
    blockers: list[str] = []
    warnings: list[str] = []

    if payload.get("schema_version") != SCHEMA_VERSION:
        blockers.append("candidate_manifest_schema_version_missing_or_invalid")
    if payload.get("kind") != CANDIDATE_KIND:
        blockers.append("candidate_manifest_kind_missing_or_invalid")
    if payload.get("candidate_manifest_contract") != CANDIDATE_MANIFEST_CONTRACT:
        blockers.append("candidate_manifest_contract_missing_or_invalid")
    if payload.get("score_claim") is not False:
        blockers.append("candidate_manifest_score_claim_must_be_false")
    if payload.get("dispatch_attempted") is not False:
        blockers.append("candidate_manifest_dispatch_attempted_must_be_false")
    if payload.get("ready_for_exact_eval_dispatch") is not False:
        blockers.append("candidate_manifest_ready_for_exact_eval_dispatch_must_be_false")
    if payload.get("fixture_only") is True:
        blockers.append("fixture_only_candidate_not_dispatchable")

    source_archive_sha256 = payload.get("source_archive_sha256")
    if source_archive_sha256 and not _is_sha256(source_archive_sha256):
        blockers.append("source_archive_sha256_invalid")

    manifest_payload: dict[str, Any] | None = None
    manifest_record = payload.get("archive_member_manifest")
    manifest_path: Path | None = None
    if not isinstance(manifest_record, dict):
        blockers.append("archive_member_manifest_record_missing")
    else:
        manifest_path = _resolve_path(
            manifest_record.get("path"),
            repo_root=root,
            manifest_dir=manifest_base,
        )
        if manifest_path is None or not manifest_path.is_file():
            blockers.append("archive_member_manifest_path_missing")
        else:
            manifest_bytes = manifest_path.read_bytes()
            manifest_sha = sha256_bytes(manifest_bytes)
            if manifest_record.get("bytes") != len(manifest_bytes):
                blockers.append("archive_member_manifest_bytes_mismatch")
            if manifest_record.get("sha256") != manifest_sha:
                blockers.append("archive_member_manifest_record_sha256_mismatch")
            if payload.get("archive_member_manifest_sha256") != manifest_sha:
                blockers.append("archive_member_manifest_sha256_mismatch")
            try:
                loaded_manifest = json.loads(manifest_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                blockers.append(f"archive_member_manifest_json_error:{type(exc).__name__}")
            else:
                if isinstance(loaded_manifest, dict):
                    manifest_payload = loaded_manifest
                    if loaded_manifest.get("schema_version") != SCHEMA_VERSION:
                        blockers.append("archive_member_manifest_schema_version_missing_or_invalid")
                    if loaded_manifest.get("kind") != ARCHIVE_MEMBER_MANIFEST_KIND:
                        blockers.append("archive_member_manifest_kind_missing_or_invalid")
                    if (
                        loaded_manifest.get("archive_member_manifest_contract")
                        != ARCHIVE_MEMBER_MANIFEST_CONTRACT
                    ):
                        blockers.append("archive_member_manifest_contract_missing_or_invalid")
                else:
                    blockers.append("archive_member_manifest_not_object")

    member_records: list[dict[str, Any]] = []
    if manifest_payload is not None:
        members = manifest_payload.get("members")
        if not isinstance(members, list) or not members:
            blockers.append("archive_member_manifest_members_missing")
        else:
            for index, record in enumerate(members):
                if not isinstance(record, dict):
                    blockers.append(f"archive_member_manifest_member_not_object:{index}")
                    continue
                name = record.get("name")
                if not _safe_member_name(name):
                    blockers.append(f"archive_member_manifest_member_name_unsafe:{index}")
                    continue
                if record.get("role") != MEMBER_ROLES.get(name):
                    blockers.append(f"archive_member_manifest_member_role_mismatch:{name}")
                if not isinstance(record.get("bytes"), int) or record.get("bytes") < 0:
                    blockers.append(f"archive_member_manifest_member_bytes_invalid:{name}")
                if not _is_sha256(record.get("sha256")):
                    blockers.append(f"archive_member_manifest_member_sha256_invalid:{name}")
                member_records.append(record)
        if manifest_payload.get("member_order") != [record.get("name") for record in member_records]:
            blockers.append("archive_member_manifest_member_order_mismatch")
        if manifest_payload.get("member_count") != len(member_records):
            blockers.append("archive_member_manifest_member_count_mismatch")
        if manifest_payload.get("payload_member") != PAYLOAD_MEMBER:
            blockers.append("archive_member_manifest_payload_member_mismatch")

    archive_record = payload.get("candidate_archive")
    archive_path: Path | None = None
    archive_members: dict[str, bytes] = {}
    duplicate_members: list[str] = []
    archive_error: str | None = None
    zip_wire_contract: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "passed": False,
        "error": "archive_not_loaded",
    }
    if not isinstance(archive_record, dict):
        blockers.append("candidate_archive_record_missing")
    else:
        archive_path = _resolve_path(
            archive_record.get("path"),
            repo_root=root,
            manifest_dir=manifest_base,
        )
        if archive_path is None or not archive_path.is_file():
            blockers.append("candidate_archive_path_missing")
        else:
            archive_bytes = archive_path.stat().st_size
            archive_sha = sha256_file(archive_path)
            if archive_record.get("bytes") != archive_bytes:
                blockers.append("candidate_archive_bytes_mismatch")
            if archive_record.get("sha256") != archive_sha:
                blockers.append("candidate_archive_sha256_mismatch")
            archive_members, duplicate_members, archive_error, zip_wire_contract = _read_archive_members(
                archive_path
            )
            if archive_error:
                blockers.append("candidate_archive_not_readable")
            if duplicate_members:
                blockers.append("candidate_archive_duplicate_members")
            if not zip_wire_contract.get("passed"):
                blockers.append("candidate_archive_zip_wire_contract_failed")
            determinism = zip_wire_contract.get("determinism_contract")
            if not isinstance(determinism, dict) or determinism.get("passed") is not True:
                blockers.append("candidate_archive_zip_determinism_contract_failed")

    record_by_name = {record["name"]: record for record in member_records if record.get("name")}
    archive_names = list(archive_members)
    manifest_names = [record.get("name") for record in member_records]
    missing_members = sorted(name for name in manifest_names if name not in archive_members)
    untracked_members = sorted(name for name in archive_names if name not in record_by_name)
    if missing_members:
        blockers.append("candidate_archive_missing_manifest_members")
    if untracked_members:
        blockers.append("candidate_archive_untracked_members")

    member_sha256_proofs: list[dict[str, Any]] = []
    for record in member_records:
        name = record.get("name")
        raw = archive_members.get(name)
        if raw is None:
            continue
        actual_sha = sha256_bytes(raw)
        actual_bytes = len(raw)
        proof = {
            "name": name,
            "role": record.get("role"),
            "manifest_bytes": record.get("bytes"),
            "actual_bytes": actual_bytes,
            "manifest_sha256": record.get("sha256"),
            "actual_sha256": actual_sha,
            "bytes_match": record.get("bytes") == actual_bytes,
            "sha256_match": record.get("sha256") == actual_sha,
        }
        member_sha256_proofs.append(proof)
        if not proof["bytes_match"]:
            blockers.append(f"archive_member_manifest_member_bytes_mismatch:{name}")
        if not proof["sha256_match"]:
            blockers.append(f"archive_member_manifest_member_sha256_mismatch:{name}")

    lfv1_record = payload.get("lfv1_payload")
    payload_member_proven = False
    payload_decode_report: dict[str, Any] = {
        "contract": LFV1_PAYLOAD_DECODE_CONTRACT,
        "accepted": False,
        "payload_member": PAYLOAD_MEMBER,
        "payload_member_sha256": "",
        "decoded_row_count": None,
        "blockers": [],
    }
    if not isinstance(lfv1_record, dict):
        blockers.append("lfv1_payload_record_missing")
        payload_decode_report["blockers"].append("lfv1_payload_record_missing")
    else:
        if lfv1_record.get("member") != PAYLOAD_MEMBER:
            blockers.append("lfv1_payload_member_mismatch")
            payload_decode_report["blockers"].append("lfv1_payload_member_mismatch")
        raw_payload = archive_members.get(PAYLOAD_MEMBER)
        if raw_payload is None:
            blockers.append("lfv1_payload_member_missing_from_archive")
            payload_decode_report["blockers"].append("lfv1_payload_member_missing_from_archive")
        else:
            payload_sha = sha256_bytes(raw_payload)
            payload_decode_report["payload_member_sha256"] = payload_sha
            if lfv1_record.get("bytes") != len(raw_payload):
                blockers.append("lfv1_payload_member_bytes_mismatch")
                payload_decode_report["blockers"].append("lfv1_payload_member_bytes_mismatch")
            if lfv1_record.get("sha256") != payload_sha:
                blockers.append("lfv1_payload_member_sha256_mismatch")
                payload_decode_report["blockers"].append("lfv1_payload_member_sha256_mismatch")
            try:
                decoded = decode_lapose_foveation_tuple_payload(raw_payload)
            except Exception as exc:
                blockers.append("lfv1_payload_decode_failed")
                payload_decode_report["blockers"].append(f"lfv1_payload_decode_failed:{type(exc).__name__}")
            else:
                payload_decode_report["decoded_row_count"] = decoded["row_count"]
                if lfv1_record.get("decoded") != decoded:
                    blockers.append("lfv1_payload_decoded_preview_mismatch")
                    payload_decode_report["blockers"].append("lfv1_payload_decoded_preview_mismatch")
                payload_member_proven = not payload_decode_report["blockers"]
    payload_decode_report["accepted"] = payload_member_proven

    runtime_path = root / RUNTIME_CONSUMER_REPO_PATH
    runtime_loader_parity, runtime_blockers = _runtime_loader_parity_report(
        payload.get("runtime_loader_parity"),
        runtime_path=runtime_path,
        member_records=member_records,
        archive_members=archive_members,
        repo_root=root,
    )
    blockers.extend(runtime_blockers)

    no_op_controls = payload.get("no_op_controls")
    no_op_summary: dict[str, Any] = {
        "declared": isinstance(no_op_controls, dict),
        "required_controls": list(REQUIRED_NO_OP_CONTROLS),
        "passed_controls": [],
        "failed_controls": [],
    }
    if not isinstance(no_op_controls, dict):
        blockers.append("no_op_controls_missing")
    else:
        for name in REQUIRED_NO_OP_CONTROLS:
            if _control_passed(no_op_controls.get(name)):
                no_op_summary["passed_controls"].append(name)
            else:
                no_op_summary["failed_controls"].append(name)
                blockers.append(f"no_op_control_not_passed:{name}")

    raw_lfv1_payload = archive_members.get(PAYLOAD_MEMBER)
    runtime_effect_controls, runtime_effect_blockers = _runtime_effect_controls_report(
        payload.get("runtime_effect_controls"),
        raw_payload=raw_lfv1_payload,
    )
    blockers.extend(runtime_effect_blockers)

    scorer_visible_bridge, bridge_validation_blockers, bridge_dispatch_blockers = (
        _scorer_visible_bridge_report(
            payload.get("scorer_visible_bridge"),
            archive_members=archive_members,
        )
    )
    blockers.extend(bridge_validation_blockers)
    blockers.extend(bridge_dispatch_blockers)

    lfv1_foveation_params_bridge, foveation_bridge_validation_blockers, foveation_bridge_dispatch_blockers = (
        _lfv1_foveation_params_bridge_report(
            payload.get("lfv1_foveation_params_bridge"),
            archive_members=archive_members,
        )
    )
    blockers.extend(foveation_bridge_validation_blockers)
    blockers.extend(foveation_bridge_dispatch_blockers)

    exact_cuda_auth_eval = payload.get("exact_cuda_auth_eval")
    if not isinstance(exact_cuda_auth_eval, dict) or exact_cuda_auth_eval.get("passed") is not True:
        blockers.append("exact_cuda_auth_eval_missing")

    archive_member_manifest_summary = {
        "path": repo_relative(manifest_path, root) if manifest_path else "",
        "declared": manifest_payload is not None,
        "member_order_matches_manifest": archive_names == manifest_names,
        "member_count_matches_manifest": len(archive_names) == len(manifest_names),
        "member_sha256_proofs": member_sha256_proofs,
    }
    candidate_archive_summary = {
        "path": repo_relative(archive_path, root) if archive_path else "",
        "bytes": archive_path.stat().st_size if archive_path and archive_path.is_file() else 0,
        "sha256": sha256_file(archive_path) if archive_path and archive_path.is_file() else "",
        "duplicate_members": duplicate_members,
        "missing_manifest_members": missing_members,
        "untracked_members": untracked_members,
        "member_order_matches_manifest": archive_names == manifest_names,
        "zip_wire_contract": zip_wire_contract,
        "zip_determinism_contract": zip_wire_contract.get("determinism_contract", {}),
    }
    custody_blocker_prefixes = (
        "archive_member_",
        "candidate_archive_",
        "lfv1_payload_",
    )
    custody_blockers = [
        blocker
        for blocker in blockers
        if blocker.startswith(custody_blocker_prefixes)
    ]
    byte_closed_local_archive = (
        payload_member_proven
        and not custody_blockers
        and candidate_archive_summary["member_order_matches_manifest"]
        and archive_member_manifest_summary["member_order_matches_manifest"]
    )
    unique_blockers = list(dict.fromkeys(blockers))
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": READINESS_KIND,
        "candidate_manifest_contract": CANDIDATE_MANIFEST_CONTRACT,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "ok": byte_closed_local_archive,
        "evidence_grade": "empirical_local_archive_custody_fail_closed",
        "payload_member_proven": payload_member_proven,
        "byte_closed_local_archive": byte_closed_local_archive,
        "candidate_archive": candidate_archive_summary,
        "archive_member_manifest": archive_member_manifest_summary,
        "lfv1_payload_decode": payload_decode_report,
        "runtime_loader_parity": runtime_loader_parity,
        "no_op_controls": no_op_summary,
        "runtime_effect_controls": runtime_effect_controls,
        "scorer_visible_bridge": scorer_visible_bridge,
        "lfv1_foveation_params_bridge": lfv1_foveation_params_bridge,
        "runtime_consumption_audit": {
            "structural_runtime_consumption": runtime_effect_controls[
                "structural_runtime_consumption"
            ],
            "scored_runtime_output_parity": runtime_effect_controls[
                "scored_runtime_output_parity"
            ],
            "scorer_visible_bridge": scorer_visible_bridge,
            "lfv1_foveation_params_bridge": lfv1_foveation_params_bridge,
            "scored_runtime_output_parity_required": True,
        },
        "runtime_contract": {
            "charged_member_required": PAYLOAD_MEMBER,
            "derived_scorer_visible_member_required": FOVEATION_PARAMS_MEMBER,
            "runtime_consumer_required": True,
            "scorer_visible_output_bridge_required": True,
            "runtime_output_parity_required": True,
            "noop_controls_required": True,
            "exact_cuda_auth_eval_required": True,
            "scorer_loads_at_pack_time": False,
        },
        "dispatch_blockers": unique_blockers,
        "warnings": warnings,
    }


__all__ = [
    "ARCHIVE_MEMBER_MANIFEST_CONTRACT",
    "BUILD_KIND",
    "CANDIDATE_MANIFEST_CONTRACT",
    "MEMBER_ORDER",
    "RUNTIME_CONSUMER_REPO_PATH",
    "RUNTIME_LOADER_PARITY_CONTRACT",
    "RUNTIME_PROOF_SKELETON_CONTRACT",
    "LaposeFoveationPayloadCandidateError",
    "audit_lapose_foveation_payload_candidate",
    "build_lapose_foveation_payload_archive_candidate",
]
