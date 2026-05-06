"""Fail-closed readiness contract for public PR91/HPM1 replay recovery."""

from __future__ import annotations

import struct
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tac.pr85_bundle import HPM1_MAGIC, Pr85BundleError, parse_pr85_bundle
from tac.pr91_hpm1_codec import (
    DEFAULT_PR91_ARCHIVE,
    DEFAULT_PR91_RUNTIME_SOURCE_DIR,
    EXPECTED_PR91_ARCHIVE_BYTES,
    EXPECTED_PR91_ARCHIVE_SHA256,
    EXPECTED_PR91_HPM1_HPAC_SHA256,
    EXPECTED_PR91_HPM1_MASK_BYTES,
    EXPECTED_PR91_HPM1_MASK_SHA256,
    EXPECTED_PR91_HPM1_TOKENS_SHA256,
    EXPECTED_PR91_MEMBER_X_BYTES,
    EXPECTED_PR91_MEMBER_X_SHA256,
    analyze_pr91_hpm1_runtime_sources,
    repo_rel,
    sha256_bytes,
    sha256_path,
    split_hpm1_mask_segment,
)

SCHEMA_VERSION = 1
KIND = "pr91_hpm1_readiness"
LOCAL_FILE_HEADER_SIGNATURE = 0x04034B50
CENTRAL_DIRECTORY_SIGNATURE = 0x02014B50
END_OF_CENTRAL_DIRECTORY_SIGNATURE = 0x06054B50


@dataclass(frozen=True)
class Gate:
    """One deterministic readiness gate."""

    status: str
    passed: bool
    required_for_dispatch: bool
    reason: str


def _gate(
    *,
    passed: bool,
    reason: str,
    required_for_dispatch: bool = True,
    failed_status: str = "failed_closed",
) -> dict[str, Any]:
    status = "passed" if passed else failed_status
    return asdict(
        Gate(
            status=status,
            passed=passed,
            required_for_dispatch=required_for_dispatch,
            reason=reason,
        )
    )


def _expected_file_record(path: Path, *, expected_bytes: int, expected_sha256: str) -> dict[str, Any]:
    exists = path.is_file()
    actual_bytes = path.stat().st_size if exists else None
    actual_sha = sha256_path(path) if exists else ""
    return {
        "path": repo_rel(path),
        "expected_bytes": expected_bytes,
        "expected_sha256": expected_sha256,
        "exists": exists,
        "bytes": actual_bytes,
        "sha256": actual_sha,
        "matches_expected": exists
        and actual_bytes == expected_bytes
        and actual_sha == expected_sha256,
    }


def _read_single_x_member(archive: Path) -> tuple[bytes | None, dict[str, Any]]:
    if not archive.is_file():
        return None, {"status": "missing_archive", "members": [], "duplicates": []}
    try:
        with zipfile.ZipFile(archive) as zf:
            infos = zf.infolist()
            names = [info.filename for info in infos]
            duplicates = sorted({name for name in names if names.count(name) > 1})
            wire_contract = _zip_wire_contract(archive, infos)
            if names != ["x"]:
                return None, {
                    "status": "not_single_x_archive",
                    "members": names,
                    "duplicates": duplicates,
                    "wire_contract": wire_contract,
                }
            try:
                member_x = zf.read("x")
            except Exception as exc:
                return None, {
                    "status": "zip_member_read_failed",
                    "members": names,
                    "duplicates": duplicates,
                    "wire_contract": wire_contract,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            return member_x, {
                "status": "passed",
                "members": names,
                "duplicates": duplicates,
                "wire_contract": wire_contract,
            }
    except Exception as exc:
        return None, {"status": "zip_read_failed", "error": f"{type(exc).__name__}: {exc}"}


def _zip_wire_contract(archive: Path, infos: list[zipfile.ZipInfo]) -> dict[str, Any]:
    raw = archive.read_bytes()
    local_headers = _scan_local_headers(raw)
    central_records = []
    mismatches = []
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
                "date_time": list(info.date_time),
            }
        )
    local_names = [row["filename"] for row in local_headers]
    duplicate_local_names = sorted({name for name in local_names if local_names.count(name) > 1})
    unsafe_names = sorted(
        name
        for name in [*local_names, *[info.filename for info in infos]]
        if not name or name.startswith("/") or ".." in name.split("/")
    )
    passed = (
        len(local_headers) == len(infos)
        and not duplicate_local_names
        and not unsafe_names
        and not mismatches
        and all(row["filename"] for row in local_headers)
    )
    return {
        "schema_version": 1,
        "passed": passed,
        "local_header_count": len(local_headers),
        "central_directory_count": len(infos),
        "duplicate_local_names": duplicate_local_names,
        "unsafe_names": unsafe_names,
        "central_local_name_mismatches": mismatches,
        "local_headers": local_headers,
        "central_records": central_records,
    }


def _scan_local_headers(raw: bytes) -> list[dict[str, Any]]:
    headers: list[dict[str, Any]] = []
    offset = 0
    while offset + 4 <= len(raw):
        signature = struct.unpack_from("<I", raw, offset)[0]
        if signature in {CENTRAL_DIRECTORY_SIGNATURE, END_OF_CENTRAL_DIRECTORY_SIGNATURE}:
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
            }
        )
        offset = data_end
    return headers


def audit_pr91_hpm1_readiness(
    *,
    archive: str | Path = DEFAULT_PR91_ARCHIVE,
    runtime_source_dir: str | Path = DEFAULT_PR91_RUNTIME_SOURCE_DIR,
) -> dict[str, Any]:
    """Return a deterministic non-dispatchable readiness report for PR91/HPM1.

    This audit only validates byte custody and records the known remaining
    gates. It does not decode the HPM1 stream, load scorers, or dispatch remote
    work.
    """

    archive_path = Path(archive)
    runtime_dir = Path(runtime_source_dir)
    gates: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []

    archive_record = _expected_file_record(
        archive_path,
        expected_bytes=EXPECTED_PR91_ARCHIVE_BYTES,
        expected_sha256=EXPECTED_PR91_ARCHIVE_SHA256,
    )
    gates["static_archive_custody"] = _gate(
        passed=bool(archive_record["matches_expected"]),
        reason="archive bytes and sha256 match public PR91 custody"
        if archive_record["matches_expected"]
        else "archive missing or does not match public PR91 custody",
        failed_status="missing" if not archive_record["exists"] else "failed_closed",
    )

    member_x, zip_report = _read_single_x_member(archive_path)
    member_x_record = {
        "name": "x",
        "expected_bytes": EXPECTED_PR91_MEMBER_X_BYTES,
        "expected_sha256": EXPECTED_PR91_MEMBER_X_SHA256,
        "exists": member_x is not None,
        "bytes": len(member_x) if member_x is not None else None,
        "sha256": sha256_bytes(member_x) if member_x is not None else "",
        "matches_expected": member_x is not None
        and len(member_x) == EXPECTED_PR91_MEMBER_X_BYTES
        and sha256_bytes(member_x) == EXPECTED_PR91_MEMBER_X_SHA256,
        "zip_report": zip_report,
    }
    gates["member_x_custody"] = _gate(
        passed=bool(member_x_record["matches_expected"]),
        reason="single ZIP member x matches public PR91 custody"
        if member_x_record["matches_expected"]
        else "archive must contain exactly one byte-matching member named x",
        failed_status="missing" if member_x is None else "failed_closed",
    )
    wire_contract_passed = (
        isinstance(zip_report.get("wire_contract"), dict)
        and zip_report["wire_contract"].get("passed") is True
    )
    gates["zip_wire_contract"] = _gate(
        passed=wire_contract_passed,
        reason="ZIP central directory and local file headers agree"
        if wire_contract_passed
        else "ZIP central directory and local file headers must agree",
        failed_status="missing" if member_x is None else "failed_closed",
    )

    mask_segment: bytes | None = None
    bundle_error = ""
    if member_x is not None:
        try:
            bundle = parse_pr85_bundle(member_x)
            mask_segment = bytes(bundle.segments["mask"])
        except (KeyError, Pr85BundleError, ValueError) as exc:
            bundle_error = f"{type(exc).__name__}: {exc}"

    hpm1_record: dict[str, Any] = {
        "expected_bytes": EXPECTED_PR91_HPM1_MASK_BYTES,
        "expected_sha256": EXPECTED_PR91_HPM1_MASK_SHA256,
        "exists": mask_segment is not None,
        "bytes": len(mask_segment) if mask_segment is not None else None,
        "sha256": sha256_bytes(mask_segment) if mask_segment is not None else "",
        "matches_expected": mask_segment is not None
        and len(mask_segment) == EXPECTED_PR91_HPM1_MASK_BYTES
        and sha256_bytes(mask_segment) == EXPECTED_PR91_HPM1_MASK_SHA256,
        "bundle_error": bundle_error,
    }
    if mask_segment is not None and not mask_segment.startswith(HPM1_MAGIC):
        hpm1_record["magic"] = mask_segment[:4].hex()
        hpm1_record["magic_matches_hpm1"] = False
    elif mask_segment is not None:
        hpm1_record["magic"] = HPM1_MAGIC.hex()
        hpm1_record["magic_matches_hpm1"] = True

    gates["hpm1_segment_custody"] = _gate(
        passed=bool(hpm1_record["matches_expected"]) and hpm1_record.get("magic_matches_hpm1") is True,
        reason="mask segment is byte-matching HPM1 payload"
        if hpm1_record["matches_expected"]
        else "HPM1 mask segment must match public PR91 custody",
        failed_status="missing" if mask_segment is None else "failed_closed",
    )

    hpm1_payload_record: dict[str, Any] = {
        "tokens_expected_sha256": EXPECTED_PR91_HPM1_TOKENS_SHA256,
        "hpac_expected_sha256": EXPECTED_PR91_HPM1_HPAC_SHA256,
        "tokens_sha256": "",
        "hpac_sha256": "",
        "config": {},
        "parse_error": "",
    }
    if mask_segment is not None and mask_segment.startswith(HPM1_MAGIC):
        try:
            payload = split_hpm1_mask_segment(mask_segment)
            hpm1_payload_record.update(
                {
                    "tokens_sha256": sha256_bytes(payload.tokens),
                    "hpac_sha256": sha256_bytes(payload.hpac),
                    "tokens_match_expected": sha256_bytes(payload.tokens)
                    == EXPECTED_PR91_HPM1_TOKENS_SHA256,
                    "hpac_matches_expected": sha256_bytes(payload.hpac)
                    == EXPECTED_PR91_HPM1_HPAC_SHA256,
                    "config": payload.config(),
                }
            )
        except (KeyError, Pr85BundleError, ValueError) as exc:
            hpm1_payload_record["parse_error"] = f"{type(exc).__name__}: {exc}"

    gates["hpm1_token_hpac_custody"] = _gate(
        passed=(
            hpm1_payload_record.get("tokens_match_expected") is True
            and hpm1_payload_record.get("hpac_matches_expected") is True
        ),
        reason="embedded HPM1 token stream and HPAC model match expected public custody"
        if hpm1_payload_record.get("tokens_match_expected") is True
        and hpm1_payload_record.get("hpac_matches_expected") is True
        else "embedded HPM1 token stream and HPAC model must match expected public custody",
        failed_status="missing" if not hpm1_payload_record["tokens_sha256"] else "failed_closed",
    )

    runtime_inventory = analyze_pr91_hpm1_runtime_sources(source_dir=runtime_dir)
    gates["runtime_source_inventory"] = _gate(
        passed=runtime_inventory.get("status") == "passed_static_source_inventory",
        reason="public PR91 runtime source inventory is present"
        if runtime_inventory.get("status") == "passed_static_source_inventory"
        else "public PR91 runtime source inventory missing",
        required_for_dispatch=False,
        failed_status="missing",
    )

    gates["full_hpm1_decode_600_frames"] = _gate(
        passed=False,
        reason="full 600-frame HPM1 probability/range decode is not recovered",
        failed_status="blocked",
    )
    gates["byte_exact_hpm1_reencode"] = _gate(
        passed=False,
        reason="byte-exact HPM1 re-encode is not recovered",
        failed_status="blocked",
    )
    gates["runtime_hpm1_loader_without_sidecars"] = _gate(
        passed=False,
        reason="contest inflate runtime has not proven HPM1 loading without uncharged sidecars or fallback",
        failed_status="blocked",
    )
    gates["exact_cuda_auth_eval_after_parity"] = _gate(
        passed=False,
        reason="exact CUDA auth eval is required after byte parity before any score claim",
        failed_status="blocked",
    )

    blockers = [
        name
        for name, gate in sorted(gates.items())
        if gate["required_for_dispatch"] and not gate["passed"]
    ]
    if member_x is not None and zip_report.get("duplicates"):
        warnings.append("duplicate ZIP members detected")

    ready = not blockers
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": ready,
        "promotion_eligible": False,
        "evidence_grade": "archive_readiness_audit" if ready else "static_custody_plus_blocked_replay",
        "source_archive": archive_record,
        "member_x": member_x_record,
        "hpm1_mask_segment": hpm1_record,
        "hpm1_payload": hpm1_payload_record,
        "runtime_source_inventory": runtime_inventory,
        "gates": gates,
        "dispatch_blockers": blockers,
        "warnings": warnings,
        "next_safe_actions": [
            "Recover full 600-frame HPM1 probability/range decode against the byte-matching PR91 payload.",
            "Prove byte-exact HPM1 decode/re-encode parity before mutating or stacking the stream.",
            "Wire the contest inflate runtime to consume HPM1 without uncharged sidecars or STBM/QMA9 fallback.",
            "Only then run exact CUDA auth eval through archive.zip -> inflate.sh -> upstream/evaluate.py.",
        ],
    }


__all__ = [
    "KIND",
    "SCHEMA_VERSION",
    "audit_pr91_hpm1_readiness",
]
