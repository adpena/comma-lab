"""Byte-level endgame decision support for PR85-family archives.

This module is local and deterministic. It validates ZIP custody, slices
PR85-family single-member bundles, probes cheap codec contracts, and estimates
rate-only transplant deltas against a named frontier archive. It never inflates
contest videos, loads scorers, dispatches jobs, or makes score claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence

from tac.archive_byte_profile import contest_rate_term
from tac.pr85_bundle import (
    HPM1_MAGIC,
    PR85_HEADERLESS_RANDMULTI_SPECS,
    Pr85BundleError,
    SEGMENT_ORDER,
    parse_hpm1_mask_segment,
    parse_pr85_bundle,
)


SCHEMA = "endgame_archive_decision_profile_v1"
TOOL = "experiments/profile_endgame_archive_decision.py"
EVIDENCE_GRADE = "byte_level_decision_support_only"
LOCAL_FILE_HEADER = 0x04034B50
QMA9_HEADER_BYTES = 20
STBM1BR_MAGIC = b"STBM1BR\0"
RMB1_MAGIC = b"RMB1"
RSB1_MAGIC = b"RSB1"


class EndgameArchiveDecisionError(ValueError):
    """Raised when endgame archive profiling inputs are malformed."""


@dataclass(frozen=True)
class _MemberPayload:
    name: str
    occurrence: int
    data: bytes
    row: dict[str, Any]


@dataclass(frozen=True)
class _ArchiveProfile:
    report: dict[str, Any]
    segments_by_name: dict[str, dict[str, Any]]


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _byte_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    total = len(data)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return round(entropy, 12)


def _display_ascii(data: bytes, *, limit: int = 8) -> str:
    return "".join(chr(value) if 32 <= value < 127 else "." for value in data[:limit])


def _safe_member_blockers(name: str) -> list[str]:
    blockers: list[str] = []
    if not name:
        blockers.append("empty_member_name")
    if "\x00" in name:
        blockers.append("nul_in_member_name")
    if "\\" in name:
        blockers.append("backslash_in_member_name")
    posix = PurePosixPath(name)
    windows = PureWindowsPath(name)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        blockers.append("absolute_or_drive_member_path")
    if any(part in ("", ".", "..") for part in posix.parts):
        blockers.append("zip_slip_member_path")
    if any(part in {"__MACOSX", ".DS_Store"} or part.startswith("._") for part in posix.parts):
        blockers.append("hidden_or_resource_fork_member")
    return blockers


def _decode_zip_name(raw: bytes, flag_bits: int) -> str:
    encoding = "utf-8" if flag_bits & 0x800 else "cp437"
    return raw.decode(encoding)


def _local_header_name(path: Path, info: zipfile.ZipInfo) -> str:
    with path.open("rb") as handle:
        handle.seek(info.header_offset)
        header = handle.read(30)
        if len(header) != 30:
            raise EndgameArchiveDecisionError(
                f"truncated local header for {info.filename!r} at {info.header_offset}"
            )
        (
            signature,
            _version_needed,
            _flag_bits,
            _compress_type,
            _mod_time,
            _mod_date,
            _crc,
            _compress_size,
            _file_size,
            name_len,
            extra_len,
        ) = struct.unpack("<IHHHHHIIIHH", header)
        if signature != LOCAL_FILE_HEADER:
            raise EndgameArchiveDecisionError(
                f"bad local header signature for {info.filename!r} at {info.header_offset}"
            )
        raw_name = handle.read(name_len)
        if len(raw_name) != name_len:
            raise EndgameArchiveDecisionError(f"truncated local filename for {info.filename!r}")
        handle.seek(extra_len, 1)
    return _decode_zip_name(raw_name, info.flag_bits)


def _import_brotli() -> Any:
    try:
        import brotli
    except ImportError as exc:  # pragma: no cover - environment guard
        raise EndgameArchiveDecisionError("brotli package is required for codec validation") from exc
    return brotli


def _brotli_probe(
    data: bytes,
    *,
    expected_magics: tuple[bytes, ...] = (),
    decoded_size: int | None = None,
) -> dict[str, Any]:
    try:
        brotli = _import_brotli()
        decoded = brotli.decompress(data)
    except Exception as exc:
        return {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "expected_magics": [magic.decode("ascii", errors="replace") for magic in expected_magics],
        }
    magic_match = True if not expected_magics else decoded.startswith(expected_magics)
    size_match = True if decoded_size is None else len(decoded) == decoded_size
    status = "ok" if magic_match and size_match else "failed"
    blockers: list[str] = []
    if not magic_match:
        blockers.append("decoded_magic_mismatch")
    if not size_match:
        blockers.append("decoded_size_mismatch")
    return {
        "status": status,
        "decoded_bytes": len(decoded),
        "decoded_sha256": _sha256_bytes(decoded),
        "decoded_magic_hex": decoded[:8].hex(),
        "decoded_magic_ascii": _display_ascii(decoded),
        "expected_magics": [magic.decode("ascii", errors="replace") for magic in expected_magics],
        "expected_magic_match": magic_match,
        "expected_decoded_bytes": decoded_size,
        "decoded_size_match": size_match,
        "blockers": blockers,
    }


def _qma9_probe(data: bytes) -> dict[str, Any]:
    if len(data) < QMA9_HEADER_BYTES or not data.startswith(b"QMA9"):
        return {"status": "failed", "blockers": ["bad_qma9_magic_or_header"]}
    frames, width, height, bitstream_bytes = struct.unpack_from("<IIII", data, 4)
    payload_bytes = len(data) - QMA9_HEADER_BYTES
    blockers: list[str] = []
    if frames <= 0 or width <= 0 or height <= 0:
        blockers.append("nonpositive_qma9_shape")
    if bitstream_bytes != payload_bytes:
        blockers.append("qma9_bitstream_length_mismatch")
    return {
        "status": "ok" if not blockers else "failed",
        "frames": int(frames),
        "width": int(width),
        "height": int(height),
        "header_bytes": QMA9_HEADER_BYTES,
        "declared_bitstream_bytes": int(bitstream_bytes),
        "actual_bitstream_bytes": int(payload_bytes),
        "bitstream_sha256": _sha256_bytes(data[QMA9_HEADER_BYTES:]),
        "blockers": blockers,
    }


def _hpm1_probe(data: bytes) -> dict[str, Any]:
    try:
        contract = parse_hpm1_mask_segment(data)
    except Pr85BundleError as exc:
        return {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
    return {"status": "ok", **dict(contract.metadata)}


def _stbm1br_probe(data: bytes) -> dict[str, Any]:
    try:
        from tac.stbm1br_mask_codec import STBM1BRError, parse_stbm1br_metadata
    except ImportError as exc:  # pragma: no cover - environment guard
        return {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
    try:
        meta = parse_stbm1br_metadata(data)
    except STBM1BRError as exc:
        return {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
    return {
        "status": "ok",
        "brotli_body_bytes": meta.brotli_body_bytes,
        "brotli_body_sha256": meta.brotli_body_sha256,
        "qtbm_blob_bytes": meta.qtbm_blob_bytes,
        "qtbm_blob_sha256": meta.qtbm_blob_sha256,
        "qtbm_magic": meta.qtbm_magic,
        "n_pairs": meta.n_pairs,
        "height": meta.height,
        "width": meta.width,
        "precision": meta.precision,
        "top_payload_bytes": meta.top_payload_bytes,
        "road_payload_bytes": meta.road_payload_bytes,
        "spatial_table_bytes": meta.spatial_table_bytes,
        "m5_table_bytes": meta.m5_table_bytes,
        "sparse_table_bytes": meta.sparse_table_bytes,
        "residual_bitstream_bytes": meta.residual_bitstream_bytes,
    }


def _parse_rmb1(data: bytes) -> dict[str, Any]:
    if len(data) < 6 or not data.startswith(RMB1_MAGIC):
        return {"status": "failed", "blockers": ["bad_rmb1_magic_or_header"]}
    mask_len = int.from_bytes(data[4:6], "little")
    if 6 + mask_len > len(data):
        return {"status": "failed", "blockers": ["rmb1_mask_brotli_overruns_segment"]}
    mask_br = data[6 : 6 + mask_len]
    vals_br = data[6 + mask_len :]
    try:
        brotli = _import_brotli()
        mask = brotli.decompress(mask_br)
        vals = brotli.decompress(vals_br)
    except Exception as exc:
        return {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
    blockers: list[str] = []
    if len(mask) % 75:
        blockers.append("rmb1_mask_length_not_row_aligned")
    rows = len(mask) // 75 if len(mask) % 75 == 0 else None
    expected_rows = sum(spec[3] for spec in PR85_HEADERLESS_RANDMULTI_SPECS)
    selected = 0
    if rows is not None:
        for row_start in range(0, len(mask), 75):
            for byte_i, byte in enumerate(mask[row_start : row_start + 75]):
                for bit in range(8):
                    frame_i = byte_i * 8 + bit
                    if frame_i >= 600:
                        break
                    selected += (byte >> bit) & 1
        if rows != expected_rows:
            blockers.append("rmb1_row_count_mismatch")
        if selected != len(vals):
            blockers.append("rmb1_value_count_mismatch")
    return {
        "status": "ok" if not blockers else "failed",
        "mask_brotli_bytes": len(mask_br),
        "values_brotli_bytes": len(vals_br),
        "mask_raw_bytes": len(mask),
        "values_raw_bytes": len(vals),
        "row_count": rows,
        "expected_row_count": expected_rows,
        "selected_value_count": int(selected),
        "mask_raw_sha256": _sha256_bytes(mask),
        "values_raw_sha256": _sha256_bytes(vals),
        "blockers": blockers,
    }


def _parse_rsb1(data: bytes) -> dict[str, Any]:
    if len(data) < 8 or not data.startswith(RSB1_MAGIC):
        return {"status": "failed", "blockers": ["bad_rsb1_magic_or_header"]}
    count = int.from_bytes(data[4:6], "little")
    table_id = int(data[6])
    reserved = int(data[7])
    try:
        brotli = _import_brotli()
        raw = brotli.decompress(data[8:])
    except Exception as exc:
        return {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
    hist = Counter(raw)
    blockers: list[str] = []
    if len(raw) != count:
        blockers.append("rsb1_action_count_mismatch")
    return {
        "status": "ok" if not blockers else "failed",
        "count": count,
        "table_id": table_id,
        "reserved": reserved,
        "brotli_body_bytes": len(data) - 8,
        "raw_action_bytes": len(raw),
        "raw_action_sha256": _sha256_bytes(raw),
        "unique_action_count": len(hist),
        "top_actions": [
            {"action": int(action), "count": int(n)}
            for action, n in sorted(hist.items(), key=lambda item: (-item[1], item[0]))[:12]
        ],
        "blockers": blockers,
    }


def _codec_label(segment_name: str, data: bytes) -> str:
    if segment_name == "mask":
        if data.startswith(b"QMA9"):
            return "QMA9_range_mask"
        if data.startswith(STBM1BR_MAGIC):
            return "STBM1BR_lossless_mask_recode"
        if data.startswith(HPM1_MAGIC):
            return "HPM1_hpac_mask"
    if segment_name == "randmulti" and data.startswith(RMB1_MAGIC):
        return "RMB1_bitmask_randmulti"
    if segment_name == "model":
        return "brotli_qh_model"
    if segment_name == "pose":
        return "brotli_p1d1_pose"
    if segment_name in {"post", "shift", "frac", "frac2", "frac3", "bias", "region", "randmulti"}:
        return "brotli_pr85_sidechannel"
    return "opaque"


def _segment_validation(name: str, data: bytes) -> dict[str, Any]:
    if name == "mask" and data.startswith(b"QMA9"):
        return _qma9_probe(data)
    if name == "mask" and data.startswith(HPM1_MAGIC):
        return _hpm1_probe(data)
    if name == "mask" and data.startswith(STBM1BR_MAGIC):
        return _stbm1br_probe(data)
    if name == "model":
        return _brotli_probe(data, expected_magics=(b"QH0", b"QH1"))
    if name == "pose":
        return _brotli_probe(data, expected_magics=(b"P1D1",))
    if name == "post":
        return _brotli_probe(data, decoded_size=2400)
    if name == "shift":
        return _brotli_probe(data, expected_magics=(b"SD4",))
    if name == "frac":
        return _brotli_probe(data, expected_magics=(b"FV1",))
    if name == "frac2":
        return _brotli_probe(data, expected_magics=(b"FH2",))
    if name == "frac3":
        return _brotli_probe(data, expected_magics=(b"FD3",))
    if name == "bias":
        return _brotli_probe(data, expected_magics=(b"BD1",))
    if name == "region":
        return _brotli_probe(data, expected_magics=(b"RH1",))
    if name == "randmulti" and data.startswith(RMB1_MAGIC):
        return _parse_rmb1(data)
    if name == "randmulti":
        return _brotli_probe(data)
    if data.startswith(RSB1_MAGIC):
        return _parse_rsb1(data)
    return {"status": "unvalidated", "reason": "no cheap local contract for payload"}


def _segment_row(name: str, data: bytes, offset: int) -> dict[str, Any]:
    validation = _segment_validation(name, data)
    return {
        "name": name,
        "bytes": len(data),
        "sha256": _sha256_bytes(data),
        "offset": int(offset),
        "magic_hex": data[:8].hex(),
        "magic_ascii": _display_ascii(data),
        "entropy_bits_per_byte": _byte_entropy(data),
        "codec": _codec_label(name, data),
        "validation": validation,
    }


def _side_member_row(member: _MemberPayload) -> dict[str, Any]:
    data = member.data
    validation = _segment_validation(member.name, data)
    codec = "RSB1_router_side_actions" if data.startswith(RSB1_MAGIC) else "opaque_side_member"
    return {
        "name": member.name,
        "occurrence": member.occurrence,
        "bytes": len(data),
        "sha256": _sha256_bytes(data),
        "magic_hex": data[:8].hex(),
        "magic_ascii": _display_ascii(data),
        "entropy_bits_per_byte": _byte_entropy(data),
        "codec": codec,
        "validation": validation,
    }


def _member_payloads(path: Path) -> tuple[list[_MemberPayload], list[dict[str, Any]], list[str]]:
    blockers: list[str] = []
    rows: list[dict[str, Any]] = []
    payloads: list[_MemberPayload] = []
    with zipfile.ZipFile(path, "r") as zf:
        infos = zf.infolist()
        if not infos:
            blockers.append("zip_has_no_members")
        name_counts = Counter(info.filename for info in infos)
        for name, count in sorted(name_counts.items()):
            if count > 1:
                blockers.append(f"duplicate_member_name:{name}:{count}")
        seen: Counter[str] = Counter()
        for info in infos:
            seen[info.filename] += 1
            occurrence = int(seen[info.filename])
            local_name = _local_header_name(path, info)
            row_blockers = _safe_member_blockers(info.filename)
            if local_name != info.filename:
                row_blockers.append("central_local_name_mismatch")
            if info.is_dir():
                row_blockers.append("directory_member")
            data = b""
            if not info.is_dir() and local_name == info.filename:
                data = zf.read(info)
            row = {
                "name": info.filename,
                "occurrence": occurrence,
                "local_header_name": local_name,
                "central_local_name_match": local_name == info.filename,
                "bytes": int(info.file_size),
                "compressed_bytes": int(info.compress_size),
                "method_id": int(info.compress_type),
                "header_offset": int(info.header_offset),
                "crc32": f"{info.CRC:08x}",
                "sha256": None if info.is_dir() or local_name != info.filename else _sha256_bytes(data),
                "blockers": row_blockers,
            }
            rows.append(row)
            blockers.extend(f"{info.filename}:{blocker}" for blocker in row_blockers)
            if not info.is_dir() and local_name == info.filename:
                payloads.append(_MemberPayload(info.filename, occurrence, data, row))
    return payloads, rows, blockers


def _try_primary_bundle(member: _MemberPayload) -> tuple[dict[str, Any], dict[str, dict[str, Any]]] | None:
    try:
        bundle = parse_pr85_bundle(member.data)
    except Pr85BundleError:
        return None
    mask = bytes(bundle.segments["mask"])
    if not mask.startswith((b"QMA9", HPM1_MAGIC, STBM1BR_MAGIC)):
        return None
    segments = [
        _segment_row(name, bytes(bundle.segments[name]), bundle.segment_offsets[name])
        for name in SEGMENT_ORDER
    ]
    primary = {
        "name": member.name,
        "occurrence": member.occurrence,
        "bytes": len(member.data),
        "sha256": _sha256_bytes(member.data),
        "magic_hex": member.data[:8].hex(),
        "magic_ascii": _display_ascii(member.data),
        "bundle_format": bundle.format,
        "bundle_header_bytes": bundle.header_bytes,
        "segment_lengths": bundle.segment_lengths,
        "segments": segments,
    }
    return primary, {row["name"]: row for row in segments}


def _profile_one_archive(path: Path, *, label: str) -> _ArchiveProfile:
    if not path.is_file():
        raise FileNotFoundError(f"archive not found: {path}")
    try:
        member_payloads, member_rows, zip_blockers = _member_payloads(path)
    except zipfile.BadZipFile as exc:
        raise EndgameArchiveDecisionError(f"bad ZIP archive: {path}") from exc

    candidates: list[tuple[_MemberPayload, dict[str, Any], dict[str, dict[str, Any]]]] = []
    for member in member_payloads:
        parsed = _try_primary_bundle(member)
        if parsed is not None:
            candidates.append((member, parsed[0], parsed[1]))
    analysis_blockers: list[str] = []
    primary: dict[str, Any] | None = None
    primary_payload: _MemberPayload | None = None
    segments_by_name: dict[str, dict[str, Any]] = {}
    if not candidates:
        analysis_blockers.append("no_pr85_family_primary_member_detected")
    elif len(candidates) > 1:
        analysis_blockers.append("multiple_pr85_family_primary_members_detected")
        candidates.sort(key=lambda row: (-len(row[0].data), row[0].name, row[0].occurrence))
        primary_payload, primary, segments_by_name = candidates[0]
    else:
        primary_payload, primary, segments_by_name = candidates[0]

    for segment in segments_by_name.values():
        if segment["validation"].get("status") == "failed":
            analysis_blockers.append(f"segment_validation_failed:{segment['name']}")

    side_payloads = [
        member
        for member in member_payloads
        if primary_payload is None
        or member.name != primary_payload.name
        or member.occurrence != primary_payload.occurrence
    ]
    side_members = [_side_member_row(member) for member in sorted(side_payloads, key=lambda row: (row.name, row.occurrence))]
    for member in side_members:
        if member["validation"].get("status") == "failed":
            analysis_blockers.append(f"side_member_validation_failed:{member['name']}#{member['occurrence']}")
    side_bytes = sum(int(member["bytes"]) for member in side_members)
    primary_bytes = 0 if primary is None else int(primary["bytes"])
    archive_bytes = path.stat().st_size
    zip_payload_bytes = sum(int(row["compressed_bytes"]) for row in member_rows)

    report = {
        "label": label,
        "archive": {
            "path": str(path),
            "bytes": archive_bytes,
            "sha256": _sha256_file(path),
            "rate_term": contest_rate_term(archive_bytes),
        },
        "strict_zip": {
            "valid": not zip_blockers,
            "blockers": sorted(zip_blockers),
            "member_count": len(member_rows),
            "members": member_rows,
            "compressed_payload_bytes": zip_payload_bytes,
            "zip_overhead_bytes": archive_bytes - zip_payload_bytes,
        },
        "primary_member": primary,
        "side_info": {
            "members": side_members,
            "member_count": len(side_members),
            "charged_bytes": side_bytes,
            "rate_term": contest_rate_term(side_bytes),
            "requires_runtime_contract_review": bool(side_members),
        },
        "byte_accounting": {
            "primary_member_bytes": primary_bytes,
            "side_info_member_bytes": side_bytes,
            "archive_zip_overhead_bytes": archive_bytes - zip_payload_bytes,
        },
        "decision_support": {
            "valid_for_byte_decision": not zip_blockers and not analysis_blockers,
            "blockers": sorted(analysis_blockers),
        },
    }
    return _ArchiveProfile(report=report, segments_by_name=segments_by_name)


def _segment_delta_rows(
    candidate: Mapping[str, Mapping[str, Any]],
    frontier: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in SEGMENT_ORDER:
        c = candidate.get(name)
        f = frontier.get(name)
        rows.append(
            {
                "segment": name,
                "same_sha256": bool(c is not None and f is not None and c["sha256"] == f["sha256"]),
                "candidate_bytes": None if c is None else int(c["bytes"]),
                "frontier_bytes": None if f is None else int(f["bytes"]),
                "delta_bytes": None if c is None or f is None else int(c["bytes"]) - int(f["bytes"]),
                "candidate_codec": None if c is None else c.get("codec"),
                "frontier_codec": None if f is None else f.get("codec"),
                "candidate_sha256": None if c is None else c.get("sha256"),
                "frontier_sha256": None if f is None else f.get("sha256"),
                "candidate_validation_status": None
                if c is None
                else c.get("validation", {}).get("status"),
                "frontier_validation_status": None
                if f is None
                else f.get("validation", {}).get("status"),
            }
        )
    return rows


def _comparison(candidate: _ArchiveProfile, frontier: _ArchiveProfile) -> dict[str, Any]:
    cand_report = candidate.report
    front_report = frontier.report
    cand_archive_bytes = int(cand_report["archive"]["bytes"])
    front_archive_bytes = int(front_report["archive"]["bytes"])
    archive_delta = cand_archive_bytes - front_archive_bytes
    cand_primary = int(cand_report["byte_accounting"]["primary_member_bytes"])
    front_primary = int(front_report["byte_accounting"]["primary_member_bytes"])
    cand_side = int(cand_report["side_info"]["charged_bytes"])
    front_side = int(front_report["side_info"]["charged_bytes"])
    cand_overhead = int(cand_report["byte_accounting"]["archive_zip_overhead_bytes"])
    front_overhead = int(front_report["byte_accounting"]["archive_zip_overhead_bytes"])
    primary_delta = cand_primary - front_primary
    side_delta = cand_side - front_side
    overhead_delta = cand_overhead - front_overhead
    segment_rows = _segment_delta_rows(candidate.segments_by_name, frontier.segments_by_name)
    changed = [row for row in segment_rows if not row["same_sha256"]]

    transplant_estimates: list[dict[str, Any]] = []
    for row in changed:
        delta = row["delta_bytes"]
        if delta is None:
            continue
        requires_side = (
            row["segment"] == "randmulti"
            and cand_side > 0
            and str(row.get("candidate_codec", "")).startswith("RMB1")
        )
        estimate = int(delta)
        included_side_delta = 0
        included_overhead_delta = 0
        if requires_side:
            included_side_delta = side_delta
            included_overhead_delta = overhead_delta
            estimate += included_side_delta + included_overhead_delta
        runtime_blockers: list[str] = []
        if row["candidate_validation_status"] != "ok":
            runtime_blockers.append("candidate_segment_validation_not_ok")
        if row["segment"] == "mask" and row["candidate_codec"] != row["frontier_codec"]:
            runtime_blockers.append("mask_codec_runtime_or_parity_change_required")
        if requires_side:
            side_statuses = [
                member["validation"].get("status")
                for member in cand_report["side_info"]["members"]
            ]
            if any(status != "ok" for status in side_statuses):
                runtime_blockers.append("required_side_info_validation_not_ok")
        transplant_estimates.append(
            {
                "segment": row["segment"],
                "candidate_codec": row["candidate_codec"],
                "frontier_codec": row["frontier_codec"],
                "segment_delta_bytes": int(delta),
                "requires_candidate_side_info": requires_side,
                "included_side_info_delta_bytes": int(included_side_delta),
                "included_zip_overhead_delta_bytes": int(included_overhead_delta),
                "estimated_archive_delta_bytes": int(estimate),
                "estimated_rate_term_delta": contest_rate_term(estimate),
                "byte_positive": estimate < 0,
                "runtime_blockers": runtime_blockers,
                "dispatch_advice": "do_not_dispatch_rate_only"
                if estimate >= 0
                else "byte_positive_but_requires_contract_gates",
            }
        )
    transplant_estimates.sort(
        key=lambda row: (
            not bool(row["byte_positive"]),
            int(row["estimated_archive_delta_bytes"]),
            row["segment"],
        )
    )
    return {
        "candidate_label": cand_report["label"],
        "frontier_label": front_report["label"],
        "archive_delta_bytes": archive_delta,
        "rate_term_delta": contest_rate_term(archive_delta),
        "primary_member_delta_bytes": primary_delta,
        "side_info_delta_bytes": side_delta,
        "zip_overhead_delta_bytes": overhead_delta,
        "changed_segments": changed,
        "unchanged_segment_count": sum(1 for row in segment_rows if row["same_sha256"]),
        "transplant_estimates": transplant_estimates,
    }


def _rank_actions(comparisons: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for comp in comparisons:
        for estimate in comp["transplant_estimates"]:
            if estimate["byte_positive"]:
                actions.append(
                    {
                        "candidate_label": comp["candidate_label"],
                        "frontier_label": comp["frontier_label"],
                        "surface": estimate["segment"],
                        "estimated_archive_delta_bytes": estimate["estimated_archive_delta_bytes"],
                        "estimated_rate_term_delta": estimate["estimated_rate_term_delta"],
                        "blocked_by": estimate["runtime_blockers"],
                        "advice": estimate["dispatch_advice"],
                    }
                )
            elif estimate["requires_candidate_side_info"]:
                actions.append(
                    {
                        "candidate_label": comp["candidate_label"],
                        "frontier_label": comp["frontier_label"],
                        "surface": f"{estimate['segment']}+side_info",
                        "estimated_archive_delta_bytes": estimate["estimated_archive_delta_bytes"],
                        "estimated_rate_term_delta": estimate["estimated_rate_term_delta"],
                        "blocked_by": ["rate_only_not_byte_positive", *estimate["runtime_blockers"]],
                        "advice": estimate["dispatch_advice"],
                    }
                )
    actions.sort(
        key=lambda row: (
            row["estimated_archive_delta_bytes"] >= 0,
            int(row["estimated_archive_delta_bytes"]),
            row["candidate_label"],
            row["surface"],
        )
    )
    return actions


def _parse_label_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise EndgameArchiveDecisionError(f"expected LABEL=PATH, got {value!r}")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    raw_path = raw_path.strip()
    if not label or not raw_path:
        raise EndgameArchiveDecisionError(f"expected LABEL=PATH, got {value!r}")
    return label, Path(raw_path)


def build_endgame_decision_profile(
    candidates: Mapping[str, Path | str],
    *,
    frontier_label: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic byte-level decision profile for candidate archives."""

    if not candidates:
        raise EndgameArchiveDecisionError("at least one candidate archive is required")
    profiles = {
        label: _profile_one_archive(Path(path), label=label)
        for label, path in sorted(candidates.items())
    }
    if frontier_label is not None and frontier_label not in profiles:
        raise EndgameArchiveDecisionError(f"frontier label {frontier_label!r} was not profiled")
    if frontier_label is None:
        frontier_label = min(
            profiles,
            key=lambda label: int(profiles[label].report["archive"]["bytes"]),
        )
    frontier = profiles[frontier_label]
    comparisons = [
        _comparison(profile, frontier)
        for label, profile in sorted(profiles.items())
        if label != frontier_label
    ]
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": EVIDENCE_GRADE,
        "frontier_label": frontier_label,
        "notes": [
            "byte-only decision support; no inflate, scorer, remote dispatch, promotion, or score claim",
            "rate deltas use only the contest archive-byte term and must not be treated as exact score evidence",
            "exact score truth remains archive.zip -> inflate.sh -> upstream/evaluate.py on CUDA",
        ],
        "archives": [profiles[label].report for label in sorted(profiles)],
        "comparisons_to_frontier": comparisons,
        "ranked_actions": _rank_actions(comparisons),
    }


def render_markdown(profile: Mapping[str, Any]) -> str:
    lines = [
        "# Endgame Archive Decision Profile",
        "",
        f"- schema: `{profile['schema']}`",
        f"- evidence_grade: `{profile['evidence_grade']}`",
        f"- score_claim: `{profile['score_claim']}`",
        f"- frontier_label: `{profile['frontier_label']}`",
        "",
        "This report is byte-only. It does not inflate videos, load scorers, dispatch jobs, "
        "promote methods, or claim contest score.",
        "",
        "## Archives",
        "",
        "| label | bytes | sha256 | strict ZIP | decision-valid | side bytes |",
        "|---|---:|---|---|---|---:|",
    ]
    for archive in profile["archives"]:
        lines.append(
            f"| {archive['label']} | {archive['archive']['bytes']} | "
            f"{archive['archive']['sha256']} | {archive['strict_zip']['valid']} | "
            f"{archive['decision_support']['valid_for_byte_decision']} | "
            f"{archive['side_info']['charged_bytes']} |"
        )
    for archive in profile["archives"]:
        primary = archive.get("primary_member")
        if not primary:
            continue
        lines.extend(
            [
                "",
                f"## {archive['label']} Segments",
                "",
                "| segment | bytes | codec | validation | sha256 |",
                "|---|---:|---|---|---|",
            ]
        )
        for segment in primary["segments"]:
            lines.append(
                f"| {segment['name']} | {segment['bytes']} | {segment['codec']} | "
                f"{segment['validation'].get('status')} | {segment['sha256']} |"
            )
        if archive["side_info"]["members"]:
            lines.extend(["", "### Side Info", "", "| member | bytes | codec | validation | sha256 |", "|---|---:|---|---|---|"])
            for member in archive["side_info"]["members"]:
                lines.append(
                    f"| {member['name']} | {member['bytes']} | {member['codec']} | "
                    f"{member['validation'].get('status')} | {member['sha256']} |"
                )
    if profile["comparisons_to_frontier"]:
        lines.extend(["", "## Frontier Comparisons", ""])
        for comp in profile["comparisons_to_frontier"]:
            lines.extend(
                [
                    f"### {comp['candidate_label']} vs {comp['frontier_label']}",
                    "",
                    f"- archive delta bytes: `{comp['archive_delta_bytes']}`",
                    f"- primary member delta bytes: `{comp['primary_member_delta_bytes']}`",
                    f"- side-info delta bytes: `{comp['side_info_delta_bytes']}`",
                    f"- ZIP overhead delta bytes: `{comp['zip_overhead_delta_bytes']}`",
                    "",
                    "| segment | delta bytes | frontier codec | candidate codec | validation |",
                    "|---|---:|---|---|---|",
                ]
            )
            for row in comp["changed_segments"]:
                lines.append(
                    f"| {row['segment']} | {row['delta_bytes']} | {row['frontier_codec']} | "
                    f"{row['candidate_codec']} | {row['candidate_validation_status']} |"
                )
            if comp["transplant_estimates"]:
                lines.extend(["", "#### Transplant Estimates", "", "| surface | est. delta bytes | side info | advice |", "|---|---:|---|---|"])
                for estimate in comp["transplant_estimates"]:
                    lines.append(
                        f"| {estimate['segment']} | {estimate['estimated_archive_delta_bytes']} | "
                        f"{estimate['requires_candidate_side_info']} | {estimate['dispatch_advice']} |"
                    )
    if profile["ranked_actions"]:
        lines.extend(["", "## Ranked Actions", "", "| candidate | surface | est. delta bytes | advice | blockers |", "|---|---|---:|---|---|"])
        for action in profile["ranked_actions"]:
            blockers = ", ".join(action["blocked_by"])
            lines.append(
                f"| {action['candidate_label']} | {action['surface']} | "
                f"{action['estimated_archive_delta_bytes']} | {action['advice']} | {blockers} |"
            )
    return "\n".join(lines) + "\n"


def write_outputs(
    profile: Mapping[str, Any],
    *,
    json_out: Path | None = None,
    markdown_out: Path | None = None,
) -> None:
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(_json_text(profile), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_markdown(profile), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Candidate archive to profile. May be repeated.",
    )
    parser.add_argument(
        "--frontier-label",
        help="Label to use as the byte frontier for comparison. Defaults to the smallest archive.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    candidates = dict(_parse_label_path(value) for value in args.candidate)
    profile = build_endgame_decision_profile(candidates, frontier_label=args.frontier_label)
    write_outputs(profile, json_out=args.json_out, markdown_out=args.markdown_out)
    if args.json_out is None and args.markdown_out is None:
        print(_json_text(profile), end="")
    return 0


__all__ = [
    "EndgameArchiveDecisionError",
    "build_endgame_decision_profile",
    "render_markdown",
    "write_outputs",
    "main",
]
