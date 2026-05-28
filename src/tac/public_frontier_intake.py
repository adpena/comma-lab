# SPDX-License-Identifier: MIT
"""Static intake and byte diffing for public frontier archives.

This module is intentionally offline and byte-only. It validates ZIP custody,
identifies PR85-family bundle segments, records charged side-channel hashes,
and supports simple cross-baseline diffing for triage.

Recovered from prior bytecode custody and promoted into a maintained, tested
implementation on 2026-05-08. The profile remains deliberately score-neutral:
it emits ZIP custody, payload hashes, known PR85 segment contracts, and
baseline byte diffs, but never inflates videos or treats byte-only evidence as
score evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import zipfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from tac.optimization.archive_bound_candidate_contract import (
    ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA,
    ARCHIVE_BOUND_CANDIDATE_CONTRACT_SURFACE_SCHEMA,
    build_archive_bound_candidate_contract_surface,
)
from tac.pr85_bundle import Pr85BundleError, parse_pr85_bundle

SCHEMA = "public_frontier_archive_intake_v1"
TOOL = "experiments/profile_public_frontier_intake.py"
EVIDENCE_GRADE = "byte_intake_only"
ARCHIVE_CONTRACT_TRANSFORM_KIND = "public_frontier_archive_intake_zip_ordering"
PUBLIC_FRONTIER_RECEIVER_CONTRACT_KIND = "public_frontier_runtime_receiver_review"

_QUARANTINE_SPEC = (
    ".recovery_quarantine_20260505T004735Z/src/tac/public_frontier_intake.recovery_spec.json"
)


class PublicFrontierIntakeError(ValueError):
    """Raised when public frontier intake inputs are malformed."""

    pass


@dataclass(frozen=True)
class _ParsedArchive:
    path: str
    total_bytes: int
    sha256: str
    member_payloads: Sequence[Mapping[str, Any]]
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _LocalZipHeader:
    filename: str
    flag_bits: int
    compress_type: int
    mod_time: int
    mod_date: int
    crc32: int
    compressed_size: int
    uncompressed_size: int


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _byte_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    n = len(data)
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)


def _rehydration_failure(symbol: str) -> NotImplementedError:
    return NotImplementedError(
        f"rehydration incomplete: {symbol} contains intricate generator/lambda "
        f"closures pycdc cannot decompile; original bytecode preserved in "
        f"{_QUARANTINE_SPEC}"
    )


def _decode_zip_name(raw_name: bytes, flag_bits: int) -> str:
    encoding = "utf-8" if flag_bits & 0x800 else "cp437"
    return raw_name.decode(encoding, errors="replace")


def _strict_zip_report(raw: bytes, infos: Sequence[zipfile.ZipInfo]) -> dict[str, Any]:
    blockers: list[str] = []
    seen: set[str] = set()
    for info in infos:
        name = info.filename
        if name in seen:
            blockers.append(f"{name}:duplicate_member")
        seen.add(name)
        try:
            _validate_archive_member_name(name)
        except PublicFrontierIntakeError as exc:
            blockers.append(f"{name}:{exc}")
        if name.startswith("__MACOSX/") or PurePosixPath(name).name.startswith("._"):
            blockers.append(f"{name}:resource_fork_or_hidden_member")
        local_header = _local_header(raw, info.header_offset)
        if local_header is None:
            blockers.append(f"{name}:missing_or_bad_local_header")
        else:
            if local_header.filename != name:
                blockers.append(f"{name}:central_local_name_mismatch")
            blockers.extend(_local_central_header_mismatch_blockers(name, local_header, info))
    return {
        "valid": not blockers,
        "blockers": blockers,
        "member_count": len(infos),
    }


def _local_central_header_mismatch_blockers(
    name: str,
    local_header: _LocalZipHeader,
    central_header: zipfile.ZipInfo,
) -> list[str]:
    blockers: list[str] = []
    if local_header.compress_type != central_header.compress_type:
        blockers.append(f"{name}:central_local_compression_method_mismatch")
    if local_header.flag_bits != central_header.flag_bits:
        blockers.append(f"{name}:central_local_general_purpose_flags_mismatch")
    if local_header.mod_time != _dos_time_from_datetime(central_header.date_time):
        blockers.append(f"{name}:central_local_mod_time_mismatch")
    if local_header.mod_date != _dos_date_from_datetime(central_header.date_time):
        blockers.append(f"{name}:central_local_mod_date_mismatch")
    if (local_header.flag_bits | central_header.flag_bits) & 0x08:
        return blockers
    if local_header.crc32 != central_header.CRC:
        blockers.append(f"{name}:central_local_crc32_mismatch")
    if (
        local_header.compressed_size != 0xFFFFFFFF
        and local_header.compressed_size != central_header.compress_size
    ):
        blockers.append(f"{name}:central_local_compressed_size_mismatch")
    if (
        local_header.uncompressed_size != 0xFFFFFFFF
        and local_header.uncompressed_size != central_header.file_size
    ):
        blockers.append(f"{name}:central_local_uncompressed_size_mismatch")
    return blockers


def _local_header(raw: bytes, offset: int) -> _LocalZipHeader | None:
    if offset < 0 or offset + 30 > len(raw):
        return None
    try:
        (
            signature,
            _version,
            flag_bits,
            method,
            mod_time,
            mod_date,
            crc,
            compressed,
            uncompressed,
            name_len,
            extra_len,
        ) = struct.unpack("<IHHHHHIIIHH", raw[offset : offset + 30])
    except struct.error:
        return None
    if signature != 0x04034B50:
        return None
    name_start = offset + 30
    name_end = name_start + name_len
    extra_end = name_end + extra_len
    if name_end > len(raw) or extra_end > len(raw):
        return None
    return _LocalZipHeader(
        filename=_decode_zip_name(raw[name_start:name_end], flag_bits),
        flag_bits=flag_bits,
        compress_type=method,
        mod_time=mod_time,
        mod_date=mod_date,
        crc32=crc,
        compressed_size=compressed,
        uncompressed_size=uncompressed,
    )


def _dos_date_from_datetime(date_time: tuple[int, int, int, int, int, int]) -> int:
    year, month, day, _hour, _minute, _second = date_time
    return ((year - 1980) << 9) | (month << 5) | day


def _dos_time_from_datetime(date_time: tuple[int, int, int, int, int, int]) -> int:
    _year, _month, _day, hour, minute, second = date_time
    return (hour << 11) | (minute << 5) | (second // 2)


def _validate_archive_member_name(name: str) -> str:
    if not name:
        raise PublicFrontierIntakeError("empty_member_name")
    if "\x00" in name:
        raise PublicFrontierIntakeError("nul_byte_in_member_name")
    if "\\" in name:
        raise PublicFrontierIntakeError("backslash_member_name")
    posix_path = PurePosixPath(name)
    if posix_path.is_absolute() or ".." in posix_path.parts:
        raise PublicFrontierIntakeError("zip_slip_member_name")
    return name


def _zip_method_name(compress_type: int) -> str:
    names = {
        zipfile.ZIP_STORED: "stored",
        zipfile.ZIP_DEFLATED: "deflated",
    }
    if hasattr(zipfile, "ZIP_BZIP2"):
        names[zipfile.ZIP_BZIP2] = "bzip2"
    if hasattr(zipfile, "ZIP_LZMA"):
        names[zipfile.ZIP_LZMA] = "lzma"
    return names.get(compress_type, f"unknown_{compress_type}")


def _zip_members(zf: zipfile.ZipFile, infos: Sequence[zipfile.ZipInfo]) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    for info in infos:
        payload = b""
        read_error: str | None = None
        try:
            payload = zf.read(info.filename)
        except zipfile.BadZipFile as exc:
            read_error = str(exc)
        members.append(
            {
                "name": info.filename,
                "compressed_bytes": int(info.compress_size),
                "uncompressed_bytes": int(info.file_size),
                "compression_method": _zip_method_name(info.compress_type),
                "crc32": f"{info.CRC:08x}",
                "sha256": _sha256_bytes(payload),
                "payload": payload,
                "read_error": read_error,
            }
        )
    return members


def _primary_member(members: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not members:
        return {
            "name": "",
            "compressed_bytes": 0,
            "uncompressed_bytes": 0,
            "compression_method": "",
            "crc32": "",
            "sha256": "",
            "payload": b"",
            "read_error": "empty_archive",
        }
    preferred = {"x", "0.bin", "p", "renderer_payload.bin", "renderer_payload.bin.br"}
    for name in preferred:
        for member in members:
            if member["name"] == name:
                return member
    return max(members, key=lambda item: int(item.get("compressed_bytes", 0)))


def _iter_baselines(
    baselines: Mapping[str, Path | str] | Sequence[Path | str] | None,
) -> Iterable[tuple[str, Path]]:
    if baselines is None:
        return []
    if isinstance(baselines, Mapping):
        return [(str(label), Path(path)) for label, path in baselines.items()]
    return [(Path(path).stem, Path(path)) for path in baselines]


def _codec_for_segment(name: str, payload: bytes) -> str:
    if name == "randmulti":
        if payload.startswith(b"RMB1"):
            return "RMB1_side_info_backed_randmulti"
        return "opaque_pr85_segment"
    if name == "mask" and payload.startswith(b"QMA9"):
        return "QMA9_mask"
    if name == "mask" and payload.startswith(b"HPM1"):
        return "HPM1_mask"
    if name == "model":
        return "brotli_qh_model"
    if name == "pose":
        return "brotli_p1d1_pose"
    return "opaque_pr85_segment"


def _profile_pr85_segments(payload: bytes) -> dict[str, Any]:
    try:
        bundle = parse_pr85_bundle(payload)
    except Pr85BundleError as exc:
        return {
            "status": "not_pr85_bundle",
            "error": str(exc),
            "segments": [],
        }
    segments = [
        {
            "segment": name,
            "offset": int(bundle.segment_offsets[name]),
            "bytes": len(bytes(segment)),
            "sha256": _sha256_bytes(bytes(segment)),
            "magic": bytes(segment[:4]).hex(),
            "codec": _codec_for_segment(name, bytes(segment)),
        }
        for name, segment in bundle.segments.items()
    ]
    return {
        "status": "parsed_pr85_bundle",
        "format": bundle.format,
        "header_bytes": bundle.header_bytes,
        "segments": segments,
    }


def _baseline_primary(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    with zipfile.ZipFile(path) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        strict = _strict_zip_report(raw, infos)
        members = _zip_members(zf, infos)
    primary = _primary_member(members)
    return {
        "strict_zip": strict,
        "primary_member": primary,
        "archive": {
            "path": path.as_posix(),
            "bytes": len(raw),
            "sha256": _sha256_bytes(raw),
        },
    }


def _segment_index(profile: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not profile or profile.get("status") != "parsed_pr85_bundle":
        return {}
    return {
        str(row["segment"]): dict(row)
        for row in profile.get("segments", [])
        if isinstance(row, dict) and row.get("segment")
    }


def _baseline_diff(
    candidate_primary: dict[str, Any],
    baseline_label: str,
    baseline_path: Path,
    *,
    inspect_segments: bool,
) -> dict[str, Any]:
    try:
        baseline = _baseline_primary(baseline_path)
    except (OSError, zipfile.BadZipFile) as exc:
        return {
            "baseline_label": baseline_label,
            "baseline_path": baseline_path.as_posix(),
            "status": "baseline_unreadable",
            "error": str(exc),
            "changed_segments": [],
        }
    candidate_profile = (
        _profile_pr85_segments(candidate_primary["payload"])
        if inspect_segments and isinstance(candidate_primary.get("payload"), bytes)
        else None
    )
    baseline_primary = baseline["primary_member"]
    baseline_profile = (
        _profile_pr85_segments(baseline_primary["payload"])
        if inspect_segments and isinstance(baseline_primary.get("payload"), bytes)
        else None
    )
    candidate_segments = _segment_index(candidate_profile)
    baseline_segments = _segment_index(baseline_profile)
    changed_segments: list[dict[str, Any]] = []
    for name in sorted(set(candidate_segments) | set(baseline_segments)):
        candidate = candidate_segments.get(name)
        prior = baseline_segments.get(name)
        if not candidate or not prior or candidate.get("sha256") != prior.get("sha256"):
            changed_segments.append(
                {
                    "segment": name,
                    "baseline_bytes": prior.get("bytes") if prior else None,
                    "candidate_bytes": candidate.get("bytes") if candidate else None,
                    "baseline_sha256": prior.get("sha256") if prior else None,
                    "candidate_sha256": candidate.get("sha256") if candidate else None,
                    "baseline_codec": prior.get("codec") if prior else None,
                    "candidate_codec": candidate.get("codec") if candidate else None,
                }
            )
    return {
        "baseline_label": baseline_label,
        "baseline_path": baseline_path.as_posix(),
        "status": "compared_pr85_segments" if candidate_segments and baseline_segments else "compared_archive_only",
        "baseline_archive": baseline["archive"],
        "baseline_primary_member": {
            key: value for key, value in baseline_primary.items() if key != "payload"
        },
        "candidate_primary_member": {
            key: value for key, value in candidate_primary.items() if key != "payload"
        },
        "changed_segments": changed_segments,
    }


def profile_public_frontier_archive(
    archive: Path | str, *,
    label: str | None = None,
    baselines: Mapping[str, Path | str] | Sequence[Path | str] | None = None,
    inspect_segments: bool = True,
) -> dict[str, Any]:
    """Profile one public frontier archive byte-by-byte.

    The profiler is score-neutral: it reads ZIP/container bytes, hashes members,
    slices known PR85-family single-member bundles, and compares optional
    baselines. It does not run inflate, load scorers, or infer score movement.
    """

    archive_path = Path(archive)
    try:
        raw = archive_path.read_bytes()
    except OSError as exc:
        raise PublicFrontierIntakeError(f"cannot read archive: {archive_path}") from exc

    try:
        with zipfile.ZipFile(archive_path) as zf:
            infos = [info for info in zf.infolist() if not info.is_dir()]
            strict_zip = _strict_zip_report(raw, infos)
            members = _zip_members(zf, infos)
    except zipfile.BadZipFile as exc:
        raise PublicFrontierIntakeError(f"bad ZIP archive: {archive_path}") from exc
    except RuntimeError as exc:
        raise PublicFrontierIntakeError(f"cannot read ZIP members: {archive_path}") from exc

    primary = _primary_member(members)
    side_members = [member for member in members if member["name"] != primary.get("name")]
    side_bytes = sum(int(member["compressed_bytes"]) for member in side_members)
    segment_profile = (
        _profile_pr85_segments(primary["payload"])
        if inspect_segments and isinstance(primary.get("payload"), bytes)
        else None
    )
    baseline_diffs = [
        _baseline_diff(primary, baseline_label, baseline_path, inspect_segments=inspect_segments)
        for baseline_label, baseline_path in _iter_baselines(baselines)
    ]
    payload_members = [
        {key: value for key, value in member.items() if key != "payload"}
        for member in members
    ]
    primary_public = {key: value for key, value in primary.items() if key != "payload"}
    archive_record = {
        "path": archive_path.as_posix(),
        "bytes": len(raw),
        "sha256": _sha256_bytes(raw),
    }
    archive_bound_surface = _archive_bound_candidate_contract_surface(
        archive=archive_record,
        strict_zip=strict_zip,
        side_info_member_count=len(side_members),
        baseline_diffs=baseline_diffs,
    )
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "evidence_grade": EVIDENCE_GRADE,
        "score_claim": False,
        "label": label or archive_path.stem,
        "archive": archive_record,
        "strict_zip": strict_zip,
        "member_count": len(members),
        "members": payload_members,
        "primary_member": primary_public,
        "side_info": {
            "member_count": len(side_members),
            "charged_bytes": side_bytes,
            "members": [
                {key: value for key, value in member.items() if key != "payload"}
                for member in side_members
            ],
            "requires_runtime_contract_review": bool(side_members),
        },
        "segment_profile": segment_profile,
        "baseline_diffs": baseline_diffs,
        "archive_bound_candidate_contract_schema": (
            ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA
        ),
        "archive_bound_candidate_contract": archive_bound_surface[
            "selected_candidate_contract"
        ],
        "archive_bound_candidate_contract_surface_schema": (
            ARCHIVE_BOUND_CANDIDATE_CONTRACT_SURFACE_SCHEMA
        ),
        "archive_bound_candidate_contract_surface": archive_bound_surface,
        "promotion_eligible": False,
        "dispatchable": False,
        "blockers": [
            "byte_intake_only_not_score_evidence",
            "requires_exact_cuda_auth_eval_for_score_claim",
        ],
    }


def _archive_bound_candidate_contract_surface(
    *,
    archive: Mapping[str, Any],
    strict_zip: Mapping[str, Any],
    side_info_member_count: int,
    baseline_diffs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    changed_segments = sum(
        len(diff.get("changed_segments") or [])
        for diff in baseline_diffs
        if isinstance(diff, Mapping)
    )
    blockers = [
        "public_frontier_byte_intake_requires_receiver_adapter_proof",
        "public_frontier_byte_intake_requires_exact_axis_replay",
    ]
    if strict_zip.get("valid") is not True:
        blockers.append("public_frontier_strict_zip_blockers_present")
    if side_info_member_count:
        blockers.append("public_frontier_side_info_requires_runtime_contract_review")
    return build_archive_bound_candidate_contract_surface(
        candidates=[
            {
                "archive_native_transform_kind": ARCHIVE_CONTRACT_TRANSFORM_KIND,
                "materialized": True,
                "path": archive.get("path"),
                "sha256": archive.get("sha256"),
                "bytes": archive.get("bytes"),
                "runtime_consumption_proof_ready": False,
                "receiver_contract_kind": PUBLIC_FRONTIER_RECEIVER_CONTRACT_KIND,
                "receiver_contract_satisfied": False,
                "semantic_payload_changed": changed_segments > 0,
                "score_affecting_payload_changed": changed_segments > 0,
                "exact_axis_score_affecting_adjudication_required": True,
                "charged_bits_changed": changed_segments > 0,
                "blockers": blockers,
            }
        ],
        selected_transform_kind=ARCHIVE_CONTRACT_TRANSFORM_KIND,
        family_id="public_frontier_archive_intake",
        typed_response_id=str(archive.get("sha256") or "")[:16] or None,
        candidate_chain_id=str(archive.get("sha256") or "") or None,
        entropy_position_label="archive_container_intake",
    )


def render_markdown(profile: dict[str, Any]) -> str:
    archive = profile.get("archive", {})
    strict_zip = profile.get("strict_zip", {})
    primary = profile.get("primary_member", {})
    side_info = profile.get("side_info", {})
    lines = [
        "# Public Frontier Archive Intake",
        "",
        f"- Label: `{profile.get('label', '')}`",
        f"- Evidence grade: `{profile.get('evidence_grade', '')}`",
        f"- Score claim: `{str(profile.get('score_claim')).lower()}`",
        f"- Archive bytes: `{archive.get('bytes')}`",
        f"- Archive SHA-256: `{archive.get('sha256', '')}`",
        f"- Strict ZIP valid: `{str(strict_zip.get('valid')).lower()}`",
        f"- Primary member: `{primary.get('name', '')}` "
        f"({primary.get('compressed_bytes', 0)} charged bytes)",
        "",
        "## Charged Side Info",
        "",
        f"- Members: `{side_info.get('member_count', 0)}`",
        f"- Charged bytes: `{side_info.get('charged_bytes', 0)}`",
        f"- Runtime contract review required: "
        f"`{str(side_info.get('requires_runtime_contract_review')).lower()}`",
    ]
    blockers = strict_zip.get("blockers") or []
    if blockers:
        lines.extend(["", "## Strict ZIP Blockers", ""])
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    diffs = profile.get("baseline_diffs") or []
    if diffs:
        lines.extend(["", "## Baseline Diffs", ""])
        for diff in diffs:
            lines.append(
                f"- `{diff.get('baseline_label')}`: "
                f"{len(diff.get('changed_segments') or [])} changed segment(s)"
            )
            for row in diff.get("changed_segments") or []:
                lines.append(
                    f"  - `{row.get('segment')}`: "
                    f"{row.get('baseline_bytes')} -> {row.get('candidate_bytes')} bytes, "
                    f"`{row.get('baseline_codec')}` -> `{row.get('candidate_codec')}`"
                )
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    profile: dict[str, Any],
    *,
    json_out: Path | None = None,
    markdown_out: Path | None = None,
) -> None:
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(_json_text(profile))
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_markdown(profile))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Static intake / byte diffing for public frontier archives."
    )
    parser.add_argument("archive", type=Path, help="archive.zip to intake")
    parser.add_argument(
        "--baseline", type=Path, action="append", default=[],
        help="baseline archive(s) to byte-diff against (may repeat)",
    )
    parser.add_argument(
        "--no-segments", action="store_true",
        help="skip per-PR85-segment inspection",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    profile = profile_public_frontier_archive(
        args.archive,
        baselines=args.baseline,
        inspect_segments=not args.no_segments,
    )
    write_outputs(profile, json_out=args.json_out, markdown_out=args.markdown_out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
