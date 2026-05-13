"""Byte-level endgame decision support for PR85-family archives.

This module is local and deterministic. It validates ZIP custody, slices
PR85-family single-member bundles, probes cheap codec contracts, and estimates
rate-only transplant deltas against a named frontier archive. It never inflates
contest videos, loads scorers, dispatches jobs, or makes score claims.

REHYDRATED 2026-05-05 from .recovery_spec.json (preserved at
.recovery_quarantine_20260505T004735Z/src/tac/endgame_archive_decision.recovery_spec.json).
Spec source: bytecode disassembly of compiled .pyc; whitespace + inline comments lost.

PARTIAL REHYDRATION: Module-level constants, error class, helpers, and the
``main`` CLI entry point are reconstructed. Many internal probes
(``_brotli_probe``, ``_qma9_probe``, ``_hpm1_probe``, ``_stbm1br_probe``,
``_parse_rmb1``, ``_parse_rsb1``, ``_profile_one_archive``, ``_comparison``,
``build_endgame_decision_profile``, etc.) are stubbed to ``NotImplementedError``
because the bytecode disassembly contains many nested closures, lambdas, and
generator expressions that pycdc cannot fully decompile. Public CLI entry
``main`` will fail loud at the first deferred internal call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence

from tac.archive_byte_profile import contest_rate_term
from tac.pr85_bundle import (
    HPM1_MAGIC,
    Pr85BundleError,
    SEGMENT_ORDER,
    parse_hpm1_mask_segment,
    parse_pr85_bundle,
)

SCHEMA = "endgame_archive_decision_profile_v1"
TOOL = "experiments/profile_endgame_archive_decision.py"
EVIDENCE_GRADE = "byte_level_decision_support_only"
LOCAL_FILE_HEADER = 0x04034B50  # 67324752
QMA9_HEADER_BYTES = 20
STBM1BR_MAGIC = b"STBM1BR\x00"
RMB1_MAGIC = b"RMB1"
RSB1_MAGIC = b"RSB1"

_QUARANTINE_SPEC = (
    ".recovery_quarantine_20260505T004735Z/src/tac/endgame_archive_decision.recovery_spec.json"
)


class EndgameArchiveDecisionError(ValueError):
    """Raised when endgame archive profiling inputs are malformed."""

    pass


@dataclass(frozen=True)
class _MemberPayload:
    """ZIP member byte payload and structural metadata."""

    name: str
    raw_name: bytes
    bytes_compressed: int
    bytes_uncompressed: int
    crc32: int
    compress_type: int
    payload: bytes
    sha256: str
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _ArchiveProfile:
    """Profiled archive byte attribution."""

    path: str
    total_bytes: int
    sha256: str
    member_payloads: Sequence[_MemberPayload]
    extra: Mapping[str, Any] = field(default_factory=dict)


def _json_text(payload: Any) -> str:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


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
    return -sum(
        (c / n) * math.log2(c / n) for c in counts.values() if c > 0
    )


def _display_ascii(data: bytes, *, limit: int = 16) -> str:
    chars = []
    for b in data[:limit]:
        if 32 <= b < 127:
            chars.append(chr(b))
        else:
            chars.append(".")
    return "".join(chars)


def _safe_member_blockers(name: str) -> list[str]:
    blockers: list[str] = []
    if not name:
        blockers.append("empty_name")
        return blockers
    if "\x00" in name:
        blockers.append("nul_byte")
    if "\\" in name:
        blockers.append("backslash")
    posix_path = PurePosixPath(name)
    if posix_path.is_absolute():
        blockers.append("absolute_posix")
    windows_path = PureWindowsPath(name)
    if windows_path.is_absolute() or windows_path.drive:
        blockers.append("absolute_windows")
    if any(p == ".." for p in posix_path.parts):
        blockers.append("dotdot")
    return blockers


def _decode_zip_name(raw: bytes, flag_bits: int) -> str:
    if flag_bits & 0x800:
        return raw.decode("utf-8")
    return raw.decode("cp437")


def _local_header_name(buf: bytes, offset: int) -> bytes:
    if offset + 30 > len(buf):
        raise EndgameArchiveDecisionError(
            f"local header offset {offset} exceeds buffer length {len(buf)}"
        )
    sig = int.from_bytes(buf[offset : offset + 4], "little")
    if sig != LOCAL_FILE_HEADER:
        raise EndgameArchiveDecisionError(
            f"local header at {offset} has wrong signature: {sig:#x}"
        )
    name_len = int.from_bytes(buf[offset + 26 : offset + 28], "little")
    name_start = offset + 30
    return buf[name_start : name_start + name_len]


def _import_brotli():
    try:
        import brotli  # noqa: F401

        return brotli
    except ImportError as exc:  # pragma: no cover - env dependent
        raise EndgameArchiveDecisionError(
            "endgame archive profiling requires the brotli package"
        ) from exc


def _rehydration_failure(symbol: str) -> NotImplementedError:
    return NotImplementedError(
        f"rehydration incomplete: {symbol} contains intricate nested closures "
        f"and generators that pycdc cannot fully decompile; original bytecode "
        f"preserved in {_QUARANTINE_SPEC}"
    )


def _brotli_probe(data: bytes, *, name: str = "", limit: int = 0) -> dict[str, Any]:
    brotli = _import_brotli()
    try:
        decoded = brotli.decompress(data)
    except brotli.error as exc:
        return {
            "status": "failed",
            "codec": "brotli",
            "name": name,
            "error": str(exc),
        }
    return {
        "status": "ok",
        "codec": "brotli",
        "name": name,
        "encoded_bytes": len(data),
        "encoded_sha256": _sha256_bytes(data),
        "decoded_bytes": len(decoded),
        "decoded_sha256": _sha256_bytes(decoded),
        "decoded_entropy": _byte_entropy(decoded[:limit] if limit else decoded),
        "decoded_prefix_ascii": _display_ascii(decoded),
    }


def _qma9_probe(data: bytes) -> dict[str, Any]:
    if not data.startswith(b"QMA9") or len(data) < QMA9_HEADER_BYTES:
        return {"status": "failed", "codec": "QMA9", "error": "bad_magic_or_truncated_header"}
    n_frames, height, width, bitstream_len = struct.unpack_from("<IIII", data, 4)
    actual_bitstream_len = len(data) - QMA9_HEADER_BYTES
    status = "ok" if actual_bitstream_len == int(bitstream_len) else "failed"
    return {
        "status": status,
        "codec": "QMA9",
        "n_frames": int(n_frames),
        "height": int(height),
        "width": int(width),
        "header_bytes": QMA9_HEADER_BYTES,
        "declared_bitstream_bytes": int(bitstream_len),
        "actual_bitstream_bytes": actual_bitstream_len,
    }


def _hpm1_probe(data: bytes) -> dict[str, Any]:
    try:
        contract = parse_hpm1_mask_segment(data)
    except Pr85BundleError as exc:
        return {"status": "failed", "codec": "HPM1", "error": str(exc)}
    return {"status": "ok", "codec": "HPM1", **dict(contract.metadata)}


def _stbm1br_probe(data: bytes) -> dict[str, Any]:
    if not data.startswith(STBM1BR_MAGIC):
        return {"status": "failed", "codec": "STBM1BR", "error": "bad_magic"}
    return {
        "status": "ok",
        "codec": "STBM1BR",
        "bytes": len(data),
        "sha256": _sha256_bytes(data),
    }


def _parse_rmb1(data: bytes) -> dict[str, Any]:
    brotli = _import_brotli()
    if len(data) < 6 or not data.startswith(RMB1_MAGIC):
        return {"status": "failed", "codec": "RMB1", "error": "bad_magic_or_truncated_header"}
    mask_len = int.from_bytes(data[4:6], "little")
    mask_br = data[6 : 6 + mask_len]
    vals_br = data[6 + mask_len :]
    try:
        mask = brotli.decompress(mask_br)
        vals = brotli.decompress(vals_br)
    except brotli.error as exc:
        return {"status": "failed", "codec": "RMB1", "error": str(exc)}
    if len(mask) % 75:
        return {
            "status": "failed",
            "codec": "RMB1",
            "error": f"mask byte count {len(mask)} is not divisible by 75",
        }
    selected = 0
    for row_start in range(0, len(mask), 75):
        for byte_i, byte in enumerate(mask[row_start : row_start + 75]):
            for bit in range(8):
                if byte_i * 8 + bit >= 600:
                    break
                selected += int(bool(byte & (1 << bit)))
    status = "ok" if selected == len(vals) else "failed"
    payload: dict[str, Any] = {
        "status": status,
        "codec": "RMB1",
        "mask_brotli_bytes": len(mask_br),
        "values_brotli_bytes": len(vals_br),
        "decoded_mask_bytes": len(mask),
        "decoded_values_bytes": len(vals),
        "row_count": len(mask) // 75,
        "selected_value_count": selected,
    }
    if status != "ok":
        payload["error"] = f"selected mask count {selected} != value bytes {len(vals)}"
    return payload


def _parse_rsb1(data: bytes) -> dict[str, Any]:
    brotli = _import_brotli()
    if len(data) < 8 or not data.startswith(RSB1_MAGIC):
        return {"status": "failed", "codec": "RSB1", "error": "bad_magic_or_truncated_header"}
    action_count = int.from_bytes(data[4:6], "little")
    table_id = int(data[6])
    flags = int(data[7])
    try:
        raw_actions = brotli.decompress(data[8:])
    except brotli.error as exc:
        return {"status": "failed", "codec": "RSB1", "error": str(exc)}
    status = "ok" if len(raw_actions) == action_count else "failed"
    payload: dict[str, Any] = {
        "status": status,
        "codec": "RSB1",
        "action_count": action_count,
        "table_id": table_id,
        "flags": flags,
        "raw_action_bytes": len(raw_actions),
        "unique_actions": len(set(raw_actions)),
        "actions_sha256": _sha256_bytes(raw_actions),
    }
    if status != "ok":
        payload["error"] = f"decoded action bytes {len(raw_actions)} != declared action count {action_count}"
    return payload


def _codec_label(name: str, payload: bytes) -> str:
    if name == "mask" and payload.startswith(b"QMA9"):
        return "QMA9_mask"
    if name == "mask" and payload.startswith(HPM1_MAGIC):
        return "HPM1_mask"
    if payload.startswith(RMB1_MAGIC):
        return "RMB1_randmulti"
    if payload.startswith(RSB1_MAGIC):
        return "RSB1_side_actions"
    if payload.startswith(STBM1BR_MAGIC):
        return "STBM1BR_mask"
    if name in SEGMENT_ORDER:
        return "brotli_or_opaque_pr85_segment"
    return "opaque"


def _segment_validation(name: str, segment: bytes) -> dict[str, Any]:
    if name == "mask" and segment.startswith(b"QMA9"):
        return _qma9_probe(segment)
    if name == "mask" and segment.startswith(HPM1_MAGIC):
        return _hpm1_probe(segment)
    if name == "mask" and segment.startswith(STBM1BR_MAGIC):
        return _stbm1br_probe(segment)
    if name == "randmulti" and segment.startswith(RMB1_MAGIC):
        return _parse_rmb1(segment)
    if name in {"model", "pose", "post", "shift", "frac", "frac2", "frac3", "bias", "region", "randmulti"}:
        return _brotli_probe(segment, name=name)
    return {"status": "ok", "codec": _codec_label(name, segment)}


def _segment_row(name: str, segment: bytes, offset: int | None = None) -> dict[str, Any]:
    validation = _segment_validation(name, segment)
    return {
        "name": name,
        "bytes": len(segment),
        "sha256": _sha256_bytes(segment),
        "offset": offset,
        "codec": _codec_label(name, segment),
        "entropy": _byte_entropy(segment),
        "prefix_ascii": _display_ascii(segment),
        "validation": validation,
    }


def _side_member_row(payload: _MemberPayload) -> dict[str, Any]:
    if payload.payload.startswith(RSB1_MAGIC):
        validation = _parse_rsb1(payload.payload)
    elif payload.payload.startswith(RMB1_MAGIC):
        validation = _parse_rmb1(payload.payload)
    else:
        validation = {"status": "ok", "codec": _codec_label(payload.name, payload.payload)}
    return {
        "name": payload.name,
        "bytes": payload.bytes_uncompressed,
        "compressed_bytes": payload.bytes_compressed,
        "sha256": payload.sha256,
        "codec": _codec_label(payload.name, payload.payload),
        "validation": validation,
    }


def _member_payloads(archive_path: Path) -> Sequence[_MemberPayload]:
    data = archive_path.read_bytes()
    members: list[_MemberPayload] = []
    with zipfile.ZipFile(archive_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            raw_name = _local_header_name(data, int(info.header_offset))
            decoded_name = _decode_zip_name(raw_name, int(info.flag_bits))
            extra: dict[str, Any] = {
                "central_local_name_match": decoded_name == info.filename,
                "local_name": decoded_name,
                "safe_name_blockers": _safe_member_blockers(info.filename),
            }
            try:
                payload = zf.read(info)
            except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
                payload = b""
                extra["read_error"] = str(exc)
            members.append(
                _MemberPayload(
                    name=info.filename,
                    raw_name=raw_name,
                    bytes_compressed=int(info.compress_size),
                    bytes_uncompressed=int(info.file_size),
                    crc32=int(info.CRC),
                    compress_type=int(info.compress_type),
                    payload=payload,
                    sha256=_sha256_bytes(payload),
                    extra=extra,
                )
            )
    return members


def _try_primary_bundle(payload: _MemberPayload) -> dict[str, Any] | None:
    if payload.name != "x" or not payload.payload:
        return None
    try:
        bundle = parse_pr85_bundle(payload.payload)
    except Pr85BundleError as exc:
        return {"status": "failed", "error": str(exc)}
    segments = [
        _segment_row(name, bytes(segment), bundle.segment_offsets.get(name))
        for name, segment in bundle.segments.items()
    ]
    return {
        "status": "ok",
        "format": bundle.format,
        "header_bytes": bundle.header_bytes,
        "member_name": payload.name,
        "member_bytes": len(payload.payload),
        "member_sha256": payload.sha256,
        "segments": segments,
        "segment_count": len(segments),
    }


def _strict_zip_report(members: Sequence[_MemberPayload]) -> dict[str, Any]:
    blockers: list[str] = []
    names = [member.name for member in members]
    for name, count in Counter(names).items():
        if count > 1:
            blockers.append(f"{name}:duplicate_member")
    for member in members:
        for blocker in member.extra.get("safe_name_blockers", ()):
            blockers.append(f"{member.name}:{blocker}")
        if not member.extra.get("central_local_name_match", False):
            blockers.append(f"{member.name}:central_local_name_mismatch")
        if member.extra.get("read_error"):
            blockers.append(f"{member.name}:read_error:{member.extra['read_error']}")
    return {
        "valid": not blockers,
        "blockers": blockers,
        "member_names": names,
        "member_count": len(names),
    }


def _profile_one_archive(archive_path: Path) -> _ArchiveProfile:
    members = _member_payloads(archive_path)
    strict_zip = _strict_zip_report(members)
    primary = next((_try_primary_bundle(member) for member in members if member.name == "x"), None)
    if primary is None:
        primary = {"status": "missing", "segments": []}
    side_members = [member for member in members if member.name != "x"]
    side_rows = [_side_member_row(member) for member in side_members]
    blockers: list[str] = []
    if not strict_zip["valid"]:
        blockers.extend(str(blocker) for blocker in strict_zip["blockers"])
    if primary.get("status") != "ok":
        blockers.append(f"primary_member_{primary.get('status', 'invalid')}")
    for index, row in enumerate(side_rows, start=1):
        if row["validation"].get("status") != "ok":
            blockers.append(f"side_member_validation_failed:{row['name']}#{index}")
    extra = {
        "label": archive_path.stem,
        "strict_zip": strict_zip,
        "primary_member": primary,
        "side_info": {
            "total_bytes": sum(member.bytes_uncompressed for member in side_members),
            "zip_member_count": len(side_members),
            "members": side_rows,
        },
        "decision_support": {
            "valid_for_byte_decision": not blockers,
            "blockers": blockers,
        },
    }
    return _ArchiveProfile(
        path=str(archive_path),
        total_bytes=int(archive_path.stat().st_size),
        sha256=_sha256_file(archive_path),
        member_payloads=members,
        extra=extra,
    )


def _archive_row(label: str, profile: _ArchiveProfile) -> dict[str, Any]:
    return {
        **dict(profile.extra),
        "label": label,
        "path": profile.path,
        "archive_bytes": profile.total_bytes,
        "archive_sha256": profile.sha256,
    }


def _segments_by_name(row: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    primary = row.get("primary_member")
    if not isinstance(primary, Mapping):
        return {}
    segments = primary.get("segments")
    if not isinstance(segments, Sequence):
        return {}
    return {
        str(segment.get("name")): segment
        for segment in segments
        if isinstance(segment, Mapping) and segment.get("name") is not None
    }


def _segment_delta_rows(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    ref_segments = _segments_by_name(reference)
    cand_segments = _segments_by_name(candidate)
    rows: list[dict[str, Any]] = []
    for name in SEGMENT_ORDER:
        ref = ref_segments.get(name)
        cand = cand_segments.get(name)
        if ref is None or cand is None:
            continue
        rows.append(
            {
                "segment": name,
                "reference_bytes": int(ref.get("bytes", 0)),
                "candidate_bytes": int(cand.get("bytes", 0)),
                "delta_bytes": int(cand.get("bytes", 0)) - int(ref.get("bytes", 0)),
                "changed": ref.get("sha256") != cand.get("sha256"),
                "reference_codec": ref.get("codec"),
                "candidate_codec": cand.get("codec"),
            }
        )
    return rows


def _comparison(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    archive_delta = int(candidate["archive_bytes"]) - int(reference["archive_bytes"])
    side_delta = int(candidate["side_info"]["total_bytes"]) - int(reference["side_info"]["total_bytes"])
    zip_overhead_delta = archive_delta
    primary_ref = reference.get("primary_member", {})
    primary_cand = candidate.get("primary_member", {})
    if isinstance(primary_ref, Mapping) and isinstance(primary_cand, Mapping):
        zip_overhead_delta -= int(primary_cand.get("member_bytes", 0)) - int(primary_ref.get("member_bytes", 0))
    zip_overhead_delta -= side_delta
    segment_deltas = _segment_delta_rows(reference, candidate)
    transplant_estimates = []
    for row in segment_deltas:
        estimate_delta = int(row["delta_bytes"])
        requires_side = False
        if row["segment"] == "randmulti" and candidate["side_info"]["total_bytes"]:
            requires_side = True
            estimate_delta += int(candidate["side_info"]["total_bytes"]) + zip_overhead_delta
        transplant_estimates.append(
            {
                "segment": row["segment"],
                "requires_candidate_side_info": requires_side,
                "estimated_archive_delta_bytes": estimate_delta,
                "segment_delta_bytes": row["delta_bytes"],
                "candidate_side_info_bytes": int(candidate["side_info"]["total_bytes"]) if requires_side else 0,
                "zip_overhead_delta_bytes": zip_overhead_delta if requires_side else 0,
            }
        )
    return {
        "reference_label": reference["label"],
        "candidate_label": candidate["label"],
        "archive_delta_bytes": archive_delta,
        "rate_delta": contest_rate_term(archive_delta),
        "primary_member_delta_bytes": int(primary_cand.get("member_bytes", 0)) - int(primary_ref.get("member_bytes", 0)),
        "side_info_delta_bytes": side_delta,
        "zip_overhead_delta_bytes": zip_overhead_delta,
        "segment_deltas": segment_deltas,
        "transplant_estimates": transplant_estimates,
        "decision_valid": bool(reference["decision_support"]["valid_for_byte_decision"])
        and bool(candidate["decision_support"]["valid_for_byte_decision"]),
    }


def _rank_actions(actions: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(actions, key=lambda row: (int(row.get("estimated_archive_delta_bytes", 0)), str(row.get("segment", ""))))


def _parse_label_path(label_path: str) -> dict[str, Any]:
    if "=" in label_path:
        label, raw_path = label_path.split("=", 1)
        if not label or not raw_path:
            raise EndgameArchiveDecisionError(f"candidate must be LABEL=PATH, got {label_path!r}")
        return {"label": label, "path": Path(raw_path)}
    path = Path(label_path)
    return {"label": path.stem, "path": path}


def _candidate_map_from_args(args: argparse.Namespace) -> tuple[dict[str, Path], str]:
    candidates: dict[str, Path] = {}
    reference = getattr(args, "reference", None)
    frontier_label = getattr(args, "frontier_label", None) or "reference"
    if reference is not None:
        candidates[frontier_label] = Path(reference)
    raw_candidates = list(getattr(args, "candidate", []) or [])
    labels = list(getattr(args, "label", []) or [])
    for index, raw in enumerate(raw_candidates):
        parsed = _parse_label_path(str(raw))
        label = labels[index] if index < len(labels) else parsed["label"]
        candidates[str(label)] = Path(parsed["path"])
    if not candidates:
        raise EndgameArchiveDecisionError("at least one --reference or --candidate is required")
    if frontier_label not in candidates:
        frontier_label = next(iter(candidates))
    return candidates, str(frontier_label)


def build_endgame_decision_profile(
    candidates_or_args: argparse.Namespace | Mapping[str, Path | str],
    *,
    frontier_label: str | None = None,
) -> dict[str, Any]:
    if isinstance(candidates_or_args, argparse.Namespace):
        candidates, frontier = _candidate_map_from_args(candidates_or_args)
    else:
        candidates = {str(label): Path(path) for label, path in candidates_or_args.items()}
        if not candidates:
            raise EndgameArchiveDecisionError("candidate mapping must not be empty")
        frontier = frontier_label or next(iter(candidates))
        if frontier not in candidates:
            raise EndgameArchiveDecisionError(f"frontier label {frontier!r} is not in candidates")
    archive_rows = []
    for label, path in candidates.items():
        archive_rows.append(_archive_row(label, _profile_one_archive(path)))
    by_label = {row["label"]: row for row in archive_rows}
    reference = by_label[frontier]
    comparisons = [
        _comparison(reference, row)
        for row in archive_rows
        if row["label"] != frontier
    ]
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "evidence_grade": EVIDENCE_GRADE,
        "score_claim": False,
        "frontier_label": frontier,
        "archives": archive_rows,
        "comparisons_to_frontier": comparisons,
        "ranked_transplant_actions": _rank_actions(
            action
            for comparison in comparisons
            for action in comparison["transplant_estimates"]
        ),
    }


def render_markdown(profile: dict[str, Any]) -> str:
    lines = [
        "# Endgame Archive Decision Profile",
        "",
        f"- evidence_grade: `{profile.get('evidence_grade')}`",
        f"- score_claim: `{profile.get('score_claim')}`",
        f"- frontier_label: `{profile.get('frontier_label')}`",
        "",
        "## Archives",
        "",
        "| label | bytes | strict_zip | decision_valid | side_bytes |",
        "|---|---:|---|---|---:|",
    ]
    for row in profile.get("archives", []):
        lines.append(
            "| `{}` | {} | `{}` | `{}` | {} |".format(
                row["label"],
                row["archive_bytes"],
                row["strict_zip"]["valid"],
                row["decision_support"]["valid_for_byte_decision"],
                row["side_info"]["total_bytes"],
            )
        )
    lines.extend(["", "## Comparisons", "", "| candidate | archive_delta | rate_delta | decision_valid |", "|---|---:|---:|---|"])
    for row in profile.get("comparisons_to_frontier", []):
        lines.append(
            "| `{}` | {} | {:.9f} | `{}` |".format(
                row["candidate_label"],
                row["archive_delta_bytes"],
                row["rate_delta"],
                row["decision_valid"],
            )
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    profile: dict[str, Any],
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
    parser = argparse.ArgumentParser(
        description="Byte-level endgame decision support for PR85-family archives."
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="reference frontier archive (PR85 single-member zip)",
    )
    parser.add_argument(
        "--candidate",
        type=str,
        action="append",
        default=[],
        help="candidate archive as PATH or LABEL=PATH (may be passed multiple times)",
    )
    parser.add_argument(
        "--frontier-label",
        type=str,
        default=None,
        help="label to use as the frontier comparator; defaults to the reference label or first candidate",
    )
    parser.add_argument(
        "--label",
        type=str,
        action="append",
        default=[],
        help="optional label for each candidate (positional 1:1 with --candidate)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="write deterministic JSON decision profile",
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        help="write markdown summary",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        profile = build_endgame_decision_profile(args)
    except (EndgameArchiveDecisionError, FileNotFoundError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))
    write_outputs(
        profile, json_out=args.json_out, markdown_out=args.markdown_out
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
