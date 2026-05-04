"""Static intake and byte diffing for public frontier archives.

This module is intentionally offline and byte-only. It validates ZIP custody,
identifies PR85-family bundle segments, records charged side-info members, and
diffs segment identities against named baselines. It never inflates contest
videos, loads scorers, submits jobs, or makes score claims.
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

from tac.archive_byte_profile import CONTEST_ORIGINAL_BYTES, contest_rate_term
from tac.pr85_bundle import Pr85BundleError, parse_pr85_bundle


SCHEMA = "public_frontier_archive_intake_v1"
TOOL = "experiments/profile_public_frontier_intake.py"
EVIDENCE_GRADE = "external_archive_byte_intake_only"
LOCAL_FILE_HEADER = 0x04034B50


class PublicFrontierIntakeError(ValueError):
    """Raised when a public-frontier intake input is malformed."""


@dataclass(frozen=True)
class _ParsedArchive:
    report: dict[str, Any]
    primary_segments: dict[str, dict[str, Any]]


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
    entropy = 0.0
    total = len(data)
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
            raise PublicFrontierIntakeError(
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
            raise PublicFrontierIntakeError(
                f"bad local header signature for {info.filename!r} at {info.header_offset}"
            )
        raw_name = handle.read(name_len)
        if len(raw_name) != name_len:
            raise PublicFrontierIntakeError(f"truncated local filename for {info.filename!r}")
        handle.seek(extra_len, 1)
    return _decode_zip_name(raw_name, info.flag_bits)


def _codec_label(segment_name: str, data: bytes) -> str:
    if segment_name == "mask":
        if data.startswith(b"QMA9"):
            return "QMA9_range_mask"
        if data.startswith(b"STBM1BR\0"):
            return "STBM1BR_lossless_mask_recode"
        if data.startswith(b"HPM1"):
            return "HPM1_hpac_mask"
    if segment_name == "randmulti" and data.startswith(b"RMB1"):
        return "RMB1_side_info_backed_randmulti"
    if segment_name in {"model", "pose", "post", "shift", "frac", "frac2", "frac3", "bias", "region", "randmulti"}:
        return "opaque_pr85_segment"
    return "unknown"


def _segment_row(name: str, data: bytes, offset: int) -> dict[str, Any]:
    return {
        "name": name,
        "bytes": len(data),
        "sha256": _sha256_bytes(data),
        "offset": int(offset),
        "magic_hex": data[:8].hex(),
        "magic_ascii": _display_ascii(data),
        "codec": _codec_label(name, data),
        "entropy_bits_per_byte": _byte_entropy(data),
    }


def _inspect_member_data(name: str, data: bytes) -> dict[str, Any]:
    row = {
        "name": name,
        "bytes": len(data),
        "sha256": _sha256_bytes(data),
        "magic_hex": data[:8].hex(),
        "magic_ascii": _display_ascii(data),
        "entropy_bits_per_byte": _byte_entropy(data),
    }
    try:
        bundle = parse_pr85_bundle(data)
    except Pr85BundleError as exc:
        row["recognized_container"] = None
        row["container_probe_error"] = str(exc)
        return row

    row["recognized_container"] = "pr85_family_bundle"
    row["bundle_format"] = bundle.format
    row["bundle_header_bytes"] = int(bundle.header_bytes)
    row["segment_lengths"] = bundle.segment_lengths
    row["segments"] = [
        _segment_row(segment_name, bytes(bundle.segments[segment_name]), bundle.segment_offsets[segment_name])
        for segment_name in bundle.segment_lengths
    ]
    return row


def _archive_members(path: Path) -> tuple[list[dict[str, Any]], dict[str, bytes], list[str]]:
    blockers: list[str] = []
    member_data: dict[str, bytes] = {}
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(path, "r") as zf:
        infos = zf.infolist()
        if not infos:
            blockers.append("zip_has_no_members")
        name_counts = Counter(info.filename for info in infos)
        for name, count in sorted(name_counts.items()):
            if count > 1:
                blockers.append(f"duplicate_member_name:{name}:{count}")
        for info in infos:
            local_name = _local_header_name(path, info)
            row_blockers = _safe_member_blockers(info.filename)
            if local_name != info.filename:
                row_blockers.append("central_local_name_mismatch")
            if info.is_dir():
                row_blockers.append("directory_member")
            # Python's ZipFile refuses to read mismatched central/local names.
            # Record the custody blocker and skip payload probing instead of
            # letting the malformed public artifact become an opaque exception.
            data = b""
            if not info.is_dir() and local_name == info.filename:
                data = zf.read(info)
            if not info.is_dir() and local_name == info.filename:
                member_data[info.filename] = data
            rows.append(
                {
                    "name": info.filename,
                    "local_header_name": local_name,
                    "central_local_name_match": local_name == info.filename,
                    "bytes": int(info.file_size),
                    "compressed_bytes": int(info.compress_size),
                    "method_id": int(info.compress_type),
                    "header_offset": int(info.header_offset),
                    "crc32": f"{info.CRC:08x}",
                    "sha256": None if info.is_dir() else _sha256_bytes(data),
                    "blockers": row_blockers,
                }
            )
            blockers.extend(f"{info.filename}:{blocker}" for blocker in row_blockers)
    return rows, member_data, blockers


def _primary_pr85_member(member_data: Mapping[str, bytes]) -> tuple[str | None, dict[str, Any] | None, dict[str, dict[str, Any]]]:
    candidates: list[tuple[str, dict[str, Any], dict[str, dict[str, Any]]]] = []
    for name, data in member_data.items():
        inspected = _inspect_member_data(name, data)
        if inspected.get("recognized_container") != "pr85_family_bundle":
            continue
        segments = {row["name"]: row for row in inspected["segments"]}
        candidates.append((name, inspected, segments))
    if not candidates:
        return None, None, {}
    candidates.sort(key=lambda item: (-int(item[1]["bytes"]), item[0]))
    name, inspected, segments = candidates[0]
    return name, inspected, segments


def _diff_against_baselines(
    primary_segments: Mapping[str, Mapping[str, Any]],
    baselines: Mapping[str, _ParsedArchive],
    *,
    candidate_archive_bytes: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, baseline in sorted(baselines.items()):
        baseline_segments = baseline.primary_segments
        changed: list[dict[str, Any]] = []
        for name in sorted(set(primary_segments) | set(baseline_segments)):
            left = baseline_segments.get(name)
            right = primary_segments.get(name)
            changed.append(
                {
                    "segment": name,
                    "same_sha256": bool(
                        left is not None and right is not None and left["sha256"] == right["sha256"]
                    ),
                    "baseline_bytes": None if left is None else int(left["bytes"]),
                    "candidate_bytes": None if right is None else int(right["bytes"]),
                    "delta_bytes": None
                    if left is None or right is None
                    else int(right["bytes"]) - int(left["bytes"]),
                    "baseline_codec": None if left is None else left.get("codec"),
                    "candidate_codec": None if right is None else right.get("codec"),
                    "baseline_sha256": None if left is None else left.get("sha256"),
                    "candidate_sha256": None if right is None else right.get("sha256"),
                }
            )
        archive_delta = int(candidate_archive_bytes) - int(baseline.report["archive"]["bytes"])
        rows.append(
            {
                "baseline_label": label,
                "baseline_archive_path": baseline.report["archive"]["path"],
                "baseline_archive_bytes": baseline.report["archive"]["bytes"],
                "candidate_minus_baseline_archive_bytes": archive_delta,
                "candidate_minus_baseline_rate_score_delta": contest_rate_term(archive_delta),
                "changed_segments": [row for row in changed if not row["same_sha256"]],
                "unchanged_segment_count": sum(1 for row in changed if row["same_sha256"]),
                "segment_count": len(changed),
            }
        )
    return rows


def _parse_baseline_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise PublicFrontierIntakeError(
            f"baseline must be LABEL=PATH, got {value!r}"
        )
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise PublicFrontierIntakeError(f"baseline must be LABEL=PATH, got {value!r}")
    return label.strip(), Path(path)


def _analyze_one(path: Path, *, label: str, candidate_bytes: int | None = None) -> _ParsedArchive:
    if not path.is_file():
        raise FileNotFoundError(f"archive not found: {path}")
    archive_bytes = path.stat().st_size
    try:
        members, member_data, blockers = _archive_members(path)
    except zipfile.BadZipFile as exc:
        raise PublicFrontierIntakeError(f"bad ZIP archive: {path}") from exc

    primary_name, primary, primary_segments = _primary_pr85_member(member_data)
    side_names = [name for name in member_data if name != primary_name]
    side_members = [_inspect_member_data(name, member_data[name]) for name in sorted(side_names)]
    total_side_bytes = sum(int(member["bytes"]) for member in side_members)
    if primary_name is None:
        blockers.append("no_pr85_family_primary_member_detected")
    report = {
        "schema": SCHEMA,
        "tool": TOOL,
        "label": label,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": EVIDENCE_GRADE,
        "archive": {
            "path": str(path),
            "sha256": _sha256_file(path),
            "bytes": archive_bytes,
            "bytes_delta_candidate_minus_baseline": 0
            if candidate_bytes is None
            else archive_bytes - int(candidate_bytes),
            "rate_term": contest_rate_term(archive_bytes),
        },
        "strict_zip": {
            "valid": not blockers,
            "blockers": sorted(blockers),
            "member_count": len(members),
            "members": members,
        },
        "primary_member": primary,
        "side_info": {
            "members": side_members,
            "member_count": len(side_members),
            "charged_bytes": total_side_bytes,
            "rate_score_cost": contest_rate_term(total_side_bytes),
            "requires_runtime_contract_review": bool(side_members),
        },
        "notes": [
            "byte-only intake; no inflate, scorer, dispatch, promotion, or score claim",
            "public-reported components remain external until exact CUDA auth replay",
        ],
    }
    return _ParsedArchive(report=report, primary_segments=primary_segments)


def profile_public_frontier_archive(
    archive: Path | str,
    *,
    label: str,
    baselines: Mapping[str, Path | str] | None = None,
) -> dict[str, Any]:
    """Build a deterministic public-frontier intake report for one archive."""

    archive_path = Path(archive)
    candidate = _analyze_one(archive_path, label=label)
    baseline_reports: dict[str, _ParsedArchive] = {}
    for baseline_label, baseline_path in (baselines or {}).items():
        baseline_reports[baseline_label] = _analyze_one(
            Path(baseline_path),
            label=baseline_label,
            candidate_bytes=int(candidate.report["archive"]["bytes"]),
        )
    candidate.report["baseline_diffs"] = _diff_against_baselines(
        candidate.primary_segments,
        baseline_reports,
        candidate_archive_bytes=int(candidate.report["archive"]["bytes"]),
    )
    candidate.report["baselines"] = {
        label: {
            "archive": parsed.report["archive"],
            "strict_zip_valid": parsed.report["strict_zip"]["valid"],
            "primary_member_name": None
            if parsed.report["primary_member"] is None
            else parsed.report["primary_member"]["name"],
            "side_info_charged_bytes": parsed.report["side_info"]["charged_bytes"],
        }
        for label, parsed in sorted(baseline_reports.items())
    }
    return candidate.report


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact Markdown view of an intake report."""

    lines = [
        "# Public Frontier Archive Intake",
        "",
        f"- label: `{report['label']}`",
        f"- evidence_grade: `{report['evidence_grade']}`",
        f"- score_claim: `{report['score_claim']}`",
        f"- archive bytes: `{report['archive']['bytes']}`",
        f"- archive sha256: `{report['archive']['sha256']}`",
        f"- strict ZIP valid: `{report['strict_zip']['valid']}`",
        f"- side-info bytes: `{report['side_info']['charged_bytes']}`",
        "",
        "This report is byte-only. It does not inflate videos, load scorers, "
        "dispatch jobs, promote methods, or claim a contest score.",
        "",
        "## Members",
        "",
        "| name | bytes | compressed | local name match | sha256 |",
        "|---|---:|---:|---|---|",
    ]
    for member in report["strict_zip"]["members"]:
        lines.append(
            f"| {member['name']} | {member['bytes']} | {member['compressed_bytes']} | "
            f"{member['central_local_name_match']} | {member['sha256'] or ''} |"
        )
    if report["strict_zip"]["blockers"]:
        lines.extend(["", "## Strict ZIP Blockers", ""])
        for blocker in report["strict_zip"]["blockers"]:
            lines.append(f"- `{blocker}`")
    primary = report.get("primary_member")
    if primary:
        lines.extend(
            [
                "",
                "## Primary PR85-Family Bundle",
                "",
                f"- member: `{primary['name']}`",
                f"- format: `{primary['bundle_format']}`",
                "",
                "| segment | bytes | codec | sha256 |",
                "|---|---:|---|---|",
            ]
        )
        for segment in primary["segments"]:
            lines.append(
                f"| {segment['name']} | {segment['bytes']} | {segment['codec']} | {segment['sha256']} |"
            )
    if report["side_info"]["members"]:
        lines.extend(["", "## Charged Side Info", "", "| member | bytes | magic | sha256 |", "|---|---:|---|---|"])
        for member in report["side_info"]["members"]:
            lines.append(
                f"| {member['name']} | {member['bytes']} | {member['magic_ascii']} | {member['sha256']} |"
            )
    if report.get("baseline_diffs"):
        lines.extend(["", "## Baseline Diffs", ""])
        for diff in report["baseline_diffs"]:
            lines.extend(
                [
                    f"### {diff['baseline_label']}",
                    "",
                    f"- archive delta bytes: `{diff['candidate_minus_baseline_archive_bytes']}`",
                    f"- rate score delta: `{diff['candidate_minus_baseline_rate_score_delta']}`",
                    "",
                    "| segment | delta bytes | baseline codec | candidate codec | same sha256 |",
                    "|---|---:|---|---|---|",
                ]
            )
            for segment in diff["changed_segments"]:
                lines.append(
                    f"| {segment['segment']} | {segment['delta_bytes']} | "
                    f"{segment['baseline_codec']} | {segment['candidate_codec']} | {segment['same_sha256']} |"
                )
    return "\n".join(lines) + "\n"


def write_outputs(report: Mapping[str, Any], *, json_out: Path | None, markdown_out: Path | None) -> None:
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(_json_text(report), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_markdown(report), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--label", default="candidate")
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        help="Named baseline as LABEL=PATH. May be repeated.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    baselines = dict(_parse_baseline_arg(value) for value in args.baseline)
    report = profile_public_frontier_archive(
        args.archive,
        label=args.label,
        baselines=baselines,
    )
    write_outputs(report, json_out=args.json_out, markdown_out=args.markdown_out)
    if args.json_out is None and args.markdown_out is None:
        print(_json_text(report), end="")
    return 0


__all__ = [
    "PublicFrontierIntakeError",
    "profile_public_frontier_archive",
    "render_markdown",
    "write_outputs",
    "main",
]
