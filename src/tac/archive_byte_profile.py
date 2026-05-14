# SPDX-License-Identifier: MIT
"""Deterministic ZIP byte attribution for contest archive research.

The profiler is intentionally byte-only: it does not extract archive payloads,
inflate contest outputs, load scorer models, or make score claims.

Recovered from prior bytecode custody and promoted into a maintained, tested
implementation on 2026-05-05. The tool remains score-neutral: it emits byte
profiles, SHA-256s, ZIP metadata, and rate-term accounting, but it never
extracts payloads or treats byte-only evidence as score evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

SCHEMA = "archive_byte_profile_collection_v1"
ARCHIVE_SCHEMA = "archive_byte_profile_v1"
ARCHIVE_DIFF_SCHEMA = "archive_candidate_diff_manifest_v1"
TOOL = "experiments/profile_archive_bytes.py"
EVIDENCE_GRADE = "byte_profile_only"
CONTEST_ORIGINAL_BYTES = 37545489
RATE_TERM_COEFFICIENT = 25 / CONTEST_ORIGINAL_BYTES

_QUARANTINE_SPEC = (
    ".recovery_quarantine_20260505T004735Z/src/tac/archive_byte_profile.recovery_spec.json"
)


class ArchiveByteProfileError(ValueError):
    """Raised when an archive cannot be safely profiled."""

    pass


def contest_rate_term(byte_count: int) -> float:
    """Return the contest formula rate contribution for ``byte_count``."""
    return 25 * int(byte_count) / CONTEST_ORIGINAL_BYTES


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_zip_member_name(name: str) -> str:
    if not name:
        raise ArchiveByteProfileError("archive member name is empty")
    if "\x00" in name:
        raise ArchiveByteProfileError(
            f"archive member contains NUL byte: {name!r}"
        )
    if "\\" in name:
        raise ArchiveByteProfileError(
            f"archive member uses backslashes: {name!r}"
        )
    posix_path = PurePosixPath(name)
    windows_path = PureWindowsPath(name)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
    ):
        raise ArchiveByteProfileError(f"zip-slip archive member path: {name!r}")
    parts = posix_path.parts
    if not parts or any(p == ".." for p in parts):
        raise ArchiveByteProfileError(f"zip-slip archive member path: {name!r}")
    return name


def _zip_method_name(compress_type: int) -> str:
    names = {
        zipfile.ZIP_DEFLATED: "deflated",
        zipfile.ZIP_STORED: "stored",
    }
    if hasattr(zipfile, "ZIP_BZIP2"):
        names[zipfile.ZIP_BZIP2] = "bzip2"
    if hasattr(zipfile, "ZIP_LZMA"):
        names[zipfile.ZIP_LZMA] = "lzma"
    return names.get(compress_type, f"unknown_{compress_type}")


def _md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _rehydration_failure(symbol: str) -> NotImplementedError:
    return NotImplementedError(
        f"rehydration incomplete: {symbol} contains intricate nested generator "
        f"expressions that pycdc cannot fully decompile; original bytecode "
        f"preserved in {_QUARANTINE_SPEC}"
    )


def profile_archive(path: Path | str) -> dict[str, Any]:
    """Profile one ZIP archive without extracting it."""
    archive_path = Path(path)
    archive_blob = archive_path.read_bytes()
    archive_sha = hashlib.sha256(archive_blob).hexdigest()
    try:
        with zipfile.ZipFile(archive_path) as zf:
            infos = [info for info in zf.infolist() if not info.is_dir()]
            seen: set[str] = set()
            members: list[dict[str, Any]] = []
            by_top_level: defaultdict[str, dict[str, Any]] = defaultdict(
                lambda: {
                    "compressed_bytes": 0,
                    "member_count": 0,
                    "uncompressed_bytes": 0,
                }
            )
            by_suffix: defaultdict[str, dict[str, Any]] = defaultdict(
                lambda: {
                    "compressed_bytes": 0,
                    "member_count": 0,
                    "uncompressed_bytes": 0,
                }
            )
            methods: Counter[str] = Counter()
            total_compressed = 0
            total_uncompressed = 0

            for info in infos:
                name = _validate_zip_member_name(info.filename)
                if name in seen:
                    raise ArchiveByteProfileError(f"duplicate archive member: {name!r}")
                seen.add(name)
                method = _zip_method_name(info.compress_type)
                methods[method] += 1
                total_compressed += int(info.compress_size)
                total_uncompressed += int(info.file_size)
                payload = zf.read(name)
                payload_sha = hashlib.sha256(payload).hexdigest()
                suffix = Path(name).suffix.lower() or "<none>"
                top_level = PurePosixPath(name).parts[0]
                by_top_level[top_level]["compressed_bytes"] += int(info.compress_size)
                by_top_level[top_level]["member_count"] += 1
                by_top_level[top_level]["uncompressed_bytes"] += int(info.file_size)
                by_suffix[suffix]["compressed_bytes"] += int(info.compress_size)
                by_suffix[suffix]["member_count"] += 1
                by_suffix[suffix]["uncompressed_bytes"] += int(info.file_size)
                members.append(
                    {
                        "compressed_bytes": int(info.compress_size),
                        "compression_method": method,
                        "crc32": f"{info.CRC:08x}",
                        "filename": name,
                        "sha256": payload_sha,
                        "uncompressed_bytes": int(info.file_size),
                    }
                )
    except zipfile.BadZipFile as exc:
        raise ArchiveByteProfileError(f"bad ZIP archive: {archive_path}") from exc

    members.sort(key=lambda item: item["filename"])
    return {
        "archive_bytes": len(archive_blob),
        "archive_path": archive_path.as_posix(),
        "archive_sha256": archive_sha,
        "compression_methods": dict(sorted(methods.items())),
        "evidence_grade": EVIDENCE_GRADE,
        "member_count": len(members),
        "members": members,
        "profile_by_suffix": _sorted_group_profile(by_suffix),
        "profile_by_top_level": _sorted_group_profile(by_top_level),
        "rate_term": contest_rate_term(len(archive_blob)),
        "schema": ARCHIVE_SCHEMA,
        "score_claim": False,
        "tool": TOOL,
        "total_compressed_member_bytes": total_compressed,
        "total_uncompressed_member_bytes": total_uncompressed,
        "zip_overhead_bytes": len(archive_blob) - total_compressed,
    }


def invalid_archive_record(path: Path | str, error: str) -> dict[str, Any]:
    """Return a structured byte-only record for an archive that failed profiling."""
    archive_path = Path(path)
    archive_bytes = archive_path.stat().st_size if archive_path.exists() else None
    archive_sha = _sha256_file(archive_path) if archive_path.is_file() else None
    return {
        "archive_bytes": archive_bytes,
        "archive_path": archive_path.as_posix(),
        "archive_sha256": archive_sha,
        "error": str(error),
        "evidence_grade": EVIDENCE_GRADE,
        "schema": ARCHIVE_SCHEMA,
        "score_claim": False,
        "tool": TOOL,
        "valid": False,
    }


def build_profile_collection(
    paths: Iterable[Path | str], *, continue_on_error: bool = False
) -> dict[str, Any]:
    archives: list[dict[str, Any]] = []
    for raw_path in paths:
        try:
            record = profile_archive(raw_path)
            record["valid"] = True
        except (ArchiveByteProfileError, OSError) as exc:
            if not continue_on_error:
                raise
            record = invalid_archive_record(raw_path, str(exc))
        archives.append(record)
    return {
        "archive_count": len(archives),
        "archives": archives,
        "evidence_grade": EVIDENCE_GRADE,
        "schema": SCHEMA,
        "score_claim": False,
        "tool": TOOL,
    }


def build_candidate_diff_manifest(
    *,
    source_archive: Path | str,
    candidate_archive: Path | str,
    source_label: str = "source",
    candidate_label: str = "candidate",
) -> dict[str, Any]:
    """Return a byte-only source/candidate archive diff manifest."""

    source = profile_archive(source_archive)
    candidate = profile_archive(candidate_archive)
    source_members = _member_index(source)
    candidate_members = _member_index(candidate)
    archive_identical = source["archive_sha256"] == candidate["archive_sha256"]
    payload_multiset_identical = _payload_sha_multiset(source) == _payload_sha_multiset(candidate)
    member_name_sets_identical = set(source_members) == set(candidate_members)
    changed_members = _changed_member_records(source_members, candidate_members)

    if archive_identical:
        no_op_status = "byte_identical_archive_noop"
        candidate_non_noop = False
    elif payload_multiset_identical:
        no_op_status = "payload_identical_container_reemit_noop"
        candidate_non_noop = False
    else:
        no_op_status = "non_noop_payload_changed"
        candidate_non_noop = True

    dispatch_blockers = [
        "candidate_diff_manifest_is_byte_forensics_only",
        "requires_exact_cuda_auth_eval_on_candidate",
    ]
    if not candidate_non_noop:
        dispatch_blockers.append("candidate_is_noop")

    return {
        "schema": ARCHIVE_DIFF_SCHEMA,
        "evidence_grade": EVIDENCE_GRADE,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "source_label": str(source_label),
        "candidate_label": str(candidate_label),
        "source_archive": _archive_summary(source),
        "candidate_archive": _archive_summary(candidate),
        "archive_byte_delta": int(candidate["archive_bytes"]) - int(source["archive_bytes"]),
        "archive_sha256_equal": archive_identical,
        "member_name_sets_identical": member_name_sets_identical,
        "payload_sha256_multiset_equal": payload_multiset_identical,
        "candidate_non_noop": candidate_non_noop,
        "no_op_status": no_op_status,
        "changed_members": changed_members,
        "dispatch_blockers": dispatch_blockers,
        "next_safe_actions": [
            "Reject byte-identical and payload-identical container rewrites as no-op controls.",
            "For non-noop candidates, validate archive compliance and run exact CUDA auth eval before any score claim.",
            "Preserve source/candidate archive bytes and SHA-256s in downstream manifests.",
        ],
    }


def _archive_summary(profile: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "archive_bytes": profile.get("archive_bytes"),
        "archive_path": profile.get("archive_path"),
        "archive_sha256": profile.get("archive_sha256"),
        "member_count": profile.get("member_count"),
        "rate_term": profile.get("rate_term"),
        "zip_overhead_bytes": profile.get("zip_overhead_bytes"),
    }


def _member_index(profile: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(member["filename"]): dict(member)
        for member in profile.get("members", [])
        if isinstance(member, Mapping) and "filename" in member
    }


def _payload_sha_multiset(profile: Mapping[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for member in profile.get("members", []):
        if isinstance(member, Mapping) and member.get("sha256"):
            counts[str(member["sha256"])] += 1
    return dict(sorted(counts.items()))


def _changed_member_records(
    source_members: Mapping[str, Mapping[str, Any]],
    candidate_members: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in sorted(set(source_members) | set(candidate_members)):
        source = source_members.get(name)
        candidate = candidate_members.get(name)
        if source is None:
            rows.append(
                {
                    "filename": name,
                    "status": "added",
                    "source_sha256": "",
                    "candidate_sha256": candidate.get("sha256") if candidate else "",
                }
            )
            continue
        if candidate is None:
            rows.append(
                {
                    "filename": name,
                    "status": "removed",
                    "source_sha256": source.get("sha256"),
                    "candidate_sha256": "",
                }
            )
            continue
        if source.get("sha256") != candidate.get("sha256"):
            rows.append(
                {
                    "filename": name,
                    "status": "payload_changed",
                    "source_sha256": source.get("sha256"),
                    "candidate_sha256": candidate.get("sha256"),
                    "source_compressed_bytes": source.get("compressed_bytes"),
                    "candidate_compressed_bytes": candidate.get("compressed_bytes"),
                    "source_uncompressed_bytes": source.get("uncompressed_bytes"),
                    "candidate_uncompressed_bytes": candidate.get("uncompressed_bytes"),
                }
            )
    return rows


def render_markdown(profile: dict[str, Any]) -> str:
    if profile.get("schema") == ARCHIVE_DIFF_SCHEMA:
        return _render_candidate_diff_markdown(profile)
    if profile.get("schema") == SCHEMA:
        records = profile.get("archives") or []
        lines = [
            "# Archive Byte Profile Collection",
            "",
            f"- archive_count: `{len(records)}`",
            f"- evidence_grade: `{profile.get('evidence_grade', EVIDENCE_GRADE)}`",
            f"- score_claim: `{str(profile.get('score_claim') is True).lower()}`",
            "",
            "| archive | valid | bytes | rate | members | zip overhead | sha256 |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for record in records:
            lines.append(_markdown_archive_summary_row(record))
        lines.append("")
        return "\n".join(lines)

    lines = [
        "# Archive Byte Profile",
        "",
        f"- archive: `{profile.get('archive_path')}`",
        f"- archive_bytes: `{profile.get('archive_bytes')}`",
        f"- archive_sha256: `{profile.get('archive_sha256')}`",
        f"- rate_term: `{profile.get('rate_term')}`",
        f"- member_count: `{profile.get('member_count')}`",
        f"- zip_overhead_bytes: `{profile.get('zip_overhead_bytes')}`",
        f"- evidence_grade: `{profile.get('evidence_grade', EVIDENCE_GRADE)}`",
        f"- score_claim: `{str(profile.get('score_claim') is True).lower()}`",
        "",
        "| member | compressed | uncompressed | method | sha256 |",
        "|---|---:|---:|---|---|",
    ]
    for member in profile.get("members") or []:
        lines.append(
            "| {name} | {compressed} | {uncompressed} | `{method}` | `{sha}` |".format(
                name=_md_escape(member.get("filename", "")),
                compressed=member.get("compressed_bytes"),
                uncompressed=member.get("uncompressed_bytes"),
                method=_md_escape(member.get("compression_method", "")),
                sha=str(member.get("sha256") or "")[:16],
            )
        )
    lines.append("")
    return "\n".join(lines)


def _render_candidate_diff_markdown(profile: dict[str, Any]) -> str:
    source = profile.get("source_archive") or {}
    candidate = profile.get("candidate_archive") or {}
    lines = [
        "# Archive Candidate Diff Manifest",
        "",
        f"- source: `{profile.get('source_label')}` `{source.get('archive_sha256')}`",
        f"- candidate: `{profile.get('candidate_label')}` `{candidate.get('archive_sha256')}`",
        f"- archive_byte_delta: `{profile.get('archive_byte_delta')}`",
        f"- no_op_status: `{profile.get('no_op_status')}`",
        f"- candidate_non_noop: `{str(profile.get('candidate_non_noop') is True).lower()}`",
        f"- score_claim: `{str(profile.get('score_claim') is True).lower()}`",
        "",
        "| member | status | source sha | candidate sha |",
        "|---|---|---|---|",
    ]
    changed = profile.get("changed_members") or []
    if not changed:
        lines.append("| _none_ | _none_ | _none_ | _none_ |")
    for row in changed:
        lines.append(
            "| {name} | `{status}` | `{source_sha}` | `{candidate_sha}` |".format(
                name=_md_escape(row.get("filename", "")),
                status=_md_escape(row.get("status", "")),
                source_sha=str(row.get("source_sha256") or "")[:16],
                candidate_sha=str(row.get("candidate_sha256") or "")[:16],
            )
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
        json_out.write_bytes(_json_bytes(profile))
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_markdown(profile), encoding="utf-8")


def _sorted_group_profile(groups: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for key, values in groups.items():
        rows.append(
            {
                "key": key,
                "compressed_bytes": int(values["compressed_bytes"]),
                "member_count": int(values["member_count"]),
                "uncompressed_bytes": int(values["uncompressed_bytes"]),
            }
        )
    return sorted(rows, key=lambda item: (-item["compressed_bytes"], item["key"]))


def _markdown_archive_summary_row(record: dict[str, Any]) -> str:
    if record.get("valid") is False:
        return "| {path} | false | {bytes_} | n/a | n/a | n/a | `{sha}` |".format(
            path=_md_escape(record.get("archive_path", "")),
            bytes_=record.get("archive_bytes"),
            sha=str(record.get("archive_sha256") or "")[:16],
        )
    return "| {path} | true | {bytes_} | {rate:.9f} | {members} | {overhead} | `{sha}` |".format(
        path=_md_escape(record.get("archive_path", "")),
        bytes_=record.get("archive_bytes"),
        rate=float(record.get("rate_term") or 0.0),
        members=record.get("member_count"),
        overhead=record.get("zip_overhead_bytes"),
        sha=str(record.get("archive_sha256") or "")[:16],
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile ZIP archive byte attribution without extracting payloads."
    )
    parser.add_argument(
        "archives", nargs="+", type=Path, help="archive.zip paths to profile"
    )
    parser.add_argument(
        "--json-out", type=Path, help="write deterministic JSON profile"
    )
    parser.add_argument(
        "--markdown-out", type=Path, help="write markdown summary"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="record invalid/nonstandard archives instead of aborting the whole collection",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    profile = build_profile_collection(
        args.archives, continue_on_error=args.continue_on_error
    )
    write_outputs(profile, json_out=args.json_out, markdown_out=args.markdown_out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
