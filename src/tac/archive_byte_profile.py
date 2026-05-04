"""Deterministic ZIP byte attribution for contest archive research.

The profiler is intentionally byte-only: it does not extract archive payloads,
inflate contest outputs, load scorer models, or make score claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Sequence


SCHEMA = "archive_byte_profile_collection_v1"
ARCHIVE_SCHEMA = "archive_byte_profile_v1"
TOOL = "experiments/profile_archive_bytes.py"
EVIDENCE_GRADE = "byte_profile_only"
CONTEST_ORIGINAL_BYTES = 37_545_489
RATE_TERM_COEFFICIENT = 25.0 / CONTEST_ORIGINAL_BYTES


class ArchiveByteProfileError(ValueError):
    """Raised when an archive cannot be safely profiled."""


def contest_rate_term(byte_count: int) -> float:
    """Return the contest formula rate contribution for ``byte_count``."""

    return 25.0 * int(byte_count) / CONTEST_ORIGINAL_BYTES


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_zip_member_name(name: str) -> str:
    if not name:
        raise ArchiveByteProfileError("archive member name is empty")
    if "\x00" in name:
        raise ArchiveByteProfileError(f"archive member contains NUL byte: {name!r}")
    if "\\" in name:
        raise ArchiveByteProfileError(f"archive member uses backslashes: {name!r}")

    posix_path = PurePosixPath(name)
    windows_path = PureWindowsPath(name)
    if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise ArchiveByteProfileError(f"zip-slip archive member path: {name!r}")

    parts = posix_path.parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise ArchiveByteProfileError(f"zip-slip archive member path: {name!r}")
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


def _extension_group(name: str, is_dir: bool) -> str:
    if is_dir:
        return "(directory)"
    suffixes = PurePosixPath(name).suffixes
    if not suffixes:
        return "(no_extension)"
    if len(suffixes) >= 2 and suffixes[-1].lower() in {".br", ".xz", ".zst", ".gz", ".bz2"}:
        return "".join(s.lower() for s in suffixes[-2:])
    return suffixes[-1].lower()


def _path_group(name: str) -> str:
    parts = PurePosixPath(name).parts
    return "(root)" if len(parts) <= 1 else parts[0]


def _top_bytes(histogram: Sequence[int], total: int, *, limit: int = 10) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value, count in sorted(
        enumerate(histogram),
        key=lambda item: (-item[1], item[0]),
    )[:limit]:
        if count == 0:
            break
        rows.append(
            {
                "byte": value,
                "hex": f"0x{value:02x}",
                "count": int(count),
                "fraction": 0.0 if total == 0 else round(count / total, 12),
            }
        )
    return rows


def _histogram_stats(histogram: Sequence[int], total: int) -> dict[str, Any]:
    entropy = 0.0
    if total:
        for count in histogram:
            if count:
                p = count / total
                entropy -= p * math.log2(p)
    printable = sum(histogram[i] for i in range(32, 127))
    zero_count = histogram[0]
    return {
        "histogram_counts": [int(count) for count in histogram],
        "shannon_entropy_bits_per_byte": round(entropy, 12),
        "zero_order_entropy_bytes": int(math.ceil(entropy * total / 8.0)),
        "unique_byte_count": int(sum(1 for count in histogram if count)),
        "zero_byte_count": int(zero_count),
        "zero_byte_fraction": 0.0 if total == 0 else round(zero_count / total, 12),
        "printable_ascii_fraction": 0.0 if total == 0 else round(printable / total, 12),
        "top_bytes": _top_bytes(histogram, total),
    }


def _read_member_stats(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> tuple[str, dict[str, Any]]:
    digest = hashlib.sha256()
    histogram = [0] * 256
    total = 0
    with zf.open(info, "r") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            total += len(chunk)
            for value in chunk:
                histogram[value] += 1
    if total != info.file_size:
        raise ArchiveByteProfileError(
            f"ZIP member size mismatch for {info.filename!r}: read {total}, expected {info.file_size}"
        )
    return digest.hexdigest(), _histogram_stats(histogram, total)


def _duplicate_rows(counter: Counter[str]) -> list[dict[str, Any]]:
    return [
        {"name": name, "count": int(count)}
        for name, count in sorted(counter.items(), key=lambda item: (item[0], item[1]))
        if count > 1
    ]


def _totals_record(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    compressed = sum(int(row["compressed_size"]) for row in rows)
    uncompressed = sum(int(row["uncompressed_size"]) for row in rows)
    return {
        "name": name,
        "member_count": len(rows),
        "compressed_size": compressed,
        "uncompressed_size": uncompressed,
        "rate_term": round(contest_rate_term(compressed), 12),
    }


def profile_archive(path: Path | str) -> dict[str, Any]:
    """Profile one ZIP archive without extracting it."""

    archive_path = Path(path)
    if not archive_path.is_file():
        raise FileNotFoundError(f"archive not found: {archive_path}")
    total_bytes = archive_path.stat().st_size
    archive_sha256 = _sha256_file(archive_path)

    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            infos = zf.infolist()
            if not infos:
                raise ArchiveByteProfileError(f"archive contains no members: {archive_path}")
            for info in infos:
                _validate_zip_member_name(info.filename)

            name_counts = Counter(info.filename for info in infos)
            seen_by_name: Counter[str] = Counter()
            members: list[dict[str, Any]] = []
            hash_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)

            for info in infos:
                seen_by_name[info.filename] += 1
                occurrence = seen_by_name[info.filename]
                is_dir = info.is_dir()
                sha256 = None
                histogram_stats: dict[str, Any] | None = None
                if not is_dir:
                    sha256, histogram_stats = _read_member_stats(zf, info)

                filename_bytes = info.filename.encode("utf-8")
                local_header_bytes = 30 + len(filename_bytes) + len(info.extra)
                central_header_bytes = (
                    46 + len(filename_bytes) + len(info.extra) + len(info.comment or b"")
                )
                compression_ratio = None
                if info.file_size:
                    compression_ratio = round(info.compress_size / info.file_size, 12)
                member = {
                    "name": info.filename,
                    "occurrence": occurrence,
                    "is_dir": is_dir,
                    "compressed_size": int(info.compress_size),
                    "uncompressed_size": int(info.file_size),
                    "method": _zip_method_name(info.compress_type),
                    "method_id": int(info.compress_type),
                    "crc32": f"{info.CRC:08x}",
                    "sha256": sha256,
                    "histogram": histogram_stats,
                    "extension_group": _extension_group(info.filename, is_dir),
                    "path_group": _path_group(info.filename),
                    "compression_ratio": compression_ratio,
                    "header_offset": int(info.header_offset),
                    "local_header_bytes_estimate": int(local_header_bytes),
                    "central_directory_header_bytes_estimate": int(central_header_bytes),
                    "rate_term": round(contest_rate_term(info.compress_size), 12),
                }
                members.append(member)
                if sha256 is not None:
                    hash_groups[(sha256, int(info.file_size))].append(
                        {
                            "name": info.filename,
                            "occurrence": occurrence,
                            "uncompressed_size": int(info.file_size),
                            "compressed_size": int(info.compress_size),
                        }
                    )
    except zipfile.BadZipFile as exc:
        raise ArchiveByteProfileError(f"bad ZIP archive: {archive_path}") from exc

    members.sort(key=lambda row: (row["name"], row["occurrence"], row["header_offset"]))
    compressed_payload_bytes = sum(int(row["compressed_size"]) for row in members)
    local_header_bytes = sum(int(row["local_header_bytes_estimate"]) for row in members)
    central_header_bytes = sum(
        int(row["central_directory_header_bytes_estimate"]) for row in members
    )
    archive_non_payload_bytes = total_bytes - compressed_payload_bytes
    unexplained_overhead = archive_non_payload_bytes - local_header_bytes - central_header_bytes

    extension_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    path_group_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for member in members:
        extension_rows[str(member["extension_group"])].append(member)
        path_group_rows[str(member["path_group"])].append(member)

    duplicate_payload_hashes = []
    for (sha256, size), rows in sorted(hash_groups.items(), key=lambda item: (item[0][0], item[0][1])):
        if len(rows) > 1:
            duplicate_payload_hashes.append(
                {
                    "sha256": sha256,
                    "uncompressed_size": size,
                    "members": sorted(rows, key=lambda row: (row["name"], row["occurrence"])),
                    "count": len(rows),
                }
            )

    return {
        "schema": ARCHIVE_SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "evidence_grade": EVIDENCE_GRADE,
        "valid_profile": True,
        "archive_path": str(archive_path),
        "archive_name": archive_path.name,
        "archive_sha256": archive_sha256,
        "total_bytes": int(total_bytes),
        "rate_term": round(contest_rate_term(total_bytes), 12),
        "member_count": len(members),
        "members": members,
        "zip_overhead_estimate": {
            "archive_non_payload_bytes": int(archive_non_payload_bytes),
            "compressed_payload_bytes": int(compressed_payload_bytes),
            "local_header_bytes_estimate": int(local_header_bytes),
            "central_directory_header_bytes_estimate": int(central_header_bytes),
            "unassigned_overhead_bytes_estimate": int(unexplained_overhead),
            "rate_term": round(contest_rate_term(max(archive_non_payload_bytes, 0)), 12),
        },
        "duplicate_detection": {
            "duplicate_member_names": _duplicate_rows(name_counts),
            "duplicate_payload_hashes": duplicate_payload_hashes,
            "has_duplicate_member_names": any(count > 1 for count in name_counts.values()),
            "has_duplicate_payload_hashes": bool(duplicate_payload_hashes),
        },
        "extension_totals": sorted(
            (_totals_record(name, rows) for name, rows in extension_rows.items()),
            key=lambda row: (-row["compressed_size"], row["name"]),
        ),
        "path_group_totals": sorted(
            (_totals_record(name, rows) for name, rows in path_group_rows.items()),
            key=lambda row: (-row["compressed_size"], row["name"]),
        ),
        "top_contributors": sorted(
            (
                {
                    "name": row["name"],
                    "occurrence": row["occurrence"],
                    "compressed_size": row["compressed_size"],
                    "uncompressed_size": row["uncompressed_size"],
                    "method": row["method"],
                    "rate_term": row["rate_term"],
                }
                for row in members
            ),
            key=lambda row: (-row["compressed_size"], row["name"], row["occurrence"]),
        )[:20],
    }


def invalid_archive_record(path: Path | str, error: BaseException) -> dict[str, Any]:
    """Return a structured byte-only record for an archive that failed profiling."""

    archive_path = Path(path)
    total_bytes = archive_path.stat().st_size if archive_path.is_file() else None
    sha256 = _sha256_file(archive_path) if archive_path.is_file() else None
    return {
        "schema": "archive_byte_profile_invalid_v1",
        "tool": TOOL,
        "score_claim": False,
        "evidence_grade": "invalid_archive_byte_profile_only",
        "archive_path": str(archive_path),
        "archive_name": archive_path.name,
        "archive_sha256": sha256,
        "total_bytes": total_bytes,
        "rate_term": round(contest_rate_term(total_bytes), 12)
        if total_bytes is not None
        else None,
        "member_count": 0,
        "valid_profile": False,
        "error_type": type(error).__name__,
        "error": str(error),
        "members": [],
        "top_contributors": [],
        "extension_totals": [],
        "path_group_totals": [],
        "duplicate_detection": {
            "duplicate_member_names": [],
            "duplicate_payload_hashes": [],
            "has_duplicate_member_names": False,
            "has_duplicate_payload_hashes": False,
        },
        "zip_overhead_estimate": {
            "archive_non_payload_bytes": None,
            "compressed_payload_bytes": None,
            "local_header_bytes_estimate": None,
            "central_directory_header_bytes_estimate": None,
            "unassigned_overhead_bytes_estimate": None,
            "rate_term": None,
        },
    }


def build_profile_collection(
    paths: Iterable[Path | str],
    *,
    continue_on_error: bool = False,
) -> dict[str, Any]:
    archives = []
    invalid_archives = []
    for path in paths:
        try:
            archives.append(profile_archive(path))
        except (ArchiveByteProfileError, FileNotFoundError, PermissionError, OSError) as exc:
            if not continue_on_error:
                raise
            record = invalid_archive_record(path, exc)
            archives.append(record)
            invalid_archives.append(record)
    if not archives:
        raise ArchiveByteProfileError("at least one archive path is required")

    cross_hashes: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for archive in archives:
        for member in archive["members"]:
            sha256 = member.get("sha256")
            if sha256 is None:
                continue
            cross_hashes[(sha256, int(member["uncompressed_size"]))].append(
                {
                    "archive_path": archive["archive_path"],
                    "name": member["name"],
                    "occurrence": member["occurrence"],
                    "uncompressed_size": member["uncompressed_size"],
                    "compressed_size": member["compressed_size"],
                }
            )

    cross_duplicates = []
    for (sha256, size), rows in sorted(cross_hashes.items(), key=lambda item: (item[0][0], item[0][1])):
        archive_paths = {row["archive_path"] for row in rows}
        if len(rows) > 1 and len(archive_paths) > 1:
            cross_duplicates.append(
                {
                    "sha256": sha256,
                    "uncompressed_size": size,
                    "members": sorted(
                        rows,
                        key=lambda row: (row["archive_path"], row["name"], row["occurrence"]),
                    ),
                    "count": len(rows),
                    "archive_count": len(archive_paths),
                }
            )

    total_bytes = sum(int(archive["total_bytes"]) for archive in archives)
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "evidence_grade": EVIDENCE_GRADE,
        "rate_formula": "25 * bytes / 37545489",
        "rate_denominator_bytes": CONTEST_ORIGINAL_BYTES,
        "archive_count": len(archives),
        "invalid_archive_count": len(invalid_archives),
        "invalid_archives": invalid_archives,
        "total_bytes": int(total_bytes),
        "rate_term": round(contest_rate_term(total_bytes), 12),
        "archives": archives,
        "cross_archive_duplicate_payload_hashes": cross_duplicates,
    }


def _md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _md_optional_int(value: Any) -> str:
    return "" if value is None else str(value)


def _md_optional_float(value: Any, *, digits: int = 9) -> str:
    return "" if value is None else f"{float(value):.{digits}f}"


def render_markdown(profile: dict[str, Any]) -> str:
    lines = [
        "# Archive Byte Profile",
        "",
        f"- schema: `{profile['schema']}`",
        f"- evidence_grade: `{profile['evidence_grade']}`",
        f"- score_claim: `{profile['score_claim']}`",
        f"- rate formula: `{profile['rate_formula']}`",
        f"- archives: `{profile['archive_count']}`",
        f"- invalid archives: `{profile.get('invalid_archive_count', 0)}`",
        "",
        "This is byte attribution only. It does not inflate payloads, run scorers, "
        "dispatch jobs, promote methods, or claim contest score.",
        "",
        "## Archives",
        "",
        "| archive | total bytes | rate term | members | ZIP overhead est. |",
        "|---|---:|---:|---:|---:|",
    ]
    for archive in profile["archives"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_escape(archive["archive_path"]),
                    _md_optional_int(archive.get("total_bytes")),
                    _md_optional_float(archive.get("rate_term")),
                    _md_optional_int(archive.get("member_count", 0)),
                    _md_optional_int(
                        archive.get("zip_overhead_estimate", {}).get(
                            "archive_non_payload_bytes"
                        )
                    ),
                ]
            )
            + " |"
        )

    if profile.get("invalid_archives"):
        lines.extend(["", "## Invalid Archives", "", "| archive | bytes | error |", "|---|---:|---|"])
        for archive in profile["invalid_archives"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _md_escape(archive["archive_path"]),
                        "" if archive["total_bytes"] is None else str(archive["total_bytes"]),
                        _md_escape(f"{archive['error_type']}: {archive['error']}"),
                    ]
                )
                + " |"
            )

    for archive in profile["archives"]:
        if archive.get("valid_profile") is False:
            continue
        lines.extend(
            [
                "",
                f"## {_md_escape(archive['archive_name'])}",
                "",
                f"- path: `{archive['archive_path']}`",
                f"- sha256: `{archive['archive_sha256']}`",
                f"- duplicate member names: `{archive['duplicate_detection']['has_duplicate_member_names']}`",
                f"- duplicate payload hashes: `{archive['duplicate_detection']['has_duplicate_payload_hashes']}`",
                "",
                "### Top Contributors",
                "",
                "| member | compressed | uncompressed | method | rate term |",
                "|---|---:|---:|---|---:|",
            ]
        )
        for row in archive["top_contributors"][:12]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _md_escape(row["name"]),
                        str(row["compressed_size"]),
                        str(row["uncompressed_size"]),
                        _md_escape(row["method"]),
                        f"{row['rate_term']:.9f}",
                    ]
                )
                + " |"
            )

        lines.extend(
            [
                "",
                "### Extension Totals",
                "",
                "| group | members | compressed | uncompressed | rate term |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in archive["extension_totals"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _md_escape(row["name"]),
                        str(row["member_count"]),
                        str(row["compressed_size"]),
                        str(row["uncompressed_size"]),
                        f"{row['rate_term']:.9f}",
                    ]
                )
                + " |"
            )

        if archive["duplicate_detection"]["duplicate_member_names"]:
            lines.extend(["", "### Duplicate Member Names", "", "| member | count |", "|---|---:|"])
            for row in archive["duplicate_detection"]["duplicate_member_names"]:
                lines.append(f"| {_md_escape(row['name'])} | {row['count']} |")

        if archive["duplicate_detection"]["duplicate_payload_hashes"]:
            lines.extend(
                [
                    "",
                    "### Duplicate Payload Hashes",
                    "",
                    "| sha256 | size | count | members |",
                    "|---|---:|---:|---|",
                ]
            )
            for row in archive["duplicate_detection"]["duplicate_payload_hashes"]:
                members = ", ".join(
                    f"{member['name']}#{member['occurrence']}" for member in row["members"]
                )
                lines.append(
                    f"| `{row['sha256']}` | {row['uncompressed_size']} | {row['count']} | {_md_escape(members)} |"
                )

    if profile["cross_archive_duplicate_payload_hashes"]:
        lines.extend(
            [
                "",
                "## Cross-Archive Duplicate Payloads",
                "",
                "| sha256 | size | archives | members |",
                "|---|---:|---:|---|",
            ]
        )
        for row in profile["cross_archive_duplicate_payload_hashes"]:
            members = ", ".join(
                f"{Path(member['archive_path']).name}:{member['name']}#{member['occurrence']}"
                for member in row["members"]
            )
            lines.append(
                f"| `{row['sha256']}` | {row['uncompressed_size']} | {row['archive_count']} | {_md_escape(members)} |"
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
        json_out.write_bytes(_json_bytes(profile))
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_markdown(profile), encoding="utf-8")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile ZIP archive byte attribution without extracting payloads."
    )
    parser.add_argument("archives", nargs="+", type=Path, help="archive.zip paths to profile")
    parser.add_argument("--json-out", type=Path, help="write deterministic JSON profile")
    parser.add_argument("--markdown-out", type=Path, help="write markdown summary")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="record invalid/nonstandard archives instead of aborting the whole collection",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    profile = build_profile_collection(args.archives, continue_on_error=args.continue_on_error)
    write_outputs(profile, json_out=args.json_out, markdown_out=args.markdown_out)
    if args.json_out is None and args.markdown_out is None:
        print(json.dumps(profile, indent=2, sort_keys=True, allow_nan=False))
    elif args.markdown_out is None:
        print(render_markdown(profile), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
