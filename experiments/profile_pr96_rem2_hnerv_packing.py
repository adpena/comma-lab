#!/usr/bin/env python3
"""Profile and safely repack the public PR96 rem2_HNeRV archive.

The emitted archives are byte-only candidates: member payload bytes are
preserved for all kept members, ZIP methods are chosen deterministically, and
the optional unused-member candidate removes only members that are not read by
the visible runtime. This tool does not inflate outputs or make score claims.
"""

from __future__ import annotations

import argparse
import ast
import binascii
import dataclasses
import hashlib
import io
import json
import lzma
import struct
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Sequence

import brotli


SCHEMA = "pr96_rem2_hnerv_packing_profile_v1"
TOOL = "experiments/profile_pr96_rem2_hnerv_packing.py"
EVIDENCE_GRADE = "byte_exact_repack_candidate_until_exact_cuda_eval"
CONTEST_ORIGINAL_BYTES = 37_545_489
DEFAULT_ARCHIVE = (
    "experiments/results/leaderboard_intel_20260504_codex/pr96_archive.zip"
)
DEFAULT_RUNTIME_PY = (
    "experiments/results/leaderboard_intel_20260504_codex/pr96_runtime/inflate.py"
)
DEFAULT_OUTPUT_DIR = "experiments/results/pr96_rem2_hnerv_packing_profile_20260504_codex"


class PR96ProfileError(ValueError):
    """Raised when the PR96 archive cannot be safely profiled or repacked."""


@dataclasses.dataclass(frozen=True)
class SourceMember:
    name: str
    data: bytes
    raw_compressed: bytes
    method_id: int
    method: str
    compressed_size: int
    uncompressed_size: int
    crc32_int: int
    crc32: str
    sha256: str
    header_offset: int
    date_time: tuple[int, int, int, int, int, int]

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "method": self.method,
            "method_id": self.method_id,
            "compressed_size": self.compressed_size,
            "uncompressed_size": self.uncompressed_size,
            "crc32": self.crc32,
            "sha256": self.sha256,
            "header_offset": self.header_offset,
            "date_time": list(self.date_time),
        }


@dataclasses.dataclass(frozen=True)
class CompressionChoice:
    method_id: int
    method: str
    compresslevel: int | None
    compressed_size: int
    strategy: str
    compressed_payload: bytes = dataclasses.field(repr=False)

    def manifest(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "method_id": self.method_id,
            "compresslevel": self.compresslevel,
            "compressed_size": self.compressed_size,
            "strategy": self.strategy,
        }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contest_rate_term(byte_count: int) -> float:
    return 25.0 * int(byte_count) / CONTEST_ORIGINAL_BYTES


def method_name(method_id: int) -> str:
    if method_id == zipfile.ZIP_STORED:
        return "stored"
    if method_id == zipfile.ZIP_DEFLATED:
        return "deflated"
    return f"method_{method_id}"


def dos_datetime() -> tuple[int, int]:
    return 0, (1 << 5) | 1


def raw_member_payload(archive_bytes: bytes, info: zipfile.ZipInfo) -> bytes:
    offset = int(info.header_offset)
    header = archive_bytes[offset : offset + 30]
    if len(header) != 30:
        raise PR96ProfileError(f"truncated local header for {info.filename!r}")
    (
        signature,
        _version_needed,
        flags,
        method_id,
        _mod_time,
        _mod_date,
        _crc32,
        _compressed_size,
        _uncompressed_size,
        name_len,
        extra_len,
    ) = struct.unpack("<IHHHHHIIIHH", header)
    if signature != 0x04034B50:
        raise PR96ProfileError(f"bad local header signature for {info.filename!r}")
    if flags & 0x0001:
        raise PR96ProfileError(f"encrypted ZIP member is not supported: {info.filename!r}")
    if method_id != info.compress_type:
        raise PR96ProfileError(f"local/central compression mismatch for {info.filename!r}")
    name_start = offset + 30
    name_end = name_start + name_len
    local_name = archive_bytes[name_start:name_end].decode("utf-8")
    if local_name != info.filename:
        raise PR96ProfileError(
            f"local/central member name mismatch: {local_name!r} != {info.filename!r}"
        )
    payload_start = name_end + extra_len
    payload_end = payload_start + int(info.compress_size)
    payload = archive_bytes[payload_start:payload_end]
    if len(payload) != info.compress_size:
        raise PR96ProfileError(f"truncated compressed payload for {info.filename!r}")
    return payload


def validate_member_name(name: str) -> str:
    if not name:
        raise PR96ProfileError("empty ZIP member name")
    if "\x00" in name:
        raise PR96ProfileError(f"NUL byte in ZIP member name: {name!r}")
    if "\\" in name:
        raise PR96ProfileError(f"backslash in ZIP member name: {name!r}")
    posix_path = PurePosixPath(name)
    windows_path = PureWindowsPath(name)
    if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise PR96ProfileError(f"zip-slip ZIP member path: {name!r}")
    if any(part in ("", ".", "..") for part in posix_path.parts):
        raise PR96ProfileError(f"zip-slip ZIP member path: {name!r}")
    return name


def read_archive(path: Path) -> list[SourceMember]:
    if not path.is_file():
        raise FileNotFoundError(f"archive not found: {path}")
    archive_bytes = path.read_bytes()
    with zipfile.ZipFile(path, "r") as zf:
        infos = zf.infolist()
        if not infos:
            raise PR96ProfileError(f"archive contains no members: {path}")
        counts = Counter(info.filename for info in infos)
        duplicates = sorted(name for name, count in counts.items() if count > 1)
        if duplicates:
            raise PR96ProfileError(f"duplicate ZIP member names are forbidden: {duplicates}")

        members: list[SourceMember] = []
        for info in infos:
            validate_member_name(info.filename)
            data = zf.read(info)
            if len(data) != info.file_size:
                raise PR96ProfileError(
                    f"member size mismatch for {info.filename!r}: "
                    f"read {len(data)}, expected {info.file_size}"
                )
            members.append(
                SourceMember(
                    name=info.filename,
                    data=data,
                    raw_compressed=raw_member_payload(archive_bytes, info),
                    method_id=int(info.compress_type),
                    method=method_name(int(info.compress_type)),
                    compressed_size=int(info.compress_size),
                    uncompressed_size=int(info.file_size),
                    crc32_int=int(info.CRC),
                    crc32=f"{info.CRC:08x}",
                    sha256=sha256_bytes(data),
                    header_offset=int(info.header_offset),
                    date_time=tuple(info.date_time),
                )
            )
    return members


def read_exact(handle: io.BytesIO, size: int, label: str) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise PR96ProfileError(f"truncated {label}: read {len(data)}, expected {size}")
    return data


def product(values: Sequence[int]) -> int:
    out = 1
    for value in values:
        out *= value
    return out


def parse_brotli_decoder_records(raw: bytes) -> list[dict[str, Any]]:
    handle = io.BytesIO(raw)
    n_records = struct.unpack("<I", read_exact(handle, 4, "brotli record count"))[0]
    records: list[dict[str, Any]] = []
    for index in range(n_records):
        name_len = struct.unpack("<I", read_exact(handle, 4, "record name length"))[0]
        name = read_exact(handle, name_len, "record name").decode("utf-8")
        n_dims = struct.unpack("<I", read_exact(handle, 4, "record dim count"))[0]
        shape = tuple(
            struct.unpack("<I", read_exact(handle, 4, "record dim"))[0]
            for _ in range(n_dims)
        )
        scale = struct.unpack("<f", read_exact(handle, 4, "record scale"))[0]
        count = product(shape)
        q = read_exact(handle, count, "record quantized payload")
        records.append(
            {
                "index": index,
                "name": name,
                "shape": list(shape),
                "scale": scale,
                "quantized_bytes": len(q),
                "sha256": sha256_bytes(q),
            }
        )
    rest = handle.read()
    if rest:
        raise PR96ProfileError(f"trailing bytes in brotli decoder stream: {len(rest)}")
    return records


def decompress_histogram(comp_id: int, blob: bytes) -> bytes:
    if comp_id == 0:
        return lzma.decompress(blob)
    if comp_id == 2:
        return brotli.decompress(blob)
    if comp_id == 1:
        import zstandard

        return zstandard.ZstdDecompressor().decompress(blob)
    raise PR96ProfileError(f"unknown histogram codec id {comp_id}")


def parse_range_meta(raw: bytes) -> list[dict[str, Any]]:
    handle = io.BytesIO(raw)
    n_records = struct.unpack("<I", read_exact(handle, 4, "range record count"))[0]
    records: list[dict[str, Any]] = []
    for index in range(n_records):
        name_len = struct.unpack("<I", read_exact(handle, 4, "range name length"))[0]
        name = read_exact(handle, name_len, "range name").decode("utf-8")
        n_dims = struct.unpack("<I", read_exact(handle, 4, "range dim count"))[0]
        shape = tuple(
            struct.unpack("<I", read_exact(handle, 4, "range dim"))[0]
            for _ in range(n_dims)
        )
        scale = struct.unpack("<f", read_exact(handle, 4, "range scale"))[0]
        count = struct.unpack("<I", read_exact(handle, 4, "range count"))[0]
        records.append(
            {
                "index": index,
                "name": name,
                "shape": list(shape),
                "scale": scale,
                "quantized_count": count,
            }
        )
    rest = handle.read()
    if rest:
        raise PR96ProfileError(f"trailing bytes in range metadata stream: {len(rest)}")
    return records


def parse_decoder_payload(blob: bytes) -> dict[str, Any]:
    handle = io.BytesIO(blob)
    br_len, hist_len, meta_len, lengths_len, comp_id = struct.unpack(
        "<IIIIB", read_exact(handle, 17, "decoder header")
    )
    br_bytes = read_exact(handle, br_len, "brotli decoder section")
    hist_bytes = read_exact(handle, hist_len, "histogram section")
    meta_bytes = read_exact(handle, meta_len, "range metadata section")
    lengths_bytes = read_exact(handle, lengths_len, "range lengths section")
    coded_bytes = handle.read()

    br_raw = brotli.decompress(br_bytes)
    br_records = parse_brotli_decoder_records(br_raw)
    summary: dict[str, Any] = {
        "format": "pr96_rem2_hnerv_decoder_v1",
        "total_bytes": len(blob),
        "sha256": sha256_bytes(blob),
        "header": {
            "br_len": br_len,
            "hist_len": hist_len,
            "meta_len": meta_len,
            "lengths_len": lengths_len,
            "comp_id": comp_id,
            "coded_bytes": len(coded_bytes),
        },
        "brotli_records": {
            "count": len(br_records),
            "raw_bytes": len(br_raw),
            "quantized_bytes": sum(int(record["quantized_bytes"]) for record in br_records),
            "records": br_records,
        },
    }

    if hist_len == 0:
        summary["range_coded_records"] = {
            "count": 0,
            "histogram_raw_bytes": 0,
            "lengths": [],
            "coded_bytes_consumed": 0,
            "coded_bytes_trailing": len(coded_bytes),
            "records": [],
        }
        return summary

    hist_raw = decompress_histogram(comp_id, hist_bytes)
    meta_raw = brotli.decompress(meta_bytes)
    lengths_raw = brotli.decompress(lengths_bytes)
    if len(lengths_raw) % 4:
        raise PR96ProfileError(f"range lengths stream is not uint32-aligned: {len(lengths_raw)}")
    lengths = list(struct.unpack(f"<{len(lengths_raw) // 4}I", lengths_raw))
    range_records = parse_range_meta(meta_raw)
    if len(lengths) != len(range_records):
        raise PR96ProfileError(
            f"range length count mismatch: {len(lengths)} lengths, "
            f"{len(range_records)} records"
        )
    coded_consumed = sum(lengths)
    if coded_consumed != len(coded_bytes):
        raise PR96ProfileError(
            f"range coded byte mismatch: lengths sum {coded_consumed}, "
            f"coded payload {len(coded_bytes)}"
        )
    summary["range_coded_records"] = {
        "count": len(range_records),
        "histogram_codec": {0: "lzma", 1: "zstd", 2: "brotli"}.get(comp_id, f"id_{comp_id}"),
        "histogram_raw_bytes": len(hist_raw),
        "histogram_count": len(hist_raw) // (256 * 2) if len(hist_raw) % (256 * 2) == 0 else None,
        "meta_raw_bytes": len(meta_raw),
        "lengths_raw_bytes": len(lengths_raw),
        "lengths": lengths,
        "coded_bytes_consumed": coded_consumed,
        "coded_bytes_trailing": 0,
        "records": [
            {
                **record,
                "coded_bytes": lengths[index],
            }
            for index, record in enumerate(range_records)
        ],
    }
    return summary


def parse_latents_payload(blob: bytes) -> dict[str, Any]:
    handle = io.BytesIO(blob)
    n_rows, n_dim = struct.unpack("<II", read_exact(handle, 8, "latent header"))
    mins = read_exact(handle, n_dim * 2, "latent mins")
    scales = read_exact(handle, n_dim * 2, "latent scales")
    q = read_exact(handle, n_rows * n_dim, "latent quantized payload")
    rest = handle.read()
    if rest:
        raise PR96ProfileError(f"trailing bytes in latents payload: {len(rest)}")
    return {
        "format": "pr96_rem2_hnerv_latents_v1",
        "total_bytes": len(blob),
        "sha256": sha256_bytes(blob),
        "n_rows": n_rows,
        "n_dim": n_dim,
        "mins_f16_bytes": len(mins),
        "scales_f16_bytes": len(scales),
        "quantized_bytes": len(q),
        "quantized_sha256": sha256_bytes(q),
    }


def archive_read_set(runtime_py: Path | None) -> dict[str, Any]:
    if runtime_py is None:
        return {
            "runtime_py": None,
            "runtime_py_exists": False,
            "read_members": [],
            "dynamic_archive_access": True,
            "safe_for_unused_member_removal": False,
            "notes": ["no runtime path supplied"],
        }
    if not runtime_py.is_file():
        return {
            "runtime_py": str(runtime_py),
            "runtime_py_exists": False,
            "read_members": [],
            "dynamic_archive_access": True,
            "safe_for_unused_member_removal": False,
            "notes": ["runtime path missing"],
        }

    tree = ast.parse(runtime_py.read_text(encoding="utf-8"))
    read_members: set[str] = set()
    dynamic_access = False
    notes: list[str] = []

    def archive_member_literal(expr: ast.AST) -> str | None:
        if not isinstance(expr, ast.BinOp) or not isinstance(expr.op, ast.Div):
            return None
        if not isinstance(expr.left, ast.Name) or expr.left.id != "archive_dir":
            return None
        if not isinstance(expr.right, ast.Constant) or not isinstance(expr.right.value, str):
            return None
        return expr.right.value

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            nonlocal dynamic_access
            func = node.func
            if isinstance(func, ast.Attribute):
                if func.attr == "read_bytes":
                    member = archive_member_literal(func.value)
                    if member is None:
                        dynamic_access = True
                        notes.append("dynamic read_bytes call could not be resolved")
                    else:
                        read_members.add(member)
                if (
                    isinstance(func.value, ast.Name)
                    and func.value.id == "archive_dir"
                    and func.attr in {"iterdir", "glob", "rglob"}
                ):
                    dynamic_access = True
                    notes.append(f"archive_dir.{func.attr}() prevents unused-member removal")
            self.generic_visit(node)

    Visitor().visit(tree)
    safe = bool(read_members) and not dynamic_access
    return {
        "runtime_py": str(runtime_py),
        "runtime_py_exists": True,
        "runtime_py_sha256": sha256_file(runtime_py),
        "read_members": sorted(read_members),
        "dynamic_archive_access": dynamic_access,
        "safe_for_unused_member_removal": safe,
        "notes": notes,
    }


def generated_deflate_payload(name: str, data: bytes, level: int) -> bytes:
    buffer = io.BytesIO()
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(buffer, "w", allowZip64=False) as zf:
        zf.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=level)
    with zipfile.ZipFile(io.BytesIO(buffer.getvalue()), "r") as zf:
        roundtrip = zf.read(name)
        if roundtrip != data:
            raise PR96ProfileError(f"generated deflate payload failed roundtrip for {name!r}")
        info = zf.getinfo(name)
        return raw_member_payload(buffer.getvalue(), info)


def parse_deflate_levels(raw: str) -> list[int]:
    levels: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            levels.extend(range(start, end + 1))
        else:
            levels.append(int(part))
    unique = sorted(set(levels))
    if not unique or any(level < 0 or level > 9 for level in unique):
        raise PR96ProfileError(f"invalid deflate levels: {raw!r}")
    return unique


def best_compression_choice(member: SourceMember, levels: Iterable[int]) -> CompressionChoice:
    choices = [
        CompressionChoice(
            method_id=zipfile.ZIP_STORED,
            method="stored",
            compresslevel=None,
            compressed_size=len(member.data),
            strategy="stored_uncompressed",
            compressed_payload=member.data,
        )
    ]
    if member.method_id in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
        choices.append(
            CompressionChoice(
                method_id=member.method_id,
                method=method_name(member.method_id),
                compresslevel=None,
                compressed_size=len(member.raw_compressed),
                strategy="source_raw_copy",
                compressed_payload=member.raw_compressed,
            )
        )
    for level in levels:
        payload = generated_deflate_payload(member.name, member.data, level)
        choices.append(
            CompressionChoice(
                method_id=zipfile.ZIP_DEFLATED,
                method="deflated",
                compresslevel=level,
                compressed_size=len(payload),
                strategy="deflate_recompressed",
                compressed_payload=payload,
            )
        )
    strategy_rank = {"source_raw_copy": 0, "stored_uncompressed": 1, "deflate_recompressed": 2}
    return min(
        choices,
        key=lambda choice: (
            choice.compressed_size,
            strategy_rank.get(choice.strategy, 99),
            -1 if choice.compresslevel is None else choice.compresslevel,
        ),
    )


def write_local_header(handle: io.BytesIO, member: SourceMember, choice: CompressionChoice) -> None:
    filename = member.name.encode("utf-8")
    mod_time, mod_date = dos_datetime()
    version_needed = 10 if choice.method_id == zipfile.ZIP_STORED else 20
    handle.write(
        struct.pack(
            "<IHHHHHIIIHH",
            0x04034B50,
            version_needed,
            0,
            choice.method_id,
            mod_time,
            mod_date,
            member.crc32_int,
            len(choice.compressed_payload),
            len(member.data),
            len(filename),
            0,
        )
    )
    handle.write(filename)
    handle.write(choice.compressed_payload)


def write_central_header(
    handle: io.BytesIO,
    member: SourceMember,
    choice: CompressionChoice,
    local_offset: int,
) -> None:
    filename = member.name.encode("utf-8")
    mod_time, mod_date = dos_datetime()
    version_needed = 10 if choice.method_id == zipfile.ZIP_STORED else 20
    version_made_by = (3 << 8) | version_needed
    handle.write(
        struct.pack(
            "<IHHHHHHIIIHHHHHII",
            0x02014B50,
            version_made_by,
            version_needed,
            0,
            choice.method_id,
            mod_time,
            mod_date,
            member.crc32_int,
            len(choice.compressed_payload),
            len(member.data),
            len(filename),
            0,
            0,
            0,
            0,
            0o644 << 16,
            local_offset,
        )
    )
    handle.write(filename)


def write_candidate_zip(
    path: Path,
    members: Sequence[SourceMember],
    choices: dict[str, CompressionChoice],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = io.BytesIO()
    local_offsets: list[int] = []
    for member in members:
        choice = choices[member.name]
        if (binascii.crc32(member.data) & 0xFFFFFFFF) != member.crc32_int:
            raise PR96ProfileError(f"CRC mismatch before writing {member.name!r}")
        local_offsets.append(handle.tell())
        write_local_header(handle, member, choice)

    central_start = handle.tell()
    for member, local_offset in zip(members, local_offsets):
        write_central_header(handle, member, choices[member.name], local_offset)
    central_size = handle.tell() - central_start
    handle.write(
        struct.pack(
            "<IHHHHIIH",
            0x06054B50,
            0,
            0,
            len(members),
            len(members),
            central_size,
            central_start,
            0,
        )
    )
    path.write_bytes(handle.getvalue())


def verify_candidate_payloads(path: Path, members: Sequence[SourceMember]) -> list[dict[str, Any]]:
    expected = {member.name: member.data for member in members}
    with zipfile.ZipFile(path, "r") as zf:
        infos = zf.infolist()
        if [info.filename for info in infos] != [member.name for member in members]:
            raise PR96ProfileError(f"candidate member ordering/name mismatch: {path}")
        rows: list[dict[str, Any]] = []
        for info, member in zip(infos, members):
            data = zf.read(info)
            if data != expected[info.filename]:
                raise PR96ProfileError(f"candidate payload mismatch for {info.filename!r}")
            rows.append(
                {
                    "name": info.filename,
                    "method": method_name(int(info.compress_type)),
                    "method_id": int(info.compress_type),
                    "compressed_size": int(info.compress_size),
                    "uncompressed_size": int(info.file_size),
                    "sha256": sha256_bytes(data),
                    "payload_identity_checked": True,
                }
            )
    return rows


def build_candidate_record(
    *,
    label: str,
    output_path: Path,
    source_archive: Path,
    source_bytes: int,
    kept_members: Sequence[SourceMember],
    removed_members: Sequence[SourceMember],
    choices: dict[str, CompressionChoice],
) -> dict[str, Any]:
    write_candidate_zip(output_path, kept_members, choices)
    member_rows = verify_candidate_payloads(output_path, kept_members)
    archive_bytes = output_path.stat().st_size
    return {
        "label": label,
        "archive": str(output_path),
        "archive_bytes": archive_bytes,
        "archive_sha256": sha256_file(output_path),
        "archive_byte_delta": archive_bytes - source_bytes,
        "rate_term": round(contest_rate_term(archive_bytes), 12),
        "rate_term_delta": round(contest_rate_term(archive_bytes) - contest_rate_term(source_bytes), 12),
        "source_archive": str(source_archive),
        "removed_members": [member.name for member in removed_members],
        "kept_members": [member.name for member in kept_members],
        "member_payload_identity_checked": True,
        "members": member_rows,
        "compression_plan": {
            name: choice.manifest()
            for name, choice in sorted(choices.items(), key=lambda item: item[0])
        },
        "score_claim": False,
        "evidence_grade": EVIDENCE_GRADE,
        "requires_exact_cuda_eval": True,
    }


def build_profile(
    archive: Path,
    runtime_py: Path | None,
    output_dir: Path,
    *,
    deflate_levels: Sequence[int],
    keep_unused: bool = False,
) -> dict[str, Any]:
    source_members = read_archive(archive)
    members_by_name = {member.name: member for member in source_members}
    missing = [name for name in ("decoder.bin", "latents.bin") if name not in members_by_name]
    if missing:
        raise PR96ProfileError(f"PR96 archive is missing required members: {missing}")

    runtime_reads = archive_read_set(runtime_py)
    read_members = set(runtime_reads["read_members"])
    unknown_reads = sorted(read_members - set(members_by_name))
    if unknown_reads:
        raise PR96ProfileError(f"runtime reads members missing from archive: {unknown_reads}")

    decoder = parse_decoder_payload(members_by_name["decoder.bin"].data)
    latents = parse_latents_payload(members_by_name["latents.bin"].data)
    compression_choices = {
        member.name: best_compression_choice(member, deflate_levels)
        for member in source_members
    }

    source_bytes = archive.stat().st_size
    candidates: list[dict[str, Any]] = []
    member_preserving_path = output_dir / "archive.pr96_member_preserving_repack.zip"
    candidates.append(
        build_candidate_record(
            label="member_preserving_repack",
            output_path=member_preserving_path,
            source_archive=archive,
            source_bytes=source_bytes,
            kept_members=source_members,
            removed_members=[],
            choices=compression_choices,
        )
    )

    unused_members: list[SourceMember] = []
    if runtime_reads["safe_for_unused_member_removal"] and not keep_unused:
        unused_members = [member for member in source_members if member.name not in read_members]
        if unused_members:
            kept = [member for member in source_members if member.name in read_members]
            choices = {member.name: compression_choices[member.name] for member in kept}
            candidates.append(
                build_candidate_record(
                    label="drop_statically_unused_members_repack",
                    output_path=output_dir / "archive.pr96_drop_unused_repack.zip",
                    source_archive=archive,
                    source_bytes=source_bytes,
                    kept_members=kept,
                    removed_members=unused_members,
                    choices=choices,
                )
            )

    best = min(candidates, key=lambda row: int(row["archive_bytes"]))
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "source_archive": str(archive),
        "source_archive_bytes": source_bytes,
        "source_archive_sha256": sha256_file(archive),
        "source_rate_term": round(contest_rate_term(source_bytes), 12),
        "source_members": [member.manifest() for member in source_members],
        "runtime_static_read_set": runtime_reads,
        "pr96_payloads": {
            "decoder": decoder,
            "latents": latents,
        },
        "byte_opportunities": {
            "unused_members": [member.name for member in unused_members],
            "zip_method_changes": [
                {
                    "name": member.name,
                    "source_method": member.method,
                    "source_compressed_size": member.compressed_size,
                    "chosen_method": compression_choices[member.name].method,
                    "chosen_compresslevel": compression_choices[member.name].compresslevel,
                    "chosen_compressed_size": compression_choices[member.name].compressed_size,
                    "payload_byte_delta": (
                        compression_choices[member.name].compressed_size
                        - member.compressed_size
                    ),
                }
                for member in source_members
                if compression_choices[member.name].compressed_size != member.compressed_size
                or compression_choices[member.name].method_id != member.method_id
            ],
        },
        "candidates": candidates,
        "recommended_candidate": best["label"],
        "recommended_candidate_archive": best["archive"],
        "recommended_candidate_bytes": best["archive_bytes"],
        "recommended_candidate_sha256": best["archive_sha256"],
        "recommended_candidate_byte_delta": best["archive_byte_delta"],
        "score_claim": False,
        "evidence_grade": EVIDENCE_GRADE,
        "safety": {
            "member_payload_bytes_preserved_for_kept_members": True,
            "no_tensor_requantization": True,
            "no_runtime_code_change": True,
            "does_not_load_scorers": True,
            "does_not_launch_remote_jobs": True,
            "requires_raw_output_parity_before_rate_only_interpretation": True,
            "requires_exact_cuda_eval_before_score_claim": True,
            "unused_member_removal_enabled": bool(unused_members),
            "unused_member_removal_basis": (
                "static archive_dir read_bytes analysis"
                if unused_members
                else "not applied"
            ),
        },
        "next_exact_eval_action": (
            "After public PR96 baseline replay and local raw-byte parity, claim a "
            "PR96 repack exact-eval lane and evaluate recommended_candidate_archive "
            "with experiments/contest_auth_eval.py --device cuda using the unchanged "
            "pr96_runtime/inflate.sh."
        ),
    }


def write_outputs(profile: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "profile_pr96_rem2_hnerv_packing.json"
    path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> int:
    runtime_py = None if args.runtime_py == "" else Path(args.runtime_py)
    profile = build_profile(
        Path(args.archive),
        runtime_py,
        Path(args.output_dir),
        deflate_levels=parse_deflate_levels(args.deflate_levels),
        keep_unused=args.keep_unused,
    )
    write_outputs(profile, Path(args.output_dir))
    print(json.dumps(profile, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--runtime-py", default=DEFAULT_RUNTIME_PY)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--deflate-levels", default="1-9")
    parser.add_argument(
        "--keep-unused",
        action="store_true",
        help="preserve all source members even if the runtime static read set does not use them",
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
