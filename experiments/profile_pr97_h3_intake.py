#!/usr/bin/env python3
# pyc-recovery pass2: rehydrated from git blob ffffbf6c848e34b9d06efe6453893131c5e96c90 via `git fsck --lost-found`
# original path: experiments/profile_pr97_h3_intake.py
# This is OUR source, dropped during commit 66c59aae filter-repo cleanup; the .pyc was the only
# orphan left behind. Original blob SHA verified intact.
# Recovered: 2026-05-05 by Sherlock pass2
"""Offline byte/profile intake for PR97 H3 public archive.

This tool is deliberately static. It parses the single-member ``p`` payload,
checks the PR97-specific H3 subformats, and builds deterministic byte-only
repack candidates. It never runs inflate, loads scorers, uses CUDA, or submits
remote work.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import io
import json
import lzma
import math
import struct
import zipfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

try:
    import brotli
except ImportError as exc:  # pragma: no cover - exercised only on broken envs
    raise SystemExit("brotli is required for PR97 H3 intake profiling") from exc


SCHEMA = "pr97_h3_static_intake_profile_v1"
TOOL = "experiments/profile_pr97_h3_intake.py"
EVIDENCE_GRADE = "external_archive_byte_intake_only_until_exact_cuda_replay"
CONTEST_ORIGINAL_BYTES = 37_545_489
DEFAULT_ARCHIVE = "experiments/results/leaderboard_intel_20260504_codex/pr97_archive.zip"
DEFAULT_RUNTIME = "experiments/results/leaderboard_intel_20260504_codex/pr97_runtime"
DEFAULT_OUTPUT_DIR = "experiments/results/pr97_h3_intake_20260504_codex"
PAYLOAD_PARTS = ("mask", "pose", "model", "sidecar")


class PR97ProfileError(ValueError):
    """Raised when the PR97 archive cannot be safely parsed."""


@dataclasses.dataclass(frozen=True)
class Member:
    name: str
    data: bytes
    method_id: int
    compressed_size: int
    uncompressed_size: int
    crc32: str
    header_offset: int

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "method_id": self.method_id,
            "method": "stored" if self.method_id == zipfile.ZIP_STORED else f"method_{self.method_id}",
            "compressed_size": self.compressed_size,
            "uncompressed_size": self.uncompressed_size,
            "crc32": self.crc32,
            "sha256": sha256_bytes(self.data),
            "header_offset": self.header_offset,
        }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def contest_rate_term(byte_count: int) -> float:
    return 25.0 * int(byte_count) / CONTEST_ORIGINAL_BYTES


def safe_member_blockers(name: str) -> list[str]:
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


def read_members(path: Path) -> list[Member]:
    if not path.is_file():
        raise FileNotFoundError(f"archive not found: {path}")
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if not infos:
            raise PR97ProfileError("archive has no file members")
        counts = Counter(info.filename for info in infos)
        duplicates = sorted(name for name, count in counts.items() if count > 1)
        if duplicates:
            raise PR97ProfileError(f"duplicate ZIP member names: {duplicates}")
        members: list[Member] = []
        for info in infos:
            blockers = safe_member_blockers(info.filename)
            if blockers:
                raise PR97ProfileError(f"unsafe ZIP member {info.filename!r}: {blockers}")
            data = zf.read(info)
            members.append(
                Member(
                    name=info.filename,
                    data=data,
                    method_id=int(info.compress_type),
                    compressed_size=int(info.compress_size),
                    uncompressed_size=int(info.file_size),
                    crc32=f"{info.CRC:08x}",
                    header_offset=int(info.header_offset),
                )
            )
        return members


def split_payload(blob: bytes) -> dict[str, bytes]:
    offset = 0
    parts: dict[str, bytes] = {}
    for name in PAYLOAD_PARTS:
        if offset + 4 > len(blob):
            raise PR97ProfileError(f"truncated payload length before {name}")
        size = struct.unpack_from("<I", blob, offset)[0]
        offset += 4
        end = offset + size
        if end > len(blob):
            raise PR97ProfileError(f"truncated payload part {name}: need {size} bytes")
        parts[name] = blob[offset:end]
        offset = end
    if offset != len(blob):
        raise PR97ProfileError(f"payload trailing bytes: {offset} vs {len(blob)}")
    return parts


def parse_mask(mask: bytes) -> dict[str, Any]:
    if len(mask) < 4:
        raise PR97ProfileError("mask payload is too short")
    offset = 0
    n_chunks = struct.unpack_from("<I", mask, offset)[0]
    offset += 4
    chunks: list[dict[str, Any]] = []
    for index in range(n_chunks):
        if offset + 4 > len(mask):
            raise PR97ProfileError(f"truncated mask chunk length at {index}")
        size = struct.unpack_from("<I", mask, offset)[0]
        offset += 4
        end = offset + size
        if end > len(mask):
            raise PR97ProfileError(f"truncated mask chunk {index}: need {size} bytes")
        data = mask[offset:end]
        chunks.append(
            {
                "index": index,
                "bytes": size,
                "sha256": sha256_bytes(data),
                "magic_ascii": data[:4].decode("ascii", errors="replace"),
                "magic_hex": data[:8].hex(),
            }
        )
        offset = end
    if offset != len(mask):
        raise PR97ProfileError(f"mask trailing bytes: {offset} vs {len(mask)}")
    sizes = [int(row["bytes"]) for row in chunks]
    return {
        "format": "pr97_h3_tiled_range_mask",
        "bytes": len(mask),
        "chunk_count": n_chunks,
        "length_header_bytes": 4 + 4 * n_chunks,
        "chunk_payload_bytes": sum(sizes),
        "min_chunk_bytes": min(sizes) if sizes else 0,
        "max_chunk_bytes": max(sizes) if sizes else 0,
        "chunks": chunks,
        "largest_chunks": sorted(chunks, key=lambda row: int(row["bytes"]), reverse=True)[:5],
    }


def parse_pose(blob: bytes) -> dict[str, Any]:
    try:
        raw = brotli.decompress(blob)
    except brotli.error as exc:
        raise PR97ProfileError(f"pose brotli decode failed: {exc}") from exc
    if len(raw) < 8:
        raise PR97ProfileError("pose raw payload is too short")
    offset = 0
    n_pairs, n_dim = struct.unpack_from("<II", raw, offset)
    offset += 8
    bits_per_dim = list(raw[offset : offset + n_dim])
    if len(bits_per_dim) != n_dim:
        raise PR97ProfileError("truncated pose bits_per_dim")
    offset += n_dim
    ranges: list[dict[str, Any]] = []
    for dim, bits in enumerate(bits_per_dim):
        if offset + 8 > len(raw):
            raise PR97ProfileError(f"truncated pose lo/scale for dim {dim}")
        lo, scale = struct.unpack_from("<ff", raw, offset)
        offset += 8
        ranges.append(
            {
                "dim": dim,
                "bits": int(bits),
                "lo": lo,
                "scale": scale,
                "covered_range": scale * ((1 << int(bits)) - 1),
            }
        )
    bitstream_bytes = len(raw) - offset
    needed_bytes = (sum(bits_per_dim) * n_pairs + 7) // 8
    if bitstream_bytes < needed_bytes:
        raise PR97ProfileError(
            f"pose bitstream too short: {bitstream_bytes} < {needed_bytes}"
        )
    return {
        "format": "pr97_per_dim_packed_pose_brotli",
        "compressed_bytes": len(blob),
        "compressed_sha256": sha256_bytes(blob),
        "raw_bytes": len(raw),
        "raw_sha256": sha256_bytes(raw),
        "n_pairs": n_pairs,
        "n_dim": n_dim,
        "bits_per_dim": bits_per_dim,
        "header_bytes": offset,
        "bitstream_bytes": bitstream_bytes,
        "needed_bitstream_bytes": needed_bytes,
        "ranges": ranges,
        "brotli_ratio": len(blob) / len(raw),
    }


def load_schema(schema_py: Path) -> list[tuple[str, str, tuple[int, ...]]]:
    tree = ast.parse(schema_py.read_text(encoding="utf-8"))
    # Catalog #168 fix 2026-05-12: handle both `SCHEMA = [...]` (Assign) and
    # `SCHEMA: list[...] = [...]` (AnnAssign) forms.
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(target, "id", None) == "SCHEMA" for target in node.targets):
            value = ast.literal_eval(node.value)
            return [(str(name), str(kind), tuple(int(x) for x in shape)) for name, kind, shape in value]
        if (isinstance(node, ast.AnnAssign)
                and node.value is not None
                and isinstance(node.target, ast.Name)
                and node.target.id == "SCHEMA"):
            value = ast.literal_eval(node.value)
            return [(str(name), str(kind), tuple(int(x) for x in shape)) for name, kind, shape in value]
    raise PR97ProfileError(f"SCHEMA assignment not found: {schema_py}")


def parse_model(blob: bytes, schema_py: Path) -> dict[str, Any]:
    try:
        raw = brotli.decompress(blob)
    except brotli.error as exc:
        raise PR97ProfileError(f"model brotli decode failed: {exc}") from exc
    schema = load_schema(schema_py)
    offset = 0
    kind_counts: Counter[str] = Counter()
    fp4_params = 0
    fp16_params = 0
    fp4_nibble_bytes = 0
    fp4_scale_bytes = 0
    fp16_bytes = 0
    rows: list[dict[str, Any]] = []
    for name, kind, shape in schema:
        kind_counts[kind] += 1
        n = math.prod(shape)
        start = offset
        if kind == "fp4_w":
            n_blocks = (n + 31) // 32
            packed_bytes = (n_blocks * 32 + 1) // 2
            scale_bytes = n_blocks * 2
            offset += packed_bytes + scale_bytes
            fp4_params += n
            fp4_nibble_bytes += packed_bytes
            fp4_scale_bytes += scale_bytes
        elif kind in {"fp16_w", "fp16_b"}:
            byte_count = n * 2
            offset += byte_count
            fp16_params += n
            fp16_bytes += byte_count
        else:
            raise PR97ProfileError(f"unknown model schema kind {kind!r}")
        rows.append(
            {
                "name": name,
                "kind": kind,
                "shape": list(shape),
                "params": n,
                "raw_bytes": offset - start,
                "offset": start,
            }
        )
    if offset != len(raw):
        raise PR97ProfileError(f"model schema consumed {offset}, raw has {len(raw)}")
    return {
        "format": "pr97_h3_flat_fp4_model_brotli",
        "compressed_bytes": len(blob),
        "compressed_sha256": sha256_bytes(blob),
        "raw_bytes": len(raw),
        "raw_sha256": sha256_bytes(raw),
        "schema_entries": len(schema),
        "schema_sha256": sha256_file(schema_py),
        "kind_counts": dict(kind_counts),
        "fp4_params": fp4_params,
        "fp16_params": fp16_params,
        "fp4_nibble_bytes": fp4_nibble_bytes,
        "fp4_scale_bytes": fp4_scale_bytes,
        "fp16_bytes": fp16_bytes,
        "brotli_ratio": len(blob) / len(raw),
        "largest_raw_entries": sorted(rows, key=lambda row: int(row["raw_bytes"]), reverse=True)[:10],
    }


def parse_sidecar(blob: bytes) -> dict[str, Any]:
    try:
        raw = lzma.decompress(blob, format=lzma.FORMAT_XZ)
    except lzma.LZMAError as exc:
        raise PR97ProfileError(f"sidecar xz decode failed: {exc}") from exc
    if raw[:4] != b"BPGD":
        raise PR97ProfileError(f"bad sidecar magic: {raw[:4]!r}")
    pos = 4
    if pos + 2 > len(raw):
        raise PR97ProfileError("truncated sidecar pair count")
    n_pairs = struct.unpack_from("<H", raw, pos)[0]
    pos += 2
    prev = 0
    pair_ids: list[int] = []
    flag_hist: Counter[int] = Counter()
    counts = Counter(
        {
            "x2_pairs": 0,
            "x2_patches": 0,
            "cmaes_pairs": 0,
            "cmaes_patches": 0,
            "pattern_pairs": 0,
            "pattern_patches": 0,
            "pose_pairs": 0,
            "pose_values": 0,
            "warp_pairs": 0,
            "warp_nonzero": 0,
        }
    )
    for index in range(n_pairs):
        if index == 0:
            if pos + 2 > len(raw):
                raise PR97ProfileError("truncated first sidecar pair id")
            pair_id = struct.unpack_from("<H", raw, pos)[0]
            pos += 2
        else:
            if pos >= len(raw):
                raise PR97ProfileError("truncated sidecar pair delta")
            delta = raw[pos]
            pos += 1
            if delta == 0xFF:
                if pos + 2 > len(raw):
                    raise PR97ProfileError("truncated sidecar escaped pair id")
                pair_id = struct.unpack_from("<H", raw, pos)[0]
                pos += 2
            else:
                pair_id = prev + delta
        prev = pair_id
        pair_ids.append(pair_id)
        if pos >= len(raw):
            raise PR97ProfileError("truncated sidecar flags")
        flags = raw[pos]
        pos += 1
        flag_hist[flags] += 1
        if flags & 1:
            n, pos = _consume_patch_list(raw, pos, "x2")
            counts["x2_pairs"] += 1
            counts["x2_patches"] += n
        if flags & 2:
            n, pos = _consume_patch_list(raw, pos, "cmaes")
            counts["cmaes_pairs"] += 1
            counts["cmaes_patches"] += n
        if flags & 4:
            n, pos = _consume_patch_list(raw, pos, "pattern")
            counts["pattern_pairs"] += 1
            counts["pattern_patches"] += n
        if flags & 8:
            if pos + 3 > len(raw):
                raise PR97ProfileError("truncated sidecar pose deltas")
            pos += 3
            counts["pose_pairs"] += 1
            counts["pose_values"] += 3
        if flags & 16:
            if pos + 2 > len(raw):
                raise PR97ProfileError("truncated sidecar warp")
            qx, qy = struct.unpack_from("<bb", raw, pos)
            pos += 2
            counts["warp_pairs"] += 1
            if qx or qy:
                counts["warp_nonzero"] += 1
    if pos != len(raw):
        raise PR97ProfileError(f"sidecar trailing bytes: {pos} vs {len(raw)}")
    return {
        "format": "pr97_bpgd_xz_sidecar",
        "compressed_bytes": len(blob),
        "compressed_sha256": sha256_bytes(blob),
        "raw_bytes": len(raw),
        "raw_sha256": sha256_bytes(raw),
        "xz_ratio": len(blob) / len(raw),
        "pair_record_count": n_pairs,
        "pair_id_min": min(pair_ids) if pair_ids else None,
        "pair_id_max": max(pair_ids) if pair_ids else None,
        "counts": dict(counts),
        "flag_histogram": {str(flag): count for flag, count in sorted(flag_hist.items())},
    }


def _consume_patch_list(raw: bytes, pos: int, label: str) -> tuple[int, int]:
    if pos >= len(raw):
        raise PR97ProfileError(f"truncated sidecar {label} count")
    n = raw[pos]
    pos += 1
    end = pos + 3 * n
    if end > len(raw):
        raise PR97ProfileError(f"truncated sidecar {label} patches")
    return n, end


def deterministic_zip_bytes(payload: bytes, *, compression: int) -> bytes:
    handle = io.BytesIO()
    compresslevel = 9 if compression == zipfile.ZIP_DEFLATED else None
    with zipfile.ZipFile(handle, "w", compression=compression, compresslevel=compresslevel) as zf:
        info = zipfile.ZipInfo("p", (1980, 1, 1, 0, 0, 0))
        info.compress_type = compression
        info.external_attr = 0o644 << 16
        zf.writestr(info, payload)
    return handle.getvalue()


def write_candidate(path: Path, payload: bytes, *, compression: int) -> dict[str, Any]:
    raw = deterministic_zip_bytes(payload, compression=compression)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    with zipfile.ZipFile(path, "r") as zf:
        info = zf.getinfo("p")
        member_sha = sha256_bytes(zf.read("p"))
    return {
        "archive": path.as_posix(),
        "archive_bytes": path.stat().st_size,
        "archive_sha256": sha256_file(path),
        "member_p_sha256": member_sha,
        "member_p_compress_type": int(info.compress_type),
        "member_p_compressed_bytes": int(info.compress_size),
        "member_p_uncompressed_bytes": int(info.file_size),
    }


def build_payload(parts: Sequence[bytes]) -> bytes:
    return b"".join(struct.pack("<I", len(part)) + part for part in parts)


def build_candidates(
    *,
    output_dir: Path,
    source_archive_bytes: int,
    source_payload: bytes,
    parts: Mapping[str, bytes],
    pose_raw_sha256: str,
    model_raw_sha256: str,
    sidecar_raw_sha256: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    def finish(label: str, payload: bytes, *, compression: int, notes: list[str]) -> None:
        row = write_candidate(output_dir / f"archive.{label}.zip", payload, compression=compression)
        row.update(
            {
                "label": label,
                "archive_byte_delta": row["archive_bytes"] - source_archive_bytes,
                "rate_score_delta": contest_rate_term(row["archive_bytes"] - source_archive_bytes),
                "score_claim": False,
                "requires_runtime_change": False,
                "exact_cuda_eval_required_before_promotion": True,
                "notes": notes,
            }
        )
        candidates.append(row)

    finish(
        "pr97_deflated_p",
        source_payload,
        compression=zipfile.ZIP_DEFLATED,
        notes=["ZIP deflates the original p member; zf.read('p') is byte-identical to source"],
    )

    pose_raw = brotli.decompress(parts["pose"])
    model_raw = brotli.decompress(parts["model"])
    pose_br10 = brotli.compress(pose_raw, quality=10)
    model_br10 = brotli.compress(model_raw, quality=10)
    if sha256_bytes(pose_raw) != pose_raw_sha256 or sha256_bytes(model_raw) != model_raw_sha256:
        raise PR97ProfileError("internal raw recompression hash mismatch")
    pm_payload = build_payload([parts["mask"], pose_br10, model_br10, parts["sidecar"]])
    finish(
        "pr97_pose_model_br10_deflated_p",
        pm_payload,
        compression=zipfile.ZIP_DEFLATED,
        notes=[
            "pose/model brotli streams decompress to the same raw bytes as source",
            "runtime source can remain unchanged",
        ],
    )

    sidecar_raw = lzma.decompress(parts["sidecar"], format=lzma.FORMAT_XZ)
    sidecar_br11 = brotli.compress(sidecar_raw, quality=11)
    side_payload = build_payload([parts["mask"], pose_br10, model_br10, sidecar_br11])
    side_runtime_row = {
        "label": "pr97_pose_model_br10_sidecar_br11_deflated_p_runtime_patch",
        "archive_bytes_estimate": len(deterministic_zip_bytes(side_payload, compression=zipfile.ZIP_DEFLATED)),
        "archive_byte_delta_estimate": len(deterministic_zip_bytes(side_payload, compression=zipfile.ZIP_DEFLATED))
        - source_archive_bytes,
        "rate_score_delta_estimate": contest_rate_term(
            len(deterministic_zip_bytes(side_payload, compression=zipfile.ZIP_DEFLATED))
            - source_archive_bytes
        ),
        "requires_runtime_change": True,
        "runtime_change": "decode_sidecar_blob must brotli.decompress the sidecar raw BPGD stream instead of lzma FORMAT_XZ",
        "raw_sidecar_sha256_preserved": sha256_bytes(sidecar_raw) == sidecar_raw_sha256,
        "score_claim": False,
        "exact_cuda_eval_required_before_promotion": True,
    }
    candidates.append(side_runtime_row)
    return candidates


def runtime_static_report(runtime_dir: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    imports: Counter[str] = Counter()
    for path in sorted(runtime_dir.iterdir()):
        if not path.is_file():
            continue
        row = {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if path.suffix == ".py":
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                row["parse_error"] = str(exc)
            else:
                mods = sorted(_top_level_imports(tree))
                row["imports"] = mods
                imports.update(mods)
        files.append(row)
    return {
        "runtime_dir": runtime_dir.as_posix(),
        "files": files,
        "runtime_file_count": len(files),
        "runtime_tree_sha256": sha256_bytes(
            b"".join(
                path.name.encode("utf-8") + b"\0" + path.read_bytes()
                for path in sorted(runtime_dir.iterdir())
                if path.is_file()
            )
        ),
        "top_level_imports": sorted(imports),
        "static_risks": [
            "inflate.sh attempts pip install brotli if missing; exact replay should pin dependency closure",
            "inflate.py compiles range_mask_codec.cpp at inflate time and requires c++/g++/clang++",
        ],
    }


def _top_level_imports(tree: ast.AST) -> Iterable[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".", 1)[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module.split(".", 1)[0]


def build_profile(archive: Path, runtime_dir: Path, output_dir: Path | None = None) -> dict[str, Any]:
    members = read_members(archive)
    if len(members) != 1 or members[0].name != "p":
        raise PR97ProfileError(f"expected single member p, got {[member.name for member in members]}")
    payload = members[0].data
    parts = split_payload(payload)
    part_rows = {
        name: {
            "bytes": len(data),
            "sha256": sha256_bytes(data),
            "magic_hex": data[:8].hex(),
            "magic_ascii": data[:8].decode("ascii", errors="replace"),
        }
        for name, data in parts.items()
    }
    mask = parse_mask(parts["mask"])
    pose = parse_pose(parts["pose"])
    model = parse_model(parts["model"], runtime_dir / "schema_h3.py")
    sidecar = parse_sidecar(parts["sidecar"])

    archive_bytes = archive.stat().st_size
    candidates: list[dict[str, Any]] = []
    if output_dir is not None:
        candidates = build_candidates(
            output_dir=output_dir,
            source_archive_bytes=archive_bytes,
            source_payload=payload,
            parts=parts,
            pose_raw_sha256=pose["raw_sha256"],
            model_raw_sha256=model["raw_sha256"],
            sidecar_raw_sha256=sidecar["raw_sha256"],
        )

    best = min(
        (row for row in candidates if row.get("requires_runtime_change") is False),
        key=lambda row: int(row.get("archive_bytes", archive_bytes)),
        default=None,
    )
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "evidence_grade": EVIDENCE_GRADE,
        "score_claim": False,
        "dispatch_performed": False,
        "archive": {
            "path": archive.as_posix(),
            "bytes": archive_bytes,
            "sha256": sha256_file(archive),
            "zip_overhead_bytes": archive_bytes - len(payload),
            "members": [member.manifest() for member in members],
        },
        "payload": {
            "member": "p",
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
            "length_header_bytes": 4 * len(PAYLOAD_PARTS),
            "parts": part_rows,
        },
        "mask": mask,
        "pose": pose,
        "model": model,
        "sidecar": sidecar,
        "runtime_static": runtime_static_report(runtime_dir),
        "byte_opportunities": {
            "source_archive_bytes": archive_bytes,
            "pr95_stemperm_frontier_bytes": 178_277,
            "archive_bytes_delta_vs_pr95_stemperm": archive_bytes - 178_277,
            "rate_score_delta_vs_pr95_stemperm": contest_rate_term(archive_bytes - 178_277),
            "safe_repack_candidates": candidates,
            "recommended_no_runtime_change_candidate": None if best is None else best["label"],
        },
        "decision_notes": [
            "No GPU score claim: replay exact T4 job remains the source of score truth.",
            "The deflated-p and pose/model br10 candidates preserve decoded member semantics and need exact CUDA replay before promotion.",
        ],
    }


def render_markdown(profile: Mapping[str, Any]) -> str:
    archive = profile["archive"]
    payload = profile["payload"]
    lines = [
        "# PR97 H3 Static Intake",
        "",
        f"- Evidence grade: `{profile['evidence_grade']}`",
        f"- Archive: `{archive['path']}`",
        f"- Archive bytes/SHA: `{archive['bytes']}` / `{archive['sha256']}`",
        f"- ZIP overhead: `{archive['zip_overhead_bytes']}` bytes",
        f"- Payload `p`: `{payload['bytes']}` bytes / `{payload['sha256']}`",
        "",
        "## Payload Split",
        "",
        "| part | bytes | sha256 |",
        "|---|---:|---|",
    ]
    for name in PAYLOAD_PARTS:
        row = payload["parts"][name]
        lines.append(f"| {name} | {row['bytes']} | `{row['sha256']}` |")
    lines.extend(
        [
            "",
            "## Parsed Subformats",
            "",
            f"- Mask: `{profile['mask']['chunk_count']}` chunks, `{profile['mask']['bytes']}` bytes.",
            f"- Pose: bits per dim `{profile['pose']['bits_per_dim']}`, raw `{profile['pose']['raw_bytes']}` bytes.",
            f"- Model: `{profile['model']['schema_entries']}` schema entries, raw `{profile['model']['raw_bytes']}` bytes.",
            f"- Sidecar: `{profile['sidecar']['pair_record_count']}` pair records, raw `{profile['sidecar']['raw_bytes']}` bytes.",
            "",
            "## Byte Candidates",
            "",
            "| label | archive bytes | delta | runtime change |",
            "|---|---:|---:|---|",
        ]
    )
    for row in profile["byte_opportunities"]["safe_repack_candidates"]:
        bytes_value = row.get("archive_bytes", row.get("archive_bytes_estimate"))
        delta = row.get("archive_byte_delta", row.get("archive_byte_delta_estimate"))
        lines.append(
            f"| {row['label']} | {bytes_value} | {delta} | {row.get('requires_runtime_change')} |"
        )
    lines.extend(
        [
            "",
            "## Risks",
            "",
        ]
    )
    for risk in profile["runtime_static"]["static_risks"]:
        lines.append(f"- {risk}")
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    archive = Path(args.archive)
    runtime_dir = Path(args.runtime_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None
    profile = build_profile(archive, runtime_dir, output_dir)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "profile_pr97_h3_intake.json").write_text(json_text(profile), encoding="utf-8")
        (output_dir / "profile_pr97_h3_intake.md").write_text(render_markdown(profile), encoding="utf-8")
    print(json_text(profile))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--runtime-dir", default=DEFAULT_RUNTIME)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
