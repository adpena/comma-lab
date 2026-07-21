# SPDX-License-Identifier: MIT
"""Strict hot-swappable container for the Task #578 S4 standalone receiver.

The container deliberately treats every learned/video-derived section as opaque
bytes.  Section interpretation belongs to the standalone receiver; this module
owns only ordering, versioning, exact lengths, hashes, and deterministic ZIP
materialization.  The result is one ``0.bin`` member and no sidecar authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

MAGIC: Final = b"S4A1\x00\x00"
CONTAINER_VERSION: Final = 1
REGISTRY_VERSION: Final = 1
PREFIX: Final = struct.Struct(">6sHH")
SECTION_PREFIX: Final = struct.Struct(">HBBQQ32s")
DIGEST_BYTES: Final = 32
SECTION_ORDER: Final = (
    "manifest.json",
    "seed.ppcs",
    "base.pbase3",
    "causal.pcr3",
    "events.pce3",
    "components.pcomp3",
)
CODEC_IDS: Final = {
    "raw": 0,
    "zlib9": 1,
    "brotli_q11": 2,
    "lzma1_raw_1MiB": 3,
    "mixed": 4,
    "range_static_v1": 5,
}
CODEC_NAMES: Final = {value: key for key, value in CODEC_IDS.items()}
SECTION_CODECS: Final = {
    "manifest.json": frozenset({"raw"}),
    "seed.ppcs": frozenset({"raw"}),
    "base.pbase3": frozenset({"mixed"}),
    "causal.pcr3": frozenset({"raw", "range_static_v1"}),
    "events.pce3": frozenset({"lzma1_raw_1MiB", "range_static_v1"}),
    "components.pcomp3": frozenset({"zlib9", "range_static_v1"}),
}


class S4ArchiveError(ValueError):
    """Malformed, noncanonical, or incomplete S4 artifact."""


@dataclass(frozen=True)
class SectionBytes:
    """One registry-bound byte section accepted by the hot-swap composer."""

    name: str
    payload: bytes
    codec: str
    decoded_bytes: int
    registry_version: int = REGISTRY_VERSION

    def __post_init__(self) -> None:
        if self.name not in SECTION_CODECS:
            raise S4ArchiveError(f"unknown section {self.name!r}")
        if self.codec not in SECTION_CODECS[self.name]:
            raise S4ArchiveError(f"codec {self.codec!r} is not registered for {self.name}")
        if self.registry_version != REGISTRY_VERSION:
            raise S4ArchiveError("section registry version mismatch")
        if not isinstance(self.payload, bytes):
            raise S4ArchiveError("section payload must be exact bytes")
        if isinstance(self.decoded_bytes, bool) or not isinstance(self.decoded_bytes, int):
            raise S4ArchiveError("decoded_bytes must be an exact integer")
        if self.decoded_bytes < 0 or (self.payload and self.decoded_bytes == 0):
            raise S4ArchiveError("decoded_bytes is inconsistent with the section payload")


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical ASCII JSON used for embedded custody manifests."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise S4ArchiveError("manifest is not canonical-JSON encodable") from exc


def _validated_sections(sections: Sequence[SectionBytes]) -> tuple[SectionBytes, ...]:
    rows = tuple(sections)
    names = tuple(row.name for row in rows)
    if names != SECTION_ORDER:
        raise S4ArchiveError(f"section order must be exactly {SECTION_ORDER!r}")
    if len(set(names)) != len(names):
        raise S4ArchiveError("section names must be unique")
    manifest = rows[0].payload
    try:
        value = json.loads(manifest.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise S4ArchiveError("manifest section is malformed") from exc
    if canonical_json_bytes(value) != manifest:
        raise S4ArchiveError("manifest section is not canonical")
    if value.get("schema") != "s4_archive_payload_manifest.v1":
        raise S4ArchiveError("manifest schema mismatch")
    declared = value.get("section_registry")
    if not isinstance(declared, list) or [row.get("name") for row in declared] != list(SECTION_ORDER[1:]):
        raise S4ArchiveError("manifest section registry is incomplete or reordered")
    actual_by_name = {row.name: row for row in rows[1:]}
    for declaration in declared:
        row = actual_by_name[declaration["name"]]
        expected = {
            "name": row.name,
            "codec": row.codec,
            "encoded_bytes": len(row.payload),
            "decoded_bytes": row.decoded_bytes,
            "sha256": hashlib.sha256(row.payload).hexdigest(),
            "registry_version": row.registry_version,
        }
        if declaration != expected:
            raise S4ArchiveError(f"manifest custody mismatch for {row.name}")
    return rows


def serialize_sections(sections: Sequence[SectionBytes]) -> bytes:
    """Serialize exact section bytes with L20-style length prefixes and hashes."""

    rows = _validated_sections(sections)
    out = bytearray(PREFIX.pack(MAGIC, CONTAINER_VERSION, len(rows)))
    for row in rows:
        name = row.name.encode("ascii")
        out.extend(
            SECTION_PREFIX.pack(
                len(name),
                row.registry_version,
                CODEC_IDS[row.codec],
                len(row.payload),
                row.decoded_bytes,
                hashlib.sha256(row.payload).digest(),
            )
        )
        out.extend(name)
        out.extend(row.payload)
    out.extend(hashlib.sha256(out).digest())
    return bytes(out)


def parse_sections(payload: bytes) -> tuple[SectionBytes, ...]:
    """Fail closed on hashes, versions, lengths, ordering, or trailing bytes."""

    if not isinstance(payload, bytes) or len(payload) < PREFIX.size + DIGEST_BYTES:
        raise S4ArchiveError("S4 container is truncated")
    if hashlib.sha256(payload[:-DIGEST_BYTES]).digest() != payload[-DIGEST_BYTES:]:
        raise S4ArchiveError("S4 outer digest mismatch")
    magic, version, count = PREFIX.unpack_from(payload)
    if magic != MAGIC or version != CONTAINER_VERSION or count != len(SECTION_ORDER):
        raise S4ArchiveError("S4 container header mismatch")
    cursor = PREFIX.size
    limit = len(payload) - DIGEST_BYTES
    rows: list[SectionBytes] = []
    for _ in range(count):
        if cursor + SECTION_PREFIX.size > limit:
            raise S4ArchiveError("S4 section header is truncated")
        name_size, registry, codec_id, encoded_size, decoded_size, digest = SECTION_PREFIX.unpack_from(
            payload, cursor
        )
        cursor += SECTION_PREFIX.size
        end = cursor + name_size + encoded_size
        if name_size == 0 or end > limit:
            raise S4ArchiveError("S4 section length is invalid")
        name_bytes = payload[cursor : cursor + name_size]
        cursor += name_size
        body = payload[cursor:end]
        cursor = end
        try:
            name = name_bytes.decode("ascii")
            codec = CODEC_NAMES[codec_id]
        except (UnicodeDecodeError, KeyError) as exc:
            raise S4ArchiveError("S4 section name or codec is unknown") from exc
        if name.encode("ascii") != name_bytes or hashlib.sha256(body).digest() != digest:
            raise S4ArchiveError("S4 section name or digest drift")
        rows.append(SectionBytes(name, body, codec, decoded_size, registry))
    if cursor != limit:
        raise S4ArchiveError("S4 container has trailing bytes")
    validated = _validated_sections(rows)
    if serialize_sections(validated) != payload:
        raise S4ArchiveError("S4 container is not byte canonical")
    return validated


def build_payload_manifest(sections: Sequence[SectionBytes], *, source_commit: str) -> dict[str, Any]:
    """Build the embedded, self-auditing registry for non-manifest sections."""

    if len(source_commit) != 40 or any(character not in "0123456789abcdef" for character in source_commit):
        raise S4ArchiveError("source_commit must be a lowercase full git SHA-1")
    return {
        "schema": "s4_archive_payload_manifest.v1",
        "container_schema": "s4_monolithic_named_sections.v1",
        "container_version": CONTAINER_VERSION,
        "source_commit": source_commit,
        "rule_118": {
            "all_video_derived_bytes_in_0_bin": True,
            "generic_receiver_code_counted": False,
            "scorer_weights_present": False,
            "source_video_present": False,
            "ground_truth_argmax_present": False,
        },
        "runtime": {
            "pair_count": 600,
            "scorer_hw": [384, 512],
            "camera_hw": [874, 1164],
            "frame_order": "pair_major_frame0_then_frame1_C_order_rgb_u8",
            "causal_policy": "PPCS_R0_plus_zero_parameter_online_policy",
            "realization": "R2_max_margin_palette_plus_exact_disjoint_factor2_support_fill",
            "standalone_receiver_closed": True,
            "semantic_and_pose_admission": False,
        },
        "section_registry": [
            {
                "name": row.name,
                "codec": row.codec,
                "encoded_bytes": len(row.payload),
                "decoded_bytes": row.decoded_bytes,
                "sha256": hashlib.sha256(row.payload).hexdigest(),
                "registry_version": row.registry_version,
            }
            for row in sections
        ],
    }


def deterministic_archive(path: Path, payload: bytes) -> dict[str, Any]:
    """Atomically emit a reproducible one-member ``archive.zip``."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        info = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        info.create_system = 3
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) != 1 or members[0].filename != "0.bin" or archive.read("0.bin") != payload:
            raise S4ArchiveError("archive ZIP parse-back mismatch")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "member": "0.bin",
        "member_bytes": len(payload),
        "member_sha256": hashlib.sha256(payload).hexdigest(),
        "member_count": 1,
        "zip_method": "deflate9",
    }


def section_map(payload: bytes) -> Mapping[str, SectionBytes]:
    """Convenience read-only mapping after the complete strict parse."""

    return {row.name: row for row in parse_sections(payload)}


__all__ = [
    "SECTION_ORDER",
    "S4ArchiveError",
    "SectionBytes",
    "build_payload_manifest",
    "canonical_json_bytes",
    "deterministic_archive",
    "parse_sections",
    "section_map",
    "serialize_sections",
]
