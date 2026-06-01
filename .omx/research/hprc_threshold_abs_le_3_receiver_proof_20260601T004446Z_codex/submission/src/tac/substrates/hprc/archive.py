# SPDX-License-Identifier: MIT
"""Deterministic ``hprc.bin`` grammar for HPRC V0.

The packet is intentionally small and boring: a fixed header, a fixed-width
section table, and raw charged section bytes. Codec sophistication belongs
inside sections, not in an ad hoc wrapper. This gives receiver proofs,
materializers, byte profilers, and exact-readiness gates one common object to
mutate and audit.
"""

from __future__ import annotations

import binascii
import hashlib
import json
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum, StrEnum

HPRC_MAGIC: bytes = b"HPRC\x00\x00\x00\x00"
HPRC_SCHEMA_VERSION: int = 1

_HEADER_FMT = "<8sBHHHHHHHI11s"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_SECTION_FMT = "<HQQI32s"
_SECTION_SIZE = struct.calcsize(_SECTION_FMT)
_RESERVED = b"\x00" * 11


class HprcArchiveError(ValueError):
    """Raised when an HPRC packet is malformed or non-deterministic."""


class HprcSectionKind(IntEnum):
    """Canonical HPRC V0 charged section IDs.

    The names are deliberately representation-level, not implementation-level.
    PR95/HNeRV, Z8 teacher residuals, Cool-Chic/C3-style latent codecs, RAFT
    motion side information, SIREN/implicit bases, and future receivers can all
    share this packet without changing the outer grammar.
    """

    DECODER_QW = 1
    LATENTS_RC = 2
    CODEBOOKS_Q = 3
    SELECTORS_RC = 4
    RESIDUAL_RC = 5
    RDO_PLAN = 6
    RECEIVER_STATE = 7
    MANIFEST_JSON = 8


class HprcMutationProofStatus(StrEnum):
    """Section proof status encoded in manifests."""

    RAW_FLIP_ONLY = "raw_flip_only"
    VALID_SEMANTIC_MUTATION = "valid_semantic_mutation"
    FULL_RECEIVER_REPLAY = "full_receiver_replay"


@dataclass(frozen=True)
class HprcPacketConfig:
    """Fixed HPRC packet header.

    ``decoder_family_id`` and ``color_transform_id`` are numeric by design:
    they are compact archive fields and force any text naming/provenance into
    ``MANIFEST_JSON`` where it is charged and hashed.
    """

    frames: int = 1200
    pairs: int = 600
    height: int = 384
    width: int = 512
    decoder_family_id: int = 0
    color_transform_id: int = 0
    gop_size: int = 2

    def __post_init__(self) -> None:
        for name in (
            "frames",
            "pairs",
            "height",
            "width",
            "decoder_family_id",
            "color_transform_id",
            "gop_size",
        ):
            value = getattr(self, name)
            if not isinstance(value, int):
                raise HprcArchiveError(f"{name} must be int, got {type(value).__name__}")
            if value < 0 or value > 0xFFFF:
                raise HprcArchiveError(f"{name}={value} outside u16 range")
        if self.frames == 0 or self.pairs == 0 or self.height == 0 or self.width == 0:
            raise HprcArchiveError("frames/pairs/height/width must be positive")
        if self.gop_size == 0:
            raise HprcArchiveError("gop_size must be positive")

    def as_dict(self) -> dict[str, int]:
        return {
            "frames": self.frames,
            "pairs": self.pairs,
            "height": self.height,
            "width": self.width,
            "decoder_family_id": self.decoder_family_id,
            "color_transform_id": self.color_transform_id,
            "gop_size": self.gop_size,
        }


@dataclass(frozen=True)
class HprcSection:
    """Parsed section table row plus bytes."""

    kind: HprcSectionKind
    offset: int
    length: int
    crc32: int
    sha256: str
    payload: bytes

    @property
    def name(self) -> str:
        return self.kind.name.lower()

    def as_manifest_row(self) -> dict[str, int | str]:
        return {
            "id": int(self.kind),
            "name": self.name,
            "offset": self.offset,
            "length": self.length,
            "crc32": f"{self.crc32:08x}",
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class HprcPacket:
    """Parsed HPRC packet."""

    config: HprcPacketConfig
    sections: tuple[HprcSection, ...]
    schema_version: int = HPRC_SCHEMA_VERSION

    def section_map(self) -> dict[HprcSectionKind, bytes]:
        return {section.kind: section.payload for section in self.sections}

    def manifest(self) -> dict[str, object]:
        packet_bytes = (
            _HEADER_SIZE
            + len(self.sections) * _SECTION_SIZE
            + sum(section.length for section in self.sections)
        )
        return {
            "schema": "hprc_packet_manifest.v1",
            "magic": HPRC_MAGIC.decode("ascii", errors="ignore").rstrip("\x00"),
            "schema_version": self.schema_version,
            "config": self.config.as_dict(),
            "byte_accounting": {
                "packet_bytes": packet_bytes,
                "header_bytes": _HEADER_SIZE,
                "section_table_bytes": len(self.sections) * _SECTION_SIZE,
                "section_payload_bytes": sum(section.length for section in self.sections),
                "runtime_bytes_included": False,
                "zip_container_bytes_included": False,
                "contest_rate_bytes_authority": False,
                "authority_note": (
                    "hprc.bin payload accounting only; contest byte ceiling must "
                    "include archive.zip, inflate runtime, config, tables, and any "
                    "bundled decoder dependencies before promotion"
                ),
            },
            "sections": [section.as_manifest_row() for section in self.sections],
            "receiver_proof": {
                "required_for_promotion": (
                    "valid semantic section mutation plus full receiver replay; raw byte "
                    "flip/hash failure is parser integrity only"
                ),
                "status": HprcMutationProofStatus.RAW_FLIP_ONLY.value,
            },
            "score_claim": False,
            "promotion_eligible": False,
        }

    def manifest_json(self) -> str:
        return json.dumps(self.manifest(), sort_keys=True, separators=(",", ":"))


def _coerce_kind(kind: HprcSectionKind | int | str) -> HprcSectionKind:
    if isinstance(kind, HprcSectionKind):
        return kind
    if isinstance(kind, str):
        key = kind.upper()
        if key.startswith("HPRCSECTIONKIND."):
            key = key.rsplit(".", 1)[-1]
        try:
            return HprcSectionKind[key]
        except KeyError as exc:
            raise HprcArchiveError(f"unknown HPRC section kind {kind!r}") from exc
    try:
        return HprcSectionKind(int(kind))
    except (ValueError, TypeError) as exc:
        raise HprcArchiveError(f"unknown HPRC section kind {kind!r}") from exc


def _validate_section_payload(payload: bytes | bytearray | memoryview) -> bytes:
    data = bytes(payload)
    if len(data) > 0xFFFFFFFFFFFFFFFF:
        raise HprcArchiveError("section payload too large for u64 length")
    return data


def pack_hprc_packet(
    sections: Mapping[HprcSectionKind | int | str, bytes | bytearray | memoryview],
    *,
    config: HprcPacketConfig | None = None,
) -> bytes:
    """Pack a deterministic HPRC packet.

    Sections are sorted by numeric section ID. Empty sections are allowed when
    the receiver contract needs an explicit charged zero-length field, but
    duplicate logical section kinds are refused.
    """

    config = config or HprcPacketConfig()
    normalized: dict[HprcSectionKind, bytes] = {}
    for raw_kind, raw_payload in sections.items():
        kind = _coerce_kind(raw_kind)
        if kind in normalized:
            raise HprcArchiveError(f"duplicate section kind {kind.name}")
        normalized[kind] = _validate_section_payload(raw_payload)
    if not normalized:
        raise HprcArchiveError("HPRC packet requires at least one section")
    if len(normalized) > 0xFFFFFFFF:
        raise HprcArchiveError("too many sections for u32 section_count")

    ordered = sorted(normalized.items(), key=lambda item: int(item[0]))
    payload_offset = _HEADER_SIZE + len(ordered) * _SECTION_SIZE
    records: list[bytes] = []
    payloads: list[bytes] = []
    offset = payload_offset
    for kind, payload in ordered:
        crc = binascii.crc32(payload) & 0xFFFFFFFF
        digest = hashlib.sha256(payload).digest()
        records.append(struct.pack(_SECTION_FMT, int(kind), offset, len(payload), crc, digest))
        payloads.append(payload)
        offset += len(payload)

    header = struct.pack(
        _HEADER_FMT,
        HPRC_MAGIC,
        HPRC_SCHEMA_VERSION,
        config.frames,
        config.pairs,
        config.height,
        config.width,
        config.decoder_family_id,
        config.color_transform_id,
        config.gop_size,
        len(ordered),
        _RESERVED,
    )
    return header + b"".join(records) + b"".join(payloads)


def parse_hprc_packet(packet: bytes | bytearray | memoryview) -> HprcPacket:
    """Parse and validate an HPRC packet."""

    data = bytes(packet)
    if len(data) < _HEADER_SIZE:
        raise HprcArchiveError("HPRC packet truncated before header")
    (
        magic,
        version,
        frames,
        pairs,
        height,
        width,
        decoder_family_id,
        color_transform_id,
        gop_size,
        section_count,
        reserved,
    ) = struct.unpack(_HEADER_FMT, data[:_HEADER_SIZE])
    if magic != HPRC_MAGIC:
        raise HprcArchiveError(f"HPRC magic mismatch: {magic!r}")
    if version != HPRC_SCHEMA_VERSION:
        raise HprcArchiveError(
            f"HPRC schema version mismatch: expected {HPRC_SCHEMA_VERSION}, got {version}"
        )
    if reserved != _RESERVED:
        raise HprcArchiveError("HPRC reserved header bytes must be zero")

    table_end = _HEADER_SIZE + int(section_count) * _SECTION_SIZE
    if len(data) < table_end:
        raise HprcArchiveError("HPRC packet truncated in section table")

    config = HprcPacketConfig(
        frames=frames,
        pairs=pairs,
        height=height,
        width=width,
        decoder_family_id=decoder_family_id,
        color_transform_id=color_transform_id,
        gop_size=gop_size,
    )
    seen: set[HprcSectionKind] = set()
    sections: list[HprcSection] = []
    expected_offset = table_end
    for idx in range(int(section_count)):
        start = _HEADER_SIZE + idx * _SECTION_SIZE
        raw_kind, offset, length, crc, digest = struct.unpack(
            _SECTION_FMT, data[start : start + _SECTION_SIZE]
        )
        kind = _coerce_kind(raw_kind)
        if kind in seen:
            raise HprcArchiveError(f"duplicate section kind {kind.name}")
        seen.add(kind)
        if offset != expected_offset:
            raise HprcArchiveError(
                f"section {kind.name} offset {offset} != expected {expected_offset}"
            )
        end = offset + length
        if end > len(data):
            raise HprcArchiveError(f"section {kind.name} extends past packet end")
        payload = data[offset:end]
        actual_crc = binascii.crc32(payload) & 0xFFFFFFFF
        if actual_crc != crc:
            raise HprcArchiveError(
                f"section {kind.name} crc mismatch: expected {crc:08x}, got {actual_crc:08x}"
            )
        actual_digest = hashlib.sha256(payload).digest()
        if actual_digest != digest:
            raise HprcArchiveError(f"section {kind.name} sha256 mismatch")
        sections.append(
            HprcSection(
                kind=kind,
                offset=int(offset),
                length=int(length),
                crc32=int(crc),
                sha256=actual_digest.hex(),
                payload=payload,
            )
        )
        expected_offset = end

    if expected_offset != len(data):
        raise HprcArchiveError(
            f"HPRC packet has trailing bytes: expected end {expected_offset}, len {len(data)}"
        )
    return HprcPacket(config=config, sections=tuple(sections), schema_version=version)


def write_hprc_manifest(packet: bytes | bytearray | memoryview) -> dict[str, object]:
    """Return the machine-readable manifest for a packet without touching disk."""

    return parse_hprc_packet(packet).manifest()
