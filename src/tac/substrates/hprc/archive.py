# SPDX-License-Identifier: MIT
"""Deterministic ``hprc.bin`` grammars for HPRC.

The packet is intentionally small and boring: a fixed header, a fixed-width
section table, and raw charged section bytes. Codec sophistication belongs
inside sections, not in an ad hoc wrapper. This gives receiver proofs,
materializers, byte profilers, and exact-readiness gates one common object to
mutate and audit.

V0 is the debug/integrity grammar: every section row carries offset, length,
CRC32, and SHA-256. G1 is the contest-compact grammar: contest-default config
fields are implicit, present sections are a bitmask, offsets are derived, and
section hashes are kept in manifests/proofs rather than charged packet bytes.
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
HPRC_G1_MAGIC: bytes = b"HPRG"
HPRC_SCHEMA_VERSION: int = 1
HPRC_V0_GRAMMAR: str = "hprc_v0_fixed_table"
HPRC_G1_GRAMMAR: str = "hprc_g1_compact_bitmask_varint"

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


_G1_IMPLICIT_CONFIG = HprcPacketConfig()
_G1_CONFIG_FIELDS = (
    "frames",
    "pairs",
    "height",
    "width",
    "decoder_family_id",
    "color_transform_id",
    "gop_size",
)


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
    grammar: str = HPRC_V0_GRAMMAR
    magic: bytes = HPRC_MAGIC
    packet_bytes: int | None = None
    header_bytes: int | None = None
    section_table_bytes: int | None = None

    def section_map(self) -> dict[HprcSectionKind, bytes]:
        return {section.kind: section.payload for section in self.sections}

    def manifest(self) -> dict[str, object]:
        packet_bytes = self.packet_bytes or (
            _HEADER_SIZE
            + len(self.sections) * _SECTION_SIZE
            + sum(section.length for section in self.sections)
        )
        header_bytes = (
            _HEADER_SIZE if self.header_bytes is None else int(self.header_bytes)
        )
        section_table_bytes = (
            len(self.sections) * _SECTION_SIZE
            if self.section_table_bytes is None
            else int(self.section_table_bytes)
        )
        return {
            "schema": "hprc_packet_manifest.v1",
            "grammar": self.grammar,
            "magic": self.magic.decode("ascii", errors="ignore").rstrip("\x00"),
            "schema_version": self.schema_version,
            "config": self.config.as_dict(),
            "byte_accounting": {
                "packet_bytes": packet_bytes,
                "header_bytes": header_bytes,
                "section_table_bytes": section_table_bytes,
                "section_payload_bytes": sum(section.length for section in self.sections),
                "wrapper_overhead_bytes": packet_bytes
                - sum(section.length for section in self.sections),
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
                "packet_integrity_scope": _packet_integrity_scope(self.grammar),
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


def _normalize_sections(
    sections: Mapping[HprcSectionKind | int | str, bytes | bytearray | memoryview],
    *,
    packet_label: str,
) -> list[tuple[HprcSectionKind, bytes]]:
    normalized: dict[HprcSectionKind, bytes] = {}
    for raw_kind, raw_payload in sections.items():
        kind = _coerce_kind(raw_kind)
        if kind in normalized:
            raise HprcArchiveError(f"duplicate section kind {kind.name}")
        normalized[kind] = _validate_section_payload(raw_payload)
    if not normalized:
        raise HprcArchiveError(f"{packet_label} requires at least one section")
    return sorted(normalized.items(), key=lambda item: int(item[0]))


def _packet_integrity_scope(grammar: str) -> str:
    if grammar == HPRC_G1_GRAMMAR:
        return (
            "grammar_bounds_only; section integrity is provided by the containing "
            "archive CRC/SHA and by external manifests, not by charged in-packet hashes"
        )
    return "grammar_bounds_plus_embedded_crc32_sha256_per_section"


def _encode_uvarint(value: int) -> bytes:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise HprcArchiveError(f"uvarint value must be a non-negative int: {value!r}")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _decode_uvarint(data: bytes, offset: int) -> tuple[int, int]:
    shift = 0
    value = 0
    pos = int(offset)
    while pos < len(data):
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, pos
        shift += 7
        if shift > 63:
            raise HprcArchiveError("uvarint exceeds u64 range")
    raise HprcArchiveError("truncated uvarint")


def _valid_g1_config_mask() -> int:
    return (1 << len(_G1_CONFIG_FIELDS)) - 1


def _valid_g1_section_mask() -> int:
    mask = 0
    for kind in HprcSectionKind:
        mask |= 1 << (int(kind) - 1)
    return mask


def _g1_config_delta_mask(config: HprcPacketConfig) -> tuple[int, list[int]]:
    defaults = _G1_IMPLICIT_CONFIG.as_dict()
    values = config.as_dict()
    mask = 0
    deltas: list[int] = []
    for bit, field in enumerate(_G1_CONFIG_FIELDS):
        value = int(values[field])
        if value != int(defaults[field]):
            mask |= 1 << bit
            deltas.append(value)
    return mask, deltas


def _g1_config_from_delta_mask(
    mask: int, data: bytes, offset: int
) -> tuple[HprcPacketConfig, int]:
    if mask & ~_valid_g1_config_mask():
        raise HprcArchiveError(f"HPRC G1 config mask has unknown bits: {mask:#x}")
    defaults = _G1_IMPLICIT_CONFIG.as_dict()
    values = dict(defaults)
    pos = offset
    for bit, field in enumerate(_G1_CONFIG_FIELDS):
        if mask & (1 << bit):
            values[field], pos = _decode_uvarint(data, pos)
    return HprcPacketConfig(**values), pos


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
    ordered = _normalize_sections(sections, packet_label="HPRC packet")
    if len(ordered) > 0xFFFFFFFF:
        raise HprcArchiveError("too many sections for u32 section_count")

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


def pack_hprc_g1_packet(
    sections: Mapping[HprcSectionKind | int | str, bytes | bytearray | memoryview],
    *,
    config: HprcPacketConfig | None = None,
) -> bytes:
    """Pack the contest-compact HPRC G1 grammar.

    G1 keeps V0's semantic sections but removes charged debug overhead:
    contest-default config fields are implicit, present sections are a bitmask,
    section lengths are unsigned varints, offsets are derived, and hashes live
    in manifests/proofs.
    """

    config = config or HprcPacketConfig()
    ordered = _normalize_sections(sections, packet_label="HPRC G1 packet")
    section_mask = 0
    for kind, _payload in ordered:
        section_mask |= 1 << (int(kind) - 1)
    config_mask, config_deltas = _g1_config_delta_mask(config)
    header = bytearray()
    header.extend(HPRC_G1_MAGIC)
    header.append(HPRC_SCHEMA_VERSION)
    header.extend(_encode_uvarint(config_mask))
    for value in config_deltas:
        header.extend(_encode_uvarint(value))
    header.extend(_encode_uvarint(section_mask))
    for _kind, payload in ordered:
        header.extend(_encode_uvarint(len(payload)))
    return bytes(header) + b"".join(payload for _kind, payload in ordered)


def parse_hprc_packet(packet: bytes | bytearray | memoryview) -> HprcPacket:
    """Parse and validate an HPRC packet."""

    data = bytes(packet)
    if len(data) >= len(HPRC_G1_MAGIC) + 1 and data[: len(HPRC_G1_MAGIC)] == HPRC_G1_MAGIC:
        return _parse_hprc_g1_packet(data)
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
    return HprcPacket(
        config=config,
        sections=tuple(sections),
        schema_version=version,
        grammar=HPRC_V0_GRAMMAR,
        magic=HPRC_MAGIC,
        packet_bytes=len(data),
        header_bytes=_HEADER_SIZE,
        section_table_bytes=int(section_count) * _SECTION_SIZE,
    )


def _parse_hprc_g1_packet(data: bytes) -> HprcPacket:
    """Parse the compact HPRC G1 grammar."""

    if len(data) < len(HPRC_G1_MAGIC) + 1:
        raise HprcArchiveError("HPRC G1 packet truncated before schema version")
    version = data[len(HPRC_G1_MAGIC)]
    if version != HPRC_SCHEMA_VERSION:
        raise HprcArchiveError(
            f"HPRC G1 schema version mismatch: expected {HPRC_SCHEMA_VERSION}, got {version}"
        )
    pos = len(HPRC_G1_MAGIC) + 1
    config_mask, pos = _decode_uvarint(data, pos)
    config, pos = _g1_config_from_delta_mask(config_mask, data, pos)
    section_mask, pos = _decode_uvarint(data, pos)
    if section_mask == 0:
        raise HprcArchiveError("HPRC G1 packet requires at least one section")
    if section_mask & ~_valid_g1_section_mask():
        raise HprcArchiveError(f"HPRC G1 section mask has unknown bits: {section_mask:#x}")

    kinds = [
        kind
        for kind in sorted(HprcSectionKind, key=int)
        if section_mask & (1 << (int(kind) - 1))
    ]
    lengths: list[int] = []
    for kind in kinds:
        length, pos = _decode_uvarint(data, pos)
        if length > len(data):
            raise HprcArchiveError(
                f"section {kind.name} declared length {length} exceeds packet size {len(data)}"
            )
        lengths.append(length)

    payload_start = pos
    sections: list[HprcSection] = []
    for kind, length in zip(kinds, lengths, strict=True):
        end = pos + length
        if end > len(data):
            raise HprcArchiveError(f"section {kind.name} extends past packet end")
        payload = data[pos:end]
        crc = binascii.crc32(payload) & 0xFFFFFFFF
        digest = hashlib.sha256(payload).digest()
        sections.append(
            HprcSection(
                kind=kind,
                offset=pos,
                length=int(length),
                crc32=crc,
                sha256=digest.hex(),
                payload=payload,
            )
        )
        pos = end

    if pos != len(data):
        raise HprcArchiveError(
            f"HPRC G1 packet has trailing bytes: expected end {pos}, len {len(data)}"
        )
    return HprcPacket(
        config=config,
        sections=tuple(sections),
        schema_version=version,
        grammar=HPRC_G1_GRAMMAR,
        magic=HPRC_G1_MAGIC,
        packet_bytes=len(data),
        header_bytes=payload_start,
        section_table_bytes=0,
    )


def write_hprc_manifest(packet: bytes | bytearray | memoryview) -> dict[str, object]:
    """Return the machine-readable manifest for a packet without touching disk."""

    return parse_hprc_packet(packet).manifest()


def hprc_grammar_byte_profile(
    sections: Mapping[HprcSectionKind | int | str, bytes | bytearray | memoryview],
    *,
    config: HprcPacketConfig | None = None,
) -> dict[str, object]:
    """Compare V0 and G1 paid bytes for the same semantic sections.

    This is acquisition signal only. It profiles the raw ``hprc.bin`` payload
    grammar and intentionally refuses contest-rate authority because ZIP/runtime
    bytes and receiver proof are outside this helper.
    """

    cfg = config or HprcPacketConfig()
    v0 = pack_hprc_packet(sections, config=cfg)
    g1 = pack_hprc_g1_packet(sections, config=cfg)
    v0_packet = parse_hprc_packet(v0)
    g1_packet = parse_hprc_packet(g1)
    payload_bytes = sum(section.length for section in v0_packet.sections)
    delta = len(v0) - len(g1)
    return {
        "schema": "hprc_grammar_byte_profile.v1",
        "semantic_section_count": len(v0_packet.sections),
        "section_payload_bytes": payload_bytes,
        "grammars": [
            {
                "grammar": HPRC_V0_GRAMMAR,
                "packet_bytes": len(v0),
                "wrapper_overhead_bytes": len(v0) - payload_bytes,
                "header_bytes": _HEADER_SIZE,
                "section_table_bytes": len(v0_packet.sections) * _SECTION_SIZE,
                "packet_integrity_scope": _packet_integrity_scope(HPRC_V0_GRAMMAR),
            },
            {
                "grammar": HPRC_G1_GRAMMAR,
                "packet_bytes": len(g1),
                "wrapper_overhead_bytes": len(g1) - payload_bytes,
                "header_bytes": g1_packet.header_bytes,
                "section_table_bytes": 0,
                "packet_integrity_scope": _packet_integrity_scope(HPRC_G1_GRAMMAR),
            },
        ],
        "recommended_grammar": HPRC_G1_GRAMMAR if delta > 0 else HPRC_V0_GRAMMAR,
        "g1_saves_bytes": max(0, delta),
        "g1_saves_fraction_of_v0_packet": 0.0 if not v0 else max(0, delta) / len(v0),
        "score_claim": False,
        "promotion_eligible": False,
        "contest_rate_bytes_authority": False,
    }


__all__ = [
    "HPRC_G1_GRAMMAR",
    "HPRC_G1_MAGIC",
    "HPRC_MAGIC",
    "HPRC_SCHEMA_VERSION",
    "HPRC_V0_GRAMMAR",
    "HprcArchiveError",
    "HprcMutationProofStatus",
    "HprcPacket",
    "HprcPacketConfig",
    "HprcSection",
    "HprcSectionKind",
    "hprc_grammar_byte_profile",
    "pack_hprc_g1_packet",
    "pack_hprc_packet",
    "parse_hprc_packet",
    "write_hprc_manifest",
]
