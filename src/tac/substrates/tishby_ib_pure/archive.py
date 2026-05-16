# SPDX-License-Identifier: MIT
"""TIBP1 archive grammar for the Tishby IB-pure L1 scaffold."""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256

TIBP1_MAGIC = b"TIBP"
TIBP1_SCHEMA_VERSION = 1
TIBP1_SECTION_ROLES = (
    "encoder_blob",
    "decoder_blob",
    "statistic_net_blob",
    "latent_t_blob",
    "scorer_class_prior_blob",
    "cdf_table_blob",
    "meta_blob",
    "reserved_blob",
)

TIBP1_HEADER_FMT = "<4sHH8I32s"
TIBP1_HEADER_SIZE = struct.calcsize(TIBP1_HEADER_FMT)


@dataclass(frozen=True)
class TishbyIBPureArchive:
    """Parsed TIBP1 archive."""

    sections: dict[str, bytes]
    content_sha256: str

    @property
    def meta(self) -> dict[str, object]:
        blob = self.sections.get("meta_blob", b"{}")
        return json.loads(blob.decode("utf-8"))


def _normalize_sections(sections: Mapping[str, bytes] | None) -> dict[str, bytes]:
    normalized = dict.fromkeys(TIBP1_SECTION_ROLES, b"")
    if sections:
        unknown = sorted(set(sections) - set(TIBP1_SECTION_ROLES))
        if unknown:
            raise ValueError(f"unknown TIBP1 section role(s): {unknown}")
        for role, blob in sections.items():
            normalized[role] = bytes(blob)
    if not normalized["meta_blob"]:
        normalized["meta_blob"] = json.dumps(
            {
                "archive_format": "TIBP1",
                "research_only": True,
                "score_claim": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    return normalized


def pack_archive(sections: Mapping[str, bytes] | None = None) -> bytes:
    """Pack deterministic TIBP1 bytes from named sections."""

    normalized = _normalize_sections(sections)
    lengths = [len(normalized[role]) for role in TIBP1_SECTION_ROLES]
    payload = b"".join(normalized[role] for role in TIBP1_SECTION_ROLES)
    digest = sha256(payload).digest()
    header = struct.pack(
        TIBP1_HEADER_FMT,
        TIBP1_MAGIC,
        TIBP1_SCHEMA_VERSION,
        len(TIBP1_SECTION_ROLES),
        *lengths,
        digest,
    )
    return header + payload


def parse_archive(data: bytes) -> TishbyIBPureArchive:
    """Parse TIBP1 archive bytes and verify payload digest."""

    if len(data) < TIBP1_HEADER_SIZE:
        raise ValueError("TIBP1 archive too short")
    unpacked = struct.unpack(TIBP1_HEADER_FMT, data[:TIBP1_HEADER_SIZE])
    magic = unpacked[0]
    version = unpacked[1]
    section_count = unpacked[2]
    lengths = list(unpacked[3:11])
    expected_digest = unpacked[11]
    if magic != TIBP1_MAGIC:
        raise ValueError(f"bad TIBP1 magic: {magic!r}")
    if version != TIBP1_SCHEMA_VERSION:
        raise ValueError(f"unsupported TIBP1 schema version: {version}")
    if section_count != len(TIBP1_SECTION_ROLES):
        raise ValueError(f"unexpected TIBP1 section count: {section_count}")
    payload = data[TIBP1_HEADER_SIZE:]
    if len(payload) != sum(lengths):
        raise ValueError("TIBP1 payload length mismatch")
    if sha256(payload).digest() != expected_digest:
        raise ValueError("TIBP1 payload digest mismatch")
    sections: dict[str, bytes] = {}
    offset = 0
    for role, length in zip(TIBP1_SECTION_ROLES, lengths, strict=True):
        sections[role] = payload[offset : offset + length]
        offset += length
    return TishbyIBPureArchive(
        sections=sections,
        content_sha256=sha256(data).hexdigest(),
    )


def parse_tibp1_archive_bytes(data: bytes) -> TishbyIBPureArchive:
    """Compatibility alias used by registry/intake tools."""

    return parse_archive(data)
