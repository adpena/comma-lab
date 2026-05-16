# SPDX-License-Identifier: MIT
"""RDIF v1 archive grammar for the Rudin floor research scaffold."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping
from dataclasses import dataclass

from tac.substrates.rudin_floor_interpretable_ml.rule_list import RudinRuleList

RDIF_MAGIC: bytes = b"RDF1"
RDIF_VERSION: int = 1
RDIF_HEADER_SIZE: int = 34
CANONICAL_K_RULES: int = 6
CANONICAL_K_RASHOMON: int = 8
CANONICAL_SLIM_COEFF_BOUND: int = 10
CANONICAL_GOSDT_DEPTH: int = 4

_SECTION_NAMES: tuple[str, ...] = (
    "encoder_tree_blob",
    "rule_list_blob",
    "scorer_priors_blob",
    "frame_0_init_blob",
    "wavelet_residuals_blob",
    "pose_residuals_blob",
    "per_pair_rule_indices_blob",
    "rashomon_disagreement_blob",
)
_DIRECTORY_ENTRY = struct.Struct("<I")
_SHA256_LEN = 32


@dataclass(frozen=True)
class RDIFv1Header:
    """Fixed-size RDIF v1 header."""

    magic: bytes
    version: int
    section_count: int
    flags: int
    payload_len: int
    payload_sha256_prefix: bytes


@dataclass(frozen=True)
class RDIFv1Archive:
    """Parsed RDIF v1 archive."""

    header: RDIFv1Header
    sections: Mapping[str, bytes]
    payload_sha256: bytes

    @property
    def rule_list(self) -> RudinRuleList:
        """Return the parsed falling rule list."""

        return RudinRuleList.from_json_bytes(self.sections["rule_list_blob"])


def pack_archive(
    *,
    rule_list: RudinRuleList,
    encoder_tree_blob: bytes = b"",
    scorer_priors_blob: bytes = b"",
    frame_0_init_blob: bytes = b"",
    wavelet_residuals_blob: bytes = b"",
    pose_residuals_blob: bytes = b"",
    per_pair_rule_indices_blob: bytes = b"",
    rashomon_disagreement_blob: bytes = b"",
    flags: int = 0,
) -> bytes:
    """Pack a monolithic RDIF v1 ``0.bin`` payload."""

    sections = {
        "encoder_tree_blob": encoder_tree_blob,
        "rule_list_blob": rule_list.to_json_bytes(),
        "scorer_priors_blob": scorer_priors_blob,
        "frame_0_init_blob": frame_0_init_blob,
        "wavelet_residuals_blob": wavelet_residuals_blob,
        "pose_residuals_blob": pose_residuals_blob,
        "per_pair_rule_indices_blob": per_pair_rule_indices_blob,
        "rashomon_disagreement_blob": rashomon_disagreement_blob,
    }
    directory = b"".join(
        _DIRECTORY_ENTRY.pack(len(sections[name])) for name in _SECTION_NAMES
    )
    payload = b"".join(sections[name] for name in _SECTION_NAMES)
    payload_sha = hashlib.sha256(directory + payload).digest()
    header = (
        RDIF_MAGIC
        + struct.pack("<H", RDIF_VERSION)
        + struct.pack("<H", len(_SECTION_NAMES))
        + struct.pack("<H", int(flags))
        + struct.pack("<Q", len(directory) + len(payload))
        + payload_sha[:16]
    )
    if len(header) != RDIF_HEADER_SIZE:
        raise AssertionError(f"RDIF header drift: {len(header)}")
    return header + directory + payload + payload_sha


def parse_archive(archive_bytes: bytes) -> RDIFv1Archive:
    """Parse a RDIF v1 payload and validate its payload hash."""

    if len(archive_bytes) < RDIF_HEADER_SIZE + _SHA256_LEN:
        raise ValueError("RDIF archive too short")
    if archive_bytes[: len(RDIF_MAGIC)] != RDIF_MAGIC:
        raise ValueError("RDIF magic mismatch")
    offset = len(RDIF_MAGIC)
    version, section_count, flags = struct.unpack(
        "<HHH", archive_bytes[offset : offset + 6]
    )
    offset += 6
    if version != RDIF_VERSION:
        raise ValueError(f"RDIF version mismatch: {version}")
    if section_count != len(_SECTION_NAMES):
        raise ValueError(f"RDIF section count mismatch: {section_count}")
    (payload_len,) = struct.unpack("<Q", archive_bytes[offset : offset + 8])
    offset += 8
    payload_sha_prefix = archive_bytes[offset : offset + 16]
    offset += 16
    expected_total = RDIF_HEADER_SIZE + payload_len + _SHA256_LEN
    if len(archive_bytes) != expected_total:
        raise ValueError(
            f"RDIF archive length mismatch: {len(archive_bytes)} != {expected_total}"
        )

    payload_region = archive_bytes[RDIF_HEADER_SIZE : RDIF_HEADER_SIZE + payload_len]
    payload_sha = archive_bytes[-_SHA256_LEN:]
    actual_sha = hashlib.sha256(payload_region).digest()
    if actual_sha != payload_sha:
        raise ValueError("RDIF payload sha256 mismatch")
    if actual_sha[:16] != payload_sha_prefix:
        raise ValueError("RDIF payload sha256 prefix mismatch")

    directory_len = len(_SECTION_NAMES) * _DIRECTORY_ENTRY.size
    directory = payload_region[:directory_len]
    payload = payload_region[directory_len:]
    lengths = [
        _DIRECTORY_ENTRY.unpack(
            directory[i : i + _DIRECTORY_ENTRY.size]
        )[0]
        for i in range(0, directory_len, _DIRECTORY_ENTRY.size)
    ]
    sections: dict[str, bytes] = {}
    cursor = 0
    for name, length in zip(_SECTION_NAMES, lengths, strict=True):
        sections[name] = payload[cursor : cursor + length]
        cursor += length
    if cursor != len(payload):
        raise ValueError("RDIF section directory length mismatch")

    return RDIFv1Archive(
        header=RDIFv1Header(
            magic=RDIF_MAGIC,
            version=version,
            section_count=section_count,
            flags=flags,
            payload_len=payload_len,
            payload_sha256_prefix=payload_sha_prefix,
        ),
        sections=sections,
        payload_sha256=payload_sha,
    )


__all__ = [
    "CANONICAL_GOSDT_DEPTH",
    "CANONICAL_K_RASHOMON",
    "CANONICAL_K_RULES",
    "CANONICAL_SLIM_COEFF_BOUND",
    "RDIF_HEADER_SIZE",
    "RDIF_MAGIC",
    "RDIF_VERSION",
    "RDIFv1Archive",
    "RDIFv1Header",
    "pack_archive",
    "parse_archive",
]
