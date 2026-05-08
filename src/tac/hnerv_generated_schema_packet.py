"""Deterministic HNGP v1 packets for generated-schema HNeRV artifacts.

HNGP v1 is a monolithic, non-score packet scaffold:

``header + hngs_decoder + latent_blob + sidecar_blob``

The header records byte lengths and SHA-256 digests for the three payload
sections. Parsing recomputes offsets, lengths, and hashes fail-closed before
returning any section bytes. This module does not integrate with a submission
runtime, does not dispatch GPU work, and does not make score or promotion
claims.
"""
from __future__ import annotations

import hashlib
import json
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

HNGP_MAGIC = b"HNGP"
HNGP_VERSION = 1
HNGP_HEADER_SCHEMA = "tac_hnerv_generated_schema_packet_header.v1"
HNGP_MANIFEST_SCHEMA = "tac_hnerv_generated_schema_packet_manifest.v1"
HNGS_DECODER_MAGIC = b"HNGS"
PACKET_PREAMBLE = struct.Struct("<4sBI")

HNGP_SECTION_ORDER = ("hngs_decoder", "latent_blob", "sidecar_blob")
HEADER_SECTION_NAME = "header"
SECTION_ROLES: dict[str, str] = {
    HEADER_SECTION_NAME: "hngp_v1_length_and_hash_manifest",
    "hngs_decoder": "generated_schema_hnerv_decoder_blob",
    "latent_blob": "generated_schema_latent_payload",
    "sidecar_blob": "generated_schema_auxiliary_sidecar_payload",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_METADATA_KEYS = frozenset(
    {
        "score",
        "score_claim",
        "promotion_eligible",
        "rank_or_kill_eligible",
        "ready_for_exact_eval_dispatch",
        "evidence_grade",
    }
)


class HNeRVGeneratedSchemaPacketError(ValueError):
    """Raised when an HNGP packet is malformed or unsupported."""


@dataclass(frozen=True)
class HNGPSection:
    """Byte accounting for one HNGP section."""

    name: str
    role: str
    offset: int
    length: int
    sha256: str

    def to_manifest(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready section manifest."""

        return {
            "name": self.name,
            "role": self.role,
            "offset": self.offset,
            "len": self.length,
            "length": self.length,
            "bytes": self.length,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class HNeRVGeneratedSchemaPacket:
    """Parsed or built HNGP v1 packet plus its non-score manifest."""

    packet: bytes
    header: dict[str, Any]
    hngs_decoder: bytes
    latent_blob: bytes
    sidecar_blob: bytes
    sections: tuple[HNGPSection, ...]
    manifest: dict[str, Any]


def build_hnerv_generated_schema_packet(
    *,
    hngs_decoder: bytes,
    latent_blob: bytes,
    sidecar_blob: bytes,
    metadata: Mapping[str, Any] | None = None,
) -> HNeRVGeneratedSchemaPacket:
    """Build a deterministic HNGP v1 packet from three payload sections.

    Args:
        hngs_decoder: Generated-schema decoder blob. It must start with the
            ``HNGS`` magic emitted by :mod:`tac.hnerv_generated_schema_codec`.
        latent_blob: Latent payload bytes. Empty bytes are allowed for scaffold
            and negative-test vectors, but the section is always present.
        sidecar_blob: Auxiliary sidecar bytes. Empty bytes are allowed.
        metadata: Optional JSON-serializable non-score metadata. Score or
            dispatch-readiness keys are rejected to keep this scaffold
            unambiguous.

    Returns:
        Parsed packet object containing the byte-identical packet and manifest.

    Raises:
        HNeRVGeneratedSchemaPacketError: If any input violates the HNGP v1
            scaffold contract.
    """

    hngs_decoder = _coerce_section_bytes("hngs_decoder", hngs_decoder)
    latent_blob = _coerce_section_bytes("latent_blob", latent_blob)
    sidecar_blob = _coerce_section_bytes("sidecar_blob", sidecar_blob)
    if not hngs_decoder.startswith(HNGS_DECODER_MAGIC):
        raise HNeRVGeneratedSchemaPacketError(
            "hngs_decoder does not start with HNGS magic"
        )

    normalized_metadata = _normalize_metadata(metadata)
    section_payloads = {
        "hngs_decoder": hngs_decoder,
        "latent_blob": latent_blob,
        "sidecar_blob": sidecar_blob,
    }
    header = {
        "schema": HNGP_HEADER_SCHEMA,
        "magic": HNGP_MAGIC.decode("ascii"),
        "version": HNGP_VERSION,
        "packet_grammar": "hngp_v1",
        "monolithic_packet": True,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "section_order": list(HNGP_SECTION_ORDER),
        "sections": [
            {
                "name": name,
                "role": SECTION_ROLES[name],
                "len": len(section_payloads[name]),
                "sha256": _sha256_bytes(section_payloads[name]),
            }
            for name in HNGP_SECTION_ORDER
        ],
        "metadata": normalized_metadata,
    }
    header_bytes = _canonical_json(header)
    packet = (
        PACKET_PREAMBLE.pack(HNGP_MAGIC, HNGP_VERSION, len(header_bytes))
        + header_bytes
        + hngs_decoder
        + latent_blob
        + sidecar_blob
    )
    return parse_hnerv_generated_schema_packet(packet)


def parse_hnerv_generated_schema_packet(packet: bytes) -> HNeRVGeneratedSchemaPacket:
    """Parse and validate one HNGP v1 monolithic packet.

    The parser rejects unsupported magic/version values, invalid JSON headers,
    duplicate or unknown sections, truncated sections, trailing bytes, and any
    section whose bytes do not match the header SHA-256.
    """

    packet = _coerce_section_bytes("packet", packet)
    if len(packet) < PACKET_PREAMBLE.size:
        raise HNeRVGeneratedSchemaPacketError("HNGP packet too short for preamble")

    magic, version, header_len = PACKET_PREAMBLE.unpack_from(packet, 0)
    if magic != HNGP_MAGIC:
        raise HNeRVGeneratedSchemaPacketError(f"bad HNGP magic: {magic!r}")
    if int(version) != HNGP_VERSION:
        raise HNeRVGeneratedSchemaPacketError(
            f"unsupported HNGP version: {int(version)}"
        )
    if int(header_len) <= 0:
        raise HNeRVGeneratedSchemaPacketError("HNGP header length must be positive")

    header_start = PACKET_PREAMBLE.size
    header_end = header_start + int(header_len)
    if header_end > len(packet):
        raise HNeRVGeneratedSchemaPacketError("truncated HNGP JSON header")
    header_bytes = packet[header_start:header_end]
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HNeRVGeneratedSchemaPacketError("invalid HNGP JSON header") from exc
    if not isinstance(header, dict):
        raise HNeRVGeneratedSchemaPacketError("HNGP JSON header must be an object")

    rows = _validate_header(header)
    cursor = header_end
    sections: list[HNGPSection] = [
        HNGPSection(
            name=HEADER_SECTION_NAME,
            role=SECTION_ROLES[HEADER_SECTION_NAME],
            offset=0,
            length=header_end,
            sha256=_sha256_bytes(packet[:header_end]),
        )
    ]
    payloads: dict[str, bytes] = {}
    for row in rows:
        name = str(row["name"])
        section_len = int(row["len"])
        section_end = cursor + section_len
        if section_end > len(packet):
            raise HNeRVGeneratedSchemaPacketError(
                f"truncated HNGP section {name!r}: "
                f"needs end offset {section_end}, packet has {len(packet)} bytes"
            )
        section = packet[cursor:section_end]
        section_sha = _sha256_bytes(section)
        expected_sha = str(row["sha256"])
        if section_sha != expected_sha:
            raise HNeRVGeneratedSchemaPacketError(
                f"HNGP section {name!r} sha256 mismatch: "
                f"{section_sha} != {expected_sha}"
            )
        payloads[name] = section
        sections.append(
            HNGPSection(
                name=name,
                role=str(row["role"]),
                offset=cursor,
                length=section_len,
                sha256=section_sha,
            )
        )
        cursor = section_end

    if cursor != len(packet):
        raise HNeRVGeneratedSchemaPacketError(
            f"trailing bytes after HNGP sections: {len(packet) - cursor}"
        )
    if not payloads["hngs_decoder"].startswith(HNGS_DECODER_MAGIC):
        raise HNeRVGeneratedSchemaPacketError(
            "hngs_decoder section does not start with HNGS magic"
        )

    manifest = _build_manifest(
        packet=packet,
        header=header,
        sections=tuple(sections),
    )
    return HNeRVGeneratedSchemaPacket(
        packet=packet,
        header=header,
        hngs_decoder=payloads["hngs_decoder"],
        latent_blob=payloads["latent_blob"],
        sidecar_blob=payloads["sidecar_blob"],
        sections=tuple(sections),
        manifest=manifest,
    )


def inspect_hnerv_generated_schema_packet(packet: bytes) -> dict[str, Any]:
    """Return the no-score HNGP v1 manifest for ``packet``."""

    return parse_hnerv_generated_schema_packet(packet).manifest


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest for ``data``."""

    return _sha256_bytes(data)


def _validate_header(header: dict[str, Any]) -> list[dict[str, Any]]:
    if header.get("schema") != HNGP_HEADER_SCHEMA:
        raise HNeRVGeneratedSchemaPacketError(
            f"unsupported HNGP header schema: {header.get('schema')!r}"
        )
    if header.get("magic") != HNGP_MAGIC.decode("ascii"):
        raise HNeRVGeneratedSchemaPacketError(
            f"bad HNGP header magic: {header.get('magic')!r}"
        )
    if header.get("version") != HNGP_VERSION:
        raise HNeRVGeneratedSchemaPacketError(
            f"bad HNGP header version: {header.get('version')!r}"
        )
    if header.get("packet_grammar") != "hngp_v1":
        raise HNeRVGeneratedSchemaPacketError(
            f"unsupported HNGP packet grammar: {header.get('packet_grammar')!r}"
        )
    if header.get("monolithic_packet") is not True:
        raise HNeRVGeneratedSchemaPacketError(
            "HNGP header must set monolithic_packet=true"
        )
    if header.get("score_claim") is not False:
        raise HNeRVGeneratedSchemaPacketError("HNGP header must set score_claim=false")
    if header.get("promotion_eligible") is not False:
        raise HNeRVGeneratedSchemaPacketError(
            "HNGP header must set promotion_eligible=false"
        )
    if header.get("ready_for_exact_eval_dispatch") is not False:
        raise HNeRVGeneratedSchemaPacketError(
            "HNGP header must set ready_for_exact_eval_dispatch=false"
        )
    if tuple(header.get("section_order") or ()) != HNGP_SECTION_ORDER:
        raise HNeRVGeneratedSchemaPacketError("HNGP section_order mismatch")
    metadata = header.get("metadata")
    if not isinstance(metadata, dict):
        raise HNeRVGeneratedSchemaPacketError("HNGP metadata must be an object")
    forbidden_metadata = sorted(set(metadata) & _FORBIDDEN_METADATA_KEYS)
    if forbidden_metadata:
        raise HNeRVGeneratedSchemaPacketError(
            "HNGP metadata contains reserved score/readiness keys: "
            + ", ".join(forbidden_metadata)
        )

    rows = header.get("sections")
    if not isinstance(rows, list):
        raise HNeRVGeneratedSchemaPacketError("HNGP sections must be a list")
    if len(rows) != len(HNGP_SECTION_ORDER):
        raise HNeRVGeneratedSchemaPacketError(
            f"HNGP expected {len(HNGP_SECTION_ORDER)} sections, got {len(rows)}"
        )

    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise HNeRVGeneratedSchemaPacketError("HNGP section row must be an object")
        name = row.get("name")
        if not isinstance(name, str):
            raise HNeRVGeneratedSchemaPacketError("HNGP section name must be a string")
        if name in seen:
            raise HNeRVGeneratedSchemaPacketError(f"duplicate HNGP section {name!r}")
        seen.add(name)
        expected = HNGP_SECTION_ORDER[index]
        if name != expected:
            raise HNeRVGeneratedSchemaPacketError(
                f"HNGP section order mismatch at index {index}: {name!r} != {expected!r}"
            )
        if row.get("role") != SECTION_ROLES[name]:
            raise HNeRVGeneratedSchemaPacketError(
                f"HNGP section {name!r} role mismatch: {row.get('role')!r}"
            )
        length = row.get("len")
        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            raise HNeRVGeneratedSchemaPacketError(
                f"HNGP section {name!r} has invalid length {length!r}"
            )
        digest = row.get("sha256")
        if not isinstance(digest, str) or not _SHA256_RE.match(digest):
            raise HNeRVGeneratedSchemaPacketError(
                f"HNGP section {name!r} has invalid sha256 {digest!r}"
            )
        validated.append(
            {
                "name": name,
                "role": row["role"],
                "len": length,
                "sha256": digest,
            }
        )

    if set(seen) != set(HNGP_SECTION_ORDER):
        raise HNeRVGeneratedSchemaPacketError("HNGP section set mismatch")
    return validated


def _build_manifest(
    *,
    packet: bytes,
    header: dict[str, Any],
    sections: tuple[HNGPSection, ...],
) -> dict[str, Any]:
    section_manifests = [section.to_manifest() for section in sections]
    section_by_name = {section.name: section for section in sections}
    return {
        "schema": HNGP_MANIFEST_SCHEMA,
        "packet_grammar": "hngp_v1",
        "monolithic_packet": True,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "dispatch_attempted": False,
        "gpu_required": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "empirical_packet_scaffold_no_score",
        "packet_bytes": len(packet),
        "packet_sha256": _sha256_bytes(packet),
        "header_bytes": section_by_name[HEADER_SECTION_NAME].length,
        "section_count": len(sections),
        "payload_section_count": len(HNGP_SECTION_ORDER),
        "sections": section_manifests,
        "payload_sections": [
            section_by_name[name].to_manifest() for name in HNGP_SECTION_ORDER
        ],
        "header": header,
        "dispatch_blockers": [
            "submissions_runtime_loader_not_wired",
            "inflate_output_parity_not_proven",
            "no_exact_cuda_auth_eval",
            "lane_dispatch_claim_required_before_gpu",
        ],
        "promotion_blockers": [
            "non_score_generated_schema_packet_scaffold",
            "runtime_parity_proof_missing",
            "contest_cuda_auth_eval_missing",
        ],
    }


def _normalize_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise HNeRVGeneratedSchemaPacketError("metadata must be a mapping")
    forbidden = sorted({str(key) for key in metadata} & _FORBIDDEN_METADATA_KEYS)
    if forbidden:
        raise HNeRVGeneratedSchemaPacketError(
            "metadata contains reserved score/readiness keys: " + ", ".join(forbidden)
        )
    normalized = dict(metadata)
    try:
        _canonical_json(normalized)
    except (TypeError, ValueError) as exc:
        raise HNeRVGeneratedSchemaPacketError(
            "metadata must be deterministic JSON-serializable"
        ) from exc
    return normalized


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _coerce_section_bytes(name: str, data: bytes) -> bytes:
    if not isinstance(data, bytes):
        raise HNeRVGeneratedSchemaPacketError(f"{name} must be bytes")
    return data


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
