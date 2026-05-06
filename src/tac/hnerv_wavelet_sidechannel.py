"""Charged wavelet sidechannel candidate bytes for HNeRV payloads.

This module is the bridge from planning-only wavelet atoms to auditable archive
bytes. It preserves the source HNeRV payload byte-for-byte, appends a charged
``WR01`` atom sidechannel, and proves that the sidechannel can be decoded and
consumed by deterministic runtime code. It does not modify pixels or claim a
score; exact CUDA eval remains blocked until an inflate path applies the atoms.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import brotli

from tac.hnerv_lowlevel_packer import (
    read_strict_single_member_zip,
    sha256_bytes,
    write_stored_single_member_zip,
)
from tac.hnerv_wavelet_residual import build_wavelet_residual_plan, plan_digest

SCHEMA_VERSION = 1
TOOL = "tac.hnerv_wavelet_sidechannel.build_wavelet_sidechannel_candidate"

OUTER_MAGIC = 0xFA
OUTER_VERSION = 1
SIDECHANNEL_MAGIC = b"WR01"
SIDECHANNEL_VERSION = 1


class HnervWaveletSidechannelError(ValueError):
    """Raised when a wavelet sidechannel payload is invalid."""


@dataclasses.dataclass(frozen=True)
class ParsedWaveletSidechannelArchive:
    """Parsed ``0xfa`` candidate wrapper."""

    source_payload: bytes
    sidechannel_blob: bytes
    decoded_sidechannel: dict[str, Any]


def build_wavelet_sidechannel_candidate(
    *,
    source_archive: str | Path,
    scorecard: Mapping[str, Any],
    source_label: str,
    output_dir: str | Path,
    target_sections: Sequence[str] = ("latents_and_sidecar_brotli",),
    top_k: int = 32,
    block_size: int = 64,
    quant_step: float = 1.0,
) -> dict[str, Any]:
    """Build a deterministic candidate archive carrying charged wavelet atoms."""

    archive = read_strict_single_member_zip(source_archive)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    blockers: list[str] = []
    plan = build_wavelet_residual_plan(
        source_archive=source_archive,
        scorecard=scorecard,
        source_label=source_label,
        target_sections=target_sections,
        top_k=top_k,
        block_size=block_size,
        quant_step=quant_step,
    )
    plan["plan_sha256"] = plan_digest(plan)
    if not plan["ready_for_wavelet_candidate_build"]:
        blockers.extend(str(item) for item in plan["blockers"])
        blockers.append("wavelet_plan_not_candidate_ready")
        return _blocked_result(archive=archive, source_label=source_label, plan=plan, blockers=blockers)

    sidechannel_blob = encode_wavelet_atom_sidechannel(plan)
    decoded = decode_wavelet_atom_sidechannel(sidechannel_blob)
    proof = runtime_consumption_proof(decoded)
    if not proof["runtime_consumed"]:
        blockers.append("runtime_consumption_proof_failed")
        return _blocked_result(archive=archive, source_label=source_label, plan=plan, blockers=blockers)

    candidate_payload = build_wavelet_sidechannel_archive_bytes(
        source_payload=archive.payload,
        sidechannel_blob=sidechannel_blob,
    )
    parsed = parse_wavelet_sidechannel_archive_bytes(candidate_payload)
    if parsed.source_payload != archive.payload:
        blockers.append("source_payload_not_preserved")
    if parsed.decoded_sidechannel != decoded:
        blockers.append("sidechannel_roundtrip_mismatch")
    if blockers:
        return _blocked_result(archive=archive, source_label=source_label, plan=plan, blockers=blockers)

    candidate_archive = output_root / "hnerv_wavelet_sidechannel_candidate.zip"
    write_stored_single_member_zip(candidate_archive, member_name=archive.member_name, payload=candidate_payload)
    candidate = read_strict_single_member_zip(candidate_archive)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "score_claim": False,
        "dispatch_attempted": False,
        "source_label": str(source_label),
        "source_archive_path": str(source_archive),
        "source_archive_sha256": archive.archive_sha256,
        "source_archive_bytes": archive.archive_bytes,
        "source_member_name": archive.member_name,
        "source_payload_sha256": sha256_bytes(archive.payload),
        "source_payload_bytes": len(archive.payload),
        "candidate_archive_path": str(candidate_archive),
        "candidate_archive_sha256": candidate.archive_sha256,
        "candidate_archive_bytes": candidate.archive_bytes,
        "candidate_payload_sha256": sha256_bytes(candidate.payload),
        "candidate_payload_bytes": len(candidate.payload),
        "candidate_member_name": candidate.member_name,
        "candidate_payload_contains_source_payload": True,
        "candidate_payload_byte_delta": len(candidate.payload) - len(archive.payload),
        "candidate_archive_byte_delta": candidate.archive_bytes - archive.archive_bytes,
        "wavelet_sidechannel_sha256": sha256_bytes(sidechannel_blob),
        "wavelet_sidechannel_bytes": len(sidechannel_blob),
        "decoded_wavelet_sidechannel": decoded,
        "runtime_consumption_proof": proof,
        "plan_sha256": plan["plan_sha256"],
        "plan": plan,
        "ready_for_wavelet_sidechannel_candidate": True,
        "ready_for_archive_preflight": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": [],
        "dispatch_blockers": [
            "candidate_sidechannel_not_applied_by_inflate_runtime",
            "requires_inflate_runtime_integration",
            "requires_archive_manifest_preflight",
            "requires_exact_cuda_auth_eval",
        ],
    }
    manifest_path = output_root / "hnerv_wavelet_sidechannel_candidate.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def encode_wavelet_atom_sidechannel(plan: Mapping[str, Any]) -> bytes:
    """Return brotli-compressed ``WR01`` bytes for a wavelet atom plan."""

    raw = io.BytesIO()
    raw.write(SIDECHANNEL_MAGIC)
    raw.write(struct.pack("<H", SIDECHANNEL_VERSION))
    sections = [section for section in plan.get("sections") or [] if isinstance(section, Mapping)]
    if len(sections) > 0xFFFF:
        raise HnervWaveletSidechannelError(f"too many sections: {len(sections)}")
    raw.write(struct.pack("<H", len(sections)))
    for section in sections:
        section_name = str(section.get("section_name") or "")
        if not section_name.isascii() or not section_name:
            raise HnervWaveletSidechannelError(f"invalid section name: {section_name!r}")
        name_bytes = section_name.encode("ascii")
        if len(name_bytes) > 0xFF:
            raise HnervWaveletSidechannelError(f"section name too long: {section_name!r}")
        source_sha = str(section.get("source_section_sha256") or "")
        if len(source_sha) != 64:
            raise HnervWaveletSidechannelError(f"invalid source section sha256 for {section_name}")
        raw_bytes = int(section.get("raw_bytes") or 0)
        atoms = [atom for atom in section.get("atoms") or [] if isinstance(atom, Mapping)]
        if raw_bytes < 0:
            raise HnervWaveletSidechannelError(f"negative raw bytes for {section_name}")
        if len(atoms) > 0xFFFF:
            raise HnervWaveletSidechannelError(f"too many atoms for {section_name}: {len(atoms)}")
        raw.write(struct.pack("<B", len(name_bytes)))
        raw.write(name_bytes)
        raw.write(bytes.fromhex(source_sha))
        raw.write(struct.pack("<I", raw_bytes))
        raw.write(struct.pack("<H", len(atoms)))
        for atom in atoms:
            raw_offset = int(atom.get("raw_offset"))
            raw_end = int(atom.get("raw_end"))
            level = int(atom.get("level"))
            coefficient_index = int(atom.get("coefficient_index"))
            coefficient_quantized = int(atom.get("coefficient_quantized"))
            if raw_offset < 0 or raw_end < raw_offset or raw_end > raw_bytes:
                raise HnervWaveletSidechannelError(
                    f"invalid atom support for {section_name}: {raw_offset}:{raw_end}/{raw_bytes}"
                )
            if not 0 <= level <= 0xFF:
                raise HnervWaveletSidechannelError(f"invalid atom level: {level}")
            raw.write(struct.pack("<IIBIi", raw_offset, raw_end, level, coefficient_index, coefficient_quantized))
    return brotli.compress(raw.getvalue(), quality=11)


def decode_wavelet_atom_sidechannel(blob: bytes) -> dict[str, Any]:
    """Decode and validate a brotli-compressed ``WR01`` atom sidechannel."""

    try:
        raw = brotli.decompress(blob)
    except brotli.error as exc:
        raise HnervWaveletSidechannelError("wavelet sidechannel brotli decode failed") from exc
    reader = _Reader(raw)
    magic = reader.read_exact(4)
    if magic != SIDECHANNEL_MAGIC:
        raise HnervWaveletSidechannelError(f"invalid sidechannel magic: {magic!r}")
    version = reader.read_u16()
    if version != SIDECHANNEL_VERSION:
        raise HnervWaveletSidechannelError(f"unsupported sidechannel version: {version}")
    section_count = reader.read_u16()
    sections: list[dict[str, Any]] = []
    total_atoms = 0
    for _ in range(section_count):
        name_len = reader.read_u8()
        section_name = reader.read_exact(name_len).decode("ascii")
        source_sha = reader.read_exact(32).hex()
        raw_bytes = reader.read_u32()
        atom_count = reader.read_u16()
        atoms: list[dict[str, Any]] = []
        seen: set[tuple[int, int, int, int]] = set()
        for _atom_index in range(atom_count):
            raw_offset = reader.read_u32()
            raw_end = reader.read_u32()
            level = reader.read_u8()
            coefficient_index = reader.read_u32()
            coefficient_quantized = reader.read_i32()
            key = (raw_offset, raw_end, level, coefficient_index)
            if key in seen:
                raise HnervWaveletSidechannelError(f"duplicate atom in {section_name}: {key}")
            seen.add(key)
            if raw_offset < 0 or raw_end < raw_offset or raw_end > raw_bytes:
                raise HnervWaveletSidechannelError(
                    f"invalid decoded atom support for {section_name}: {raw_offset}:{raw_end}/{raw_bytes}"
                )
            atoms.append(
                {
                    "raw_offset": raw_offset,
                    "raw_end": raw_end,
                    "level": level,
                    "coefficient_index": coefficient_index,
                    "coefficient_quantized": coefficient_quantized,
                }
            )
        total_atoms += atom_count
        sections.append(
            {
                "section_name": section_name,
                "source_section_sha256": source_sha,
                "raw_bytes": raw_bytes,
                "atom_count": atom_count,
                "atoms": atoms,
            }
        )
    reader.assert_eof()
    return {
        "magic": SIDECHANNEL_MAGIC.decode("ascii"),
        "schema_version": SIDECHANNEL_VERSION,
        "section_count": section_count,
        "total_atom_count": total_atoms,
        "sections": sections,
    }


def runtime_consumption_proof(decoded_sidechannel: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deterministic proof that runtime code consumed decoded atoms."""

    h = hashlib.sha256()
    total_atoms = 0
    for section in decoded_sidechannel.get("sections") or []:
        if not isinstance(section, Mapping):
            continue
        h.update(str(section.get("section_name")).encode("utf-8"))
        h.update(str(section.get("source_section_sha256")).encode("utf-8"))
        h.update(str(section.get("raw_bytes")).encode("utf-8"))
        for atom in section.get("atoms") or []:
            if not isinstance(atom, Mapping):
                continue
            total_atoms += 1
            h.update(
                (
                    f"{atom.get('raw_offset')}:{atom.get('raw_end')}:"
                    f"{atom.get('level')}:{atom.get('coefficient_index')}:"
                    f"{atom.get('coefficient_quantized')};"
                ).encode()
            )
    return {
        "runtime_consumed": total_atoms > 0,
        "decoded_section_count": int(decoded_sidechannel.get("section_count") or 0),
        "decoded_atom_count": total_atoms,
        "atom_coordinate_sha256": h.hexdigest(),
        "score_claim": False,
    }


def build_wavelet_sidechannel_archive_bytes(*, source_payload: bytes, sidechannel_blob: bytes) -> bytes:
    """Wrap source payload and charged sidechannel in a deterministic outer stream."""

    if len(source_payload) >= (1 << 24):
        raise HnervWaveletSidechannelError(f"source payload too large for uint24: {len(source_payload)}")
    if len(sidechannel_blob) >= (1 << 32):
        raise HnervWaveletSidechannelError(f"sidechannel too large for uint32: {len(sidechannel_blob)}")
    return (
        bytes([OUTER_MAGIC, OUTER_VERSION])
        + len(source_payload).to_bytes(3, "little")
        + source_payload
        + struct.pack("<I", len(sidechannel_blob))
        + sidechannel_blob
    )


def parse_wavelet_sidechannel_archive_bytes(payload: bytes) -> ParsedWaveletSidechannelArchive:
    """Parse the ``0xfa`` wavelet-sidechannel archive wrapper."""

    if len(payload) < 9:
        raise HnervWaveletSidechannelError("wavelet sidechannel wrapper is truncated")
    if payload[0] != OUTER_MAGIC:
        raise HnervWaveletSidechannelError(f"invalid outer magic: 0x{payload[0]:02x}")
    if payload[1] != OUTER_VERSION:
        raise HnervWaveletSidechannelError(f"unsupported outer version: {payload[1]}")
    source_len = int.from_bytes(payload[2:5], "little")
    source_start = 5
    source_end = source_start + source_len
    if source_end + 4 > len(payload):
        raise HnervWaveletSidechannelError("wrapper truncated before sidechannel length")
    side_len = struct.unpack_from("<I", payload, source_end)[0]
    side_start = source_end + 4
    side_end = side_start + side_len
    if side_end != len(payload):
        raise HnervWaveletSidechannelError(
            f"sidechannel length mismatch: end={side_end} total={len(payload)}"
        )
    sidechannel_blob = payload[side_start:side_end]
    return ParsedWaveletSidechannelArchive(
        source_payload=payload[source_start:source_end],
        sidechannel_blob=sidechannel_blob,
        decoded_sidechannel=decode_wavelet_atom_sidechannel(sidechannel_blob),
    )


def _blocked_result(
    *,
    archive: Any,
    source_label: str,
    plan: Mapping[str, Any],
    blockers: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "score_claim": False,
        "dispatch_attempted": False,
        "source_label": str(source_label),
        "source_archive_sha256": archive.archive_sha256,
        "source_archive_bytes": archive.archive_bytes,
        "source_member_name": archive.member_name,
        "source_payload_sha256": sha256_bytes(archive.payload),
        "candidate_archive_path": None,
        "ready_for_wavelet_sidechannel_candidate": False,
        "ready_for_archive_preflight": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": list(blockers),
        "plan": dict(plan),
    }


class _Reader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def read_exact(self, n: int) -> bytes:
        end = self._pos + n
        if end > len(self._data):
            raise HnervWaveletSidechannelError("sidechannel truncated")
        out = self._data[self._pos : end]
        self._pos = end
        return out

    def read_u8(self) -> int:
        return self.read_exact(1)[0]

    def read_u16(self) -> int:
        return struct.unpack("<H", self.read_exact(2))[0]

    def read_u32(self) -> int:
        return struct.unpack("<I", self.read_exact(4))[0]

    def read_i32(self) -> int:
        return struct.unpack("<i", self.read_exact(4))[0]

    def assert_eof(self) -> None:
        if self._pos != len(self._data):
            raise HnervWaveletSidechannelError(
                f"sidechannel trailing bytes: pos={self._pos} total={len(self._data)}"
            )


__all__ = [
    "OUTER_MAGIC",
    "OUTER_VERSION",
    "SIDECHANNEL_MAGIC",
    "SIDECHANNEL_VERSION",
    "HnervWaveletSidechannelError",
    "build_wavelet_sidechannel_archive_bytes",
    "build_wavelet_sidechannel_candidate",
    "decode_wavelet_atom_sidechannel",
    "encode_wavelet_atom_sidechannel",
    "parse_wavelet_sidechannel_archive_bytes",
    "runtime_consumption_proof",
]
