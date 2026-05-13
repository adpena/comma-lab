"""DPW1 charged archive grammar for a driving-prior world-model scaffold.

The monolithic ``0.bin`` placeholder grammar is:

    MAGIC(4)                  b"DPW1"
    VERSION(1)                u8, currently 1
    OUTPUT_HEIGHT(2)          u16 big-endian
    OUTPUT_WIDTH(2)           u16 big-endian
    NUM_PAIRS(2)              u16 big-endian
    CODEBOOK_ENTRIES(2)       u16 big-endian
    RESIDUAL_GRID_HEIGHT(2)   u16 big-endian
    RESIDUAL_GRID_WIDTH(2)    u16 big-endian
    PRIOR_WEIGHT_LEN(4)       u32 big-endian, charged uint8 RGB codebook bytes
    RESIDUAL_LEN(4)           u32 big-endian, charged int8 residual-grid bytes
    METADATA_LEN(4)           u32 big-endian, canonical UTF-8 JSON
    PRIOR_WEIGHT_BYTES        codebook_entries * 3 bytes
    RESIDUAL_BYTES            num_pairs * 2 * grid_h * grid_w * 3 bytes
    METADATA_BYTES            proxy-safe custody metadata

The parser consumes every byte and rejects trailing data. Metadata is fixed
false for score/dispatch authority because this is not training or exact eval.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .config import DEFAULT_LANE_ID, DrivingPriorWorldModelConfig

DPWM_MAGIC = b"DPW1"
DPWM_SCHEMA_VERSION = 1
DPWM_PROXY_EVIDENCE_GRADE = "proxy_only_driving_prior_world_model_scaffold"

_HEADER_STRUCT = struct.Struct(">4sBHHHHHHIII")
DPWM_HEADER_BYTES = _HEADER_STRUCT.size


class DrivingPriorWorldModelError(ValueError):
    """Raised when a DPW1 archive or apply request is invalid."""


@dataclass(frozen=True)
class ArchiveSection:
    """Byte location and hash for a charged archive section."""

    name: str
    offset: int
    length: int
    sha256: str

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "offset": self.offset,
            "length": self.length,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class DrivingPriorWorldModelArchive:
    """Parsed DPW1 archive and fixed-offset section manifest."""

    config: DrivingPriorWorldModelConfig
    prior_weights: bytes
    residual_bytes: bytes
    metadata: dict[str, Any]
    sections: tuple[ArchiveSection, ...]
    total_bytes: int
    schema_version: int = DPWM_SCHEMA_VERSION

    @property
    def structural_noop(self) -> bool:
        """True only when the charged prior and residual cannot change output."""

        return not any(self.prior_weights) and not any(self.residual_bytes)

    def section_manifest(self) -> list[dict[str, object]]:
        return [section.to_json() for section in self.sections]


_FORCED_METADATA: dict[str, object] = {
    "schema_version": DPWM_SCHEMA_VERSION,
    "format_family": "dpw1_driving_prior_world_model",
    "substrate": "driving_prior_world_model",
    "lane_id": DEFAULT_LANE_ID,
    "evidence_grade": DPWM_PROXY_EVIDENCE_GRADE,
    "research_only": True,
    "proxy": True,
    "proxy_only": True,
    "score_claim": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "scorer_at_inflate": False,
    "dispatch_authority": "none",
    "prior_dtype": "uint8_rgb_codebook",
    "residual_dtype": "int8_twos_complement_residual_grid",
    "archive_grammar": "monolithic_0_bin_dpw1_fixed_offsets",
}

_FIXED_AUTHORITY_FIELDS: dict[str, object] = {
    "research_only": True,
    "proxy": True,
    "proxy_only": True,
    "score_claim": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "scorer_at_inflate": False,
    "dispatch_authority": "none",
}


def expected_prior_weight_bytes(config: DrivingPriorWorldModelConfig) -> int:
    return int(config.codebook_entries) * 3


def expected_residual_bytes(config: DrivingPriorWorldModelConfig) -> int:
    return (
        int(config.num_pairs)
        * 2
        * int(config.residual_grid_height)
        * int(config.residual_grid_width)
        * 3
    )


def deterministic_prior_weights(
    config: DrivingPriorWorldModelConfig,
    *,
    seed: int = 2032,
) -> bytes:
    """Return deterministic non-trained RGB codebook bytes."""

    out = bytearray()
    for idx in range(expected_prior_weight_bytes(config)):
        digest = hashlib.sha256(f"dpw1:prior:{seed}:{idx}".encode("ascii")).digest()
        out.append(digest[0])
    return bytes(out)


def deterministic_residual_bytes(
    config: DrivingPriorWorldModelConfig,
    *,
    seed: int = 2032,
    max_abs_delta: int = 3,
) -> bytes:
    """Return deterministic signed-int8 residual-grid bytes."""

    if max_abs_delta < 0 or max_abs_delta > 127:
        raise ValueError("max_abs_delta must be in [0, 127]")
    if max_abs_delta == 0:
        return bytes(expected_residual_bytes(config))
    width = 2 * max_abs_delta + 1
    out = bytearray()
    for idx in range(expected_residual_bytes(config)):
        digest = hashlib.sha256(f"dpw1:residual:{seed}:{idx}".encode("ascii")).digest()
        signed = int(digest[0] % width) - max_abs_delta
        out.append(signed & 0xFF)
    return bytes(out)


def pack_archive(
    config: DrivingPriorWorldModelConfig,
    prior_weights: bytes | None = None,
    residual_bytes: bytes | None = None,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> bytes:
    """Serialize charged prior weights and residual bytes into DPW1 bytes."""

    prior = bytes(
        prior_weights if prior_weights is not None else deterministic_prior_weights(config)
    )
    residual = bytes(
        residual_bytes
        if residual_bytes is not None
        else deterministic_residual_bytes(config)
    )
    _validate_section_lengths(config, prior, residual)
    meta_bytes = _canonical_metadata_bytes(config, prior, residual, metadata)
    header = _HEADER_STRUCT.pack(
        DPWM_MAGIC,
        DPWM_SCHEMA_VERSION,
        int(config.output_height),
        int(config.output_width),
        int(config.num_pairs),
        int(config.codebook_entries),
        int(config.residual_grid_height),
        int(config.residual_grid_width),
        len(prior),
        len(residual),
        len(meta_bytes),
    )
    return header + prior + residual + meta_bytes


def parse_archive(blob: bytes) -> DrivingPriorWorldModelArchive:
    """Parse DPW1 bytes and reject malformed/trailing data."""

    data = bytes(blob)
    if len(data) < DPWM_HEADER_BYTES:
        raise DrivingPriorWorldModelError(
            f"archive too short ({len(data)} bytes; need >= {DPWM_HEADER_BYTES})"
        )
    (
        magic,
        version,
        output_height,
        output_width,
        num_pairs,
        codebook_entries,
        residual_grid_height,
        residual_grid_width,
        prior_len,
        residual_len,
        metadata_len,
    ) = _HEADER_STRUCT.unpack(data[:DPWM_HEADER_BYTES])
    if magic != DPWM_MAGIC:
        raise DrivingPriorWorldModelError(f"bad magic {magic!r}; expected {DPWM_MAGIC!r}")
    if version != DPWM_SCHEMA_VERSION:
        raise DrivingPriorWorldModelError(f"unsupported schema version {version}")

    config = DrivingPriorWorldModelConfig(
        output_height=int(output_height),
        output_width=int(output_width),
        num_pairs=int(num_pairs),
        codebook_entries=int(codebook_entries),
        residual_grid_height=int(residual_grid_height),
        residual_grid_width=int(residual_grid_width),
    )
    if prior_len != expected_prior_weight_bytes(config):
        raise DrivingPriorWorldModelError(
            f"prior weight byte count {prior_len} != expected {expected_prior_weight_bytes(config)}"
        )
    if residual_len != expected_residual_bytes(config):
        raise DrivingPriorWorldModelError(
            f"residual byte count {residual_len} != expected {expected_residual_bytes(config)}"
        )

    offset = DPWM_HEADER_BYTES
    prior_offset = offset
    prior = data[offset : offset + prior_len]
    offset += prior_len
    residual_offset = offset
    residual = data[offset : offset + residual_len]
    offset += residual_len
    metadata_offset = offset
    metadata_blob = data[offset : offset + metadata_len]
    offset += metadata_len
    if offset != len(data):
        raise DrivingPriorWorldModelError("trailing bytes after metadata")
    if len(metadata_blob) != metadata_len:
        raise DrivingPriorWorldModelError("truncated metadata blob")
    try:
        metadata_obj = json.loads(metadata_blob.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise DrivingPriorWorldModelError("invalid metadata JSON") from exc
    if not isinstance(metadata_obj, dict):
        raise DrivingPriorWorldModelError("metadata must decode to a JSON object")
    metadata = dict(metadata_obj)
    _validate_proxy_safe_metadata(metadata)
    lane_id = str(metadata.get("lane_id", DEFAULT_LANE_ID))
    config = DrivingPriorWorldModelConfig(
        output_height=int(output_height),
        output_width=int(output_width),
        num_pairs=int(num_pairs),
        codebook_entries=int(codebook_entries),
        residual_grid_height=int(residual_grid_height),
        residual_grid_width=int(residual_grid_width),
        lane_id=lane_id,
    )
    sections = (
        _section("prior_weights", prior_offset, prior),
        _section("residual_bytes", residual_offset, residual),
        _section("metadata", metadata_offset, metadata_blob),
    )
    return DrivingPriorWorldModelArchive(
        config=config,
        prior_weights=prior,
        residual_bytes=residual,
        metadata=metadata,
        sections=sections,
        total_bytes=len(data),
        schema_version=int(version),
    )


def build_readiness_manifest(
    archive: DrivingPriorWorldModelArchive | bytes,
) -> dict[str, object]:
    """Return proxy-safe readiness metadata for this scaffold."""

    parsed = parse_archive(archive) if isinstance(archive, (bytes, bytearray)) else archive
    blockers = [
        "research_only_scaffold_not_trained",
        "requires_byte_closed_2032_trained_prior",
        "requires_exact_cuda_auth_eval_before_dispatch",
    ]
    if parsed.structural_noop:
        blockers.insert(0, "driving_prior_world_model_structural_noop")
    return {
        "schema_version": DPWM_SCHEMA_VERSION,
        "lane_id": parsed.config.lane_id,
        "substrate": "driving_prior_world_model",
        "archive_bytes": parsed.total_bytes,
        "prior_weights_charged_bytes": len(parsed.prior_weights),
        "residual_charged_bytes": len(parsed.residual_bytes),
        "sections": parsed.section_manifest(),
        "structural_noop": parsed.structural_noop,
        "research_only": True,
        "proxy_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "scorer_at_inflate": False,
        "dispatch_blockers": blockers,
    }


def _validate_section_lengths(
    config: DrivingPriorWorldModelConfig,
    prior_weights: bytes,
    residual_bytes: bytes,
) -> None:
    prior_expected = expected_prior_weight_bytes(config)
    residual_expected = expected_residual_bytes(config)
    if len(prior_weights) != prior_expected:
        raise DrivingPriorWorldModelError(
            f"prior_weights must be {prior_expected} bytes; got {len(prior_weights)}"
        )
    if len(residual_bytes) != residual_expected:
        raise DrivingPriorWorldModelError(
            f"residual_bytes must be {residual_expected} bytes; got {len(residual_bytes)}"
        )


def _canonical_metadata_bytes(
    config: DrivingPriorWorldModelConfig,
    prior_weights: bytes,
    residual_bytes: bytes,
    metadata: Mapping[str, Any] | None,
) -> bytes:
    out: dict[str, Any] = dict(_FORCED_METADATA)
    if metadata is not None:
        for key, expected in _FIXED_AUTHORITY_FIELDS.items():
            if key in metadata and metadata[key] != expected:
                raise DrivingPriorWorldModelError(
                    f"metadata field {key} must be {expected!r}"
                )
        out.update(dict(metadata))
    out.update(
        {
            **_FORCED_METADATA,
            "lane_id": config.lane_id,
            "output_height": config.output_height,
            "output_width": config.output_width,
            "num_pairs": config.num_pairs,
            "codebook_entries": config.codebook_entries,
            "residual_grid_height": config.residual_grid_height,
            "residual_grid_width": config.residual_grid_width,
            "prior_weights_charged_bytes": len(prior_weights),
            "residual_charged_bytes": len(residual_bytes),
            "prior_weights_sha256": _sha256(prior_weights),
            "residual_sha256": _sha256(residual_bytes),
            "score_claim_blockers": [
                "proxy_only_scaffold",
                "not_trained",
                "not_exact_evaled",
            ],
        }
    )
    _validate_proxy_safe_metadata(out)
    return json.dumps(out, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _validate_proxy_safe_metadata(metadata: Mapping[str, Any]) -> None:
    for key, expected in _FIXED_AUTHORITY_FIELDS.items():
        if metadata.get(key) != expected:
            raise DrivingPriorWorldModelError(
                f"metadata {key} must be {expected!r}"
            )


def _section(name: str, offset: int, payload: bytes) -> ArchiveSection:
    return ArchiveSection(name=name, offset=offset, length=len(payload), sha256=_sha256(payload))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
