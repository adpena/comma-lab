"""Coord-MLP residual sidecar archive contract.

This module is the lightweight H15 readiness surface: it defines charged
Coord-MLP residual sidecar bytes, parses them into typed sections, and emits a
proxy-safe readiness manifest. It does not load scorers and does not claim
score authority.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from tac.optimization.proxy_candidate_contract import (
    PROXY_FALSE_AUTHORITY_FIELDS,
    apply_proxy_evidence_boundary,
    proxy_authority_fields,
)

SIDECAR_MAGIC: Final[bytes] = b"CMLR"
SIDECAR_VERSION: Final[int] = 1
SIDECAR_SCHEMA: Final[str] = "coord_mlp_residual_sidecar_v1"
READINESS_SCHEMA: Final[str] = "coord_mlp_residual_sidecar_readiness_v1"
H15_LANE_ID: Final[str] = "h15_coord_mlp_residual_sidecar"

COORD_DIM: Final[int] = 3
RGB_CHANNELS: Final[int] = 3
HEADER_STRUCT: Final[struct.Struct] = struct.Struct("<4sBBHII")
PATCH_STRUCT: Final[struct.Struct] = struct.Struct("<HHHHH")
HEADER_BYTES: Final[int] = HEADER_STRUCT.size
PATCH_BYTES: Final[int] = PATCH_STRUCT.size


class CoordMlpResidualSidecarError(ValueError):
    """Raised when a Coord-MLP residual sidecar violates its byte contract."""


@dataclass(frozen=True)
class CoordMlpPatch:
    """A rectangular RGB residual application region."""

    frame_index: int
    y: int
    x: int
    height: int
    width: int

    def assert_wire_safe(self) -> None:
        for name, value in (
            ("frame_index", self.frame_index),
            ("y", self.y),
            ("x", self.x),
            ("height", self.height),
            ("width", self.width),
        ):
            if int(value) != value or value < 0 or value > 0xFFFF:
                raise CoordMlpResidualSidecarError(
                    f"{name}={value!r} is outside uint16 wire range"
                )
        if self.height == 0 or self.width == 0:
            raise CoordMlpResidualSidecarError("patch height/width must be positive")

    def to_bytes(self) -> bytes:
        self.assert_wire_safe()
        return PATCH_STRUCT.pack(
            int(self.frame_index),
            int(self.y),
            int(self.x),
            int(self.height),
            int(self.width),
        )

    @classmethod
    def from_bytes(cls, blob: bytes, offset: int) -> CoordMlpPatch:
        values = PATCH_STRUCT.unpack_from(blob, offset)
        patch = cls(*map(int, values))
        patch.assert_wire_safe()
        return patch


def _readonly_array(value: Any, *, dtype: np.dtype, shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.shape != shape:
        raise CoordMlpResidualSidecarError(
            f"{name} shape {arr.shape} does not match expected {shape}"
        )
    if not np.all(np.isfinite(arr.astype(np.float64, copy=False))):
        raise CoordMlpResidualSidecarError(f"{name} contains non-finite values")
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        if np.any(arr < info.min) or np.any(arr > info.max):
            raise CoordMlpResidualSidecarError(
                f"{name} has values outside {dtype} range"
            )
    out = arr.astype(dtype, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class CoordMlpResidualWeights:
    """Fixed-point one-hidden-layer Coord-MLP residual weights."""

    w1_int8: Any
    b1_int16: Any
    w2_int8: Any
    b2_int16: Any

    def __post_init__(self) -> None:
        w1 = np.asarray(self.w1_int8)
        if w1.ndim != 2 or w1.shape[1] != COORD_DIM:
            raise CoordMlpResidualSidecarError(
                f"w1_int8 must have shape (hidden_dim, {COORD_DIM}); got {w1.shape}"
            )
        hidden_dim = int(w1.shape[0])
        if hidden_dim <= 0 or hidden_dim > 0xFF:
            raise CoordMlpResidualSidecarError(
                f"hidden_dim={hidden_dim} outside uint8 wire range"
            )
        object.__setattr__(
            self,
            "w1_int8",
            _readonly_array(
                w1,
                dtype=np.dtype("int8"),
                shape=(hidden_dim, COORD_DIM),
                name="w1_int8",
            ),
        )
        object.__setattr__(
            self,
            "b1_int16",
            _readonly_array(
                self.b1_int16,
                dtype=np.dtype("<i2"),
                shape=(hidden_dim,),
                name="b1_int16",
            ),
        )
        object.__setattr__(
            self,
            "w2_int8",
            _readonly_array(
                self.w2_int8,
                dtype=np.dtype("int8"),
                shape=(RGB_CHANNELS, hidden_dim),
                name="w2_int8",
            ),
        )
        object.__setattr__(
            self,
            "b2_int16",
            _readonly_array(
                self.b2_int16,
                dtype=np.dtype("<i2"),
                shape=(RGB_CHANNELS,),
                name="b2_int16",
            ),
        )

    @property
    def hidden_dim(self) -> int:
        return int(self.w1_int8.shape[0])

    @property
    def byte_len(self) -> int:
        return int(8 * self.hidden_dim + 6)

    @property
    def structurally_nonzero(self) -> bool:
        if np.any(self.b2_int16):
            return True
        return bool(np.any(self.w2_int8) and (np.any(self.w1_int8) or np.any(self.b1_int16)))

    def to_bytes(self) -> bytes:
        return b"".join(
            (
                self.w1_int8.astype("int8", copy=False).tobytes(order="C"),
                self.b1_int16.astype("<i2", copy=False).tobytes(order="C"),
                self.w2_int8.astype("int8", copy=False).tobytes(order="C"),
                self.b2_int16.astype("<i2", copy=False).tobytes(order="C"),
            )
        )

    @classmethod
    def from_bytes(cls, *, hidden_dim: int, blob: bytes) -> CoordMlpResidualWeights:
        expected = 8 * int(hidden_dim) + 6
        if len(blob) != expected:
            raise CoordMlpResidualSidecarError(
                f"weight blob length {len(blob)} does not match expected {expected}"
            )
        offset = 0
        w1_count = hidden_dim * COORD_DIM
        w1 = np.frombuffer(blob[offset : offset + w1_count], dtype=np.int8).reshape(
            hidden_dim, COORD_DIM
        )
        offset += w1_count
        b1_bytes = hidden_dim * 2
        b1 = np.frombuffer(blob[offset : offset + b1_bytes], dtype="<i2")
        offset += b1_bytes
        w2_count = RGB_CHANNELS * hidden_dim
        w2 = np.frombuffer(blob[offset : offset + w2_count], dtype=np.int8).reshape(
            RGB_CHANNELS, hidden_dim
        )
        offset += w2_count
        b2 = np.frombuffer(blob[offset : offset + 6], dtype="<i2")
        return cls(w1, b1, w2, b2)


@dataclass(frozen=True)
class CoordMlpSection:
    """A charged byte section inside a Coord-MLP residual sidecar."""

    name: str
    offset: int
    size: int
    sha256: str
    charged: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "offset": self.offset,
            "size": self.size,
            "sha256": self.sha256,
            "charged": self.charged,
        }


@dataclass(frozen=True)
class ParsedCoordMlpResidualSidecar:
    """Typed parse result for one Coord-MLP residual sidecar."""

    hidden_dim: int
    patches: tuple[CoordMlpPatch, ...]
    weights: CoordMlpResidualWeights
    metadata: Mapping[str, Any]
    sections: tuple[CoordMlpSection, ...]
    raw_bytes: bytes = field(repr=False)
    score_claim: bool = field(default=False, init=False)
    promotion_eligible: bool = field(default=False, init=False)
    ready_for_exact_eval_dispatch: bool = field(default=False, init=False)

    @property
    def sidecar_sha256(self) -> str:
        return hashlib.sha256(self.raw_bytes).hexdigest()

    @property
    def charged_bytes(self) -> int:
        return sum(section.size for section in self.sections if section.charged)

    @property
    def structural_noop(self) -> bool:
        return len(self.patches) == 0 or not self.weights.structurally_nonzero

    def to_bytes(self) -> bytes:
        return bytes(self.raw_bytes)

    def section_manifest(self) -> list[dict[str, Any]]:
        return [section.to_dict() for section in self.sections]

    def assert_invariants(self) -> None:
        if self.hidden_dim <= 0 or self.hidden_dim > 0xFF:
            raise CoordMlpResidualSidecarError(f"bad hidden_dim={self.hidden_dim}")
        if self.weights.hidden_dim != self.hidden_dim:
            raise CoordMlpResidualSidecarError("weights hidden_dim mismatch")
        if self.charged_bytes != len(self.raw_bytes):
            raise CoordMlpResidualSidecarError("all sidecar bytes must be charged")
        if self.score_claim or self.promotion_eligible or self.ready_for_exact_eval_dispatch:
            raise CoordMlpResidualSidecarError("sidecar parse result cannot claim authority")
        _validate_false_authority_metadata(self.metadata)


def pack_sidecar(
    patches: Sequence[CoordMlpPatch],
    weights: CoordMlpResidualWeights,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> bytes:
    """Pack a deterministic charged Coord-MLP residual sidecar."""

    patch_tuple = tuple(patches)
    if len(patch_tuple) > 0xFFFF:
        raise CoordMlpResidualSidecarError("too many patches for uint16 count")
    for patch in patch_tuple:
        patch.assert_wire_safe()

    meta_bytes = _canonical_metadata_bytes(metadata)
    weight_bytes = weights.to_bytes()
    header = HEADER_STRUCT.pack(
        SIDECAR_MAGIC,
        SIDECAR_VERSION,
        weights.hidden_dim,
        len(patch_tuple),
        len(weight_bytes),
        len(meta_bytes),
    )
    patch_bytes = b"".join(patch.to_bytes() for patch in patch_tuple)
    blob = header + patch_bytes + weight_bytes + meta_bytes
    parsed = parse_sidecar(blob)
    parsed.assert_invariants()
    return blob


def parse_sidecar(blob: bytes) -> ParsedCoordMlpResidualSidecar:
    """Parse and validate a Coord-MLP residual sidecar."""

    if not isinstance(blob, (bytes, bytearray)):
        raise CoordMlpResidualSidecarError(
            f"sidecar blob must be bytes-like, got {type(blob).__name__}"
        )
    raw = bytes(blob)
    if len(raw) < HEADER_BYTES:
        raise CoordMlpResidualSidecarError("sidecar too short for header")
    magic, version, hidden_dim, patch_count, weight_len, meta_len = HEADER_STRUCT.unpack_from(
        raw, 0
    )
    if magic != SIDECAR_MAGIC:
        raise CoordMlpResidualSidecarError(f"bad sidecar magic {magic!r}")
    if version != SIDECAR_VERSION:
        raise CoordMlpResidualSidecarError(f"unsupported sidecar version {version}")
    if hidden_dim <= 0:
        raise CoordMlpResidualSidecarError("hidden_dim must be positive")

    offset = HEADER_BYTES
    patch_table_len = int(patch_count) * PATCH_BYTES
    patch_end = offset + patch_table_len
    if patch_end > len(raw):
        raise CoordMlpResidualSidecarError("truncated patch table")
    patches = tuple(
        CoordMlpPatch.from_bytes(raw, offset + index * PATCH_BYTES)
        for index in range(int(patch_count))
    )
    offset = patch_end

    weight_end = offset + int(weight_len)
    if weight_end > len(raw):
        raise CoordMlpResidualSidecarError("truncated weight blob")
    weight_bytes = raw[offset:weight_end]
    weights = CoordMlpResidualWeights.from_bytes(
        hidden_dim=int(hidden_dim), blob=weight_bytes
    )
    offset = weight_end

    meta_end = offset + int(meta_len)
    if meta_end > len(raw):
        raise CoordMlpResidualSidecarError("truncated metadata blob")
    meta_bytes = raw[offset:meta_end]
    offset = meta_end
    if offset != len(raw):
        raise CoordMlpResidualSidecarError("trailing bytes after metadata")
    metadata = json.loads(meta_bytes.decode("utf-8"))
    if not isinstance(metadata, dict):
        raise CoordMlpResidualSidecarError("metadata must decode to a JSON object")
    _validate_false_authority_metadata(metadata)

    sections = (
        _section("HEADER", raw, 0, HEADER_BYTES),
        _section("PATCH_TABLE", raw, HEADER_BYTES, patch_table_len),
        _section("WEIGHT_BLOB", raw, HEADER_BYTES + patch_table_len, weight_len),
        _section("META_JSON", raw, HEADER_BYTES + patch_table_len + weight_len, meta_len),
    )
    parsed = ParsedCoordMlpResidualSidecar(
        hidden_dim=int(hidden_dim),
        patches=patches,
        weights=weights,
        metadata=metadata,
        sections=sections,
        raw_bytes=raw,
    )
    parsed.assert_invariants()
    return parsed


def build_readiness_manifest(
    sidecar_blob: bytes,
    *,
    archive_sha256: str | None = None,
    runtime_tree_sha256: str | None = None,
) -> dict[str, Any]:
    """Return proxy-safe readiness metadata for an H15 sidecar candidate."""

    parsed = parse_sidecar(sidecar_blob)
    blockers: list[str] = []
    if parsed.structural_noop:
        blockers.append("coord_mlp_residual_sidecar_structural_noop")
    if not _is_sha256(archive_sha256):
        blockers.append("archive_custody_missing")
    if not _is_sha256(runtime_tree_sha256):
        blockers.append("runtime_tree_custody_missing")
    blockers.append("coord_mlp_sidecar_requires_byte_closed_exact_eval_adjudication")

    row: dict[str, Any] = {
        "schema": READINESS_SCHEMA,
        "sidecar_schema": SIDECAR_SCHEMA,
        "lane_id": H15_LANE_ID,
        "source_recommendation": "H15",
        "charged_section": True,
        "charged_bytes": parsed.charged_bytes,
        "section_manifest": parsed.section_manifest(),
        "sidecar_sha256": parsed.sidecar_sha256,
        "structural_noop": parsed.structural_noop,
        "archive_sha256": archive_sha256,
        "runtime_tree_sha256": runtime_tree_sha256,
        "scorer_at_inflate": False,
    }
    return apply_proxy_evidence_boundary(row, dispatch_blockers=blockers)


def _section(name: str, blob: bytes, offset: int, size: int) -> CoordMlpSection:
    data = blob[offset : offset + int(size)]
    return CoordMlpSection(
        name=name,
        offset=int(offset),
        size=int(size),
        sha256=hashlib.sha256(data).hexdigest(),
        charged=True,
    )


def _canonical_metadata_bytes(metadata: Mapping[str, Any] | None) -> bytes:
    out: dict[str, Any] = {
        "schema": SIDECAR_SCHEMA,
        "lane_id": H15_LANE_ID,
        "family": "coord_mlp_residual_sidecar",
        "charged_section": True,
        "scorer_at_inflate": False,
    }
    out.update(proxy_authority_fields())
    if metadata:
        out.update(dict(metadata))
    _validate_false_authority_metadata(out)
    out["schema"] = SIDECAR_SCHEMA
    out["lane_id"] = H15_LANE_ID
    out["charged_section"] = True
    out["scorer_at_inflate"] = False
    out.update(proxy_authority_fields())
    return json.dumps(out, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _validate_false_authority_metadata(metadata: Mapping[str, Any]) -> None:
    for key, expected in PROXY_FALSE_AUTHORITY_FIELDS.items():
        if metadata.get(key, expected) is not expected:
            raise CoordMlpResidualSidecarError(
                f"metadata field {key} must be {expected!r}"
            )
    if metadata.get("score_claim") is not False:
        raise CoordMlpResidualSidecarError("metadata score_claim must be false")
    if metadata.get("ready_for_exact_eval_dispatch") is not False:
        raise CoordMlpResidualSidecarError(
            "metadata ready_for_exact_eval_dispatch must be false"
        )


def _is_sha256(value: str | None) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(ch in "0123456789abcdef" for ch in value)


__all__ = [
    "H15_LANE_ID",
    "READINESS_SCHEMA",
    "SIDECAR_MAGIC",
    "SIDECAR_SCHEMA",
    "SIDECAR_VERSION",
    "CoordMlpPatch",
    "CoordMlpResidualSidecarError",
    "CoordMlpResidualWeights",
    "CoordMlpSection",
    "ParsedCoordMlpResidualSidecar",
    "build_readiness_manifest",
    "pack_sidecar",
    "parse_sidecar",
]
