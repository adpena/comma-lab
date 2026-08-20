# SPDX-License-Identifier: MIT
"""Predictor-preserving conditional camera preimage grammar (A3).

This module is the counted ``A`` receiver for a coupled chronological pair.
It consumes the exact ephemeral predictor frames ``P0/P1`` and the exact
predictor-preserving ``G`` overlay for corrected ``Y1``.  It emits ``Y0/Y1``
without ever serializing either dense predictor frame.

The wire universe is deliberately finite and versioned:

* ``PASS_P0_V1``: exact P0 pass-through, with no control rows or mutation body;
* ``SPARSE_CONSTANT_RGB_V1``: replace disjoint camera-lattice supports with a
  canonical RGB value, thereby realizing that exact scorer numerator; and
* ``COPY_CORRECTED_Y1_SUPPORT_V1``: copy selected disjoint supports from the
  already-bound corrected Y1 onto P0.  This is the only defensible
  corrected-Y1-derived form that still preserves P0 everywhere A does not own.

Every packet binds P, G, P0, predictor P1, and corrected Y1.  The receiver
proves exact byte preservation outside A ownership and exact scorer-numerator
preservation outside the corresponding scorer cells.  It invokes no scorer and
makes no score, candidate, originality, or promotion claim.  Selecting useful
rows/values remains an encoder-side inverse problem; this file only makes the
finite receiver primitive real and byte-closed.
"""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from dataclasses import asdict, dataclass
from enum import StrEnum
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np

from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator, Uint8LatticeError
from tac.witness_dsl.ep725_levelset_predictor_adapter import Ep725EphemeralRuntimeSurfaceV2
from tac.witness_dsl.generative_taskspace_correction import DecodedGenerativeCorrectionV1
from tac.witness_dsl.predictor_preserving_taskspace_overlay import (
    PredictorPreservingOverlayResultV1,
    parse_predictor_preserving_overlay_receipt,
)
from tac.witness_dsl.taskspace_predictor_state_v2 import TaskspacePredictorStateV2

if TYPE_CHECKING:
    from tac.optimization.direct_description_carrier_compose import ReceiverRealizationProfileV1

CAMERA_HEIGHT: Final = 874
CAMERA_WIDTH: Final = 1164
SCORER_HEIGHT: Final = 384
SCORER_WIDTH: Final = 512
CHANNELS: Final = 3
MAX_BOUNDED_PAIRS: Final = 4

PACKET_MAGIC: Final = b"TACA3P1\x00"
PACKET_VERSION: Final = 1
PACKET_HEADER: Final = struct.Struct(">8sBBHH32sIIII")
RGB_ROW: Final = struct.Struct(">HHHBBB")
COPY_ROW: Final = struct.Struct(">HHH")
RECEIPT_SCHEMA: Final = "tac.predictor_preserving_coupled_preimage_receipt.v1"
SOURCE_SCHEMA: Final = "tac.predictor_preserving_coupled_preimage_source.v1"
SURFACE_SCHEMA: Final = "tac.predictor_camera_pair_surface.v1"


class PredictorPreservingA3Error(ValueError):
    """The A3 packet, source binding, or exact preservation proof failed."""


class PredictorPreservingA3Mode(StrEnum):
    """Closed A3 wire universe; each value has a distinct receiver action."""

    PASS_P0_V1 = "PASS_P0_V1"
    SPARSE_CONSTANT_RGB_V1 = "SPARSE_CONSTANT_RGB_V1"
    COPY_CORRECTED_Y1_SUPPORT_V1 = "COPY_CORRECTED_Y1_SUPPORT_V1"


_MODE_TO_WIRE: Final = {
    PredictorPreservingA3Mode.PASS_P0_V1: 0,
    PredictorPreservingA3Mode.SPARSE_CONSTANT_RGB_V1: 1,
    PredictorPreservingA3Mode.COPY_CORRECTED_Y1_SUPPORT_V1: 2,
}
_WIRE_TO_MODE: Final = {value: key for key, value in _MODE_TO_WIRE.items()}


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return _sha256(memoryview(contiguous).cast("B"))


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise PredictorPreservingA3Error("A3 record is not finite canonical ASCII JSON") from exc


def _require_sha256(value: object, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PredictorPreservingA3Error(f"{field} must be canonical lowercase SHA-256")
    return value


def _require_exact_int(value: object, field: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise PredictorPreservingA3Error(f"{field} must be an exact integer in [{minimum},{maximum}]")
    return value


def _immutable_array(value: np.ndarray, *, dtype: np.dtype[Any]) -> np.ndarray:
    copied = np.ascontiguousarray(value, dtype=dtype).copy()
    copied.setflags(write=False)
    return copied


@lru_cache(maxsize=1)
def _operator() -> DisjointResizeOperator:
    try:
        return DisjointResizeOperator.build(
            camera_h=CAMERA_HEIGHT,
            camera_w=CAMERA_WIDTH,
            scorer_h=SCORER_HEIGHT,
            scorer_w=SCORER_WIDTH,
        )
    except Uint8LatticeError as exc:
        raise PredictorPreservingA3Error("frozen camera/scorer geometry is not disjoint") from exc


def _profile_sha256(profile: ReceiverRealizationProfileV1) -> str:
    return _sha256(
        _canonical_json(
            {
                "amplitude_u8": profile.amplitude_u8,
                "coverage_radius": profile.coverage_radius,
                "role_rgb_u8": [list(row) for row in profile.role_rgb_u8],
            }
        )
    )


@dataclass(frozen=True, slots=True, order=True)
class SparseConstantRGBCellV1:
    """One globally addressed scorer cell and its exact constant RGB target."""

    source_pair_id: int
    scorer_row: int
    scorer_col: int
    rgb_u8: tuple[int, int, int]

    def __post_init__(self) -> None:
        _require_exact_int(self.source_pair_id, "source_pair_id", minimum=0, maximum=599)
        _require_exact_int(self.scorer_row, "scorer_row", minimum=0, maximum=SCORER_HEIGHT - 1)
        _require_exact_int(self.scorer_col, "scorer_col", minimum=0, maximum=SCORER_WIDTH - 1)
        if type(self.rgb_u8) is not tuple or len(self.rgb_u8) != CHANNELS:
            raise PredictorPreservingA3Error("rgb_u8 must be one exact RGB tuple")
        for channel, value in enumerate(self.rgb_u8):
            _require_exact_int(value, f"rgb_u8[{channel}]", minimum=0, maximum=255)

    @property
    def address(self) -> tuple[int, int, int]:
        return (self.source_pair_id, self.scorer_row, self.scorer_col)


@dataclass(frozen=True, slots=True, order=True)
class CorrectedY1SupportCopyCellV1:
    """One scorer cell whose exact camera support is copied from corrected Y1."""

    source_pair_id: int
    scorer_row: int
    scorer_col: int

    def __post_init__(self) -> None:
        _require_exact_int(self.source_pair_id, "source_pair_id", minimum=0, maximum=599)
        _require_exact_int(self.scorer_row, "scorer_row", minimum=0, maximum=SCORER_HEIGHT - 1)
        _require_exact_int(self.scorer_col, "scorer_col", minimum=0, maximum=SCORER_WIDTH - 1)

    @property
    def address(self) -> tuple[int, int, int]:
        return (self.source_pair_id, self.scorer_row, self.scorer_col)


@dataclass(frozen=True, slots=True)
class PredictorPreservingA3ProgramV1:
    """A canonical finite A3 control program."""

    mode: PredictorPreservingA3Mode
    constant_rgb_cells: tuple[SparseConstantRGBCellV1, ...] = ()
    corrected_y1_copy_cells: tuple[CorrectedY1SupportCopyCellV1, ...] = ()

    def __post_init__(self) -> None:
        if type(self.mode) is not PredictorPreservingA3Mode:
            raise PredictorPreservingA3Error("A3 mode must use the closed exact discriminant")
        if type(self.constant_rgb_cells) is not tuple or type(self.corrected_y1_copy_cells) is not tuple:
            raise PredictorPreservingA3Error("A3 control rows must be exact tuples")
        if any(type(row) is not SparseConstantRGBCellV1 for row in self.constant_rgb_cells):
            raise PredictorPreservingA3Error("constant RGB rows changed exact type")
        if any(type(row) is not CorrectedY1SupportCopyCellV1 for row in self.corrected_y1_copy_cells):
            raise PredictorPreservingA3Error("corrected-Y1 copy rows changed exact type")

        if self.mode is PredictorPreservingA3Mode.PASS_P0_V1:
            if self.constant_rgb_cells or self.corrected_y1_copy_cells:
                raise PredictorPreservingA3Error("PASS_P0_V1 must contain exactly zero control rows")
        elif self.mode is PredictorPreservingA3Mode.SPARSE_CONSTANT_RGB_V1:
            if not self.constant_rgb_cells or self.corrected_y1_copy_cells:
                raise PredictorPreservingA3Error("sparse RGB mode requires only nonempty RGB rows")
            addresses = tuple(row.address for row in self.constant_rgb_cells)
            if addresses != tuple(sorted(addresses)) or len(set(addresses)) != len(addresses):
                raise PredictorPreservingA3Error("sparse RGB cells must be unique canonical address order")
        elif self.mode is PredictorPreservingA3Mode.COPY_CORRECTED_Y1_SUPPORT_V1:
            if self.constant_rgb_cells or not self.corrected_y1_copy_cells:
                raise PredictorPreservingA3Error("corrected-Y1 copy mode requires only nonempty copy rows")
            addresses = tuple(row.address for row in self.corrected_y1_copy_cells)
            if addresses != tuple(sorted(addresses)) or len(set(addresses)) != len(addresses):
                raise PredictorPreservingA3Error("corrected-Y1 copy cells must be unique canonical address order")
        else:  # pragma: no cover - exact enum exhaustiveness guard
            raise PredictorPreservingA3Error("unsupported A3 mode")

    @property
    def control_row_count(self) -> int:
        return len(self.constant_rgb_cells) + len(self.corrected_y1_copy_cells)

    @property
    def coordinate_bytes(self) -> int:
        return self.control_row_count * COPY_ROW.size

    @property
    def rgb_value_bytes(self) -> int:
        return len(self.constant_rgb_cells) * CHANNELS

    @property
    def body_bytes(self) -> int:
        return self.coordinate_bytes + self.rgb_value_bytes


@dataclass(frozen=True, slots=True)
class PredictorCameraPairSurfaceV1:
    """Exact ephemeral P0/P1 frames bound to one counted predictor state.

    The dense frames are runtime inputs only.  This type has no serialization
    method by design.
    """

    predictor_state: TaskspacePredictorStateV2
    chronological_camera_frames: np.ndarray
    upstream_decode_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.predictor_state) is not TaskspacePredictorStateV2:
            raise PredictorPreservingA3Error("predictor surface requires exact TaskspacePredictorStateV2")
        if not 1 <= self.predictor_state.pair_count <= MAX_BOUNDED_PAIRS:
            raise PredictorPreservingA3Error("A3 predictor surface is bounded to at most four pairs")
        frames = np.asarray(self.chronological_camera_frames)
        expected = (
            self.predictor_state.pair_count,
            2,
            CAMERA_HEIGHT,
            CAMERA_WIDTH,
            CHANNELS,
        )
        if frames.dtype != np.uint8 or frames.shape != expected:
            raise PredictorPreservingA3Error("predictor surface must be uint8 [pair,2,874,1164,3]")
        if self.upstream_decode_receipt_sha256 is not None:
            _require_sha256(self.upstream_decode_receipt_sha256, "upstream_decode_receipt_sha256")
        object.__setattr__(self, "chronological_camera_frames", _immutable_array(frames, dtype=np.dtype(np.uint8)))

    @classmethod
    def from_ep725(cls, surface: Ep725EphemeralRuntimeSurfaceV2) -> PredictorCameraPairSurfaceV1:
        if type(surface) is not Ep725EphemeralRuntimeSurfaceV2:
            raise PredictorPreservingA3Error("ep725 adapter requires exact ephemeral runtime surface")
        return cls(
            predictor_state=surface.predictor_state,
            chronological_camera_frames=surface.chronological_camera_frames,
            upstream_decode_receipt_sha256=surface.bounded_decode.receipt_sha256,
        )

    @property
    def camera_p0(self) -> np.ndarray:
        return self.chronological_camera_frames[:, 0]

    @property
    def camera_p1(self) -> np.ndarray:
        return self.chronological_camera_frames[:, 1]

    @property
    def camera_p0_sha256(self) -> str:
        return _array_sha256(self.camera_p0)

    @property
    def camera_p1_sha256(self) -> str:
        return _array_sha256(self.camera_p1)

    @property
    def chronological_sha256(self) -> str:
        return _array_sha256(self.chronological_camera_frames)

    @property
    def binding_sha256(self) -> str:
        return _sha256(
            _canonical_json(
                {
                    "schema": SURFACE_SCHEMA,
                    "predictor_state_binding_sha256": self.predictor_state.binding_sha256,
                    "source_pair_ids": list(self.predictor_state.source_pair_ids),
                    "camera_p0_sha256": self.camera_p0_sha256,
                    "camera_p1_sha256": self.camera_p1_sha256,
                    "chronological_sha256": self.chronological_sha256,
                    "upstream_decode_receipt_sha256": self.upstream_decode_receipt_sha256,
                    "dense_frames_persisted": False,
                    "additional_counted_bytes": 0,
                }
            )
        )


@dataclass(frozen=True, slots=True)
class PredictorPreservingA3SourceBindingV1:
    """All P/G/camera foreign keys required to decode one A3 packet."""

    schema: Literal["tac.predictor_preserving_coupled_preimage_source.v1"]
    source_pair_ids: tuple[int, ...]
    predictor_state_binding_sha256: str
    predictor_semantic_binding_sha256: str
    predictor_program_sha256: str
    predictor_renderer_sha256: str
    predictor_surface_binding_sha256: str
    upstream_decode_receipt_sha256: str | None
    predictor_labels_sha256: str
    predictor_camera_p0_sha256: str
    predictor_camera_p1_sha256: str
    correction_packet_sha256: str
    correction_labels_sha256: str
    realization_profile_sha256: str
    overlay_receipt_sha256: str
    corrected_camera_y1_sha256: str

    def __post_init__(self) -> None:
        if self.schema != SOURCE_SCHEMA:
            raise PredictorPreservingA3Error("A3 source schema changed")
        if type(self.source_pair_ids) is not tuple or not self.source_pair_ids:
            raise PredictorPreservingA3Error("A3 source pair IDs must be a nonempty exact tuple")
        if any(type(value) is not int for value in self.source_pair_ids):
            raise PredictorPreservingA3Error("A3 source pair IDs must be exact integers")
        expected = tuple(range(self.source_pair_ids[0], self.source_pair_ids[0] + len(self.source_pair_ids)))
        if self.source_pair_ids != expected or not 0 <= expected[0] < expected[-1] + 1 <= 600:
            raise PredictorPreservingA3Error("A3 source pair IDs must be contiguous inside [0,600)")
        if len(expected) > MAX_BOUNDED_PAIRS:
            raise PredictorPreservingA3Error("A3 source binding exceeds bounded pair scope")
        for field in (
            "predictor_state_binding_sha256",
            "predictor_semantic_binding_sha256",
            "predictor_program_sha256",
            "predictor_renderer_sha256",
            "predictor_surface_binding_sha256",
            "predictor_labels_sha256",
            "predictor_camera_p0_sha256",
            "predictor_camera_p1_sha256",
            "correction_packet_sha256",
            "correction_labels_sha256",
            "realization_profile_sha256",
            "overlay_receipt_sha256",
            "corrected_camera_y1_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        if self.upstream_decode_receipt_sha256 is not None:
            _require_sha256(self.upstream_decode_receipt_sha256, "upstream_decode_receipt_sha256")

    @property
    def pair_start(self) -> int:
        return self.source_pair_ids[0]

    @property
    def pair_count(self) -> int:
        return len(self.source_pair_ids)

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_pair_ids"] = list(self.source_pair_ids)
        return value

    @property
    def binding_sha256(self) -> str:
        return _sha256(_canonical_json(self.as_dict()))


def derive_predictor_preserving_a3_source_binding(
    predictor_surface: PredictorCameraPairSurfaceV1,
    decoded_g: DecodedGenerativeCorrectionV1,
    corrected_y1_overlay: PredictorPreservingOverlayResultV1,
) -> PredictorPreservingA3SourceBindingV1:
    """Validate and bind the exact coupled P/G/P0/corrected-Y1 object."""

    if type(predictor_surface) is not PredictorCameraPairSurfaceV1:
        raise PredictorPreservingA3Error("A3 requires exact PredictorCameraPairSurfaceV1")
    if type(decoded_g) is not DecodedGenerativeCorrectionV1:
        raise PredictorPreservingA3Error("A3 requires exact DecodedGenerativeCorrectionV1")
    if type(corrected_y1_overlay) is not PredictorPreservingOverlayResultV1:
        raise PredictorPreservingA3Error("A3 requires exact PredictorPreservingOverlayResultV1")
    state = predictor_surface.predictor_state
    receipt = parse_predictor_preserving_overlay_receipt(
        corrected_y1_overlay.receipt.to_receipt_bytes()
    )
    if decoded_g.realization_profile is None:
        raise PredictorPreservingA3Error("A3 corrected Y1 must be backed by a finite G realization profile")
    if decoded_g.labels.shape != state.labels.shape:
        raise PredictorPreservingA3Error("G pair/semantic shape differs from predictor state")
    if receipt.pair_count != state.pair_count:
        raise PredictorPreservingA3Error("corrected Y1 pair count differs from predictor state")

    expected_equalities = {
        "correction_packet_sha256": decoded_g.correction_packet_sha256,
        "realization_profile_sha256": _profile_sha256(decoded_g.realization_profile),
        "predictor_labels_sha256": state.labels_sha256,
        "corrected_labels_sha256": _array_sha256(decoded_g.labels),
        "changed_label_mask_sha256": _array_sha256(decoded_g.labels != state.labels),
        "base_camera_y1_sha256": predictor_surface.camera_p1_sha256,
        "corrected_camera_y1_sha256": _array_sha256(corrected_y1_overlay.camera_y1),
    }
    mismatches = [
        field for field, expected in expected_equalities.items() if getattr(receipt, field) != expected
    ]
    if mismatches:
        raise PredictorPreservingA3Error(
            f"corrected Y1 overlay differs from live P/G sources: {sorted(mismatches)}"
        )

    return PredictorPreservingA3SourceBindingV1(
        schema=SOURCE_SCHEMA,
        source_pair_ids=state.source_pair_ids,
        predictor_state_binding_sha256=state.binding_sha256,
        predictor_semantic_binding_sha256=state.semantic_binding_sha256,
        predictor_program_sha256=state.predictor_program_sha256,
        predictor_renderer_sha256=state.predictor_renderer_sha256,
        predictor_surface_binding_sha256=predictor_surface.binding_sha256,
        upstream_decode_receipt_sha256=predictor_surface.upstream_decode_receipt_sha256,
        predictor_labels_sha256=state.labels_sha256,
        predictor_camera_p0_sha256=predictor_surface.camera_p0_sha256,
        predictor_camera_p1_sha256=predictor_surface.camera_p1_sha256,
        correction_packet_sha256=decoded_g.correction_packet_sha256,
        correction_labels_sha256=_array_sha256(decoded_g.labels),
        realization_profile_sha256=_profile_sha256(decoded_g.realization_profile),
        overlay_receipt_sha256=receipt.receipt_sha256,
        corrected_camera_y1_sha256=receipt.corrected_camera_y1_sha256,
    )


def _encode_packet(
    program: PredictorPreservingA3ProgramV1,
    source: PredictorPreservingA3SourceBindingV1,
) -> bytes:
    if type(program) is not PredictorPreservingA3ProgramV1:
        raise PredictorPreservingA3Error("A3 compile requires exact program type")
    if type(source) is not PredictorPreservingA3SourceBindingV1:
        raise PredictorPreservingA3Error("A3 packet requires exact source binding")
    addressed = (
        tuple(row.source_pair_id for row in program.constant_rgb_cells)
        + tuple(row.source_pair_id for row in program.corrected_y1_copy_cells)
    )
    if any(pair_id not in source.source_pair_ids for pair_id in addressed):
        raise PredictorPreservingA3Error("A3 control row addresses a pair outside the bound source window")

    body = bytearray()
    if program.mode is PredictorPreservingA3Mode.SPARSE_CONSTANT_RGB_V1:
        for row in program.constant_rgb_cells:
            body.extend(
                RGB_ROW.pack(
                    row.source_pair_id,
                    row.scorer_row,
                    row.scorer_col,
                    *row.rgb_u8,
                )
            )
    elif program.mode is PredictorPreservingA3Mode.COPY_CORRECTED_Y1_SUPPORT_V1:
        for row in program.corrected_y1_copy_cells:
            body.extend(COPY_ROW.pack(row.source_pair_id, row.scorer_row, row.scorer_col))
    if len(body) != program.body_bytes:
        raise PredictorPreservingA3Error("A3 packet body accounting changed")
    if program.control_row_count > 0xFFFFFFFF or len(body) > 0xFFFFFFFF:
        raise PredictorPreservingA3Error("A3 packet exceeds the version-1 integer ABI")
    packet_bytes = PACKET_HEADER.size + len(body)
    header_zero_crc = PACKET_HEADER.pack(
        PACKET_MAGIC,
        PACKET_VERSION,
        _MODE_TO_WIRE[program.mode],
        source.pair_start,
        source.pair_count,
        bytes.fromhex(source.binding_sha256),
        program.control_row_count,
        len(body),
        packet_bytes,
        0,
    )
    crc = zlib.crc32(header_zero_crc + body) & 0xFFFFFFFF
    header = PACKET_HEADER.pack(
        PACKET_MAGIC,
        PACKET_VERSION,
        _MODE_TO_WIRE[program.mode],
        source.pair_start,
        source.pair_count,
        bytes.fromhex(source.binding_sha256),
        program.control_row_count,
        len(body),
        packet_bytes,
        crc,
    )
    return header + bytes(body)


@dataclass(frozen=True, slots=True)
class ParsedPredictorPreservingA3PacketV1:
    packet: bytes
    source_binding_sha256: str
    pair_start: int
    pair_count: int
    packet_bytes: int
    packet_header_bytes: int
    packet_body_bytes: int
    coordinate_bytes: int
    rgb_value_bytes: int
    program: PredictorPreservingA3ProgramV1


def parse_predictor_preserving_a3_packet(
    packet: bytes,
    *,
    expected_source_binding: PredictorPreservingA3SourceBindingV1,
) -> ParsedPredictorPreservingA3PacketV1:
    """Strict packet parse with CRC, source-FK, and byte re-encode closure."""

    if type(packet) is not bytes or len(packet) < PACKET_HEADER.size:
        raise PredictorPreservingA3Error("A3 packet is shorter than its exact header")
    if type(expected_source_binding) is not PredictorPreservingA3SourceBindingV1:
        raise PredictorPreservingA3Error("A3 parse requires exact expected source binding")
    try:
        (
            magic,
            version,
            mode_wire,
            pair_start,
            pair_count,
            source_binding_raw,
            row_count,
            body_bytes,
            packet_bytes,
            crc,
        ) = PACKET_HEADER.unpack_from(packet, 0)
    except struct.error as exc:  # pragma: no cover - guarded length
        raise PredictorPreservingA3Error("A3 packet header is malformed") from exc
    if magic != PACKET_MAGIC or version != PACKET_VERSION:
        raise PredictorPreservingA3Error("A3 packet magic/version is unsupported")
    try:
        mode = _WIRE_TO_MODE[mode_wire]
    except KeyError as exc:
        raise PredictorPreservingA3Error("A3 packet mode is outside the closed universe") from exc
    if pair_start != expected_source_binding.pair_start or pair_count != expected_source_binding.pair_count:
        raise PredictorPreservingA3Error("A3 packet pair window differs from exact source binding")
    if source_binding_raw.hex() != expected_source_binding.binding_sha256:
        raise PredictorPreservingA3Error("A3 packet source binding differs from live P/G/P0/Y1")
    if packet_bytes != len(packet) or body_bytes != len(packet) - PACKET_HEADER.size:
        raise PredictorPreservingA3Error("A3 packet length accounting or trailing bytes changed")
    row_size = {
        PredictorPreservingA3Mode.PASS_P0_V1: 0,
        PredictorPreservingA3Mode.SPARSE_CONSTANT_RGB_V1: RGB_ROW.size,
        PredictorPreservingA3Mode.COPY_CORRECTED_Y1_SUPPORT_V1: COPY_ROW.size,
    }[mode]
    if mode is PredictorPreservingA3Mode.PASS_P0_V1:
        if row_count != 0 or body_bytes != 0:
            raise PredictorPreservingA3Error("PASS_P0_V1 packet must have zero rows and zero mutation body")
    elif row_count == 0 or body_bytes != row_count * row_size:
        raise PredictorPreservingA3Error("A3 row count/body length differs from its mode ABI")
    body = packet[PACKET_HEADER.size :]
    header_zero_crc = PACKET_HEADER.pack(
        magic,
        version,
        mode_wire,
        pair_start,
        pair_count,
        source_binding_raw,
        row_count,
        body_bytes,
        packet_bytes,
        0,
    )
    if zlib.crc32(header_zero_crc + body) & 0xFFFFFFFF != crc:
        raise PredictorPreservingA3Error("A3 packet CRC differs from exact bytes")

    rgb_rows: list[SparseConstantRGBCellV1] = []
    copy_rows: list[CorrectedY1SupportCopyCellV1] = []
    offset = 0
    try:
        if mode is PredictorPreservingA3Mode.SPARSE_CONSTANT_RGB_V1:
            for _ in range(row_count):
                pair_id, scorer_row, scorer_col, red, green, blue = RGB_ROW.unpack_from(body, offset)
                offset += RGB_ROW.size
                rgb_rows.append(
                    SparseConstantRGBCellV1(pair_id, scorer_row, scorer_col, (red, green, blue))
                )
        elif mode is PredictorPreservingA3Mode.COPY_CORRECTED_Y1_SUPPORT_V1:
            for _ in range(row_count):
                pair_id, scorer_row, scorer_col = COPY_ROW.unpack_from(body, offset)
                offset += COPY_ROW.size
                copy_rows.append(CorrectedY1SupportCopyCellV1(pair_id, scorer_row, scorer_col))
    except struct.error as exc:  # pragma: no cover - exact length checked above
        raise PredictorPreservingA3Error("A3 control row is truncated") from exc
    if offset != len(body):
        raise PredictorPreservingA3Error("A3 packet body was not consumed exactly")
    program = PredictorPreservingA3ProgramV1(
        mode=mode,
        constant_rgb_cells=tuple(rgb_rows),
        corrected_y1_copy_cells=tuple(copy_rows),
    )
    addressed = (
        tuple(row.source_pair_id for row in program.constant_rgb_cells)
        + tuple(row.source_pair_id for row in program.corrected_y1_copy_cells)
    )
    if any(pair_id not in expected_source_binding.source_pair_ids for pair_id in addressed):
        raise PredictorPreservingA3Error("A3 packet row addresses a pair outside its exact source window")
    if _encode_packet(program, expected_source_binding) != packet:
        raise PredictorPreservingA3Error("A3 packet changed on strict typed parse/re-encode")
    return ParsedPredictorPreservingA3PacketV1(
        packet=packet,
        source_binding_sha256=expected_source_binding.binding_sha256,
        pair_start=pair_start,
        pair_count=pair_count,
        packet_bytes=packet_bytes,
        packet_header_bytes=PACKET_HEADER.size,
        packet_body_bytes=body_bytes,
        coordinate_bytes=program.coordinate_bytes,
        rgb_value_bytes=program.rgb_value_bytes,
        program=program,
    )


def _numerator_stack(frames: np.ndarray) -> tuple[np.ndarray, int]:
    values: list[np.ndarray] = []
    denominator: int | None = None
    for frame in frames:
        try:
            numerators, current = _operator().apply_numerators(frame)
        except Uint8LatticeError as exc:
            raise PredictorPreservingA3Error("camera-to-scorer exact numerator proof failed") from exc
        if denominator is not None and current != denominator:
            raise PredictorPreservingA3Error("scorer denominator changed between pair frames")
        denominator = current
        values.append(np.ascontiguousarray(numerators))
    if denominator is None:  # pragma: no cover - source pair count is nonempty
        raise PredictorPreservingA3Error("empty A3 numerator stack")
    return np.stack(values, axis=0), denominator


@dataclass(frozen=True, slots=True)
class ExactDisjointCameraCellReconstructionV1:
    """Generic exact owned-cell reconstruction result.

    This kernel is intentionally independent of A packet/source bindings so G
    can reuse the same camera-lattice mechanism for same-class realization
    repair.  A and G remain distinct packet domains around this primitive.
    """

    output_camera: np.ndarray
    owned_scorer_mask: np.ndarray
    owned_camera_mask: np.ndarray
    base_scorer_numerators: np.ndarray
    output_scorer_numerators: np.ndarray
    scorer_denominator: int
    actually_changed_camera_values: int
    constant_rgb_target_numerators_exact: bool
    copy_source_target_numerators_exact: bool

    def __post_init__(self) -> None:
        output = np.asarray(self.output_camera)
        scorer_owned = np.asarray(self.owned_scorer_mask)
        camera_owned = np.asarray(self.owned_camera_mask)
        base_num = np.asarray(self.base_scorer_numerators)
        output_num = np.asarray(self.output_scorer_numerators)
        if output.dtype != np.uint8 or output.ndim != 4 or output.shape[1:] != (
            CAMERA_HEIGHT,
            CAMERA_WIDTH,
            CHANNELS,
        ):
            raise PredictorPreservingA3Error("owned-cell reconstruction output camera ABI changed")
        pair_count = output.shape[0]
        if scorer_owned.dtype != np.bool_ or scorer_owned.shape != (
            pair_count,
            SCORER_HEIGHT,
            SCORER_WIDTH,
        ):
            raise PredictorPreservingA3Error("owned-cell reconstruction scorer mask ABI changed")
        if camera_owned.dtype != np.bool_ or camera_owned.shape != (
            pair_count,
            CAMERA_HEIGHT,
            CAMERA_WIDTH,
        ):
            raise PredictorPreservingA3Error("owned-cell reconstruction camera mask ABI changed")
        expected_numerator_shape = (pair_count, SCORER_HEIGHT, SCORER_WIDTH, CHANNELS)
        if (
            base_num.dtype != np.int64
            or output_num.dtype != np.int64
            or base_num.shape != expected_numerator_shape
            or output_num.shape != expected_numerator_shape
        ):
            raise PredictorPreservingA3Error("owned-cell reconstruction numerator ABI changed")
        if type(self.scorer_denominator) is not int or self.scorer_denominator <= 0:
            raise PredictorPreservingA3Error("owned-cell reconstruction denominator changed")
        if type(self.actually_changed_camera_values) is not int or self.actually_changed_camera_values < 0:
            raise PredictorPreservingA3Error("owned-cell reconstruction changed-value count is invalid")
        if type(self.constant_rgb_target_numerators_exact) is not bool:
            raise PredictorPreservingA3Error("constant-RGB proof flag changed exact bool type")
        if type(self.copy_source_target_numerators_exact) is not bool:
            raise PredictorPreservingA3Error("copy-source proof flag changed exact bool type")
        object.__setattr__(self, "output_camera", _immutable_array(output, dtype=np.dtype(np.uint8)))
        object.__setattr__(
            self,
            "owned_scorer_mask",
            _immutable_array(scorer_owned, dtype=np.dtype(np.bool_)),
        )
        object.__setattr__(
            self,
            "owned_camera_mask",
            _immutable_array(camera_owned, dtype=np.dtype(np.bool_)),
        )
        object.__setattr__(
            self,
            "base_scorer_numerators",
            _immutable_array(base_num, dtype=np.dtype(np.int64)),
        )
        object.__setattr__(
            self,
            "output_scorer_numerators",
            _immutable_array(output_num, dtype=np.dtype(np.int64)),
        )


def apply_disjoint_camera_cell_reconstruction_v1(
    base_camera: np.ndarray,
    *,
    source_pair_ids: tuple[int, ...],
    constant_rgb_cells: tuple[SparseConstantRGBCellV1, ...] = (),
    copy_source_camera: np.ndarray | None = None,
    copy_source_cells: tuple[CorrectedY1SupportCopyCellV1, ...] = (),
) -> ExactDisjointCameraCellReconstructionV1:
    """Apply one exact disjoint-support reconstruction mechanism.

    The empty-row form is exact pass-through.  A nonempty call must choose one
    mechanism: constant RGB targets or camera taps copied from a bound source.
    Duplicate, noncanonical, overlapping, foreign-pair, and no-op rows fail.
    """

    if type(source_pair_ids) is not tuple or not 1 <= len(source_pair_ids) <= MAX_BOUNDED_PAIRS:
        raise PredictorPreservingA3Error("owned-cell source_pair_ids must be a bounded exact tuple")
    if any(type(pair_id) is not int for pair_id in source_pair_ids) or len(set(source_pair_ids)) != len(
        source_pair_ids
    ):
        raise PredictorPreservingA3Error("owned-cell source_pair_ids must be unique exact integers")
    if type(constant_rgb_cells) is not tuple or any(
        type(row) is not SparseConstantRGBCellV1 for row in constant_rgb_cells
    ):
        raise PredictorPreservingA3Error("owned-cell constant RGB rows changed exact tuple/type")
    if type(copy_source_cells) is not tuple or any(
        type(row) is not CorrectedY1SupportCopyCellV1 for row in copy_source_cells
    ):
        raise PredictorPreservingA3Error("owned-cell copy rows changed exact tuple/type")
    if constant_rgb_cells and copy_source_cells:
        raise PredictorPreservingA3Error("one owned-cell reconstruction cannot mix mechanisms")
    if copy_source_cells and copy_source_camera is None:
        raise PredictorPreservingA3Error("copy rows require one exact source camera")
    if not copy_source_cells and copy_source_camera is not None:
        raise PredictorPreservingA3Error("copy source camera without copy rows is an alias")

    base = np.asarray(base_camera)
    expected_shape = (len(source_pair_ids), CAMERA_HEIGHT, CAMERA_WIDTH, CHANNELS)
    if base.dtype != np.uint8 or base.shape != expected_shape:
        raise PredictorPreservingA3Error("owned-cell base camera must be uint8 [pair,874,1164,3]")
    copy_source: np.ndarray | None = None
    if copy_source_camera is not None:
        copy_source = np.asarray(copy_source_camera)
        if copy_source.dtype != np.uint8 or copy_source.shape != expected_shape:
            raise PredictorPreservingA3Error("owned-cell copy source camera changed exact ABI")

    rows: tuple[SparseConstantRGBCellV1 | CorrectedY1SupportCopyCellV1, ...]
    rows = constant_rgb_cells if constant_rgb_cells else copy_source_cells
    addresses = tuple(row.address for row in rows)
    if addresses != tuple(sorted(addresses)) or len(addresses) != len(set(addresses)):
        raise PredictorPreservingA3Error("owned-cell rows must be unique canonical address order")
    if any(row.source_pair_id not in source_pair_ids for row in rows):
        raise PredictorPreservingA3Error("owned-cell row addresses a foreign source pair")

    output = np.ascontiguousarray(base.copy())
    scorer_owned = np.zeros((len(source_pair_ids), SCORER_HEIGHT, SCORER_WIDTH), dtype=np.bool_)
    camera_owned = np.zeros((len(source_pair_ids), CAMERA_HEIGHT, CAMERA_WIDTH), dtype=np.bool_)
    pair_to_index = {pair_id: index for index, pair_id in enumerate(source_pair_ids)}
    operator = _operator()
    for row in rows:
        pair_index = pair_to_index[row.source_pair_id]
        if scorer_owned[pair_index, row.scorer_row, row.scorer_col]:
            raise PredictorPreservingA3Error("owned-cell duplicate scorer ownership survived validation")
        scorer_owned[pair_index, row.scorer_row, row.scorer_col] = True
        row_support = operator.row_supports[row.scorer_row]
        col_support = operator.col_supports[row.scorer_col]
        camera_indices = tuple(
            (camera_row, camera_col)
            for camera_row in row_support.indices
            for camera_col in col_support.indices
        )
        if any(camera_owned[pair_index, camera_row, camera_col] for camera_row, camera_col in camera_indices):
            raise PredictorPreservingA3Error("owned scorer cells unexpectedly overlap on camera lattice")
        before = np.stack(
            [base[pair_index, camera_row, camera_col] for camera_row, camera_col in camera_indices],
            axis=0,
        )
        if type(row) is SparseConstantRGBCellV1:
            replacement = np.broadcast_to(np.asarray(row.rgb_u8, dtype=np.uint8), before.shape)
        else:
            assert copy_source is not None  # guarded above; narrows the exact mechanism
            replacement = np.stack(
                [copy_source[pair_index, camera_row, camera_col] for camera_row, camera_col in camera_indices],
                axis=0,
            )
        if np.array_equal(before, replacement):
            raise PredictorPreservingA3Error("owned-cell reconstruction contains a no-op alias row")
        for camera_row, camera_col in camera_indices:
            camera_owned[pair_index, camera_row, camera_col] = True
            if type(row) is SparseConstantRGBCellV1:
                output[pair_index, camera_row, camera_col] = row.rgb_u8
            else:
                assert copy_source is not None
                output[pair_index, camera_row, camera_col] = copy_source[
                    pair_index, camera_row, camera_col
                ]

    base_numerators, denominator = _numerator_stack(base)
    output_numerators, output_denominator = _numerator_stack(output)
    if denominator != output_denominator:
        raise PredictorPreservingA3Error("owned-cell reconstruction changed scorer denominator")
    if not np.array_equal(output[~camera_owned], base[~camera_owned]):
        raise PredictorPreservingA3Error("owned-cell reconstruction changed an unowned camera value")
    if not np.array_equal(output_numerators[~scorer_owned], base_numerators[~scorer_owned]):
        raise PredictorPreservingA3Error("owned-cell reconstruction changed an unowned scorer numerator")

    rgb_exact = False
    copy_exact = False
    if constant_rgb_cells:
        for row in constant_rgb_cells:
            pair_index = pair_to_index[row.source_pair_id]
            expected = np.asarray(row.rgb_u8, dtype=np.int64) * denominator
            if not np.array_equal(output_numerators[pair_index, row.scorer_row, row.scorer_col], expected):
                raise PredictorPreservingA3Error("constant RGB row missed its exact target numerator")
        rgb_exact = True
    elif copy_source_cells:
        assert copy_source is not None
        copy_numerators, copy_denominator = _numerator_stack(copy_source)
        if copy_denominator != denominator:
            raise PredictorPreservingA3Error("owned-cell copy source scorer denominator changed")
        if not np.array_equal(output_numerators[scorer_owned], copy_numerators[scorer_owned]):
            raise PredictorPreservingA3Error("copy-source row missed its exact source numerator")
        copy_exact = True

    return ExactDisjointCameraCellReconstructionV1(
        output_camera=output,
        owned_scorer_mask=scorer_owned,
        owned_camera_mask=camera_owned,
        base_scorer_numerators=base_numerators,
        output_scorer_numerators=output_numerators,
        scorer_denominator=denominator,
        actually_changed_camera_values=int(np.count_nonzero(output != base)),
        constant_rgb_target_numerators_exact=rgb_exact,
        copy_source_target_numerators_exact=copy_exact,
    )


@dataclass(frozen=True, slots=True)
class PredictorPreservingA3ReceiptV1:
    schema: Literal["tac.predictor_preserving_coupled_preimage_receipt.v1"]
    mode: str
    source_pair_ids: tuple[int, ...]
    source_binding_sha256: str
    predictor_state_binding_sha256: str
    predictor_semantic_binding_sha256: str
    predictor_program_sha256: str
    predictor_renderer_sha256: str
    predictor_surface_binding_sha256: str
    upstream_decode_receipt_sha256: str | None
    predictor_labels_sha256: str
    correction_packet_sha256: str
    correction_labels_sha256: str
    realization_profile_sha256: str
    overlay_receipt_sha256: str
    predictor_camera_p0_sha256: str
    predictor_camera_p1_sha256: str
    corrected_camera_y1_sha256: str
    output_camera_y0_sha256: str
    output_camera_y1_sha256: str
    chronological_camera_frames_sha256: str
    owned_scorer_mask_sha256: str
    owned_camera_mask_sha256: str
    predictor_p0_scorer_numerators_sha256: str
    output_y0_scorer_numerators_sha256: str
    corrected_y1_scorer_numerators_sha256: str
    packet_sha256: str
    packet_bytes: int
    packet_header_bytes: int
    packet_body_bytes: int
    serialized_coordinate_bytes: int
    serialized_rgb_value_bytes: int
    control_row_count: int
    owned_scorer_cells: int
    owned_camera_values: int
    actually_changed_camera_values: int
    scorer_denominator: int
    p0_pass_through_zero_mutation: bool
    constant_rgb_target_numerators_exact: bool
    corrected_y1_support_numerators_exact: bool
    implementation_scope: Literal[
        "predictor_preserving_camera_seam_control_not_full_a3_se3_xi_basis.v1"
    ] = "predictor_preserving_camera_seam_control_not_full_a3_se3_xi_basis.v1"
    full_a3_se3_xi_basis_universe_implemented: Literal[False] = False
    se3_xi_inverse_solve_implemented: Literal[False] = False
    encoder_side_row_selection_solved: Literal[False] = False
    corrected_y1_unchanged: Literal[True] = True
    outside_a_owned_p0_camera_values_preserved: Literal[True] = True
    outside_a_owned_p0_scorer_numerators_preserved: Literal[True] = True
    duplicate_or_overlapping_cells_rejected: Literal[True] = True
    no_op_alias_rows_rejected: Literal[True] = True
    deterministic_double_decode: Literal[True] = True
    packet_parse_reencode_exact: Literal[True] = True
    receipt_parse_reencode_exact: Literal[True] = True
    dense_predictor_frames_serialized: Literal[False] = False
    dense_corrected_y1_serialized: Literal[False] = False
    dense_output_frames_serialized: Literal[False] = False
    scorer_invoked: Literal[False] = False
    score_claim: Literal[False] = False
    candidate_claim: Literal[False] = False
    originality_claim: Literal[False] = False
    promotion_eligible: Literal[False] = False
    research_only: Literal[True] = True

    def __post_init__(self) -> None:
        if self.schema != RECEIPT_SCHEMA:
            raise PredictorPreservingA3Error("A3 receipt schema changed")
        try:
            mode = PredictorPreservingA3Mode(self.mode)
        except (TypeError, ValueError) as exc:
            raise PredictorPreservingA3Error("A3 receipt mode escaped the closed universe") from exc
        if type(self.source_pair_ids) is not tuple or not self.source_pair_ids:
            raise PredictorPreservingA3Error("A3 receipt source_pair_ids changed exact tuple type")
        if any(type(value) is not int for value in self.source_pair_ids):
            raise PredictorPreservingA3Error("A3 receipt source_pair_ids must contain exact integers")
        expected_ids = tuple(
            range(self.source_pair_ids[0], self.source_pair_ids[0] + len(self.source_pair_ids))
        )
        if (
            self.source_pair_ids != expected_ids
            or len(expected_ids) > MAX_BOUNDED_PAIRS
            or not 0 <= expected_ids[0] < expected_ids[-1] + 1 <= 600
        ):
            raise PredictorPreservingA3Error(
                "A3 receipt source_pair_ids must be bounded contiguous IDs inside [0,600)"
            )
        for field in (
            "source_binding_sha256",
            "predictor_state_binding_sha256",
            "predictor_semantic_binding_sha256",
            "predictor_program_sha256",
            "predictor_renderer_sha256",
            "predictor_surface_binding_sha256",
            "predictor_labels_sha256",
            "correction_packet_sha256",
            "correction_labels_sha256",
            "realization_profile_sha256",
            "overlay_receipt_sha256",
            "predictor_camera_p0_sha256",
            "predictor_camera_p1_sha256",
            "corrected_camera_y1_sha256",
            "output_camera_y0_sha256",
            "output_camera_y1_sha256",
            "chronological_camera_frames_sha256",
            "owned_scorer_mask_sha256",
            "owned_camera_mask_sha256",
            "predictor_p0_scorer_numerators_sha256",
            "output_y0_scorer_numerators_sha256",
            "corrected_y1_scorer_numerators_sha256",
            "packet_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        if self.upstream_decode_receipt_sha256 is not None:
            _require_sha256(self.upstream_decode_receipt_sha256, "upstream_decode_receipt_sha256")
        reconstructed_source = PredictorPreservingA3SourceBindingV1(
            schema=SOURCE_SCHEMA,
            source_pair_ids=self.source_pair_ids,
            predictor_state_binding_sha256=self.predictor_state_binding_sha256,
            predictor_semantic_binding_sha256=self.predictor_semantic_binding_sha256,
            predictor_program_sha256=self.predictor_program_sha256,
            predictor_renderer_sha256=self.predictor_renderer_sha256,
            predictor_surface_binding_sha256=self.predictor_surface_binding_sha256,
            upstream_decode_receipt_sha256=self.upstream_decode_receipt_sha256,
            predictor_labels_sha256=self.predictor_labels_sha256,
            predictor_camera_p0_sha256=self.predictor_camera_p0_sha256,
            predictor_camera_p1_sha256=self.predictor_camera_p1_sha256,
            correction_packet_sha256=self.correction_packet_sha256,
            correction_labels_sha256=self.correction_labels_sha256,
            realization_profile_sha256=self.realization_profile_sha256,
            overlay_receipt_sha256=self.overlay_receipt_sha256,
            corrected_camera_y1_sha256=self.corrected_camera_y1_sha256,
        )
        if reconstructed_source.binding_sha256 != self.source_binding_sha256:
            raise PredictorPreservingA3Error(
                "A3 receipt source binding does not recompose from exact foreign keys"
            )
        exact_ints = (
            "packet_bytes",
            "packet_header_bytes",
            "packet_body_bytes",
            "serialized_coordinate_bytes",
            "serialized_rgb_value_bytes",
            "control_row_count",
            "owned_scorer_cells",
            "owned_camera_values",
            "actually_changed_camera_values",
            "scorer_denominator",
        )
        if any(type(getattr(self, field)) is not int for field in exact_ints):
            raise PredictorPreservingA3Error("A3 receipt byte/count fields must be exact integers")
        if self.packet_header_bytes != PACKET_HEADER.size:
            raise PredictorPreservingA3Error("A3 receipt packet header size changed")
        if self.packet_bytes != self.packet_header_bytes + self.packet_body_bytes:
            raise PredictorPreservingA3Error("A3 receipt packet byte decomposition does not close")
        if self.packet_body_bytes != self.serialized_coordinate_bytes + self.serialized_rgb_value_bytes:
            raise PredictorPreservingA3Error("A3 receipt body byte decomposition does not close")
        if self.control_row_count != self.owned_scorer_cells:
            raise PredictorPreservingA3Error("A3 receipt ownership differs from one-cell-per-row grammar")
        if self.serialized_coordinate_bytes != self.control_row_count * COPY_ROW.size:
            raise PredictorPreservingA3Error("A3 receipt coordinate byte count changed")
        if self.owned_camera_values < 0 or self.actually_changed_camera_values < 0:
            raise PredictorPreservingA3Error("A3 receipt camera counts became negative")
        if self.actually_changed_camera_values > self.owned_camera_values:
            raise PredictorPreservingA3Error("A3 receipt changed more camera values than it owns")
        if self.scorer_denominator <= 0:
            raise PredictorPreservingA3Error("A3 receipt scorer denominator must be positive")

        expected_mode_truth = {
            PredictorPreservingA3Mode.PASS_P0_V1: (0, 0, 0, True, False, False),
            PredictorPreservingA3Mode.SPARSE_CONSTANT_RGB_V1: (
                self.control_row_count,
                self.control_row_count * CHANNELS,
                1,
                False,
                True,
                False,
            ),
            PredictorPreservingA3Mode.COPY_CORRECTED_Y1_SUPPORT_V1: (
                self.control_row_count,
                0,
                1,
                False,
                False,
                True,
            ),
        }[mode]
        expected_rows, expected_rgb_bytes, require_rows, pass_truth, rgb_truth, y1_truth = expected_mode_truth
        if self.control_row_count != expected_rows or self.serialized_rgb_value_bytes != expected_rgb_bytes:
            raise PredictorPreservingA3Error("A3 receipt row/value byte count differs from its mode")
        if require_rows and (self.control_row_count < 1 or self.actually_changed_camera_values < 1):
            raise PredictorPreservingA3Error("mutating A3 mode must contain at least one realized non-no-op row")
        if self.p0_pass_through_zero_mutation is not pass_truth:
            raise PredictorPreservingA3Error("A3 pass-through truth differs from mode")
        if self.constant_rgb_target_numerators_exact is not rgb_truth:
            raise PredictorPreservingA3Error("A3 constant-RGB numerator truth differs from mode")
        if self.corrected_y1_support_numerators_exact is not y1_truth:
            raise PredictorPreservingA3Error("A3 corrected-Y1 numerator truth differs from mode")
        truth = {
            "full_a3_se3_xi_basis_universe_implemented": False,
            "se3_xi_inverse_solve_implemented": False,
            "encoder_side_row_selection_solved": False,
            "corrected_y1_unchanged": True,
            "outside_a_owned_p0_camera_values_preserved": True,
            "outside_a_owned_p0_scorer_numerators_preserved": True,
            "duplicate_or_overlapping_cells_rejected": True,
            "no_op_alias_rows_rejected": True,
            "deterministic_double_decode": True,
            "packet_parse_reencode_exact": True,
            "receipt_parse_reencode_exact": True,
            "dense_predictor_frames_serialized": False,
            "dense_corrected_y1_serialized": False,
            "dense_output_frames_serialized": False,
            "scorer_invoked": False,
            "score_claim": False,
            "candidate_claim": False,
            "originality_claim": False,
            "promotion_eligible": False,
            "research_only": True,
        }
        if any(getattr(self, field) is not expected for field, expected in truth.items()):
            raise PredictorPreservingA3Error("A3 receipt truth labels became permissive")
        if (
            self.implementation_scope
            != "predictor_preserving_camera_seam_control_not_full_a3_se3_xi_basis.v1"
        ):
            raise PredictorPreservingA3Error("A3 camera-seam implementation scope changed")
        if mode is PredictorPreservingA3Mode.PASS_P0_V1:
            if self.output_camera_y0_sha256 != self.predictor_camera_p0_sha256:
                raise PredictorPreservingA3Error("PASS_P0_V1 receipt changed exact P0 bytes")
            if self.predictor_p0_scorer_numerators_sha256 != self.output_y0_scorer_numerators_sha256:
                raise PredictorPreservingA3Error("PASS_P0_V1 receipt changed P0 scorer numerators")
            if self.owned_camera_values != 0 or self.actually_changed_camera_values != 0:
                raise PredictorPreservingA3Error("PASS_P0_V1 receipt owns or changes camera values")
        if self.output_camera_y1_sha256 != self.corrected_camera_y1_sha256:
            raise PredictorPreservingA3Error("A3 receipt changed corrected Y1")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_pair_ids"] = list(self.source_pair_ids)
        return value

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(self.as_dict())

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())


@dataclass(frozen=True, slots=True)
class DecodedPredictorPreservingA3V1:
    camera_y0: np.ndarray
    camera_y1: np.ndarray
    chronological_camera_frames: np.ndarray
    owned_scorer_mask: np.ndarray
    owned_camera_mask: np.ndarray
    receipt: PredictorPreservingA3ReceiptV1

    def __post_init__(self) -> None:
        pair_count = len(self.receipt.source_pair_ids)
        y0 = np.asarray(self.camera_y0)
        y1 = np.asarray(self.camera_y1)
        chronological = np.asarray(self.chronological_camera_frames)
        scorer_mask = np.asarray(self.owned_scorer_mask)
        camera_mask = np.asarray(self.owned_camera_mask)
        if y0.dtype != np.uint8 or y0.shape != (pair_count, CAMERA_HEIGHT, CAMERA_WIDTH, CHANNELS):
            raise PredictorPreservingA3Error("decoded A3 Y0 camera ABI changed")
        if y1.dtype != np.uint8 or y1.shape != y0.shape:
            raise PredictorPreservingA3Error("decoded A3 Y1 camera ABI changed")
        if chronological.dtype != np.uint8 or chronological.shape != (
            pair_count,
            2,
            CAMERA_HEIGHT,
            CAMERA_WIDTH,
            CHANNELS,
        ):
            raise PredictorPreservingA3Error("decoded A3 chronological camera ABI changed")
        if scorer_mask.dtype != np.bool_ or scorer_mask.shape != (pair_count, SCORER_HEIGHT, SCORER_WIDTH):
            raise PredictorPreservingA3Error("decoded A3 scorer ownership ABI changed")
        if camera_mask.dtype != np.bool_ or camera_mask.shape != (pair_count, CAMERA_HEIGHT, CAMERA_WIDTH):
            raise PredictorPreservingA3Error("decoded A3 camera ownership ABI changed")
        expected_hashes = {
            "output_camera_y0_sha256": _array_sha256(y0),
            "output_camera_y1_sha256": _array_sha256(y1),
            "chronological_camera_frames_sha256": _array_sha256(chronological),
            "owned_scorer_mask_sha256": _array_sha256(scorer_mask),
            "owned_camera_mask_sha256": _array_sha256(camera_mask),
        }
        if any(getattr(self.receipt, field) != expected for field, expected in expected_hashes.items()):
            raise PredictorPreservingA3Error("decoded A3 arrays differ from exact receipt hashes")
        if not np.array_equal(chronological[:, 0], y0) or not np.array_equal(chronological[:, 1], y1):
            raise PredictorPreservingA3Error("decoded A3 chronological ordering is not exact [Y0,Y1]")
        object.__setattr__(self, "camera_y0", _immutable_array(y0, dtype=np.dtype(np.uint8)))
        object.__setattr__(self, "camera_y1", _immutable_array(y1, dtype=np.dtype(np.uint8)))
        object.__setattr__(
            self,
            "chronological_camera_frames",
            _immutable_array(chronological, dtype=np.dtype(np.uint8)),
        )
        object.__setattr__(self, "owned_scorer_mask", _immutable_array(scorer_mask, dtype=np.dtype(np.bool_)))
        object.__setattr__(self, "owned_camera_mask", _immutable_array(camera_mask, dtype=np.dtype(np.bool_)))


def _decode_parsed_packet(
    parsed: ParsedPredictorPreservingA3PacketV1,
    source: PredictorPreservingA3SourceBindingV1,
    predictor_surface: PredictorCameraPairSurfaceV1,
    corrected_y1_overlay: PredictorPreservingOverlayResultV1,
) -> DecodedPredictorPreservingA3V1:
    program = parsed.program
    p0 = predictor_surface.camera_p0
    corrected_y1 = corrected_y1_overlay.camera_y1
    reconstruction = apply_disjoint_camera_cell_reconstruction_v1(
        p0,
        source_pair_ids=source.source_pair_ids,
        constant_rgb_cells=program.constant_rgb_cells,
        copy_source_camera=(
            corrected_y1
            if program.mode is PredictorPreservingA3Mode.COPY_CORRECTED_Y1_SUPPORT_V1
            else None
        ),
        copy_source_cells=program.corrected_y1_copy_cells,
    )
    y0 = reconstruction.output_camera
    scorer_owned = reconstruction.owned_scorer_mask
    camera_owned = reconstruction.owned_camera_mask
    p0_numerators = reconstruction.base_scorer_numerators
    y0_numerators = reconstruction.output_scorer_numerators
    denominator = reconstruction.scorer_denominator
    y1_numerators, y1_denominator = _numerator_stack(corrected_y1)
    if denominator != y1_denominator:
        raise PredictorPreservingA3Error("A3 scorer denominator differs across P0/Y0/Y1")

    if program.mode is PredictorPreservingA3Mode.PASS_P0_V1:
        if not np.array_equal(y0, p0) or np.any(scorer_owned) or np.any(camera_owned):
            raise PredictorPreservingA3Error("PASS_P0_V1 failed exact zero-mutation pass-through")
    elif program.mode is PredictorPreservingA3Mode.SPARSE_CONSTANT_RGB_V1:
        if not reconstruction.constant_rgb_target_numerators_exact:
            raise PredictorPreservingA3Error("sparse RGB reconstruction lacked exact target proof")
    else:
        if not reconstruction.copy_source_target_numerators_exact:
            raise PredictorPreservingA3Error("corrected-Y1 reconstruction lacked exact copy proof")

    if _array_sha256(corrected_y1) != source.corrected_camera_y1_sha256:
        raise PredictorPreservingA3Error("corrected Y1 changed after source binding")
    chronological = np.stack((y0, corrected_y1), axis=1)
    actual_changes = reconstruction.actually_changed_camera_values
    owned_camera_values = int(np.count_nonzero(camera_owned)) * CHANNELS
    receipt = PredictorPreservingA3ReceiptV1(
        schema=RECEIPT_SCHEMA,
        mode=program.mode.value,
        source_pair_ids=source.source_pair_ids,
        source_binding_sha256=source.binding_sha256,
        predictor_state_binding_sha256=source.predictor_state_binding_sha256,
        predictor_semantic_binding_sha256=source.predictor_semantic_binding_sha256,
        predictor_program_sha256=source.predictor_program_sha256,
        predictor_renderer_sha256=source.predictor_renderer_sha256,
        predictor_surface_binding_sha256=source.predictor_surface_binding_sha256,
        upstream_decode_receipt_sha256=source.upstream_decode_receipt_sha256,
        predictor_labels_sha256=source.predictor_labels_sha256,
        correction_packet_sha256=source.correction_packet_sha256,
        correction_labels_sha256=source.correction_labels_sha256,
        realization_profile_sha256=source.realization_profile_sha256,
        overlay_receipt_sha256=source.overlay_receipt_sha256,
        predictor_camera_p0_sha256=source.predictor_camera_p0_sha256,
        predictor_camera_p1_sha256=source.predictor_camera_p1_sha256,
        corrected_camera_y1_sha256=source.corrected_camera_y1_sha256,
        output_camera_y0_sha256=_array_sha256(y0),
        output_camera_y1_sha256=_array_sha256(corrected_y1),
        chronological_camera_frames_sha256=_array_sha256(chronological),
        owned_scorer_mask_sha256=_array_sha256(scorer_owned),
        owned_camera_mask_sha256=_array_sha256(camera_owned),
        predictor_p0_scorer_numerators_sha256=_array_sha256(p0_numerators),
        output_y0_scorer_numerators_sha256=_array_sha256(y0_numerators),
        corrected_y1_scorer_numerators_sha256=_array_sha256(y1_numerators),
        packet_sha256=_sha256(parsed.packet),
        packet_bytes=parsed.packet_bytes,
        packet_header_bytes=parsed.packet_header_bytes,
        packet_body_bytes=parsed.packet_body_bytes,
        serialized_coordinate_bytes=parsed.coordinate_bytes,
        serialized_rgb_value_bytes=parsed.rgb_value_bytes,
        control_row_count=program.control_row_count,
        owned_scorer_cells=int(np.count_nonzero(scorer_owned)),
        owned_camera_values=owned_camera_values,
        actually_changed_camera_values=actual_changes,
        scorer_denominator=denominator,
        p0_pass_through_zero_mutation=program.mode is PredictorPreservingA3Mode.PASS_P0_V1,
        constant_rgb_target_numerators_exact=(
            program.mode is PredictorPreservingA3Mode.SPARSE_CONSTANT_RGB_V1
        ),
        corrected_y1_support_numerators_exact=(
            program.mode is PredictorPreservingA3Mode.COPY_CORRECTED_Y1_SUPPORT_V1
        ),
    )
    return DecodedPredictorPreservingA3V1(
        camera_y0=y0,
        camera_y1=corrected_y1,
        chronological_camera_frames=chronological,
        owned_scorer_mask=scorer_owned,
        owned_camera_mask=camera_owned,
        receipt=receipt,
    )


def decode_predictor_preserving_a3_packet(
    packet: bytes,
    *,
    predictor_surface: PredictorCameraPairSurfaceV1,
    decoded_g: DecodedGenerativeCorrectionV1,
    corrected_y1_overlay: PredictorPreservingOverlayResultV1,
) -> DecodedPredictorPreservingA3V1:
    """Decode twice and receipt-close exact A bytes against ephemeral P/G."""

    source = derive_predictor_preserving_a3_source_binding(
        predictor_surface,
        decoded_g,
        corrected_y1_overlay,
    )
    parsed = parse_predictor_preserving_a3_packet(packet, expected_source_binding=source)
    first = _decode_parsed_packet(parsed, source, predictor_surface, corrected_y1_overlay)
    second = _decode_parsed_packet(parsed, source, predictor_surface, corrected_y1_overlay)
    if first.receipt != second.receipt:
        raise PredictorPreservingA3Error("A3 deterministic double decode changed receipt")
    for field in (
        "camera_y0",
        "camera_y1",
        "chronological_camera_frames",
        "owned_scorer_mask",
        "owned_camera_mask",
    ):
        if not np.array_equal(getattr(first, field), getattr(second, field)):
            raise PredictorPreservingA3Error(f"A3 deterministic double decode changed {field}")
    receipt_payload = first.receipt.to_receipt_bytes()
    if parse_predictor_preserving_a3_receipt(receipt_payload) != first.receipt:
        raise PredictorPreservingA3Error("A3 receipt failed strict parse/re-emit closure")
    return first


def parse_predictor_preserving_a3_receipt(payload: bytes) -> PredictorPreservingA3ReceiptV1:
    """Strict JSON parse with duplicate-key/type/canonical re-emit checks."""

    if type(payload) is not bytes or not payload:
        raise PredictorPreservingA3Error("A3 receipt must be nonempty exact bytes")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PredictorPreservingA3Error(f"A3 receipt repeats key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("ascii"), object_pairs_hook=unique_pairs)
    except PredictorPreservingA3Error:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PredictorPreservingA3Error("A3 receipt is not strict ASCII JSON") from exc
    if type(value) is not dict or set(value) != set(PredictorPreservingA3ReceiptV1.__dataclass_fields__):
        raise PredictorPreservingA3Error("A3 receipt fields differ from the closed schema")
    if _canonical_json(value) != payload:
        raise PredictorPreservingA3Error("A3 receipt is not canonical JSON")
    if type(value["source_pair_ids"]) is not list or any(
        type(pair_id) is not int for pair_id in value["source_pair_ids"]
    ):
        raise PredictorPreservingA3Error("A3 receipt source_pair_ids must be a JSON list of exact integers")
    value["source_pair_ids"] = tuple(value["source_pair_ids"])
    try:
        receipt = PredictorPreservingA3ReceiptV1(**value)
    except TypeError as exc:
        raise PredictorPreservingA3Error("A3 receipt contains invalid typed fields") from exc
    if receipt.to_receipt_bytes() != payload:
        raise PredictorPreservingA3Error("A3 receipt changed on typed parse/re-emit")
    return receipt


@dataclass(frozen=True, slots=True)
class CompiledPredictorPreservingA3V1:
    packet: bytes
    program: PredictorPreservingA3ProgramV1
    source_binding: PredictorPreservingA3SourceBindingV1
    decoded: DecodedPredictorPreservingA3V1
    receipt: PredictorPreservingA3ReceiptV1

    def __post_init__(self) -> None:
        if type(self.packet) is not bytes or not self.packet:
            raise PredictorPreservingA3Error("compiled A3 packet must be nonempty exact bytes")
        if type(self.program) is not PredictorPreservingA3ProgramV1:
            raise PredictorPreservingA3Error("compiled A3 program changed exact type")
        if type(self.source_binding) is not PredictorPreservingA3SourceBindingV1:
            raise PredictorPreservingA3Error("compiled A3 source binding changed exact type")
        if type(self.decoded) is not DecodedPredictorPreservingA3V1 or self.decoded.receipt != self.receipt:
            raise PredictorPreservingA3Error("compiled A3 decoded result/receipt diverged")
        if self.receipt.packet_sha256 != _sha256(self.packet) or self.receipt.packet_bytes != len(self.packet):
            raise PredictorPreservingA3Error("compiled A3 packet differs from receipt")
        if self.receipt.source_binding_sha256 != self.source_binding.binding_sha256:
            raise PredictorPreservingA3Error("compiled A3 source differs from receipt")


def compile_predictor_preserving_a3(
    program: PredictorPreservingA3ProgramV1,
    *,
    predictor_surface: PredictorCameraPairSurfaceV1,
    decoded_g: DecodedGenerativeCorrectionV1,
    corrected_y1_overlay: PredictorPreservingOverlayResultV1,
) -> CompiledPredictorPreservingA3V1:
    """Compile, parse back, decode twice, and close the exact A3 receipt."""

    if type(program) is not PredictorPreservingA3ProgramV1:
        raise PredictorPreservingA3Error("A3 compile requires exact program type")
    source = derive_predictor_preserving_a3_source_binding(
        predictor_surface,
        decoded_g,
        corrected_y1_overlay,
    )
    packet = _encode_packet(program, source)
    decoded = decode_predictor_preserving_a3_packet(
        packet,
        predictor_surface=predictor_surface,
        decoded_g=decoded_g,
        corrected_y1_overlay=corrected_y1_overlay,
    )
    return CompiledPredictorPreservingA3V1(
        packet=packet,
        program=program,
        source_binding=source,
        decoded=decoded,
        receipt=decoded.receipt,
    )


__all__ = [
    "CAMERA_HEIGHT",
    "CAMERA_WIDTH",
    "CHANNELS",
    "MAX_BOUNDED_PAIRS",
    "PACKET_HEADER",
    "SCORER_HEIGHT",
    "SCORER_WIDTH",
    "CompiledPredictorPreservingA3V1",
    "CorrectedY1SupportCopyCellV1",
    "DecodedPredictorPreservingA3V1",
    "ExactDisjointCameraCellReconstructionV1",
    "ParsedPredictorPreservingA3PacketV1",
    "PredictorCameraPairSurfaceV1",
    "PredictorPreservingA3Error",
    "PredictorPreservingA3Mode",
    "PredictorPreservingA3ProgramV1",
    "PredictorPreservingA3ReceiptV1",
    "PredictorPreservingA3SourceBindingV1",
    "SparseConstantRGBCellV1",
    "apply_disjoint_camera_cell_reconstruction_v1",
    "compile_predictor_preserving_a3",
    "decode_predictor_preserving_a3_packet",
    "derive_predictor_preserving_a3_source_binding",
    "parse_predictor_preserving_a3_packet",
    "parse_predictor_preserving_a3_receipt",
]
