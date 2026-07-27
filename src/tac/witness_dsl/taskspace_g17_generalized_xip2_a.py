# SPDX-License-Identifier: MIT
"""Generalized conditional chronological A packet and NumPy receiver.

``TACX2A4`` keeps G13's exact XIP2 transport and geometry semantics while
binding them to a G17 receiver-derived conditional Y1 surface.  Scorer and
forward evidence remains encoder-only and is never needed by the decoder.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from dataclasses import dataclass, field, fields
from enum import IntEnum, StrEnum
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from tac.boundary_math.warp_real_luma_frame0 import GroundHomographyGeom, warp_frame0_native_numpy
from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator
from tac.witness_dsl.predictor_preserving_coupled_preimage import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CHANNELS,
    SCORER_HEIGHT,
    SCORER_WIDTH,
)
from tac.witness_dsl.taskspace_predictor_state_v2 import SE3XiTransportV2, TaskspacePredictorStateV2Error

if TYPE_CHECKING:
    from tac.witness_dsl.taskspace_g17_forward_observation import (
        G17CandidateForwardObservationV1,
        G17TargetForwardObservationV1,
    )

PACKET_MAGIC: Final = b"TACX2A4\x00"
PACKET_VERSION: Final = 1
PACKET_HEADER: Final = struct.Struct(">8sBBBBBHHHHHHf32s32sII")
PACKET_FOOTER: Final = struct.Struct(">I")
PACKET_HEADER_BYTES: Final = 101
PACKET_FOOTER_BYTES: Final = 4
SOURCE_SCHEMA: Final = "tac.taskspace_g17_conditional_y1_source.v1"
ACQUISITION_SCHEMA: Final = "tac.taskspace_g17_conditional_y1_acquisition_custody.v1"
RECEIPT_SCHEMA: Final = "tac.taskspace_g17_generalized_xip2_a_receipt.v1"
MAX_XIP2_BODY_BYTES: Final = 1_048_576
ABSENT_DIGEST: Final = "0" * 64
_POSITIVE_ZERO_F32: Final = b"\x00\x00\x00\x00"

assert PACKET_HEADER.size == PACKET_HEADER_BYTES
assert PACKET_FOOTER.size == PACKET_FOOTER_BYTES


class G17GeneralizedXIP2AError(ValueError):
    """A source, packet, transport, or decoded chronology failed closed."""


class G17ConditionalSemanticModeV1(StrEnum):
    PASS_PREDICTOR_V1 = "PASS_PREDICTOR_V1"
    SELECTIVE_ROW3_TACG1C_V1 = "SELECTIVE_ROW3_TACG1C_V1"
    EXACT_SEMANTIC_DIAGNOSTIC_V1 = "EXACT_SEMANTIC_DIAGNOSTIC_V1"


class G17ConditionalG8ModeV1(StrEnum):
    NONE_V1 = "NONE_V1"
    FRESH_POST_TOPOLOGY_G8_V1 = "FRESH_POST_TOPOLOGY_G8_V1"
    MIXED_SHARDED_V1 = "MIXED_SHARDED_V1"


class G17RealizationExtensionV1(StrEnum):
    NONE_V1 = "NONE_V1"


class G17GeneralizedAModeV1(IntEnum):
    PASS_P0 = 0
    GLOBAL_COPY_FINAL_Y1 = 1
    XIP2_WARP = 2


class G17GeneralizedAInterpretationV1(IntEnum):
    CAMERA_THEN_R = 0
    SCORER_THEN_FACTOR2 = 1


class G17GeneralizedAGeometryProfileV1(IntEnum):
    EON_GROUND_874X1164_TO_384X512_V1 = 1


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return _sha256(memoryview(np.ascontiguousarray(value)).cast("B"))


def _immutable(value: np.ndarray, *, dtype: np.dtype[Any], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != dtype:
        raise G17GeneralizedXIP2AError(f"{name} dtype changed")
    copied = np.ascontiguousarray(array).copy()
    copied.setflags(write=False)
    return copied


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
        raise G17GeneralizedXIP2AError("binding is not finite canonical ASCII JSON") from exc


def _strict_json(payload: bytes, *, expected_fields: set[str], schema: str) -> dict[str, Any]:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise G17GeneralizedXIP2AError(f"binding repeats JSON key {key!r}")
            result[key] = value
        return result

    if type(payload) is not bytes or not payload:
        raise G17GeneralizedXIP2AError("binding must be nonempty exact bytes")
    try:
        value = json.loads(payload.decode("ascii"), object_pairs_hook=unique_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G17GeneralizedXIP2AError("binding is not strict ASCII JSON") from exc
    if type(value) is not dict or set(value) != expected_fields or value.get("schema") != schema:
        raise G17GeneralizedXIP2AError("binding schema or field set changed")
    if _canonical_json(value) != payload:
        raise G17GeneralizedXIP2AError("binding changed on canonical parse/re-emit")
    return value


def _pair_ids(value: object) -> tuple[int, ...]:
    if type(value) is not tuple or not value or any(type(item) is not int for item in value):
        raise G17GeneralizedXIP2AError("source pair IDs must be a nonempty exact tuple")
    if value != tuple(range(value[0], value[0] + len(value))) or value[0] < 0 or value[-1] >= 600:
        raise G17GeneralizedXIP2AError("source pair IDs must be contiguous inside [0,600)")
    return value


def _finite_f32(value: object, *, positive: bool) -> float:
    if type(value) not in {float, np.float32}:
        raise G17GeneralizedXIP2AError("pitch must be an exact floating value")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise G17GeneralizedXIP2AError("pitch must be finite and positive for active XIP2")
    packed = struct.pack(">f", result)
    roundtrip = struct.unpack(">f", packed)[0]
    if struct.pack(">f", roundtrip) != packed:
        raise G17GeneralizedXIP2AError("pitch is not canonical IEEE-754 binary32")
    return roundtrip


@dataclass(frozen=True, slots=True)
class G17ConditionalY1SourceBindingV1:
    schema: str
    source_pair_ids: tuple[int, ...]
    semantic_mode: str
    g8_mode: str
    realization_extension: str
    predictor_state_binding_sha256: str
    predictor_semantic_binding_sha256: str
    predictor_program_sha256: str
    predictor_renderer_sha256: str
    predictor_surface_binding_sha256: str
    upstream_decode_receipt_sha256: str
    predictor_labels_sha256: str
    predictor_camera_p0_sha256: str
    predictor_camera_p1_sha256: str
    p_section_sha256: str
    g_section_sha256: str
    post_topology_population_receipt_sha256: str
    semantic_labels_sha256: str
    post_topology_camera_y1_sha256: str
    fresh_g8_population_receipt_sha256: str | None
    post_g8_camera_y1_sha256: str
    realization_extension_population_receipt_sha256: str | None
    final_conditional_camera_y1_sha256: str
    causal_p_receipt_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            item.name: list(value) if item.name == "source_pair_ids" else value
            for item in fields(self)
            if (value := getattr(self, item.name)) is not None or item.name.endswith("receipt_sha256")
        }

    def to_bytes(self) -> bytes:
        return _canonical_json(self.as_dict())

    @property
    def binding_sha256(self) -> str:
        return _sha256(self.to_bytes())


def parse_g17_conditional_y1_source_binding(payload: bytes) -> G17ConditionalY1SourceBindingV1:
    expected = {item.name for item in fields(G17ConditionalY1SourceBindingV1)}
    value = _strict_json(payload, expected_fields=expected, schema=SOURCE_SCHEMA)
    value["source_pair_ids"] = tuple(value["source_pair_ids"])
    try:
        binding = G17ConditionalY1SourceBindingV1(**value)
    except (TypeError, ValueError) as exc:
        raise G17GeneralizedXIP2AError("conditional source contains invalid typed fields") from exc
    _pair_ids(binding.source_pair_ids)
    if binding.semantic_mode not in {item.value for item in G17ConditionalSemanticModeV1}:
        raise G17GeneralizedXIP2AError("conditional source semantic mode escaped its closed set")
    if binding.g8_mode not in {item.value for item in G17ConditionalG8ModeV1}:
        raise G17GeneralizedXIP2AError("conditional source G8 mode escaped its closed set")
    if binding.realization_extension != G17RealizationExtensionV1.NONE_V1.value:
        raise G17GeneralizedXIP2AError("V1 realization extension must be NONE")
    if binding.g8_mode == G17ConditionalG8ModeV1.NONE_V1.value:
        if binding.fresh_g8_population_receipt_sha256 is not None:
            raise G17GeneralizedXIP2AError("NONE G8 source cannot carry a fresh-G8 receipt")
        if binding.post_g8_camera_y1_sha256 != binding.post_topology_camera_y1_sha256:
            raise G17GeneralizedXIP2AError("NONE G8 source changed Y1")
    elif binding.fresh_g8_population_receipt_sha256 is None:
        raise G17GeneralizedXIP2AError("active/mixed G8 source omitted its receipt")
    if binding.realization_extension_population_receipt_sha256 is not None:
        raise G17GeneralizedXIP2AError("NONE realization extension cannot carry a receipt")
    if binding.final_conditional_camera_y1_sha256 != binding.post_g8_camera_y1_sha256:
        raise G17GeneralizedXIP2AError("V1 final conditional Y1 must equal post-G8 Y1")
    for item in fields(binding):
        value = getattr(binding, item.name)
        if (
            item.name.endswith("_sha256")
            and value is not None
            and (type(value) is not str or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value))
        ):
            raise G17GeneralizedXIP2AError(f"{item.name} is not canonical SHA-256")
    if binding.to_bytes() != payload:
        raise G17GeneralizedXIP2AError("conditional source changed on typed parse/re-emit")
    return binding


@dataclass(frozen=True, slots=True)
class G17ConditionalY1SurfaceV1:
    """Exact receiver-owned pre-A population surface, including retained bytes."""

    source_pair_ids: tuple[int, ...]
    semantic_mode: G17ConditionalSemanticModeV1
    g8_mode: G17ConditionalG8ModeV1
    p_section_bytes: bytes = field(repr=False)
    g_section_bytes: bytes = field(repr=False)
    predictor_state_bytes: bytes = field(repr=False)
    predictor_semantic_bytes: bytes = field(repr=False)
    predictor_program_bytes: bytes = field(repr=False)
    predictor_renderer_bytes: bytes = field(repr=False)
    predictor_surface_bytes: bytes = field(repr=False)
    upstream_decode_receipt_bytes: bytes = field(repr=False)
    causal_p_receipt_bytes: bytes = field(repr=False)
    post_topology_population_receipt_bytes: bytes = field(repr=False)
    fresh_g8_population_receipt_bytes: bytes | None = field(repr=False)
    predictor_labels: np.ndarray = field(repr=False)
    predictor_camera_p0: np.ndarray = field(repr=False)
    predictor_camera_p1: np.ndarray = field(repr=False)
    semantic_labels: np.ndarray = field(repr=False)
    post_topology_camera_y1: np.ndarray = field(repr=False)
    post_g8_camera_y1: np.ndarray = field(repr=False)
    final_conditional_camera_y1: np.ndarray = field(repr=False)
    g_active_parser: object | None = field(default=None, repr=False, compare=False)
    source_binding: G17ConditionalY1SourceBindingV1 = field(init=False)

    def __post_init__(self) -> None:
        pair_ids = _pair_ids(self.source_pair_ids)
        if type(self.semantic_mode) is not G17ConditionalSemanticModeV1:
            raise G17GeneralizedXIP2AError("surface semantic mode is not typed")
        if type(self.g8_mode) is not G17ConditionalG8ModeV1:
            raise G17GeneralizedXIP2AError("surface G8 mode is not typed")
        byte_names = (
            "p_section_bytes",
            "g_section_bytes",
            "predictor_state_bytes",
            "predictor_semantic_bytes",
            "predictor_program_bytes",
            "predictor_renderer_bytes",
            "predictor_surface_bytes",
            "upstream_decode_receipt_bytes",
            "causal_p_receipt_bytes",
            "post_topology_population_receipt_bytes",
        )
        for name in byte_names:
            if type(getattr(self, name)) is not bytes or not getattr(self, name):
                raise G17GeneralizedXIP2AError(f"{name} must retain nonempty exact bytes")
        arrays = {
            "predictor_labels": _immutable(self.predictor_labels, dtype=np.dtype(np.uint8), name="predictor_labels"),
            "predictor_camera_p0": _immutable(
                self.predictor_camera_p0,
                dtype=np.dtype(np.uint8),
                name="predictor_camera_p0",
            ),
            "predictor_camera_p1": _immutable(
                self.predictor_camera_p1,
                dtype=np.dtype(np.uint8),
                name="predictor_camera_p1",
            ),
            "semantic_labels": _immutable(self.semantic_labels, dtype=np.dtype(np.uint8), name="semantic_labels"),
            "post_topology_camera_y1": _immutable(
                self.post_topology_camera_y1,
                dtype=np.dtype(np.uint8),
                name="post_topology_camera_y1",
            ),
            "post_g8_camera_y1": _immutable(
                self.post_g8_camera_y1,
                dtype=np.dtype(np.uint8),
                name="post_g8_camera_y1",
            ),
            "final_conditional_camera_y1": _immutable(
                self.final_conditional_camera_y1,
                dtype=np.dtype(np.uint8),
                name="final_conditional_camera_y1",
            ),
        }
        camera_shape = (len(pair_ids), CAMERA_HEIGHT, CAMERA_WIDTH, CHANNELS)
        for name in (
            "predictor_camera_p0",
            "predictor_camera_p1",
            "post_topology_camera_y1",
            "post_g8_camera_y1",
            "final_conditional_camera_y1",
        ):
            if arrays[name].shape != camera_shape:
                raise G17GeneralizedXIP2AError(f"{name} changed the production camera ABI")
        label_shape = (len(pair_ids), SCORER_HEIGHT, SCORER_WIDTH)
        if arrays["predictor_labels"].shape != label_shape or arrays["semantic_labels"].shape != label_shape:
            raise G17GeneralizedXIP2AError("semantic label arrays changed the scorer-plane ABI")
        for name, value in arrays.items():
            object.__setattr__(self, name, value)
        from tac.witness_dsl.taskspace_g17_production_envelope import (
            parse_g17_g_section,
            parse_g17_post_g8_population_receipt,
            parse_g17_post_topology_population_receipt,
        )

        if self.g_active_parser is not None and not callable(self.g_active_parser):
            raise G17GeneralizedXIP2AError("G active parser must be callable when provided")
        parsed_g = parse_g17_g_section(
            self.g_section_bytes,
            expected_p_section=self.p_section_bytes,
            active_parser=self.g_active_parser,
        )
        expected_semantic = parsed_g.semantic_summary.generalized_source_value
        expected_g8 = parsed_g.g8_summary.generalized_source_value
        if self.semantic_mode.value != expected_semantic or self.g8_mode.value != expected_g8:
            raise G17GeneralizedXIP2AError("surface modes differ from reopened G descriptor summaries")
        post_top = parse_g17_post_topology_population_receipt(
            self.post_topology_population_receipt_bytes,
            p_section_bytes=self.p_section_bytes,
            g_section_bytes=self.g_section_bytes,
            causal_p_receipt_bytes=self.causal_p_receipt_bytes,
            predictor_state_binding_sha256=_sha256(self.predictor_state_bytes),
            semantic_labels=arrays["semantic_labels"],
            post_topology_camera_y1=arrays["post_topology_camera_y1"],
            g_active_parser=self.g_active_parser,
        )
        fresh_hash: str | None = None
        if self.g8_mode is G17ConditionalG8ModeV1.NONE_V1:
            if self.fresh_g8_population_receipt_bytes is not None:
                raise G17GeneralizedXIP2AError("NONE G8 surface retained a fresh-G8 receipt")
            if not np.array_equal(arrays["post_g8_camera_y1"], arrays["post_topology_camera_y1"]):
                raise G17GeneralizedXIP2AError("NONE G8 surface changed post-topology Y1")
        else:
            if type(self.fresh_g8_population_receipt_bytes) is not bytes:
                raise G17GeneralizedXIP2AError("active/mixed G8 surface omitted exact receipt bytes")
            parsed_post_g8 = parse_g17_post_g8_population_receipt(
                self.fresh_g8_population_receipt_bytes,
                post_topology_receipt=post_top,
                p_section_bytes=self.p_section_bytes,
                g_section_bytes=self.g_section_bytes,
                semantic_labels=arrays["semantic_labels"],
                post_g8_camera_y1=arrays["post_g8_camera_y1"],
                g_active_parser=self.g_active_parser,
            )
            fresh_hash = parsed_post_g8.receipt_sha256
        if not np.array_equal(arrays["final_conditional_camera_y1"], arrays["post_g8_camera_y1"]):
            raise G17GeneralizedXIP2AError("V1 final conditional Y1 must equal actual post-G8 Y1")
        source = G17ConditionalY1SourceBindingV1(
            schema=SOURCE_SCHEMA,
            source_pair_ids=pair_ids,
            semantic_mode=self.semantic_mode.value,
            g8_mode=self.g8_mode.value,
            realization_extension=G17RealizationExtensionV1.NONE_V1.value,
            predictor_state_binding_sha256=_sha256(self.predictor_state_bytes),
            predictor_semantic_binding_sha256=_sha256(self.predictor_semantic_bytes),
            predictor_program_sha256=_sha256(self.predictor_program_bytes),
            predictor_renderer_sha256=_sha256(self.predictor_renderer_bytes),
            predictor_surface_binding_sha256=_sha256(self.predictor_surface_bytes),
            upstream_decode_receipt_sha256=_sha256(self.upstream_decode_receipt_bytes),
            predictor_labels_sha256=_array_sha256(arrays["predictor_labels"]),
            predictor_camera_p0_sha256=_array_sha256(arrays["predictor_camera_p0"]),
            predictor_camera_p1_sha256=_array_sha256(arrays["predictor_camera_p1"]),
            p_section_sha256=_sha256(self.p_section_bytes),
            g_section_sha256=_sha256(self.g_section_bytes),
            post_topology_population_receipt_sha256=post_top.receipt_sha256,
            semantic_labels_sha256=_array_sha256(arrays["semantic_labels"]),
            post_topology_camera_y1_sha256=_array_sha256(arrays["post_topology_camera_y1"]),
            fresh_g8_population_receipt_sha256=fresh_hash,
            post_g8_camera_y1_sha256=_array_sha256(arrays["post_g8_camera_y1"]),
            realization_extension_population_receipt_sha256=None,
            final_conditional_camera_y1_sha256=_array_sha256(arrays["final_conditional_camera_y1"]),
            causal_p_receipt_sha256=_sha256(self.causal_p_receipt_bytes),
        )
        if parse_g17_conditional_y1_source_binding(source.to_bytes()) != source:
            raise G17GeneralizedXIP2AError("surface source binding failed strict closure")
        object.__setattr__(self, "source_binding", source)


@dataclass(frozen=True, slots=True)
class G17ConditionalY1AcquisitionCustodyV1:
    surface: G17ConditionalY1SurfaceV1 = field(repr=False)
    g_descriptor_acquisition_custody_bytes: bytes = field(repr=False)
    target_observation: G17TargetForwardObservationV1 = field(repr=False)
    post_topology_observation: G17CandidateForwardObservationV1 = field(repr=False)
    post_g8_observation: G17CandidateForwardObservationV1 | None = field(repr=False)
    guidance_payload_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.surface) is not G17ConditionalY1SurfaceV1:
            raise G17GeneralizedXIP2AError("acquisition custody requires exact conditional surface")
        for name in ("g_descriptor_acquisition_custody_bytes", "guidance_payload_bytes"):
            if type(getattr(self, name)) is not bytes or not getattr(self, name):
                raise G17GeneralizedXIP2AError(f"{name} must retain exact nonempty bytes")
        if self.post_topology_observation.target is not self.target_observation:
            raise G17GeneralizedXIP2AError("post-topology observation targets another object")
        if self.surface.g8_mode is G17ConditionalG8ModeV1.NONE_V1:
            if self.post_g8_observation is not None:
                raise G17GeneralizedXIP2AError("NONE G8 custody cannot carry post-G8 scorer evidence")
        elif self.post_g8_observation is None or self.post_g8_observation.target is not self.target_observation:
            raise G17GeneralizedXIP2AError("active/mixed G8 custody requires same-target post-G8 evidence")

    def to_bytes(self) -> bytes:
        value = {
            "schema": ACQUISITION_SCHEMA,
            "g17_conditional_source_sha256": self.surface.source_binding.binding_sha256,
            "g_descriptor_acquisition_custody_sha256": _sha256(self.g_descriptor_acquisition_custody_bytes),
            "target_forward_receipt_sha256": self.target_observation.receipt.receipt_sha256,
            "post_topology_forward_observation_sha256": self.post_topology_observation.receipt.receipt_sha256,
            "post_g8_forward_observation_sha256": None
            if self.post_g8_observation is None
            else self.post_g8_observation.receipt.receipt_sha256,
            "frozen_scorer_sha256": self.target_observation.receipt.frozen_scorer_sha256,
            "scorer_runtime_environment_sha256": self.target_observation.receipt.scorer_runtime_environment_sha256,
            "guidance_payload_sha256": _sha256(self.guidance_payload_bytes),
        }
        return _canonical_json(value)

    @property
    def custody_sha256(self) -> str:
        return _sha256(self.to_bytes())


@dataclass(frozen=True, slots=True)
class G17GeneralizedXIP2ProgramV1:
    mode: G17GeneralizedAModeV1
    interpretation: G17GeneralizedAInterpretationV1 = G17GeneralizedAInterpretationV1.CAMERA_THEN_R
    geometry_profile: G17GeneralizedAGeometryProfileV1 = (
        G17GeneralizedAGeometryProfileV1.EON_GROUND_874X1164_TO_384X512_V1
    )
    pitch: float = 0.0
    xip2_body: bytes = b""
    encoder_guidance_binding_sha256: str = ABSENT_DIGEST

    def __post_init__(self) -> None:
        if type(self.mode) is not G17GeneralizedAModeV1:
            raise G17GeneralizedXIP2AError("program mode is not exact G17 enum")
        if type(self.interpretation) is not G17GeneralizedAInterpretationV1:
            raise G17GeneralizedXIP2AError("program interpretation is not exact G17 enum")
        if type(self.geometry_profile) is not G17GeneralizedAGeometryProfileV1:
            raise G17GeneralizedXIP2AError("program geometry is not exact G17 enum")
        if type(self.xip2_body) is not bytes or len(self.xip2_body) > MAX_XIP2_BODY_BYTES:
            raise G17GeneralizedXIP2AError("program XIP2 body escaped its bounded byte ABI")
        if self.mode in {G17GeneralizedAModeV1.PASS_P0, G17GeneralizedAModeV1.GLOBAL_COPY_FINAL_Y1}:
            if (
                self.interpretation is not G17GeneralizedAInterpretationV1.CAMERA_THEN_R
                or struct.pack(">f", float(self.pitch)) != _POSITIVE_ZERO_F32
                or self.xip2_body
                or self.encoder_guidance_binding_sha256 != ABSENT_DIGEST
            ):
                raise G17GeneralizedXIP2AError("PASS/global-copy require canonical empty-body encoding")
        else:
            _finite_f32(self.pitch, positive=True)
            if not self.xip2_body.startswith(b"XIP2") or self.encoder_guidance_binding_sha256 == ABSENT_DIGEST:
                raise G17GeneralizedXIP2AError("active XIP2 requires exact body and nonzero guidance custody")


@dataclass(frozen=True, slots=True)
class ParsedG17GeneralizedXIP2PacketV1:
    packet: bytes
    program: G17GeneralizedXIP2ProgramV1
    source_pair_ids: tuple[int, ...]
    source_binding_sha256: str
    body_crc32: int
    packet_crc32: int
    transport: SE3XiTransportV2 | None

    @property
    def packet_sha256(self) -> str:
        return _sha256(self.packet)


def build_g17_generalized_xip2_packet(
    *,
    surface: G17ConditionalY1SurfaceV1,
    program: G17GeneralizedXIP2ProgramV1,
) -> bytes:
    if type(surface) is not G17ConditionalY1SurfaceV1 or type(program) is not G17GeneralizedXIP2ProgramV1:
        raise G17GeneralizedXIP2AError("packet build requires exact surface and program types")
    body_crc = zlib.crc32(program.xip2_body) & 0xFFFFFFFF
    header = PACKET_HEADER.pack(
        PACKET_MAGIC,
        PACKET_VERSION,
        1,
        int(program.mode),
        int(program.interpretation),
        int(program.geometry_profile),
        surface.source_pair_ids[0],
        len(surface.source_pair_ids),
        CAMERA_HEIGHT,
        CAMERA_WIDTH,
        SCORER_HEIGHT,
        SCORER_WIDTH,
        float(program.pitch),
        bytes.fromhex(surface.source_binding.binding_sha256),
        bytes.fromhex(program.encoder_guidance_binding_sha256),
        len(program.xip2_body),
        body_crc,
    )
    without_footer = header + program.xip2_body
    packet = without_footer + PACKET_FOOTER.pack(zlib.crc32(without_footer) & 0xFFFFFFFF)
    parsed = parse_g17_generalized_xip2_packet(packet, expected_surface=surface)
    if parsed.packet != packet:
        raise G17GeneralizedXIP2AError("built packet changed on strict parse")
    return packet


def parse_g17_generalized_xip2_packet(
    packet: bytes,
    *,
    expected_surface: G17ConditionalY1SurfaceV1,
) -> ParsedG17GeneralizedXIP2PacketV1:
    if type(packet) is not bytes or len(packet) < PACKET_HEADER.size + PACKET_FOOTER.size:
        raise G17GeneralizedXIP2AError("packet is truncated")
    if type(expected_surface) is not G17ConditionalY1SurfaceV1:
        raise G17GeneralizedXIP2AError("packet parse requires exact conditional surface")
    try:
        unpacked = PACKET_HEADER.unpack_from(packet)
    except struct.error as exc:
        raise G17GeneralizedXIP2AError("packet header is malformed") from exc
    (
        magic,
        version,
        source_domain,
        mode_wire,
        interpretation_wire,
        geometry_wire,
        pair_start,
        pair_count,
        camera_h,
        camera_w,
        scorer_h,
        scorer_w,
        pitch,
        source_digest,
        guidance_digest,
        body_bytes,
        body_crc,
    ) = unpacked
    if magic != PACKET_MAGIC or version != PACKET_VERSION or source_domain != 1:
        raise G17GeneralizedXIP2AError("packet magic/version/source domain changed")
    try:
        mode = G17GeneralizedAModeV1(mode_wire)
        interpretation = G17GeneralizedAInterpretationV1(interpretation_wire)
        geometry = G17GeneralizedAGeometryProfileV1(geometry_wire)
    except ValueError as exc:
        raise G17GeneralizedXIP2AError("packet contains an unknown discriminator") from exc
    if (camera_h, camera_w, scorer_h, scorer_w) != (CAMERA_HEIGHT, CAMERA_WIDTH, SCORER_HEIGHT, SCORER_WIDTH):
        raise G17GeneralizedXIP2AError("packet geometry dimensions changed")
    pair_ids = tuple(range(pair_start, pair_start + pair_count))
    if pair_ids != expected_surface.source_pair_ids:
        raise G17GeneralizedXIP2AError("packet source window differs from exact surface")
    if source_digest.hex() != expected_surface.source_binding.binding_sha256:
        raise G17GeneralizedXIP2AError("packet source binding differs from exact surface")
    if len(packet) != PACKET_HEADER.size + body_bytes + PACKET_FOOTER.size or body_bytes > MAX_XIP2_BODY_BYTES:
        raise G17GeneralizedXIP2AError("packet length/body is not exact EOF")
    body = packet[PACKET_HEADER.size : -PACKET_FOOTER.size]
    if zlib.crc32(body) & 0xFFFFFFFF != body_crc:
        raise G17GeneralizedXIP2AError("packet body CRC changed")
    (packet_crc,) = PACKET_FOOTER.unpack_from(packet, len(packet) - PACKET_FOOTER.size)
    if zlib.crc32(packet[: -PACKET_FOOTER.size]) & 0xFFFFFFFF != packet_crc:
        raise G17GeneralizedXIP2AError("packet CRC changed")
    program = G17GeneralizedXIP2ProgramV1(
        mode=mode,
        interpretation=interpretation,
        geometry_profile=geometry,
        pitch=pitch,
        xip2_body=body,
        encoder_guidance_binding_sha256=guidance_digest.hex(),
    )
    transport: SE3XiTransportV2 | None = None
    if mode is G17GeneralizedAModeV1.XIP2_WARP:
        try:
            transport = SE3XiTransportV2(
                counted_payload=body,
                source_pair_ids=pair_ids,
                predictor_program_sha256=_sha256(expected_surface.predictor_program_bytes),
            )
        except TaskspacePredictorStateV2Error as exc:
            raise G17GeneralizedXIP2AError("active body failed frozen SE3XiTransportV2 parse") from exc
    canonical = build_g17_generalized_xip2_packet_unchecked(surface=expected_surface, program=program)
    if canonical != packet:
        raise G17GeneralizedXIP2AError("packet changed on strict parse/re-encode")
    return ParsedG17GeneralizedXIP2PacketV1(
        packet=packet,
        program=program,
        source_pair_ids=pair_ids,
        source_binding_sha256=source_digest.hex(),
        body_crc32=body_crc,
        packet_crc32=packet_crc,
        transport=transport,
    )


def build_g17_generalized_xip2_packet_unchecked(
    *, surface: G17ConditionalY1SurfaceV1, program: G17GeneralizedXIP2ProgramV1
) -> bytes:
    body_crc = zlib.crc32(program.xip2_body) & 0xFFFFFFFF
    header = PACKET_HEADER.pack(
        PACKET_MAGIC,
        PACKET_VERSION,
        1,
        int(program.mode),
        int(program.interpretation),
        int(program.geometry_profile),
        surface.source_pair_ids[0],
        len(surface.source_pair_ids),
        CAMERA_HEIGHT,
        CAMERA_WIDTH,
        SCORER_HEIGHT,
        SCORER_WIDTH,
        float(program.pitch),
        bytes.fromhex(surface.source_binding.binding_sha256),
        bytes.fromhex(program.encoder_guidance_binding_sha256),
        len(program.xip2_body),
        body_crc,
    )
    without_footer = header + program.xip2_body
    return without_footer + PACKET_FOOTER.pack(zlib.crc32(without_footer) & 0xFFFFFFFF)


@lru_cache(maxsize=1)
def _resize_operator() -> DisjointResizeOperator:
    return DisjointResizeOperator.eon((CAMERA_HEIGHT, CAMERA_WIDTH), (SCORER_HEIGHT, SCORER_WIDTH))


@dataclass(frozen=True, slots=True)
class DecodedG17GeneralizedXIP2V1:
    parsed: ParsedG17GeneralizedXIP2PacketV1
    camera_y0: np.ndarray = field(repr=False)
    camera_y1: np.ndarray = field(repr=False)
    chronological_camera_frames: np.ndarray = field(repr=False)
    receipt_bytes: bytes = field(repr=False)

    @property
    def decoded_output_sha256(self) -> str:
        return _array_sha256(self.chronological_camera_frames)


def _decode_once(packet: bytes, *, surface: G17ConditionalY1SurfaceV1) -> DecodedG17GeneralizedXIP2V1:
    parsed = parse_g17_generalized_xip2_packet(packet, expected_surface=surface)
    y1 = np.ascontiguousarray(surface.final_conditional_camera_y1).copy()
    if parsed.program.mode is G17GeneralizedAModeV1.PASS_P0:
        y0 = np.ascontiguousarray(surface.predictor_camera_p0).copy()
    elif parsed.program.mode is G17GeneralizedAModeV1.GLOBAL_COPY_FINAL_Y1:
        y0 = y1.copy()
    else:
        assert parsed.transport is not None
        output: list[np.ndarray] = []
        if parsed.program.interpretation is G17GeneralizedAInterpretationV1.CAMERA_THEN_R:
            geometry = GroundHomographyGeom.eon((CAMERA_HEIGHT, CAMERA_WIDTH), pitch=parsed.program.pitch)
            for pair_index, xi in enumerate(parsed.transport.xi):
                warped = warp_frame0_native_numpy(y1[pair_index], xi, geometry, compute_dtype=np.float32)
                output.append(np.clip(np.round(warped), 0.0, 255.0).astype(np.uint8))
        else:
            operator = _resize_operator()
            geometry = GroundHomographyGeom.eon((SCORER_HEIGHT, SCORER_WIDTH), pitch=parsed.program.pitch)
            for pair_index, xi in enumerate(parsed.transport.xi):
                scorer_y1 = operator.apply(y1[pair_index])
                warped = warp_frame0_native_numpy(scorer_y1, xi, geometry, compute_dtype=np.float32)
                scorer_u8 = np.clip(np.round(warped), 0.0, 255.0).astype(np.uint8)
                output.append(operator.realize_factor2_uint8(scorer_u8))
        y0 = np.stack(output, axis=0)
    chronology = np.stack((y0, y1), axis=1)
    receipt = _canonical_json(
        {
            "schema": RECEIPT_SCHEMA,
            "packet_sha256": parsed.packet_sha256,
            "source_binding_sha256": parsed.source_binding_sha256,
            "source_pair_ids": list(parsed.source_pair_ids),
            "mode": parsed.program.mode.name,
            "interpretation": parsed.program.interpretation.name,
            "output_camera_y0_sha256": _array_sha256(y0),
            "output_camera_y1_sha256": _array_sha256(y1),
            "output_chronology_sha256": _array_sha256(chronology),
            "y1_byte_identical": _array_sha256(y1) == surface.source_binding.final_conditional_camera_y1_sha256,
            "research_only": True,
            "candidate_claim": False,
            "score_claim": False,
        }
    )
    return DecodedG17GeneralizedXIP2V1(
        parsed=parsed,
        camera_y0=_immutable(y0, dtype=np.dtype(np.uint8), name="decoded Y0"),
        camera_y1=_immutable(y1, dtype=np.dtype(np.uint8), name="decoded Y1"),
        chronological_camera_frames=_immutable(
            chronology,
            dtype=np.dtype(np.uint8),
            name="decoded chronology",
        ),
        receipt_bytes=receipt,
    )


def decode_g17_generalized_xip2_packet(
    packet: bytes,
    *,
    surface: G17ConditionalY1SurfaceV1,
) -> DecodedG17GeneralizedXIP2V1:
    """Execute the deterministic generic A receiver twice."""

    first = _decode_once(packet, surface=surface)
    second = _decode_once(packet, surface=surface)
    if (
        first.receipt_bytes != second.receipt_bytes
        or not np.array_equal(first.camera_y0, second.camera_y0)
        or not np.array_equal(first.camera_y1, second.camera_y1)
        or not np.array_equal(first.chronological_camera_frames, second.chronological_camera_frames)
    ):
        raise G17GeneralizedXIP2AError("generalized A receiver failed double-decode equality")
    if not np.array_equal(first.camera_y1, surface.final_conditional_camera_y1):
        raise G17GeneralizedXIP2AError("A mode changed final conditional Y1")
    return first


__all__ = [
    "ABSENT_DIGEST",
    "ACQUISITION_SCHEMA",
    "PACKET_FOOTER",
    "PACKET_FOOTER_BYTES",
    "PACKET_HEADER",
    "PACKET_HEADER_BYTES",
    "PACKET_MAGIC",
    "SOURCE_SCHEMA",
    "DecodedG17GeneralizedXIP2V1",
    "G17ConditionalG8ModeV1",
    "G17ConditionalSemanticModeV1",
    "G17ConditionalY1AcquisitionCustodyV1",
    "G17ConditionalY1SourceBindingV1",
    "G17ConditionalY1SurfaceV1",
    "G17GeneralizedAGeometryProfileV1",
    "G17GeneralizedAInterpretationV1",
    "G17GeneralizedAModeV1",
    "G17GeneralizedXIP2AError",
    "G17GeneralizedXIP2ProgramV1",
    "G17RealizationExtensionV1",
    "ParsedG17GeneralizedXIP2PacketV1",
    "build_g17_generalized_xip2_packet",
    "decode_g17_generalized_xip2_packet",
    "parse_g17_conditional_y1_source_binding",
    "parse_g17_generalized_xip2_packet",
]
