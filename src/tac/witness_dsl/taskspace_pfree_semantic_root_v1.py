# SPDX-License-Identifier: MIT
"""Strict P-free counted semantic-root wire and deterministic RGB receiver.

``SemanticRootY1V1`` is a counted receiver section, not a second compiler.
Placement, logical ownership, pair-population identity, obligations, lifecycle,
and proof authority remain exclusively owned by
``taskspace_selected_solution_compiler`` (G17/G21).  The adapter at the bottom
of this module accepts those canonical objects without copying their schemas.

The closed typed section union deliberately cannot carry a V15/P/PVSA raster,
a dense scorer plane, or an untyped opaque residual.  Topology is only an
optional factor in a mandatory scorer-native RGB field.  Every valid packet
contains counted quantized shared-generator tensors, a canonical temporally
coded full-n600 latent stream, and an exclusive typed choice between
latent-derived or explicit post-generator texture/chroma/parallax gauges.
Source-lineage custody is packet-bound G17 encoder-only evidence and is never
charged to the decoder.  Palette-only/direct-label realization is rejected.
An explicit sparse RGB quotient section remains available for the irreducible
joint-descent residue.

This is generic deterministic receiver machinery.  It does not compile source
video, claim evaluator equivalence, provide a public ``inflate.sh`` dispatch,
or establish a score.
"""

from __future__ import annotations

import hashlib
import struct
import zlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Final

import numpy as np

from tac.optimization.uint8_lattice_feasibility import (
    DisjointResizeOperator,
    Factor2ExactVerification,
    realize_factor2_uint8_scorer_plane,
    verify_factor2_uint8_scorer_plane,
)
from tac.witness_dsl.taskspace_selected_solution_compiler import (
    G17AnalyticResidualOwnershipV1,
    G17EncoderOnlyTeacherOracleEvidenceV1,
    G17LearnedResidualOwnershipV1,
    G17LogicalOwnershipKindV1,
    G17LogicalOwnershipV1,
    G17PairPopulationV1,
    G17PopulationSharingV1,
    G17RealizationGaugeV1,
    G17SemanticTopologyV1,
)

MAGIC: Final = b"SRY1V1\x00\x00"
VERSION: Final = 1
PAIR_COUNT_N600: Final = 600
SCORER_H: Final = 384
SCORER_W: Final = 512
SCORER_CHANNELS: Final = 3
MAX_PACKET_BYTES: Final = 2_000_000
MAX_TEMPLATES: Final = 256
MAX_EVENTS: Final = 1_024
MAX_GENERATOR_TENSORS: Final = 24
MAX_RGB_BASIS_ATOMS: Final = 128
MAX_QUOTIENT_ATOMS: Final = 2_048
ALL_ROLE_MASK: Final = 0b1_1111

PALETTE_ONLY_DIRECT_LABEL_BLOCKER: Final = "FEED_AH_PALETTE_ONLY_DIRECT_LABEL_REALIZATION_IS_TRIPLY_DOMINATED"
SOURCE_BACKED_COMPILER_BLOCKER: Final = (
    "FRESH_SOURCE_BACKED_G17_COMPILER_MUST_FIT_TOPOLOGY_RGB_GAUGE_AND_IRREDUCIBLE_QUOTIENT"
)
PUBLIC_RECEIVER_BLOCKER: Final = "PUBLIC_INFLATE_DISPATCH_AND_EXTRACTED_DIRECTORY_PARSE_BACK_OWED"
FINAL_Y1_G94_BINDING_BLOCKER: Final = "G94_V2_MUST_BIND_TO_THE_FROZEN_FINAL_SEMANTIC_ROOT_Y1_POPULATION_SHA"
V9_PHASE_ADVECTION_ADAPTER_BLOCKER: Final = "V9_FILM_TANH_BETA_SIN_PHASE_ADVECTION_TYPED_ABI_AND_REFERENCE_FORWARD_OWED"
V9_Y1_ODD_ROW_PROJECTION_CONTRACT: Final = "COUNT_ONLY_CODE_2P_PLUS_1_Y1_ROWS_DISCARD_EVEN_Y0_ROWS"
G94_V2_SOLE_Y0_OWNER_CONTRACT: Final = "G94_V2_IS_SOLE_CONDITIONAL_Y0_OWNER_AFTER_FINAL_Y1_FREEZE"
EXPLICIT_GAUGE_ARBITRATION_BLOCKER: Final = "EXPLICIT_RGB_GAUGE_OVER_TEMPORAL_LATENT_VALUE_PER_BYTE_ARBITRATION_OWED"
UNSUPPORTED_LEARNED_RECEIVER_BLOCKER: Final = "SEMANTIC_ROOT_LEARNED_GENERATOR_ARCHITECTURE_HAS_NO_REVIEWED_RECEIVER"
OPEN_PRODUCT_BLOCKERS: Final = (
    SOURCE_BACKED_COMPILER_BLOCKER,
    PUBLIC_RECEIVER_BLOCKER,
    FINAL_Y1_G94_BINDING_BLOCKER,
    V9_PHASE_ADVECTION_ADAPTER_BLOCKER,
    EXPLICIT_GAUGE_ARBITRATION_BLOCKER,
)

_HEADER = struct.Struct(">8sBBHHHBBHHHHHHH")
_SECTION_META = struct.Struct(">4sI")
_FOOTER = struct.Struct(">I")
_PROFILE = struct.Struct(">4s15B4BI")
_TEMPLATE = struct.Struct(">HBB6h")
_EVENT = struct.Struct(">5H4h")
_MODEL_HEADER = struct.Struct(">4sBBBBHHHHH")
_TENSOR_HEADER = struct.Struct(">HBBbB4HhI")
_LATENT_HEADER = struct.Struct(">4sBBHHhhI")
_RGB_HEADER = struct.Struct(">4sBBHH")
_RGB_BASIS = struct.Struct(">HBBHHHhhhBB")
_PAIR_GAUGE = struct.Struct(">HH6h")
_QUOTIENT_HEADER = struct.Struct(">4sH")
_QUOTIENT = struct.Struct(">HHHBBhhhhHHhhh")
_LINEAGE = struct.Struct(">4s7s32s32s32s32s32s32s32s32s")

_PROFILE_MAGIC: Final = b"SRP1"
_MODEL_MAGIC: Final = b"SRM1"
_LATENT_MAGIC: Final = b"SRL1"
_RGB_MAGIC: Final = b"SRF1"
_QUOTIENT_MAGIC: Final = b"SRQ1"
_LINEAGE_MAGIC: Final = b"SRN1"

SECTION_TAGS: Final = (
    b"PROF",
    b"TOPO",
    b"EVNT",
    b"MODL",
    b"LATN",
    b"RGBF",
    b"IRRQ",
)

_FORBIDDEN_NESTED_MAGICS: Final = (
    b"PK\x03\x04",
    b"TACPVSA",
    b"DDV15S1",
    b"TACV10R",
    b"PVSA",
    b"TSPPV1",
    b"TSPPV2",
)

# Exact historical identities are deny-listed only as a second line of defense.
# The typed fixed-width sections already make those raster/payload homes
# structurally unrepresentable.
_FORBIDDEN_WHOLE_PAYLOAD_SHA256: Final = frozenset(
    {
        "759e2833f31d2182b80e1b2f434214f24d75cb487bbec554dc58abdc7d53e6bb",
        "b9c8ab2a5e2bf6cb775539156be1220d9f3f6b44fce38a2ecae70164027f512b",
        "d50aac6ea72df527f1630485c174b73ed25c2c7b41b685a24a53ccac21e6cf6c",
        "736d9c751b1578cead45bccb5e71a4bab2373353f079f96d5c6ec96694ae8d95",
        "e6f99e435fcbd45673bebea4049f8b8322d927a2276c37c995056e1ac4bbf4fe",
        "2b82e28e23e3b37fc305dc42f2320ed643726d27fe5e6805bf0978ac0e5c8fa8",
        "e4cd154fbd5540bf176102374c968dd9a07f7bd647108a4f24b28d19fb10dad7",
        "e3d0581f70ac91493ed9897e5e3d49819961477c56cac161a3e577010e683c7e",
    }
)


class SemanticRootY1V1Error(ValueError):
    """A counted semantic-root packet or deterministic decode failed closed."""


class SemanticRoleV1(IntEnum):
    ROAD = 0
    LANE = 1
    UNDRIVABLE_BOUNDARY = 2
    MOVABLE = 3
    MY_CAR = 4


class TopologyShapeV1(IntEnum):
    RECT = 0
    ELLIPSE = 1
    TRAPEZOID = 2
    QUADRATIC_STRIP = 3


class RGBWaveKindV1(IntEnum):
    TRIANGLE = 0
    CHECKER = 1
    RIDGE = 2


class GeneratorArchitectureV1(IntEnum):
    """Closed receiver implementations; new IDs require reviewed decode code."""

    # An original fixed-point MLP family.  It is NOT the V9
    # FiLM-conditioned tanh(beta * sin(w*x)) phase-advection trunk.
    ORIGINAL_COORDINR_FILM_MLP_V1 = 0


class GeneratorNumericContractV1(IntEnum):
    INT8_WEIGHT_INT16_STATE_INT32_ACCUM_Q12 = 0


class GeneratorActivationV1(IntEnum):
    HARD_TANH_Q12 = 0
    RELU6_Q12 = 1


class QuantizedTensorRoleV1(IntEnum):
    INPUT_WEIGHT = 0
    INPUT_BIAS = 1
    HIDDEN_WEIGHT = 2
    HIDDEN_BIAS = 3
    FILM_WEIGHT = 4
    FILM_BIAS = 5
    OUTPUT_WEIGHT = 6
    OUTPUT_BIAS = 7


class QuantizedTensorDTypeV1(IntEnum):
    INT8 = 0
    INT16_BE = 1


class TemporalLatentCodecV1(IntEnum):
    DELTA_RICE_SIGNED_I16_V1 = 0


class RGBGaugeOwnershipV1(IntEnum):
    DERIVED_BY_SHARED_GENERATOR = 0
    EXPLICIT_NONOVERLAPPING_POST_GENERATOR = 1


def _int(
    value: object,
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise SemanticRootY1V1Error(f"{name} must be an exact integer in [{minimum},{maximum}]")
    return value


def _typed_enum(value: object, enum_type: type[IntEnum], name: str) -> IntEnum:
    if type(value) is not enum_type:
        raise SemanticRootY1V1Error(f"{name} is not {enum_type.__name__}")
    return value


def _sha256(payload: bytes | memoryview) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise SemanticRootY1V1Error(f"{name} must be lowercase SHA-256")
    return value


def _round_div_signed_scalar(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise AssertionError("internal denominator must be positive")
    sign = -1 if numerator < 0 else 1
    return sign * ((abs(numerator) + denominator // 2) // denominator)


def _round_div_signed_array(numerator: np.ndarray, denominator: int) -> np.ndarray:
    magnitude = (np.abs(numerator) + denominator // 2) // denominator
    return np.where(numerator < 0, -magnitude, magnitude)


def _tensor_element_count(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


@dataclass(frozen=True, slots=True)
class QuantizedGeneratorTensorV1:
    """One counted tensor with an architecture-constrained semantic role."""

    tensor_id: int
    role: QuantizedTensorRoleV1
    dtype: QuantizedTensorDTypeV1
    shape: tuple[int, ...]
    scale_exponent: int
    zero_point: int
    data: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _int(self.tensor_id, "tensor_id", minimum=0, maximum=0xFFFF)
        _typed_enum(self.role, QuantizedTensorRoleV1, "tensor role")
        _typed_enum(self.dtype, QuantizedTensorDTypeV1, "tensor dtype")
        if type(self.shape) is not tuple or not 1 <= len(self.shape) <= 4:
            raise SemanticRootY1V1Error("tensor shape must have rank 1..4")
        for dimension in self.shape:
            _int(dimension, "tensor dimension", minimum=1, maximum=0xFFFF)
        if self.shape in {
            (SCORER_H, SCORER_W, SCORER_CHANNELS),
            (PAIR_COUNT_N600, SCORER_H, SCORER_W, SCORER_CHANNELS),
        }:
            raise SemanticRootY1V1Error("dense scorer/raster tensor shape is forbidden")
        _int(
            self.scale_exponent,
            "tensor scale_exponent",
            minimum=-32,
            maximum=31,
        )
        _int(self.zero_point, "tensor zero_point", minimum=-0x8000, maximum=0x7FFF)
        if type(self.data) is not bytes:
            raise SemanticRootY1V1Error("tensor data must be exact counted bytes")
        itemsize = 1 if self.dtype is QuantizedTensorDTypeV1.INT8 else 2
        if len(self.data) != _tensor_element_count(self.shape) * itemsize:
            raise SemanticRootY1V1Error("tensor byte length disagrees with shape/dtype")
        if not self.data:
            raise SemanticRootY1V1Error("generator tensor bytes must be nonempty")
        if _sha256(self.data) in _FORBIDDEN_WHOLE_PAYLOAD_SHA256 or any(
            self.data.startswith(magic) for magic in _FORBIDDEN_NESTED_MAGICS
        ):
            raise SemanticRootY1V1Error("generator tensor reuses a foreign/raster payload home")

    @property
    def array(self) -> np.ndarray:
        dtype = np.dtype("i1") if self.dtype is QuantizedTensorDTypeV1.INT8 else np.dtype(">i2")
        array = np.frombuffer(self.data, dtype=dtype).astype(np.int32)
        return array.reshape(self.shape)


@dataclass(frozen=True, slots=True)
class QuantizedSharedGeneratorV1:
    architecture: GeneratorArchitectureV1
    numeric_contract: GeneratorNumericContractV1
    activation: GeneratorActivationV1
    input_dim: int
    hidden_dim: int
    hidden_layer_count: int
    modulation_dim: int
    tensors: tuple[QuantizedGeneratorTensorV1, ...]

    def __post_init__(self) -> None:
        _typed_enum(self.architecture, GeneratorArchitectureV1, "architecture")
        _typed_enum(
            self.numeric_contract,
            GeneratorNumericContractV1,
            "numeric contract",
        )
        _typed_enum(self.activation, GeneratorActivationV1, "activation")
        _int(self.input_dim, "input_dim", minimum=4, maximum=32)
        _int(self.hidden_dim, "hidden_dim", minimum=8, maximum=256)
        _int(
            self.hidden_layer_count,
            "hidden_layer_count",
            minimum=1,
            maximum=8,
        )
        _int(self.modulation_dim, "modulation_dim", minimum=1, maximum=64)
        if type(self.tensors) is not tuple or not self.tensors:
            raise SemanticRootY1V1Error("shared generator tensors must be nonempty")
        if any(type(item) is not QuantizedGeneratorTensorV1 for item in self.tensors):
            raise SemanticRootY1V1Error("shared generator contains an untyped tensor")
        if tuple(item.tensor_id for item in self.tensors) != tuple(range(len(self.tensors))):
            raise SemanticRootY1V1Error("shared generator tensor IDs must be canonical contiguous order")
        if all(all(value == 0 for value in tensor.data) for tensor in self.tensors):
            raise SemanticRootY1V1Error("all-zero shared generator is refused")
        expected_roles_and_shapes = [
            (QuantizedTensorRoleV1.INPUT_WEIGHT, (self.hidden_dim, self.input_dim)),
            (QuantizedTensorRoleV1.INPUT_BIAS, (self.hidden_dim,)),
        ]
        for _ in range(self.hidden_layer_count):
            expected_roles_and_shapes.extend(
                [
                    (
                        QuantizedTensorRoleV1.HIDDEN_WEIGHT,
                        (self.hidden_dim, self.hidden_dim),
                    ),
                    (QuantizedTensorRoleV1.HIDDEN_BIAS, (self.hidden_dim,)),
                ]
            )
        expected_roles_and_shapes.extend(
            [
                (
                    QuantizedTensorRoleV1.FILM_WEIGHT,
                    (2 * self.hidden_dim, self.modulation_dim),
                ),
                (QuantizedTensorRoleV1.FILM_BIAS, (2 * self.hidden_dim,)),
                (
                    QuantizedTensorRoleV1.OUTPUT_WEIGHT,
                    (SCORER_CHANNELS, self.hidden_dim),
                ),
                (QuantizedTensorRoleV1.OUTPUT_BIAS, (SCORER_CHANNELS,)),
            ]
        )
        observed = tuple((item.role, item.shape) for item in self.tensors)
        if observed != tuple(expected_roles_and_shapes):
            raise SemanticRootY1V1Error("shared generator tensor roles/shapes disagree with reviewed CoordINR ABI")
        for tensor in self.tensors:
            expected_dtype = (
                QuantizedTensorDTypeV1.INT8
                if tensor.role
                in {
                    QuantizedTensorRoleV1.INPUT_WEIGHT,
                    QuantizedTensorRoleV1.HIDDEN_WEIGHT,
                    QuantizedTensorRoleV1.FILM_WEIGHT,
                    QuantizedTensorRoleV1.OUTPUT_WEIGHT,
                }
                else QuantizedTensorDTypeV1.INT16_BE
            )
            if tensor.dtype is not expected_dtype:
                raise SemanticRootY1V1Error("shared generator tensor dtype disagrees with numeric contract")
            if (
                tensor.scale_exponent != (-7 if tensor.dtype is QuantizedTensorDTypeV1.INT8 else -12)
                or tensor.zero_point != 0
            ):
                raise SemanticRootY1V1Error("shared generator tensor quantization disagrees with Q12 ABI")


class _BitWriter:
    def __init__(self) -> None:
        self._bytes = bytearray()
        self._current = 0
        self._bits = 0

    def bit(self, value: int) -> None:
        self._current = (self._current << 1) | value
        self._bits += 1
        if self._bits == 8:
            self._bytes.append(self._current)
            self._current = 0
            self._bits = 0

    def finish(self) -> bytes:
        if self._bits:
            self._bytes.append(self._current << (8 - self._bits))
        return bytes(self._bytes)


class _BitReader:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.bit_offset = 0

    def bit(self) -> int:
        if self.bit_offset >= len(self.payload) * 8:
            raise SemanticRootY1V1Error("temporal Rice stream is truncated")
        byte = self.payload[self.bit_offset // 8]
        value = (byte >> (7 - self.bit_offset % 8)) & 1
        self.bit_offset += 1
        return value

    def require_zero_padding(self) -> None:
        while self.bit_offset < len(self.payload) * 8:
            if self.bit() != 0:
                raise SemanticRootY1V1Error("temporal Rice stream has noncanonical trailing bits")


def _rice_encode_signed(values: np.ndarray, rice_k: int) -> bytes:
    writer = _BitWriter()
    previous = np.zeros(values.shape[1], dtype=np.int64)
    for row in values.astype(np.int64, copy=False):
        for column, current in enumerate(row):
            delta = int(current - previous[column])
            previous[column] = current
            unsigned = 2 * delta if delta >= 0 else -2 * delta - 1
            quotient = unsigned >> rice_k
            if quotient > 0xFFFF:
                raise SemanticRootY1V1Error("temporal Rice quotient exceeds closed decoder bound")
            for _ in range(quotient):
                writer.bit(1)
            writer.bit(0)
            for shift in range(rice_k - 1, -1, -1):
                writer.bit((unsigned >> shift) & 1)
    return writer.finish()


def _rice_decode_signed(
    payload: bytes,
    *,
    pair_count: int,
    latent_dim: int,
    rice_k: int,
) -> np.ndarray:
    reader = _BitReader(payload)
    output = np.empty((pair_count, latent_dim), dtype=np.int16)
    previous = np.zeros(latent_dim, dtype=np.int64)
    for pair_id in range(pair_count):
        for column in range(latent_dim):
            quotient = 0
            while reader.bit():
                quotient += 1
                if quotient > 0xFFFF:
                    raise SemanticRootY1V1Error("temporal Rice unary quotient exceeds decoder bound")
            remainder = 0
            for _ in range(rice_k):
                remainder = (remainder << 1) | reader.bit()
            unsigned = (quotient << rice_k) | remainder
            delta = unsigned // 2 if unsigned % 2 == 0 else -(unsigned // 2) - 1
            value = int(previous[column]) + delta
            if not -0x8000 <= value <= 0x7FFF:
                raise SemanticRootY1V1Error("temporal latent delta leaves the int16 range")
            output[pair_id, column] = value
            previous[column] = value
    reader.require_zero_padding()
    output.setflags(write=False)
    return output


@dataclass(frozen=True, slots=True)
class TemporalLatentStreamV1:
    codec: TemporalLatentCodecV1
    rice_k: int
    latent_dim: int
    value_min: int
    value_max: int
    decoded_sha256: str
    encoded_bytes: bytes = field(repr=False)
    pair_count: int = field(default=PAIR_COUNT_N600, init=False)

    def __post_init__(self) -> None:
        _typed_enum(self.codec, TemporalLatentCodecV1, "temporal latent codec")
        _int(self.rice_k, "rice_k", minimum=0, maximum=15)
        _int(self.latent_dim, "latent_dim", minimum=1, maximum=64)
        _int(self.value_min, "latent value_min", minimum=-0x8000, maximum=0x7FFF)
        _int(self.value_max, "latent value_max", minimum=-0x8000, maximum=0x7FFF)
        if self.value_min > self.value_max:
            raise SemanticRootY1V1Error("latent declared range is inverted")
        _require_sha256(self.decoded_sha256, "decoded latent")
        if type(self.encoded_bytes) is not bytes or not self.encoded_bytes:
            raise SemanticRootY1V1Error("temporal latent bytes must be counted/nonempty")
        if _sha256(self.encoded_bytes) in _FORBIDDEN_WHOLE_PAYLOAD_SHA256 or any(
            self.encoded_bytes.startswith(magic) for magic in _FORBIDDEN_NESTED_MAGICS
        ):
            raise SemanticRootY1V1Error("temporal latent stream reuses a foreign/raster payload home")
        decoded = self.decode()
        if int(decoded.min()) != self.value_min or int(decoded.max()) != self.value_max:
            raise SemanticRootY1V1Error("temporal latent declared range disagrees with decoded values")
        if _sha256(memoryview(np.ascontiguousarray(decoded.astype(">i2"))).cast("B")) != self.decoded_sha256:
            raise SemanticRootY1V1Error("temporal latent decoded SHA-256 mismatch")

    @classmethod
    def from_array(
        cls,
        values: np.ndarray,
        *,
        rice_k: int,
    ) -> TemporalLatentStreamV1:
        raw = np.asarray(values)
        if raw.shape[0] != PAIR_COUNT_N600 or raw.ndim != 2:
            raise SemanticRootY1V1Error("temporal latent source must have shape (600, latent_dim)")
        if raw.dtype.kind not in ("i", "u") or np.any(raw < -0x8000) or np.any(raw > 0x7FFF):
            raise SemanticRootY1V1Error("temporal latent source must be exact int16-range integers")
        canonical = raw.astype(np.int16)
        encoded = _rice_encode_signed(canonical, rice_k)
        decoded_bytes = memoryview(np.ascontiguousarray(canonical.astype(">i2"))).cast("B")
        return cls(
            codec=TemporalLatentCodecV1.DELTA_RICE_SIGNED_I16_V1,
            rice_k=rice_k,
            latent_dim=canonical.shape[1],
            value_min=int(canonical.min()),
            value_max=int(canonical.max()),
            decoded_sha256=_sha256(decoded_bytes),
            encoded_bytes=encoded,
        )

    def decode(self) -> np.ndarray:
        return _rice_decode_signed(
            self.encoded_bytes,
            pair_count=PAIR_COUNT_N600,
            latent_dim=self.latent_dim,
            rice_k=self.rice_k,
        )


def temporal_y1_latents_from_interleaved_v9_codes(
    interleaved_codes: np.ndarray,
    *,
    rice_k: int,
) -> TemporalLatentStreamV1:
    """Count only V9 odd/Y1 rows; even/Y0 rows remain encoder-only.

    This projection does not make the original MLP ABI source-compatible with
    V9's dual-head phase-advection decoder.  It only closes chronological
    ownership so a future reviewed adapter cannot duplicate G94-V2's Y0 state.
    """

    raw = np.asarray(interleaved_codes)
    if raw.ndim != 2 or raw.shape[0] != 2 * PAIR_COUNT_N600:
        raise SemanticRootY1V1Error("interleaved producer code must have exact shape (1200, latent_dim)")
    if raw.dtype.kind not in ("i", "u"):
        raise SemanticRootY1V1Error("interleaved producer code must contain exact integers")
    y1_only = np.ascontiguousarray(raw[1::2])
    return TemporalLatentStreamV1.from_array(y1_only, rice_k=rice_k)


@dataclass(frozen=True, slots=True)
class SemanticRootSourceLineageV1:
    """External encoder-only custody; never serialized in candidate bytes."""

    compiler_id: str
    root_packet_sha256: str
    source_video_sha256: str
    target_custody_sha256: str
    compiler_source_sha256: str
    compile_config_sha256: str
    originality_declaration_sha256: str
    model_section_sha256: str
    latent_decoded_sha256: str

    def __post_init__(self) -> None:
        if type(self.compiler_id) is not str or not 1 <= len(self.compiler_id) <= 7 or not self.compiler_id.isascii():
            raise SemanticRootY1V1Error("lineage compiler_id must be 1..7 ASCII characters")
        for name in (
            "root_packet_sha256",
            "source_video_sha256",
            "target_custody_sha256",
            "compiler_source_sha256",
            "compile_config_sha256",
            "originality_declaration_sha256",
            "model_section_sha256",
            "latent_decoded_sha256",
        ):
            _require_sha256(getattr(self, name), f"lineage {name}")


@dataclass(frozen=True, slots=True)
class SemanticRealizationProfileV1:
    """Baseline prototypes plus mandatory RGB-field gain controls."""

    role_rgb: tuple[tuple[int, int, int], ...]
    texture_gain_q4: int
    edge_gain_q4: int
    chroma_gain_q4: int
    parallax_gain_q4: int
    renderer_seed: int

    def __post_init__(self) -> None:
        if type(self.role_rgb) is not tuple or len(self.role_rgb) != len(SemanticRoleV1):
            raise SemanticRootY1V1Error("profile must have exactly five RGB prototypes")
        for index, rgb in enumerate(self.role_rgb):
            if type(rgb) is not tuple or len(rgb) != 3:
                raise SemanticRootY1V1Error(f"profile role_rgb[{index}] is not an RGB tuple")
            for channel in rgb:
                _int(channel, f"profile role_rgb[{index}]", minimum=0, maximum=255)
        if len(set(self.role_rgb)) != len(self.role_rgb):
            raise SemanticRootY1V1Error("role RGB prototypes must be distinct")
        for name in (
            "texture_gain_q4",
            "edge_gain_q4",
            "chroma_gain_q4",
            "parallax_gain_q4",
        ):
            _int(getattr(self, name), name, minimum=1, maximum=31)
        _int(self.renderer_seed, "renderer_seed", minimum=0, maximum=0xFFFFFFFF)


@dataclass(frozen=True, slots=True)
class SemanticTopologyTemplateV1:
    template_id: int
    role: SemanticRoleV1
    shape: TopologyShapeV1
    params_q: tuple[int, int, int, int, int, int]

    def __post_init__(self) -> None:
        _int(self.template_id, "template_id", minimum=0, maximum=0xFFFF)
        _typed_enum(self.role, SemanticRoleV1, "template role")
        _typed_enum(self.shape, TopologyShapeV1, "template shape")
        if type(self.params_q) is not tuple or len(self.params_q) != 6:
            raise SemanticRootY1V1Error("template params_q must contain six int16 values")
        for value in self.params_q:
            _int(value, "template parameter", minimum=-0x8000, maximum=0x7FFF)
        if self.shape in {TopologyShapeV1.RECT, TopologyShapeV1.ELLIPSE}:
            if self.params_q[2] <= 0 or self.params_q[3] <= 0:
                raise SemanticRootY1V1Error("RECT/ELLIPSE template extents must be positive")
        elif self.shape is TopologyShapeV1.TRAPEZOID:
            if self.params_q[2] <= self.params_q[1] or self.params_q[3] <= 0 or self.params_q[4] <= 0:
                raise SemanticRootY1V1Error("TRAPEZOID requires bottom>top and positive half-widths")
        elif self.shape is TopologyShapeV1.QUADRATIC_STRIP and (self.params_q[4] <= 0 or self.params_q[5] <= 0):
            raise SemanticRootY1V1Error("QUADRATIC_STRIP requires positive half-width and y-span")


@dataclass(frozen=True, slots=True)
class SemanticTopologyEventV1:
    event_id: int
    template_id: int
    pair_start: int
    pair_stop: int
    z_order: int
    anchor_x_q4: int
    anchor_y_q4: int
    velocity_x_q8: int
    velocity_y_q8: int

    def __post_init__(self) -> None:
        _int(self.event_id, "event_id", minimum=0, maximum=0xFFFF)
        _int(self.template_id, "event template_id", minimum=0, maximum=0xFFFF)
        _int(self.pair_start, "event pair_start", minimum=0, maximum=PAIR_COUNT_N600 - 1)
        _int(self.pair_stop, "event pair_stop", minimum=1, maximum=PAIR_COUNT_N600)
        if self.pair_start >= self.pair_stop:
            raise SemanticRootY1V1Error("event pair interval must be nonempty")
        _int(self.z_order, "event z_order", minimum=0, maximum=0xFFFF)
        for name in (
            "anchor_x_q4",
            "anchor_y_q4",
            "velocity_x_q8",
            "velocity_y_q8",
        ):
            _int(getattr(self, name), name, minimum=-0x8000, maximum=0x7FFF)


@dataclass(frozen=True, slots=True)
class RGBBasisAtomV1:
    atom_id: int
    role_mask: int
    wave_kind: RGBWaveKindV1
    frequency_x: int
    frequency_y: int
    phase_u16: int
    amplitude_rgb: tuple[int, int, int]
    edge_width: int

    def __post_init__(self) -> None:
        _int(self.atom_id, "RGB atom_id", minimum=0, maximum=0xFFFF)
        _int(self.role_mask, "RGB role_mask", minimum=1, maximum=ALL_ROLE_MASK)
        _typed_enum(self.wave_kind, RGBWaveKindV1, "RGB wave_kind")
        _int(self.frequency_x, "frequency_x", minimum=0, maximum=64)
        _int(self.frequency_y, "frequency_y", minimum=0, maximum=64)
        if self.frequency_x == self.frequency_y == 0:
            raise SemanticRootY1V1Error("RGB basis frequency cannot be DC")
        _int(self.phase_u16, "phase_u16", minimum=0, maximum=0xFFFF)
        if type(self.amplitude_rgb) is not tuple or len(self.amplitude_rgb) != 3:
            raise SemanticRootY1V1Error("RGB amplitude must contain three int16 values")
        for amplitude in self.amplitude_rgb:
            _int(amplitude, "RGB amplitude", minimum=-255, maximum=255)
        if self.amplitude_rgb == (0, 0, 0):
            raise SemanticRootY1V1Error("RGB basis amplitude cannot be zero")
        _int(self.edge_width, "edge_width", minimum=0, maximum=16)


@dataclass(frozen=True, slots=True)
class PairRGBGaugeV1:
    pair_id: int
    phase_u16: int
    parallax_x_q8: int
    parallax_y_q8: int
    luma_bias: int
    chroma_u_bias: int
    chroma_v_bias: int
    texture_gain_q8: int

    def __post_init__(self) -> None:
        _int(self.pair_id, "gauge pair_id", minimum=0, maximum=PAIR_COUNT_N600 - 1)
        _int(self.phase_u16, "gauge phase_u16", minimum=0, maximum=0xFFFF)
        for name in (
            "parallax_x_q8",
            "parallax_y_q8",
            "luma_bias",
            "chroma_u_bias",
            "chroma_v_bias",
        ):
            _int(getattr(self, name), f"gauge {name}", minimum=-255, maximum=255)
        _int(
            self.texture_gain_q8,
            "gauge texture_gain_q8",
            minimum=1,
            maximum=0x7FFF,
        )


@dataclass(frozen=True, slots=True)
class RGBQuotientAtomV1:
    """Sparse scorer-native RGB atom; never an opaque or dense-plane home."""

    atom_id: int
    pair_start: int
    pair_stop: int
    role_mask: int
    edge_only: bool
    center_x_q4: int
    center_y_q4: int
    velocity_x_q8: int
    velocity_y_q8: int
    radius_x_q4: int
    radius_y_q4: int
    amplitude_rgb: tuple[int, int, int]

    def __post_init__(self) -> None:
        _int(self.atom_id, "quotient atom_id", minimum=0, maximum=0xFFFF)
        _int(self.pair_start, "quotient pair_start", minimum=0, maximum=PAIR_COUNT_N600 - 1)
        _int(self.pair_stop, "quotient pair_stop", minimum=1, maximum=PAIR_COUNT_N600)
        if self.pair_start >= self.pair_stop:
            raise SemanticRootY1V1Error("quotient pair interval must be nonempty")
        _int(self.role_mask, "quotient role_mask", minimum=1, maximum=ALL_ROLE_MASK)
        if type(self.edge_only) is not bool:
            raise SemanticRootY1V1Error("quotient edge_only must be bool")
        for name in (
            "center_x_q4",
            "center_y_q4",
            "velocity_x_q8",
            "velocity_y_q8",
        ):
            _int(getattr(self, name), name, minimum=-0x8000, maximum=0x7FFF)
        _int(self.radius_x_q4, "radius_x_q4", minimum=1, maximum=0xFFFF)
        _int(self.radius_y_q4, "radius_y_q4", minimum=1, maximum=0xFFFF)
        if type(self.amplitude_rgb) is not tuple or len(self.amplitude_rgb) != 3:
            raise SemanticRootY1V1Error("quotient amplitude must contain three int16 values")
        for amplitude in self.amplitude_rgb:
            _int(amplitude, "quotient amplitude", minimum=-255, maximum=255)
        if self.amplitude_rgb == (0, 0, 0):
            raise SemanticRootY1V1Error("quotient amplitude cannot be zero")


@dataclass(frozen=True, slots=True)
class SemanticRootY1V1:
    """Typed counted payload only; lifecycle and ownership live in G17/G21."""

    background_role: SemanticRoleV1
    profile: SemanticRealizationProfileV1
    shared_generator: QuantizedSharedGeneratorV1
    temporal_latents: TemporalLatentStreamV1
    rgb_gauge_ownership: RGBGaugeOwnershipV1
    topology_templates: tuple[SemanticTopologyTemplateV1, ...]
    topology_events: tuple[SemanticTopologyEventV1, ...]
    rgb_basis: tuple[RGBBasisAtomV1, ...]
    pair_rgb_gauges: tuple[PairRGBGaugeV1, ...]
    irreducible_rgb_quotient: tuple[RGBQuotientAtomV1, ...] = ()
    pair_count: int = field(default=PAIR_COUNT_N600, init=False)

    def __post_init__(self) -> None:
        _typed_enum(self.background_role, SemanticRoleV1, "background_role")
        if type(self.profile) is not SemanticRealizationProfileV1:
            raise SemanticRootY1V1Error("profile is not SemanticRealizationProfileV1")
        if type(self.shared_generator) is not QuantizedSharedGeneratorV1:
            raise SemanticRootY1V1Error("shared_generator is not QuantizedSharedGeneratorV1")
        if type(self.temporal_latents) is not TemporalLatentStreamV1:
            raise SemanticRootY1V1Error("temporal_latents is not TemporalLatentStreamV1")
        if self.temporal_latents.latent_dim != self.shared_generator.modulation_dim:
            raise SemanticRootY1V1Error("temporal latent width disagrees with generator modulation ABI")
        _typed_enum(
            self.rgb_gauge_ownership,
            RGBGaugeOwnershipV1,
            "RGB gauge ownership",
        )
        expected_gauge_count = (
            0 if self.rgb_gauge_ownership is RGBGaugeOwnershipV1.DERIVED_BY_SHARED_GENERATOR else PAIR_COUNT_N600
        )
        typed_sequences = (
            ("topology_templates", self.topology_templates, SemanticTopologyTemplateV1, 0, MAX_TEMPLATES),
            ("topology_events", self.topology_events, SemanticTopologyEventV1, 0, MAX_EVENTS),
            ("rgb_basis", self.rgb_basis, RGBBasisAtomV1, 0, MAX_RGB_BASIS_ATOMS),
            (
                "pair_rgb_gauges",
                self.pair_rgb_gauges,
                PairRGBGaugeV1,
                expected_gauge_count,
                expected_gauge_count,
            ),
            (
                "irreducible_rgb_quotient",
                self.irreducible_rgb_quotient,
                RGBQuotientAtomV1,
                0,
                MAX_QUOTIENT_ATOMS,
            ),
        )
        for name, values, item_type, minimum, maximum in typed_sequences:
            if type(values) is not tuple or not minimum <= len(values) <= maximum:
                raise SemanticRootY1V1Error(f"{name} cardinality must be in [{minimum},{maximum}]")
            if any(type(item) is not item_type for item in values):
                raise SemanticRootY1V1Error(f"{name} contains an untyped item")
        for name, values in (
            ("template", self.topology_templates),
            ("event", self.topology_events),
            ("RGB basis", self.rgb_basis),
            ("quotient", self.irreducible_rgb_quotient),
        ):
            ids = tuple(
                item.template_id if name == "template" else item.event_id if name == "event" else item.atom_id
                for item in values
            )
            if ids != tuple(range(len(ids))):
                raise SemanticRootY1V1Error(f"{name} IDs must be canonical contiguous order")
        if self.pair_rgb_gauges and tuple(item.pair_id for item in self.pair_rgb_gauges) != tuple(
            range(PAIR_COUNT_N600)
        ):
            raise SemanticRootY1V1Error("pair RGB gauges must cover full n600 in exact upstream order")
        template_ids = {item.template_id for item in self.topology_templates}
        if any(event.template_id not in template_ids for event in self.topology_events):
            raise SemanticRootY1V1Error("topology event references an absent template")
        if self.rgb_gauge_ownership is RGBGaugeOwnershipV1.DERIVED_BY_SHARED_GENERATOR and self.rgb_basis:
            raise SemanticRootY1V1Error("procedural RGB basis requires explicit post-generator gauge ownership")
        if self.rgb_basis and not any(
            atom.role_mask == ALL_ROLE_MASK and atom.edge_width == 0 for atom in self.rgb_basis
        ):
            raise SemanticRootY1V1Error(
                f"{PALETTE_ONLY_DIRECT_LABEL_BLOCKER}: mandatory all-role non-edge scorer-native RGB basis is absent"
            )

    @property
    def packet_sha256(self) -> str:
        return _sha256(encode_semantic_root_y1_v1(self))


def _reject_foreign_payload(payload: bytes, *, name: str) -> None:
    if _sha256(payload) in _FORBIDDEN_WHOLE_PAYLOAD_SHA256:
        raise SemanticRootY1V1Error(f"{name} reuses a forbidden historical payload")
    for magic in _FORBIDDEN_NESTED_MAGICS:
        if payload.startswith(magic):
            raise SemanticRootY1V1Error(f"{name} contains forbidden foreign/raster payload magic {magic!r}")


def _encode_profile(profile: SemanticRealizationProfileV1) -> bytes:
    flat_rgb = tuple(channel for rgb in profile.role_rgb for channel in rgb)
    return _PROFILE.pack(
        _PROFILE_MAGIC,
        *flat_rgb,
        profile.texture_gain_q4,
        profile.edge_gain_q4,
        profile.chroma_gain_q4,
        profile.parallax_gain_q4,
        profile.renderer_seed,
    )


def _decode_profile(payload: bytes) -> SemanticRealizationProfileV1:
    if len(payload) != _PROFILE.size:
        raise SemanticRootY1V1Error("profile section has noncanonical length")
    values = _PROFILE.unpack(payload)
    if values[0] != _PROFILE_MAGIC:
        raise SemanticRootY1V1Error("profile section magic changed")
    role_rgb = tuple(
        tuple(int(channel) for channel in values[1 + offset : 1 + offset + 3]) for offset in range(0, 15, 3)
    )
    return SemanticRealizationProfileV1(
        role_rgb=role_rgb,  # type: ignore[arg-type]
        texture_gain_q4=int(values[16]),
        edge_gain_q4=int(values[17]),
        chroma_gain_q4=int(values[18]),
        parallax_gain_q4=int(values[19]),
        renderer_seed=int(values[20]),
    )


def _encode_templates(
    templates: tuple[SemanticTopologyTemplateV1, ...],
) -> bytes:
    return b"".join(
        _TEMPLATE.pack(
            item.template_id,
            int(item.role),
            int(item.shape),
            *item.params_q,
        )
        for item in templates
    )


def _decode_templates(payload: bytes, count: int) -> tuple[SemanticTopologyTemplateV1, ...]:
    if len(payload) != count * _TEMPLATE.size:
        raise SemanticRootY1V1Error("topology-template section length changed")
    result = []
    for offset in range(0, len(payload), _TEMPLATE.size):
        values = _TEMPLATE.unpack_from(payload, offset)
        try:
            role = SemanticRoleV1(values[1])
            shape = TopologyShapeV1(values[2])
        except ValueError as exc:
            raise SemanticRootY1V1Error("unknown topology role or shape") from exc
        result.append(
            SemanticTopologyTemplateV1(
                template_id=int(values[0]),
                role=role,
                shape=shape,
                params_q=tuple(int(value) for value in values[3:]),  # type: ignore[arg-type]
            )
        )
    return tuple(result)


def _encode_events(events: tuple[SemanticTopologyEventV1, ...]) -> bytes:
    return b"".join(
        _EVENT.pack(
            item.event_id,
            item.template_id,
            item.pair_start,
            item.pair_stop,
            item.z_order,
            item.anchor_x_q4,
            item.anchor_y_q4,
            item.velocity_x_q8,
            item.velocity_y_q8,
        )
        for item in events
    )


def _decode_events(payload: bytes, count: int) -> tuple[SemanticTopologyEventV1, ...]:
    if len(payload) != count * _EVENT.size:
        raise SemanticRootY1V1Error("topology-event section length changed")
    result = []
    for offset in range(0, len(payload), _EVENT.size):
        values = _EVENT.unpack_from(payload, offset)
        result.append(
            SemanticTopologyEventV1(
                event_id=int(values[0]),
                template_id=int(values[1]),
                pair_start=int(values[2]),
                pair_stop=int(values[3]),
                z_order=int(values[4]),
                anchor_x_q4=int(values[5]),
                anchor_y_q4=int(values[6]),
                velocity_x_q8=int(values[7]),
                velocity_y_q8=int(values[8]),
            )
        )
    return tuple(result)


def _encode_shared_generator(model: QuantizedSharedGeneratorV1) -> bytes:
    header = _MODEL_HEADER.pack(
        _MODEL_MAGIC,
        int(model.architecture),
        int(model.numeric_contract),
        int(model.activation),
        0,
        model.input_dim,
        model.hidden_dim,
        model.hidden_layer_count,
        model.modulation_dim,
        len(model.tensors),
    )
    rows = []
    for tensor in model.tensors:
        padded_shape = tensor.shape + (0,) * (4 - len(tensor.shape))
        rows.append(
            _TENSOR_HEADER.pack(
                tensor.tensor_id,
                int(tensor.role),
                int(tensor.dtype),
                tensor.scale_exponent,
                len(tensor.shape),
                *padded_shape,
                tensor.zero_point,
                len(tensor.data),
            )
            + tensor.data
        )
    return header + b"".join(rows)


def quantized_shared_generator_section_sha256(
    model: QuantizedSharedGeneratorV1,
) -> str:
    """Return the exact counted model-section identity for source custody."""

    if type(model) is not QuantizedSharedGeneratorV1:
        raise SemanticRootY1V1Error("model-section identity requires QuantizedSharedGeneratorV1")
    return _sha256(_encode_shared_generator(model))


def _decode_shared_generator(payload: bytes, *, tensor_count: int) -> QuantizedSharedGeneratorV1:
    if len(payload) < _MODEL_HEADER.size:
        raise SemanticRootY1V1Error("shared-generator section is truncated")
    values = _MODEL_HEADER.unpack_from(payload)
    (
        magic,
        architecture_raw,
        numeric_raw,
        activation_raw,
        reserved,
        input_dim,
        hidden_dim,
        hidden_layer_count,
        modulation_dim,
        inner_tensor_count,
    ) = values
    if magic != _MODEL_MAGIC or reserved != 0 or inner_tensor_count != tensor_count:
        raise SemanticRootY1V1Error("shared-generator inner header disagrees with packet")
    try:
        architecture = GeneratorArchitectureV1(architecture_raw)
        numeric_contract = GeneratorNumericContractV1(numeric_raw)
        activation = GeneratorActivationV1(activation_raw)
    except ValueError as exc:
        raise SemanticRootY1V1Error(f"{UNSUPPORTED_LEARNED_RECEIVER_BLOCKER}: unknown model contract") from exc
    cursor = _MODEL_HEADER.size
    tensors = []
    for _ in range(tensor_count):
        if cursor + _TENSOR_HEADER.size > len(payload):
            raise SemanticRootY1V1Error("shared-generator tensor header is truncated")
        row = _TENSOR_HEADER.unpack_from(payload, cursor)
        cursor += _TENSOR_HEADER.size
        (
            tensor_id,
            role_raw,
            dtype_raw,
            scale_exponent,
            rank,
            *tail,
        ) = row
        dimensions = tuple(int(value) for value in tail[:4])
        zero_point = int(tail[4])
        byte_length = int(tail[5])
        if (
            not 1 <= rank <= 4
            or any(dimensions[index] <= 0 for index in range(rank))
            or any(dimensions[index] != 0 for index in range(rank, 4))
        ):
            raise SemanticRootY1V1Error("tensor rank/padded shape is noncanonical")
        if cursor + byte_length > len(payload):
            raise SemanticRootY1V1Error("shared-generator tensor bytes are truncated")
        data = payload[cursor : cursor + byte_length]
        cursor += byte_length
        try:
            role = QuantizedTensorRoleV1(role_raw)
            dtype = QuantizedTensorDTypeV1(dtype_raw)
        except ValueError as exc:
            raise SemanticRootY1V1Error(f"{UNSUPPORTED_LEARNED_RECEIVER_BLOCKER}: unknown tensor role/dtype") from exc
        tensors.append(
            QuantizedGeneratorTensorV1(
                tensor_id=int(tensor_id),
                role=role,
                dtype=dtype,
                shape=dimensions[:rank],
                scale_exponent=int(scale_exponent),
                zero_point=zero_point,
                data=data,
            )
        )
    if cursor != len(payload):
        raise SemanticRootY1V1Error("shared-generator section has hidden/trailing tensor bytes")
    return QuantizedSharedGeneratorV1(
        architecture=architecture,
        numeric_contract=numeric_contract,
        activation=activation,
        input_dim=int(input_dim),
        hidden_dim=int(hidden_dim),
        hidden_layer_count=int(hidden_layer_count),
        modulation_dim=int(modulation_dim),
        tensors=tuple(tensors),
    )


def _encode_temporal_latents(stream: TemporalLatentStreamV1) -> bytes:
    return (
        _LATENT_HEADER.pack(
            _LATENT_MAGIC,
            int(stream.codec),
            stream.rice_k,
            PAIR_COUNT_N600,
            stream.latent_dim,
            stream.value_min,
            stream.value_max,
            len(stream.encoded_bytes),
        )
        + stream.encoded_bytes
    )


def _decode_temporal_latents(payload: bytes, *, latent_dim: int) -> TemporalLatentStreamV1:
    if len(payload) < _LATENT_HEADER.size:
        raise SemanticRootY1V1Error("temporal-latent section is truncated")
    (
        magic,
        codec_raw,
        rice_k,
        pair_count,
        inner_latent_dim,
        value_min,
        value_max,
        byte_length,
    ) = _LATENT_HEADER.unpack_from(payload)
    if (
        magic != _LATENT_MAGIC
        or pair_count != PAIR_COUNT_N600
        or inner_latent_dim != latent_dim
        or len(payload) != _LATENT_HEADER.size + byte_length
    ):
        raise SemanticRootY1V1Error("temporal-latent inner header disagrees with packet")
    try:
        codec = TemporalLatentCodecV1(codec_raw)
    except ValueError as exc:
        raise SemanticRootY1V1Error(f"{UNSUPPORTED_LEARNED_RECEIVER_BLOCKER}: unknown latent entropy codec") from exc
    encoded_bytes = payload[_LATENT_HEADER.size :]
    decoded = _rice_decode_signed(
        encoded_bytes,
        pair_count=PAIR_COUNT_N600,
        latent_dim=int(inner_latent_dim),
        rice_k=int(rice_k),
    )
    decoded_sha = _sha256(memoryview(np.ascontiguousarray(decoded.astype(">i2"))).cast("B"))
    return TemporalLatentStreamV1(
        codec=codec,
        rice_k=int(rice_k),
        latent_dim=int(inner_latent_dim),
        value_min=int(value_min),
        value_max=int(value_max),
        decoded_sha256=decoded_sha,
        encoded_bytes=encoded_bytes,
    )


def _encode_lineage(lineage: SemanticRootSourceLineageV1) -> bytes:
    compiler_id = lineage.compiler_id.encode("ascii").ljust(7, b"\0")
    return _LINEAGE.pack(
        _LINEAGE_MAGIC,
        compiler_id,
        bytes.fromhex(lineage.root_packet_sha256),
        bytes.fromhex(lineage.source_video_sha256),
        bytes.fromhex(lineage.target_custody_sha256),
        bytes.fromhex(lineage.compiler_source_sha256),
        bytes.fromhex(lineage.compile_config_sha256),
        bytes.fromhex(lineage.originality_declaration_sha256),
        bytes.fromhex(lineage.model_section_sha256),
        bytes.fromhex(lineage.latent_decoded_sha256),
    )


def _decode_lineage(payload: bytes) -> SemanticRootSourceLineageV1:
    if len(payload) != _LINEAGE.size:
        raise SemanticRootY1V1Error("source-lineage section has noncanonical length")
    values = _LINEAGE.unpack(payload)
    if values[0] != _LINEAGE_MAGIC:
        raise SemanticRootY1V1Error("source-lineage section magic changed")
    raw_compiler_id = values[1]
    compiler_id = raw_compiler_id.rstrip(b"\0")
    if not compiler_id or raw_compiler_id != compiler_id.ljust(7, b"\0"):
        raise SemanticRootY1V1Error("source-lineage compiler ID padding changed")
    try:
        compiler_text = compiler_id.decode("ascii")
    except UnicodeDecodeError as exc:
        raise SemanticRootY1V1Error("source-lineage compiler ID is not ASCII") from exc
    return SemanticRootSourceLineageV1(
        compiler_id=compiler_text,
        root_packet_sha256=values[2].hex(),
        source_video_sha256=values[3].hex(),
        target_custody_sha256=values[4].hex(),
        compiler_source_sha256=values[5].hex(),
        compile_config_sha256=values[6].hex(),
        originality_declaration_sha256=values[7].hex(),
        model_section_sha256=values[8].hex(),
        latent_decoded_sha256=values[9].hex(),
    )


def _encode_rgb_field(root: SemanticRootY1V1) -> bytes:
    basis = b"".join(
        _RGB_BASIS.pack(
            item.atom_id,
            item.role_mask,
            int(item.wave_kind),
            item.frequency_x,
            item.frequency_y,
            item.phase_u16,
            *item.amplitude_rgb,
            item.edge_width,
            0,
        )
        for item in root.rgb_basis
    )
    gauges = b"".join(
        _PAIR_GAUGE.pack(
            item.pair_id,
            item.phase_u16,
            item.parallax_x_q8,
            item.parallax_y_q8,
            item.luma_bias,
            item.chroma_u_bias,
            item.chroma_v_bias,
            item.texture_gain_q8,
        )
        for item in root.pair_rgb_gauges
    )
    return (
        _RGB_HEADER.pack(
            _RGB_MAGIC,
            int(root.rgb_gauge_ownership),
            0,
            len(root.rgb_basis),
            len(root.pair_rgb_gauges),
        )
        + basis
        + gauges
    )


def _decode_rgb_field(
    payload: bytes,
    *,
    basis_count: int,
    gauge_count: int,
) -> tuple[
    RGBGaugeOwnershipV1,
    tuple[RGBBasisAtomV1, ...],
    tuple[PairRGBGaugeV1, ...],
]:
    expected = _RGB_HEADER.size + basis_count * _RGB_BASIS.size + gauge_count * _PAIR_GAUGE.size
    if len(payload) != expected:
        raise SemanticRootY1V1Error("RGB-field section length changed")
    (
        magic,
        ownership_raw,
        reserved,
        inner_basis_count,
        inner_gauge_count,
    ) = _RGB_HEADER.unpack_from(payload)
    if (
        magic != _RGB_MAGIC
        or reserved != 0
        or (
            inner_basis_count,
            inner_gauge_count,
        )
        != (
            basis_count,
            gauge_count,
        )
    ):
        raise SemanticRootY1V1Error("RGB-field inner header disagrees with packet")
    try:
        ownership = RGBGaugeOwnershipV1(ownership_raw)
    except ValueError as exc:
        raise SemanticRootY1V1Error("unknown RGB gauge ownership mode") from exc
    cursor = _RGB_HEADER.size
    basis = []
    for _ in range(basis_count):
        values = _RGB_BASIS.unpack_from(payload, cursor)
        cursor += _RGB_BASIS.size
        if values[-1] != 0:
            raise SemanticRootY1V1Error("RGB basis reserved byte is nonzero")
        try:
            wave_kind = RGBWaveKindV1(values[2])
        except ValueError as exc:
            raise SemanticRootY1V1Error("unknown RGB wave kind") from exc
        basis.append(
            RGBBasisAtomV1(
                atom_id=int(values[0]),
                role_mask=int(values[1]),
                wave_kind=wave_kind,
                frequency_x=int(values[3]),
                frequency_y=int(values[4]),
                phase_u16=int(values[5]),
                amplitude_rgb=tuple(int(value) for value in values[6:9]),  # type: ignore[arg-type]
                edge_width=int(values[9]),
            )
        )
    gauges = []
    for _ in range(gauge_count):
        values = _PAIR_GAUGE.unpack_from(payload, cursor)
        cursor += _PAIR_GAUGE.size
        gauges.append(
            PairRGBGaugeV1(
                pair_id=int(values[0]),
                phase_u16=int(values[1]),
                parallax_x_q8=int(values[2]),
                parallax_y_q8=int(values[3]),
                luma_bias=int(values[4]),
                chroma_u_bias=int(values[5]),
                chroma_v_bias=int(values[6]),
                texture_gain_q8=int(values[7]),
            )
        )
    if cursor != len(payload):
        raise AssertionError("RGB-field decoder did not consume its exact section")
    expected_gauge_count = 0 if ownership is RGBGaugeOwnershipV1.DERIVED_BY_SHARED_GENERATOR else PAIR_COUNT_N600
    if gauge_count != expected_gauge_count:
        raise SemanticRootY1V1Error("RGB gauge count disagrees with exclusive ownership mode")
    return ownership, tuple(basis), tuple(gauges)


def _encode_quotient(atoms: tuple[RGBQuotientAtomV1, ...]) -> bytes:
    rows = b"".join(
        _QUOTIENT.pack(
            item.atom_id,
            item.pair_start,
            item.pair_stop,
            item.role_mask,
            int(item.edge_only),
            item.center_x_q4,
            item.center_y_q4,
            item.velocity_x_q8,
            item.velocity_y_q8,
            item.radius_x_q4,
            item.radius_y_q4,
            *item.amplitude_rgb,
        )
        for item in atoms
    )
    return _QUOTIENT_HEADER.pack(_QUOTIENT_MAGIC, len(atoms)) + rows


def _decode_quotient(payload: bytes, count: int) -> tuple[RGBQuotientAtomV1, ...]:
    expected = _QUOTIENT_HEADER.size + count * _QUOTIENT.size
    if len(payload) != expected:
        raise SemanticRootY1V1Error("RGB-quotient section length changed")
    magic, inner_count = _QUOTIENT_HEADER.unpack_from(payload)
    if magic != _QUOTIENT_MAGIC or inner_count != count:
        raise SemanticRootY1V1Error("RGB-quotient inner header disagrees with packet")
    result = []
    cursor = _QUOTIENT_HEADER.size
    for _ in range(count):
        values = _QUOTIENT.unpack_from(payload, cursor)
        cursor += _QUOTIENT.size
        if values[4] not in (0, 1):
            raise SemanticRootY1V1Error("quotient edge_only byte is not canonical bool")
        result.append(
            RGBQuotientAtomV1(
                atom_id=int(values[0]),
                pair_start=int(values[1]),
                pair_stop=int(values[2]),
                role_mask=int(values[3]),
                edge_only=bool(values[4]),
                center_x_q4=int(values[5]),
                center_y_q4=int(values[6]),
                velocity_x_q8=int(values[7]),
                velocity_y_q8=int(values[8]),
                radius_x_q4=int(values[9]),
                radius_y_q4=int(values[10]),
                amplitude_rgb=tuple(int(value) for value in values[11:14]),  # type: ignore[arg-type]
            )
        )
    return tuple(result)


def _semantic_root_sections(root: SemanticRootY1V1) -> tuple[bytes, ...]:
    return (
        _encode_profile(root.profile),
        _encode_templates(root.topology_templates),
        _encode_events(root.topology_events),
        _encode_shared_generator(root.shared_generator),
        _encode_temporal_latents(root.temporal_latents),
        _encode_rgb_field(root),
        _encode_quotient(root.irreducible_rgb_quotient),
    )


def semantic_root_g17_logical_values(
    root: SemanticRootY1V1,
) -> tuple[
    G17SemanticTopologyV1,
    G17RealizationGaugeV1,
    G17LearnedResidualOwnershipV1,
    G17PopulationSharingV1,
    G17LearnedResidualOwnershipV1 | G17AnalyticResidualOwnershipV1,
]:
    """Expose exact section values through the canonical G17 logical types."""

    if type(root) is not SemanticRootY1V1:
        raise SemanticRootY1V1Error("G17 logical-value adapter requires exact SemanticRootY1V1")
    sections = _semantic_root_sections(root)
    quotient_type = G17LearnedResidualOwnershipV1 if root.irreducible_rgb_quotient else G17AnalyticResidualOwnershipV1
    return (
        G17SemanticTopologyV1(sections[1] + sections[2]),
        G17RealizationGaugeV1(sections[0] + sections[5]),
        G17LearnedResidualOwnershipV1(sections[3]),
        G17PopulationSharingV1(sections[4]),
        quotient_type(sections[6]),
    )


def encode_semantic_root_y1_v1(root: SemanticRootY1V1) -> bytes:
    if type(root) is not SemanticRootY1V1:
        raise SemanticRootY1V1Error("encode requires exact SemanticRootY1V1")
    sections = _semantic_root_sections(root)
    for tag, section in zip(SECTION_TAGS, sections, strict=True):
        _reject_foreign_payload(section, name=f"{tag.decode('ascii')} section")
    header = _HEADER.pack(
        MAGIC,
        VERSION,
        0,
        PAIR_COUNT_N600,
        SCORER_H,
        SCORER_W,
        SCORER_CHANNELS,
        int(root.background_role),
        len(root.topology_templates),
        len(root.topology_events),
        len(root.shared_generator.tensors),
        root.temporal_latents.latent_dim,
        len(root.rgb_basis),
        len(root.pair_rgb_gauges),
        len(root.irreducible_rgb_quotient),
    )
    metadata = b"".join(
        _SECTION_META.pack(tag, len(section)) for tag, section in zip(SECTION_TAGS, sections, strict=True)
    )
    body = b"".join(sections)
    packet = header + metadata + body + _FOOTER.pack(zlib.crc32(body) & 0xFFFFFFFF)
    if len(packet) > MAX_PACKET_BYTES:
        raise SemanticRootY1V1Error("semantic-root packet exceeds sparse typed packet cap; dense plane refused")
    _reject_foreign_payload(packet, name="semantic-root packet")
    return packet


def parse_semantic_root_y1_v1(payload: bytes) -> SemanticRootY1V1:
    if type(payload) is not bytes:
        raise SemanticRootY1V1Error("packet must be exact bytes")
    if len(payload) > MAX_PACKET_BYTES:
        raise SemanticRootY1V1Error("packet exceeds sparse typed cap; dense plane refused")
    minimum = _HEADER.size + len(SECTION_TAGS) * _SECTION_META.size + _FOOTER.size
    if len(payload) < minimum:
        raise SemanticRootY1V1Error("packet is truncated")
    _reject_foreign_payload(payload, name="semantic-root packet")
    values = _HEADER.unpack_from(payload)
    (
        magic,
        version,
        flags,
        pair_count,
        scorer_h,
        scorer_w,
        channels,
        background_role_raw,
        template_count,
        event_count,
        tensor_count,
        latent_dim,
        basis_count,
        gauge_count,
        quotient_count,
    ) = values
    if magic != MAGIC or version != VERSION or flags != 0:
        raise SemanticRootY1V1Error("packet magic/version/flags changed")
    if (pair_count, scorer_h, scorer_w, channels) != (
        PAIR_COUNT_N600,
        SCORER_H,
        SCORER_W,
        SCORER_CHANNELS,
    ):
        raise SemanticRootY1V1Error("packet is not exact full-n600 scorer geometry")
    if not (
        0 <= template_count <= MAX_TEMPLATES
        and 0 <= event_count <= MAX_EVENTS
        and (template_count == 0) == (event_count == 0)
        and 1 <= tensor_count <= MAX_GENERATOR_TENSORS
        and 1 <= latent_dim <= 64
        and 0 <= basis_count <= MAX_RGB_BASIS_ATOMS
        and gauge_count in (0, PAIR_COUNT_N600)
        and quotient_count <= MAX_QUOTIENT_ATOMS
    ):
        raise SemanticRootY1V1Error("packet section cardinality is outside closed bounds")
    try:
        background_role = SemanticRoleV1(background_role_raw)
    except ValueError as exc:
        raise SemanticRootY1V1Error("unknown background role") from exc
    cursor = _HEADER.size
    metas: list[tuple[bytes, int]] = []
    for expected_tag in SECTION_TAGS:
        tag, length = _SECTION_META.unpack_from(payload, cursor)
        cursor += _SECTION_META.size
        if tag != expected_tag:
            raise SemanticRootY1V1Error("section tags are absent, duplicated, or reordered")
        metas.append((tag, int(length)))
    total_section_bytes = sum(length for _, length in metas)
    exact_length = cursor + total_section_bytes + _FOOTER.size
    if exact_length != len(payload):
        raise SemanticRootY1V1Error("packet has truncation or trailing bytes after exact EOF")
    sections = []
    for tag, length in metas:
        section = payload[cursor : cursor + length]
        cursor += length
        _reject_foreign_payload(section, name=f"{tag.decode('ascii')} section")
        sections.append(section)
    (expected_crc,) = _FOOTER.unpack_from(payload, cursor)
    if expected_crc != zlib.crc32(b"".join(sections)) & 0xFFFFFFFF:
        raise SemanticRootY1V1Error("packet body CRC32 mismatch")
    profile = _decode_profile(sections[0])
    templates = _decode_templates(sections[1], template_count)
    events = _decode_events(sections[2], event_count)
    shared_generator = _decode_shared_generator(sections[3], tensor_count=tensor_count)
    temporal_latents = _decode_temporal_latents(sections[4], latent_dim=latent_dim)
    rgb_gauge_ownership, basis, gauges = _decode_rgb_field(
        sections[5], basis_count=basis_count, gauge_count=gauge_count
    )
    quotient = _decode_quotient(sections[6], quotient_count)
    root = SemanticRootY1V1(
        background_role=background_role,
        profile=profile,
        shared_generator=shared_generator,
        temporal_latents=temporal_latents,
        rgb_gauge_ownership=rgb_gauge_ownership,
        topology_templates=templates,
        topology_events=events,
        rgb_basis=basis,
        pair_rgb_gauges=gauges,
        irreducible_rgb_quotient=quotient,
    )
    if encode_semantic_root_y1_v1(root) != payload:
        raise SemanticRootY1V1Error("packet is not canonical under strict re-emission")
    return root


def _active_topology_labels(root: SemanticRootY1V1, pair_id: int) -> np.ndarray:
    pair_id = _int(pair_id, "pair_id", minimum=0, maximum=PAIR_COUNT_N600 - 1)
    labels = np.full((SCORER_H, SCORER_W), int(root.background_role), dtype=np.uint8)
    y_q4 = np.arange(SCORER_H, dtype=np.int32)[:, None] * 16
    x_q4 = np.arange(SCORER_W, dtype=np.int32)[None, :] * 16
    by_id = {template.template_id: template for template in root.topology_templates}
    active = sorted(
        (event for event in root.topology_events if event.pair_start <= pair_id < event.pair_stop),
        key=lambda item: (item.z_order, item.event_id),
    )
    for event in active:
        template = by_id[event.template_id]
        dt = pair_id - event.pair_start
        anchor_x = event.anchor_x_q4 + _round_div_signed_scalar(event.velocity_x_q8 * dt, 16)
        anchor_y = event.anchor_y_q4 + _round_div_signed_scalar(event.velocity_y_q8 * dt, 16)
        p0, p1, p2, p3, p4, p5 = template.params_q
        local_x = x_q4 - anchor_x
        local_y = y_q4 - anchor_y
        if template.shape is TopologyShapeV1.RECT:
            mask = (np.abs(local_x - p0) <= p2) & (np.abs(local_y - p1) <= p3)
        elif template.shape is TopologyShapeV1.ELLIPSE:
            dx = local_x - p0
            dy = local_y - p1
            mask = dx * dx * p3 * p3 + dy * dy * p2 * p2 <= p2 * p2 * p3 * p3
        elif template.shape is TopologyShapeV1.TRAPEZOID:
            # params: center_x, top_y, bottom_y, top_half, bottom_half, unused.
            span = p2 - p1
            within_y = (local_y >= p1) & (local_y <= p2)
            half_width = p3 + ((local_y - p1) * (p4 - p3)) // span
            mask = within_y & (np.abs(local_x - p0) <= half_width)
        else:
            # params: x0, y0, slope_q8, curve_q12, half_width_q4, y_span_q4.
            dy = local_y - p1
            curve_x = p0 + (p2 * dy) // 256 + (p3 * dy * dy) // (4096 * 16)
            mask = (np.abs(dy) <= p5) & (np.abs(local_x - curve_x) <= p4)
        labels[mask] = int(template.role)
    return labels


def _edge_mask(labels: np.ndarray) -> np.ndarray:
    edge = np.zeros_like(labels, dtype=bool)
    edge[1:, :] |= labels[1:, :] != labels[:-1, :]
    edge[:-1, :] |= labels[:-1, :] != labels[1:, :]
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return edge


def _dilate(mask: np.ndarray, width: int) -> np.ndarray:
    result = mask.copy()
    for _ in range(max(0, width - 1)):
        expanded = result.copy()
        expanded[1:, :] |= result[:-1, :]
        expanded[:-1, :] |= result[1:, :]
        expanded[:, 1:] |= result[:, :-1]
        expanded[:, :-1] |= result[:, 1:]
        result = expanded
    return result


def _activate_q12(values: np.ndarray, activation: GeneratorActivationV1) -> np.ndarray:
    if activation is GeneratorActivationV1.HARD_TANH_Q12:
        return np.clip(values, -4096, 4096)
    if activation is GeneratorActivationV1.RELU6_Q12:
        return np.clip(values, 0, 6 * 4096)
    raise SemanticRootY1V1Error(f"{UNSUPPORTED_LEARNED_RECEIVER_BLOCKER}: activation has no implementation")


def _coordinate_features_q12(
    *,
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    input_dim: int,
    renderer_seed: int,
) -> np.ndarray:
    count = x.size
    features = np.empty((count, input_dim), dtype=np.int64)
    features[:, 0] = (x.astype(np.int64) * 8192 // (SCORER_W - 1)) - 4096
    features[:, 1] = (y.astype(np.int64) * 8192 // (SCORER_H - 1)) - 4096
    features[:, 2] = labels.astype(np.int64) * 2048 - 4096
    features[:, 3] = 4096
    for column in range(4, input_dim):
        fx = 1 + ((renderer_seed + 17 * column) % 31)
        fy = 1 + (((renderer_seed >> 8) + 29 * column) % 31)
        phase = (
            fx * x.astype(np.int64) * 256 // SCORER_W
            + fy * y.astype(np.int64) * 256 // SCORER_H
            + (renderer_seed >> ((column % 4) * 8))
        ) & 0xFF
        features[:, column] = (127 - 2 * np.abs(phase - 128)) * 4096 // 127
    return features


def _render_learned_generator_residual(
    root: SemanticRootY1V1,
    *,
    pair_id: int,
    labels: np.ndarray,
) -> np.ndarray:
    model = root.shared_generator
    if (
        model.architecture is not GeneratorArchitectureV1.ORIGINAL_COORDINR_FILM_MLP_V1
        or model.numeric_contract is not GeneratorNumericContractV1.INT8_WEIGHT_INT16_STATE_INT32_ACCUM_Q12
    ):
        raise SemanticRootY1V1Error(f"{UNSUPPORTED_LEARNED_RECEIVER_BLOCKER}: model ABI is not implemented")
    tensors = model.tensors
    input_weight = tensors[0].array.astype(np.int64)
    input_bias = tensors[1].array.astype(np.int64)
    hidden_pairs = tuple(
        (
            tensors[2 + 2 * index].array.astype(np.int64),
            tensors[3 + 2 * index].array.astype(np.int64),
        )
        for index in range(model.hidden_layer_count)
    )
    film_offset = 2 + 2 * model.hidden_layer_count
    film_weight = tensors[film_offset].array.astype(np.int64)
    film_bias = tensors[film_offset + 1].array.astype(np.int64)
    output_weight = tensors[film_offset + 2].array.astype(np.int64)
    output_bias = tensors[film_offset + 3].array.astype(np.int64)
    latent = root.temporal_latents.decode()[pair_id].astype(np.int64)
    film = _round_div_signed_array(film_weight @ latent, 128) + film_bias
    gamma = film[: model.hidden_dim]
    beta = film[model.hidden_dim :]
    flat_labels = labels.reshape(-1)
    flat_y, flat_x = np.indices((SCORER_H, SCORER_W), dtype=np.int32)
    flat_y = flat_y.reshape(-1)
    flat_x = flat_x.reshape(-1)
    output = np.empty((flat_labels.size, SCORER_CHANNELS), dtype=np.int32)
    tile_size = 4_096
    for start in range(0, flat_labels.size, tile_size):
        stop = min(start + tile_size, flat_labels.size)
        features = _coordinate_features_q12(
            x=flat_x[start:stop],
            y=flat_y[start:stop],
            labels=flat_labels[start:stop],
            input_dim=model.input_dim,
            renderer_seed=root.profile.renderer_seed,
        )
        hidden = _round_div_signed_array(features @ input_weight.T, 128) + input_bias
        hidden = _activate_q12(hidden, model.activation)
        for weight, bias in hidden_pairs:
            hidden = _round_div_signed_array(hidden @ weight.T, 128) + bias
            hidden += _round_div_signed_array(hidden * gamma, 4096) + beta
            hidden = _activate_q12(hidden, model.activation)
        residual_q12 = _round_div_signed_array(hidden @ output_weight.T, 128) + output_bias
        output[start:stop] = _round_div_signed_array(residual_q12, 4096).astype(np.int32)
    return output.reshape(SCORER_H, SCORER_W, SCORER_CHANNELS)


def render_semantic_root_y1_scorer(root: SemanticRootY1V1, pair_id: int) -> np.ndarray:
    """Render one deterministic scorer-grid RGB Y1 witness."""

    if type(root) is not SemanticRootY1V1:
        raise SemanticRootY1V1Error("render requires exact SemanticRootY1V1")
    pair_id = _int(pair_id, "pair_id", minimum=0, maximum=PAIR_COUNT_N600 - 1)
    labels = _active_topology_labels(root, pair_id)
    palette = np.asarray(root.profile.role_rgb, dtype=np.int32)
    rgb = palette[labels].copy()
    gauge = (
        root.pair_rgb_gauges[pair_id]
        if root.rgb_gauge_ownership is RGBGaugeOwnershipV1.EXPLICIT_NONOVERLAPPING_POST_GENERATOR
        else PairRGBGaugeV1(
            pair_id=pair_id,
            phase_u16=0,
            parallax_x_q8=0,
            parallax_y_q8=0,
            luma_bias=0,
            chroma_u_bias=0,
            chroma_v_bias=0,
            texture_gain_q8=256,
        )
    )
    chroma_gain = root.profile.chroma_gain_q4
    rgb[:, :, 0] += gauge.luma_bias + _round_div_signed_scalar(gauge.chroma_v_bias * chroma_gain, 16)
    rgb[:, :, 1] += gauge.luma_bias - _round_div_signed_scalar(
        (gauge.chroma_u_bias + gauge.chroma_v_bias) * chroma_gain, 32
    )
    rgb[:, :, 2] += gauge.luma_bias + _round_div_signed_scalar(gauge.chroma_u_bias * chroma_gain, 16)
    y = np.arange(SCORER_H, dtype=np.int64)[:, None]
    x = np.arange(SCORER_W, dtype=np.int64)[None, :]
    edges: np.ndarray | None = None
    for atom in root.rgb_basis:
        phase = (
            atom.frequency_x * x * 256 // SCORER_W
            + atom.frequency_y * y * 256 // SCORER_H
            + atom.phase_u16
            + gauge.phase_u16
            + _round_div_signed_scalar(
                (atom.frequency_x * gauge.parallax_x_q8 + atom.frequency_y * gauge.parallax_y_q8)
                * root.profile.parallax_gain_q4,
                16,
            )
        ) & 0xFF
        if atom.wave_kind is RGBWaveKindV1.TRIANGLE:
            wave = 127 - 2 * np.abs(phase - 128)
        elif atom.wave_kind is RGBWaveKindV1.CHECKER:
            wave = np.where(phase < 128, -127, 127)
        else:
            wave = 127 - np.minimum(np.abs(phase - 64), np.abs(phase - 192)) * 4
        role_allowed = ((atom.role_mask >> labels) & 1).astype(bool)
        if atom.edge_width:
            if edges is None:
                edges = _edge_mask(labels)
            role_allowed &= _dilate(edges, atom.edge_width)
            profile_gain = root.profile.edge_gain_q4
        else:
            profile_gain = root.profile.texture_gain_q4
        for channel, amplitude in enumerate(atom.amplitude_rgb):
            numerator = wave * int(amplitude) * gauge.texture_gain_q8 * profile_gain
            delta = _round_div_signed_array(numerator, 127 * 256 * 16)
            rgb[:, :, channel] += np.where(role_allowed, delta, 0).astype(np.int32)
    rgb += _render_learned_generator_residual(
        root,
        pair_id=pair_id,
        labels=labels,
    )
    for atom in root.irreducible_rgb_quotient:
        if not atom.pair_start <= pair_id < atom.pair_stop:
            continue
        dt = pair_id - atom.pair_start
        center_x = atom.center_x_q4 + _round_div_signed_scalar(atom.velocity_x_q8 * dt, 16)
        center_y = atom.center_y_q4 + _round_div_signed_scalar(atom.velocity_y_q8 * dt, 16)
        wx = np.maximum(0, atom.radius_x_q4 - np.abs(x * 16 - center_x))
        wy = np.maximum(0, atom.radius_y_q4 - np.abs(y * 16 - center_y))
        weight_q8 = wx * wy * 256 // (atom.radius_x_q4 * atom.radius_y_q4)
        allowed = ((atom.role_mask >> labels) & 1).astype(bool)
        if atom.edge_only:
            if edges is None:
                edges = _edge_mask(labels)
            allowed &= edges
        for channel, amplitude in enumerate(atom.amplitude_rgb):
            delta = _round_div_signed_array(weight_q8 * int(amplitude), 256)
            rgb[:, :, channel] += np.where(allowed, delta, 0).astype(np.int32)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def iter_semantic_root_y1_batches(
    root: SemanticRootY1V1,
    *,
    batch_size: int = 16,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield exact upstream-order full-n600 scorer RGB batches."""

    batch_size = _int(batch_size, "batch_size", minimum=1, maximum=16)
    for start in range(0, PAIR_COUNT_N600, batch_size):
        stop = min(start + batch_size, PAIR_COUNT_N600)
        batch = np.stack([render_semantic_root_y1_scorer(root, pair_id) for pair_id in range(start, stop)])
        batch.setflags(write=False)
        yield start, batch


def semantic_root_y1_population_sha256(
    root: SemanticRootY1V1,
    *,
    batch_size: int = 16,
) -> str:
    digest = hashlib.sha256(b"SEMANTIC-ROOT-Y1-SCORER-POPULATION-V1\0")
    observed = 0
    for start, batch in iter_semantic_root_y1_batches(root, batch_size=batch_size):
        if start != observed:
            raise AssertionError("internal semantic-root pair order drifted")
        for local, frame in enumerate(batch):
            pair_id = start + local
            digest.update(struct.pack(">H", pair_id))
            digest.update(memoryview(np.ascontiguousarray(frame)).cast("B"))
            observed += 1
    if observed != PAIR_COUNT_N600:
        raise AssertionError("internal semantic-root population is incomplete")
    return digest.hexdigest()


def realize_semantic_root_y1_v10_factor2(
    root: SemanticRootY1V1,
    pair_id: int,
) -> tuple[np.ndarray, Factor2ExactVerification]:
    """Realize one scorer Y1 through the public V10 factor-2 path."""

    target = render_semantic_root_y1_scorer(root, pair_id)
    operator = DisjointResizeOperator.build(
        camera_h=874,
        camera_w=1164,
        scorer_h=SCORER_H,
        scorer_w=SCORER_W,
    )
    frame = realize_factor2_uint8_scorer_plane(operator, target)
    proof = verify_factor2_uint8_scorer_plane(operator, frame, target)
    if not proof.certified_exact:
        raise SemanticRootY1V1Error("V10 factor-2 realization was not certified exact")
    frame.setflags(write=False)
    return frame, proof


def bind_semantic_root_to_g17(
    root: SemanticRootY1V1,
    *,
    population: G17PairPopulationV1,
    topology_owner: G17LogicalOwnershipV1,
    realization_owner: G17LogicalOwnershipV1,
    generator_owner: G17LogicalOwnershipV1,
    temporal_owner: G17LogicalOwnershipV1,
    quotient_owner: G17LogicalOwnershipV1,
) -> str:
    """Bind exact wire sections to canonical G17 owners without a parallel IR."""

    if type(root) is not SemanticRootY1V1:
        raise SemanticRootY1V1Error("G17 adapter requires exact SemanticRootY1V1")
    if type(population) is not G17PairPopulationV1:
        raise SemanticRootY1V1Error("G17 adapter requires canonical PairPopulation")
    if root.rgb_gauge_ownership is RGBGaugeOwnershipV1.EXPLICIT_NONOVERLAPPING_POST_GENERATOR:
        raise SemanticRootY1V1Error(
            f"{EXPLICIT_GAUGE_ARBITRATION_BLOCKER}: G17 must retain measured non-overlap/value-per-byte evidence"
        )
    canonical_pairs = tuple(range(PAIR_COUNT_N600))
    if (
        population.global_pair_ids != canonical_pairs
        or population.source_pair_ids != canonical_pairs
        or population.v9_pair_coordinates != canonical_pairs
        or population.pbr_pair_coordinates != canonical_pairs
        or population.obligation_ir_coordinates != canonical_pairs
        or population.v10_local_coordinates != canonical_pairs
    ):
        raise SemanticRootY1V1Error("G17 population is not the exact ordered full-n600 identity map")
    sections = _semantic_root_sections(root)
    expected = (
        (
            topology_owner,
            G17LogicalOwnershipKindV1.SEMANTIC_TOPOLOGY,
            G17SemanticTopologyV1,
            sections[1] + sections[2],
        ),
        (
            realization_owner,
            G17LogicalOwnershipKindV1.REALIZATION_GAUGE,
            G17RealizationGaugeV1,
            sections[0] + sections[5],
        ),
        (
            generator_owner,
            G17LogicalOwnershipKindV1.LEARNED_RESIDUAL,
            G17LearnedResidualOwnershipV1,
            sections[3],
        ),
        (
            temporal_owner,
            G17LogicalOwnershipKindV1.POPULATION_SHARED,
            G17PopulationSharingV1,
            sections[4],
        ),
        (
            quotient_owner,
            (
                G17LogicalOwnershipKindV1.LEARNED_RESIDUAL
                if root.irreducible_rgb_quotient
                else G17LogicalOwnershipKindV1.ANALYTIC_RESIDUAL
            ),
            (G17LearnedResidualOwnershipV1 if root.irreducible_rgb_quotient else G17AnalyticResidualOwnershipV1),
            sections[6],
        ),
    )
    for owner, ownership_kind, value_type, exact_bytes in expected:
        if (
            type(owner) is not G17LogicalOwnershipV1
            or owner.ownership_kind is not ownership_kind
            or type(owner.value) is not value_type
            or owner.value.exact_bytes != exact_bytes
        ):
            raise SemanticRootY1V1Error("semantic-root section is not exactly retained by its canonical G17 owner")
    digest = hashlib.sha256(b"G17-SEMANTIC-ROOT-Y1-ADAPTER-V1\0")
    digest.update(bytes.fromhex(root.packet_sha256))
    digest.update(bytes.fromhex(population.binding_sha256))
    for owner, _, _, _ in expected:
        digest.update(bytes.fromhex(owner.identity_sha256))
    return digest.hexdigest()


def encode_semantic_root_source_lineage_manifest(
    root: SemanticRootY1V1,
    lineage: SemanticRootSourceLineageV1,
) -> bytes:
    """Encode packet-bound lineage as external encoder-only evidence."""

    if type(root) is not SemanticRootY1V1 or type(lineage) is not SemanticRootSourceLineageV1:
        raise SemanticRootY1V1Error("lineage manifest requires exact root and lineage objects")
    if (
        lineage.root_packet_sha256 != root.packet_sha256
        or lineage.model_section_sha256 != quantized_shared_generator_section_sha256(root.shared_generator)
        or lineage.latent_decoded_sha256 != root.temporal_latents.decoded_sha256
    ):
        raise SemanticRootY1V1Error("external lineage does not bind exact packet/model/latent identities")
    return _encode_lineage(lineage)


def parse_semantic_root_source_lineage_manifest(
    payload: bytes,
) -> SemanticRootSourceLineageV1:
    """Strictly reopen external encoder-only lineage evidence."""

    if type(payload) is not bytes:
        raise SemanticRootY1V1Error("lineage manifest must be exact bytes")
    lineage = _decode_lineage(payload)
    if _encode_lineage(lineage) != payload:
        raise SemanticRootY1V1Error("lineage manifest is not canonical")
    return lineage


def bind_semantic_root_source_lineage_to_g17(
    root: SemanticRootY1V1,
    lineage: SemanticRootSourceLineageV1,
    *,
    lineage_owner: G17LogicalOwnershipV1,
) -> str:
    """Bind external custody only to canonical G17 encoder-evidence ownership."""

    manifest = encode_semantic_root_source_lineage_manifest(root, lineage)
    if (
        type(lineage_owner) is not G17LogicalOwnershipV1
        or lineage_owner.ownership_kind is not G17LogicalOwnershipKindV1.ENCODER_EVIDENCE
        or type(lineage_owner.value) is not G17EncoderOnlyTeacherOracleEvidenceV1
        or lineage_owner.value.exact_bytes != manifest
    ):
        raise SemanticRootY1V1Error("lineage manifest is not retained as canonical G17 encoder-only evidence")
    return _sha256(
        b"G17-SEMANTIC-ROOT-LINEAGE-ADAPTER-V1\0"
        + bytes.fromhex(root.packet_sha256)
        + bytes.fromhex(lineage_owner.identity_sha256)
    )


def final_semantic_root_y1_binding_sha256(
    *,
    root_packet_sha256: str,
    g17_population_binding_sha256: str,
    scorer_y1_population_sha256: str,
) -> str:
    """Domain-separated final-Y1 identity for a later G94-V2 conditional fit."""

    digest = hashlib.sha256(b"FINAL-SEMANTIC-ROOT-Y1-G94-V2-BINDING\0")
    for name, value in (
        ("root packet", root_packet_sha256),
        ("G17 population", g17_population_binding_sha256),
        ("scorer Y1 population", scorer_y1_population_sha256),
    ):
        digest.update(bytes.fromhex(_require_sha256(value, name)))
    return digest.hexdigest()
