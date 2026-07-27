# SPDX-License-Identifier: MIT
"""Counted P-once Pose preimage chart bound to one exact G94 population state.

The production wire is deliberately split:

* one population-global learned basis object, counted exactly once; and
* ordered coefficient/selector chunks containing at most sixteen pairs.

Every chunk references the exact basis bytes and the same whole-population
custody key.  The NumPy receiver is scorer-free and streams chunks without
duplicating the shared basis.  Pair-0 reachability uses these same production
types; it is not a special packet shape.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Final, Literal

import numpy as np

CAMERA_HEIGHT: Final = 874
CAMERA_WIDTH: Final = 1164
CHANNELS: Final = 3
PAIR_COUNT: Final = 600
MAX_BATCH_PAIRS: Final = 16
MAX_GRID_HEIGHT: Final = 512
MAX_GRID_WIDTH: Final = 512
MAX_RANK: Final = 255
MAX_OBJECT_BYTES: Final = 64 << 20
BOUND_G94_PARENT_COMMIT: Final = "9e84c69b8a389337270b70fd4023a4174ef3c552"
BOUND_G94_PRODUCT_MEMBER_SHA256: Final = "84335c287ccee915fafce19f0258dc3fa6939095b5bdd773e09dbf7e47fd934a"
BOUND_G94_CONDITIONING_STATE_SHA256: Final = "7ab4829d0ecf53b973629be518cc0be575cf826f8a33eceffcb13cb00d678c9b"
REACHABILITY_THRESHOLD: Final = 0.00047366
REACHABILITY_COORDINATE_SCOPE: Final = (
    "SUFFICIENT_POSE_COORDINATE_AT_CURRENT_TEACHER_AND_132132_BYTE_SEAM_NOT_UNIVERSAL_PASS_FAIL"
)

BASIS_MAGIC: Final = b"G95BAS1\x00"
CHUNK_MAGIC: Final = b"G95CHK1\x00"
WIRE_VERSION: Final = 1
_BASIS_HEADER: Final = struct.Struct(">8sBBBBHHH32s32s32s32s32s32s32sI32sI32s")
_CHUNK_HEADER: Final = struct.Struct(">8sBBBBHH32s32s32s32sI32sI32sI32s")
_CRC32: Final = struct.Struct(">I")

RECEIVER_ID: Final = "tac.g95.population_pose_preimage_chart_numpy_receiver.v1"
WIRE_POLICY_ID: Final = "P_ONCE_GLOBAL_I8_BASIS_PLUS_INDEXED_I16_COEFFICIENT_CHUNKS_V1"
BASE_LAW_ID: Final = "COPY_EXACT_CONDITIONAL_Y1"
BILINEAR_REFERENCE_ID: Final = "NUMPY_FP32_ALIGN_CORNERS_FALSE_HALF_PIXEL_BILINEAR_V1"
ROUNDING_POLICY_ID: Final = "CLAMP_0_255_THEN_IEEE754_ROUND_TO_NEAREST_EVEN_UINT8_V1"
MISSING_INTEGRATION: Final = "G88_G94_NEW_TYPED_G95_POPULATION_CHART_MODE_AND_OUTER_ARCHIVE_RACE_OWED"
OUTER_ZIP_SCORE_ADMISSION: Final = "OWED_EXACT_OUTER_ZIP_BUILD_AND_UPSTREAM_EVALUATE_PENALTY"
FORMULATION_SCOPE: Final = (
    "G95_POPULATION_SHARED_LINEAR_LOW_RESIDUAL_POSE_PREIMAGE_CHART_BOUND_TO_EXACT_G94_CONDITIONING_STATE"
)
ONE_STATE_MISS_AXIS: Final = "HIGHER_STATIC_RANK_OR_GRID"
POPULATION_TRANSFER_REQUEST: Final = "Y1_CONDITIONED_SHARED_GENERATOR_OR_FEATURE_MODULATED_BASIS"
MISS_VERDICT_SCOPE: Final = "STATIC_SHARED_BASIS_AT_EXACT_G94_CONDITIONING_STATE"


class PopulationPosePreimageChartError(ValueError):
    """A G95 wire, custody, numeric, shape, or receiver invariant failed."""


class G95ControlModeV1(IntEnum):
    PASS_PRECONDITIONAL_Y0 = 0
    COPY_EXACT_CONDITIONAL_Y1 = 1
    POPULATION_SHARED_PREIMAGE_CHART = 2


class G95BaseLawV1(IntEnum):
    COPY_EXACT_CONDITIONAL_Y1 = 1


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return _sha256(memoryview(np.ascontiguousarray(value)).cast("B"))


def _require_sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PopulationPosePreimageChartError(f"{label} must be canonical lowercase SHA-256")
    return value


def _exact_int(value: object, *, label: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise PopulationPosePreimageChartError(f"{label} must be an exact integer in [{minimum},{maximum}]")
    return value


def _validate_chunk_pair_ids(source_pair_ids: tuple[int, ...]) -> None:
    if (
        type(source_pair_ids) is not tuple
        or not 1 <= len(source_pair_ids) <= MAX_BATCH_PAIRS
        or any(type(value) is not int or not 0 <= value < PAIR_COUNT for value in source_pair_ids)
        or source_pair_ids != tuple(range(source_pair_ids[0], source_pair_ids[0] + len(source_pair_ids)))
    ):
        raise PopulationPosePreimageChartError("source_pair_ids must be 1..16 ordered contiguous exact n600 pair IDs")


def _pair_ids_wire(source_pair_ids: tuple[int, ...]) -> bytes:
    _validate_chunk_pair_ids(source_pair_ids)
    return struct.pack(f">{len(source_pair_ids)}H", *source_pair_ids)


def population_pair_ids_sha256() -> str:
    """Identity of the exact ordered n600 population selector."""

    return _sha256(struct.pack(f">{PAIR_COUNT}H", *range(PAIR_COUNT)))


def population_state_key(
    *,
    g94_product_member_sha256: str,
    g94_conditioning_state_sha256: str,
    whole_preconditional_camera_sha256: str,
    selected_target_table_sha256: str,
    posenet_weights_sha256: str,
) -> str:
    """Canonical whole-state foreign key shared by the basis and every chunk."""

    payload = {
        "schema": "tac.g95.population_state_key.v1",
        "source_pair_ids_sha256": population_pair_ids_sha256(),
        "g94_product_member_sha256": _require_sha256(
            g94_product_member_sha256,
            label="g94_product_member_sha256",
        ),
        "g94_conditioning_state_sha256": _require_sha256(
            g94_conditioning_state_sha256,
            label="g94_conditioning_state_sha256",
        ),
        "whole_preconditional_camera_sha256": _require_sha256(
            whole_preconditional_camera_sha256,
            label="whole_preconditional_camera_sha256",
        ),
        "selected_target_table_sha256": _require_sha256(
            selected_target_table_sha256,
            label="selected_target_table_sha256",
        ),
        "posenet_weights_sha256": _require_sha256(
            posenet_weights_sha256,
            label="posenet_weights_sha256",
        ),
    }
    wire = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return _sha256(wire)


def _require_exact_array(
    value: np.ndarray,
    *,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
    label: str,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype != dtype or raw.shape != shape:
        raise PopulationPosePreimageChartError(f"{label} must have exact dtype {dtype} and shape {shape}")
    return np.ascontiguousarray(raw)


def _positive_f32_array(value: np.ndarray, *, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = _require_exact_array(
        value,
        dtype=np.dtype(np.float32),
        shape=shape,
        label=label,
    )
    if not np.all(np.isfinite(result)) or np.any(result <= np.float32(0.0)) or np.any(np.signbit(result)):
        raise PopulationPosePreimageChartError(f"{label} must contain canonical finite positive fp32 values")
    return result


def _immutable_copy(value: np.ndarray, *, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _basis_wire(basis_q: np.ndarray, *, rank: int, grid_height: int, grid_width: int) -> bytes:
    exact = _require_exact_array(
        basis_q,
        dtype=np.dtype(np.int8),
        shape=(rank, grid_height, grid_width, CHANNELS),
        label="basis_q",
    )
    return exact.tobytes(order="C")


def _f32_wire(value: np.ndarray, *, rank: int, label: str) -> bytes:
    exact = _positive_f32_array(value, shape=(rank,), label=label)
    return exact.astype(">f4", copy=False).tobytes(order="C")


def _coefficients_wire(coefficients_q: np.ndarray, *, pair_count: int, rank: int) -> bytes:
    exact = _require_exact_array(
        coefficients_q,
        dtype=np.dtype(np.int16),
        shape=(pair_count, rank),
        label="coefficients_q",
    )
    return exact.astype(">i2", copy=False).tobytes(order="C")


def _finish_object(prefix: bytes, *, label: str) -> bytes:
    if len(prefix) + _CRC32.size > MAX_OBJECT_BYTES:
        raise PopulationPosePreimageChartError(f"G95 {label} exceeds bounded V1 wire")
    return prefix + _CRC32.pack(zlib.crc32(prefix) & 0xFFFFFFFF)


def encode_population_pose_preimage_basis(
    *,
    g94_product_member_sha256: str,
    g94_conditioning_state_sha256: str,
    whole_preconditional_camera_sha256: str,
    selected_target_table_sha256: str,
    posenet_weights_sha256: str,
    basis_q: np.ndarray,
    basis_scales: np.ndarray,
) -> bytes:
    """Encode the learned population-global basis exactly once."""

    raw_basis = np.asarray(basis_q)
    if raw_basis.ndim != 4 or raw_basis.shape[-1] != CHANNELS:
        raise PopulationPosePreimageChartError("basis_q must have exact [rank,h,w,3] shape")
    rank, grid_height, grid_width, _channels = raw_basis.shape
    _exact_int(rank, label="rank", minimum=1, maximum=MAX_RANK)
    _exact_int(grid_height, label="grid_height", minimum=1, maximum=MAX_GRID_HEIGHT)
    _exact_int(grid_width, label="grid_width", minimum=1, maximum=MAX_GRID_WIDTH)
    product = _require_sha256(g94_product_member_sha256, label="g94_product_member_sha256")
    conditioning = _require_sha256(
        g94_conditioning_state_sha256,
        label="g94_conditioning_state_sha256",
    )
    whole_preconditional = _require_sha256(
        whole_preconditional_camera_sha256,
        label="whole_preconditional_camera_sha256",
    )
    targets = _require_sha256(
        selected_target_table_sha256,
        label="selected_target_table_sha256",
    )
    weights = _require_sha256(posenet_weights_sha256, label="posenet_weights_sha256")
    state_key = population_state_key(
        g94_product_member_sha256=product,
        g94_conditioning_state_sha256=conditioning,
        whole_preconditional_camera_sha256=whole_preconditional,
        selected_target_table_sha256=targets,
        posenet_weights_sha256=weights,
    )
    sections = (
        _basis_wire(
            basis_q,
            rank=rank,
            grid_height=grid_height,
            grid_width=grid_width,
        ),
        _f32_wire(basis_scales, rank=rank, label="basis_scales"),
    )
    prefix = _BASIS_HEADER.pack(
        BASIS_MAGIC,
        WIRE_VERSION,
        int(G95BaseLawV1.COPY_EXACT_CONDITIONAL_Y1),
        0,
        0,
        rank,
        grid_height,
        grid_width,
        bytes.fromhex(product),
        bytes.fromhex(conditioning),
        bytes.fromhex(population_pair_ids_sha256()),
        bytes.fromhex(whole_preconditional),
        bytes.fromhex(targets),
        bytes.fromhex(weights),
        bytes.fromhex(state_key),
        *(item for section in sections for item in (len(section), bytes.fromhex(_sha256(section)))),
    ) + b"".join(sections)
    return _finish_object(prefix, label="basis object")


@dataclass(frozen=True, slots=True)
class ParsedPopulationPosePreimageBasisV1:
    object_bytes: bytes = field(repr=False)
    g94_product_member_sha256: str
    g94_conditioning_state_sha256: str
    population_pair_ids_sha256: str
    whole_preconditional_camera_sha256: str
    selected_target_table_sha256: str
    posenet_weights_sha256: str
    population_state_key: str
    grid_height: int
    grid_width: int
    rank: int
    basis_bytes: bytes = field(repr=False)
    basis_scales_bytes: bytes = field(repr=False)
    basis_q: np.ndarray = field(repr=False)
    basis_scales: np.ndarray = field(repr=False)
    receiver_id: Literal["tac.g95.population_pose_preimage_chart_numpy_receiver.v1"] = RECEIVER_ID
    wire_policy_id: Literal["P_ONCE_GLOBAL_I8_BASIS_PLUS_INDEXED_I16_COEFFICIENT_CHUNKS_V1"] = WIRE_POLICY_ID
    outer_zip_score_admission: Literal["OWED_EXACT_OUTER_ZIP_BUILD_AND_UPSTREAM_EVALUATE_PENALTY"] = (
        OUTER_ZIP_SCORE_ADMISSION
    )
    research_only: Literal[True] = True
    candidate_claim: Literal[False] = False
    score_claim: Literal[False] = False
    promotion_eligible: Literal[False] = False
    pointer_moved: Literal[False] = False

    def __post_init__(self) -> None:
        if type(self.object_bytes) is not bytes:
            raise PopulationPosePreimageChartError("basis object lost exact byte custody")
        for value, label in (
            (self.g94_product_member_sha256, "g94_product_member_sha256"),
            (self.g94_conditioning_state_sha256, "g94_conditioning_state_sha256"),
            (self.population_pair_ids_sha256, "population_pair_ids_sha256"),
            (self.whole_preconditional_camera_sha256, "whole_preconditional_camera_sha256"),
            (self.selected_target_table_sha256, "selected_target_table_sha256"),
            (self.posenet_weights_sha256, "posenet_weights_sha256"),
            (self.population_state_key, "population_state_key"),
        ):
            _require_sha256(value, label=label)
        if self.population_pair_ids_sha256 != population_pair_ids_sha256():
            raise PopulationPosePreimageChartError("basis population pair IDs differ from exact n600")
        expected_key = population_state_key(
            g94_product_member_sha256=self.g94_product_member_sha256,
            g94_conditioning_state_sha256=self.g94_conditioning_state_sha256,
            whole_preconditional_camera_sha256=self.whole_preconditional_camera_sha256,
            selected_target_table_sha256=self.selected_target_table_sha256,
            posenet_weights_sha256=self.posenet_weights_sha256,
        )
        if self.population_state_key != expected_key:
            raise PopulationPosePreimageChartError("basis whole-population state key differs")
        exact_basis = _require_exact_array(
            self.basis_q,
            dtype=np.dtype(np.int8),
            shape=(self.rank, self.grid_height, self.grid_width, CHANNELS),
            label="basis_q",
        )
        exact_scales = _positive_f32_array(
            self.basis_scales,
            shape=(self.rank,),
            label="basis_scales",
        )
        if self.basis_bytes != _basis_wire(
            exact_basis,
            rank=self.rank,
            grid_height=self.grid_height,
            grid_width=self.grid_width,
        ) or self.basis_scales_bytes != _f32_wire(exact_scales, rank=self.rank, label="basis_scales"):
            raise PopulationPosePreimageChartError("basis arrays differ from counted sections")
        if (
            self.receiver_id != RECEIVER_ID
            or self.wire_policy_id != WIRE_POLICY_ID
            or self.outer_zip_score_admission != OUTER_ZIP_SCORE_ADMISSION
            or self.research_only is not True
            or self.candidate_claim is not False
            or self.score_claim is not False
            or self.promotion_eligible is not False
            or self.pointer_moved is not False
        ):
            raise PopulationPosePreimageChartError("G95 basis truth labels became permissive")
        reencoded = encode_population_pose_preimage_basis(
            g94_product_member_sha256=self.g94_product_member_sha256,
            g94_conditioning_state_sha256=self.g94_conditioning_state_sha256,
            whole_preconditional_camera_sha256=self.whole_preconditional_camera_sha256,
            selected_target_table_sha256=self.selected_target_table_sha256,
            posenet_weights_sha256=self.posenet_weights_sha256,
            basis_q=exact_basis,
            basis_scales=exact_scales,
        )
        if reencoded != self.object_bytes:
            raise PopulationPosePreimageChartError("basis object changed on strict re-encoding")
        object.__setattr__(self, "basis_q", _immutable_copy(exact_basis, dtype=np.dtype(np.int8)))
        object.__setattr__(
            self,
            "basis_scales",
            _immutable_copy(exact_scales, dtype=np.dtype(np.float32)),
        )

    @property
    def object_sha256(self) -> str:
        return _sha256(self.object_bytes)

    @property
    def counted_bytes(self) -> int:
        return len(self.object_bytes)


def _check_object_envelope(
    object_bytes: bytes,
    *,
    header: struct.Struct,
    expected_sha256: str | None,
    label: str,
) -> tuple[tuple[Any, ...], bytes]:
    if type(object_bytes) is not bytes:
        raise PopulationPosePreimageChartError(f"{label} must be exact bytes")
    if not header.size + _CRC32.size <= len(object_bytes) <= MAX_OBJECT_BYTES:
        raise PopulationPosePreimageChartError(f"{label} is truncated or exceeds the ceiling")
    if expected_sha256 is not None and _sha256(object_bytes) != _require_sha256(
        expected_sha256,
        label=f"expected_{label}_sha256",
    ):
        raise PopulationPosePreimageChartError(f"{label} exact SHA custody differs")
    prefix = object_bytes[: -_CRC32.size]
    (expected_crc,) = _CRC32.unpack_from(object_bytes, len(prefix))
    if zlib.crc32(prefix) & 0xFFFFFFFF != expected_crc:
        raise PopulationPosePreimageChartError(f"{label} CRC32 mismatch")
    return header.unpack_from(object_bytes), prefix


def parse_population_pose_preimage_basis(
    object_bytes: bytes,
    *,
    expected_object_sha256: str | None = None,
) -> ParsedPopulationPosePreimageBasisV1:
    header, prefix = _check_object_envelope(
        object_bytes,
        header=_BASIS_HEADER,
        expected_sha256=expected_object_sha256,
        label="basis_object",
    )
    (
        magic,
        version,
        base_law,
        reserved0,
        reserved1,
        rank,
        grid_height,
        grid_width,
        product,
        conditioning,
        pair_ids_sha,
        whole_preconditional,
        targets,
        weights,
        state_key,
        basis_length,
        basis_sha,
        scales_length,
        scales_sha,
    ) = header
    if (
        magic != BASIS_MAGIC
        or version != WIRE_VERSION
        or base_law != int(G95BaseLawV1.COPY_EXACT_CONDITIONAL_Y1)
        or reserved0 != 0
        or reserved1 != 0
    ):
        raise PopulationPosePreimageChartError("basis object magic/version/base-law/reserved mismatch")
    _exact_int(rank, label="rank", minimum=1, maximum=MAX_RANK)
    _exact_int(grid_height, label="grid_height", minimum=1, maximum=MAX_GRID_HEIGHT)
    _exact_int(grid_width, label="grid_width", minimum=1, maximum=MAX_GRID_WIDTH)
    expected_lengths = (
        rank * grid_height * grid_width * CHANNELS,
        rank * 4,
    )
    if (basis_length, scales_length) != expected_lengths:
        raise PopulationPosePreimageChartError("basis section length/shape/rank contract differs")
    if len(object_bytes) != _BASIS_HEADER.size + sum(expected_lengths) + _CRC32.size:
        raise PopulationPosePreimageChartError("basis object exact EOF differs")
    basis_offset = _BASIS_HEADER.size
    basis_bytes = object_bytes[basis_offset : basis_offset + basis_length]
    scales_bytes = object_bytes[basis_offset + basis_length : len(prefix)]
    if _sha256(basis_bytes) != basis_sha.hex() or _sha256(scales_bytes) != scales_sha.hex():
        raise PopulationPosePreimageChartError("basis section SHA-256 mismatch")
    basis_q = np.frombuffer(basis_bytes, dtype=np.int8).reshape(
        rank,
        grid_height,
        grid_width,
        CHANNELS,
    )
    basis_scales = np.frombuffer(scales_bytes, dtype=">f4").astype(np.float32)
    return ParsedPopulationPosePreimageBasisV1(
        object_bytes=object_bytes,
        g94_product_member_sha256=product.hex(),
        g94_conditioning_state_sha256=conditioning.hex(),
        population_pair_ids_sha256=pair_ids_sha.hex(),
        whole_preconditional_camera_sha256=whole_preconditional.hex(),
        selected_target_table_sha256=targets.hex(),
        posenet_weights_sha256=weights.hex(),
        population_state_key=state_key.hex(),
        grid_height=grid_height,
        grid_width=grid_width,
        rank=rank,
        basis_bytes=basis_bytes,
        basis_scales_bytes=scales_bytes,
        basis_q=basis_q,
        basis_scales=basis_scales,
    )


def encode_population_pose_preimage_coefficient_chunk(
    *,
    basis_object_sha256: str,
    population_state_key_sha256: str,
    preconditional_camera_sha256: str,
    selected_target_sha256: str,
    source_pair_ids: tuple[int, ...],
    rank: int,
    coefficients_q: np.ndarray,
    coefficient_scales: np.ndarray,
) -> bytes:
    """Encode one indexed <=16-pair coefficient chunk without the shared basis."""

    _validate_chunk_pair_ids(source_pair_ids)
    _exact_int(rank, label="rank", minimum=1, maximum=MAX_RANK)
    basis_ref = _require_sha256(basis_object_sha256, label="basis_object_sha256")
    state_key = _require_sha256(
        population_state_key_sha256,
        label="population_state_key_sha256",
    )
    preconditional = _require_sha256(
        preconditional_camera_sha256,
        label="preconditional_camera_sha256",
    )
    selected_target = _require_sha256(
        selected_target_sha256,
        label="selected_target_sha256",
    )
    sections = (
        _pair_ids_wire(source_pair_ids),
        _coefficients_wire(
            coefficients_q,
            pair_count=len(source_pair_ids),
            rank=rank,
        ),
        _f32_wire(coefficient_scales, rank=rank, label="coefficient_scales"),
    )
    prefix = _CHUNK_HEADER.pack(
        CHUNK_MAGIC,
        WIRE_VERSION,
        int(G95BaseLawV1.COPY_EXACT_CONDITIONAL_Y1),
        len(source_pair_ids),
        0,
        source_pair_ids[0],
        rank,
        bytes.fromhex(basis_ref),
        bytes.fromhex(state_key),
        bytes.fromhex(preconditional),
        bytes.fromhex(selected_target),
        *(item for section in sections for item in (len(section), bytes.fromhex(_sha256(section)))),
    ) + b"".join(sections)
    return _finish_object(prefix, label="coefficient chunk")


@dataclass(frozen=True, slots=True)
class ParsedPopulationPosePreimageCoefficientChunkV1:
    object_bytes: bytes = field(repr=False)
    basis_object_sha256: str
    population_state_key: str
    preconditional_camera_sha256: str
    selected_target_sha256: str
    source_pair_ids: tuple[int, ...]
    rank: int
    pair_ids_bytes: bytes = field(repr=False)
    coefficients_bytes: bytes = field(repr=False)
    coefficient_scales_bytes: bytes = field(repr=False)
    coefficients_q: np.ndarray = field(repr=False)
    coefficient_scales: np.ndarray = field(repr=False)
    outer_zip_score_admission: Literal["OWED_EXACT_OUTER_ZIP_BUILD_AND_UPSTREAM_EVALUATE_PENALTY"] = (
        OUTER_ZIP_SCORE_ADMISSION
    )
    research_only: Literal[True] = True
    candidate_claim: Literal[False] = False
    score_claim: Literal[False] = False
    promotion_eligible: Literal[False] = False
    pointer_moved: Literal[False] = False

    def __post_init__(self) -> None:
        if type(self.object_bytes) is not bytes:
            raise PopulationPosePreimageChartError("coefficient chunk lost exact byte custody")
        for value, label in (
            (self.basis_object_sha256, "basis_object_sha256"),
            (self.population_state_key, "population_state_key"),
            (self.preconditional_camera_sha256, "preconditional_camera_sha256"),
            (self.selected_target_sha256, "selected_target_sha256"),
        ):
            _require_sha256(value, label=label)
        _validate_chunk_pair_ids(self.source_pair_ids)
        coefficients = _require_exact_array(
            self.coefficients_q,
            dtype=np.dtype(np.int16),
            shape=(len(self.source_pair_ids), self.rank),
            label="coefficients_q",
        )
        scales = _positive_f32_array(
            self.coefficient_scales,
            shape=(self.rank,),
            label="coefficient_scales",
        )
        if (
            self.pair_ids_bytes != _pair_ids_wire(self.source_pair_ids)
            or self.coefficients_bytes
            != _coefficients_wire(
                coefficients,
                pair_count=len(self.source_pair_ids),
                rank=self.rank,
            )
            or self.coefficient_scales_bytes != _f32_wire(scales, rank=self.rank, label="coefficient_scales")
        ):
            raise PopulationPosePreimageChartError("coefficient arrays differ from counted sections")
        if (
            self.outer_zip_score_admission != OUTER_ZIP_SCORE_ADMISSION
            or self.research_only is not True
            or self.candidate_claim is not False
            or self.score_claim is not False
            or self.promotion_eligible is not False
            or self.pointer_moved is not False
        ):
            raise PopulationPosePreimageChartError("G95 chunk truth labels became permissive")
        reencoded = encode_population_pose_preimage_coefficient_chunk(
            basis_object_sha256=self.basis_object_sha256,
            population_state_key_sha256=self.population_state_key,
            preconditional_camera_sha256=self.preconditional_camera_sha256,
            selected_target_sha256=self.selected_target_sha256,
            source_pair_ids=self.source_pair_ids,
            rank=self.rank,
            coefficients_q=coefficients,
            coefficient_scales=scales,
        )
        if reencoded != self.object_bytes:
            raise PopulationPosePreimageChartError("coefficient chunk changed on strict re-encoding")
        object.__setattr__(
            self,
            "coefficients_q",
            _immutable_copy(coefficients, dtype=np.dtype(np.int16)),
        )
        object.__setattr__(
            self,
            "coefficient_scales",
            _immutable_copy(scales, dtype=np.dtype(np.float32)),
        )

    @property
    def object_sha256(self) -> str:
        return _sha256(self.object_bytes)

    @property
    def counted_bytes(self) -> int:
        return len(self.object_bytes)

    @property
    def pair_selector_bytes(self) -> int:
        return len(self.pair_ids_bytes)


@dataclass(frozen=True, slots=True)
class PopulationPosePreimageChartWireSetV1:
    """One P-once basis reference plus one coefficient chunk.

    This is an in-memory typed view only.  It has no combined wire encoding, so
    constructing many views cannot duplicate the counted basis bytes.
    """

    basis: ParsedPopulationPosePreimageBasisV1
    chunk: ParsedPopulationPosePreimageCoefficientChunkV1

    def __post_init__(self) -> None:
        if (
            type(self.basis) is not ParsedPopulationPosePreimageBasisV1
            or type(self.chunk) is not ParsedPopulationPosePreimageCoefficientChunkV1
        ):
            raise PopulationPosePreimageChartError("wire set requires exact parsed basis and chunk")
        if self.chunk.basis_object_sha256 != self.basis.object_sha256:
            raise PopulationPosePreimageChartError("wire-set chunk basis reference mismatch")
        if self.chunk.population_state_key != self.basis.population_state_key:
            raise PopulationPosePreimageChartError("wire-set chunk whole-state reference mismatch")
        if self.chunk.rank != self.basis.rank:
            raise PopulationPosePreimageChartError("wire-set chunk rank differs from basis")

    @property
    def wire_set_sha256(self) -> str:
        return _sha256(bytes.fromhex(self.basis.object_sha256) + bytes.fromhex(self.chunk.object_sha256))

    @property
    def rank(self) -> int:
        return self.basis.rank

    @property
    def grid_height(self) -> int:
        return self.basis.grid_height

    @property
    def grid_width(self) -> int:
        return self.basis.grid_width

    @property
    def source_pair_ids(self) -> tuple[int, ...]:
        return self.chunk.source_pair_ids

    @property
    def basis_q(self) -> np.ndarray:
        return self.basis.basis_q

    @property
    def basis_scales(self) -> np.ndarray:
        return self.basis.basis_scales

    @property
    def coefficients_q(self) -> np.ndarray:
        return self.chunk.coefficients_q

    @property
    def coefficient_scales(self) -> np.ndarray:
        return self.chunk.coefficient_scales

    @property
    def counted_sections(self) -> tuple[tuple[str, int, str], ...]:
        return (
            ("shared_basis_i8", len(self.basis.basis_bytes), _sha256(self.basis.basis_bytes)),
            (
                "basis_scales_f32be",
                len(self.basis.basis_scales_bytes),
                _sha256(self.basis.basis_scales_bytes),
            ),
            (
                "source_pair_ids_u16be",
                len(self.chunk.pair_ids_bytes),
                _sha256(self.chunk.pair_ids_bytes),
            ),
            (
                "per_pair_coefficients_i16be",
                len(self.chunk.coefficients_bytes),
                _sha256(self.chunk.coefficients_bytes),
            ),
            (
                "coefficient_scales_f32be",
                len(self.chunk.coefficient_scales_bytes),
                _sha256(self.chunk.coefficient_scales_bytes),
            ),
        )

    @property
    def learned_payload_bytes(self) -> int:
        return (
            len(self.basis.basis_bytes)
            + len(self.basis.basis_scales_bytes)
            + len(self.chunk.coefficients_bytes)
            + len(self.chunk.coefficient_scales_bytes)
        )

    @property
    def pair_selector_bytes(self) -> int:
        return self.chunk.pair_selector_bytes

    @property
    def total_counted_bytes(self) -> int:
        return self.basis.counted_bytes + self.chunk.counted_bytes


def parse_population_pose_preimage_coefficient_chunk(
    object_bytes: bytes,
    *,
    expected_object_sha256: str | None = None,
) -> ParsedPopulationPosePreimageCoefficientChunkV1:
    header, prefix = _check_object_envelope(
        object_bytes,
        header=_CHUNK_HEADER,
        expected_sha256=expected_object_sha256,
        label="coefficient_chunk",
    )
    (
        magic,
        version,
        base_law,
        pair_count,
        reserved,
        pair_start,
        rank,
        basis_ref,
        state_key,
        preconditional,
        selected_target,
        pair_ids_length,
        pair_ids_sha,
        coefficients_length,
        coefficients_sha,
        scales_length,
        scales_sha,
    ) = header
    if (
        magic != CHUNK_MAGIC
        or version != WIRE_VERSION
        or base_law != int(G95BaseLawV1.COPY_EXACT_CONDITIONAL_Y1)
        or reserved != 0
    ):
        raise PopulationPosePreimageChartError("coefficient chunk magic/version/base-law/reserved mismatch")
    _exact_int(pair_count, label="pair_count", minimum=1, maximum=MAX_BATCH_PAIRS)
    _exact_int(pair_start, label="pair_start", minimum=0, maximum=PAIR_COUNT - 1)
    _exact_int(rank, label="rank", minimum=1, maximum=MAX_RANK)
    if pair_start + pair_count > PAIR_COUNT:
        raise PopulationPosePreimageChartError("coefficient chunk escapes exact n600 domain")
    expected_lengths = (pair_count * 2, pair_count * rank * 2, rank * 4)
    if (pair_ids_length, coefficients_length, scales_length) != expected_lengths:
        raise PopulationPosePreimageChartError("coefficient chunk length/shape/rank contract differs")
    if len(object_bytes) != _CHUNK_HEADER.size + sum(expected_lengths) + _CRC32.size:
        raise PopulationPosePreimageChartError("coefficient chunk exact EOF differs")
    offset = _CHUNK_HEADER.size
    sections: list[bytes] = []
    for length, expected_sha in (
        (pair_ids_length, pair_ids_sha),
        (coefficients_length, coefficients_sha),
        (scales_length, scales_sha),
    ):
        section = object_bytes[offset : offset + length]
        offset += length
        if _sha256(section) != expected_sha.hex():
            raise PopulationPosePreimageChartError("coefficient chunk section SHA-256 mismatch")
        sections.append(section)
    if offset != len(prefix):
        raise PopulationPosePreimageChartError("coefficient chunk parser did not consume exact EOF")
    source_pair_ids = tuple(struct.unpack(f">{pair_count}H", sections[0]))
    _validate_chunk_pair_ids(source_pair_ids)
    if source_pair_ids[0] != pair_start:
        raise PopulationPosePreimageChartError("coefficient chunk selector differs from header range")
    coefficients_q = (
        np.frombuffer(sections[1], dtype=">i2")
        .astype(np.int16)
        .reshape(
            pair_count,
            rank,
        )
    )
    coefficient_scales = np.frombuffer(sections[2], dtype=">f4").astype(np.float32)
    return ParsedPopulationPosePreimageCoefficientChunkV1(
        object_bytes=object_bytes,
        basis_object_sha256=basis_ref.hex(),
        population_state_key=state_key.hex(),
        preconditional_camera_sha256=preconditional.hex(),
        selected_target_sha256=selected_target.hex(),
        source_pair_ids=source_pair_ids,
        rank=rank,
        pair_ids_bytes=sections[0],
        coefficients_bytes=sections[1],
        coefficient_scales_bytes=sections[2],
        coefficients_q=coefficients_q,
        coefficient_scales=coefficient_scales,
    )


def validate_population_chunk_coverage(
    basis: ParsedPopulationPosePreimageBasisV1,
    chunks: Sequence[ParsedPopulationPosePreimageCoefficientChunkV1],
) -> int:
    """Prove exact 0..599 coverage with one basis reference and no gaps/overlap."""

    if type(basis) is not ParsedPopulationPosePreimageBasisV1:
        raise PopulationPosePreimageChartError("population coverage requires exact parsed basis")
    expected_next = 0
    for chunk in chunks:
        if type(chunk) is not ParsedPopulationPosePreimageCoefficientChunkV1:
            raise PopulationPosePreimageChartError("population coverage requires exact parsed chunks")
        if chunk.basis_object_sha256 != basis.object_sha256:
            raise PopulationPosePreimageChartError("chunk basis reference mismatch")
        if chunk.population_state_key != basis.population_state_key:
            raise PopulationPosePreimageChartError("chunk whole-state reference mismatch")
        if chunk.rank != basis.rank:
            raise PopulationPosePreimageChartError("chunk rank differs from population basis")
        if chunk.source_pair_ids[0] != expected_next:
            raise PopulationPosePreimageChartError("population chunks contain a gap, overlap, or reordering")
        expected_next += len(chunk.source_pair_ids)
    if expected_next != PAIR_COUNT:
        raise PopulationPosePreimageChartError("population chunks do not cover all 600 rows")
    return expected_next


def bilinear_resize_align_corners_false_numpy(
    value: np.ndarray,
    *,
    output_height: int,
    output_width: int,
) -> np.ndarray:
    _exact_int(output_height, label="output_height", minimum=1, maximum=CAMERA_HEIGHT)
    _exact_int(output_width, label="output_width", minimum=1, maximum=CAMERA_WIDTH)
    raw = np.asarray(value)
    squeeze = raw.ndim == 3
    if squeeze:
        raw = raw[None, ...]
    if (
        raw.dtype != np.float32
        or raw.ndim != 4
        or raw.shape[0] < 1
        or raw.shape[-1] != CHANNELS
        or raw.shape[1] < 1
        or raw.shape[2] < 1
        or not np.all(np.isfinite(raw))
    ):
        raise PopulationPosePreimageChartError("bilinear input must be finite float32 [N,h,w,3] or [h,w,3]")
    source = np.ascontiguousarray(raw)
    input_height, input_width = source.shape[1:3]
    y = (np.arange(output_height, dtype=np.float32) + np.float32(0.5)) * np.float32(
        input_height / output_height
    ) - np.float32(0.5)
    x = (np.arange(output_width, dtype=np.float32) + np.float32(0.5)) * np.float32(
        input_width / output_width
    ) - np.float32(0.5)
    y = np.clip(y, np.float32(0.0), np.float32(input_height - 1))
    x = np.clip(x, np.float32(0.0), np.float32(input_width - 1))
    y0 = np.floor(y).astype(np.intp)
    x0 = np.floor(x).astype(np.intp)
    y1 = np.minimum(y0 + 1, input_height - 1)
    x1 = np.minimum(x0 + 1, input_width - 1)
    wy = (y - y0.astype(np.float32)).reshape(1, output_height, 1, 1)
    wx = (x - x0.astype(np.float32)).reshape(1, 1, output_width, 1)
    vertical = source[:, y0, :, :] * (np.float32(1.0) - wy) + source[:, y1, :, :] * wy
    resized = vertical[:, :, x0, :] * (np.float32(1.0) - wx) + vertical[:, :, x1, :] * wx
    result = np.ascontiguousarray(resized, dtype=np.float32)
    return result[0] if squeeze else result


@dataclass(frozen=True, slots=True)
class PopulationPosePreimageChartBatchResultV1:
    basis_object_sha256: str
    coefficient_chunk_sha256: str
    population_state_key: str
    g94_product_member_sha256: str
    g94_conditioning_state_sha256: str
    preconditional_camera_sha256: str
    selected_target_sha256: str
    source_pair_ids: tuple[int, ...]
    preconditional_camera_pairs: np.ndarray = field(repr=False)
    camera_pairs: np.ndarray = field(repr=False)
    camera_sha256: str
    exact_y1_sha256: str
    changed_y0_values: int
    changed_y0_pixels: int
    deterministic_double_decode: Literal[True] = True
    exact_y1_preserved: Literal[True] = True
    all_chunk_pairs_addressed: Literal[True] = True
    outer_zip_score_admission: Literal["OWED_EXACT_OUTER_ZIP_BUILD_AND_UPSTREAM_EVALUATE_PENALTY"] = (
        OUTER_ZIP_SCORE_ADMISSION
    )
    research_only: Literal[True] = True
    candidate_claim: Literal[False] = False
    score_claim: Literal[False] = False
    promotion_eligible: Literal[False] = False
    pointer_moved: Literal[False] = False

    def __post_init__(self) -> None:
        for value, label in (
            (self.basis_object_sha256, "basis_object_sha256"),
            (self.coefficient_chunk_sha256, "coefficient_chunk_sha256"),
            (self.population_state_key, "population_state_key"),
            (self.g94_product_member_sha256, "g94_product_member_sha256"),
            (self.g94_conditioning_state_sha256, "g94_conditioning_state_sha256"),
            (self.preconditional_camera_sha256, "preconditional_camera_sha256"),
            (self.selected_target_sha256, "selected_target_sha256"),
            (self.camera_sha256, "camera_sha256"),
            (self.exact_y1_sha256, "exact_y1_sha256"),
        ):
            _require_sha256(value, label=label)
        _validate_chunk_pair_ids(self.source_pair_ids)
        shape = (len(self.source_pair_ids), 2, CAMERA_HEIGHT, CAMERA_WIDTH, CHANNELS)
        before = _require_exact_array(
            self.preconditional_camera_pairs,
            dtype=np.dtype(np.uint8),
            shape=shape,
            label="preconditional_camera_pairs",
        )
        after = _require_exact_array(
            self.camera_pairs,
            dtype=np.dtype(np.uint8),
            shape=shape,
            label="camera_pairs",
        )
        if (
            _array_sha256(before) != self.preconditional_camera_sha256
            or _array_sha256(after) != self.camera_sha256
            or _array_sha256(before[:, 1]) != self.exact_y1_sha256
            or not np.array_equal(after[:, 1], before[:, 1])
            or self.changed_y0_values != int(np.count_nonzero(after[:, 0] != before[:, 0]))
            or self.changed_y0_pixels != int(np.count_nonzero(np.any(after[:, 0] != before[:, 0], axis=-1)))
            or self.deterministic_double_decode is not True
            or self.exact_y1_preserved is not True
            or self.all_chunk_pairs_addressed is not True
            or self.outer_zip_score_admission != OUTER_ZIP_SCORE_ADMISSION
            or self.research_only is not True
            or self.candidate_claim is not False
            or self.score_claim is not False
            or self.promotion_eligible is not False
            or self.pointer_moved is not False
        ):
            raise PopulationPosePreimageChartError("G95 receiver hashes, ownership, counts, or truth labels differ")
        object.__setattr__(
            self,
            "preconditional_camera_pairs",
            _immutable_copy(before, dtype=np.dtype(np.uint8)),
        )
        object.__setattr__(
            self,
            "camera_pairs",
            _immutable_copy(after, dtype=np.dtype(np.uint8)),
        )


@dataclass(frozen=True, slots=True, init=False)
class PopulationPosePreimageChartReceiverV1:
    basis: ParsedPopulationPosePreimageBasisV1
    _basis_identity: int

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("PopulationPosePreimageChartReceiverV1 must be constructed through .open()")

    @classmethod
    def open(
        cls,
        basis: ParsedPopulationPosePreimageBasisV1,
        *,
        expected_g94_product_member_sha256: str,
        expected_g94_conditioning_state_sha256: str,
        expected_whole_preconditional_camera_sha256: str,
        expected_selected_target_table_sha256: str,
        expected_posenet_weights_sha256: str,
    ) -> PopulationPosePreimageChartReceiverV1:
        if type(basis) is not ParsedPopulationPosePreimageBasisV1:
            raise PopulationPosePreimageChartError("receiver requires an exact parsed P-once basis")
        expected = (
            _require_sha256(
                expected_g94_product_member_sha256,
                label="expected_g94_product_member_sha256",
            ),
            _require_sha256(
                expected_g94_conditioning_state_sha256,
                label="expected_g94_conditioning_state_sha256",
            ),
            _require_sha256(
                expected_whole_preconditional_camera_sha256,
                label="expected_whole_preconditional_camera_sha256",
            ),
            _require_sha256(
                expected_selected_target_table_sha256,
                label="expected_selected_target_table_sha256",
            ),
            _require_sha256(
                expected_posenet_weights_sha256,
                label="expected_posenet_weights_sha256",
            ),
        )
        actual = (
            basis.g94_product_member_sha256,
            basis.g94_conditioning_state_sha256,
            basis.whole_preconditional_camera_sha256,
            basis.selected_target_table_sha256,
            basis.posenet_weights_sha256,
        )
        if actual != expected:
            raise PopulationPosePreimageChartError("receiver whole-state foreign key mismatch")
        instance = object.__new__(cls)
        object.__setattr__(instance, "basis", basis)
        object.__setattr__(instance, "_basis_identity", id(basis))
        return instance

    def _decode_once(
        self,
        chunk: ParsedPopulationPosePreimageCoefficientChunkV1,
        preconditional_camera_pairs: np.ndarray,
    ) -> np.ndarray:
        if id(self.basis) != self._basis_identity:
            raise PopulationPosePreimageChartError("receiver basis custody drifted")
        if type(chunk) is not ParsedPopulationPosePreimageCoefficientChunkV1:
            raise PopulationPosePreimageChartError("receiver requires an exact parsed coefficient chunk")
        if chunk.basis_object_sha256 != self.basis.object_sha256:
            raise PopulationPosePreimageChartError("chunk basis reference mismatch")
        if chunk.population_state_key != self.basis.population_state_key:
            raise PopulationPosePreimageChartError("chunk whole-state reference mismatch")
        if chunk.rank != self.basis.rank:
            raise PopulationPosePreimageChartError("chunk rank differs from basis")
        shape = (len(chunk.source_pair_ids), 2, CAMERA_HEIGHT, CAMERA_WIDTH, CHANNELS)
        before = _require_exact_array(
            preconditional_camera_pairs,
            dtype=np.dtype(np.uint8),
            shape=shape,
            label="preconditional_camera_pairs",
        )
        if _array_sha256(before) != chunk.preconditional_camera_sha256:
            raise PopulationPosePreimageChartError("receiver preconditional bytes differ from chunk-bound state")
        basis = self.basis.basis_q.astype(np.float32) * self.basis.basis_scales.reshape(
            self.basis.rank,
            1,
            1,
            1,
        )
        coefficients = chunk.coefficients_q.astype(np.float32) * chunk.coefficient_scales.reshape(
            1,
            self.basis.rank,
        )
        residual_grid = np.zeros(
            (
                len(chunk.source_pair_ids),
                self.basis.grid_height,
                self.basis.grid_width,
                CHANNELS,
            ),
            dtype=np.float32,
        )
        for rank_index in range(self.basis.rank):
            residual_grid += coefficients[:, rank_index, None, None, None] * basis[rank_index][None]
        residual_camera = bilinear_resize_align_corners_false_numpy(
            residual_grid,
            output_height=CAMERA_HEIGHT,
            output_width=CAMERA_WIDTH,
        )
        result = np.ascontiguousarray(before).copy()
        result[:, 0] = np.rint(
            np.clip(
                before[:, 1].astype(np.float32) + residual_camera,
                np.float32(0.0),
                np.float32(255.0),
            )
        ).astype(np.uint8)
        if not np.array_equal(result[:, 1], before[:, 1]):
            raise PopulationPosePreimageChartError("receiver changed exact conditional Y1")
        return result

    def decode_preconditional_chunk(
        self,
        chunk: ParsedPopulationPosePreimageCoefficientChunkV1,
        preconditional_camera_pairs: np.ndarray,
    ) -> PopulationPosePreimageChartBatchResultV1:
        first = self._decode_once(chunk, preconditional_camera_pairs)
        second = self._decode_once(chunk, preconditional_camera_pairs)
        if not np.array_equal(first, second):
            raise PopulationPosePreimageChartError("deterministic double decode differs")
        before = np.asarray(preconditional_camera_pairs)
        return PopulationPosePreimageChartBatchResultV1(
            basis_object_sha256=self.basis.object_sha256,
            coefficient_chunk_sha256=chunk.object_sha256,
            population_state_key=self.basis.population_state_key,
            g94_product_member_sha256=self.basis.g94_product_member_sha256,
            g94_conditioning_state_sha256=self.basis.g94_conditioning_state_sha256,
            preconditional_camera_sha256=chunk.preconditional_camera_sha256,
            selected_target_sha256=chunk.selected_target_sha256,
            source_pair_ids=chunk.source_pair_ids,
            preconditional_camera_pairs=before,
            camera_pairs=first,
            camera_sha256=_array_sha256(first),
            exact_y1_sha256=_array_sha256(before[:, 1]),
            changed_y0_values=int(np.count_nonzero(first[:, 0] != before[:, 0])),
            changed_y0_pixels=int(np.count_nonzero(np.any(first[:, 0] != before[:, 0], axis=-1))),
        )


@dataclass(frozen=True, slots=True)
class RicherControlRequestV1:
    g94_product_member_sha256: str
    g94_conditioning_state_sha256: str
    source_pair_ids: tuple[int, ...]
    attempted_rank: int
    attempted_grid_height: int
    attempted_grid_width: int
    exact_d_pose: float
    reachability_threshold: float
    exact_residual: float
    requested_minimum_rank: int
    requested_minimum_grid_height: int
    requested_minimum_grid_width: int
    one_state_miss_axis: Literal["HIGHER_STATIC_RANK_OR_GRID"] = ONE_STATE_MISS_AXIS
    population_transfer_failure_classification: Literal[
        "Y1_CONDITIONED_SHARED_GENERATOR_OR_FEATURE_MODULATED_BASIS"
    ] = POPULATION_TRANSFER_REQUEST
    population_transfer_request: Literal["Y1_CONDITIONED_SHARED_GENERATOR_OR_FEATURE_MODULATED_BASIS"] = (
        POPULATION_TRANSFER_REQUEST
    )
    pair0_population_viability_claim: Literal[False] = False
    formulation_scope: Literal[
        "G95_POPULATION_SHARED_LINEAR_LOW_RESIDUAL_POSE_PREIMAGE_CHART_BOUND_TO_EXACT_G94_CONDITIONING_STATE"
    ] = FORMULATION_SCOPE
    missing_integration: Literal["G88_G94_NEW_TYPED_G95_POPULATION_CHART_MODE_AND_OUTER_ARCHIVE_RACE_OWED"] = (
        MISSING_INTEGRATION
    )
    family_dead_claim: Literal[False] = False
    research_only: Literal[True] = True
    candidate_claim: Literal[False] = False
    score_claim: Literal[False] = False
    promotion_eligible: Literal[False] = False
    pointer_moved: Literal[False] = False

    def __post_init__(self) -> None:
        _require_sha256(self.g94_product_member_sha256, label="g94_product_member_sha256")
        _require_sha256(
            self.g94_conditioning_state_sha256,
            label="g94_conditioning_state_sha256",
        )
        _validate_chunk_pair_ids(self.source_pair_ids)
        for value, label, maximum in (
            (self.attempted_rank, "attempted_rank", MAX_RANK),
            (self.requested_minimum_rank, "requested_minimum_rank", MAX_RANK),
            (self.attempted_grid_height, "attempted_grid_height", MAX_GRID_HEIGHT),
            (self.requested_minimum_grid_height, "requested_minimum_grid_height", MAX_GRID_HEIGHT),
            (self.attempted_grid_width, "attempted_grid_width", MAX_GRID_WIDTH),
            (self.requested_minimum_grid_width, "requested_minimum_grid_width", MAX_GRID_WIDTH),
        ):
            _exact_int(value, label=label, minimum=1, maximum=maximum)
        if (
            type(self.exact_d_pose) is not float
            or not math.isfinite(self.exact_d_pose)
            or self.reachability_threshold != REACHABILITY_THRESHOLD
            or self.exact_d_pose <= self.reachability_threshold
            or not math.isclose(
                self.exact_residual,
                self.exact_d_pose - self.reachability_threshold,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or (
                self.requested_minimum_rank <= self.attempted_rank
                and self.requested_minimum_grid_height <= self.attempted_grid_height
                and self.requested_minimum_grid_width <= self.attempted_grid_width
            )
            or self.one_state_miss_axis != ONE_STATE_MISS_AXIS
            or self.population_transfer_failure_classification != POPULATION_TRANSFER_REQUEST
            or self.population_transfer_request != POPULATION_TRANSFER_REQUEST
            or self.pair0_population_viability_claim is not False
            or self.formulation_scope != FORMULATION_SCOPE
            or self.missing_integration != MISSING_INTEGRATION
            or self.family_dead_claim is not False
            or self.research_only is not True
            or self.candidate_claim is not False
            or self.score_claim is not False
            or self.promotion_eligible is not False
            or self.pointer_moved is not False
        ):
            raise PopulationPosePreimageChartError(
                "G95 richer-control request is unscoped, inconsistent, or permissive"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "tac.g95_richer_control_request.v1",
            "g94_product_member_sha256": self.g94_product_member_sha256,
            "g94_conditioning_state_sha256": self.g94_conditioning_state_sha256,
            "source_pair_ids": list(self.source_pair_ids),
            "attempted": {
                "rank": self.attempted_rank,
                "grid_height": self.attempted_grid_height,
                "grid_width": self.attempted_grid_width,
                "exact_d_pose": self.exact_d_pose,
                "reachability_threshold": self.reachability_threshold,
                "exact_residual": self.exact_residual,
            },
            "requested_minimum": {
                "rank": self.requested_minimum_rank,
                "grid_height": self.requested_minimum_grid_height,
                "grid_width": self.requested_minimum_grid_width,
            },
            "one_state_miss_axis": ONE_STATE_MISS_AXIS,
            "population_transfer_failure_classification": POPULATION_TRANSFER_REQUEST,
            "population_transfer_request": POPULATION_TRANSFER_REQUEST,
            "pair0_population_viability_claim": False,
            "formulation_scope": FORMULATION_SCOPE,
            "missing_integration": MISSING_INTEGRATION,
            "family_dead_claim": False,
            "research_only": True,
            "candidate_claim": False,
            "score_claim": False,
            "promotion_eligible": False,
            "pointer_moved": False,
        }


def richer_control_request_for_miss(
    *,
    g94_product_member_sha256: str,
    g94_conditioning_state_sha256: str,
    source_pair_ids: tuple[int, ...],
    attempted_rank: int,
    attempted_grid_height: int,
    attempted_grid_width: int,
    exact_d_pose: float,
    reachability_threshold: float,
    requested_minimum_rank: int,
    requested_minimum_grid_height: int,
    requested_minimum_grid_width: int,
) -> RicherControlRequestV1:
    d_pose = float(exact_d_pose)
    threshold = float(reachability_threshold)
    if threshold != REACHABILITY_THRESHOLD:
        raise PopulationPosePreimageChartError("reachability_threshold must equal frozen exact 0.00047366")
    return RicherControlRequestV1(
        g94_product_member_sha256=g94_product_member_sha256,
        g94_conditioning_state_sha256=g94_conditioning_state_sha256,
        source_pair_ids=source_pair_ids,
        attempted_rank=attempted_rank,
        attempted_grid_height=attempted_grid_height,
        attempted_grid_width=attempted_grid_width,
        exact_d_pose=d_pose,
        reachability_threshold=threshold,
        exact_residual=d_pose - threshold,
        requested_minimum_rank=requested_minimum_rank,
        requested_minimum_grid_height=requested_minimum_grid_height,
        requested_minimum_grid_width=requested_minimum_grid_width,
    )


__all__ = [
    "BASE_LAW_ID",
    "BASIS_MAGIC",
    "BILINEAR_REFERENCE_ID",
    "BOUND_G94_CONDITIONING_STATE_SHA256",
    "BOUND_G94_PARENT_COMMIT",
    "BOUND_G94_PRODUCT_MEMBER_SHA256",
    "CHUNK_MAGIC",
    "FORMULATION_SCOPE",
    "MAX_BATCH_PAIRS",
    "MISSING_INTEGRATION",
    "MISS_VERDICT_SCOPE",
    "ONE_STATE_MISS_AXIS",
    "OUTER_ZIP_SCORE_ADMISSION",
    "POPULATION_TRANSFER_REQUEST",
    "REACHABILITY_COORDINATE_SCOPE",
    "REACHABILITY_THRESHOLD",
    "RECEIVER_ID",
    "ROUNDING_POLICY_ID",
    "WIRE_POLICY_ID",
    "G95ControlModeV1",
    "ParsedPopulationPosePreimageBasisV1",
    "ParsedPopulationPosePreimageCoefficientChunkV1",
    "PopulationPosePreimageChartBatchResultV1",
    "PopulationPosePreimageChartError",
    "PopulationPosePreimageChartReceiverV1",
    "PopulationPosePreimageChartWireSetV1",
    "RicherControlRequestV1",
    "bilinear_resize_align_corners_false_numpy",
    "encode_population_pose_preimage_basis",
    "encode_population_pose_preimage_coefficient_chunk",
    "parse_population_pose_preimage_basis",
    "parse_population_pose_preimage_coefficient_chunk",
    "population_pair_ids_sha256",
    "population_state_key",
    "richer_control_request_for_miss",
    "validate_population_chunk_coverage",
]
