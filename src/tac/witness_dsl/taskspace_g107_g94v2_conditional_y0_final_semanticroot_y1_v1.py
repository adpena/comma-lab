# SPDX-License-Identifier: MIT
"""G107: one conditional Y0 owner over a frozen final SemanticRootY1V1.

The counted object is the direct product

    frozen SemanticRootY1V1 * conditional-Y0(basis, scales, coefficients)

There is no union arm, second temporal stream, raw RGB-gauge table, scorer,
teacher, source video, target table, or historical payload decoder.  The
receiver first reconstructs the exact ordered full-n600 final Y1 population,
checks its externally frozen binding, and only then derives Y0 as a compact
conditional residual over that Y1.  Both scorer planes are realizable through
the public V10 factor-2 operator.

This module is generic receiver and accounting machinery.  It does not fit the
operands, establish fresh source custody, close public ``inflate.sh``, or claim
an evaluator score.  Encoder-only lineage and exact G17 ownership adapters are
external to the counted packet.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Final, Literal

import numpy as np

from tac.contest_score import (
    UNCOMPRESSED_SIZE_BYTES,
    compute_contest_score,
    pose_term,
    rate_term,
    seg_term,
)
from tac.optimization.uint8_lattice_feasibility import (
    DisjointResizeOperator,
    Factor2ExactVerification,
    realize_factor2_uint8_scorer_plane,
    verify_factor2_uint8_scorer_plane,
)
from tac.witness_dsl.taskspace_g95_population_pose_preimage_chart_v1 import (
    bilinear_resize_align_corners_false_numpy,
)
from tac.witness_dsl.taskspace_outer_archive_codec import (
    TaskspaceOuterArchiveBuild,
    build_taskspace_outer_archive,
)
from tac.witness_dsl.taskspace_pfree_semantic_root_v1 import (
    PAIR_COUNT_N600,
    SCORER_CHANNELS,
    SCORER_H,
    SCORER_W,
    RGBGaugeOwnershipV1,
    SemanticRootY1V1,
    bind_semantic_root_to_g17,
    encode_semantic_root_y1_v1,
    final_semantic_root_y1_binding_sha256,
    parse_semantic_root_y1_v1,
    render_semantic_root_y1_scorer,
    semantic_root_y1_population_sha256,
)
from tac.witness_dsl.taskspace_selected_solution_compiler import (
    G17ChronologicalPosePreimageV1,
    G17EncoderOnlyTeacherOracleEvidenceV1,
    G17LogicalOwnershipKindV1,
    G17LogicalOwnershipV1,
    G17PairPopulationV1,
)

MAGIC: Final = b"G107CY01"
VERSION: Final = 1
MAX_PACKET_BYTES: Final = 2_000_000
MAX_RANK: Final = 64
MAX_GRID_SIDE: Final = 64
MAX_STREAM_BATCH_PAIRS: Final = 16
CAMERA_H: Final = 874
CAMERA_W: Final = 1164
UPSTREAM_POSE_BATCH_SIZE: Final = 16
POSE_TARGET_CONTRACT_ID: Final = "UPSTREAM_POSENET_SOURCE_TARGET_ORDERED_N600_BATCH16_V1"
POSE_CANDIDATE_CONTRACT_ID: Final = "UPSTREAM_POSENET_FINAL_Y0_Y1_ORDERED_N600_BATCH16_V1"
CONDITIONAL_OWNER_ID: Final = "G107_G94V2_SOLE_CONDITIONAL_Y0_OWNER"
WIRE_POLICY_ID: Final = "FROZEN_FINAL_SEMANTIC_ROOT_Y1_TIMES_ONE_CONDITIONAL_Y0_OWNER_V1"
PUBLIC_RECEIVER_BLOCKER: Final = "G107_PUBLIC_INFLATE_SH_RECURSIVE_RUNTIME_CLOSURE_OWED"
FRESH_SOURCE_LINEAGE_BLOCKER: Final = "G107_FRESH_OWN_LINEAGE_CONDITIONAL_Y0_OPERANDS_OWED"
EXACT_AUTHORITY_BLOCKER: Final = "G107_SAME_ARCHIVE_UPSTREAM_EVALUATE_PY_N600_CPU_CUDA_OWED"
OPEN_BLOCKERS: Final = (
    PUBLIC_RECEIVER_BLOCKER,
    FRESH_SOURCE_LINEAGE_BLOCKER,
    EXACT_AUTHORITY_BLOCKER,
)

_HEADER: Final = struct.Struct(">8sBBHHHBBHHIIIII32s")
_FOOTER: Final = struct.Struct(">I")
_F32_BE: Final = np.dtype(">f4")
_I16_BE: Final = np.dtype(">i2")
_SECTION_NAMES: Final = (
    "semantic_root_y1",
    "conditional_basis_q",
    "conditional_basis_scales",
    "conditional_coefficients_q",
    "conditional_coefficient_scales",
)
_FORBIDDEN_NESTED_MAGICS: Final = (
    b"PK\x03\x04",
    b"DDV15S1",
    b"TACPVSA",
    b"PVSA",
    b"G95BAS1",
    b"G95CHK1",
    b"TSPPV1",
    b"TSPPV2",
)
# Exact historical whole-payload identities already denied by SemanticRootY1V1.
# Repeating them here closes direct reuse in any new conditional section.
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


class G107ConditionalY0Error(ValueError):
    """The G107 wire, frozen-Y1 binding, ownership, or decode failed."""


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return _sha256(memoryview(np.ascontiguousarray(value)).cast("B"))


def _require_sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise G107ConditionalY0Error(f"{name} must be canonical lowercase SHA-256")
    return value


def _exact_int(value: object, *, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise G107ConditionalY0Error(f"{name} must be an exact integer in [{minimum},{maximum}]")
    return value


def _canonical_population(population: G17PairPopulationV1) -> None:
    if type(population) is not G17PairPopulationV1:
        raise G107ConditionalY0Error("population must be canonical G17PairPopulationV1")
    exact = tuple(range(PAIR_COUNT_N600))
    if (
        population.global_pair_ids != exact
        or population.source_pair_ids != exact
        or population.v9_pair_coordinates != exact
        or population.pbr_pair_coordinates != exact
        or population.obligation_ir_coordinates != exact
        or population.v10_local_coordinates != exact
    ):
        raise G107ConditionalY0Error("G17 population is not the exact ordered full-n600 identity map")


def _reject_foreign_payload(payload: bytes, *, name: str) -> None:
    if _sha256(payload) in _FORBIDDEN_WHOLE_PAYLOAD_SHA256:
        raise G107ConditionalY0Error(f"{name} reuses a denied historical payload identity")
    if any(payload.startswith(magic) for magic in _FORBIDDEN_NESTED_MAGICS):
        raise G107ConditionalY0Error(f"{name} begins with a denied historical/raster payload magic")


def _require_root_ownership(root: SemanticRootY1V1) -> None:
    if type(root) is not SemanticRootY1V1:
        raise G107ConditionalY0Error("conditional packet requires exact SemanticRootY1V1")
    if root.rgb_gauge_ownership is not RGBGaugeOwnershipV1.DERIVED_BY_SHARED_GENERATOR or root.pair_rgb_gauges != ():
        raise G107ConditionalY0Error(
            "frozen Y1 must own RGB gauge through its shared generator; duplicate raw pair gauges are forbidden"
        )


def _immutable_array(
    value: object,
    *,
    dtype: np.dtype[object] | type[np.generic],
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype != np.dtype(dtype) or raw.shape != shape or not raw.flags.c_contiguous:
        raise G107ConditionalY0Error(f"{name} must be C-contiguous {np.dtype(dtype)} with shape {shape}")
    result = np.array(raw, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _conditional_sections(
    *,
    basis_q: np.ndarray,
    basis_scales: np.ndarray,
    coefficients_q: np.ndarray,
    coefficient_scales: np.ndarray,
) -> tuple[bytes, bytes, bytes, bytes]:
    return (
        np.asarray(basis_q, dtype=np.int8).tobytes(order="C"),
        np.asarray(basis_scales, dtype=_F32_BE).tobytes(order="C"),
        np.asarray(coefficients_q, dtype=_I16_BE).tobytes(order="C"),
        np.asarray(coefficient_scales, dtype=_F32_BE).tobytes(order="C"),
    )


def _validate_conditional_operands(
    *,
    basis_q: np.ndarray,
    basis_scales: np.ndarray,
    coefficients_q: np.ndarray,
    coefficient_scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if basis_q.ndim != 4 or basis_q.shape[-1] != SCORER_CHANNELS:
        raise G107ConditionalY0Error("basis_q must have shape [rank,grid_h,grid_w,3]")
    rank, grid_h, grid_w, _ = basis_q.shape
    _exact_int(rank, name="rank", minimum=1, maximum=MAX_RANK)
    _exact_int(grid_h, name="grid_h", minimum=1, maximum=MAX_GRID_SIDE)
    _exact_int(grid_w, name="grid_w", minimum=1, maximum=MAX_GRID_SIDE)
    basis = _immutable_array(
        basis_q,
        dtype=np.int8,
        shape=(rank, grid_h, grid_w, SCORER_CHANNELS),
        name="basis_q",
    )
    basis_scale = _immutable_array(
        basis_scales,
        dtype=np.float32,
        shape=(rank,),
        name="basis_scales",
    )
    coefficients = _immutable_array(
        coefficients_q,
        dtype=np.int16,
        shape=(PAIR_COUNT_N600, rank),
        name="coefficients_q",
    )
    coefficient_scale = _immutable_array(
        coefficient_scales,
        dtype=np.float32,
        shape=(rank,),
        name="coefficient_scales",
    )
    if not np.all(np.isfinite(basis_scale)) or np.any(basis_scale <= 0):
        raise G107ConditionalY0Error("basis scales must be positive finite float32")
    if not np.all(np.isfinite(coefficient_scale)) or np.any(coefficient_scale <= 0):
        raise G107ConditionalY0Error("coefficient scales must be positive finite float32")
    if np.any(np.all(basis == 0, axis=(1, 2, 3))):
        raise G107ConditionalY0Error("every conditional rank must have a receiver-live basis value")
    if np.any(np.all(coefficients == 0, axis=1)):
        raise G107ConditionalY0Error("every one of the ordered n600 rows must own a conditional Y0 value")
    if np.any(np.all(coefficients == 0, axis=0)):
        raise G107ConditionalY0Error("every conditional rank must be used by the n600 coefficient population")
    decoded_grid = np.einsum(
        "pr,rhwc,r,r->phwc",
        coefficients.astype(np.float32),
        basis.astype(np.float32),
        basis_scale,
        coefficient_scale,
        optimize=True,
        dtype=np.float32,
    )
    if not np.all(np.isfinite(decoded_grid)) or np.any(np.all(decoded_grid == 0, axis=(1, 2, 3))):
        raise G107ConditionalY0Error("every conditional row must produce a finite nonzero Y0 residual")
    return basis, basis_scale, coefficients, coefficient_scale


@dataclass(frozen=True, slots=True, eq=False)
class ParsedG107ConditionalY0V1:
    """Strict counted packet reopened into exact typed receiver operands."""

    root: SemanticRootY1V1
    final_y1_binding_sha256: str
    basis_q: np.ndarray = field(repr=False)
    basis_scales: np.ndarray = field(repr=False)
    coefficients_q: np.ndarray = field(repr=False)
    coefficient_scales: np.ndarray = field(repr=False)
    section_bytes: tuple[bytes, bytes, bytes, bytes, bytes] = field(repr=False)
    packet_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _require_root_ownership(self.root)
        _require_sha256(self.final_y1_binding_sha256, name="final Y1 binding")
        if type(self.section_bytes) is not tuple or len(self.section_bytes) != len(_SECTION_NAMES):
            raise G107ConditionalY0Error("packet must expose exactly five counted sections")
        if any(type(section) is not bytes for section in self.section_bytes):
            raise G107ConditionalY0Error("counted sections must remain immutable bytes")
        if type(self.packet_bytes) is not bytes:
            raise G107ConditionalY0Error("packet custody must remain exact immutable bytes")
        basis, basis_scale, coefficients, coefficient_scale = _validate_conditional_operands(
            basis_q=self.basis_q,
            basis_scales=self.basis_scales,
            coefficients_q=self.coefficients_q,
            coefficient_scales=self.coefficient_scales,
        )
        expected_sections = (
            encode_semantic_root_y1_v1(self.root),
            *_conditional_sections(
                basis_q=basis,
                basis_scales=basis_scale,
                coefficients_q=coefficients,
                coefficient_scales=coefficient_scale,
            ),
        )
        if self.section_bytes != expected_sections:
            raise G107ConditionalY0Error("parsed arrays do not exactly spell the counted sections")
        expected_packet = _encode_packet(
            root_packet=expected_sections[0],
            final_y1_binding_sha256=self.final_y1_binding_sha256,
            rank=int(basis.shape[0]),
            grid_h=int(basis.shape[1]),
            grid_w=int(basis.shape[2]),
            conditional_sections=expected_sections[1:],
        )
        if self.packet_bytes != expected_packet:
            raise G107ConditionalY0Error("parsed object does not retain exact canonical packet custody")
        object.__setattr__(self, "basis_q", basis)
        object.__setattr__(self, "basis_scales", basis_scale)
        object.__setattr__(self, "coefficients_q", coefficients)
        object.__setattr__(self, "coefficient_scales", coefficient_scale)

    @property
    def rank(self) -> int:
        return int(self.basis_q.shape[0])

    @property
    def grid_height(self) -> int:
        return int(self.basis_q.shape[1])

    @property
    def grid_width(self) -> int:
        return int(self.basis_q.shape[2])

    @property
    def packet_sha256(self) -> str:
        return _sha256(self.packet_bytes)

    @property
    def conditional_operand_bytes(self) -> bytes:
        parts = self.section_bytes[1:]
        return b"G107-CONDITIONAL-Y0-OPERANDS-V1\0" + b"".join(struct.pack(">I", len(part)) + part for part in parts)

    @property
    def conditional_operand_sha256(self) -> str:
        return _sha256(self.conditional_operand_bytes)


def _encode_packet(
    *,
    root_packet: bytes,
    final_y1_binding_sha256: str,
    rank: int,
    grid_h: int,
    grid_w: int,
    conditional_sections: tuple[bytes, bytes, bytes, bytes],
) -> bytes:
    sections = (root_packet, *conditional_sections)
    for name, section in zip(_SECTION_NAMES, sections, strict=True):
        _reject_foreign_payload(section, name=name)
    header = _HEADER.pack(
        MAGIC,
        VERSION,
        0,
        PAIR_COUNT_N600,
        SCORER_H,
        SCORER_W,
        SCORER_CHANNELS,
        rank,
        grid_h,
        grid_w,
        *(len(section) for section in sections),
        bytes.fromhex(_require_sha256(final_y1_binding_sha256, name="final Y1 binding")),
    )
    body = b"".join(sections)
    packet = header + body + _FOOTER.pack(zlib.crc32(body) & 0xFFFFFFFF)
    if len(packet) > MAX_PACKET_BYTES:
        raise G107ConditionalY0Error("packet exceeds compact typed cap; dense conditional plane refused")
    return packet


def compute_final_y1_binding(
    root: SemanticRootY1V1,
    *,
    population: G17PairPopulationV1,
    batch_size: int = MAX_STREAM_BATCH_PAIRS,
) -> str:
    """Hash the actual ordered n600 decoded Y1 population into its final bind."""

    _require_root_ownership(root)
    _canonical_population(population)
    _exact_int(batch_size, name="batch_size", minimum=1, maximum=MAX_STREAM_BATCH_PAIRS)
    root_packet = encode_semantic_root_y1_v1(root)
    scorer_population_sha = semantic_root_y1_population_sha256(root, batch_size=batch_size)
    return final_semantic_root_y1_binding_sha256(
        root_packet_sha256=_sha256(root_packet),
        g17_population_binding_sha256=population.binding_sha256,
        scorer_y1_population_sha256=scorer_population_sha,
    )


def encode_g107_conditional_y0_v1(
    root: SemanticRootY1V1,
    *,
    population: G17PairPopulationV1,
    expected_final_y1_binding_sha256: str,
    basis_q: np.ndarray,
    basis_scales: np.ndarray,
    coefficients_q: np.ndarray,
    coefficient_scales: np.ndarray,
    batch_size: int = MAX_STREAM_BATCH_PAIRS,
) -> bytes:
    """Encode only after actual decoded full-n600 Y1 matches the frozen bind."""

    _require_root_ownership(root)
    expected = _require_sha256(expected_final_y1_binding_sha256, name="expected final Y1 binding")
    actual = compute_final_y1_binding(root, population=population, batch_size=batch_size)
    if actual != expected:
        raise G107ConditionalY0Error("wrong frozen final Y1 population binding")
    basis, basis_scale, coefficients, coefficient_scale = _validate_conditional_operands(
        basis_q=basis_q,
        basis_scales=basis_scales,
        coefficients_q=coefficients_q,
        coefficient_scales=coefficient_scales,
    )
    conditional_sections = _conditional_sections(
        basis_q=basis,
        basis_scales=basis_scale,
        coefficients_q=coefficients,
        coefficient_scales=coefficient_scale,
    )
    return _encode_packet(
        root_packet=encode_semantic_root_y1_v1(root),
        final_y1_binding_sha256=actual,
        rank=int(basis.shape[0]),
        grid_h=int(basis.shape[1]),
        grid_w=int(basis.shape[2]),
        conditional_sections=conditional_sections,
    )


def parse_g107_conditional_y0_v1(
    payload: bytes,
    *,
    population: G17PairPopulationV1,
    expected_final_y1_binding_sha256: str,
    batch_size: int = MAX_STREAM_BATCH_PAIRS,
) -> ParsedG107ConditionalY0V1:
    """Strict parse-back with EOF, CRC, canonical re-emission, and Y1 replay."""

    if type(payload) is not bytes:
        raise G107ConditionalY0Error("packet must be exact immutable bytes")
    if not _HEADER.size + _FOOTER.size <= len(payload) <= MAX_PACKET_BYTES:
        raise G107ConditionalY0Error("packet is truncated or exceeds the compact typed cap")
    _canonical_population(population)
    expected = _require_sha256(expected_final_y1_binding_sha256, name="expected final Y1 binding")
    values = _HEADER.unpack_from(payload)
    (
        magic,
        version,
        flags,
        pair_count,
        scorer_h,
        scorer_w,
        channels,
        rank,
        grid_h,
        grid_w,
        root_length,
        basis_length,
        basis_scale_length,
        coefficient_length,
        coefficient_scale_length,
        binding_raw,
    ) = values
    if magic != MAGIC or version != VERSION or flags != 0:
        raise G107ConditionalY0Error("packet magic/version/flags changed")
    if (pair_count, scorer_h, scorer_w, channels) != (
        PAIR_COUNT_N600,
        SCORER_H,
        SCORER_W,
        SCORER_CHANNELS,
    ):
        raise G107ConditionalY0Error("packet is not exact ordered n600 scorer geometry")
    _exact_int(rank, name="rank", minimum=1, maximum=MAX_RANK)
    _exact_int(grid_h, name="grid_h", minimum=1, maximum=MAX_GRID_SIDE)
    _exact_int(grid_w, name="grid_w", minimum=1, maximum=MAX_GRID_SIDE)
    expected_lengths = (
        root_length,
        rank * grid_h * grid_w * SCORER_CHANNELS,
        rank * 4,
        PAIR_COUNT_N600 * rank * 2,
        rank * 4,
    )
    observed_lengths = (
        root_length,
        basis_length,
        basis_scale_length,
        coefficient_length,
        coefficient_scale_length,
    )
    if observed_lengths != expected_lengths or root_length <= 0:
        raise G107ConditionalY0Error("typed section lengths disagree with rank/grid/n600 ABI")
    body_length = sum(observed_lengths)
    if _HEADER.size + body_length + _FOOTER.size != len(payload):
        raise G107ConditionalY0Error("packet has truncation or trailing bytes after exact EOF")
    cursor = _HEADER.size
    sections: list[bytes] = []
    for name, length in zip(_SECTION_NAMES, observed_lengths, strict=True):
        section = payload[cursor : cursor + length]
        cursor += length
        _reject_foreign_payload(section, name=name)
        sections.append(section)
    (expected_crc,) = _FOOTER.unpack_from(payload, cursor)
    if expected_crc != zlib.crc32(b"".join(sections)) & 0xFFFFFFFF:
        raise G107ConditionalY0Error("packet body CRC32 mismatch")
    root = parse_semantic_root_y1_v1(sections[0])
    _require_root_ownership(root)
    binding = binding_raw.hex()
    if binding != expected:
        raise G107ConditionalY0Error("packet does not name the externally frozen final Y1 binding")
    actual = compute_final_y1_binding(root, population=population, batch_size=batch_size)
    if actual != binding:
        raise G107ConditionalY0Error("embedded root decodes to the wrong frozen final Y1 population")
    basis = np.frombuffer(sections[1], dtype=np.int8).reshape(rank, grid_h, grid_w, SCORER_CHANNELS).copy()
    basis_scale = np.frombuffer(sections[2], dtype=_F32_BE).astype(np.float32)
    coefficients = np.frombuffer(sections[3], dtype=_I16_BE).astype(np.int16).reshape(PAIR_COUNT_N600, rank)
    coefficient_scale = np.frombuffer(sections[4], dtype=_F32_BE).astype(np.float32)
    parsed = ParsedG107ConditionalY0V1(
        root=root,
        final_y1_binding_sha256=binding,
        basis_q=np.ascontiguousarray(basis),
        basis_scales=np.ascontiguousarray(basis_scale),
        coefficients_q=np.ascontiguousarray(coefficients),
        coefficient_scales=np.ascontiguousarray(coefficient_scale),
        section_bytes=tuple(sections),  # type: ignore[arg-type]
        packet_bytes=payload,
    )
    canonical = _encode_packet(
        root_packet=sections[0],
        final_y1_binding_sha256=binding,
        rank=rank,
        grid_h=grid_h,
        grid_w=grid_w,
        conditional_sections=tuple(sections[1:]),  # type: ignore[arg-type]
    )
    if canonical != payload:
        raise G107ConditionalY0Error("packet is not canonical under strict re-emission")
    return parsed


def _conditional_grid(parsed: ParsedG107ConditionalY0V1, pair_id: int) -> np.ndarray:
    _exact_int(pair_id, name="pair_id", minimum=0, maximum=PAIR_COUNT_N600 - 1)
    weights = parsed.coefficients_q[pair_id].astype(np.float32) * parsed.coefficient_scales * parsed.basis_scales
    grid = np.einsum("r,rhwc->hwc", weights, parsed.basis_q.astype(np.float32), optimize=True)
    result = np.ascontiguousarray(grid, dtype=np.float32)
    if not np.all(np.isfinite(result)) or not np.any(result):
        raise G107ConditionalY0Error("conditional Y0 residual is dead or nonfinite")
    result.setflags(write=False)
    return result


def _render_y0_given_operands(
    *,
    root: SemanticRootY1V1,
    pair_id: int,
    basis_q: np.ndarray,
    basis_scales: np.ndarray,
    coefficients_q: np.ndarray,
    coefficient_scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared generic receiver equation used by decode and effect audit."""

    _exact_int(pair_id, name="pair_id", minimum=0, maximum=PAIR_COUNT_N600 - 1)
    y1 = render_semantic_root_y1_scorer(root, pair_id)
    if (
        type(y1) is not np.ndarray
        or y1.dtype != np.uint8
        or y1.shape
        != (
            SCORER_H,
            SCORER_W,
            SCORER_CHANNELS,
        )
    ):
        raise G107ConditionalY0Error("SemanticRoot Y1 receiver violated scorer RGB ABI")
    weights = coefficients_q[pair_id].astype(np.float32) * coefficient_scales * basis_scales
    grid = np.einsum("r,rhwc->hwc", weights, basis_q.astype(np.float32), optimize=True)
    residual = bilinear_resize_align_corners_false_numpy(
        np.ascontiguousarray(grid, dtype=np.float32),
        output_height=SCORER_H,
        output_width=SCORER_W,
    )
    y0 = np.clip(np.rint(y1.astype(np.float32) + residual), 0, 255).astype(np.uint8)
    if not np.any(y0 != y1):
        raise G107ConditionalY0Error("conditional owner did not affect realized Y0")
    return np.ascontiguousarray(y0), np.ascontiguousarray(y1)


@dataclass(frozen=True, slots=True, eq=False)
class G107ScorerPairV1:
    pair_id: int
    scorer_pair: np.ndarray = field(repr=False)
    y1_sha256: str
    y0_sha256: str
    changed_y0_values: int
    final_y1_preserved: Literal[True] = True
    conditional_y0_owner_count: Literal[1] = 1

    def __post_init__(self) -> None:
        _exact_int(self.pair_id, name="pair_id", minimum=0, maximum=PAIR_COUNT_N600 - 1)
        pair = _immutable_array(
            self.scorer_pair,
            dtype=np.uint8,
            shape=(2, SCORER_H, SCORER_W, SCORER_CHANNELS),
            name="scorer_pair",
        )
        _require_sha256(self.y1_sha256, name="Y1 SHA")
        _require_sha256(self.y0_sha256, name="Y0 SHA")
        if (
            _array_sha256(pair[1]) != self.y1_sha256
            or _array_sha256(pair[0]) != self.y0_sha256
            or type(self.changed_y0_values) is not int
            or self.changed_y0_values <= 0
            or self.final_y1_preserved is not True
            or self.conditional_y0_owner_count != 1
        ):
            raise G107ConditionalY0Error("scorer-pair ownership or decode evidence differs")
        object.__setattr__(self, "scorer_pair", pair)


@dataclass(frozen=True, slots=True, eq=False)
class G107V10Factor2PairV1:
    pair_id: int
    camera_pair: np.ndarray = field(repr=False)
    scorer_pair_sha256: str
    y0_proof: Factor2ExactVerification
    y1_proof: Factor2ExactVerification
    both_scorer_planes_exact: Literal[True] = True

    def __post_init__(self) -> None:
        _exact_int(self.pair_id, name="pair_id", minimum=0, maximum=PAIR_COUNT_N600 - 1)
        pair = _immutable_array(
            self.camera_pair,
            dtype=np.uint8,
            shape=(2, CAMERA_H, CAMERA_W, SCORER_CHANNELS),
            name="camera_pair",
        )
        _require_sha256(self.scorer_pair_sha256, name="scorer pair SHA")
        if (
            type(self.y0_proof) is not Factor2ExactVerification
            or type(self.y1_proof) is not Factor2ExactVerification
            or not self.y0_proof.certified_exact
            or not self.y1_proof.certified_exact
            or self.both_scorer_planes_exact is not True
        ):
            raise G107ConditionalY0Error("both V10 scorer planes must be factor-2 exact")
        object.__setattr__(self, "camera_pair", pair)


@dataclass(frozen=True, slots=True, init=False)
class G107ConditionalY0ReceiverV1:
    """Source/scorer-free receiver over one parsed counted object."""

    parsed: ParsedG107ConditionalY0V1
    _parsed_identity: int
    _operator: DisjointResizeOperator

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("G107ConditionalY0ReceiverV1 must be constructed through .open()")

    @classmethod
    def open(cls, parsed: ParsedG107ConditionalY0V1) -> G107ConditionalY0ReceiverV1:
        if type(parsed) is not ParsedG107ConditionalY0V1:
            raise G107ConditionalY0Error("receiver requires exact parsed G107 packet")
        instance = object.__new__(cls)
        object.__setattr__(instance, "parsed", parsed)
        object.__setattr__(instance, "_parsed_identity", id(parsed))
        object.__setattr__(
            instance,
            "_operator",
            DisjointResizeOperator.build(
                camera_h=CAMERA_H,
                camera_w=CAMERA_W,
                scorer_h=SCORER_H,
                scorer_w=SCORER_W,
            ),
        )
        return instance

    def _custody(self) -> None:
        if id(self.parsed) != self._parsed_identity:
            raise G107ConditionalY0Error("receiver parsed-object custody drifted")

    def render_scorer_pair(self, pair_id: int) -> G107ScorerPairV1:
        self._custody()
        y0, y1 = _render_y0_given_operands(
            root=self.parsed.root,
            pair_id=pair_id,
            basis_q=self.parsed.basis_q,
            basis_scales=self.parsed.basis_scales,
            coefficients_q=self.parsed.coefficients_q,
            coefficient_scales=self.parsed.coefficient_scales,
        )
        changed = int(np.count_nonzero(y0 != y1))
        pair = np.ascontiguousarray(np.stack((y0, y1), axis=0))
        return G107ScorerPairV1(
            pair_id=pair_id,
            scorer_pair=pair,
            y1_sha256=_array_sha256(y1),
            y0_sha256=_array_sha256(y0),
            changed_y0_values=changed,
        )

    def render_scorer_batch(self, pair_ids: tuple[int, ...]) -> np.ndarray:
        if (
            type(pair_ids) is not tuple
            or not 1 <= len(pair_ids) <= MAX_STREAM_BATCH_PAIRS
            or any(type(pair_id) is not int for pair_id in pair_ids)
            or pair_ids != tuple(range(pair_ids[0], pair_ids[0] + len(pair_ids)))
            or pair_ids[0] < 0
            or pair_ids[-1] >= PAIR_COUNT_N600
        ):
            raise G107ConditionalY0Error("batch IDs must be one exact contiguous ordered slice of n600")
        batch = np.stack([self.render_scorer_pair(pair_id).scorer_pair for pair_id in pair_ids])
        result = np.ascontiguousarray(batch, dtype=np.uint8)
        result.setflags(write=False)
        return result

    def iter_n600(self, *, batch_size: int = MAX_STREAM_BATCH_PAIRS) -> Iterator[tuple[int, np.ndarray]]:
        _exact_int(batch_size, name="batch_size", minimum=1, maximum=MAX_STREAM_BATCH_PAIRS)
        observed = 0
        for start in range(0, PAIR_COUNT_N600, batch_size):
            stop = min(start + batch_size, PAIR_COUNT_N600)
            pair_ids = tuple(range(start, stop))
            batch = self.render_scorer_batch(pair_ids)
            if start != observed:
                raise AssertionError("internal G107 n600 ordering drifted")
            observed += len(pair_ids)
            yield start, batch
        if observed != PAIR_COUNT_N600:
            raise AssertionError("internal G107 n600 population is incomplete")

    def realize_v10_factor2_pair(self, pair_id: int) -> G107V10Factor2PairV1:
        scorer = self.render_scorer_pair(pair_id)
        camera_frames: list[np.ndarray] = []
        proofs: list[Factor2ExactVerification] = []
        for plane in scorer.scorer_pair:
            camera = realize_factor2_uint8_scorer_plane(self._operator, plane)
            proof = verify_factor2_uint8_scorer_plane(self._operator, camera, plane)
            if not proof.certified_exact:
                raise G107ConditionalY0Error("V10 factor-2 realization was not exact for both planes")
            camera_frames.append(camera)
            proofs.append(proof)
        return G107V10Factor2PairV1(
            pair_id=pair_id,
            camera_pair=np.ascontiguousarray(np.stack(camera_frames)),
            scorer_pair_sha256=_array_sha256(scorer.scorer_pair),
            y0_proof=proofs[0],
            y1_proof=proofs[1],
        )


@dataclass(frozen=True, slots=True)
class G107OperandEffectAuditV1:
    """Section-level perturbation receipt; it is behavior evidence, not a score."""

    pair_id: int
    baseline_y0_sha256: str
    basis_q_changes_y0: bool
    basis_scales_change_y0: bool
    coefficients_q_change_y0: bool
    coefficient_scales_change_y0: bool
    all_conditional_operands_receiver_live: Literal[True] = True
    research_only: Literal[True] = True
    score_claim: Literal[False] = False

    def __post_init__(self) -> None:
        _exact_int(self.pair_id, name="pair_id", minimum=0, maximum=PAIR_COUNT_N600 - 1)
        _require_sha256(self.baseline_y0_sha256, name="baseline Y0 SHA")
        flags = (
            self.basis_q_changes_y0,
            self.basis_scales_change_y0,
            self.coefficients_q_change_y0,
            self.coefficient_scales_change_y0,
        )
        if (
            any(type(flag) is not bool or not flag for flag in flags)
            or self.all_conditional_operands_receiver_live is not True
            or self.research_only is not True
            or self.score_claim is not False
        ):
            raise G107ConditionalY0Error("not every conditional operand changed realized Y0")


def audit_conditional_operand_effects(
    parsed: ParsedG107ConditionalY0V1,
    *,
    pair_id: int,
) -> G107OperandEffectAuditV1:
    """Perturb each of the four semantic operands and require Y0 movement."""

    if type(parsed) is not ParsedG107ConditionalY0V1:
        raise G107ConditionalY0Error("effect audit requires exact parsed packet")
    _exact_int(pair_id, name="pair_id", minimum=0, maximum=PAIR_COUNT_N600 - 1)

    def render(
        basis: np.ndarray,
        basis_scale: np.ndarray,
        coefficients: np.ndarray,
        coefficient_scale: np.ndarray,
    ) -> np.ndarray:
        y0, _ = _render_y0_given_operands(
            root=parsed.root,
            basis_q=np.ascontiguousarray(basis, dtype=np.int8),
            basis_scales=np.ascontiguousarray(basis_scale, dtype=np.float32),
            coefficients_q=np.ascontiguousarray(coefficients, dtype=np.int16),
            coefficient_scales=np.ascontiguousarray(coefficient_scale, dtype=np.float32),
            pair_id=pair_id,
        )
        return y0

    baseline = G107ConditionalY0ReceiverV1.open(parsed).render_scorer_pair(pair_id).scorer_pair[0]
    basis = parsed.basis_q.copy()
    basis[0, 0, 0, 0] = np.int8(-int(basis[0, 0, 0, 0]) or 1)
    basis_scale = parsed.basis_scales.copy()
    basis_scale[0] = np.float32(basis_scale[0] * np.float32(2.0))
    coefficients = parsed.coefficients_q.copy()
    original_coefficient = int(coefficients[pair_id, 0])
    coefficients[pair_id, 0] = np.int16(
        max(-32768, min(32767, -original_coefficient if original_coefficient != 0 else 1))
    )
    coefficient_scale = parsed.coefficient_scales.copy()
    coefficient_scale[0] = np.float32(coefficient_scale[0] * np.float32(2.0))
    changed = (
        not np.array_equal(
            render(basis, parsed.basis_scales, parsed.coefficients_q, parsed.coefficient_scales),
            baseline,
        ),
        not np.array_equal(
            render(parsed.basis_q, basis_scale, parsed.coefficients_q, parsed.coefficient_scales),
            baseline,
        ),
        not np.array_equal(
            render(parsed.basis_q, parsed.basis_scales, coefficients, parsed.coefficient_scales),
            baseline,
        ),
        not np.array_equal(
            render(parsed.basis_q, parsed.basis_scales, parsed.coefficients_q, coefficient_scale),
            baseline,
        ),
    )
    return G107OperandEffectAuditV1(
        pair_id=pair_id,
        baseline_y0_sha256=_array_sha256(baseline),
        basis_q_changes_y0=changed[0],
        basis_scales_change_y0=changed[1],
        coefficients_q_change_y0=changed[2],
        coefficient_scales_change_y0=changed[3],
    )


def conditional_y0_g17_logical_value(
    parsed: ParsedG107ConditionalY0V1,
) -> G17ChronologicalPosePreimageV1:
    if type(parsed) is not ParsedG107ConditionalY0V1:
        raise G107ConditionalY0Error("G17 value adapter requires exact parsed packet")
    return G17ChronologicalPosePreimageV1(parsed.conditional_operand_bytes)


def bind_g107_to_g17(
    parsed: ParsedG107ConditionalY0V1,
    *,
    population: G17PairPopulationV1,
    topology_owner: G17LogicalOwnershipV1,
    realization_owner: G17LogicalOwnershipV1,
    generator_owner: G17LogicalOwnershipV1,
    temporal_owner: G17LogicalOwnershipV1,
    quotient_owner: G17LogicalOwnershipV1,
    conditional_pose_owner: G17LogicalOwnershipV1,
) -> str:
    """Join existing G103 ownership with exactly one G17 Y0-fiber owner."""

    if type(parsed) is not ParsedG107ConditionalY0V1:
        raise G107ConditionalY0Error("G17 seam requires exact parsed packet")
    _canonical_population(population)
    root_binding = bind_semantic_root_to_g17(
        parsed.root,
        population=population,
        topology_owner=topology_owner,
        realization_owner=realization_owner,
        generator_owner=generator_owner,
        temporal_owner=temporal_owner,
        quotient_owner=quotient_owner,
    )
    expected_value = conditional_y0_g17_logical_value(parsed)
    if (
        type(conditional_pose_owner) is not G17LogicalOwnershipV1
        or conditional_pose_owner.ownership_kind is not G17LogicalOwnershipKindV1.CHRONOLOGICAL_POSE
        or type(conditional_pose_owner.value) is not G17ChronologicalPosePreimageV1
        or conditional_pose_owner.value.exact_bytes != expected_value.exact_bytes
    ):
        raise G107ConditionalY0Error("conditional Y0 bytes are not exactly retained by one G17 pose owner")
    digest = hashlib.sha256(b"G17-G107-G94V2-CONDITIONAL-Y0-FINAL-Y1-SEAM-V1\0")
    digest.update(bytes.fromhex(parsed.packet_sha256))
    digest.update(bytes.fromhex(parsed.final_y1_binding_sha256))
    digest.update(bytes.fromhex(population.binding_sha256))
    digest.update(bytes.fromhex(root_binding))
    digest.update(bytes.fromhex(conditional_pose_owner.identity_sha256))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class G107ConditionalY0SourceLineageV1:
    """External fresh-own-lineage evidence; never part of candidate bytes."""

    packet_sha256: str
    final_y1_binding_sha256: str
    conditional_operand_sha256: str
    source_video_sha256: str
    target_custody_sha256: str
    pose_target_binding_sha256: str
    pose_candidate_batch16_contract_sha256: str
    compiler_source_sha256: str
    compile_config_sha256: str
    originality_declaration_sha256: str
    pose_target_mode: Literal[
        "FRESH_UPSTREAM_BATCH16_TARGET",
        "EXACT_BATCH1_TO_BATCH16_PARITY_RECEIPT",
    ]
    compiler_id: Literal["G107_FRESH_CONDITIONAL_Y0_COMPILER_V1"] = "G107_FRESH_CONDITIONAL_Y0_COMPILER_V1"
    pose_target_contract_id: Literal["UPSTREAM_POSENET_SOURCE_TARGET_ORDERED_N600_BATCH16_V1"] = POSE_TARGET_CONTRACT_ID
    pose_candidate_contract_id: Literal["UPSTREAM_POSENET_FINAL_Y0_Y1_ORDERED_N600_BATCH16_V1"] = (
        POSE_CANDIDATE_CONTRACT_ID
    )
    pose_target_batch_size: Literal[16] = UPSTREAM_POSE_BATCH_SIZE
    pose_candidate_batch_size: Literal[16] = UPSTREAM_POSE_BATCH_SIZE
    legacy_batch1_gt_poses_reused: Literal[False] = False
    batch32_verdict_rows_are_authority: Literal[False] = False
    historical_payload_reused: Literal[False] = False
    encoder_only: Literal[True] = True

    def __post_init__(self) -> None:
        for name in (
            "packet_sha256",
            "final_y1_binding_sha256",
            "conditional_operand_sha256",
            "source_video_sha256",
            "target_custody_sha256",
            "pose_target_binding_sha256",
            "pose_candidate_batch16_contract_sha256",
            "compiler_source_sha256",
            "compile_config_sha256",
            "originality_declaration_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        if (
            self.compiler_id != "G107_FRESH_CONDITIONAL_Y0_COMPILER_V1"
            or self.pose_target_mode
            not in {
                "FRESH_UPSTREAM_BATCH16_TARGET",
                "EXACT_BATCH1_TO_BATCH16_PARITY_RECEIPT",
            }
            or self.pose_target_contract_id != POSE_TARGET_CONTRACT_ID
            or self.pose_candidate_contract_id != POSE_CANDIDATE_CONTRACT_ID
            or self.pose_target_batch_size != UPSTREAM_POSE_BATCH_SIZE
            or self.pose_candidate_batch_size != UPSTREAM_POSE_BATCH_SIZE
            or self.legacy_batch1_gt_poses_reused is not False
            or self.batch32_verdict_rows_are_authority is not False
            or self.historical_payload_reused is not False
            or self.encoder_only is not True
        ):
            raise G107ConditionalY0Error(
                "lineage must retain upstream-batch16 pose custody and remain fresh, external, and encoder-only"
            )


def encode_g107_source_lineage_manifest(
    parsed: ParsedG107ConditionalY0V1,
    lineage: G107ConditionalY0SourceLineageV1,
) -> bytes:
    if type(parsed) is not ParsedG107ConditionalY0V1 or type(lineage) is not G107ConditionalY0SourceLineageV1:
        raise G107ConditionalY0Error("lineage manifest requires exact parsed packet and lineage")
    if (
        lineage.packet_sha256 != parsed.packet_sha256
        or lineage.final_y1_binding_sha256 != parsed.final_y1_binding_sha256
        or lineage.conditional_operand_sha256 != parsed.conditional_operand_sha256
    ):
        raise G107ConditionalY0Error("external fresh lineage does not bind the exact packet and operands")
    data = {
        "compile_config_sha256": lineage.compile_config_sha256,
        "compiler_id": lineage.compiler_id,
        "compiler_source_sha256": lineage.compiler_source_sha256,
        "conditional_operand_sha256": lineage.conditional_operand_sha256,
        "encoder_only": lineage.encoder_only,
        "final_y1_binding_sha256": lineage.final_y1_binding_sha256,
        "batch32_verdict_rows_are_authority": lineage.batch32_verdict_rows_are_authority,
        "historical_payload_reused": lineage.historical_payload_reused,
        "legacy_batch1_gt_poses_reused": lineage.legacy_batch1_gt_poses_reused,
        "originality_declaration_sha256": lineage.originality_declaration_sha256,
        "packet_sha256": lineage.packet_sha256,
        "pose_candidate_batch16_contract_sha256": lineage.pose_candidate_batch16_contract_sha256,
        "pose_candidate_batch_size": lineage.pose_candidate_batch_size,
        "pose_candidate_contract_id": lineage.pose_candidate_contract_id,
        "pose_target_batch_size": lineage.pose_target_batch_size,
        "pose_target_binding_sha256": lineage.pose_target_binding_sha256,
        "pose_target_contract_id": lineage.pose_target_contract_id,
        "pose_target_mode": lineage.pose_target_mode,
        "source_video_sha256": lineage.source_video_sha256,
        "target_custody_sha256": lineage.target_custody_sha256,
    }
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def bind_g107_source_lineage_to_g17(
    parsed: ParsedG107ConditionalY0V1,
    lineage: G107ConditionalY0SourceLineageV1,
    *,
    lineage_owner: G17LogicalOwnershipV1,
) -> str:
    manifest = encode_g107_source_lineage_manifest(parsed, lineage)
    if (
        type(lineage_owner) is not G17LogicalOwnershipV1
        or lineage_owner.ownership_kind is not G17LogicalOwnershipKindV1.ENCODER_EVIDENCE
        or type(lineage_owner.value) is not G17EncoderOnlyTeacherOracleEvidenceV1
        or lineage_owner.value.exact_bytes != manifest
    ):
        raise G107ConditionalY0Error("fresh lineage is not retained as canonical G17 encoder evidence")
    return _sha256(
        b"G17-G107-FRESH-CONDITIONAL-Y0-LINEAGE-V1\0"
        + bytes.fromhex(parsed.packet_sha256)
        + bytes.fromhex(lineage_owner.identity_sha256)
    )


@dataclass(frozen=True, slots=True)
class G107CandidateByteAccountingV1:
    packet_nbytes: int
    semantic_root_nbytes: int
    final_y1_binding_nbytes: Literal[32]
    conditional_basis_nbytes: int
    conditional_basis_scales_nbytes: int
    conditional_coefficients_nbytes: int
    conditional_coefficient_scales_nbytes: int
    envelope_and_integrity_nbytes: int
    stored_archive_nbytes: int
    deflated_archive_nbytes: int
    selected_archive_nbytes: int
    selected_archive_sha256: str
    selected_member_sha256: str
    selected_encoding: str
    exact_outer_archive: TaskspaceOuterArchiveBuild = field(repr=False)
    research_only: Literal[True] = True
    candidate_claim: Literal[False] = False
    score_claim: Literal[False] = False

    def __post_init__(self) -> None:
        for name in (
            "packet_nbytes",
            "semantic_root_nbytes",
            "conditional_basis_nbytes",
            "conditional_basis_scales_nbytes",
            "conditional_coefficients_nbytes",
            "conditional_coefficient_scales_nbytes",
            "envelope_and_integrity_nbytes",
            "stored_archive_nbytes",
            "deflated_archive_nbytes",
            "selected_archive_nbytes",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise G107ConditionalY0Error(f"{name} must be an exact nonnegative byte count")
        _require_sha256(self.selected_archive_sha256, name="selected archive SHA")
        _require_sha256(self.selected_member_sha256, name="selected member SHA")
        if (
            self.final_y1_binding_nbytes != 32
            or self.selected_archive_nbytes != self.exact_outer_archive.selected.archive_nbytes
            or self.selected_archive_sha256 != self.exact_outer_archive.selected.archive_sha256
            or self.selected_member_sha256 != self.exact_outer_archive.selected.member_sha256
            or self.research_only is not True
            or self.candidate_claim is not False
            or self.score_claim is not False
        ):
            raise G107ConditionalY0Error("candidate-byte accounting is not exact or truth labels drifted")


def account_g107_candidate_bytes(
    parsed: ParsedG107ConditionalY0V1,
) -> G107CandidateByteAccountingV1:
    if type(parsed) is not ParsedG107ConditionalY0V1:
        raise G107ConditionalY0Error("byte accounting requires exact parsed packet")
    outer = build_taskspace_outer_archive(parsed.packet_bytes, max_member_bytes=MAX_PACKET_BYTES)
    section_lengths = tuple(len(section) for section in parsed.section_bytes)
    envelope = len(parsed.packet_bytes) - sum(section_lengths) - 32
    if envelope < 0:
        raise AssertionError("internal packet accounting underflow")
    return G107CandidateByteAccountingV1(
        packet_nbytes=len(parsed.packet_bytes),
        semantic_root_nbytes=section_lengths[0],
        final_y1_binding_nbytes=32,
        conditional_basis_nbytes=section_lengths[1],
        conditional_basis_scales_nbytes=section_lengths[2],
        conditional_coefficients_nbytes=section_lengths[3],
        conditional_coefficient_scales_nbytes=section_lengths[4],
        envelope_and_integrity_nbytes=envelope,
        stored_archive_nbytes=outer.stored_price_bytes,
        deflated_archive_nbytes=outer.deflated_price_bytes,
        selected_archive_nbytes=outer.selected.archive_nbytes,
        selected_archive_sha256=outer.selected.archive_sha256,
        selected_member_sha256=outer.selected.member_sha256,
        selected_encoding=outer.selected.encoding.value,
        exact_outer_archive=outer,
    )


def target_byte_budget_for_score(
    *,
    target_score: float,
    d_seg: float,
    d_pose: float,
    uncompressed_size: int = UNCOMPRESSED_SIZE_BYTES,
) -> int | None:
    """Largest integer archive size with strict score below ``target_score``."""

    if (
        type(target_score) not in (int, float)
        or isinstance(target_score, bool)
        or not math.isfinite(float(target_score))
        or float(target_score) <= 0
    ):
        raise G107ConditionalY0Error("target_score must be positive finite")
    remaining = float(target_score) - seg_term(d_seg) - pose_term(d_pose)
    if remaining <= 0:
        return None
    boundary = remaining * uncompressed_size / 25.0
    candidate = max(0, math.ceil(boundary) - 1)
    while candidate >= 0 and compute_contest_score(
        d_seg,
        d_pose,
        candidate,
        uncompressed_size=uncompressed_size,
    ) >= float(target_score):
        candidate -= 1
    while compute_contest_score(
        d_seg,
        d_pose,
        candidate + 1,
        uncompressed_size=uncompressed_size,
    ) < float(target_score):
        candidate += 1
    return candidate if candidate >= 0 else None


@dataclass(frozen=True, slots=True)
class G107JointFeasibilityV1:
    """Coupled Seg/Pose/rate model only; exact evaluator evidence remains owed."""

    d_seg: float
    d_pose: float
    selected_archive_nbytes: int
    target_score: float
    seg_score_term: float
    pose_score_term: float
    rate_score_term: float
    modeled_score: float
    strict_score_slack: float
    target_byte_budget: int | None
    strict_sublevel_feasible: bool
    pose_measurement_pair_count: Literal[600] = PAIR_COUNT_N600
    pose_measurement_batch_size: Literal[16] = UPSTREAM_POSE_BATCH_SIZE
    pose_measurement_chronological: Literal[True] = True
    batch32_verdict_rows_are_authority: Literal[False] = False
    research_only: Literal[True] = True
    authority: Literal[False] = False
    score_claim: Literal[False] = False

    def __post_init__(self) -> None:
        numeric = (
            self.d_seg,
            self.d_pose,
            self.target_score,
            self.seg_score_term,
            self.pose_score_term,
            self.rate_score_term,
            self.modeled_score,
            self.strict_score_slack,
        )
        if any(
            type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(float(value))
            for value in numeric
        ):
            raise G107ConditionalY0Error("joint feasibility fields must be finite")
        if type(self.selected_archive_nbytes) is not int or self.selected_archive_nbytes < 0:
            raise G107ConditionalY0Error("selected archive bytes must be nonnegative exact int")
        expected = compute_contest_score(self.d_seg, self.d_pose, self.selected_archive_nbytes)
        if (
            self.modeled_score != expected
            or self.strict_score_slack != self.target_score - expected
            or self.strict_sublevel_feasible is not (expected < self.target_score)
            or (
                self.target_byte_budget is not None
                and self.strict_sublevel_feasible is not (self.selected_archive_nbytes <= self.target_byte_budget)
            )
            or self.pose_measurement_pair_count != PAIR_COUNT_N600
            or self.pose_measurement_batch_size != UPSTREAM_POSE_BATCH_SIZE
            or self.pose_measurement_chronological is not True
            or self.batch32_verdict_rows_are_authority is not False
            or self.research_only is not True
            or self.authority is not False
            or self.score_claim is not False
        ):
            raise G107ConditionalY0Error("joint feasibility arithmetic or authority labels drifted")


def model_g107_joint_feasibility(
    accounting: G107CandidateByteAccountingV1,
    *,
    d_seg: float,
    d_pose: float,
    target_score: float,
    pose_measurement_batch_size: int,
) -> G107JointFeasibilityV1:
    if type(accounting) is not G107CandidateByteAccountingV1:
        raise G107ConditionalY0Error("joint feasibility requires exact candidate-byte accounting")
    if pose_measurement_batch_size != UPSTREAM_POSE_BATCH_SIZE:
        raise G107ConditionalY0Error(
            "joint feasibility rejects legacy batch1 and trainer batch32 pose rows; exact upstream batch16 is required"
        )
    score = compute_contest_score(d_seg, d_pose, accounting.selected_archive_nbytes)
    budget = target_byte_budget_for_score(
        target_score=target_score,
        d_seg=d_seg,
        d_pose=d_pose,
    )
    return G107JointFeasibilityV1(
        d_seg=float(d_seg),
        d_pose=float(d_pose),
        selected_archive_nbytes=accounting.selected_archive_nbytes,
        target_score=float(target_score),
        seg_score_term=seg_term(d_seg),
        pose_score_term=pose_term(d_pose),
        rate_score_term=rate_term(accounting.selected_archive_nbytes),
        modeled_score=score,
        strict_score_slack=float(target_score) - score,
        target_byte_budget=budget,
        strict_sublevel_feasible=score < float(target_score),
    )


__all__ = [
    "CONDITIONAL_OWNER_ID",
    "EXACT_AUTHORITY_BLOCKER",
    "FRESH_SOURCE_LINEAGE_BLOCKER",
    "OPEN_BLOCKERS",
    "PUBLIC_RECEIVER_BLOCKER",
    "WIRE_POLICY_ID",
    "G107CandidateByteAccountingV1",
    "G107ConditionalY0Error",
    "G107ConditionalY0ReceiverV1",
    "G107ConditionalY0SourceLineageV1",
    "G107JointFeasibilityV1",
    "G107OperandEffectAuditV1",
    "G107ScorerPairV1",
    "G107V10Factor2PairV1",
    "ParsedG107ConditionalY0V1",
    "account_g107_candidate_bytes",
    "audit_conditional_operand_effects",
    "bind_g107_source_lineage_to_g17",
    "bind_g107_to_g17",
    "compute_final_y1_binding",
    "conditional_y0_g17_logical_value",
    "encode_g107_conditional_y0_v1",
    "encode_g107_source_lineage_manifest",
    "model_g107_joint_feasibility",
    "parse_g107_conditional_y0_v1",
    "target_byte_budget_for_score",
]
