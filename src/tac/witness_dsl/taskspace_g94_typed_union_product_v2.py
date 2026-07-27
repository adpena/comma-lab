# SPDX-License-Identifier: MIT
"""G94 V2: G98 cumulative Y1 product plus one exclusive conditional-Y0 owner.

The counted wire is a product whose final factor is a closed sum type::

    (base PVSA1 * G98 sparse cumulative Y1)
        * (G88 conditional Y0 | G95 P-once basis * indexed coefficient chunks)

The inner preconditional member is an actual serialized object.  G95's
legacy-named ``g94_product_member_sha256`` therefore binds the SHA-256 of that
owner-independent member without a circular outer-product hash.

Decode is scorer-free.  G98 constructs final Y1 once; the selected Y0 owner may
change frame 0 only.  Both branches prove final Y1 byte identity.  This module
does not close public ``inflate.sh`` or exact upstream evaluation.
"""

from __future__ import annotations

import hashlib
import struct
import zlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Final, Literal

import numpy as np

from tac.optimization.direct_description_carrier_compose import (
    REALIZATION_PAINT_ORDER,
    CarrierComposeReceiverV1,
)
from tac.witness_dsl.taskspace_g88_population_conditional_y0_pvsa_v1 import (
    ConditionalY0ModeV1,
    PopulationConditionalOperandV1,
    PopulationConditionalPVSAError,
    apply_population_conditional_to_decoded_batch,
    parse_population_conditional_operand,
)
from tac.witness_dsl.taskspace_g95_population_pose_preimage_chart_v1 import (
    ParsedPopulationPosePreimageBasisV1,
    ParsedPopulationPosePreimageCoefficientChunkV1,
    PopulationPosePreimageChartError,
    PopulationPosePreimageChartReceiverV1,
    parse_population_pose_preimage_basis,
    parse_population_pose_preimage_coefficient_chunk,
    validate_population_chunk_coverage,
)
from tac.witness_dsl.taskspace_outer_archive_codec import (
    OuterArchiveEncoding,
    ParsedTaskspaceOuterArchive,
    TaskspaceOuterArchiveBuild,
    TaskspaceOuterArchiveError,
    build_taskspace_outer_archive,
    parse_taskspace_outer_archive,
)
from tac.witness_dsl.taskspace_pvsa_compact_container_v1 import (
    MAX_STREAM_BATCH_PAIRS,
    CompactPVSAError,
    ParsedCompactPVSAMemberV1,
    parse_compact_pvsa_member,
)
from tac.witness_dsl.taskspace_sparse_atlas_cumulative_lowering_v1 import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CHANNELS,
    PAIR_COUNT,
    SparseAtlasCumulativeLoweringError,
    SparseAtlasCumulativeReceiverV1,
    SparseAtlasY1OperandV1,
    parse_sparse_atlas_y1_operand,
)

MAX_PRODUCT_BYTES: Final = 64 << 20
PRECONDITIONAL_MAGIC: Final = b"G94PRE2\x00"
PRODUCT_MAGIC: Final = b"G94UN2\x00\x00"
PRODUCT_VERSION: Final = 2
_PRECONDITIONAL_HEADER: Final = struct.Struct(">8sBII32s32s")
_PRODUCT_HEADER: Final = struct.Struct(">8sBBHI32sH")
_SECTION_HEADER: Final = struct.Struct(">I32s")
_CRC32: Final = struct.Struct(">I")

RECEIVER_ID: Final = "tac.g94.sparse_y1_exclusive_y0_union_receiver.v2"
WIRE_POLICY_ID: Final = "PRODUCT_BASE_G98_TIMES_SUM_G88_OR_G95_PONCE_V2"
PUBLIC_INFLATE_BLOCKER: Final = "G94_V2_PUBLIC_INFLATE_SH_RECURSIVE_RUNTIME_CLOSURE_OWED"
UPSTREAM_N600_BLOCKER: Final = "G94_V2_SAME_ARCHIVE_UPSTREAM_EVALUATE_PY_N600_AUTHORITY_OWED"
FINAL_Y1_G95_REFIT_BLOCKER: Final = "G94_V2_G95_FINAL_Y1_WHOLE_STATE_REFIT_AND_CHUNK_REEMISSION_OWED"
BASE_BLOCKERS: Final = (PUBLIC_INFLATE_BLOCKER, UPSTREAM_N600_BLOCKER)


class G94V2TypedUnionProductError(ValueError):
    """The V2 sum/product wire, custody, coverage, or decode contract failed."""


class ConditionalY0OwnerTagV2(IntEnum):
    G88_POPULATION_CONDITIONAL = 1
    G95_POPULATION_CHART = 2


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
        raise G94V2TypedUnionProductError(f"{label} is not canonical SHA-256")
    return value


def _exact_int(value: object, *, label: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise G94V2TypedUnionProductError(f"{label} must be an exact integer in [{minimum},{maximum}]")
    return value


def _validate_pair_ids(local_pair_ids: tuple[int, ...]) -> None:
    if (
        type(local_pair_ids) is not tuple
        or not 1 <= len(local_pair_ids) <= MAX_STREAM_BATCH_PAIRS
        or any(type(value) is not int or not 0 <= value < PAIR_COUNT for value in local_pair_ids)
        or local_pair_ids != tuple(range(local_pair_ids[0], local_pair_ids[0] + len(local_pair_ids)))
    ):
        raise G94V2TypedUnionProductError("pair IDs must be 1..16 contiguous exact n600 selectors")


def encode_g94_v2_preconditional_product(
    *,
    base_pvsa_member_bytes: bytes,
    sparse_y1_operand_bytes: bytes,
) -> bytes:
    """Encode base and G98 once as the actual owner-independent parent member."""

    for payload, label in (
        (base_pvsa_member_bytes, "base_pvsa_member_bytes"),
        (sparse_y1_operand_bytes, "sparse_y1_operand_bytes"),
    ):
        if type(payload) is not bytes or not payload:
            raise G94V2TypedUnionProductError(f"{label} must be nonempty exact bytes")
        if len(payload) > MAX_PRODUCT_BYTES:
            raise G94V2TypedUnionProductError(f"{label} exceeds V2 ceiling")
    prefix = (
        _PRECONDITIONAL_HEADER.pack(
            PRECONDITIONAL_MAGIC,
            PRODUCT_VERSION,
            len(base_pvsa_member_bytes),
            len(sparse_y1_operand_bytes),
            bytes.fromhex(_sha256(base_pvsa_member_bytes)),
            bytes.fromhex(_sha256(sparse_y1_operand_bytes)),
        )
        + base_pvsa_member_bytes
        + sparse_y1_operand_bytes
    )
    if len(prefix) + _CRC32.size > MAX_PRODUCT_BYTES:
        raise G94V2TypedUnionProductError("preconditional product exceeds V2 ceiling")
    return prefix + _CRC32.pack(zlib.crc32(prefix) & 0xFFFFFFFF)


@dataclass(frozen=True, slots=True)
class ParsedG94V2PreconditionalProduct:
    """Exact counted base*G98 product."""

    member_bytes: bytes = field(repr=False)
    base_pvsa_member_bytes: bytes = field(repr=False)
    base_pvsa: ParsedCompactPVSAMemberV1
    sparse_y1_operand_bytes: bytes = field(repr=False)
    sparse_y1_operand: SparseAtlasY1OperandV1

    def __post_init__(self) -> None:
        if (
            type(self.member_bytes) is not bytes
            or type(self.base_pvsa_member_bytes) is not bytes
            or type(self.base_pvsa) is not ParsedCompactPVSAMemberV1
            or type(self.sparse_y1_operand_bytes) is not bytes
            or type(self.sparse_y1_operand) is not SparseAtlasY1OperandV1
        ):
            raise G94V2TypedUnionProductError("preconditional typed custody changed")
        if (
            self.base_pvsa.member_bytes != self.base_pvsa_member_bytes
            or self.sparse_y1_operand.to_bytes() != self.sparse_y1_operand_bytes
            or self.sparse_y1_operand.base_pvsa_member_sha256 != self.base_pvsa.member_sha256
            or self.sparse_y1_operand.semantic_p_sha256 != self.base_pvsa.semantic_p_sha256
        ):
            raise G94V2TypedUnionProductError("G98 and base P/member foreign keys differ")
        if (
            encode_g94_v2_preconditional_product(
                base_pvsa_member_bytes=self.base_pvsa_member_bytes,
                sparse_y1_operand_bytes=self.sparse_y1_operand_bytes,
            )
            != self.member_bytes
        ):
            raise G94V2TypedUnionProductError("preconditional product changed on re-encode")

    @property
    def member_sha256(self) -> str:
        return _sha256(self.member_bytes)

    @property
    def conditioning_state_sha256(self) -> str:
        return self.sparse_y1_operand.conditioning_state_sha256

    @property
    def byte_homes(self) -> tuple[tuple[str, int, str], ...]:
        return (
            (
                "base_pvsa1_incumbent_g74_both",
                len(self.base_pvsa_member_bytes),
                _sha256(self.base_pvsa_member_bytes),
            ),
            (
                "g98_sparse_cumulative_y1_operand",
                len(self.sparse_y1_operand_bytes),
                _sha256(self.sparse_y1_operand_bytes),
            ),
        )


def parse_g94_v2_preconditional_product(
    member_bytes: bytes,
    *,
    expected_member_sha256: str | None = None,
    maximum_member_bytes: int = MAX_PRODUCT_BYTES,
) -> ParsedG94V2PreconditionalProduct:
    """Strictly parse and re-encode the owner-independent product."""

    if type(member_bytes) is not bytes:
        raise G94V2TypedUnionProductError("preconditional product must be exact bytes")
    limit = _exact_int(
        maximum_member_bytes,
        label="maximum_member_bytes",
        minimum=1,
        maximum=(1 << 32) - 1,
    )
    if not _PRECONDITIONAL_HEADER.size + 2 + _CRC32.size <= len(member_bytes) <= limit:
        raise G94V2TypedUnionProductError("preconditional product is truncated or oversized")
    if expected_member_sha256 is not None and _sha256(member_bytes) != _require_sha256(
        expected_member_sha256,
        label="expected_member_sha256",
    ):
        raise G94V2TypedUnionProductError("preconditional member SHA differs")
    prefix = member_bytes[: -_CRC32.size]
    (crc,) = _CRC32.unpack_from(member_bytes, len(prefix))
    if zlib.crc32(prefix) & 0xFFFFFFFF != crc:
        raise G94V2TypedUnionProductError("preconditional product CRC differs")
    magic, version, base_len, sparse_len, base_sha, sparse_sha = _PRECONDITIONAL_HEADER.unpack_from(member_bytes)
    if magic != PRECONDITIONAL_MAGIC or version != PRODUCT_VERSION:
        raise G94V2TypedUnionProductError("preconditional product magic/version differs")
    if (
        base_len < 1
        or sparse_len < 1
        or len(member_bytes) != _PRECONDITIONAL_HEADER.size + base_len + sparse_len + _CRC32.size
    ):
        raise G94V2TypedUnionProductError("preconditional section length/EOF differs")
    base_start = _PRECONDITIONAL_HEADER.size
    sparse_start = base_start + base_len
    base_bytes = member_bytes[base_start:sparse_start]
    sparse_bytes = member_bytes[sparse_start : -_CRC32.size]
    if _sha256(base_bytes) != base_sha.hex() or _sha256(sparse_bytes) != sparse_sha.hex():
        raise G94V2TypedUnionProductError("preconditional section SHA differs")
    try:
        base = parse_compact_pvsa_member(
            base_bytes,
            maximum_member_bytes=len(base_bytes),
            maximum_section_bytes=MAX_PRODUCT_BYTES,
        )
        sparse = parse_sparse_atlas_y1_operand(
            sparse_bytes,
            expected_sha256=sparse_sha.hex(),
        )
    except (CompactPVSAError, SparseAtlasCumulativeLoweringError) as exc:
        raise G94V2TypedUnionProductError("preconditional nested parse failed") from exc
    return ParsedG94V2PreconditionalProduct(
        member_bytes=member_bytes,
        base_pvsa_member_bytes=base_bytes,
        base_pvsa=base,
        sparse_y1_operand_bytes=sparse_bytes,
        sparse_y1_operand=sparse,
    )


@dataclass(frozen=True, slots=True)
class G88ConditionalY0OwnerV2:
    """The sole G88 branch of the closed Y0-owner sum."""

    operand_bytes: bytes = field(repr=False)
    operand: PopulationConditionalOperandV1
    tag: Literal[ConditionalY0OwnerTagV2.G88_POPULATION_CONDITIONAL] = (
        ConditionalY0OwnerTagV2.G88_POPULATION_CONDITIONAL
    )

    def __post_init__(self) -> None:
        if (
            type(self.operand_bytes) is not bytes
            or type(self.operand) is not PopulationConditionalOperandV1
            or self.operand.to_bytes() != self.operand_bytes
            or self.tag is not ConditionalY0OwnerTagV2.G88_POPULATION_CONDITIONAL
        ):
            raise G94V2TypedUnionProductError("G88 Y0 owner lost exact one-object custody")

    @classmethod
    def parse(cls, operand_bytes: bytes) -> G88ConditionalY0OwnerV2:
        try:
            operand = parse_population_conditional_operand(operand_bytes)
        except PopulationConditionalPVSAError as exc:
            raise G94V2TypedUnionProductError("G88 owner parse failed") from exc
        return cls(operand_bytes=operand_bytes, operand=operand)

    @property
    def section_bytes(self) -> tuple[bytes, ...]:
        return (self.operand_bytes,)

    @property
    def byte_homes(self) -> tuple[tuple[str, int, str], ...]:
        return (
            (
                "g88_exclusive_conditional_y0_operand",
                len(self.operand_bytes),
                _sha256(self.operand_bytes),
            ),
        )


@dataclass(frozen=True, slots=True)
class G95PopulationChartY0OwnerV2:
    """The sole G95 branch: one basis object and a complete indexed chunk stream."""

    basis_bytes: bytes = field(repr=False)
    basis: ParsedPopulationPosePreimageBasisV1
    chunk_bytes: tuple[bytes, ...] = field(repr=False)
    chunks: tuple[ParsedPopulationPosePreimageCoefficientChunkV1, ...]
    tag: Literal[ConditionalY0OwnerTagV2.G95_POPULATION_CHART] = ConditionalY0OwnerTagV2.G95_POPULATION_CHART

    def __post_init__(self) -> None:
        if (
            type(self.basis_bytes) is not bytes
            or type(self.basis) is not ParsedPopulationPosePreimageBasisV1
            or self.basis.object_bytes != self.basis_bytes
            or type(self.chunk_bytes) is not tuple
            or type(self.chunks) is not tuple
            or len(self.chunk_bytes) != len(self.chunks)
            or not self.chunks
            or any(type(value) is not bytes for value in self.chunk_bytes)
            or any(
                type(chunk) is not ParsedPopulationPosePreimageCoefficientChunkV1 or chunk.object_bytes != raw
                for raw, chunk in zip(self.chunk_bytes, self.chunks, strict=True)
            )
            or self.tag is not ConditionalY0OwnerTagV2.G95_POPULATION_CHART
        ):
            raise G94V2TypedUnionProductError("G95 owner lost basis/chunk custody")
        try:
            validate_population_chunk_coverage(self.basis, self.chunks)
        except PopulationPosePreimageChartError as exc:
            raise G94V2TypedUnionProductError("G95 chunk stream is not exact n600 coverage") from exc

    @classmethod
    def parse(
        cls,
        *,
        basis_bytes: bytes,
        chunk_bytes: tuple[bytes, ...],
    ) -> G95PopulationChartY0OwnerV2:
        try:
            basis = parse_population_pose_preimage_basis(basis_bytes)
            chunks = tuple(parse_population_pose_preimage_coefficient_chunk(value) for value in chunk_bytes)
        except PopulationPosePreimageChartError as exc:
            raise G94V2TypedUnionProductError("G95 owner nested parse failed") from exc
        return cls(
            basis_bytes=basis_bytes,
            basis=basis,
            chunk_bytes=chunk_bytes,
            chunks=chunks,
        )

    @property
    def section_bytes(self) -> tuple[bytes, ...]:
        return (self.basis_bytes, *self.chunk_bytes)

    @property
    def byte_homes(self) -> tuple[tuple[str, int, str], ...]:
        return (
            (
                "g95_population_global_basis_once",
                len(self.basis_bytes),
                _sha256(self.basis_bytes),
            ),
            *tuple(
                (
                    f"g95_indexed_coefficient_chunk_{index:03d}",
                    len(payload),
                    _sha256(payload),
                )
                for index, payload in enumerate(self.chunk_bytes)
            ),
        )


ConditionalY0OwnerV2 = G88ConditionalY0OwnerV2 | G95PopulationChartY0OwnerV2


def encode_g94_v2_typed_union_product(
    *,
    preconditional: ParsedG94V2PreconditionalProduct,
    y0_owner: ConditionalY0OwnerV2,
) -> bytes:
    """Encode exactly one branch; no API surface accepts two Y0 owners."""

    if type(preconditional) is not ParsedG94V2PreconditionalProduct:
        raise G94V2TypedUnionProductError("V2 encode requires exact preconditional product")
    if type(y0_owner) not in (G88ConditionalY0OwnerV2, G95PopulationChartY0OwnerV2):
        raise G94V2TypedUnionProductError("V2 encode requires exactly one closed Y0 owner")
    sections = y0_owner.section_bytes
    if (y0_owner.tag is ConditionalY0OwnerTagV2.G88_POPULATION_CONDITIONAL and len(sections) != 1) or (
        y0_owner.tag is ConditionalY0OwnerTagV2.G95_POPULATION_CHART and len(sections) < 2
    ):
        raise G94V2TypedUnionProductError("Y0 owner section cardinality differs from tag")
    owner_wire = b"".join(
        _SECTION_HEADER.pack(len(payload), bytes.fromhex(_sha256(payload))) + payload for payload in sections
    )
    prefix = (
        _PRODUCT_HEADER.pack(
            PRODUCT_MAGIC,
            PRODUCT_VERSION,
            int(y0_owner.tag),
            0,
            len(preconditional.member_bytes),
            bytes.fromhex(preconditional.member_sha256),
            len(sections),
        )
        + preconditional.member_bytes
        + owner_wire
    )
    if len(prefix) + _CRC32.size > MAX_PRODUCT_BYTES:
        raise G94V2TypedUnionProductError("G94 V2 product exceeds byte ceiling")
    return prefix + _CRC32.pack(zlib.crc32(prefix) & 0xFFFFFFFF)


@dataclass(frozen=True, slots=True)
class ParsedG94V2TypedUnionProduct:
    """Strict product with one statically exclusive Y0 owner."""

    member_bytes: bytes = field(repr=False)
    preconditional: ParsedG94V2PreconditionalProduct
    y0_owner: ConditionalY0OwnerV2
    receiver_id: Literal["tac.g94.sparse_y1_exclusive_y0_union_receiver.v2"] = RECEIVER_ID
    wire_policy_id: Literal["PRODUCT_BASE_G98_TIMES_SUM_G88_OR_G95_PONCE_V2"] = WIRE_POLICY_ID
    research_only: Literal[True] = True
    candidate_claim: Literal[False] = False
    score_claim: Literal[False] = False
    public_runtime_closed: Literal[False] = False

    def __post_init__(self) -> None:
        if (
            type(self.member_bytes) is not bytes
            or type(self.preconditional) is not ParsedG94V2PreconditionalProduct
            or type(self.y0_owner) not in (G88ConditionalY0OwnerV2, G95PopulationChartY0OwnerV2)
        ):
            raise G94V2TypedUnionProductError("parsed V2 product lost exact sum/product type")
        if self.y0_owner.tag is ConditionalY0OwnerTagV2.G88_POPULATION_CONDITIONAL:
            owner = self.y0_owner
            if type(owner) is not G88ConditionalY0OwnerV2:
                raise G94V2TypedUnionProductError("G88 tag carries non-G88 owner")
            if (
                owner.operand.base_pvsa_member_sha256 != self.preconditional.base_pvsa.member_sha256
                or owner.operand.semantic_p_sha256 != self.preconditional.base_pvsa.semantic_p_sha256
            ):
                raise G94V2TypedUnionProductError("G88 owner belongs to another base/P")
        else:
            owner = self.y0_owner
            if type(owner) is not G95PopulationChartY0OwnerV2:
                raise G94V2TypedUnionProductError("G95 tag carries non-G95 owner")
            if (
                owner.basis.g94_product_member_sha256 != self.preconditional.member_sha256
                or owner.basis.g94_conditioning_state_sha256 != self.preconditional.conditioning_state_sha256
            ):
                raise G94V2TypedUnionProductError("G95 basis belongs to another preconditional product/state")
        if (
            self.receiver_id != RECEIVER_ID
            or self.wire_policy_id != WIRE_POLICY_ID
            or self.research_only is not True
            or self.candidate_claim is not False
            or self.score_claim is not False
            or self.public_runtime_closed is not False
        ):
            raise G94V2TypedUnionProductError("V2 product truth labels became permissive")
        if (
            encode_g94_v2_typed_union_product(
                preconditional=self.preconditional,
                y0_owner=self.y0_owner,
            )
            != self.member_bytes
        ):
            raise G94V2TypedUnionProductError("V2 product changed on strict re-encode")

    @property
    def member_sha256(self) -> str:
        return _sha256(self.member_bytes)

    @property
    def owner_tag(self) -> ConditionalY0OwnerTagV2:
        return self.y0_owner.tag

    @property
    def byte_homes(self) -> tuple[tuple[str, int, str], ...]:
        return (*self.preconditional.byte_homes, *self.y0_owner.byte_homes)

    @property
    def envelope_bytes(self) -> int:
        return len(self.member_bytes) - sum(size for _name, size, _sha in self.byte_homes)

    @property
    def open_blockers(self) -> tuple[str, ...]:
        if self.owner_tag is ConditionalY0OwnerTagV2.G95_POPULATION_CHART:
            return (*BASE_BLOCKERS, FINAL_Y1_G95_REFIT_BLOCKER)
        return BASE_BLOCKERS

    def open_receiver(
        self,
        *,
        verify_member_effects: bool = True,
    ) -> G94V2TypedUnionReceiver:
        return G94V2TypedUnionReceiver.open(
            self,
            verify_member_effects=verify_member_effects,
        )


def parse_g94_v2_typed_union_product(
    member_bytes: bytes,
    *,
    expected_member_sha256: str | None = None,
    maximum_member_bytes: int = MAX_PRODUCT_BYTES,
) -> ParsedG94V2TypedUnionProduct:
    """Parse one exact branch and reject every dead/trailing owner byte."""

    if type(member_bytes) is not bytes:
        raise G94V2TypedUnionProductError("G94 V2 product must be exact bytes")
    limit = _exact_int(
        maximum_member_bytes,
        label="maximum_member_bytes",
        minimum=1,
        maximum=(1 << 32) - 1,
    )
    if not _PRODUCT_HEADER.size + 2 + _CRC32.size <= len(member_bytes) <= limit:
        raise G94V2TypedUnionProductError("G94 V2 product is truncated or oversized")
    if expected_member_sha256 is not None and _sha256(member_bytes) != _require_sha256(
        expected_member_sha256,
        label="expected_member_sha256",
    ):
        raise G94V2TypedUnionProductError("G94 V2 exact member SHA differs")
    prefix = member_bytes[: -_CRC32.size]
    (crc,) = _CRC32.unpack_from(member_bytes, len(prefix))
    if zlib.crc32(prefix) & 0xFFFFFFFF != crc:
        raise G94V2TypedUnionProductError("G94 V2 product CRC differs")
    magic, version, raw_tag, reserved, pre_len, pre_sha, owner_count = _PRODUCT_HEADER.unpack_from(member_bytes)
    if magic != PRODUCT_MAGIC or version != PRODUCT_VERSION or reserved != 0:
        raise G94V2TypedUnionProductError("G94 V2 magic/version/reserved differs")
    try:
        tag = ConditionalY0OwnerTagV2(raw_tag)
    except ValueError as exc:
        raise G94V2TypedUnionProductError("G94 V2 Y0 owner tag is unknown") from exc
    if (
        pre_len < 1
        or (tag is ConditionalY0OwnerTagV2.G88_POPULATION_CONDITIONAL and owner_count != 1)
        or (tag is ConditionalY0OwnerTagV2.G95_POPULATION_CHART and owner_count < 2)
    ):
        raise G94V2TypedUnionProductError("G94 V2 owner cardinality differs from closed tag")
    pre_start = _PRODUCT_HEADER.size
    owner_start = pre_start + pre_len
    if owner_start > len(prefix):
        raise G94V2TypedUnionProductError("G94 V2 preconditional length escapes EOF")
    pre_bytes = member_bytes[pre_start:owner_start]
    if _sha256(pre_bytes) != pre_sha.hex():
        raise G94V2TypedUnionProductError("G94 V2 preconditional SHA differs")
    preconditional = parse_g94_v2_preconditional_product(
        pre_bytes,
        expected_member_sha256=pre_sha.hex(),
        maximum_member_bytes=pre_len,
    )
    cursor = owner_start
    owner_sections: list[bytes] = []
    for index in range(owner_count):
        if cursor + _SECTION_HEADER.size > len(prefix):
            raise G94V2TypedUnionProductError(f"G94 V2 owner section {index} header truncated")
        section_len, section_sha = _SECTION_HEADER.unpack_from(member_bytes, cursor)
        cursor += _SECTION_HEADER.size
        if section_len < 1 or cursor + section_len > len(prefix):
            raise G94V2TypedUnionProductError(f"G94 V2 owner section {index} escapes EOF")
        section = member_bytes[cursor : cursor + section_len]
        cursor += section_len
        if _sha256(section) != section_sha.hex():
            raise G94V2TypedUnionProductError(f"G94 V2 owner section {index} SHA differs")
        owner_sections.append(section)
    if cursor != len(prefix):
        raise G94V2TypedUnionProductError("G94 V2 has trailing or double-owner bytes")
    if tag is ConditionalY0OwnerTagV2.G88_POPULATION_CONDITIONAL:
        owner: ConditionalY0OwnerV2 = G88ConditionalY0OwnerV2.parse(owner_sections[0])
    else:
        owner = G95PopulationChartY0OwnerV2.parse(
            basis_bytes=owner_sections[0],
            chunk_bytes=tuple(owner_sections[1:]),
        )
    return ParsedG94V2TypedUnionProduct(
        member_bytes=member_bytes,
        preconditional=preconditional,
        y0_owner=owner,
    )


def _visible_role_camera_masks(
    receiver: CarrierComposeReceiverV1,
    local_pair_ids: tuple[int, ...],
) -> np.ndarray:
    """Derive disjoint final-G98 visible supports for G88 role-bounded modes."""

    _validate_pair_ids(local_pair_ids)
    layer_by_role = {row.role: row for row in receiver.layers}
    if not set(REALIZATION_PAINT_ORDER).issubset(layer_by_role):
        raise G94V2TypedUnionProductError("final G98 Y1 lacks all semantic roles")
    ys = (np.arange(CAMERA_HEIGHT) * 384 // CAMERA_HEIGHT).clip(0, 383)
    xs = (np.arange(CAMERA_WIDTH) * 512 // CAMERA_WIDTH).clip(0, 511)
    result = np.zeros(
        (len(local_pair_ids), len(REALIZATION_PAINT_ORDER), CAMERA_HEIGHT, CAMERA_WIDTH),
        dtype=bool,
    )
    for local_index, pair_id in enumerate(local_pair_ids):
        for role_index, role in enumerate(REALIZATION_PAINT_ORDER):
            mask = receiver._mask_for_layer(
                layer_by_role[role],
                pair_id,
                replace_g1_movable=True,
            ).copy()
            for later_role in REALIZATION_PAINT_ORDER[role_index + 1 :]:
                mask &= ~receiver._mask_for_layer(
                    layer_by_role[later_role],
                    pair_id,
                    replace_g1_movable=True,
                )
            result[local_index, role_index] = mask[np.ix_(ys, xs)]
    if np.any(np.sum(result.astype(np.uint8), axis=1) > 1):
        raise G94V2TypedUnionProductError("final G98 visible role supports overlap")
    return np.ascontiguousarray(result)


@dataclass(frozen=True, slots=True)
class G94V2TypedUnionBatchResult:
    product_member_sha256: str
    preconditional_member_sha256: str
    conditioning_state_sha256: str
    owner_tag: ConditionalY0OwnerTagV2
    owner_object_sha256: str
    local_pair_ids: tuple[int, ...]
    preconditional_camera_pairs: np.ndarray
    camera_pairs: np.ndarray
    preconditional_sha256: str
    camera_sha256: str
    final_y1_sha256: str
    changed_y0_values: int
    final_y1_preserved: Literal[True] = True
    deterministic_double_decode: Literal[True] = True
    research_only: Literal[True] = True
    candidate_claim: Literal[False] = False
    score_claim: Literal[False] = False

    def __post_init__(self) -> None:
        for value, label in (
            (self.product_member_sha256, "product_member_sha256"),
            (self.preconditional_member_sha256, "preconditional_member_sha256"),
            (self.conditioning_state_sha256, "conditioning_state_sha256"),
            (self.owner_object_sha256, "owner_object_sha256"),
            (self.preconditional_sha256, "preconditional_sha256"),
            (self.camera_sha256, "camera_sha256"),
            (self.final_y1_sha256, "final_y1_sha256"),
        ):
            _require_sha256(value, label=label)
        _validate_pair_ids(self.local_pair_ids)
        shape = (len(self.local_pair_ids), 2, CAMERA_HEIGHT, CAMERA_WIDTH, CHANNELS)
        before = np.asarray(self.preconditional_camera_pairs)
        after = np.asarray(self.camera_pairs)
        if before.dtype != np.uint8 or before.shape != shape:
            raise G94V2TypedUnionProductError("preconditional camera ABI differs")
        if after.dtype != np.uint8 or after.shape != shape:
            raise G94V2TypedUnionProductError("final camera ABI differs")
        if (
            _array_sha256(before) != self.preconditional_sha256
            or _array_sha256(after) != self.camera_sha256
            or _array_sha256(before[:, 1]) != self.final_y1_sha256
            or not np.array_equal(after[:, 1], before[:, 1])
            or self.changed_y0_values != int(np.count_nonzero(after[:, 0] != before[:, 0]))
            or self.final_y1_preserved is not True
            or self.deterministic_double_decode is not True
            or self.research_only is not True
            or self.candidate_claim is not False
            or self.score_claim is not False
        ):
            raise G94V2TypedUnionProductError("V2 transition ownership/determinism differs")
        immutable_before = np.array(before, dtype=np.uint8, copy=True, order="C")
        immutable_after = np.array(after, dtype=np.uint8, copy=True, order="C")
        immutable_before.setflags(write=False)
        immutable_after.setflags(write=False)
        object.__setattr__(self, "preconditional_camera_pairs", immutable_before)
        object.__setattr__(self, "camera_pairs", immutable_after)


@dataclass(frozen=True, slots=True)
class G95WholeStateCoverageProofV2:
    basis_object_sha256: str
    whole_preconditional_camera_sha256: str
    chunk_count: int
    pair_count: Literal[600] = PAIR_COUNT
    final_y1_preserved: Literal[True] = True
    deterministic_double_decode: Literal[True] = True
    research_only: Literal[True] = True
    score_claim: Literal[False] = False

    def __post_init__(self) -> None:
        _require_sha256(self.basis_object_sha256, label="basis_object_sha256")
        _require_sha256(
            self.whole_preconditional_camera_sha256,
            label="whole_preconditional_camera_sha256",
        )
        if (
            type(self.chunk_count) is not int
            or self.chunk_count < 1
            or self.pair_count != PAIR_COUNT
            or self.final_y1_preserved is not True
            or self.deterministic_double_decode is not True
            or self.research_only is not True
            or self.score_claim is not False
        ):
            raise G94V2TypedUnionProductError("G95 whole-state proof became permissive")


@dataclass(frozen=True, slots=True, init=False)
class G94V2TypedUnionReceiver:
    parsed: ParsedG94V2TypedUnionProduct
    sparse_receiver: SparseAtlasCumulativeReceiverV1
    g95_receiver: PopulationPosePreimageChartReceiverV1 | None
    _parsed_identity: int

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("G94V2TypedUnionReceiver must be constructed through .open()")

    @classmethod
    def open(
        cls,
        parsed: ParsedG94V2TypedUnionProduct,
        *,
        verify_member_effects: bool = True,
    ) -> G94V2TypedUnionReceiver:
        if type(parsed) is not ParsedG94V2TypedUnionProduct:
            raise G94V2TypedUnionProductError("receiver requires exact parsed V2 product")
        try:
            sparse_receiver = SparseAtlasCumulativeReceiverV1.open(
                base_pvsa_member_bytes=parsed.preconditional.base_pvsa_member_bytes,
                sparse_operand_bytes=parsed.preconditional.sparse_y1_operand_bytes,
                expected_sparse_operand_sha256=parsed.preconditional.sparse_y1_operand.sha256,
                verify_member_effects=verify_member_effects,
            )
        except SparseAtlasCumulativeLoweringError as exc:
            raise G94V2TypedUnionProductError("G98 receiver open failed") from exc
        g95_receiver: PopulationPosePreimageChartReceiverV1 | None = None
        if type(parsed.y0_owner) is G95PopulationChartY0OwnerV2:
            basis = parsed.y0_owner.basis
            try:
                g95_receiver = PopulationPosePreimageChartReceiverV1.open(
                    basis,
                    expected_g94_product_member_sha256=parsed.preconditional.member_sha256,
                    expected_g94_conditioning_state_sha256=parsed.preconditional.conditioning_state_sha256,
                    expected_whole_preconditional_camera_sha256=basis.whole_preconditional_camera_sha256,
                    expected_selected_target_table_sha256=basis.selected_target_table_sha256,
                    expected_posenet_weights_sha256=basis.posenet_weights_sha256,
                )
            except PopulationPosePreimageChartError as exc:
                raise G94V2TypedUnionProductError("G95 scorer-free receiver open failed") from exc
        instance = object.__new__(cls)
        object.__setattr__(instance, "parsed", parsed)
        object.__setattr__(instance, "sparse_receiver", sparse_receiver)
        object.__setattr__(instance, "g95_receiver", g95_receiver)
        object.__setattr__(instance, "_parsed_identity", id(parsed))
        return instance

    def _validate_custody(self) -> None:
        if (
            id(self.parsed) != self._parsed_identity
            or self.sparse_receiver.operand.sha256 != self.parsed.preconditional.sparse_y1_operand.sha256
        ):
            raise G94V2TypedUnionProductError("V2 receiver custody drifted")

    def _preconditional(self, local_pair_ids: tuple[int, ...]) -> np.ndarray:
        _validate_pair_ids(local_pair_ids)
        self._validate_custody()
        try:
            result = self.sparse_receiver.render_final_preconditional_batch(local_pair_ids)
        except SparseAtlasCumulativeLoweringError as exc:
            raise G94V2TypedUnionProductError("G98 final preconditional decode failed") from exc
        return np.ascontiguousarray(result.preconditional_camera_pairs)

    def _g95_chunk_for_exact_ids(
        self,
        local_pair_ids: tuple[int, ...],
    ) -> ParsedPopulationPosePreimageCoefficientChunkV1:
        owner = self.parsed.y0_owner
        if type(owner) is not G95PopulationChartY0OwnerV2:
            raise G94V2TypedUnionProductError("G95 chunk requested on G88 branch")
        for chunk in owner.chunks:
            if chunk.source_pair_ids == local_pair_ids:
                return chunk
        raise G94V2TypedUnionProductError("G95 streaming decode must use one exact indexed chunk selector")

    def render_camera_pair_batch(
        self,
        local_pair_ids: tuple[int, ...],
    ) -> G94V2TypedUnionBatchResult:
        """Run selected Y0 owner twice over exact final G98 Y1."""

        preconditional = self._preconditional(local_pair_ids)
        owner = self.parsed.y0_owner
        if type(owner) is G88ConditionalY0OwnerV2:
            needs_masks = any(
                row.source_pair_id in set(local_pair_ids) and row.mode is ConditionalY0ModeV1.ROLE_TRANSLATE_RGB
                for row in owner.operand.controls
            )
            masks = (
                _visible_role_camera_masks(
                    self.sparse_receiver.receiver_for_prefix(len(self.sparse_receiver.operand.steps)),
                    local_pair_ids,
                )
                if needs_masks
                else None
            )
            try:
                result = apply_population_conditional_to_decoded_batch(
                    operand=owner.operand,
                    first_base_camera_pairs=preconditional,
                    second_base_camera_pairs=np.ascontiguousarray(preconditional).copy(),
                    local_pair_ids=local_pair_ids,
                    first_visible_role_masks=masks,
                    second_visible_role_masks=None if masks is None else masks.copy(),
                )
            except PopulationConditionalPVSAError as exc:
                raise G94V2TypedUnionProductError("G88 exclusive Y0 transition failed") from exc
            camera_pairs = result.camera_pairs
            owner_sha = _sha256(owner.operand_bytes)
        else:
            if type(owner) is not G95PopulationChartY0OwnerV2 or self.g95_receiver is None:
                raise G94V2TypedUnionProductError("G95 branch receiver is absent")
            chunk = self._g95_chunk_for_exact_ids(local_pair_ids)
            try:
                result = self.g95_receiver.decode_preconditional_chunk(
                    chunk,
                    preconditional,
                )
            except PopulationPosePreimageChartError as exc:
                raise G94V2TypedUnionProductError(
                    f"G95 exclusive Y0 transition failed at chunk/state custody: {exc}"
                ) from exc
            camera_pairs = result.camera_pairs
            owner_sha = chunk.object_sha256
        if not np.array_equal(camera_pairs[:, 1], preconditional[:, 1]):
            raise G94V2TypedUnionProductError("selected Y0 owner changed final G98 Y1")
        return G94V2TypedUnionBatchResult(
            product_member_sha256=self.parsed.member_sha256,
            preconditional_member_sha256=self.parsed.preconditional.member_sha256,
            conditioning_state_sha256=self.parsed.preconditional.conditioning_state_sha256,
            owner_tag=owner.tag,
            owner_object_sha256=owner_sha,
            local_pair_ids=local_pair_ids,
            preconditional_camera_pairs=preconditional,
            camera_pairs=camera_pairs,
            preconditional_sha256=_array_sha256(preconditional),
            camera_sha256=_array_sha256(camera_pairs),
            final_y1_sha256=_array_sha256(preconditional[:, 1]),
            changed_y0_values=int(np.count_nonzero(camera_pairs[:, 0] != preconditional[:, 0])),
        )

    def verify_g95_whole_state(self) -> G95WholeStateCoverageProofV2:
        """Stream every exact chunk and bind basis whole-state SHA to actual G98 bytes."""

        owner = self.parsed.y0_owner
        if type(owner) is not G95PopulationChartY0OwnerV2:
            raise G94V2TypedUnionProductError("whole G95 proof requested on G88 branch")
        digest = hashlib.sha256()
        covered = 0
        for chunk in owner.chunks:
            preconditional = self._preconditional(chunk.source_pair_ids)
            digest.update(memoryview(preconditional).cast("B"))
            result = self.render_camera_pair_batch(chunk.source_pair_ids)
            if not np.array_equal(
                result.camera_pairs[:, 1],
                result.preconditional_camera_pairs[:, 1],
            ):
                raise G94V2TypedUnionProductError("G95 whole-state replay changed final Y1")
            covered += len(chunk.source_pair_ids)
        whole_sha = digest.hexdigest()
        if covered != PAIR_COUNT or whole_sha != owner.basis.whole_preconditional_camera_sha256:
            raise G94V2TypedUnionProductError("G95 basis whole-state hash differs from exact final G98 population")
        return G95WholeStateCoverageProofV2(
            basis_object_sha256=owner.basis.object_sha256,
            whole_preconditional_camera_sha256=whole_sha,
            chunk_count=len(owner.chunks),
        )

    def decode_pair(self, pair_index: int) -> G94V2TypedUnionBatchResult:
        if type(pair_index) is not int:
            raise G94V2TypedUnionProductError("pair_index must be an exact integer")
        owner = self.parsed.y0_owner
        if type(owner) is G95PopulationChartY0OwnerV2:
            for chunk in owner.chunks:
                if pair_index in chunk.source_pair_ids:
                    batch = self.render_camera_pair_batch(chunk.source_pair_ids)
                    offset = chunk.source_pair_ids.index(pair_index)
                    before = batch.preconditional_camera_pairs[offset : offset + 1]
                    after = batch.camera_pairs[offset : offset + 1]
                    return G94V2TypedUnionBatchResult(
                        product_member_sha256=batch.product_member_sha256,
                        preconditional_member_sha256=batch.preconditional_member_sha256,
                        conditioning_state_sha256=batch.conditioning_state_sha256,
                        owner_tag=batch.owner_tag,
                        owner_object_sha256=batch.owner_object_sha256,
                        local_pair_ids=(pair_index,),
                        preconditional_camera_pairs=before,
                        camera_pairs=after,
                        preconditional_sha256=_array_sha256(before),
                        camera_sha256=_array_sha256(after),
                        final_y1_sha256=_array_sha256(before[:, 1]),
                        changed_y0_values=int(np.count_nonzero(after[:, 0] != before[:, 0])),
                    )
            raise G94V2TypedUnionProductError("pair is absent from G95 chunk coverage")
        return self.render_camera_pair_batch((pair_index,))

    def iter_camera_pair_batches(
        self,
        *,
        batch_pairs: int = MAX_STREAM_BATCH_PAIRS,
        verify_g95_whole_state: bool = True,
    ) -> Iterator[G94V2TypedUnionBatchResult]:
        owner = self.parsed.y0_owner
        if type(owner) is G95PopulationChartY0OwnerV2:
            if verify_g95_whole_state:
                self.verify_g95_whole_state()
            for chunk in owner.chunks:
                yield self.render_camera_pair_batch(chunk.source_pair_ids)
            return
        _exact_int(
            batch_pairs,
            label="batch_pairs",
            minimum=1,
            maximum=MAX_STREAM_BATCH_PAIRS,
        )
        for start in range(0, PAIR_COUNT, batch_pairs):
            yield self.render_camera_pair_batch(tuple(range(start, min(start + batch_pairs, PAIR_COUNT))))


@dataclass(frozen=True, slots=True)
class G94V2OuterArchiveBuild:
    """STORE/DEFLATE race with equal typed parse-back."""

    outer_build: TaskspaceOuterArchiveBuild
    stored: ParsedG94V2TypedUnionProduct
    deflated: ParsedG94V2TypedUnionProduct
    selected: ParsedG94V2TypedUnionProduct
    research_only: Literal[True] = True
    candidate_claim: Literal[False] = False
    score_claim: Literal[False] = False

    def __post_init__(self) -> None:
        expected = self.stored if self.outer_build.selected.encoding is OuterArchiveEncoding.STORED else self.deflated
        if (
            self.selected.member_bytes != expected.member_bytes
            or self.stored.member_bytes != self.deflated.member_bytes
            or self.selected.member_bytes != self.outer_build.selected.member_bytes
            or self.research_only is not True
            or self.candidate_claim is not False
            or self.score_claim is not False
        ):
            raise G94V2TypedUnionProductError("outer coding or truth custody differs")


def _parse_outer_v2(
    exact: ParsedTaskspaceOuterArchive,
) -> ParsedG94V2TypedUnionProduct:
    try:
        reopened = parse_taskspace_outer_archive(
            exact.archive_bytes,
            expected_encoding=exact.encoding,
            expected_archive_sha256=exact.archive_sha256,
            expected_member_sha256=exact.member_sha256,
            max_member_bytes=MAX_PRODUCT_BYTES,
        )
    except TaskspaceOuterArchiveError as exc:
        raise G94V2TypedUnionProductError("outer V2 reopen failed") from exc
    return parse_g94_v2_typed_union_product(
        reopened.member_bytes,
        expected_member_sha256=reopened.member_sha256,
    )


def build_g94_v2_outer_archive(
    *,
    preconditional: ParsedG94V2PreconditionalProduct,
    y0_owner: ConditionalY0OwnerV2,
) -> G94V2OuterArchiveBuild:
    """Build both exact outer encodings and reopen the selected sum/product."""

    member = encode_g94_v2_typed_union_product(
        preconditional=preconditional,
        y0_owner=y0_owner,
    )
    try:
        outer = build_taskspace_outer_archive(
            member,
            max_member_bytes=MAX_PRODUCT_BYTES,
        )
    except TaskspaceOuterArchiveError as exc:
        raise G94V2TypedUnionProductError("outer V2 build failed") from exc
    stored = _parse_outer_v2(outer.stored)
    deflated = _parse_outer_v2(outer.deflated)
    selected = stored if outer.selected.encoding is OuterArchiveEncoding.STORED else deflated
    return G94V2OuterArchiveBuild(
        outer_build=outer,
        stored=stored,
        deflated=deflated,
        selected=selected,
    )


__all__ = [
    "BASE_BLOCKERS",
    "FINAL_Y1_G95_REFIT_BLOCKER",
    "PRECONDITIONAL_MAGIC",
    "PRODUCT_MAGIC",
    "PUBLIC_INFLATE_BLOCKER",
    "RECEIVER_ID",
    "UPSTREAM_N600_BLOCKER",
    "WIRE_POLICY_ID",
    "ConditionalY0OwnerTagV2",
    "G88ConditionalY0OwnerV2",
    "G94V2OuterArchiveBuild",
    "G94V2TypedUnionBatchResult",
    "G94V2TypedUnionProductError",
    "G94V2TypedUnionReceiver",
    "G95PopulationChartY0OwnerV2",
    "G95WholeStateCoverageProofV2",
    "ParsedG94V2PreconditionalProduct",
    "ParsedG94V2TypedUnionProduct",
    "build_g94_v2_outer_archive",
    "encode_g94_v2_preconditional_product",
    "encode_g94_v2_typed_union_product",
    "parse_g94_v2_preconditional_product",
    "parse_g94_v2_typed_union_product",
]
