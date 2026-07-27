# SPDX-License-Identifier: MIT
"""Fixed-order counted product for incumbent BOTH -> G89 Y1 -> G88 Y0|Y1.

The wire owns three source-specific byte homes exactly once:

1. one compact PVSA1 member containing exactly one incumbent G74 ``BOTH``;
2. one G89 class-complete Y1 semantic program bound to the same semantic P;
3. one G88 conditional-Y0 operand bound to the same PVSA1 member and P.

Decode is sequential.  Incumbent PVSA Y0 is rendered exactly.  Receiver-side
generic logic merges the incumbent boundary atoms into G89's native Y1 state,
refusing every donor-address collision.  G88 then receives the chronological
pair ``(exact incumbent Y0, combined native Y1)`` and may modify Y0 only; the
combined Y1 is immutable.

This is receiver and archive closure, not public ``inflate.sh`` closure or an
n600 score.  All truth labels remain research-only until the exact selected
archive is recursively public-decoded and scored upstream.
"""

from __future__ import annotations

import hashlib
import struct
import zlib
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import Any, Final, Literal

import numpy as np

from tac.optimization.direct_description_carrier_compose import (
    REALIZATION_PAINT_ORDER,
    BoundaryShearletAtomV1,
    CarrierComposeReceiverV1,
)
from tac.witness_dsl.taskspace_g74_v15_roleaware_overlay_decoder_v1 import (
    RoleAwareBoundaryShearletOperandV1,
)
from tac.witness_dsl.taskspace_g88_population_conditional_y0_pvsa_v1 import (
    CAUSAL_TRANSITION_ID as G88_CAUSAL_TRANSITION_ID,
)
from tac.witness_dsl.taskspace_g88_population_conditional_y0_pvsa_v1 import (
    ConditionalBatchResultV1,
    ConditionalY0ModeV1,
    PopulationConditionalOperandV1,
    PopulationConditionalPVSAError,
    apply_population_conditional_to_decoded_batch,
    parse_population_conditional_operand,
)
from tac.witness_dsl.taskspace_g89_class_complete_semantic_compiler_v1 import (
    G83_DECODER_TRANSITION_ID as G89_DECODER_TRANSITION_ID,
)
from tac.witness_dsl.taskspace_g89_class_complete_semantic_compiler_v1 import (
    ClassCompleteSemanticError,
    ClassCompleteSemanticProgramV1,
    ClassCompleteSemanticReceiverV1,
    parse_class_complete_semantic_program,
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
    CompactActuatorTypeV1,
    CompactPVSAError,
    CompactPVSAReceiverV1,
    ParsedCompactPVSAMemberV1,
    parse_compact_pvsa_member,
)
from tac.witness_dsl.taskspace_selected_preimage_program_v1 import (
    SelectedPreimageFrameSelectorV1,
)

CAMERA_HEIGHT: Final = 874
CAMERA_WIDTH: Final = 1164
CHANNELS: Final = 3
PAIR_COUNT: Final = 600
ROLE_COUNT: Final = len(REALIZATION_PAINT_ORDER)
MAX_PRODUCT_BYTES: Final = 64 << 20

PRODUCT_MAGIC: Final = b"G94SEQ1\x00"
PRODUCT_VERSION: Final = 1
_PRODUCT_HEADER: Final = struct.Struct(">8sBIII32s32s32s")
_CRC32: Final = struct.Struct(">I")

RECEIVER_ID: Final = "tac.g94.sequential_typed_mixed_selector_receiver.v1"
WIRE_POLICY_ID: Final = "PVSA1_G74_BOTH_THEN_G89_Y1_THEN_G88_CONDITIONAL_Y0_V1"
CAUSAL_TRANSITION_ID: Final = "G85_INCUMBENT_BOTH_TO_G89_COMBINED_Y1_TO_G88_CONDITIONAL_Y0_V1"
DYNAMIC_MERGE_POLICY_ID: Final = "RECEIVER_SIDE_INCUMBENT_G74_BOUNDARY_MERGE_INTO_G89_Y1_COLLISION_REFUSE_V1"
PUBLIC_RUNTIME_BLOCKER: Final = "G94_PUBLIC_INFLATE_SH_RECURSIVE_RUNTIME_CLOSURE_OWED"
UPSTREAM_N600_BLOCKER: Final = "G94_SAME_ARCHIVE_UPSTREAM_EVALUATE_PY_N600_AUTHORITY_OWED"
G83_MEASUREMENT_BLOCKER: Final = "G94_COMPLETE_SEG_POSE_RATE_ROW_FOR_EXACT_ARCHIVE_OWED"
G95_FIT_RECEIPT_BLOCKER: Final = "G94_G95_FIT_RECEIPT_BINDING_EXACT_CONDITIONING_STATE_SHA256_OWED"
OPEN_BLOCKERS: Final = (
    PUBLIC_RUNTIME_BLOCKER,
    UPSTREAM_N600_BLOCKER,
    G83_MEASUREMENT_BLOCKER,
    G95_FIT_RECEIPT_BLOCKER,
)
CONDITIONAL_FIT_AUTHORITY: Final = "STRUCTURAL_EXECUTABLE_ONLY_NO_POSE_MARGINAL_TRANSFER_UNTIL_G95_FIT_RECEIPT"
_CONDITIONING_DOMAIN: Final = b"G94_CONDITIONING_STATE_V1\x00"

_ROLE_WIRE: Final = {"UndrivableBoundary": 0, "Road": 1}


class SequentialTypedProductError(ValueError):
    """Wire custody, transition ordering, collision, or execution failed."""


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return _sha256(memoryview(contiguous).cast("B"))


def _require_sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SequentialTypedProductError(f"{label} is not canonical SHA-256")
    return value


def _exact_positive_int(value: object, *, label: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise SequentialTypedProductError(f"{label} must be an exact integer in [1,{maximum}]")
    return value


def _conditioning_state_sha256(
    *,
    base_pvsa_member_sha256: str,
    g89_program_sha256: str,
) -> str:
    """Hash the exact pre-G88 conditioning state and transition semantics."""

    base_sha = _require_sha256(
        base_pvsa_member_sha256,
        label="base_pvsa_member_sha256",
    )
    program_sha = _require_sha256(
        g89_program_sha256,
        label="g89_program_sha256",
    )
    transitions = (
        "PVSA1:G74:BOTH",
        G89_DECODER_TRANSITION_ID,
        DYNAMIC_MERGE_POLICY_ID,
        CAUSAL_TRANSITION_ID,
    )
    encoded_transitions = b"".join(
        struct.pack(">H", len(value.encode("ascii"))) + value.encode("ascii") for value in transitions
    )
    return _sha256(_CONDITIONING_DOMAIN + bytes.fromhex(base_sha) + bytes.fromhex(program_sha) + encoded_transitions)


def _atom_key(atom: BoundaryShearletAtomV1) -> tuple[int, int, int, int]:
    return (
        atom.pair_index,
        _ROLE_WIRE[atom.role],
        atom.center_y,
        atom.center_x,
    )


def _validate_pair_ids(local_pair_ids: tuple[int, ...]) -> None:
    if (
        type(local_pair_ids) is not tuple
        or not 1 <= len(local_pair_ids) <= MAX_STREAM_BATCH_PAIRS
        or any(type(value) is not int or not 0 <= value < PAIR_COUNT for value in local_pair_ids)
        or local_pair_ids != tuple(range(local_pair_ids[0], local_pair_ids[0] + len(local_pair_ids)))
    ):
        raise SequentialTypedProductError("G94 stream batch must be 1..16 contiguous exact n600 pair IDs")


def encode_sequential_typed_product(
    *,
    base_pvsa_member_bytes: bytes,
    g89_program_bytes: bytes,
    g88_conditional_operand_bytes: bytes,
) -> bytes:
    """Encode the three counted byte homes in fixed causal order."""

    sections = (
        (base_pvsa_member_bytes, "base_pvsa_member_bytes"),
        (g89_program_bytes, "g89_program_bytes"),
        (g88_conditional_operand_bytes, "g88_conditional_operand_bytes"),
    )
    for payload, label in sections:
        if type(payload) is not bytes:
            raise SequentialTypedProductError(f"{label} must be exact bytes")
        _exact_positive_int(len(payload), label=label, maximum=MAX_PRODUCT_BYTES)
    prefix = (
        _PRODUCT_HEADER.pack(
            PRODUCT_MAGIC,
            PRODUCT_VERSION,
            len(base_pvsa_member_bytes),
            len(g89_program_bytes),
            len(g88_conditional_operand_bytes),
            bytes.fromhex(_sha256(base_pvsa_member_bytes)),
            bytes.fromhex(_sha256(g89_program_bytes)),
            bytes.fromhex(_sha256(g88_conditional_operand_bytes)),
        )
        + base_pvsa_member_bytes
        + g89_program_bytes
        + g88_conditional_operand_bytes
    )
    if len(prefix) + _CRC32.size > MAX_PRODUCT_BYTES:
        raise SequentialTypedProductError("G94 product exceeds its bounded wire")
    return prefix + _CRC32.pack(zlib.crc32(prefix) & 0xFFFFFFFF)


@dataclass(frozen=True, slots=True)
class ParsedSequentialTypedProductV1:
    """Strict parsed product and all typed foreign-key-bound sections."""

    member_bytes: bytes = field(repr=False)
    base_pvsa_member_bytes: bytes = field(repr=False)
    base_pvsa: ParsedCompactPVSAMemberV1
    g89_program_bytes: bytes = field(repr=False)
    g89_program: ClassCompleteSemanticProgramV1
    g88_conditional_operand_bytes: bytes = field(repr=False)
    g88_conditional_operand: PopulationConditionalOperandV1
    receiver_id: Literal["tac.g94.sequential_typed_mixed_selector_receiver.v1"] = RECEIVER_ID
    wire_policy_id: Literal["PVSA1_G74_BOTH_THEN_G89_Y1_THEN_G88_CONDITIONAL_Y0_V1"] = WIRE_POLICY_ID
    research_only: Literal[True] = True
    candidate_claim: Literal[False] = False
    score_claim: Literal[False] = False
    public_runtime_closed: Literal[False] = False
    conditional_fit_authority: Literal["STRUCTURAL_EXECUTABLE_ONLY_NO_POSE_MARGINAL_TRANSFER_UNTIL_G95_FIT_RECEIPT"] = (
        CONDITIONAL_FIT_AUTHORITY
    )

    def __post_init__(self) -> None:
        if (
            type(self.member_bytes) is not bytes
            or type(self.base_pvsa_member_bytes) is not bytes
            or type(self.base_pvsa) is not ParsedCompactPVSAMemberV1
            or type(self.g89_program_bytes) is not bytes
            or type(self.g89_program) is not ClassCompleteSemanticProgramV1
            or type(self.g88_conditional_operand_bytes) is not bytes
            or type(self.g88_conditional_operand) is not PopulationConditionalOperandV1
        ):
            raise SequentialTypedProductError("G94 parsed product lost exact typed custody")
        if (
            len(self.base_pvsa.actuators) != 1
            or self.base_pvsa.actuators[0].actuator_type is not CompactActuatorTypeV1.G74_ROLE_AWARE_PREPAINT
            or self.base_pvsa.actuators[0].operand.frame_selector is not SelectedPreimageFrameSelectorV1.BOTH
        ):
            raise SequentialTypedProductError("G94 base must contain exactly one incumbent G74 BOTH operand")
        if (
            self.base_pvsa.member_bytes != self.base_pvsa_member_bytes
            or self.g89_program.to_bytes() != self.g89_program_bytes
            or self.g88_conditional_operand.to_bytes() != self.g88_conditional_operand_bytes
            or self.g89_program.semantic_archive_sha256 != self.base_pvsa.semantic_p_sha256
            or self.g88_conditional_operand.base_pvsa_member_sha256 != self.base_pvsa.member_sha256
            or self.g88_conditional_operand.semantic_p_sha256 != self.base_pvsa.semantic_p_sha256
        ):
            raise SequentialTypedProductError("G94 section foreign keys differ from exact base PVSA/P")
        incumbent = self.base_pvsa.actuators[0].operand.atoms
        incumbent_keys = {_atom_key(atom) for atom in incumbent}
        program_keys = {_atom_key(atom) for atom in self.g89_program.boundary_shearlets}
        if incumbent_keys.intersection(program_keys):
            raise SequentialTypedProductError(
                "G94 incumbent/G89 boundary donor collision requires an explicit replacement law"
            )
        if (
            self.receiver_id != RECEIVER_ID
            or self.wire_policy_id != WIRE_POLICY_ID
            or self.research_only is not True
            or self.candidate_claim is not False
            or self.score_claim is not False
            or self.public_runtime_closed is not False
            or self.conditional_fit_authority != CONDITIONAL_FIT_AUTHORITY
        ):
            raise SequentialTypedProductError("G94 truth labels became permissive")

    @property
    def member_sha256(self) -> str:
        return _sha256(self.member_bytes)

    @property
    def conditioning_state_sha256(self) -> str:
        return _conditioning_state_sha256(
            base_pvsa_member_sha256=self.base_pvsa.member_sha256,
            g89_program_sha256=self.g89_program.sha256,
        )

    @property
    def byte_homes(self) -> tuple[tuple[str, int, str], ...]:
        return (
            (
                "base_pvsa1_incumbent_g74_both",
                len(self.base_pvsa_member_bytes),
                _sha256(self.base_pvsa_member_bytes),
            ),
            (
                "g89_class_complete_y1_program",
                len(self.g89_program_bytes),
                _sha256(self.g89_program_bytes),
            ),
            (
                "g88_conditional_y0_given_combined_y1",
                len(self.g88_conditional_operand_bytes),
                _sha256(self.g88_conditional_operand_bytes),
            ),
        )

    @property
    def open_blockers(self) -> tuple[str, ...]:
        return OPEN_BLOCKERS

    def open_receiver(
        self,
        *,
        verify_member_effects: bool = True,
    ) -> SequentialTypedProductReceiverV1:
        return SequentialTypedProductReceiverV1.open(
            self,
            verify_member_effects=verify_member_effects,
        )


def parse_sequential_typed_product(
    member_bytes: bytes,
    *,
    expected_member_sha256: str | None = None,
    maximum_member_bytes: int = MAX_PRODUCT_BYTES,
    maximum_section_bytes: int = MAX_PRODUCT_BYTES,
) -> ParsedSequentialTypedProductV1:
    """CRC/SHA parse every byte home and prove exact re-encoding."""

    if type(member_bytes) is not bytes:
        raise SequentialTypedProductError("G94 product must be exact bytes")
    member_limit = _exact_positive_int(
        maximum_member_bytes,
        label="maximum_member_bytes",
        maximum=(1 << 32) - 1,
    )
    section_limit = _exact_positive_int(
        maximum_section_bytes,
        label="maximum_section_bytes",
        maximum=(1 << 32) - 1,
    )
    minimum = _PRODUCT_HEADER.size + 3 + _CRC32.size
    if not minimum <= len(member_bytes) <= member_limit:
        raise SequentialTypedProductError("G94 product is truncated or exceeds caller ceiling")
    if expected_member_sha256 is not None and _sha256(member_bytes) != _require_sha256(
        expected_member_sha256,
        label="expected_member_sha256",
    ):
        raise SequentialTypedProductError("G94 product exact SHA custody differs")
    (
        magic,
        version,
        base_bytes,
        g89_bytes,
        g88_bytes,
        base_sha,
        g89_sha,
        g88_sha,
    ) = _PRODUCT_HEADER.unpack_from(member_bytes)
    if magic != PRODUCT_MAGIC or version != PRODUCT_VERSION:
        raise SequentialTypedProductError("G94 product magic/version mismatch")
    lengths = (base_bytes, g89_bytes, g88_bytes)
    if (
        any(length < 1 or length > section_limit for length in lengths)
        or len(member_bytes) != _PRODUCT_HEADER.size + sum(lengths) + _CRC32.size
    ):
        raise SequentialTypedProductError("G94 product section length/EOF mismatch")
    prefix = member_bytes[: -_CRC32.size]
    (expected_crc,) = _CRC32.unpack_from(member_bytes, len(prefix))
    if zlib.crc32(prefix) & 0xFFFFFFFF != expected_crc:
        raise SequentialTypedProductError("G94 product CRC32 mismatch")
    base_start = _PRODUCT_HEADER.size
    g89_start = base_start + base_bytes
    g88_start = g89_start + g89_bytes
    base_payload = member_bytes[base_start:g89_start]
    g89_payload = member_bytes[g89_start:g88_start]
    g88_payload = member_bytes[g88_start : -_CRC32.size]
    for payload, expected_sha, label in (
        (base_payload, base_sha.hex(), "base PVSA"),
        (g89_payload, g89_sha.hex(), "G89 program"),
        (g88_payload, g88_sha.hex(), "G88 operand"),
    ):
        if _sha256(payload) != expected_sha:
            raise SequentialTypedProductError(f"G94 {label} SHA mismatch")
    try:
        base = parse_compact_pvsa_member(
            base_payload,
            maximum_member_bytes=section_limit,
            maximum_section_bytes=section_limit,
        )
        g89 = parse_class_complete_semantic_program(
            g89_payload,
            expected_sha256=g89_sha.hex(),
            maximum_operand_bytes=section_limit,
        )
        g88 = parse_population_conditional_operand(
            g88_payload,
            expected_sha256=g88_sha.hex(),
            maximum_operand_bytes=section_limit,
        )
    except (CompactPVSAError, ClassCompleteSemanticError, PopulationConditionalPVSAError) as exc:
        raise SequentialTypedProductError("G94 nested typed section parse failed") from exc
    parsed = ParsedSequentialTypedProductV1(
        member_bytes=member_bytes,
        base_pvsa_member_bytes=base_payload,
        base_pvsa=base,
        g89_program_bytes=g89_payload,
        g89_program=g89,
        g88_conditional_operand_bytes=g88_payload,
        g88_conditional_operand=g88,
    )
    if (
        encode_sequential_typed_product(
            base_pvsa_member_bytes=base_payload,
            g89_program_bytes=g89_payload,
            g88_conditional_operand_bytes=g88_payload,
        )
        != member_bytes
    ):
        raise SequentialTypedProductError("G94 product changed on parse/re-encode")
    return parsed


def _merge_incumbent_into_g89_y1(
    *,
    g89_receiver: ClassCompleteSemanticReceiverV1,
    incumbent_atoms: tuple[BoundaryShearletAtomV1, ...],
) -> CarrierComposeReceiverV1:
    """Merge exact incumbent boundaries into G89's native state, never payload."""

    existing = g89_receiver.mutated.boundary_shearlets
    existing_keys = {_atom_key(atom) for atom in existing}
    incumbent_keys = tuple(_atom_key(atom) for atom in incumbent_atoms)
    if len(incumbent_keys) != len(set(incumbent_keys)) or existing_keys.intersection(incumbent_keys):
        raise SequentialTypedProductError("G94 dynamic incumbent/G89 boundary merge found a donor collision")
    combined = tuple(sorted((*existing, *incumbent_atoms), key=_atom_key))
    try:
        validated = RoleAwareBoundaryShearletOperandV1(
            frame_selector=SelectedPreimageFrameSelectorV1.Y1,
            atoms=combined,
        )
    except (ValueError, TypeError) as exc:
        raise SequentialTypedProductError("G94 combined Y1 boundary state is invalid") from exc
    if validated.atoms != combined:
        raise SequentialTypedProductError("G94 combined Y1 boundary order drifted")
    return replace(g89_receiver.mutated, boundary_shearlets=combined)


def _visible_role_camera_masks(
    receiver: CarrierComposeReceiverV1,
    local_pair_ids: tuple[int, ...],
) -> np.ndarray:
    """Derive disjoint camera role supports for the complete combined Y1."""

    _validate_pair_ids(local_pair_ids)
    layer_by_role = {row.role: row for row in receiver.layers}
    if not set(REALIZATION_PAINT_ORDER).issubset(layer_by_role):
        raise SequentialTypedProductError("G94 combined Y1 lacks all visible semantic roles")
    ys = (np.arange(CAMERA_HEIGHT) * 384 // CAMERA_HEIGHT).clip(0, 383)
    xs = (np.arange(CAMERA_WIDTH) * 512 // CAMERA_WIDTH).clip(0, 511)
    result = np.zeros(
        (
            len(local_pair_ids),
            ROLE_COUNT,
            CAMERA_HEIGHT,
            CAMERA_WIDTH,
        ),
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
        raise SequentialTypedProductError("G94 visible role supports overlap")
    return np.ascontiguousarray(result)


@dataclass(frozen=True, slots=True)
class SequentialTypedBatchResultV1:
    """One exact bounded sequential decode with per-transition custody."""

    product_member_sha256: str
    conditioning_state_sha256: str
    local_pair_ids: tuple[int, ...]
    base_incumbent_camera_pairs: np.ndarray
    preconditional_camera_pairs: np.ndarray
    camera_pairs: np.ndarray
    conditional_result: ConditionalBatchResultV1
    base_incumbent_sha256: str
    preconditional_sha256: str
    camera_sha256: str
    incumbent_y0_sha256: str
    combined_y1_sha256: str
    g89_changed_y1_values: int
    g88_changed_y0_values: int
    incumbent_y0_preserved_before_g88: Literal[True] = True
    combined_y1_preserved_by_g88: Literal[True] = True
    deterministic_double_decode: Literal[True] = True
    research_only: Literal[True] = True
    candidate_claim: Literal[False] = False
    score_claim: Literal[False] = False

    def __post_init__(self) -> None:
        _require_sha256(self.product_member_sha256, label="product_member_sha256")
        _require_sha256(
            self.conditioning_state_sha256,
            label="conditioning_state_sha256",
        )
        _validate_pair_ids(self.local_pair_ids)
        expected = (
            len(self.local_pair_ids),
            2,
            CAMERA_HEIGHT,
            CAMERA_WIDTH,
            CHANNELS,
        )
        arrays = (
            (
                self.base_incumbent_camera_pairs,
                self.base_incumbent_sha256,
                "base incumbent",
            ),
            (
                self.preconditional_camera_pairs,
                self.preconditional_sha256,
                "preconditional",
            ),
            (self.camera_pairs, self.camera_sha256, "final camera"),
        )
        for value, expected_sha, label in arrays:
            raw = np.asarray(value)
            _require_sha256(expected_sha, label=f"{label}_sha256")
            if raw.dtype != np.uint8 or raw.shape != expected:
                raise SequentialTypedProductError(f"G94 {label} array changed exact uint8 camera ABI")
            if _array_sha256(raw) != expected_sha:
                raise SequentialTypedProductError(f"G94 {label} array differs from receipt SHA")
        _require_sha256(self.incumbent_y0_sha256, label="incumbent_y0_sha256")
        _require_sha256(self.combined_y1_sha256, label="combined_y1_sha256")
        if (
            _array_sha256(self.base_incumbent_camera_pairs[:, 0]) != self.incumbent_y0_sha256
            or _array_sha256(self.preconditional_camera_pairs[:, 1]) != self.combined_y1_sha256
            or not np.array_equal(
                self.preconditional_camera_pairs[:, 0],
                self.base_incumbent_camera_pairs[:, 0],
            )
            or not np.array_equal(
                self.camera_pairs[:, 1],
                self.preconditional_camera_pairs[:, 1],
            )
        ):
            raise SequentialTypedProductError("G94 batch transition hashes or frame ownership differ")
        if (
            type(self.conditional_result) is not ConditionalBatchResultV1
            or self.conditional_result.local_pair_ids != self.local_pair_ids
            or not np.array_equal(
                self.conditional_result.base_camera_pairs,
                self.preconditional_camera_pairs,
            )
            or not np.array_equal(
                self.conditional_result.camera_pairs,
                self.camera_pairs,
            )
            or self.conditional_result.exact_y1_sha256 != self.combined_y1_sha256
        ):
            raise SequentialTypedProductError("G94 nested conditional result is not the same exact transition")
        exact_g89_changes = int(
            np.count_nonzero(self.preconditional_camera_pairs[:, 1] != self.base_incumbent_camera_pairs[:, 1])
        )
        exact_g88_changes = int(np.count_nonzero(self.camera_pairs[:, 0] != self.preconditional_camera_pairs[:, 0]))
        if (
            type(self.g89_changed_y1_values) is not int
            or self.g89_changed_y1_values != exact_g89_changes
            or type(self.g88_changed_y0_values) is not int
            or self.g88_changed_y0_values != exact_g88_changes
            or self.g88_changed_y0_values != self.conditional_result.changed_y0_values
            or self.incumbent_y0_preserved_before_g88 is not True
            or self.combined_y1_preserved_by_g88 is not True
            or self.deterministic_double_decode is not True
            or self.conditional_result.deterministic_double_decode is not True
            or self.research_only is not True
            or self.candidate_claim is not False
            or self.score_claim is not False
        ):
            raise SequentialTypedProductError("G94 batch counts, determinism, or truth labels differ")
        for field_name in (
            "base_incumbent_camera_pairs",
            "preconditional_camera_pairs",
            "camera_pairs",
        ):
            immutable = np.array(
                getattr(self, field_name),
                dtype=np.uint8,
                copy=True,
                order="C",
            )
            immutable.setflags(write=False)
            object.__setattr__(self, field_name, immutable)


@dataclass(frozen=True, slots=True, init=False)
class SequentialTypedProductReceiverV1:
    """Cached fixed-order receiver over one immutable parsed G94 product."""

    parsed: ParsedSequentialTypedProductV1
    base_receiver: CompactPVSAReceiverV1
    g89_receiver: ClassCompleteSemanticReceiverV1
    combined_y1_receiver: CarrierComposeReceiverV1
    _parsed_identity: int

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("SequentialTypedProductReceiverV1 must be constructed through .open()")

    @classmethod
    def open(
        cls,
        parsed: ParsedSequentialTypedProductV1,
        *,
        verify_member_effects: bool = True,
    ) -> SequentialTypedProductReceiverV1:
        if type(parsed) is not ParsedSequentialTypedProductV1:
            raise SequentialTypedProductError("G94 receiver requires an exact parsed product")
        try:
            base_receiver = parsed.base_pvsa.open_receiver(
                verify_member_effects=verify_member_effects,
            )
            g89_receiver = ClassCompleteSemanticReceiverV1.open(
                parsed.base_pvsa.semantic_p_archive,
                parsed.g89_program_bytes,
                expected_semantic_archive_sha256=parsed.base_pvsa.semantic_p_sha256,
                expected_program_sha256=parsed.g89_program.sha256,
                verify_member_effects=verify_member_effects,
            )
        except (CompactPVSAError, ClassCompleteSemanticError) as exc:
            raise SequentialTypedProductError("G94 nested receiver open failed") from exc
        combined = _merge_incumbent_into_g89_y1(
            g89_receiver=g89_receiver,
            incumbent_atoms=parsed.base_pvsa.actuators[0].operand.atoms,
        )
        instance = object.__new__(cls)
        object.__setattr__(instance, "parsed", parsed)
        object.__setattr__(instance, "base_receiver", base_receiver)
        object.__setattr__(instance, "g89_receiver", g89_receiver)
        object.__setattr__(instance, "combined_y1_receiver", combined)
        object.__setattr__(instance, "_parsed_identity", id(parsed))
        return instance

    def _preconditional_once(
        self,
        local_pair_ids: tuple[int, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        _validate_pair_ids(local_pair_ids)
        base = self.base_receiver.render_camera_pair_batch(local_pair_ids)
        combined_native = self.combined_y1_receiver.render_camera_pairs(local_pair_ids)
        expected = (
            len(local_pair_ids),
            2,
            CAMERA_HEIGHT,
            CAMERA_WIDTH,
            CHANNELS,
        )
        if (
            base.dtype != np.uint8
            or combined_native.dtype != np.uint8
            or base.shape != expected
            or combined_native.shape != expected
        ):
            raise SequentialTypedProductError("G94 nested receiver changed camera ABI")
        preconditional = np.ascontiguousarray(base).copy()
        preconditional[:, 1] = combined_native[:, 1]
        if not np.array_equal(preconditional[:, 0], base[:, 0]):
            raise SequentialTypedProductError("G89 Y1 execution changed incumbent Y0")
        return np.ascontiguousarray(base), preconditional

    def render_camera_pair_batch(
        self,
        local_pair_ids: tuple[int, ...],
    ) -> SequentialTypedBatchResultV1:
        """Decode the complete sequence twice and admit only byte identity."""

        if id(self.parsed) != self._parsed_identity:
            raise SequentialTypedProductError("G94 parsed receiver custody drifted")
        first_base, first_pre = self._preconditional_once(local_pair_ids)
        second_base, second_pre = self._preconditional_once(local_pair_ids)
        if not np.array_equal(first_base, second_base) or not np.array_equal(
            first_pre,
            second_pre,
        ):
            raise SequentialTypedProductError("G94 preconditional double decode differs")
        needs_masks = any(
            row.source_pair_id in set(local_pair_ids) and row.mode is ConditionalY0ModeV1.ROLE_TRANSLATE_RGB
            for row in self.parsed.g88_conditional_operand.controls
        )
        first_masks = _visible_role_camera_masks(self.combined_y1_receiver, local_pair_ids) if needs_masks else None
        second_masks = _visible_role_camera_masks(self.combined_y1_receiver, local_pair_ids) if needs_masks else None
        try:
            conditional = apply_population_conditional_to_decoded_batch(
                operand=self.parsed.g88_conditional_operand,
                first_base_camera_pairs=first_pre,
                second_base_camera_pairs=second_pre,
                local_pair_ids=local_pair_ids,
                first_visible_role_masks=first_masks,
                second_visible_role_masks=second_masks,
            )
        except PopulationConditionalPVSAError as exc:
            raise SequentialTypedProductError("G94 G88 conditional transition failed") from exc
        if not np.array_equal(conditional.camera_pairs[:, 1], first_pre[:, 1]):
            raise SequentialTypedProductError("G88 changed combined G89 Y1")
        return SequentialTypedBatchResultV1(
            product_member_sha256=self.parsed.member_sha256,
            conditioning_state_sha256=self.parsed.conditioning_state_sha256,
            local_pair_ids=local_pair_ids,
            base_incumbent_camera_pairs=first_base,
            preconditional_camera_pairs=first_pre,
            camera_pairs=conditional.camera_pairs,
            conditional_result=conditional,
            base_incumbent_sha256=_array_sha256(first_base),
            preconditional_sha256=_array_sha256(first_pre),
            camera_sha256=_array_sha256(conditional.camera_pairs),
            incumbent_y0_sha256=_array_sha256(first_base[:, 0]),
            combined_y1_sha256=_array_sha256(first_pre[:, 1]),
            g89_changed_y1_values=int(np.count_nonzero(first_pre[:, 1] != first_base[:, 1])),
            g88_changed_y0_values=conditional.changed_y0_values,
        )

    def decode_pair(self, pair_index: int) -> SequentialTypedBatchResultV1:
        if type(pair_index) is not int:
            raise SequentialTypedProductError("pair_index must be an exact integer")
        return self.render_camera_pair_batch((pair_index,))

    def iter_camera_pair_batches(
        self,
        *,
        batch_pairs: int = MAX_STREAM_BATCH_PAIRS,
    ) -> Iterator[SequentialTypedBatchResultV1]:
        if type(batch_pairs) is not int or not 1 <= batch_pairs <= MAX_STREAM_BATCH_PAIRS:
            raise SequentialTypedProductError("batch_pairs must be an exact integer in [1,16]")
        for start in range(0, PAIR_COUNT, batch_pairs):
            yield self.render_camera_pair_batch(tuple(range(start, min(start + batch_pairs, PAIR_COUNT))))


@dataclass(frozen=True, slots=True)
class SequentialTypedArchiveBuildV1:
    """Exact STORE/DEFLATE race and equal typed parse-back."""

    outer_build: TaskspaceOuterArchiveBuild
    stored: ParsedSequentialTypedProductV1
    deflated: ParsedSequentialTypedProductV1
    selected: ParsedSequentialTypedProductV1
    research_only: Literal[True] = True
    candidate_claim: Literal[False] = False
    score_claim: Literal[False] = False
    public_runtime_closed: Literal[False] = False
    conditional_fit_authority: Literal["STRUCTURAL_EXECUTABLE_ONLY_NO_POSE_MARGINAL_TRANSFER_UNTIL_G95_FIT_RECEIPT"] = (
        CONDITIONAL_FIT_AUTHORITY
    )

    def __post_init__(self) -> None:
        expected = self.stored if self.outer_build.selected.encoding is OuterArchiveEncoding.STORED else self.deflated
        if (
            self.selected != expected
            or self.stored != self.deflated
            or self.research_only is not True
            or self.candidate_claim is not False
            or self.score_claim is not False
            or self.public_runtime_closed is not False
            or self.conditional_fit_authority != CONDITIONAL_FIT_AUTHORITY
        ):
            raise SequentialTypedProductError("G94 outer coding/truth custody differs")

    @property
    def conditioning_state_sha256(self) -> str:
        return self.selected.conditioning_state_sha256


def _parse_outer(
    exact: ParsedTaskspaceOuterArchive,
    *,
    maximum_member_bytes: int,
    maximum_section_bytes: int,
) -> ParsedSequentialTypedProductV1:
    try:
        reopened = parse_taskspace_outer_archive(
            exact.archive_bytes,
            expected_encoding=exact.encoding,
            expected_archive_sha256=exact.archive_sha256,
            expected_member_sha256=exact.member_sha256,
            max_member_bytes=maximum_member_bytes,
        )
    except TaskspaceOuterArchiveError as exc:
        raise SequentialTypedProductError("G94 outer strict reopen failed") from exc
    return parse_sequential_typed_product(
        reopened.member_bytes,
        expected_member_sha256=reopened.member_sha256,
        maximum_member_bytes=maximum_member_bytes,
        maximum_section_bytes=maximum_section_bytes,
    )


def build_sequential_typed_archive(
    *,
    base_pvsa_member_bytes: bytes,
    g89_program_bytes: bytes,
    g88_conditional_operand_bytes: bytes,
    maximum_member_bytes: int = MAX_PRODUCT_BYTES,
    maximum_section_bytes: int = MAX_PRODUCT_BYTES,
) -> SequentialTypedArchiveBuildV1:
    """Build both exact outer codings and strictly parse every section back."""

    member = encode_sequential_typed_product(
        base_pvsa_member_bytes=base_pvsa_member_bytes,
        g89_program_bytes=g89_program_bytes,
        g88_conditional_operand_bytes=g88_conditional_operand_bytes,
    )
    try:
        outer = build_taskspace_outer_archive(
            member,
            max_member_bytes=maximum_member_bytes,
        )
    except TaskspaceOuterArchiveError as exc:
        raise SequentialTypedProductError("G94 outer archive build failed") from exc
    stored = _parse_outer(
        outer.stored,
        maximum_member_bytes=maximum_member_bytes,
        maximum_section_bytes=maximum_section_bytes,
    )
    deflated = _parse_outer(
        outer.deflated,
        maximum_member_bytes=maximum_member_bytes,
        maximum_section_bytes=maximum_section_bytes,
    )
    selected = stored if outer.selected.encoding is OuterArchiveEncoding.STORED else deflated
    return SequentialTypedArchiveBuildV1(
        outer_build=outer,
        stored=stored,
        deflated=deflated,
        selected=selected,
    )


@dataclass(frozen=True, slots=True)
class G94G83StateMetadataV1:
    """Allocator-shaped exact state identity, not an admitted score row."""

    archive_bytes: int
    archive_sha256: str
    member_bytes: int
    member_sha256: str
    byte_homes: tuple[tuple[str, int, str], ...]
    bounded_proof_pair_ids: tuple[int, ...]
    bounded_camera_sha256: str
    conditioning_state_sha256: str
    receiver_transition_chain: tuple[str, ...]
    conditional_fit_authority: str = CONDITIONAL_FIT_AUTHORITY
    g95_fit_receipt_sha256: None = None
    complete_component_row: None = None
    public_inflate_closed: Literal[False] = False
    upstream_n600_closed: Literal[False] = False
    allocator_schema_ready: Literal[True] = True
    g83_admission_ready: Literal[False] = False
    research_only: Literal[True] = True
    candidate_claim: Literal[False] = False
    score_claim: Literal[False] = False
    pointer_moved: Literal[False] = False
    blockers: tuple[str, ...] = OPEN_BLOCKERS

    def __post_init__(self) -> None:
        _exact_positive_int(
            self.archive_bytes,
            label="archive_bytes",
            maximum=(1 << 32) - 1,
        )
        _exact_positive_int(
            self.member_bytes,
            label="member_bytes",
            maximum=(1 << 32) - 1,
        )
        for value, label in (
            (self.archive_sha256, "archive_sha256"),
            (self.member_sha256, "member_sha256"),
            (self.bounded_camera_sha256, "bounded_camera_sha256"),
            (self.conditioning_state_sha256, "conditioning_state_sha256"),
        ):
            _require_sha256(value, label=label)
        if (
            not self.bounded_proof_pair_ids
            or self.receiver_transition_chain
            != (
                "PVSA1:G74:BOTH",
                G89_DECODER_TRANSITION_ID,
                G88_CAUSAL_TRANSITION_ID,
                CAUSAL_TRANSITION_ID,
            )
            or self.complete_component_row is not None
            or self.conditional_fit_authority != CONDITIONAL_FIT_AUTHORITY
            or self.g95_fit_receipt_sha256 is not None
            or self.g83_admission_ready is not False
            or self.blockers != OPEN_BLOCKERS
        ):
            raise SequentialTypedProductError("G94 G83 metadata weakened admission boundary")

    def to_dict(self) -> dict[str, Any]:
        return {
            "archive": {
                "bytes": self.archive_bytes,
                "sha256": self.archive_sha256,
            },
            "member": {
                "bytes": self.member_bytes,
                "sha256": self.member_sha256,
            },
            "byte_homes": [{"name": name, "bytes": size, "sha256": sha} for name, size, sha in self.byte_homes],
            "bounded_proof_pair_ids": list(self.bounded_proof_pair_ids),
            "bounded_camera_sha256": self.bounded_camera_sha256,
            "conditioning_state_sha256": self.conditioning_state_sha256,
            "receiver_transition_chain": list(self.receiver_transition_chain),
            "conditional_fit_authority": self.conditional_fit_authority,
            "g95_fit_receipt_sha256": None,
            "complete_component_row": None,
            "public_inflate_closed": False,
            "upstream_n600_closed": False,
            "allocator_schema_ready": True,
            "g83_admission_ready": False,
            "blockers": list(self.blockers),
            "research_only": True,
            "candidate_claim": False,
            "score_claim": False,
            "pointer_moved": False,
        }


def build_g83_state_metadata(
    *,
    build: SequentialTypedArchiveBuildV1,
    bounded_proof: SequentialTypedBatchResultV1,
) -> G94G83StateMetadataV1:
    """Bind exact selected archive identity to one deterministic bounded proof."""

    if bounded_proof.deterministic_double_decode is not True:
        raise SequentialTypedProductError("G94 metadata requires deterministic double decode")
    if (
        bounded_proof.product_member_sha256 != build.selected.member_sha256
        or bounded_proof.conditioning_state_sha256 != build.conditioning_state_sha256
    ):
        raise SequentialTypedProductError("G94 bounded proof belongs to a different exact product state")
    selected = build.outer_build.selected
    return G94G83StateMetadataV1(
        archive_bytes=selected.archive_nbytes,
        archive_sha256=selected.archive_sha256,
        member_bytes=len(build.selected.member_bytes),
        member_sha256=build.selected.member_sha256,
        byte_homes=build.selected.byte_homes,
        bounded_proof_pair_ids=bounded_proof.local_pair_ids,
        bounded_camera_sha256=bounded_proof.camera_sha256,
        conditioning_state_sha256=build.conditioning_state_sha256,
        receiver_transition_chain=(
            "PVSA1:G74:BOTH",
            G89_DECODER_TRANSITION_ID,
            G88_CAUSAL_TRANSITION_ID,
            CAUSAL_TRANSITION_ID,
        ),
    )


__all__ = [
    "CAUSAL_TRANSITION_ID",
    "CONDITIONAL_FIT_AUTHORITY",
    "DYNAMIC_MERGE_POLICY_ID",
    "G83_MEASUREMENT_BLOCKER",
    "G95_FIT_RECEIPT_BLOCKER",
    "OPEN_BLOCKERS",
    "PRODUCT_MAGIC",
    "PUBLIC_RUNTIME_BLOCKER",
    "RECEIVER_ID",
    "UPSTREAM_N600_BLOCKER",
    "WIRE_POLICY_ID",
    "G94G83StateMetadataV1",
    "ParsedSequentialTypedProductV1",
    "SequentialTypedArchiveBuildV1",
    "SequentialTypedBatchResultV1",
    "SequentialTypedProductError",
    "SequentialTypedProductReceiverV1",
    "build_g83_state_metadata",
    "build_sequential_typed_archive",
    "encode_sequential_typed_product",
    "parse_sequential_typed_product",
]
