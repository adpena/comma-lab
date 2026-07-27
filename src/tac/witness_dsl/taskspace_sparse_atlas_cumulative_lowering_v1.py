# SPDX-License-Identifier: MIT
"""Exact cumulative lowering for selected sparse G90/G92 Y1 interventions.

G90 V2 measures each physical BoundaryShearlet group in isolation at one
incumbent state.  G92 retains those exact atoms but deliberately refuses to
pretend that isolated component deltas add.  This module closes the next
generic seam:

* encoder-side lowering verifies one sealed V2 atlas, one G92 plan, the exact
  incumbent outer archive, and the selected donor IDs/atoms;
* the counted sparse operand carries every selected ID and atom, plus exact
  provenance/foreign-key hashes, so decode never depends on research files;
* the standalone receiver opens P/base once and realizes every cumulative
  prefix from the actual current atom state in canonical order; and
* resumable encoder-only checkpoints record each exact prefix/batch replay
  without entering the counted wire or claiming a score/candidate.

The sparse actuator is intentionally not a ``ClassCompleteSemanticProgramV1``.
It may lawfully contain only Road or UndrivableBoundary atoms.  G89's
class-complete topology/Lane/Movable invariants remain unchanged and no dead
filler is synthesized.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import zlib
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np

from tac.optimization.direct_description_carrier_compose import (
    _ROLE_TO_WIRE,
    BoundaryShearletAtomV1,
    CarrierComposeReceiverV1,
    DirectDescriptionError,
    _encode_boundary_shearlet_atoms,
)
from tac.witness_dsl.c0b_semantic_quotient import exact_resize_round_u8
from tac.witness_dsl.taskspace_g74_v15_roleaware_overlay_decoder_v1 import (
    RoleAwareBoundaryShearletOperandV1,
    V15RoleAwareOverlayError,
)
from tac.witness_dsl.taskspace_g92_population_global_program_induction_v1 import (
    G90_V2_AGGREGATE_SCHEMA,
    PopulationProgramPlanV1,
    SealedG90PopulationV1,
    canonical_json_bytes,
    sha256_bytes,
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

PAIR_COUNT: Final = 600
CAMERA_HEIGHT: Final = 874
CAMERA_WIDTH: Final = 1164
SCORER_HEIGHT: Final = 384
SCORER_WIDTH: Final = 512
CHANNELS: Final = 3
MAX_OPERAND_BYTES: Final = 64 << 20
MAX_SELECTED_STEPS: Final = 1 << 16

OPERAND_MAGIC: Final = b"G98SAT1\x00"
OPERAND_VERSION: Final = 1
_OPERAND_HEADER: Final = struct.Struct(">8sB32s32s32s32s32s32sH")
_STEP_HEADER: Final = struct.Struct(">HI32s")
_CRC32: Final = struct.Struct(">I")

RECEIVER_ID: Final = "tac.g98.sparse_atlas_cumulative_y1_receiver.v1"
WIRE_POLICY_ID: Final = "COUNTED_SELECTED_IDS_AND_G74_Y1_ATOMS_CANONICAL_CUMULATIVE_REPLAY_V1"
G94_PRECONDITIONAL_ABI_ID: Final = "G94_PRECONDITIONAL_UINT8_CAMERA_PAIR_PLUS_COMBINED_Y1_SHA256_V1"
CHECKPOINT_SCHEMA: Final = "tac.g98_sparse_atlas_prefix_batch_checkpoint.v1"
PREFIX_SCHEMA: Final = "tac.g98_sparse_atlas_prefix_stage_receipt.v1"
REAL_AGGREGATE_BLOCKER: Final = "G98_REQUIRES_COMPLETE_PERSISTED_G90_V2_AGGREGATE_BEFORE_REAL_N600_MATERIALIZATION"

_ID_RE: Final = re.compile(
    r"g72:[0-9]{4}_[0-9]{4}:(?:Road|UndrivableBoundary):"
    r"d[01]:a(?:0\.5|1):p[0-9]+\Z"
)
_SHA_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_CONDITIONING_DOMAIN: Final = b"G98_SPARSE_ATLAS_CONDITIONING_STATE_V1\x00"
_STATE_DOMAIN: Final = b"G98_SPARSE_ATLAS_REALIZED_PREFIX_STATE_V1\x00"


class SparseAtlasCumulativeLoweringError(ValueError):
    """A sealed-input, sparse wire, state-chain, or checkpoint invariant failed."""


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return _sha256(memoryview(contiguous).cast("B"))


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA_RE.fullmatch(value) is None:
        raise SparseAtlasCumulativeLoweringError(f"{label} is not canonical SHA-256")
    return value


def _require_exact_int(
    value: object,
    *,
    label: str,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise SparseAtlasCumulativeLoweringError(f"{label} must be an exact integer in [{minimum},{maximum}]")
    return value


def _atom_key(atom: BoundaryShearletAtomV1) -> tuple[int, int, int, int]:
    return (
        atom.pair_index,
        _ROLE_TO_WIRE[atom.role],
        atom.center_y,
        atom.center_x,
    )


def _immutable_u8(value: np.ndarray, *, shape: tuple[int, ...], label: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype != np.uint8 or raw.shape != shape:
        raise SparseAtlasCumulativeLoweringError(f"{label} changed exact uint8 ABI")
    copied = np.ascontiguousarray(raw).copy()
    copied.setflags(write=False)
    return copied


def _plan_sha256(plan: PopulationProgramPlanV1) -> str:
    """Hash every plan field without inventing a second persisted plan format."""

    if type(plan) is not PopulationProgramPlanV1:
        raise SparseAtlasCumulativeLoweringError("G98 requires an exact G92 plan type")
    value = {
        "g90_aggregate_sha256": plan.g90_aggregate_sha256,
        "g90_aggregate_self_sha256": plan.g90_aggregate_self_sha256,
        "g90_source_schema": plan.g90_source_schema,
        "exact_replay_atlas_complete": plan.exact_replay_atlas_complete,
        "g51_receipt_sha256": plan.g51_receipt_sha256,
        "current_base_archive_bytes": plan.current_base_archive_bytes,
        "current_base_archive_sha256": plan.current_base_archive_sha256,
        "shared_families": [
            {
                "family_id": row.family_id,
                "role": row.role,
                "direction_rank": row.direction_rank,
                "amplitude_scale": row.amplitude_scale,
                "intervention_ids": list(row.intervention_ids),
            }
            for row in plan.shared_families
        ],
        "branches": [list(branch) for branch in plan.branches],
        "screening_only_projection_ids": list(plan.screening_only_projection_ids),
        "lowering_blocker": plan.lowering_blocker,
    }
    return sha256_bytes(canonical_json_bytes(value))


@dataclass(frozen=True, slots=True)
class SparseAtlasY1StepV1:
    """One donor-exact selected intervention embedded in the counted wire."""

    operand_id: str
    operand: RoleAwareBoundaryShearletOperandV1

    def __post_init__(self) -> None:
        if type(self.operand_id) is not str or _ID_RE.fullmatch(self.operand_id) is None:
            raise SparseAtlasCumulativeLoweringError("sparse step lost canonical G90 operand ID")
        if type(self.operand) is not RoleAwareBoundaryShearletOperandV1:
            raise SparseAtlasCumulativeLoweringError("sparse step lost exact G74 operand type")
        if self.operand.frame_selector is not SelectedPreimageFrameSelectorV1.Y1:
            raise SparseAtlasCumulativeLoweringError("sparse step must own Y1 only")

    @property
    def atoms(self) -> tuple[BoundaryShearletAtomV1, ...]:
        return self.operand.atoms

    @property
    def sha256(self) -> str:
        return self.operand.sha256


@dataclass(frozen=True, slots=True)
class SparseAtlasY1OperandV1:
    """Hermetic counted sparse operand; G90/G92 hashes are provenance only."""

    semantic_p_sha256: str
    base_archive_sha256: str
    base_pvsa_member_sha256: str
    g90_aggregate_sha256: str
    g90_aggregate_self_sha256: str
    g92_plan_sha256: str
    steps: tuple[SparseAtlasY1StepV1, ...]
    receiver_id: Literal["tac.g98.sparse_atlas_cumulative_y1_receiver.v1"] = RECEIVER_ID
    wire_policy_id: Literal["COUNTED_SELECTED_IDS_AND_G74_Y1_ATOMS_CANONICAL_CUMULATIVE_REPLAY_V1"] = WIRE_POLICY_ID
    research_only: Literal[True] = True
    candidate_claim: Literal[False] = False
    score_claim: Literal[False] = False

    def __post_init__(self) -> None:
        for value, label in (
            (self.semantic_p_sha256, "semantic_p_sha256"),
            (self.base_archive_sha256, "base_archive_sha256"),
            (self.base_pvsa_member_sha256, "base_pvsa_member_sha256"),
            (self.g90_aggregate_sha256, "g90_aggregate_sha256"),
            (self.g90_aggregate_self_sha256, "g90_aggregate_self_sha256"),
            (self.g92_plan_sha256, "g92_plan_sha256"),
        ):
            _require_sha256(value, label=label)
        if (
            type(self.steps) is not tuple
            or not self.steps
            or len(self.steps) > MAX_SELECTED_STEPS
            or any(type(row) is not SparseAtlasY1StepV1 for row in self.steps)
        ):
            raise SparseAtlasCumulativeLoweringError("sparse operand requires bounded exact steps")
        ids = tuple(row.operand_id for row in self.steps)
        if len(ids) != len(set(ids)):
            raise SparseAtlasCumulativeLoweringError("selected intervention IDs must be unique in declared order")
        keys = tuple(_atom_key(atom) for row in self.steps for atom in row.atoms)
        if len(keys) != len(set(keys)):
            raise SparseAtlasCumulativeLoweringError(
                "selected sparse interventions collide; no replacement law is encoded"
            )
        if (
            self.receiver_id != RECEIVER_ID
            or self.wire_policy_id != WIRE_POLICY_ID
            or self.research_only is not True
            or self.candidate_claim is not False
            or self.score_claim is not False
        ):
            raise SparseAtlasCumulativeLoweringError("sparse operand truth labels became permissive")
        # The donor encoder is the canonical collision/order authority.
        try:
            _encode_boundary_shearlet_atoms(
                tuple(
                    sorted(
                        (atom for row in self.steps for atom in row.atoms),
                        key=_atom_key,
                    )
                )
            )
        except DirectDescriptionError as exc:
            raise SparseAtlasCumulativeLoweringError("sparse union failed donor atom validation") from exc

    @property
    def selected_operand_ids(self) -> tuple[str, ...]:
        return tuple(row.operand_id for row in self.steps)

    @property
    def atom_count(self) -> int:
        return sum(len(row.atoms) for row in self.steps)

    def to_bytes(self) -> bytes:
        encoded_steps: list[bytes] = []
        for step in self.steps:
            operand_payload = step.operand.to_bytes()
            id_bytes = step.operand_id.encode("ascii")
            encoded_steps.append(
                _STEP_HEADER.pack(
                    len(id_bytes),
                    len(operand_payload),
                    bytes.fromhex(step.sha256),
                )
                + id_bytes
                + operand_payload
            )
        prefix = _OPERAND_HEADER.pack(
            OPERAND_MAGIC,
            OPERAND_VERSION,
            bytes.fromhex(self.semantic_p_sha256),
            bytes.fromhex(self.base_archive_sha256),
            bytes.fromhex(self.base_pvsa_member_sha256),
            bytes.fromhex(self.g90_aggregate_sha256),
            bytes.fromhex(self.g90_aggregate_self_sha256),
            bytes.fromhex(self.g92_plan_sha256),
            len(self.steps),
        ) + b"".join(encoded_steps)
        if len(prefix) + _CRC32.size > MAX_OPERAND_BYTES:
            raise SparseAtlasCumulativeLoweringError("sparse operand exceeds byte ceiling")
        return prefix + _CRC32.pack(zlib.crc32(prefix) & 0xFFFFFFFF)

    @property
    def sha256(self) -> str:
        return _sha256(self.to_bytes())

    @property
    def conditioning_state_sha256(self) -> str:
        transitions = (
            b"PVSA1:G74:BOTH",
            b"G98:SPARSE:Y1:CUMULATIVE",
            G94_PRECONDITIONAL_ABI_ID.encode("ascii"),
        )
        encoded = b"".join(struct.pack(">H", len(value)) + value for value in transitions)
        return _sha256(
            _CONDITIONING_DOMAIN + bytes.fromhex(self.base_pvsa_member_sha256) + bytes.fromhex(self.sha256) + encoded
        )


def parse_sparse_atlas_y1_operand(
    payload: bytes,
    *,
    expected_sha256: str | None = None,
    maximum_operand_bytes: int = MAX_OPERAND_BYTES,
) -> SparseAtlasY1OperandV1:
    """Strict CRC/SHA parser with exact donor parse/re-encode."""

    if type(payload) is not bytes:
        raise SparseAtlasCumulativeLoweringError("sparse operand must be exact bytes")
    limit = _require_exact_int(
        maximum_operand_bytes,
        label="maximum_operand_bytes",
        minimum=_OPERAND_HEADER.size + _STEP_HEADER.size + _CRC32.size + 2,
        maximum=(1 << 32) - 1,
    )
    if not _OPERAND_HEADER.size + _STEP_HEADER.size + _CRC32.size + 2 <= len(payload) <= limit:
        raise SparseAtlasCumulativeLoweringError("sparse operand is truncated or exceeds ceiling")
    if expected_sha256 is not None and _sha256(payload) != _require_sha256(
        expected_sha256,
        label="expected_sha256",
    ):
        raise SparseAtlasCumulativeLoweringError("sparse operand exact SHA differs")
    prefix = payload[: -_CRC32.size]
    (expected_crc,) = _CRC32.unpack_from(payload, len(prefix))
    if zlib.crc32(prefix) & 0xFFFFFFFF != expected_crc:
        raise SparseAtlasCumulativeLoweringError("sparse operand CRC32 differs")
    (
        magic,
        version,
        semantic_sha,
        base_archive_sha,
        base_member_sha,
        aggregate_sha,
        aggregate_self_sha,
        plan_sha,
        step_count,
    ) = _OPERAND_HEADER.unpack_from(payload)
    if magic != OPERAND_MAGIC or version != OPERAND_VERSION:
        raise SparseAtlasCumulativeLoweringError("sparse operand magic/version differs")
    _require_exact_int(
        step_count,
        label="step_count",
        minimum=1,
        maximum=MAX_SELECTED_STEPS,
    )
    cursor = _OPERAND_HEADER.size
    steps: list[SparseAtlasY1StepV1] = []
    for index in range(step_count):
        if cursor + _STEP_HEADER.size > len(prefix):
            raise SparseAtlasCumulativeLoweringError(f"sparse step {index} header is truncated")
        id_bytes, operand_bytes, operand_sha = _STEP_HEADER.unpack_from(payload, cursor)
        cursor += _STEP_HEADER.size
        if id_bytes < 1 or operand_bytes < 1 or cursor + id_bytes + operand_bytes > len(prefix):
            raise SparseAtlasCumulativeLoweringError(f"sparse step {index} length escapes EOF")
        id_payload = payload[cursor : cursor + id_bytes]
        cursor += id_bytes
        operand_payload = payload[cursor : cursor + operand_bytes]
        cursor += operand_bytes
        try:
            operand_id = id_payload.decode("ascii")
            from tac.witness_dsl.taskspace_g74_v15_roleaware_overlay_decoder_v1 import (
                parse_role_aware_boundary_shearlet_operand,
            )

            operand = parse_role_aware_boundary_shearlet_operand(
                operand_payload,
                expected_sha256=operand_sha.hex(),
                maximum_operand_bytes=operand_bytes,
            )
        except (UnicodeDecodeError, V15RoleAwareOverlayError) as exc:
            raise SparseAtlasCumulativeLoweringError(f"sparse step {index} donor parse failed") from exc
        steps.append(SparseAtlasY1StepV1(operand_id=operand_id, operand=operand))
    if cursor != len(prefix):
        raise SparseAtlasCumulativeLoweringError("sparse operand has trailing or unowned bytes")
    result = SparseAtlasY1OperandV1(
        semantic_p_sha256=semantic_sha.hex(),
        base_archive_sha256=base_archive_sha.hex(),
        base_pvsa_member_sha256=base_member_sha.hex(),
        g90_aggregate_sha256=aggregate_sha.hex(),
        g90_aggregate_self_sha256=aggregate_self_sha.hex(),
        g92_plan_sha256=plan_sha.hex(),
        steps=tuple(steps),
    )
    if result.to_bytes() != payload:
        raise SparseAtlasCumulativeLoweringError("sparse operand changed on parse/re-encode")
    return result


@dataclass(frozen=True, slots=True)
class LoweredSparseAtlasY1V1:
    """Encoder-side lowering result; decode needs only these exact bytes."""

    base_pvsa_member_bytes: bytes = field(repr=False)
    operand: SparseAtlasY1OperandV1
    selected_source_operand_bytes: int
    counted_sparse_operand_bytes: int
    research_only: Literal[True] = True
    candidate_claim: Literal[False] = False
    score_claim: Literal[False] = False

    def __post_init__(self) -> None:
        if (
            type(self.base_pvsa_member_bytes) is not bytes
            or _sha256(self.base_pvsa_member_bytes) != self.operand.base_pvsa_member_sha256
            or type(self.selected_source_operand_bytes) is not int
            or self.selected_source_operand_bytes <= 0
            or self.counted_sparse_operand_bytes != len(self.operand.to_bytes())
            or self.research_only is not True
            or self.candidate_claim is not False
            or self.score_claim is not False
        ):
            raise SparseAtlasCumulativeLoweringError("encoder lowering byte/truth custody differs")


def lower_selected_sparse_atlas(
    *,
    g90: SealedG90PopulationV1,
    plan: PopulationProgramPlanV1,
    selected_operand_ids: tuple[str, ...],
    base_outer_archive_bytes: bytes,
    verify_member_effects: bool = True,
) -> LoweredSparseAtlasY1V1:
    """Verify V2 source custody and emit a hermetic sparse counted operand."""

    if type(g90) is not SealedG90PopulationV1 or type(plan) is not PopulationProgramPlanV1:
        raise SparseAtlasCumulativeLoweringError("lowering requires exact G90/G92 typed inputs")
    if (
        g90.source_schema != G90_V2_AGGREGATE_SCHEMA
        or not g90.exact_replay_atlas_complete
        or g90.unresolved_projection_ids
        or plan.g90_source_schema != G90_V2_AGGREGATE_SCHEMA
        or not plan.exact_replay_atlas_complete
        or plan.screening_only_projection_ids
    ):
        raise SparseAtlasCumulativeLoweringError(REAL_AGGREGATE_BLOCKER)
    g90.aggregate.verify(label="G90 V2 persisted exact aggregate")
    if (
        plan.g90_aggregate_sha256 != g90.aggregate.sha256
        or plan.g90_aggregate_self_sha256 != g90.aggregate_self_sha256
        or plan.current_base_archive_bytes != g90.base_archive_bytes
        or plan.current_base_archive_sha256 != g90.base_archive_sha256
    ):
        raise SparseAtlasCumulativeLoweringError("G92 plan belongs to a different G90/base state")
    if (
        type(selected_operand_ids) is not tuple
        or not selected_operand_ids
        or len(selected_operand_ids) != len(set(selected_operand_ids))
    ):
        raise SparseAtlasCumulativeLoweringError("selected intervention IDs must be nonempty and unique")
    planned_ids = {value for branch in plan.branches for value in branch}
    interventions = {row.operand_id: row for row in g90.interventions}
    if any(value not in planned_ids or value not in interventions for value in selected_operand_ids):
        raise SparseAtlasCumulativeLoweringError("selected intervention ID is unknown to exact G90/G92 custody")

    if type(base_outer_archive_bytes) is not bytes:
        raise SparseAtlasCumulativeLoweringError("base outer archive must be exact bytes")
    if (
        len(base_outer_archive_bytes) != g90.base_archive_bytes
        or _sha256(base_outer_archive_bytes) != g90.base_archive_sha256
    ):
        raise SparseAtlasCumulativeLoweringError("base outer archive differs from G90/G92 state")
    try:
        outer = parse_taskspace_outer_archive(
            base_outer_archive_bytes,
            expected_archive_sha256=g90.base_archive_sha256,
        )
        base = parse_compact_pvsa_member(
            outer.member_bytes,
            maximum_member_bytes=len(outer.member_bytes),
            maximum_section_bytes=MAX_OPERAND_BYTES,
        )
    except (TaskspaceOuterArchiveError, CompactPVSAError) as exc:
        raise SparseAtlasCumulativeLoweringError("base G94-compatible PVSA reopen failed") from exc
    if (
        len(base.actuators) != 1
        or base.actuators[0].actuator_type is not CompactActuatorTypeV1.G74_ROLE_AWARE_PREPAINT
        or base.actuators[0].operand.frame_selector is not SelectedPreimageFrameSelectorV1.BOTH
    ):
        raise SparseAtlasCumulativeLoweringError("base must be one exact incumbent G74 BOTH state")

    steps: list[SparseAtlasY1StepV1] = []
    selected_source_bytes = 0
    for operand_id in selected_operand_ids:
        row = interventions[operand_id]
        donor = RoleAwareBoundaryShearletOperandV1(
            frame_selector=SelectedPreimageFrameSelectorV1.Y1,
            atoms=row.atoms,
        )
        if (
            donor.sha256 != row.operand_sha256
            or donor.sha256 != row.proposed_atoms_sha256
            or len(donor.to_bytes()) != row.operand_member_bytes
        ):
            raise SparseAtlasCumulativeLoweringError("persisted G90 exact atom custody differs during sparse lowering")
        steps.append(SparseAtlasY1StepV1(operand_id=operand_id, operand=donor))
        selected_source_bytes += len(donor.to_bytes())

    # Validate against P and the incumbent current state, not merely against
    # the selected atoms in isolation.
    try:
        base_receiver = base.open_receiver(verify_member_effects=verify_member_effects)
    except CompactPVSAError as exc:
        raise SparseAtlasCumulativeLoweringError("base P/incumbent receiver open failed") from exc
    current_atoms = (
        *base_receiver.overlay_decoder.receiver.boundary_shearlets,
        *base.actuators[0].operand.atoms,
    )
    selected_atoms = tuple(atom for step in steps for atom in step.atoms)
    current_keys = {_atom_key(atom) for atom in current_atoms}
    selected_keys = tuple(_atom_key(atom) for atom in selected_atoms)
    if len(selected_keys) != len(set(selected_keys)) or current_keys.intersection(selected_keys):
        raise SparseAtlasCumulativeLoweringError("selected sparse atom collides with actual P/incumbent state")

    operand = SparseAtlasY1OperandV1(
        semantic_p_sha256=base.semantic_p_sha256,
        base_archive_sha256=g90.base_archive_sha256,
        base_pvsa_member_sha256=base.member_sha256,
        g90_aggregate_sha256=g90.aggregate.sha256,
        g90_aggregate_self_sha256=g90.aggregate_self_sha256,
        g92_plan_sha256=_plan_sha256(plan),
        steps=tuple(steps),
    )
    return LoweredSparseAtlasY1V1(
        base_pvsa_member_bytes=base.member_bytes,
        operand=operand,
        selected_source_operand_bytes=selected_source_bytes,
        counted_sparse_operand_bytes=len(operand.to_bytes()),
    )


@dataclass(frozen=True, slots=True)
class SparseAtlasPrefixBatchV1:
    """One exact cumulative prefix at the G94 preconditional camera ABI."""

    sparse_operand_sha256: str
    conditioning_state_sha256: str
    prefix_index: int
    selected_operand_ids: tuple[str, ...]
    local_pair_ids: tuple[int, ...]
    previous_state_sha256: str
    current_state_sha256: str
    previous_combined_y1_sha256: str
    combined_y1_sha256: str
    previous_exact_r_y1_sha256: str
    exact_r_y1_sha256: str
    base_incumbent_camera_pairs: np.ndarray
    preconditional_camera_pairs: np.ndarray
    exact_r_preconditional_pairs: np.ndarray
    changed_y1_values_from_previous_prefix: int
    cumulative_changed_y1_values_from_base: int
    g94_preconditional_abi_id: Literal["G94_PRECONDITIONAL_UINT8_CAMERA_PAIR_PLUS_COMBINED_Y1_SHA256_V1"] = (
        G94_PRECONDITIONAL_ABI_ID
    )
    y0_preserved: Literal[True] = True
    deterministic_double_decode: Literal[True] = True
    research_only: Literal[True] = True
    candidate_claim: Literal[False] = False
    score_claim: Literal[False] = False

    def __post_init__(self) -> None:
        for value, label in (
            (self.sparse_operand_sha256, "sparse_operand_sha256"),
            (self.conditioning_state_sha256, "conditioning_state_sha256"),
            (self.previous_state_sha256, "previous_state_sha256"),
            (self.current_state_sha256, "current_state_sha256"),
            (self.previous_combined_y1_sha256, "previous_combined_y1_sha256"),
            (self.combined_y1_sha256, "combined_y1_sha256"),
            (self.previous_exact_r_y1_sha256, "previous_exact_r_y1_sha256"),
            (self.exact_r_y1_sha256, "exact_r_y1_sha256"),
        ):
            _require_sha256(value, label=label)
        _require_exact_int(
            self.prefix_index,
            label="prefix_index",
            minimum=1,
            maximum=MAX_SELECTED_STEPS,
        )
        if (
            type(self.selected_operand_ids) is not tuple
            or len(self.selected_operand_ids) != self.prefix_index
            or len(self.selected_operand_ids) != len(set(self.selected_operand_ids))
            or type(self.local_pair_ids) is not tuple
            or not self.local_pair_ids
            or self.local_pair_ids
            != tuple(range(self.local_pair_ids[0], self.local_pair_ids[0] + len(self.local_pair_ids)))
            or len(self.local_pair_ids) > MAX_STREAM_BATCH_PAIRS
        ):
            raise SparseAtlasCumulativeLoweringError("prefix IDs/pairs changed declared or bounded order")
        shape = (
            len(self.local_pair_ids),
            2,
            CAMERA_HEIGHT,
            CAMERA_WIDTH,
            CHANNELS,
        )
        r_shape = (
            len(self.local_pair_ids),
            2,
            SCORER_HEIGHT,
            SCORER_WIDTH,
            CHANNELS,
        )
        base = _immutable_u8(
            self.base_incumbent_camera_pairs,
            shape=shape,
            label="base_incumbent_camera_pairs",
        )
        pre = _immutable_u8(
            self.preconditional_camera_pairs,
            shape=shape,
            label="preconditional_camera_pairs",
        )
        exact_r = _immutable_u8(
            self.exact_r_preconditional_pairs,
            shape=r_shape,
            label="exact_r_preconditional_pairs",
        )
        if (
            not np.array_equal(pre[:, 0], base[:, 0])
            or _array_sha256(pre[:, 1]) != self.combined_y1_sha256
            or _array_sha256(exact_r[:, 1]) != self.exact_r_y1_sha256
            or type(self.changed_y1_values_from_previous_prefix) is not int
            or self.changed_y1_values_from_previous_prefix < 0
            or type(self.cumulative_changed_y1_values_from_base) is not int
            or self.cumulative_changed_y1_values_from_base < 0
            or self.g94_preconditional_abi_id != G94_PRECONDITIONAL_ABI_ID
            or self.y0_preserved is not True
            or self.deterministic_double_decode is not True
            or self.research_only is not True
            or self.candidate_claim is not False
            or self.score_claim is not False
        ):
            raise SparseAtlasCumulativeLoweringError("prefix realization/truth custody differs")
        object.__setattr__(self, "base_incumbent_camera_pairs", base)
        object.__setattr__(self, "preconditional_camera_pairs", pre)
        object.__setattr__(self, "exact_r_preconditional_pairs", exact_r)


@dataclass(frozen=True, slots=True, init=False)
class SparseAtlasCumulativeReceiverV1:
    """Standalone P-once/base-once cumulative sparse Y1 receiver."""

    base: ParsedCompactPVSAMemberV1
    operand: SparseAtlasY1OperandV1
    base_receiver: CompactPVSAReceiverV1
    semantic_receiver: CarrierComposeReceiverV1
    incumbent_atoms: tuple[BoundaryShearletAtomV1, ...]
    _base_identity: int
    _operand_identity: int

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("SparseAtlasCumulativeReceiverV1 must be constructed through .open()")

    @classmethod
    def open(
        cls,
        *,
        base_pvsa_member_bytes: bytes,
        sparse_operand_bytes: bytes,
        expected_sparse_operand_sha256: str | None = None,
        verify_member_effects: bool = True,
    ) -> SparseAtlasCumulativeReceiverV1:
        if type(base_pvsa_member_bytes) is not bytes:
            raise SparseAtlasCumulativeLoweringError("base PVSA member must be exact bytes")
        try:
            base = parse_compact_pvsa_member(
                base_pvsa_member_bytes,
                maximum_member_bytes=len(base_pvsa_member_bytes),
                maximum_section_bytes=MAX_OPERAND_BYTES,
            )
            operand = parse_sparse_atlas_y1_operand(
                sparse_operand_bytes,
                expected_sha256=expected_sparse_operand_sha256,
            )
        except CompactPVSAError as exc:
            raise SparseAtlasCumulativeLoweringError("sparse receiver base parse failed") from exc
        if base.member_sha256 != operand.base_pvsa_member_sha256 or base.semantic_p_sha256 != operand.semantic_p_sha256:
            raise SparseAtlasCumulativeLoweringError("sparse operand is bound to a different P/base state")
        if (
            len(base.actuators) != 1
            or base.actuators[0].actuator_type is not CompactActuatorTypeV1.G74_ROLE_AWARE_PREPAINT
            or base.actuators[0].operand.frame_selector is not SelectedPreimageFrameSelectorV1.BOTH
        ):
            raise SparseAtlasCumulativeLoweringError("sparse receiver requires one incumbent G74 BOTH")
        try:
            base_receiver = base.open_receiver(verify_member_effects=verify_member_effects)
        except CompactPVSAError as exc:
            raise SparseAtlasCumulativeLoweringError("sparse receiver could not open P/base once") from exc
        semantic_receiver = base_receiver.overlay_decoder.receiver
        incumbent_atoms = base.actuators[0].operand.atoms
        current_keys = {_atom_key(atom) for atom in (*semantic_receiver.boundary_shearlets, *incumbent_atoms)}
        selected_keys = tuple(_atom_key(atom) for step in operand.steps for atom in step.atoms)
        if len(selected_keys) != len(set(selected_keys)) or current_keys.intersection(selected_keys):
            raise SparseAtlasCumulativeLoweringError("sparse operand collides with reopened P/base state")
        instance = object.__new__(cls)
        object.__setattr__(instance, "base", base)
        object.__setattr__(instance, "operand", operand)
        object.__setattr__(instance, "base_receiver", base_receiver)
        object.__setattr__(instance, "semantic_receiver", semantic_receiver)
        object.__setattr__(instance, "incumbent_atoms", incumbent_atoms)
        object.__setattr__(instance, "_base_identity", id(base))
        object.__setattr__(instance, "_operand_identity", id(operand))
        return instance

    @property
    def conditioning_state_sha256(self) -> str:
        return self.operand.conditioning_state_sha256

    def _validate_custody(self) -> None:
        if (
            id(self.base) != self._base_identity
            or id(self.operand) != self._operand_identity
            or self.base.member_sha256 != self.operand.base_pvsa_member_sha256
            or self.base.semantic_p_sha256 != self.operand.semantic_p_sha256
        ):
            raise SparseAtlasCumulativeLoweringError("sparse receiver custody drifted")

    def receiver_for_prefix(self, prefix_index: int) -> CarrierComposeReceiverV1:
        """Return the actual cumulative semantic state for one selected prefix."""

        self._validate_custody()
        _require_exact_int(
            prefix_index,
            label="prefix_index",
            minimum=1,
            maximum=len(self.operand.steps),
        )
        cumulative = tuple(atom for step in self.operand.steps[:prefix_index] for atom in step.atoms)
        combined = tuple(
            sorted(
                (
                    *self.semantic_receiver.boundary_shearlets,
                    *self.incumbent_atoms,
                    *cumulative,
                ),
                key=_atom_key,
            )
        )
        try:
            _encode_boundary_shearlet_atoms(combined)
        except DirectDescriptionError as exc:
            raise SparseAtlasCumulativeLoweringError("cumulative current-state atom union became invalid") from exc
        return replace(self.semantic_receiver, boundary_shearlets=combined)

    def _exact_r(self, camera_pairs: np.ndarray) -> np.ndarray:
        result = np.empty(
            (
                camera_pairs.shape[0],
                2,
                SCORER_HEIGHT,
                SCORER_WIDTH,
                CHANNELS,
            ),
            dtype=np.uint8,
        )
        operator = self.base_receiver.overlay_decoder.operator
        for pair_offset in range(camera_pairs.shape[0]):
            for frame_index in range(2):
                result[pair_offset, frame_index] = exact_resize_round_u8(
                    operator,
                    camera_pairs[pair_offset, frame_index],
                )
        return np.ascontiguousarray(result)

    def render_cumulative_prefixes(
        self,
        local_pair_ids: tuple[int, ...],
        *,
        stop_after_prefix: int | None = None,
    ) -> tuple[SparseAtlasPrefixBatchV1, ...]:
        """Render every requested prefix from its complete current state."""

        self._validate_custody()
        if (
            type(local_pair_ids) is not tuple
            or not 1 <= len(local_pair_ids) <= MAX_STREAM_BATCH_PAIRS
            or local_pair_ids != tuple(range(local_pair_ids[0], local_pair_ids[0] + len(local_pair_ids)))
            or any(type(value) is not int or not 0 <= value < PAIR_COUNT for value in local_pair_ids)
        ):
            raise SparseAtlasCumulativeLoweringError("sparse batch must be 1..16 contiguous exact n600 pair IDs")
        final_prefix = len(self.operand.steps)
        if stop_after_prefix is not None:
            final_prefix = _require_exact_int(
                stop_after_prefix,
                label="stop_after_prefix",
                minimum=1,
                maximum=len(self.operand.steps),
            )
        try:
            base = self.base_receiver.render_camera_pair_batch(local_pair_ids)
        except CompactPVSAError as exc:
            raise SparseAtlasCumulativeLoweringError("incumbent base replay failed") from exc
        base_r = self._exact_r(base)
        previous_camera = base
        previous_y1_sha = _array_sha256(base[:, 1])
        previous_r_y1_sha = _array_sha256(base_r[:, 1])
        previous_state_sha = _sha256(
            _STATE_DOMAIN
            + bytes.fromhex(self.base.member_sha256)
            + bytes.fromhex(previous_y1_sha)
            + bytes.fromhex(previous_r_y1_sha)
        )
        output: list[SparseAtlasPrefixBatchV1] = []
        for prefix_index in range(1, final_prefix + 1):
            current_receiver = self.receiver_for_prefix(prefix_index)
            try:
                first_native = current_receiver.render_camera_pairs(local_pair_ids)
                second_native = current_receiver.render_camera_pairs(local_pair_ids)
            except (DirectDescriptionError, ValueError) as exc:
                raise SparseAtlasCumulativeLoweringError("cumulative Y1 receiver execution failed") from exc
            if (
                first_native.dtype != np.uint8
                or first_native.shape != base.shape
                or not np.array_equal(first_native, second_native)
            ):
                raise SparseAtlasCumulativeLoweringError(
                    "cumulative Y1 receiver is nondeterministic or changed camera ABI"
                )
            preconditional = np.ascontiguousarray(base).copy()
            preconditional[:, 1] = first_native[:, 1]
            if not np.array_equal(preconditional[:, 0], base[:, 0]):
                raise SparseAtlasCumulativeLoweringError("sparse cumulative replay mutated Y0")
            exact_r = self._exact_r(preconditional)
            combined_y1_sha = _array_sha256(preconditional[:, 1])
            exact_r_y1_sha = _array_sha256(exact_r[:, 1])
            structural = canonical_json_bytes(
                {
                    "base_pvsa_member_sha256": self.base.member_sha256,
                    "sparse_operand_sha256": self.operand.sha256,
                    "conditioning_state_sha256": self.conditioning_state_sha256,
                    "prefix_index": prefix_index,
                    "selected_operand_ids": list(self.operand.selected_operand_ids[:prefix_index]),
                    "local_pair_ids": list(local_pair_ids),
                }
            )
            current_state_sha = _sha256(
                _STATE_DOMAIN
                + structural
                + bytes.fromhex(previous_state_sha)
                + bytes.fromhex(combined_y1_sha)
                + bytes.fromhex(exact_r_y1_sha)
            )
            output.append(
                SparseAtlasPrefixBatchV1(
                    sparse_operand_sha256=self.operand.sha256,
                    conditioning_state_sha256=self.conditioning_state_sha256,
                    prefix_index=prefix_index,
                    selected_operand_ids=self.operand.selected_operand_ids[:prefix_index],
                    local_pair_ids=local_pair_ids,
                    previous_state_sha256=previous_state_sha,
                    current_state_sha256=current_state_sha,
                    previous_combined_y1_sha256=previous_y1_sha,
                    combined_y1_sha256=combined_y1_sha,
                    previous_exact_r_y1_sha256=previous_r_y1_sha,
                    exact_r_y1_sha256=exact_r_y1_sha,
                    base_incumbent_camera_pairs=base,
                    preconditional_camera_pairs=preconditional,
                    exact_r_preconditional_pairs=exact_r,
                    changed_y1_values_from_previous_prefix=int(
                        np.count_nonzero(preconditional[:, 1] != previous_camera[:, 1])
                    ),
                    cumulative_changed_y1_values_from_base=int(np.count_nonzero(preconditional[:, 1] != base[:, 1])),
                )
            )
            previous_camera = preconditional
            previous_y1_sha = combined_y1_sha
            previous_r_y1_sha = exact_r_y1_sha
            previous_state_sha = current_state_sha
        return tuple(output)

    def render_final_preconditional_batch(
        self,
        local_pair_ids: tuple[int, ...],
    ) -> SparseAtlasPrefixBatchV1:
        """Return the final G94-compatible preconditional state."""

        return self.render_cumulative_prefixes(
            local_pair_ids,
            stop_after_prefix=len(self.operand.steps),
        )[-1]

    def iter_final_preconditional_batches(
        self,
        *,
        batch_pairs: int = MAX_STREAM_BATCH_PAIRS,
    ) -> Iterator[SparseAtlasPrefixBatchV1]:
        _require_exact_int(
            batch_pairs,
            label="batch_pairs",
            minimum=1,
            maximum=MAX_STREAM_BATCH_PAIRS,
        )
        for start in range(0, PAIR_COUNT, batch_pairs):
            yield self.render_final_preconditional_batch(tuple(range(start, min(start + batch_pairs, PAIR_COUNT))))


@dataclass(frozen=True, slots=True)
class SparseAtlasOuterArchiveBuildV1:
    """Exact STORE/DEFLATE race for the counted sparse byte home."""

    outer_build: TaskspaceOuterArchiveBuild
    stored: SparseAtlasY1OperandV1
    deflated: SparseAtlasY1OperandV1
    selected: SparseAtlasY1OperandV1
    research_only: Literal[True] = True
    candidate_claim: Literal[False] = False
    score_claim: Literal[False] = False

    def __post_init__(self) -> None:
        expected = self.stored if self.outer_build.selected.encoding is OuterArchiveEncoding.STORED else self.deflated
        if (
            self.selected != expected
            or self.stored != self.deflated
            or self.research_only is not True
            or self.candidate_claim is not False
            or self.score_claim is not False
        ):
            raise SparseAtlasCumulativeLoweringError("sparse outer archive custody differs")


def _parse_sparse_outer(
    exact: ParsedTaskspaceOuterArchive,
    *,
    maximum_operand_bytes: int,
) -> SparseAtlasY1OperandV1:
    try:
        reopened = parse_taskspace_outer_archive(
            exact.archive_bytes,
            expected_encoding=exact.encoding,
            expected_archive_sha256=exact.archive_sha256,
            expected_member_sha256=exact.member_sha256,
            max_member_bytes=maximum_operand_bytes,
        )
    except TaskspaceOuterArchiveError as exc:
        raise SparseAtlasCumulativeLoweringError("sparse outer strict reopen failed") from exc
    return parse_sparse_atlas_y1_operand(
        reopened.member_bytes,
        expected_sha256=reopened.member_sha256,
        maximum_operand_bytes=maximum_operand_bytes,
    )


def build_sparse_atlas_outer_archive(
    operand: SparseAtlasY1OperandV1,
    *,
    maximum_operand_bytes: int = MAX_OPERAND_BYTES,
) -> SparseAtlasOuterArchiveBuildV1:
    """Count every selected ID/atom and prove both outer encodings reopen."""

    if type(operand) is not SparseAtlasY1OperandV1:
        raise SparseAtlasCumulativeLoweringError("outer build requires exact sparse operand")
    payload = operand.to_bytes()
    try:
        outer = build_taskspace_outer_archive(
            payload,
            max_member_bytes=maximum_operand_bytes,
        )
    except TaskspaceOuterArchiveError as exc:
        raise SparseAtlasCumulativeLoweringError("sparse outer archive build failed") from exc
    stored = _parse_sparse_outer(
        outer.stored,
        maximum_operand_bytes=maximum_operand_bytes,
    )
    deflated = _parse_sparse_outer(
        outer.deflated,
        maximum_operand_bytes=maximum_operand_bytes,
    )
    selected = stored if outer.selected.encoding is OuterArchiveEncoding.STORED else deflated
    return SparseAtlasOuterArchiveBuildV1(
        outer_build=outer,
        stored=stored,
        deflated=deflated,
        selected=selected,
    )


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Write one immutable checkpoint atomically; never replace an existing stage."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise SparseAtlasCumulativeLoweringError("existing sparse checkpoint differs; immutable resume refused")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sealed(value: dict[str, Any], *, field_name: str) -> dict[str, Any]:
    if field_name in value:
        raise SparseAtlasCumulativeLoweringError("checkpoint seal field already exists")
    return {
        **value,
        field_name: sha256_bytes(canonical_json_bytes(value)),
    }


def _load_sealed(path: Path, *, field_name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise SparseAtlasCumulativeLoweringError("sparse checkpoint cannot be reopened") from exc
    if type(value) is not dict:
        raise SparseAtlasCumulativeLoweringError("sparse checkpoint is not one object")
    expected = _require_sha256(value.get(field_name), label=field_name)
    body = {key: item for key, item in value.items() if key != field_name}
    if sha256_bytes(canonical_json_bytes(body)) != expected:
        raise SparseAtlasCumulativeLoweringError("sparse checkpoint self seal differs")
    return value


def materialize_next_prefix_checkpoint(
    *,
    receiver: SparseAtlasCumulativeReceiverV1,
    checkpoint_root: Path,
    population_pair_ids: tuple[int, ...] = tuple(range(PAIR_COUNT)),
    batch_pairs: int = MAX_STREAM_BATCH_PAIRS,
) -> dict[str, object]:
    """Resume exactly one prefix-batch or finalize one prefix stage.

    Checkpoints are encoder-only hashes/receipts.  They are not part of the
    sparse operand, G94 product, archive price, or decoder dependency.
    """

    if type(receiver) is not SparseAtlasCumulativeReceiverV1:
        raise SparseAtlasCumulativeLoweringError("materializer requires exact sparse receiver")
    if (
        type(population_pair_ids) is not tuple
        or not population_pair_ids
        or population_pair_ids
        != tuple(range(population_pair_ids[0], population_pair_ids[0] + len(population_pair_ids)))
        or any(type(value) is not int or not 0 <= value < PAIR_COUNT for value in population_pair_ids)
    ):
        raise SparseAtlasCumulativeLoweringError("materializer pair population is not contiguous/canonical")
    _require_exact_int(
        batch_pairs,
        label="batch_pairs",
        minimum=1,
        maximum=MAX_STREAM_BATCH_PAIRS,
    )
    root = Path(checkpoint_root)
    ranges = tuple(
        (
            population_pair_ids[offset],
            population_pair_ids[min(offset + batch_pairs, len(population_pair_ids)) - 1] + 1,
        )
        for offset in range(0, len(population_pair_ids), batch_pairs)
    )
    for prefix_index in range(1, len(receiver.operand.steps) + 1):
        prefix_dir = root / f"prefix_{prefix_index:04d}"
        final_path = prefix_dir / "prefix_receipt.json"
        if final_path.is_file():
            final = _load_sealed(final_path, field_name="prefix_receipt_sha256")
            if (
                final.get("schema") != PREFIX_SCHEMA
                or final.get("prefix_index") != prefix_index
                or final.get("sparse_operand_sha256") != receiver.operand.sha256
                or final.get("base_pvsa_member_sha256") != receiver.base.member_sha256
                or final.get("selected_operand_ids") != list(receiver.operand.selected_operand_ids[:prefix_index])
                or final.get("population_pair_ids") != list(population_pair_ids)
                or final.get("encoder_only") is not True
                or final.get("candidate_claim") is not False
                or final.get("score_claim") is not False
            ):
                raise SparseAtlasCumulativeLoweringError("completed prefix resume state differs")
            continue

        bindings: list[dict[str, object]] = []
        for start, stop in ranges:
            batch_path = prefix_dir / "batches" / f"batch_{start:04d}_{stop:04d}.json"
            pair_ids = tuple(range(start, stop))
            if batch_path.is_file():
                checkpoint = _load_sealed(
                    batch_path,
                    field_name="batch_checkpoint_sha256",
                )
                if (
                    checkpoint.get("schema") != CHECKPOINT_SCHEMA
                    or checkpoint.get("prefix_index") != prefix_index
                    or checkpoint.get("pair_ids") != list(pair_ids)
                    or checkpoint.get("sparse_operand_sha256") != receiver.operand.sha256
                    or checkpoint.get("base_pvsa_member_sha256") != receiver.base.member_sha256
                    or checkpoint.get("selected_operand_ids")
                    != list(receiver.operand.selected_operand_ids[:prefix_index])
                    or checkpoint.get("encoder_only") is not True
                    or checkpoint.get("candidate_claim") is not False
                    or checkpoint.get("score_claim") is not False
                ):
                    raise SparseAtlasCumulativeLoweringError("resumed prefix batch differs from source-derived state")
                bindings.append(
                    {
                        "pair_range": [start, stop],
                        "path": str(batch_path),
                        "bytes": batch_path.stat().st_size,
                        "sha256": _sha256(batch_path.read_bytes()),
                        "batch_checkpoint_sha256": checkpoint["batch_checkpoint_sha256"],
                    }
                )
                continue
            result = receiver.render_cumulative_prefixes(
                pair_ids,
                stop_after_prefix=prefix_index,
            )[-1]
            body = {
                "schema": CHECKPOINT_SCHEMA,
                "prefix_index": prefix_index,
                "pair_ids": list(pair_ids),
                "selected_operand_ids": list(result.selected_operand_ids),
                "sparse_operand_sha256": receiver.operand.sha256,
                "conditioning_state_sha256": receiver.conditioning_state_sha256,
                "base_pvsa_member_sha256": receiver.base.member_sha256,
                "g90_aggregate_sha256": receiver.operand.g90_aggregate_sha256,
                "g90_aggregate_self_sha256": receiver.operand.g90_aggregate_self_sha256,
                "g92_plan_sha256": receiver.operand.g92_plan_sha256,
                "previous_state_sha256": result.previous_state_sha256,
                "current_state_sha256": result.current_state_sha256,
                "previous_combined_y1_sha256": result.previous_combined_y1_sha256,
                "combined_y1_sha256": result.combined_y1_sha256,
                "previous_exact_r_y1_sha256": result.previous_exact_r_y1_sha256,
                "exact_r_y1_sha256": result.exact_r_y1_sha256,
                "changed_y1_values_from_previous_prefix": (result.changed_y1_values_from_previous_prefix),
                "cumulative_changed_y1_values_from_base": (result.cumulative_changed_y1_values_from_base),
                "deterministic_double_decode": True,
                "y0_preserved": True,
                "checkpoint_policy": "atomic_immutable_every_prefix_batch_then_every_prefix_stage",
                "population_complete_n600": population_pair_ids == tuple(range(PAIR_COUNT)),
                "encoder_only": True,
                "research_only": True,
                "candidate_claim": False,
                "score_claim": False,
                "pointer_moved": False,
            }
            checkpoint = _sealed(body, field_name="batch_checkpoint_sha256")
            _atomic_write_json(batch_path, checkpoint)
            return {
                "status": "batch_complete",
                "prefix_index": prefix_index,
                "pair_range": [start, stop],
                "checkpoint_path": str(batch_path),
                "encoder_only": True,
            }

        # All batches for this prefix have been strictly reopened.  Finalize
        # the immutable prefix stage without recomputing or inferring scores.
        prefix_body = {
            "schema": PREFIX_SCHEMA,
            "prefix_index": prefix_index,
            "selected_operand_ids": list(receiver.operand.selected_operand_ids[:prefix_index]),
            "population_pair_ids": list(population_pair_ids),
            "batch_count": len(bindings),
            "batches": bindings,
            "sparse_operand_sha256": receiver.operand.sha256,
            "conditioning_state_sha256": receiver.conditioning_state_sha256,
            "base_pvsa_member_sha256": receiver.base.member_sha256,
            "g90_aggregate_sha256": receiver.operand.g90_aggregate_sha256,
            "g90_aggregate_self_sha256": receiver.operand.g90_aggregate_self_sha256,
            "g92_plan_sha256": receiver.operand.g92_plan_sha256,
            "population_complete_n600": population_pair_ids == tuple(range(PAIR_COUNT)),
            "checkpoint_policy": "atomic_immutable_every_prefix_batch_then_every_prefix_stage",
            "encoder_only": True,
            "research_only": True,
            "candidate_claim": False,
            "score_claim": False,
            "pointer_moved": False,
        }
        prefix_receipt = _sealed(prefix_body, field_name="prefix_receipt_sha256")
        _atomic_write_json(final_path, prefix_receipt)
        return {
            "status": "prefix_complete",
            "prefix_index": prefix_index,
            "checkpoint_path": str(final_path),
            "encoder_only": True,
        }
    return {
        "status": "complete",
        "prefix_count": len(receiver.operand.steps),
        "population_complete_n600": population_pair_ids == tuple(range(PAIR_COUNT)),
        "encoder_only": True,
    }


__all__ = [
    "CHECKPOINT_SCHEMA",
    "G94_PRECONDITIONAL_ABI_ID",
    "OPERAND_MAGIC",
    "PREFIX_SCHEMA",
    "REAL_AGGREGATE_BLOCKER",
    "RECEIVER_ID",
    "WIRE_POLICY_ID",
    "LoweredSparseAtlasY1V1",
    "SparseAtlasCumulativeLoweringError",
    "SparseAtlasCumulativeReceiverV1",
    "SparseAtlasOuterArchiveBuildV1",
    "SparseAtlasPrefixBatchV1",
    "SparseAtlasY1OperandV1",
    "SparseAtlasY1StepV1",
    "build_sparse_atlas_outer_archive",
    "lower_selected_sparse_atlas",
    "materialize_next_prefix_checkpoint",
    "parse_sparse_atlas_y1_operand",
]
