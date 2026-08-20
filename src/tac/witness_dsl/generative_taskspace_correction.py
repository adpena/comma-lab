# SPDX-License-Identifier: MIT
"""Finite candidate-payload grammar for a generative V9 task-space correction.

PBR1/PBR2, target labels, obligation rows, oracle observations, and dense RGB
preimages are encoder-only teachers.  They are intentionally absent from this
wire format.  The counted packet contains only finite parameters of receiver-
closed V9 primitives: boundary gauge coefficients, topology events, compact
boundary shearlets, island shapes, movable worldsheets, and one shared palette.

This is an L0 research compiler.  Structural candidate-payload eligibility is
not a score or promotion claim.  The packet self-describes exact resource
cardinalities and refuses only ABI-unrepresentable values; arbitrary pre-score
budgets are deliberately absent.  Exact-score economics require reopening the
archive, evaluator, runtime, report, and raw-output custody bytes at the bottom
of this module.  Literal packet presence is still insufficient: exact admission
fails closed until the runtime also emits a provenance-bound strict parse/apply
receipt and a matched G-only counterfactual proves raw-output causality.
Caller-attested score coordinates or authority booleans are never accepted.

Lane-periodic chart programs are deliberately not accepted by this v1 wrapper.
Their exact receiver consumes inherited Lane chart state, while this decoder's
public input is only the predictor semantic partition and Pose6 state.  Adding
inert lane bytes or reconstructing hidden chart state would be a fake decoder.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import stat
import struct
import zipfile
import zlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final, Literal

import numpy as np

from tac.canonical_frontier_pointer import (
    CANONICAL_FRONTIER_POINTER_PATH,
    POINTER_SCHEMA_VERSION,
    POINTER_STALE_SECONDS,
    CanonicalFrontierPointer,
    recompute_effective_frontier,
)
from tac.contest_compliance import compute_upstream_snapshot_sha256
from tac.exact_eval_custody import (
    CONTEST_REFERENCE_BYTES,
    contest_score,
    validate_exact_eval_evidence,
)
from tac.optimization.direct_description_carrier_compose import (
    _ROLE_TO_WIRE,
    REALIZATION_PAINT_ORDER,
    ROLE_CLASS_IDS,
    BoundaryCoefficientDelta,
    BoundaryShearletAtomV1,
    IslandShapeAtomV1,
    MovableWorldsheetKnotV1,
    MovableWorldsheetTrackV1,
    ReceiverRealizationProfileV1,
    TopologyEventV1,
    _apply_boundary_coefficients,
    _apply_boundary_shearlet_atoms,
    _decode_boundary_coefficient_deltas,
    _decode_boundary_shearlet_atoms,
    _decode_island_shape_atoms,
    _decode_realization_profile,
    _decode_topology_events,
    _decode_worldsheet_knots,
    _decode_worldsheet_tracks,
    _encode_boundary_coefficient_deltas,
    _encode_boundary_shearlet_atoms,
    _encode_island_shape_atoms,
    _encode_realization_profile,
    _encode_topology_events,
    _encode_worldsheet_knots,
    _encode_worldsheet_tracks,
    _event_mask,
    _island_shape_mask,
    _worldsheet_track_mask,
    requires_pose6_transport,
)
from tac.optimization.direct_description_minimizer import DirectDescriptionError
from tac.witness_dsl.factorized_v9_predictor import (
    PREDICTOR_CONTRACT_ID,
    SEMANTIC_HEIGHT,
    SEMANTIC_WIDTH,
)

PACKET_MAGIC: Final = b"TACG1C\x00\x00"
PACKET_VERSION: Final = 1
PACKET_SCHEMA: Final = "tac.generative_taskspace_correction.v1"
ENCODER_ONLY_LINEAGE_POLICY: Final = "pbr1_pbr2_target_labels_obligation_ir_oracle_evidence_dense_y_encoder_only.v1"
PRIMITIVE_LINEAGE_POLICY: Final = "original_v9_v19c_receiver_closed_finite_taskspace_primitives.v1"
DECODER_PAYLOAD_POLICY: Final = "finite_primitive_parameters_only_no_target_table_dense_plane_or_preimage.v1"
JOINT_OBJECTIVE_ID: Final = "100*d_seg+sqrt(10*d_pose)+25*archive_bytes/37545489.v1"
EXACT_JOINT_AUTHORITY: Final = "exact_coupled_score_after_same_object_archive_decode.v1"
RECEIVER_CONSUMPTION_CUSTODY_ABSENT: Final = "receiver_consumption_custody_absent"

# Packet cardinalities are observations, not ceilings:
# magic/version, pair window, predictor binding, exact packet bytes, exact
# seven family counts, body bytes, CRC32.
_PACKET_PREFIX: Final = struct.Struct(">8sBHH32sI7HII")
_SECTION_PREFIX: Final = struct.Struct(">2sI")
_SECTION_ORDER: Final = (b"BC", b"TE", b"SH", b"IS", b"WT", b"WK", b"RP")
_SHA256_HEX_LENGTH: Final = 64
_ABI_UINT16_MAX: Final = 0xFFFF
_ABI_UINT32_MAX: Final = 0xFFFFFFFF
_CLASS_ID_BY_ROLE: Final = {role: ROLE_CLASS_IDS[role] for role in REALIZATION_PAINT_ORDER}
_REPORT_SAMPLES_RE: Final = re.compile(r"Evaluation results over\s+([0-9]+)\s+samples")
_REPORT_POSE_RE: Final = re.compile(r"Average PoseNet Distortion:\s*([0-9]+(?:\.[0-9]+)?)")
_REPORT_SEG_RE: Final = re.compile(r"Average SegNet Distortion:\s*([0-9]+(?:\.[0-9]+)?)")
_REPORT_BYTES_RE: Final = re.compile(r"Submission file size:\s*([0-9,]+)\s+bytes")
_REPORT_DEVICE_RE: Final = re.compile(r"^\s*device:\s*(\S+)\s*$", re.MULTILINE)
_EXACT_EVAL_TOOL: Final = "experiments/contest_auth_eval.py"


class GenerativeTaskspaceCorrectionError(ValueError):
    """Malformed G packet, forbidden lineage, or failed bounded admission."""


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _require_sha256(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_HEX_LENGTH
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise GenerativeTaskspaceCorrectionError(f"{label} must be lowercase SHA-256 hex")
    return value


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise GenerativeTaskspaceCorrectionError("correction identity is not canonical JSON") from exc


def _immutable_array(value: np.ndarray, *, dtype: np.dtype[Any]) -> np.ndarray:
    copied = np.ascontiguousarray(value, dtype=dtype).copy()
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True, slots=True)
class PredictorSemanticStateV1:
    """The only decoder-side state besides counted G bytes.

    Both arrays are re-derived from the counted V9 predictor.  No target label,
    scorer observation, PBR stream, or dense RGB plane belongs here.
    """

    predictor_program_sha256: str
    predictor_renderer_sha256: str
    source_pair_ids: tuple[int, ...]
    labels: np.ndarray
    pose6_codes: np.ndarray

    def __post_init__(self) -> None:
        _require_sha256(self.predictor_program_sha256, "predictor_program_sha256")
        _require_sha256(self.predictor_renderer_sha256, "predictor_renderer_sha256")
        if type(self.source_pair_ids) is not tuple or not self.source_pair_ids:
            raise GenerativeTaskspaceCorrectionError("source_pair_ids must be a nonempty exact tuple")
        if any(type(value) is not int for value in self.source_pair_ids):
            raise GenerativeTaskspaceCorrectionError("source_pair_ids must contain exact integers")
        expected = tuple(range(self.source_pair_ids[0], self.source_pair_ids[0] + len(self.source_pair_ids)))
        if self.source_pair_ids != expected or not 0 <= expected[0] < expected[-1] + 1 <= 600:
            raise GenerativeTaskspaceCorrectionError("source_pair_ids must be one contiguous subset of [0,600)")
        labels = np.asarray(self.labels)
        expected_shape = (len(expected), SEMANTIC_HEIGHT, SEMANTIC_WIDTH)
        if labels.dtype != np.uint8 or labels.shape != expected_shape:
            raise GenerativeTaskspaceCorrectionError("predictor labels must be canonical uint8 scorer-grid semantics")
        if int(labels.min()) < 0 or int(labels.max()) > 4:
            raise GenerativeTaskspaceCorrectionError("predictor labels escaped the five-class semantic universe")
        pose6 = np.asarray(self.pose6_codes)
        if pose6.shape != (len(expected), 6) or pose6.dtype.kind not in "iu":
            raise GenerativeTaskspaceCorrectionError("Pose6 state must be an integer [pair,6] array")
        if int(pose6.min()) < -32768 or int(pose6.max()) > 32767:
            raise GenerativeTaskspaceCorrectionError("Pose6 state escaped canonical int16 transport range")
        object.__setattr__(self, "labels", _immutable_array(labels, dtype=np.dtype(np.uint8)))
        object.__setattr__(self, "pose6_codes", _immutable_array(pose6, dtype=np.dtype(">i2")))

    @property
    def pair_start(self) -> int:
        return self.source_pair_ids[0]

    @property
    def pair_count(self) -> int:
        return len(self.source_pair_ids)

    @property
    def labels_sha256(self) -> str:
        return _sha256(memoryview(self.labels).cast("B"))

    @property
    def pose6_sha256(self) -> str:
        return _sha256(memoryview(self.pose6_codes).cast("B"))

    @property
    def binding_sha256(self) -> str:
        return _sha256(
            _canonical_json(
                {
                    "schema": "tac.generative_taskspace_predictor_binding.v1",
                    "predictor_contract_id": PREDICTOR_CONTRACT_ID,
                    "predictor_program_sha256": self.predictor_program_sha256,
                    "predictor_renderer_sha256": self.predictor_renderer_sha256,
                    "source_pair_ids": list(self.source_pair_ids),
                    "labels_sha256": self.labels_sha256,
                    "pose6_sha256": self.pose6_sha256,
                }
            )
        )


@dataclass(frozen=True, slots=True)
class GenerativeCorrectionProgramV1:
    """Finite primitive program; every field has an existing V9 receiver codec."""

    boundary_coefficients: tuple[BoundaryCoefficientDelta, ...] = ()
    topology_events: tuple[TopologyEventV1, ...] = ()
    boundary_shearlets: tuple[BoundaryShearletAtomV1, ...] = ()
    island_shapes: tuple[IslandShapeAtomV1, ...] = ()
    worldsheet_tracks: tuple[MovableWorldsheetTrackV1, ...] = ()
    worldsheet_knots: tuple[MovableWorldsheetKnotV1, ...] = ()
    realization_profile: ReceiverRealizationProfileV1 | None = None

    def __post_init__(self) -> None:
        typed_rows = (
            ("boundary_coefficients", BoundaryCoefficientDelta),
            ("topology_events", TopologyEventV1),
            ("boundary_shearlets", BoundaryShearletAtomV1),
            ("island_shapes", IslandShapeAtomV1),
            ("worldsheet_tracks", MovableWorldsheetTrackV1),
            ("worldsheet_knots", MovableWorldsheetKnotV1),
        )
        for field, row_type in typed_rows:
            rows = getattr(self, field)
            if type(rows) is not tuple or any(type(row) is not row_type for row in rows):
                raise GenerativeTaskspaceCorrectionError(f"{field} must contain only exact {row_type.__name__} rows")
        if self.realization_profile is not None and type(self.realization_profile) is not ReceiverRealizationProfileV1:
            raise GenerativeTaskspaceCorrectionError("realization_profile must use the existing finite V9 palette type")

    @property
    def family_counts(self) -> Mapping[str, int]:
        return {
            "boundary_coefficients": len(self.boundary_coefficients),
            "topology_events": len(self.topology_events),
            "boundary_shearlets": len(self.boundary_shearlets),
            "island_shapes": len(self.island_shapes),
            "worldsheet_tracks": len(self.worldsheet_tracks),
            "worldsheet_knots": len(self.worldsheet_knots),
            "realization_profile": int(self.realization_profile is not None),
        }

    @property
    def atom_count(self) -> int:
        return sum(self.family_counts.values())

    @property
    def semantic_atom_count(self) -> int:
        return self.atom_count - int(self.realization_profile is not None)


@dataclass(frozen=True, slots=True)
class CorrectionResourceCountsV1:
    """Exact packet cardinalities encoded in the ABI; none is an admission cap."""

    boundary_coefficients: int
    topology_events: int
    boundary_shearlets: int
    island_shapes: int
    worldsheet_tracks: int
    worldsheet_knots: int
    realization_profile: int

    def __post_init__(self) -> None:
        values = tuple(getattr(self, field) for field in self.__dataclass_fields__)
        if any(type(value) is not int for value in values):
            raise GenerativeTaskspaceCorrectionError("all correction resource counts must be exact integers")
        if any(not 0 <= value <= _ABI_UINT16_MAX for value in values):
            raise GenerativeTaskspaceCorrectionError("a correction resource count is not uint16-representable")
        if self.realization_profile not in {0, 1}:
            raise GenerativeTaskspaceCorrectionError("realization-profile cardinality must be exactly zero or one")

    @classmethod
    def from_program(cls, program: GenerativeCorrectionProgramV1) -> CorrectionResourceCountsV1:
        return cls(**dict(program.family_counts))

    @property
    def wire_counts(self) -> tuple[int, ...]:
        return tuple(getattr(self, field) for field in self.__dataclass_fields__)

    @property
    def total_atoms(self) -> int:
        return sum(self.wire_counts)

    def as_dict(self) -> dict[str, int]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class EncoderOnlyTeacherEvidenceV1:
    """Encoder-side PBR2 fit evidence.  None of these fields enters the packet."""

    pbr1_sha256: str
    pbr2_sha256: str
    target_labels_sha256: str
    obligation_ir_sha256: str
    oracle_evidence_sha256: str
    dense_y_sha256: str
    target_labels: np.ndarray
    teacher_event_count: int
    lineage_policy: Literal["pbr1_pbr2_target_labels_obligation_ir_oracle_evidence_dense_y_encoder_only.v1"] = (
        ENCODER_ONLY_LINEAGE_POLICY
    )
    serialized_teacher_bytes: Literal[0] = 0

    def __post_init__(self) -> None:
        for field in (
            "pbr1_sha256",
            "pbr2_sha256",
            "target_labels_sha256",
            "obligation_ir_sha256",
            "oracle_evidence_sha256",
            "dense_y_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        if self.lineage_policy != ENCODER_ONLY_LINEAGE_POLICY or self.serialized_teacher_bytes != 0:
            raise GenerativeTaskspaceCorrectionError("teacher evidence must remain wholly encoder-only")
        labels = np.asarray(self.target_labels)
        if (
            labels.dtype != np.uint8
            or labels.ndim != 3
            or labels.shape[0] < 1
            or labels.shape[1:] != (SEMANTIC_HEIGHT, SEMANTIC_WIDTH)
        ):
            raise GenerativeTaskspaceCorrectionError("encoder-only target labels changed semantic ABI")
        if int(labels.min()) < 0 or int(labels.max()) > 4:
            raise GenerativeTaskspaceCorrectionError("encoder-only target labels escaped the five-class universe")
        immutable = _immutable_array(labels, dtype=np.dtype(np.uint8))
        if _sha256(memoryview(immutable).cast("B")) != self.target_labels_sha256:
            raise GenerativeTaskspaceCorrectionError("encoder-only target label hash mismatches exact bytes")
        object.__setattr__(self, "target_labels", immutable)
        if type(self.teacher_event_count) is not int or self.teacher_event_count < 0:
            raise GenerativeTaskspaceCorrectionError("teacher event evidence is invalid")

    @property
    def binding_sha256(self) -> str:
        """Bind every encoder-only identity without serializing teacher payload."""

        return _sha256(
            _canonical_json(
                {
                    "schema": "tac.generative_taskspace_teacher_evidence_binding.v1",
                    "pbr1_sha256": self.pbr1_sha256,
                    "pbr2_sha256": self.pbr2_sha256,
                    "target_labels_sha256": self.target_labels_sha256,
                    "target_labels_shape": list(self.target_labels.shape),
                    "obligation_ir_sha256": self.obligation_ir_sha256,
                    "oracle_evidence_sha256": self.oracle_evidence_sha256,
                    "dense_y_sha256": self.dense_y_sha256,
                    "teacher_event_count": self.teacher_event_count,
                    "lineage_policy": self.lineage_policy,
                    "serialized_teacher_bytes": self.serialized_teacher_bytes,
                }
            )
        )


@dataclass(frozen=True, slots=True)
class ParsedGenerativeCorrectionV1:
    packet: bytes
    predictor_binding_sha256: str
    pair_start: int
    pair_count: int
    packet_bytes: int
    resource_counts: CorrectionResourceCountsV1
    program: GenerativeCorrectionProgramV1


@dataclass(frozen=True, slots=True)
class DecodedGenerativeCorrectionV1:
    """Generated frame-1 obligations plus the optional counted shared palette.

    The dense array is generic receiver output, not serialized payload.  It is
    still only a semantic obligation field: independent frame-0/Pose preimage
    construction and exact evaluator realization remain downstream work.
    """

    labels: np.ndarray
    realization_profile: ReceiverRealizationProfileV1 | None
    correction_packet_sha256: str

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels)
        if (
            labels.dtype != np.uint8
            or labels.ndim != 3
            or labels.shape[0] < 1
            or labels.shape[1:] != (SEMANTIC_HEIGHT, SEMANTIC_WIDTH)
        ):
            raise GenerativeTaskspaceCorrectionError("decoded correction labels changed semantic ABI")
        if int(labels.min()) < 0 or int(labels.max()) > 4:
            raise GenerativeTaskspaceCorrectionError("decoded correction labels escaped the five-class universe")
        object.__setattr__(self, "labels", _immutable_array(labels, dtype=np.dtype(np.uint8)))
        _require_sha256(self.correction_packet_sha256, "correction_packet_sha256")

    def paint_rgb(self) -> np.ndarray:
        """Consume the finite palette over corrected semantics; no scorer is used."""

        if self.realization_profile is None:
            raise GenerativeTaskspaceCorrectionError("RGB painting requires a counted realization profile")
        output = np.empty((*self.labels.shape, 3), dtype=np.uint8)
        for role in REALIZATION_PAINT_ORDER:
            output[self.labels == _CLASS_ID_BY_ROLE[role]] = self.realization_profile.colour_for(role)
        return np.ascontiguousarray(output)


@dataclass(frozen=True, slots=True)
class GenerativeCorrectionCompileReceiptV1:
    schema: Literal["tac.generative_taskspace_correction.v1"]
    packet_bytes: int
    packet_body_bytes: int
    packet_sha256: str
    predictor_binding_sha256: str
    resource_counts: CorrectionResourceCountsV1
    total_atoms: int
    pair_addressed_atoms: int
    active_atom_pair_incidence: int
    max_active_atoms_per_pair: int
    changed_cells: int
    debt_before_cells: int
    debt_after_cells: int
    debt_delta_cells: int
    residual_debt_cells: int
    teacher_evidence_binding_sha256: str
    pbr1_sha256: str
    pbr2_sha256: str
    target_labels_sha256: str
    obligation_ir_sha256: str
    oracle_evidence_sha256: str
    dense_y_sha256: str
    teacher_event_count: int
    encoder_only_lineage_policy: Literal[
        "pbr1_pbr2_target_labels_obligation_ir_oracle_evidence_dense_y_encoder_only.v1"
    ]
    serialized_teacher_bytes: Literal[0]
    serialized_dense_semantic_bytes: Literal[0]
    serialized_dense_y_bytes: Literal[0]
    serialized_explicit_preimage_bytes: Literal[0]
    primitive_lineage_policy: Literal["original_v9_v19c_receiver_closed_finite_taskspace_primitives.v1"]
    decoder_payload_policy: Literal["finite_primitive_parameters_only_no_target_table_dense_plane_or_preimage.v1"]
    encoder_teacher_role: Literal["pbr2_acquisition_strata_never_candidate_payload"]
    decoded_obligation_scope: Literal["generated_frame1_semantic_obligations_only"]
    independent_frame0_pose_preimage_owed: Literal[True]
    evaluator_realization_and_exact_score_owed: Literal[True]
    exact_semantic_target_reconstructed: bool
    exact_target_match_is_not_lineage_authority: Literal[True]
    abi_representable: Literal[True]
    arbitrary_pre_score_caps_applied: Literal[False]
    candidate_payload_eligible: Literal[True]
    research_only: Literal[True]
    score_claim: Literal[False]
    promotion_eligible: Literal[False]

    @property
    def family_counts(self) -> Mapping[str, int]:
        return self.resource_counts.as_dict()

    @property
    def binding_sha256(self) -> str:
        payload = {field: getattr(self, field) for field in self.__dataclass_fields__}
        payload["resource_counts"] = self.resource_counts.as_dict()
        return _sha256(
            _canonical_json(
                {
                    "schema": "tac.generative_taskspace_compile_receipt_binding.v1",
                    "receipt": payload,
                }
            )
        )


@dataclass(frozen=True, slots=True)
class CompiledGenerativeCorrectionV1:
    packet: bytes
    decoded: DecodedGenerativeCorrectionV1
    receipt: GenerativeCorrectionCompileReceiptV1
    receipt_binding_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.receipt_binding_sha256, "receipt_binding_sha256")
        if self.receipt_binding_sha256 != self.receipt.binding_sha256:
            raise GenerativeTaskspaceCorrectionError("compiled correction receipt binding is inconsistent")


def _canonical_program(program: GenerativeCorrectionProgramV1) -> GenerativeCorrectionProgramV1:
    return GenerativeCorrectionProgramV1(
        boundary_coefficients=tuple(
            sorted(program.boundary_coefficients, key=lambda row: (row.pair_index, row.coefficient_index))
        ),
        topology_events=tuple(
            sorted(
                program.topology_events,
                key=lambda row: (
                    row.pair_index,
                    _ROLE_TO_WIRE[row.role],
                    0 if row.action == "birth" else 1,
                    0 if row.shape == "ellipse" else 1,
                    row.y0,
                    row.x0,
                    row.y1,
                    row.x1,
                ),
            )
        ),
        boundary_shearlets=tuple(
            sorted(
                program.boundary_shearlets,
                key=lambda row: (row.pair_index, _ROLE_TO_WIRE[row.role], row.center_y, row.center_x),
            )
        ),
        island_shapes=tuple(
            sorted(
                program.island_shapes,
                key=lambda row: (row.pair_index, 0 if row.action == "birth" else 1, row.center_y, row.center_x),
            )
        ),
        worldsheet_tracks=tuple(sorted(program.worldsheet_tracks, key=lambda row: row.object_id)),
        worldsheet_knots=tuple(sorted(program.worldsheet_knots, key=lambda row: (row.object_id, row.pair_index))),
        realization_profile=program.realization_profile,
    )


def _validate_program_window(program: GenerativeCorrectionProgramV1, state: PredictorSemanticStateV1) -> None:
    start = state.pair_start
    stop = start + state.pair_count
    addressed = (
        *program.boundary_coefficients,
        *program.topology_events,
        *program.boundary_shearlets,
        *program.island_shapes,
    )
    if any(not start <= row.pair_index < stop for row in addressed):
        raise GenerativeTaskspaceCorrectionError("primitive address escaped the bound predictor pair population")
    if any(row.pair_index + row.lifetime > stop for row in (*program.topology_events, *program.island_shapes)):
        raise GenerativeTaskspaceCorrectionError("primitive lifetime escaped the bound predictor pair population")
    if any(
        row.birth_pair < start or row.death_pair_exclusive > stop or row.death_pair_exclusive <= row.birth_pair
        for row in program.worldsheet_tracks
    ):
        raise GenerativeTaskspaceCorrectionError("worldsheet lifecycle escaped the bound predictor pair population")
    tracks = {row.object_id: row for row in program.worldsheet_tracks}
    if len(tracks) != len(program.worldsheet_tracks):
        raise GenerativeTaskspaceCorrectionError("worldsheet object IDs must be unique")
    if any(
        row.object_id not in tracks
        or not tracks[row.object_id].birth_pair <= row.pair_index < tracks[row.object_id].death_pair_exclusive
        for row in program.worldsheet_knots
    ):
        raise GenerativeTaskspaceCorrectionError("worldsheet knot escaped its declared finite lifecycle")
    if program.semantic_atom_count == 0:
        raise GenerativeTaskspaceCorrectionError("a palette-only or empty packet is not a task-space correction")


def _pair_address_counts(program: GenerativeCorrectionProgramV1) -> Counter[int]:
    counts: Counter[int] = Counter()
    for row in (
        *program.boundary_coefficients,
        *program.topology_events,
        *program.boundary_shearlets,
        *program.island_shapes,
        *program.worldsheet_knots,
    ):
        counts[row.pair_index] += 1
    for row in program.worldsheet_tracks:
        counts[row.birth_pair] += 1
    return counts


def _active_pair_counts(
    program: GenerativeCorrectionProgramV1,
    state: PredictorSemanticStateV1,
) -> Counter[int]:
    """Count every atom on every pair whose receiver output it can affect."""

    counts: Counter[int] = Counter()
    for row in (*program.boundary_coefficients, *program.boundary_shearlets):
        counts[row.pair_index] += 1
    for row in (*program.topology_events, *program.island_shapes):
        for pair_index in range(row.pair_index, row.pair_index + row.lifetime):
            counts[pair_index] += 1
    knots_by_object = Counter(row.object_id for row in program.worldsheet_knots)
    for track in program.worldsheet_tracks:
        active_parameters = 1 + knots_by_object[track.object_id]
        for pair_index in range(track.birth_pair, track.death_pair_exclusive):
            counts[pair_index] += active_parameters
    if program.realization_profile is not None:
        for pair_index in state.source_pair_ids:
            counts[pair_index] += 1
    return counts


def _encode_sections(program: GenerativeCorrectionProgramV1) -> bytes:
    try:
        payloads = (
            _encode_boundary_coefficient_deltas(program.boundary_coefficients),
            _encode_topology_events(program.topology_events),
            _encode_boundary_shearlet_atoms(program.boundary_shearlets),
            _encode_island_shape_atoms(program.island_shapes),
            _encode_worldsheet_tracks(program.worldsheet_tracks),
            _encode_worldsheet_knots(program.worldsheet_knots),
            _encode_realization_profile(program.realization_profile),
        )
    except DirectDescriptionError as exc:
        raise GenerativeTaskspaceCorrectionError("existing V9 primitive codec refused the correction program") from exc
    if any(len(payload) > _ABI_UINT32_MAX for payload in payloads):
        raise GenerativeTaskspaceCorrectionError("a correction section is not uint32-length-representable")
    body = b"".join(
        _SECTION_PREFIX.pack(section_id, len(payload)) + payload
        for section_id, payload in zip(_SECTION_ORDER, payloads, strict=True)
    )
    if len(body) > _ABI_UINT32_MAX:
        raise GenerativeTaskspaceCorrectionError("correction body is not uint32-length-representable")
    return body


def _encode_packet(
    state: PredictorSemanticStateV1,
    program: GenerativeCorrectionProgramV1,
) -> bytes:
    body = _encode_sections(program)
    resource_counts = CorrectionResourceCountsV1.from_program(program)
    packet_bytes = _PACKET_PREFIX.size + len(body)
    if packet_bytes > _ABI_UINT32_MAX:
        raise GenerativeTaskspaceCorrectionError("correction packet is not uint32-length-representable")
    prefix_values = (
        PACKET_MAGIC,
        PACKET_VERSION,
        state.pair_start,
        state.pair_count,
        bytes.fromhex(state.binding_sha256),
        packet_bytes,
        *resource_counts.wire_counts,
        len(body),
    )
    zero_checksum_prefix = _PACKET_PREFIX.pack(*prefix_values, 0)
    checksum = zlib.crc32(zero_checksum_prefix + body) & 0xFFFFFFFF
    return _PACKET_PREFIX.pack(*prefix_values, checksum) + body


def _decode_sections(body: bytes) -> GenerativeCorrectionProgramV1:
    payloads: list[bytes] = []
    offset = 0
    for expected_id in _SECTION_ORDER:
        if offset + _SECTION_PREFIX.size > len(body):
            raise GenerativeTaskspaceCorrectionError("correction section table is truncated")
        section_id, length = _SECTION_PREFIX.unpack_from(body, offset)
        offset += _SECTION_PREFIX.size
        stop = offset + length
        if section_id != expected_id or stop > len(body):
            raise GenerativeTaskspaceCorrectionError("correction section order or length is invalid")
        payloads.append(body[offset:stop])
        offset = stop
    if offset != len(body):
        raise GenerativeTaskspaceCorrectionError("correction packet has trailing or extra sections")
    try:
        return GenerativeCorrectionProgramV1(
            boundary_coefficients=_decode_boundary_coefficient_deltas(payloads[0]),
            topology_events=_decode_topology_events(payloads[1]),
            boundary_shearlets=_decode_boundary_shearlet_atoms(payloads[2]),
            island_shapes=_decode_island_shape_atoms(payloads[3]),
            worldsheet_tracks=_decode_worldsheet_tracks(payloads[4]),
            worldsheet_knots=_decode_worldsheet_knots(payloads[5]),
            realization_profile=_decode_realization_profile(payloads[6]),
        )
    except DirectDescriptionError as exc:
        raise GenerativeTaskspaceCorrectionError("existing V9 primitive parse-back refused the packet") from exc


def _enforce_declared_resource_counts(
    program: GenerativeCorrectionProgramV1,
    declared: CorrectionResourceCountsV1,
) -> None:
    actual = CorrectionResourceCountsV1.from_program(program)
    if actual != declared:
        raise GenerativeTaskspaceCorrectionError("correction packet resource counts differ from decoded sections")


def parse_generative_taskspace_correction(
    packet: bytes,
    *,
    predictor_state: PredictorSemanticStateV1,
) -> ParsedGenerativeCorrectionV1:
    """Strictly parse, bind, resource-check, and canonicalize one counted G packet."""

    if type(packet) is not bytes or len(packet) < _PACKET_PREFIX.size:
        raise GenerativeTaskspaceCorrectionError("correction packet is empty, non-bytes, or truncated")
    values = _PACKET_PREFIX.unpack_from(packet)
    magic, version, pair_start, pair_count, binding, packet_bytes, *tail = values
    resource_counts = CorrectionResourceCountsV1(*tail[:7])
    body_length, checksum = tail[7:]
    if magic != PACKET_MAGIC or version != PACKET_VERSION:
        raise GenerativeTaskspaceCorrectionError("correction packet magic/version is invalid")
    if (pair_start, pair_count) != (predictor_state.pair_start, predictor_state.pair_count):
        raise GenerativeTaskspaceCorrectionError("correction packet pair population differs from predictor semantics")
    if binding.hex() != predictor_state.binding_sha256:
        raise GenerativeTaskspaceCorrectionError("correction packet predictor identity binding mismatch")
    if len(packet) != packet_bytes or packet_bytes != _PACKET_PREFIX.size + body_length:
        raise GenerativeTaskspaceCorrectionError("correction packet length differs from its self-described bytes")
    body = packet[_PACKET_PREFIX.size :]
    zero_checksum_prefix = _PACKET_PREFIX.pack(
        magic,
        version,
        pair_start,
        pair_count,
        binding,
        packet_bytes,
        *resource_counts.wire_counts,
        body_length,
        0,
    )
    if (zlib.crc32(zero_checksum_prefix + body) & 0xFFFFFFFF) != checksum:
        raise GenerativeTaskspaceCorrectionError("correction packet CRC mismatch")
    program = _decode_sections(body)
    _validate_program_window(program, predictor_state)
    _enforce_declared_resource_counts(program, resource_counts)
    if _encode_packet(predictor_state, program) != packet:
        raise GenerativeTaskspaceCorrectionError("correction packet is not byte-canonical on parse-back")
    return ParsedGenerativeCorrectionV1(
        packet=packet,
        predictor_binding_sha256=binding.hex(),
        pair_start=pair_start,
        pair_count=pair_count,
        packet_bytes=packet_bytes,
        resource_counts=resource_counts,
        program=program,
    )


def _apply_program(
    program: GenerativeCorrectionProgramV1,
    state: PredictorSemanticStateV1,
) -> DecodedGenerativeCorrectionV1:
    output = np.empty_like(state.labels)
    knots_by_object: dict[int, tuple[MovableWorldsheetKnotV1, ...]] = {
        track.object_id: tuple(row for row in program.worldsheet_knots if row.object_id == track.object_id)
        for track in program.worldsheet_tracks
    }
    for local_pair, source_pair in enumerate(state.source_pair_ids):
        masks = {role: state.labels[local_pair] == class_id for role, class_id in _CLASS_ID_BY_ROLE.items()}
        for role in ("Road", "UndrivableBoundary"):
            coefficients = tuple(
                row for row in program.boundary_coefficients if row.pair_index == source_pair and row.role == role
            )
            shearlets = tuple(
                row for row in program.boundary_shearlets if row.pair_index == source_pair and row.role == role
            )
            masks[role] = _apply_boundary_coefficients(masks[role], coefficients)
            masks[role] = _apply_boundary_shearlet_atoms(masks[role], shearlets)
        for event in program.topology_events:
            event_sites = _event_mask(
                event,
                source_pair_id=source_pair,
                source_pair_start=state.pair_start,
                pose6_codes=state.pose6_codes if requires_pose6_transport(event) else None,
            )
            masks[event.role] = (
                masks[event.role] | event_sites if event.action == "birth" else masks[event.role] & ~event_sites
            )
        for atom in program.island_shapes:
            atom_sites = _island_shape_mask(
                atom,
                source_pair_id=source_pair,
                source_pair_start=state.pair_start,
                pose6_codes=state.pose6_codes if requires_pose6_transport(atom) else None,
            )
            masks["Movable"] = (
                masks["Movable"] | atom_sites if atom.action == "birth" else masks["Movable"] & ~atom_sites
            )
        for track in program.worldsheet_tracks:
            masks["Movable"] |= _worldsheet_track_mask(
                track,
                knots_by_object[track.object_id],
                source_pair_id=source_pair,
                source_pair_start=state.pair_start,
                pose6_codes=state.pose6_codes,
            )
        merged = np.full((SEMANTIC_HEIGHT, SEMANTIC_WIDTH), ROLE_CLASS_IDS["UndrivableBoundary"], dtype=np.uint8)
        for role in REALIZATION_PAINT_ORDER:
            merged[masks[role]] = np.uint8(_CLASS_ID_BY_ROLE[role])
        output[local_pair] = merged
    return DecodedGenerativeCorrectionV1(
        labels=np.ascontiguousarray(output),
        realization_profile=program.realization_profile,
        correction_packet_sha256="0" * 64,
    )


def apply_generative_taskspace_correction(
    packet: bytes,
    *,
    predictor_state: PredictorSemanticStateV1,
) -> DecodedGenerativeCorrectionV1:
    """Decode G using only identity-bound predictor semantics and counted bytes."""

    parsed = parse_generative_taskspace_correction(packet, predictor_state=predictor_state)
    decoded = _apply_program(parsed.program, predictor_state)
    return DecodedGenerativeCorrectionV1(
        labels=decoded.labels,
        realization_profile=decoded.realization_profile,
        correction_packet_sha256=_sha256(packet),
    )


def _validate_teacher_debt_custody(
    evidence: EncoderOnlyTeacherEvidenceV1,
    *,
    debt_before_cells: int,
) -> None:
    if evidence.teacher_event_count != debt_before_cells:
        raise GenerativeTaskspaceCorrectionError("PBR2 teacher event count differs from exact predictor-target debt")


def compile_generative_taskspace_correction(
    predictor_state: PredictorSemanticStateV1,
    program: GenerativeCorrectionProgramV1,
    *,
    teacher_evidence: EncoderOnlyTeacherEvidenceV1,
) -> CompiledGenerativeCorrectionV1:
    """Compile and behaviorally close a finite, lineage-clean G packet."""

    canonical = _canonical_program(program)
    _validate_program_window(canonical, predictor_state)
    resource_counts = CorrectionResourceCountsV1.from_program(canonical)
    packet = _encode_packet(predictor_state, canonical)
    parsed = parse_generative_taskspace_correction(packet, predictor_state=predictor_state)
    if parsed.program != canonical or parsed.resource_counts != resource_counts:
        raise GenerativeTaskspaceCorrectionError("compiled primitive program changed on strict parse-back")
    decoded = apply_generative_taskspace_correction(packet, predictor_state=predictor_state)
    replay = apply_generative_taskspace_correction(packet, predictor_state=predictor_state)
    if not np.array_equal(decoded.labels, replay.labels):
        raise GenerativeTaskspaceCorrectionError("generative correction decode is nondeterministic")
    changed_cells = int(np.count_nonzero(decoded.labels != predictor_state.labels))
    if changed_cells == 0:
        raise GenerativeTaskspaceCorrectionError("correction program is a receiver-output no-op")
    target_labels = teacher_evidence.target_labels
    if target_labels.shape != predictor_state.labels.shape:
        raise GenerativeTaskspaceCorrectionError("encoder-only target pair population differs from predictor semantics")
    debt_before_cells = int(np.count_nonzero(predictor_state.labels != target_labels))
    debt_after_cells = int(np.count_nonzero(decoded.labels != target_labels))
    _validate_teacher_debt_custody(
        teacher_evidence,
        debt_before_cells=debt_before_cells,
    )
    pair_counts = _pair_address_counts(canonical)
    active_pair_counts = _active_pair_counts(canonical, predictor_state)
    receipt = GenerativeCorrectionCompileReceiptV1(
        schema=PACKET_SCHEMA,
        packet_bytes=len(packet),
        packet_body_bytes=len(packet) - _PACKET_PREFIX.size,
        packet_sha256=_sha256(packet),
        predictor_binding_sha256=predictor_state.binding_sha256,
        resource_counts=resource_counts,
        total_atoms=canonical.atom_count,
        pair_addressed_atoms=sum(pair_counts.values()),
        active_atom_pair_incidence=sum(active_pair_counts.values()),
        max_active_atoms_per_pair=max(active_pair_counts.values(), default=0),
        changed_cells=changed_cells,
        debt_before_cells=debt_before_cells,
        debt_after_cells=debt_after_cells,
        debt_delta_cells=debt_after_cells - debt_before_cells,
        residual_debt_cells=debt_after_cells,
        teacher_evidence_binding_sha256=teacher_evidence.binding_sha256,
        pbr1_sha256=teacher_evidence.pbr1_sha256,
        pbr2_sha256=teacher_evidence.pbr2_sha256,
        target_labels_sha256=teacher_evidence.target_labels_sha256,
        obligation_ir_sha256=teacher_evidence.obligation_ir_sha256,
        oracle_evidence_sha256=teacher_evidence.oracle_evidence_sha256,
        dense_y_sha256=teacher_evidence.dense_y_sha256,
        teacher_event_count=teacher_evidence.teacher_event_count,
        encoder_only_lineage_policy=teacher_evidence.lineage_policy,
        serialized_teacher_bytes=0,
        serialized_dense_semantic_bytes=0,
        serialized_dense_y_bytes=0,
        serialized_explicit_preimage_bytes=0,
        primitive_lineage_policy=PRIMITIVE_LINEAGE_POLICY,
        decoder_payload_policy=DECODER_PAYLOAD_POLICY,
        encoder_teacher_role="pbr2_acquisition_strata_never_candidate_payload",
        decoded_obligation_scope="generated_frame1_semantic_obligations_only",
        independent_frame0_pose_preimage_owed=True,
        evaluator_realization_and_exact_score_owed=True,
        exact_semantic_target_reconstructed=debt_after_cells == 0,
        exact_target_match_is_not_lineage_authority=True,
        abi_representable=True,
        arbitrary_pre_score_caps_applied=False,
        candidate_payload_eligible=True,
        research_only=True,
        score_claim=False,
        promotion_eligible=False,
    )
    return CompiledGenerativeCorrectionV1(
        packet=packet,
        decoded=decoded,
        receipt=receipt,
        receipt_binding_sha256=receipt.binding_sha256,
    )


@dataclass(frozen=True, slots=True)
class ExactEvalCustodyPathsV1:
    """Paths whose bytes are reopened; this type contains no score assertions."""

    custody_root: Path
    result_json_path: Path
    archive_path: Path
    provenance_json_path: Path
    report_path: Path
    runtime_root: Path
    inflated_root: Path
    inflated_outputs_manifest_path: Path
    upstream_evaluate_path: Path
    video_names_path: Path

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if not isinstance(value, Path):
                raise GenerativeTaskspaceCorrectionError(f"{field} must be an explicit pathlib.Path")


@dataclass(frozen=True, slots=True)
class _ReopenedExactEvalV1:
    result_artifact_sha256: str
    provenance_artifact_sha256: str
    report_sha256: str
    archive_sha256: str
    archive_bytes: int
    runtime_tree_sha256: str
    runtime_content_tree_sha256: str
    evaluator_sha256: str
    raw_output_aggregate_sha256: str
    inflated_outputs_manifest_sha256: str
    video_names_sha256: str
    upstream_snapshot_sha256: str
    upstream_commit: str | None
    source_object_binding_sha256: str
    authority_axis: Literal["contest_cpu", "contest_cuda"]
    n_samples: Literal[600]
    d_seg: float
    d_pose: float
    joint_score: float
    literal_g_archive_member_path: str | None


@dataclass(frozen=True, slots=True)
class ValuePerByteAdmissionV1:
    baseline_joint_score: float
    candidate_joint_score: float
    exact_archive_byte_delta: int
    delta_d_seg: float
    delta_d_pose: float
    segmentation_score_delta: float
    pose_score_delta: float
    rate_score_delta: float
    joint_score_delta: float
    joint_score_reduction: float
    joint_score_reduction_per_added_byte: float | None
    archive_byte_relation: Literal["smaller", "same", "larger"]
    local_step_improved: bool
    competitive_frontier_broken: bool
    local_step_rule: Literal["candidate_exact_joint_score_less_than_same_object_baseline.v1"]
    competitive_frontier_rule: Literal["candidate_exact_joint_score_less_than_live_effective_frontier.v1"]
    competitive_frontier_score: float
    competitive_frontier_axis: str
    canonical_frontier_pointer_sha256: str
    baseline_archive_sha256: str
    candidate_archive_sha256: str
    baseline_result_artifact_sha256: str
    candidate_result_artifact_sha256: str
    baseline_raw_output_aggregate_sha256: str
    candidate_raw_output_aggregate_sha256: str
    literal_g_archive_member_path: str
    literal_g_packet_sha256: str
    evaluator_sha256: str
    authority: Literal["exact_coupled_score_after_same_object_archive_decode.v1"]
    authority_axis: Literal["contest_cpu", "contest_cuda"]
    exact_custody_reopened: Literal[True]
    canonical_pointer_mutated: Literal[False]
    research_only: Literal[True]
    score_claim: Literal[False]
    promotion_eligible: Literal[False]


@dataclass(frozen=True, slots=True)
class _StableFileSnapshotV1:
    size: int
    sha256: str
    payload: bytes | None


@dataclass(frozen=True, slots=True)
class _StableArchiveSnapshotV1:
    size: int
    sha256: str
    literal_g_archive_member_path: str | None


def _stable_file_snapshot(
    path: Path,
    *,
    label: str,
    capture_payload: bool,
) -> _StableFileSnapshotV1:
    """Read/hash one regular file from one O_NOFOLLOW fd and stable fstats."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise GenerativeTaskspaceCorrectionError(f"{label} cannot be opened without symlink following") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise GenerativeTaskspaceCorrectionError(f"{label} is not a regular custody file")
        digest = hashlib.sha256()
        chunks: list[bytes] | None = [] if capture_payload else None
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = os.fstat(descriptor)
        stable_fields_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        stable_fields_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if stable_fields_before != stable_fields_after or total != before.st_size:
            raise GenerativeTaskspaceCorrectionError(f"{label} changed during its custody snapshot")
        return _StableFileSnapshotV1(
            size=total,
            sha256=digest.hexdigest(),
            payload=b"".join(chunks) if chunks is not None else None,
        )
    except OSError as exc:
        raise GenerativeTaskspaceCorrectionError(f"{label} could not be read as one stable snapshot") from exc
    finally:
        os.close(descriptor)


def _stable_archive_snapshot(
    path: Path,
    *,
    label: str,
    required_member_path: str | None = None,
    required_member_payload: bytes | None = None,
) -> _StableArchiveSnapshotV1:
    """Hash one ZIP and optionally prove one exact member on the same stable fd.

    The selected member is caller-addressed but never caller-attested: its
    decompressed bytes must equal ``required_member_payload`` exactly.  Only
    that one exact-size member is decompressed, and comparison is streaming,
    so a ZIP bomb cannot turn admission into an unbounded extraction.
    """

    if (required_member_path is None) != (required_member_payload is None):
        raise GenerativeTaskspaceCorrectionError("literal G archive member path and payload must be paired")
    if required_member_path is not None:
        if not isinstance(required_member_path, str) or type(required_member_payload) is not bytes:
            raise GenerativeTaskspaceCorrectionError("literal G archive member selector has noncanonical types")
        pure_required = PurePosixPath(required_member_path)
        if (
            pure_required.is_absolute()
            or not pure_required.parts
            or "\\" in required_member_path
            or any(part in {"", ".", ".."} for part in pure_required.parts)
        ):
            raise GenerativeTaskspaceCorrectionError("literal G archive member path is unsafe")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise GenerativeTaskspaceCorrectionError(f"{label} cannot be opened without symlink following") from exc

    def hash_descriptor() -> tuple[int, str]:
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        return total, digest.hexdigest()

    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise GenerativeTaskspaceCorrectionError(f"{label} is not a regular custody file")
        first_size, first_sha = hash_descriptor()
        literal_member: str | None = None
        if required_member_path is not None:
            assert required_member_payload is not None
            os.lseek(descriptor, 0, os.SEEK_SET)
            with (
                os.fdopen(os.dup(descriptor), "rb") as archive_file,
                zipfile.ZipFile(archive_file) as archive,
            ):
                seen_names: set[str] = set()
                selected: list[zipfile.ZipInfo] = []
                for info in archive.infolist():
                    pure = PurePosixPath(info.filename)
                    if (
                        pure.is_absolute()
                        or not pure.parts
                        or "\\" in info.filename
                        or any(part in {"", ".", ".."} for part in pure.parts)
                    ):
                        raise GenerativeTaskspaceCorrectionError("candidate archive contains an unsafe member path")
                    if info.filename in seen_names:
                        raise GenerativeTaskspaceCorrectionError("candidate archive contains duplicate member names")
                    seen_names.add(info.filename)
                    if info.filename == required_member_path:
                        selected.append(info)
                if len(selected) != 1:
                    raise GenerativeTaskspaceCorrectionError("candidate archive lacks one unique literal G member")
                info = selected[0]
                if info.is_dir() or info.flag_bits & 0x1:
                    raise GenerativeTaskspaceCorrectionError("literal G archive member is a directory or encrypted")
                if info.file_size != len(required_member_payload):
                    raise GenerativeTaskspaceCorrectionError(
                        "literal G archive member size differs from compiled canonical G bytes"
                    )
                offset = 0
                with archive.open(info, "r") as member:
                    while offset < len(required_member_payload):
                        chunk = member.read(min(1024 * 1024, len(required_member_payload) - offset))
                        if not chunk or chunk != required_member_payload[offset : offset + len(chunk)]:
                            raise GenerativeTaskspaceCorrectionError(
                                "literal G archive member differs from compiled canonical G bytes"
                            )
                        offset += len(chunk)
                    if member.read(1) != b"":
                        raise GenerativeTaskspaceCorrectionError(
                            "literal G archive member exceeds compiled canonical G bytes"
                        )
                literal_member = info.filename

        second_size, second_sha = hash_descriptor()
        after = os.fstat(descriptor)
        stable_fields_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        stable_fields_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if (
            stable_fields_before != stable_fields_after
            or first_size != before.st_size
            or second_size != before.st_size
            or first_sha != second_sha
        ):
            raise GenerativeTaskspaceCorrectionError(f"{label} changed during its custody snapshot")
        return _StableArchiveSnapshotV1(
            size=first_size,
            sha256=first_sha,
            literal_g_archive_member_path=literal_member,
        )
    except (OSError, RuntimeError, EOFError, zipfile.BadZipFile, zlib.error) as exc:
        raise GenerativeTaskspaceCorrectionError(f"{label} could not be read as one stable ZIP snapshot") from exc
    finally:
        os.close(descriptor)


def _json_without_duplicate_keys(raw: bytes, *, label: str) -> Mapping[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise GenerativeTaskspaceCorrectionError(f"{label} contains duplicate JSON key {key!r}")
            output[key] = value
        return output

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=object_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GenerativeTaskspaceCorrectionError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise GenerativeTaskspaceCorrectionError(f"{label} root must be a JSON object")
    return value


def _resolve_custody_path(
    root: Path,
    path: Path,
    *,
    label: str,
    directory: bool = False,
) -> Path:
    try:
        resolved = (path if path.is_absolute() else root / path).resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise GenerativeTaskspaceCorrectionError(f"{label} is missing or escaped the custody root") from exc
    path_text = resolved.as_posix()
    if path_text.startswith(("/tmp/", "/var/tmp/", "/private/tmp/")):
        raise GenerativeTaskspaceCorrectionError(f"{label} is transient rather than durable custody")
    if directory and not resolved.is_dir():
        raise GenerativeTaskspaceCorrectionError(f"{label} must be a custody directory")
    if not directory and not resolved.is_file():
        raise GenerativeTaskspaceCorrectionError(f"{label} must be a custody file")
    return resolved


def _safe_member_path(root: Path, relative_path: object, *, label: str) -> Path:
    if not isinstance(relative_path, str):
        raise GenerativeTaskspaceCorrectionError(f"{label} relative path is not text")
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise GenerativeTaskspaceCorrectionError(f"{label} relative path is unsafe")
    try:
        resolved = (root / Path(*pure.parts)).resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise GenerativeTaskspaceCorrectionError(f"{label} file is missing or escaped custody") from exc
    if not resolved.is_file():
        raise GenerativeTaskspaceCorrectionError(f"{label} does not resolve to a file")
    return resolved


def _exact_nonnegative_float(mapping: Mapping[str, Any], key: str, *, label: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise GenerativeTaskspaceCorrectionError(f"{label}.{key} must be an exact JSON number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise GenerativeTaskspaceCorrectionError(f"{label}.{key} must be finite and nonnegative")
    return result


def _exact_positive_int(mapping: Mapping[str, Any], key: str, *, label: str) -> int:
    value = mapping.get(key)
    if type(value) is not int or value <= 0:
        raise GenerativeTaskspaceCorrectionError(f"{label}.{key} must be an exact positive integer")
    return value


def _parse_exact_report(report_payload: bytes) -> tuple[int, float, float, int, str]:
    try:
        text = report_payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GenerativeTaskspaceCorrectionError("exact evaluator report is unreadable") from exc
    matches = (
        _REPORT_SAMPLES_RE.search(text),
        _REPORT_POSE_RE.search(text),
        _REPORT_SEG_RE.search(text),
        _REPORT_BYTES_RE.search(text),
        _REPORT_DEVICE_RE.search(text),
    )
    if any(match is None for match in matches):
        raise GenerativeTaskspaceCorrectionError("exact evaluator report is missing canonical metric lines")
    samples_match, pose_match, seg_match, bytes_match, device_match = matches
    assert samples_match is not None
    assert pose_match is not None
    assert seg_match is not None
    assert bytes_match is not None
    assert device_match is not None
    n_samples = int(samples_match.group(1))
    d_pose = float(pose_match.group(1))
    d_seg = float(seg_match.group(1))
    archive_bytes = int(bytes_match.group(1).replace(",", ""))
    if n_samples != 600 or not all(math.isfinite(value) and value >= 0.0 for value in (d_pose, d_seg)):
        raise GenerativeTaskspaceCorrectionError("exact evaluator report is not a finite n600 observation")
    return n_samples, d_seg, d_pose, archive_bytes, device_match.group(1)


def _validate_manifest_rows(rows: object, root: Path, *, label: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(rows, list):
        raise GenerativeTaskspaceCorrectionError(f"{label} rows must be an exact JSON list")
    validated: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise GenerativeTaskspaceCorrectionError(f"{label}[{index}] must be an object")
        relative = row.get("relative_path")
        if not isinstance(relative, str) or relative in seen:
            raise GenerativeTaskspaceCorrectionError(f"{label} paths must be unique text")
        seen.add(relative)
        path = _safe_member_path(root, relative, label=f"{label}[{index}]")
        expected_bytes = _exact_positive_int(row, "bytes", label=f"{label}[{index}]")
        expected_sha = _require_sha256(str(row.get("sha256")), f"{label}[{index}].sha256")
        snapshot = _stable_file_snapshot(
            path,
            label=f"{label}[{index}]",
            capture_payload=False,
        )
        if snapshot.size != expected_bytes or snapshot.sha256 != expected_sha:
            raise GenerativeTaskspaceCorrectionError(f"{label}[{index}] bytes differ from its manifest")
        validated.append(row)
    return tuple(validated)


def _validate_runtime_custody(
    provenance: Mapping[str, Any],
    *,
    runtime_root: Path,
    evaluator_path: Path,
) -> tuple[str, str, str]:
    manifest = provenance.get("inflate_runtime_manifest")
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema") != "contest_auth_eval_runtime_dependency_manifest_v1"
    ):
        raise GenerativeTaskspaceCorrectionError("exact evaluation lacks the canonical runtime dependency manifest")
    files = _validate_manifest_rows(manifest.get("files"), runtime_root, label="runtime_manifest.files")
    if manifest.get("runtime_file_count") != len(files):
        raise GenerativeTaskspaceCorrectionError("runtime manifest file cardinality is inconsistent")
    if not files or not any(row.get("relative_path") == "inflate.sh" for row in files):
        raise GenerativeTaskspaceCorrectionError("runtime manifest does not close the inflate entrypoint")
    if manifest.get("external_dependency_roots") != []:
        raise GenerativeTaskspaceCorrectionError("runtime has external dependency roots outside this custody bundle")
    repo_tac = manifest.get("repo_local_tac_import_manifest")
    if not isinstance(repo_tac, Mapping):
        raise GenerativeTaskspaceCorrectionError("runtime repo-local import manifest is missing")
    if (
        repo_tac.get("module_count") != 0
        or repo_tac.get("file_count") != 0
        or repo_tac.get("files") != []
        or repo_tac.get("unresolved_modules") != []
        or repo_tac.get("parse_errors") != []
    ):
        raise GenerativeTaskspaceCorrectionError("runtime is not standalone inside the reopened custody root")
    evaluator_snapshot = _stable_file_snapshot(
        evaluator_path,
        label="upstream evaluate.py",
        capture_payload=False,
    )
    evaluator_sha = evaluator_snapshot.sha256
    evaluator_row = manifest.get("upstream_evaluate_py")
    if not isinstance(evaluator_row, Mapping):
        raise GenerativeTaskspaceCorrectionError("runtime manifest omits upstream evaluate.py")
    if (
        evaluator_row.get("relative_path") != "evaluate.py"
        or evaluator_row.get("bytes") != evaluator_snapshot.size
        or evaluator_row.get("sha256") != evaluator_sha
    ):
        raise GenerativeTaskspaceCorrectionError("upstream evaluator bytes differ from the runtime manifest")
    recorded_root = manifest.get("runtime_root")
    if not isinstance(recorded_root, str) or Path(recorded_root).name != runtime_root.name:
        raise GenerativeTaskspaceCorrectionError("runtime root identity differs from the evaluated manifest")
    tree_payload = {
        "runtime_root_name": runtime_root.name,
        "files": list(files),
        "external_dependency_roots": [],
        "repo_local_tac_import_manifest": repo_tac,
        "upstream_evaluate_py": evaluator_row,
    }
    content_payload = {
        "files": [{key: row[key] for key in ("relative_path", "bytes", "sha256")} for row in files],
        "external_dependency_roots": [],
        "repo_local_tac_import_manifest": {key: value for key, value in repo_tac.items() if key != "runtime_root_name"},
        "upstream_evaluate_py": evaluator_row,
    }
    tree_sha = _sha256(json.dumps(tree_payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    content_sha = _sha256(json.dumps(content_payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    if manifest.get("runtime_tree_sha256") != tree_sha or manifest.get("runtime_content_tree_sha256") != content_sha:
        raise GenerativeTaskspaceCorrectionError("runtime aggregate identity does not recompose from reopened files")
    return tree_sha, content_sha, evaluator_sha


def _validate_raw_output_custody(
    provenance: Mapping[str, Any],
    *,
    manifest_path: Path,
    inflated_root: Path,
    video_names_path: Path,
) -> tuple[str, str, str]:
    outer = provenance.get("inflated_output_manifest")
    if not isinstance(outer, Mapping):
        raise GenerativeTaskspaceCorrectionError("exact evaluation lacks inflated-output custody")
    manifest_snapshot = _stable_file_snapshot(
        manifest_path,
        label="inflated outputs manifest",
        capture_payload=True,
    )
    assert manifest_snapshot.payload is not None
    manifest = _json_without_duplicate_keys(
        manifest_snapshot.payload,
        label="inflated outputs manifest",
    )
    manifest_sha = manifest_snapshot.sha256
    if outer.get("sha256") != manifest_sha or outer.get("payload") != manifest:
        raise GenerativeTaskspaceCorrectionError("inflated-output manifest differs from embedded provenance")
    if manifest.get("schema") != "contest_auth_eval_inflated_output_manifest_v1":
        raise GenerativeTaskspaceCorrectionError("inflated-output manifest schema is not canonical")
    video_names_snapshot = _stable_file_snapshot(
        video_names_path,
        label="video-name source",
        capture_payload=True,
    )
    assert video_names_snapshot.payload is not None
    try:
        video_names_text = video_names_snapshot.payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GenerativeTaskspaceCorrectionError("video-name population is unreadable") from exc
    video_names = tuple(line.strip() for line in video_names_text.splitlines() if line.strip())
    rows = manifest.get("files")
    if not isinstance(rows, list) or len(rows) != len(video_names) or manifest.get("raw_file_count") != len(rows):
        raise GenerativeTaskspaceCorrectionError("raw-output population differs from the video-name source")
    aggregate_rows: list[dict[str, Any]] = []
    total_bytes = 0
    for index, (row, video_name) in enumerate(zip(rows, video_names, strict=True)):
        if not isinstance(row, Mapping) or row.get("video_name") != video_name or row.get("exists") is not True:
            raise GenerativeTaskspaceCorrectionError(f"raw-output row {index} is incomplete or misaddressed")
        path = _safe_member_path(inflated_root, row.get("relative_path"), label=f"raw_output[{index}]")
        row_bytes = _exact_positive_int(row, "bytes", label=f"raw_output[{index}]")
        row_sha = _require_sha256(str(row.get("sha256")), f"raw_output[{index}].sha256")
        snapshot = _stable_file_snapshot(
            path,
            label=f"raw_output[{index}]",
            capture_payload=False,
        )
        if snapshot.size != row_bytes or snapshot.sha256 != row_sha:
            raise GenerativeTaskspaceCorrectionError(f"raw-output row {index} differs from reopened bytes")
        total_bytes += row_bytes
        aggregate_rows.append({"relative_path": row["relative_path"], "bytes": row_bytes, "sha256": row_sha})
    aggregate_sha = _sha256(
        json.dumps({"files": aggregate_rows}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    if manifest.get("aggregate_sha256") != aggregate_sha or manifest.get("total_bytes") != total_bytes:
        raise GenerativeTaskspaceCorrectionError("raw-output aggregate identity does not recompose")
    return aggregate_sha, manifest_sha, video_names_snapshot.sha256


def _derive_authority_axis(provenance: Mapping[str, Any], report_device: str) -> Literal["contest_cpu", "contest_cuda"]:
    device = provenance.get("device")
    if device != report_device:
        raise GenerativeTaskspaceCorrectionError("report and provenance evaluate different devices")
    system = provenance.get("platform_system")
    machine = str(provenance.get("platform_machine") or "").lower()
    if system != "Linux" or machine not in {"x86_64", "amd64"}:
        raise GenerativeTaskspaceCorrectionError("exact admission requires contest Linux x86_64 custody")
    if device == "cpu":
        return "contest_cpu"
    gpu_model = str(provenance.get("gpu_model") or "").lower()
    cuda_version = str(provenance.get("cuda_version") or "").strip()
    if (
        device != "cuda"
        or provenance.get("cuda_available") is not True
        or not cuda_version
        or not any(token in gpu_model for token in ("nvidia", "tesla", "geforce", "a100", "h100", "a10", "l4", "t4"))
    ):
        raise GenerativeTaskspaceCorrectionError("CUDA exact admission lacks observed contest GPU custody")
    return "contest_cuda"


def _command_flag(argv: Sequence[str], flag: str) -> str | None:
    for index, token in enumerate(argv[:-1]):
        if token == flag:
            return argv[index + 1]
    prefix = f"{flag}="
    for token in argv:
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def _validate_upstream_snapshot_custody(
    provenance: Mapping[str, Any],
    *,
    custody_root: Path,
    evaluator_path: Path,
    video_names_path: Path,
    argv: Sequence[str],
) -> str:
    recorded_root = provenance.get("upstream_dir")
    if not isinstance(recorded_root, str) or not recorded_root.strip():
        raise GenerativeTaskspaceCorrectionError("exact provenance omits the full upstream custody root")
    upstream_root = _resolve_custody_path(
        custody_root,
        Path(recorded_root),
        label="full upstream snapshot root",
        directory=True,
    )
    try:
        canonical_evaluator = (upstream_root / "evaluate.py").resolve(strict=True)
    except OSError as exc:
        raise GenerativeTaskspaceCorrectionError("full upstream snapshot lacks canonical evaluate.py") from exc
    if evaluator_path != canonical_evaluator:
        raise GenerativeTaskspaceCorrectionError("evaluator path escaped the full upstream snapshot root")
    recorded_video_names = provenance.get("video_names_file")
    if not isinstance(recorded_video_names, str):
        raise GenerativeTaskspaceCorrectionError("exact provenance omits its video-name source")
    if (
        _resolve_custody_path(
            custody_root,
            Path(recorded_video_names),
            label="recorded video-name source",
        )
        != video_names_path
    ):
        raise GenerativeTaskspaceCorrectionError("video-name source differs from exact provenance")
    argv_upstream = _command_flag(argv, "--upstream-dir")
    if not isinstance(argv_upstream, str) or (
        _resolve_custody_path(
            custody_root,
            Path(argv_upstream),
            label="argv upstream snapshot root",
            directory=True,
        )
        != upstream_root
    ):
        raise GenerativeTaskspaceCorrectionError("governed evaluator argv names a different upstream snapshot")

    recorded_sha = provenance.get("upstream_snapshot_sha256")
    if not isinstance(recorded_sha, str):
        raise GenerativeTaskspaceCorrectionError(
            "exact producer lacks canonical upstream_snapshot_sha256; admission fails closed"
        )
    _require_sha256(recorded_sha, "provenance.upstream_snapshot_sha256")
    try:
        first_sha = compute_upstream_snapshot_sha256(
            upstream_root,
            upstream_subdir=".",
            reject_executable_artifacts=True,
        )
        second_sha = compute_upstream_snapshot_sha256(
            upstream_root,
            upstream_subdir=".",
            reject_executable_artifacts=True,
        )
    except (OSError, ValueError) as exc:
        raise GenerativeTaskspaceCorrectionError("full upstream snapshot could not be recomputed") from exc
    if first_sha is None or first_sha != second_sha:
        raise GenerativeTaskspaceCorrectionError("full upstream snapshot changed during canonical recomputation")
    if first_sha != recorded_sha:
        raise GenerativeTaskspaceCorrectionError("recorded upstream snapshot differs from canonical full-tree bytes")
    return recorded_sha


def _reopen_exact_eval(
    paths: ExactEvalCustodyPathsV1,
    *,
    required_literal_g_member_path: str | None = None,
    required_literal_g_packet: bytes | None = None,
) -> _ReopenedExactEvalV1:
    try:
        root = paths.custody_root.resolve(strict=True)
    except OSError as exc:
        raise GenerativeTaskspaceCorrectionError("exact-eval custody root is missing") from exc
    if not root.is_dir():
        raise GenerativeTaskspaceCorrectionError("exact-eval custody root is not a directory")
    result_path = _resolve_custody_path(root, paths.result_json_path, label="exact result JSON")
    archive_path = _resolve_custody_path(root, paths.archive_path, label="exact archive")
    provenance_path = _resolve_custody_path(root, paths.provenance_json_path, label="exact provenance JSON")
    report_path = _resolve_custody_path(root, paths.report_path, label="upstream evaluator report")
    runtime_root = _resolve_custody_path(root, paths.runtime_root, label="inflate runtime root", directory=True)
    inflated_root = _resolve_custody_path(root, paths.inflated_root, label="inflated raw root", directory=True)
    manifest_path = _resolve_custody_path(
        root,
        paths.inflated_outputs_manifest_path,
        label="inflated outputs manifest",
    )
    evaluator_path = _resolve_custody_path(root, paths.upstream_evaluate_path, label="upstream evaluate.py")
    video_names_path = _resolve_custody_path(root, paths.video_names_path, label="video-name source")

    result_snapshot = _stable_file_snapshot(
        result_path,
        label="exact result JSON",
        capture_payload=True,
    )
    provenance_snapshot = _stable_file_snapshot(
        provenance_path,
        label="exact provenance JSON",
        capture_payload=True,
    )
    report_snapshot = _stable_file_snapshot(
        report_path,
        label="upstream evaluator report",
        capture_payload=True,
    )
    archive_snapshot = _stable_archive_snapshot(
        archive_path,
        label="exact archive",
        required_member_path=required_literal_g_member_path,
        required_member_payload=required_literal_g_packet,
    )
    assert result_snapshot.payload is not None
    assert provenance_snapshot.payload is not None
    assert report_snapshot.payload is not None
    result = _json_without_duplicate_keys(result_snapshot.payload, label="exact result JSON")
    provenance = _json_without_duplicate_keys(provenance_snapshot.payload, label="exact provenance JSON")
    if result.get("schema_version") != 1 or result.get("provenance") != provenance:
        raise GenerativeTaskspaceCorrectionError("exact result does not embed the reopened provenance artifact")
    if provenance.get("schema_version") != 1 or provenance.get("tool") != _EXACT_EVAL_TOOL:
        raise GenerativeTaskspaceCorrectionError("exact provenance was not emitted by contest_auth_eval")
    if provenance.get("modal_auth_eval_advisory_only") is True or result.get("diagnostic_blockers"):
        raise GenerativeTaskspaceCorrectionError("diagnostic/advisory exact-eval evidence cannot admit")

    n_samples, d_seg, d_pose, report_archive_bytes, report_device = _parse_exact_report(report_snapshot.payload)
    archive_sha = archive_snapshot.sha256
    archive_bytes = archive_snapshot.size
    if archive_bytes <= 0 or report_archive_bytes != archive_bytes:
        raise GenerativeTaskspaceCorrectionError("report archive bytes differ from the reopened archive")
    if provenance.get("archive_sha256") != archive_sha or provenance.get("archive_size_bytes") != archive_bytes:
        raise GenerativeTaskspaceCorrectionError("provenance archive identity differs from reopened bytes")
    if result.get("archive_size_bytes") != archive_bytes or result.get("n_samples") != n_samples:
        raise GenerativeTaskspaceCorrectionError("result archive/sample cardinality differs from reopened evidence")
    if (
        _exact_nonnegative_float(result, "avg_segnet_dist", label="result") != d_seg
        or _exact_nonnegative_float(result, "avg_posenet_dist", label="result") != d_pose
    ):
        raise GenerativeTaskspaceCorrectionError("result distortion coordinates differ from upstream report")
    expected_score = contest_score(d_seg, d_pose, archive_bytes)
    for key in ("canonical_score", "score_recomputed_from_components"):
        if not math.isclose(
            _exact_nonnegative_float(result, key, label="result"),
            expected_score,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise GenerativeTaskspaceCorrectionError("result score does not recompose from report and archive bytes")
    expected_rate = archive_bytes / CONTEST_REFERENCE_BYTES
    if not math.isclose(
        _exact_nonnegative_float(result, "rate_unscaled", label="result"),
        expected_rate,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise GenerativeTaskspaceCorrectionError("result rate differs from exact reopened archive bytes")
    if result.get("original_uncompressed_size_bytes") != CONTEST_REFERENCE_BYTES:
        raise GenerativeTaskspaceCorrectionError("result uses a noncanonical contest reference cardinality")

    axis = _derive_authority_axis(provenance, report_device)
    if result.get("score_axis") != axis:
        raise GenerativeTaskspaceCorrectionError("result axis differs from independently derived hardware axis")
    upstream_commit_raw = str(provenance.get("upstream_commit") or "").lower()
    upstream_commit = (
        upstream_commit_raw
        if len(upstream_commit_raw) in {40, 64}
        and all(character in "0123456789abcdef" for character in upstream_commit_raw)
        else None
    )
    argv = provenance.get("sys_argv")
    if not isinstance(argv, list) or any(not isinstance(token, str) for token in argv):
        raise GenerativeTaskspaceCorrectionError("exact provenance argv is not a literal string vector")
    if (
        not argv
        or not argv[0].endswith("contest_auth_eval.py")
        or _command_flag(argv, "--device") != report_device
        or _command_flag(argv, "--archive") is None
        or _command_flag(argv, "--inflate-sh") is None
        or _command_flag(argv, "--upstream-dir") is None
        or _command_flag(argv, "--work-dir") is None
        or "--keep-work-dir" not in argv
    ):
        raise GenerativeTaskspaceCorrectionError("exact provenance argv does not close the governed evaluator path")
    upstream_snapshot_sha = _validate_upstream_snapshot_custody(
        provenance,
        custody_root=root,
        evaluator_path=evaluator_path,
        video_names_path=video_names_path,
        argv=argv,
    )
    runtime_tree_sha, runtime_content_sha, evaluator_sha = _validate_runtime_custody(
        provenance,
        runtime_root=runtime_root,
        evaluator_path=evaluator_path,
    )
    raw_aggregate_sha, inflated_manifest_sha, video_names_sha = _validate_raw_output_custody(
        provenance,
        manifest_path=manifest_path,
        inflated_root=inflated_root,
        video_names_path=video_names_path,
    )
    hardware = (
        "Linux x86_64 CPU" if axis == "contest_cpu" else f"Linux x86_64 {provenance.get('gpu_model') or 'CUDA GPU'}"
    )
    canonical_validation = validate_exact_eval_evidence(
        {
            "axis": axis,
            "archive_sha256": archive_sha,
            "runtime_tree_sha256": runtime_tree_sha,
            "artifact_sha256": result_snapshot.sha256,
            "n_samples": n_samples,
            "archive_bytes": archive_bytes,
            "seg_dist": d_seg,
            "pose_dist": d_pose,
            "score": expected_score,
            "hardware": hardware,
            "auth_eval_command": shlex.join(argv),
            "log_path": str(report_path),
            "artifact_path": str(result_path),
            "raw_output_aggregate_sha256": raw_aggregate_sha,
            "inflated_outputs_manifest_path": str(manifest_path),
            "inflated_outputs_manifest_sha256": inflated_manifest_sha,
        },
        expected_axis=axis,
        expected_archive_sha256=archive_sha,
        expected_runtime_tree_sha256=runtime_tree_sha,
        require_artifact_path=True,
        require_hardware=True,
        require_auth_eval_command=True,
        require_log_path=True,
        require_artifact_sha256=True,
        require_inflated_outputs_manifest=True,
        require_raw_output_aggregate_sha256=True,
    )
    if canonical_validation.blockers:
        raise GenerativeTaskspaceCorrectionError(
            "canonical exact-eval custody validator refused: " + ",".join(canonical_validation.blockers)
        )
    source_object_binding_sha = _sha256(
        _canonical_json(
            {
                "schema": "tac.generative_taskspace_exact_source_object_binding.v1",
                "upstream_evaluate_sha256": evaluator_sha,
                "video_names_sha256": video_names_sha,
                "upstream_snapshot_sha256": upstream_snapshot_sha,
                "original_uncompressed_size_bytes": CONTEST_REFERENCE_BYTES,
                "upstream_commit_when_recorded": upstream_commit,
            }
        )
    )
    return _ReopenedExactEvalV1(
        result_artifact_sha256=result_snapshot.sha256,
        provenance_artifact_sha256=provenance_snapshot.sha256,
        report_sha256=report_snapshot.sha256,
        archive_sha256=archive_sha,
        archive_bytes=archive_bytes,
        runtime_tree_sha256=runtime_tree_sha,
        runtime_content_tree_sha256=runtime_content_sha,
        evaluator_sha256=evaluator_sha,
        raw_output_aggregate_sha256=raw_aggregate_sha,
        inflated_outputs_manifest_sha256=inflated_manifest_sha,
        video_names_sha256=video_names_sha,
        upstream_snapshot_sha256=upstream_snapshot_sha,
        upstream_commit=upstream_commit,
        source_object_binding_sha256=source_object_binding_sha,
        authority_axis=axis,
        n_samples=600,
        d_seg=d_seg,
        d_pose=d_pose,
        joint_score=expected_score,
        literal_g_archive_member_path=archive_snapshot.literal_g_archive_member_path,
    )


def _timestamp_is_stale(value: object, *, now: datetime | None = None) -> bool:
    if not isinstance(value, str) or not value:
        return True
    try:
        observed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return True
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=UTC)
    current = now or datetime.now(UTC)
    age_seconds = (current - observed).total_seconds()
    return age_seconds < -300 or age_seconds > POINTER_STALE_SECONDS


def _load_live_competitive_frontier(repo_root: Path) -> tuple[float, str, str]:
    pointer_path = repo_root / CANONICAL_FRONTIER_POINTER_PATH
    snapshot = _stable_file_snapshot(
        pointer_path,
        label="canonical frontier pointer",
        capture_payload=True,
    )
    assert snapshot.payload is not None
    payload = _json_without_duplicate_keys(snapshot.payload, label="canonical frontier pointer")
    try:
        pointer = CanonicalFrontierPointer.from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise GenerativeTaskspaceCorrectionError("canonical frontier pointer constituents are invalid") from exc
    if pointer.schema_version != POINTER_SCHEMA_VERSION or pointer.is_stale():
        raise GenerativeTaskspaceCorrectionError("canonical frontier pointer is stale or has the wrong schema")
    recomputed = recompute_effective_frontier(pointer)
    serialized = payload.get("effective_frontier")
    if not isinstance(recomputed, Mapping) or not isinstance(serialized, Mapping):
        raise GenerativeTaskspaceCorrectionError("canonical pointer lacks recomposable frontier constituents")
    if _canonical_json({"effective_frontier": serialized}) != _canonical_json({"effective_frontier": recomputed}):
        raise GenerativeTaskspaceCorrectionError(
            "serialized effective frontier differs from its canonical constituents"
        )
    if _timestamp_is_stale(pointer.upstream_leaderboard_snapshot_at_utc):
        raise GenerativeTaskspaceCorrectionError(
            "official leaderboard snapshot is stale; refresh pointer before exact G admission"
        )
    if recomputed.get("source") == "upstream_official_leaderboard" and (
        recomputed.get("snapshot_at_utc") != pointer.upstream_leaderboard_snapshot_at_utc
    ):
        raise GenerativeTaskspaceCorrectionError("effective frontier winning snapshot timestamp differs")
    score_raw = recomputed.get("score")
    axis = recomputed.get("axis")
    if (
        isinstance(score_raw, bool)
        or not isinstance(score_raw, int | float)
        or not math.isfinite(float(score_raw))
        or float(score_raw) <= 0.0
        or not isinstance(axis, str)
        or not axis
    ):
        raise GenerativeTaskspaceCorrectionError("recomputed competitive frontier is invalid")
    return float(score_raw), axis, snapshot.sha256


def _validate_compiled_for_exact_admission(
    compiled: CompiledGenerativeCorrectionV1,
    predictor_state: PredictorSemanticStateV1,
    teacher_evidence: EncoderOnlyTeacherEvidenceV1,
) -> None:
    if type(compiled) is not CompiledGenerativeCorrectionV1:
        raise GenerativeTaskspaceCorrectionError("compiled correction must use the exact canonical type")
    if type(predictor_state) is not PredictorSemanticStateV1:
        raise GenerativeTaskspaceCorrectionError("predictor state must use the exact canonical type")
    if type(teacher_evidence) is not EncoderOnlyTeacherEvidenceV1:
        raise GenerativeTaskspaceCorrectionError("teacher evidence must use the exact canonical type")
    if (
        type(compiled.packet) is not bytes
        or type(compiled.receipt) is not GenerativeCorrectionCompileReceiptV1
        or type(compiled.decoded) is not DecodedGenerativeCorrectionV1
    ):
        raise GenerativeTaskspaceCorrectionError("compiled correction contains a noncanonical component type")
    if compiled.receipt_binding_sha256 != compiled.receipt.binding_sha256:
        raise GenerativeTaskspaceCorrectionError("compiled correction receipt changed after compilation")
    parsed = parse_generative_taskspace_correction(compiled.packet, predictor_state=predictor_state)
    recomputed = compile_generative_taskspace_correction(
        predictor_state,
        parsed.program,
        teacher_evidence=teacher_evidence,
    )
    if compiled.packet != recomputed.packet:
        raise GenerativeTaskspaceCorrectionError("compiled correction packet differs from semantic recompilation")
    if compiled.receipt != recomputed.receipt or compiled.receipt_binding_sha256 != recomputed.receipt_binding_sha256:
        raise GenerativeTaskspaceCorrectionError(
            "compiled correction receipt differs from complete semantic recomputation"
        )
    if not np.array_equal(recomputed.decoded.labels, compiled.decoded.labels) or (
        recomputed.decoded.realization_profile != compiled.decoded.realization_profile
        or recomputed.decoded.correction_packet_sha256 != compiled.decoded.correction_packet_sha256
    ):
        raise GenerativeTaskspaceCorrectionError("compiled correction output differs from semantic recompilation")


def admit_by_exact_coupled_score(
    compiled: CompiledGenerativeCorrectionV1,
    predictor_state: PredictorSemanticStateV1,
    *,
    teacher_evidence: EncoderOnlyTeacherEvidenceV1,
    baseline_custody: ExactEvalCustodyPathsV1,
    candidate_custody: ExactEvalCustodyPathsV1,
    candidate_g_archive_member_path: str,
    repo_root: Path,
) -> ValuePerByteAdmissionV1:
    """Fail closed until exact G consumption, not mere presence, is proved.

    Reopening a candidate ZIP member equal to ``compiled.packet`` proves byte
    presence only.  Exact admission additionally requires a governed runtime
    receipt joining strict G parse/apply to decoded state and evaluated raw
    outputs, plus a matched G-only counterfactual.  That producer contract is
    not landed yet, so this function deliberately cannot emit authority.
    """

    if not isinstance(repo_root, Path):
        raise GenerativeTaskspaceCorrectionError("repo_root must be an explicit pathlib.Path")
    _validate_compiled_for_exact_admission(compiled, predictor_state, teacher_evidence)
    baseline = _reopen_exact_eval(baseline_custody)
    candidate = _reopen_exact_eval(
        candidate_custody,
        required_literal_g_member_path=candidate_g_archive_member_path,
        required_literal_g_packet=compiled.packet,
    )
    if baseline.archive_sha256 == candidate.archive_sha256:
        raise GenerativeTaskspaceCorrectionError("baseline and candidate exact archives must differ")
    if (
        baseline.authority_axis != candidate.authority_axis
        or baseline.source_object_binding_sha256 != candidate.source_object_binding_sha256
    ):
        raise GenerativeTaskspaceCorrectionError("baseline and candidate are not exact same-object observations")
    if candidate.literal_g_archive_member_path != candidate_g_archive_member_path:
        raise GenerativeTaskspaceCorrectionError("candidate exact archive did not prove its literal G member")

    # Keep live-pointer validation in the pre-admission path so a stale or
    # incoherent competitive target cannot hide behind the receiver blocker.
    _load_live_competitive_frontier(repo_root.resolve(strict=True))
    raise GenerativeTaskspaceCorrectionError(
        f"{RECEIVER_CONSUMPTION_CUSTODY_ABSENT}: literal G archive-member equality proves presence, not "
        "archive-member -> strict-G-decode -> evaluated-raw-output causality; require a provenance-bound "
        "receiver parse/apply receipt and matched G-only counterfactual"
    )


__all__ = [
    "DECODER_PAYLOAD_POLICY",
    "ENCODER_ONLY_LINEAGE_POLICY",
    "EXACT_JOINT_AUTHORITY",
    "JOINT_OBJECTIVE_ID",
    "PACKET_SCHEMA",
    "PRIMITIVE_LINEAGE_POLICY",
    "RECEIVER_CONSUMPTION_CUSTODY_ABSENT",
    "CompiledGenerativeCorrectionV1",
    "CorrectionResourceCountsV1",
    "DecodedGenerativeCorrectionV1",
    "EncoderOnlyTeacherEvidenceV1",
    "ExactEvalCustodyPathsV1",
    "GenerativeCorrectionCompileReceiptV1",
    "GenerativeCorrectionProgramV1",
    "GenerativeTaskspaceCorrectionError",
    "ParsedGenerativeCorrectionV1",
    "PredictorSemanticStateV1",
    "ValuePerByteAdmissionV1",
    "admit_by_exact_coupled_score",
    "apply_generative_taskspace_correction",
    "compile_generative_taskspace_correction",
    "parse_generative_taskspace_correction",
]
