# SPDX-License-Identifier: MIT
"""Encoder-only acquisition for counted predictor-preserving A3 rows.

The counted receiver primitive already exists in
``predictor_preserving_coupled_preimage``.  This module supplies the missing
bounded encoder seam: given exact ephemeral ``P0/P1``, exact counted ``G``, the
exact corrected camera ``Y1``, and an encoder-only target camera ``Y0``, it
builds finite prefixes of two currently executable A3 interpretations:

* constant-RGB scorer-cell supports chosen against exact integer target
  numerators; and
* coordinate-only copies from the already-bound corrected ``Y1`` when that
  exact numerator is closer to the target than the predictor ``P0``.

Every returned packet is compiled through the public A3 compiler, strictly
parsed twice, and publicly decoded twice (each public decode itself performs
the receiver's deterministic double replay).  The allocator transforms retain
only exact P/G hashes and counted A bytes; dense target RGB, target Pose,
scorer weights, and scorer observations are never captured by them.

This is an acquisition/control surface, not an optimizer or score.  Exact
integer numerator error reduction ranks rows but never admits them.  Admission
belongs to the whole-object nonlinear allocator after complete archive rebuild,
receiver replay, and realized measurement.

The module also exposes both preregistered XIP2 warp-domain interpretations as
encoder-only guidance targets.  They deliberately do *not* pretend that the
current A3 wire grammar ships XIP2: the current packet remains one of the two
sparse row modes above, while a true counted XIP2 A3 receiver mode (including
its pitch/geometry ABI) remains an explicit blocker.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from functools import lru_cache, partial
from typing import Any, Final, Literal

import numpy as np

from tac.boundary_math.warp_real_luma_frame0 import (
    GroundHomographyGeom,
    WarpRealLumaFrame0Error,
    warp_frame0_native_numpy,
    warp_frame0_uint8_numpy,
)
from tac.boundary_math.xi_pose_coder import (
    XiPoseCoderError,
    quantize_xi,
    serialize_xi_payload,
)
from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator, Uint8LatticeError
from tac.witness_dsl.generative_taskspace_correction import DecodedGenerativeCorrectionV1
from tac.witness_dsl.predictor_preserving_coupled_preimage import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CHANNELS,
    PACKET_HEADER,
    SCORER_HEIGHT,
    SCORER_WIDTH,
    CompiledPredictorPreservingA3V1,
    CorrectedY1SupportCopyCellV1,
    PredictorCameraPairSurfaceV1,
    PredictorPreservingA3Error,
    PredictorPreservingA3Mode,
    PredictorPreservingA3ProgramV1,
    PredictorPreservingA3SourceBindingV1,
    SparseConstantRGBCellV1,
    compile_predictor_preserving_a3,
    decode_predictor_preserving_a3_packet,
    derive_predictor_preserving_a3_source_binding,
    parse_predictor_preserving_a3_packet,
)
from tac.witness_dsl.predictor_preserving_taskspace_overlay import PredictorPreservingOverlayResultV1
from tac.witness_dsl.taskspace_predictor_state_v2 import (
    SE3XiTransportV2,
    TaskspacePredictorStateV2Error,
)
from tac.witness_dsl.taskspace_whole_archive_allocator import (
    TaskspaceSectionBundleV1,
    TaskspaceWholeArchiveProposalV1,
)

RECEIPT_SCHEMA: Final = "tac.taskspace_chronological_a3_encoder_receipt.v1"
ACQUISITION_ID: Final = "exact_integer_numerator_prefixes_over_predictor_preserving_a3.v1"
_TARGET_BINDING_SCHEMA: Final = "tac.encoder_only_a3_target_camera.v1"
_XIP2_GUIDANCE_SCHEMA: Final = "tac.encoder_only_a3_xip2_guidance.v1"
_PROPOSAL_PREFIX_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,31}\Z")


class ChronologicalA3EncoderError(ValueError):
    """Target custody, acquisition, A3 compile, or allocator substitution failed."""


class ChronologicalA3Interpretation(StrEnum):
    """The two executable sparse-row acquisition interpretations."""

    TARGET_CONSTANT_RGB_V1 = "TARGET_CONSTANT_RGB_V1"
    CORRECTED_Y1_SUPPORT_COPY_V1 = "CORRECTED_Y1_SUPPORT_COPY_V1"


class XIP2WarpDomain(StrEnum):
    """The two parent-spec warp-domain interpretations kept alive for probing."""

    CAMERA_THEN_R = "CAMERA_THEN_R"
    SCORER_THEN_FACTOR2 = "SCORER_THEN_FACTOR2"


class XIP2Coder(StrEnum):
    """Closed public coder names accepted by ``xi_pose_coder``."""

    NONE = "none"
    DELTA_AR = "delta_ar"
    SPLINE_RESIDUAL = "spline_residual"
    DELTA_RES = "delta_res"


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return _sha256(memoryview(np.ascontiguousarray(value)).cast("B"))


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
        raise ChronologicalA3EncoderError("A3 encoder record is not finite canonical ASCII JSON") from exc


def _require_sha256(value: object, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ChronologicalA3EncoderError(f"{field} must be canonical lowercase SHA-256")
    return value


def _require_exact_nonnegative_int(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ChronologicalA3EncoderError(f"{field} must be an exact nonnegative integer")
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
    except Uint8LatticeError as exc:  # pragma: no cover - frozen geometry
        raise ChronologicalA3EncoderError("frozen camera/scorer operator is unavailable") from exc


def _numerator_stack(frames: np.ndarray) -> tuple[np.ndarray, int]:
    array = np.asarray(frames)
    if array.dtype != np.uint8 or array.ndim != 4 or array.shape[1:] != (
        CAMERA_HEIGHT,
        CAMERA_WIDTH,
        CHANNELS,
    ):
        raise ChronologicalA3EncoderError("camera stack must be uint8 [pair,874,1164,3]")
    numerators: list[np.ndarray] = []
    denominator: int | None = None
    for frame in array:
        try:
            current, current_denominator = _operator().apply_numerators(frame)
        except Uint8LatticeError as exc:
            raise ChronologicalA3EncoderError("exact scorer numerator application failed") from exc
        if denominator is not None and current_denominator != denominator:
            raise ChronologicalA3EncoderError("scorer denominator changed across pair rows")
        denominator = current_denominator
        numerators.append(np.ascontiguousarray(current, dtype=np.int64))
    if denominator is None:  # pragma: no cover - every public source is nonempty
        raise ChronologicalA3EncoderError("empty camera stack has no scorer denominator")
    return np.stack(numerators, axis=0), denominator


@dataclass(frozen=True, slots=True)
class EncoderOnlyA3TargetV1:
    """Dense target camera evidence that intentionally has no serializer.

    ``custody_sha256`` identifies the frozen source object or the deterministic
    encoder-only guidance derivation.  Only ``binding_sha256`` may enter an
    acquisition receipt; neither this array nor its raw bytes enter an A packet
    or allocator transform.
    """

    source_pair_ids: tuple[int, ...]
    target_camera_y0: np.ndarray
    custody_sha256: str
    evidence_kind: str = "frozen_target_camera_y0_encoder_only.v1"

    def __post_init__(self) -> None:
        if type(self.source_pair_ids) is not tuple or not self.source_pair_ids:
            raise ChronologicalA3EncoderError("target source_pair_ids must be a nonempty exact tuple")
        if any(type(pair_id) is not int for pair_id in self.source_pair_ids):
            raise ChronologicalA3EncoderError("target source_pair_ids must contain exact integers")
        expected_ids = tuple(
            range(self.source_pair_ids[0], self.source_pair_ids[0] + len(self.source_pair_ids))
        )
        if self.source_pair_ids != expected_ids or not 0 <= expected_ids[0] < expected_ids[-1] + 1 <= 600:
            raise ChronologicalA3EncoderError("target source_pair_ids must be contiguous inside [0,600)")
        _require_sha256(self.custody_sha256, "target custody_sha256")
        if (
            type(self.evidence_kind) is not str
            or not self.evidence_kind
            or len(self.evidence_kind) > 128
            or not self.evidence_kind.isascii()
        ):
            raise ChronologicalA3EncoderError("target evidence_kind must be bounded nonempty ASCII")
        target = np.asarray(self.target_camera_y0)
        expected_shape = (
            len(self.source_pair_ids),
            CAMERA_HEIGHT,
            CAMERA_WIDTH,
            CHANNELS,
        )
        if target.dtype != np.uint8 or target.shape != expected_shape:
            raise ChronologicalA3EncoderError("target Y0 must be uint8 [pair,874,1164,3]")
        object.__setattr__(self, "target_camera_y0", _immutable_array(target, dtype=np.dtype(np.uint8)))

    @property
    def target_camera_y0_sha256(self) -> str:
        return _array_sha256(self.target_camera_y0)

    @property
    def binding_sha256(self) -> str:
        return _sha256(
            _canonical_json(
                {
                    "schema": _TARGET_BINDING_SCHEMA,
                    "source_pair_ids": list(self.source_pair_ids),
                    "target_camera_y0_sha256": self.target_camera_y0_sha256,
                    "custody_sha256": self.custody_sha256,
                    "evidence_kind": self.evidence_kind,
                    "serialized_into_a3_packet": False,
                }
            )
        )


@dataclass(frozen=True, slots=True)
class A3PrefixPlanV1:
    """Caller-chosen finite prefix sizes for one interpretation.

    The encoder intentionally does not invent a universal row budget.  Prefix
    sizes are experiment coordinates; the whole-object allocator arbitrates
    them under the actual nonlinear score and archive bytes.
    """

    interpretation: ChronologicalA3Interpretation
    row_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.interpretation) is not ChronologicalA3Interpretation:
            raise ChronologicalA3EncoderError("prefix interpretation escaped the closed universe")
        if type(self.row_counts) is not tuple or not self.row_counts:
            raise ChronologicalA3EncoderError("prefix row_counts must be a nonempty exact tuple")
        if any(type(value) is not int or value < 1 for value in self.row_counts):
            raise ChronologicalA3EncoderError("prefix row_counts must contain positive exact integers")
        if self.row_counts != tuple(sorted(set(self.row_counts))):
            raise ChronologicalA3EncoderError("prefix row_counts must be sorted and unique")
        maximum_cells = 4 * SCORER_HEIGHT * SCORER_WIDTH
        if self.row_counts[-1] > maximum_cells:
            raise ChronologicalA3EncoderError("prefix row count exceeds the bounded four-pair cell universe")


@dataclass(frozen=True, slots=True, order=True)
class RankedA3CellV1:
    """One exact positive numerator-SSE acquisition row."""

    rank: int
    source_pair_id: int
    scorer_row: int
    scorer_col: int
    numerator_sse_before: int
    numerator_sse_after: int
    numerator_sse_reduction: int
    constant_rgb_u8: tuple[int, int, int] | None

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank < 0:
            raise ChronologicalA3EncoderError("ranked A3 cell rank must be an exact nonnegative integer")
        if type(self.source_pair_id) is not int or not 0 <= self.source_pair_id < 600:
            raise ChronologicalA3EncoderError("ranked A3 cell pair ID escaped [0,600)")
        if type(self.scorer_row) is not int or not 0 <= self.scorer_row < SCORER_HEIGHT:
            raise ChronologicalA3EncoderError("ranked A3 cell row escaped scorer geometry")
        if type(self.scorer_col) is not int or not 0 <= self.scorer_col < SCORER_WIDTH:
            raise ChronologicalA3EncoderError("ranked A3 cell column escaped scorer geometry")
        for field in ("numerator_sse_before", "numerator_sse_after", "numerator_sse_reduction"):
            _require_exact_nonnegative_int(getattr(self, field), field)
        if (
            self.numerator_sse_reduction < 1
            or self.numerator_sse_before - self.numerator_sse_after != self.numerator_sse_reduction
        ):
            raise ChronologicalA3EncoderError("ranked A3 cell integer SSE arithmetic does not close")
        if self.constant_rgb_u8 is not None:
            if type(self.constant_rgb_u8) is not tuple or len(self.constant_rgb_u8) != CHANNELS:
                raise ChronologicalA3EncoderError("constant RGB acquisition row must hold one exact triple")
            if any(type(value) is not int or not 0 <= value <= 255 for value in self.constant_rgb_u8):
                raise ChronologicalA3EncoderError("constant RGB acquisition value escaped uint8")


@dataclass(frozen=True, slots=True)
class ChronologicalA3EncoderReceiptV1:
    """Canonical encoder receipt; never part of the counted A packet."""

    schema: Literal["tac.taskspace_chronological_a3_encoder_receipt.v1"]
    acquisition_id: Literal["exact_integer_numerator_prefixes_over_predictor_preserving_a3.v1"]
    proposal_id: str
    interpretation: str
    packet_mode: str
    source_pair_ids: tuple[int, ...]
    source_binding_sha256: str
    predictor_program_sha256: str
    correction_packet_sha256: str
    corrected_camera_y1_sha256: str
    target_evidence_binding_sha256: str
    packet_sha256: str
    packet_bytes: int
    control_row_count: int
    available_positive_rows: int
    selected_numerator_sse_before: int
    selected_numerator_sse_after: int
    selected_numerator_sse_reduction: int
    serialized_coordinate_bytes: int
    serialized_rgb_value_bytes: int
    serialized_dense_target_camera_bytes: Literal[0] = 0
    serialized_dense_target_scorer_bytes: Literal[0] = 0
    serialized_target_pose_bytes: Literal[0] = 0
    serialized_scorer_evidence_bytes: Literal[0] = 0
    exact_p_g_corrected_y1_source_binding: Literal[True] = True
    packet_strict_parse_twice: Literal[True] = True
    packet_public_decode_twice: Literal[True] = True
    corrected_y1_preserved_exactly: Literal[True] = True
    nonselected_p0_camera_values_preserved: Literal[True] = True
    nonselected_p0_scorer_numerators_preserved: Literal[True] = True
    allocator_transform_captures_target_evidence: Literal[False] = False
    full_xip2_receiver_mode_implemented: Literal[False] = False
    xip2_payload_serialized_in_current_a3_packet: Literal[False] = False
    scorer_invoked: Literal[False] = False
    through_r_verified: Literal[False] = False
    n600_evidence_claim: Literal[False] = False
    exact_score_claim: Literal[False] = False
    rate_optimality_claim: Literal[False] = False
    candidate_claim: Literal[False] = False
    originality_claim: Literal[False] = False
    promotion_eligible: Literal[False] = False
    research_only: Literal[True] = True

    def __post_init__(self) -> None:
        if self.schema != RECEIPT_SCHEMA or self.acquisition_id != ACQUISITION_ID:
            raise ChronologicalA3EncoderError("A3 encoder receipt schema/algorithm changed")
        if (
            type(self.proposal_id) is not str
            or not self.proposal_id
            or len(self.proposal_id) > 128
            or not self.proposal_id.isascii()
        ):
            raise ChronologicalA3EncoderError("A3 encoder proposal_id must be bounded nonempty ASCII")
        try:
            interpretation = ChronologicalA3Interpretation(self.interpretation)
            packet_mode = PredictorPreservingA3Mode(self.packet_mode)
        except (TypeError, ValueError) as exc:
            raise ChronologicalA3EncoderError("A3 encoder receipt discriminant escaped its closed set") from exc
        expected_packet_mode = {
            ChronologicalA3Interpretation.TARGET_CONSTANT_RGB_V1: (
                PredictorPreservingA3Mode.SPARSE_CONSTANT_RGB_V1
            ),
            ChronologicalA3Interpretation.CORRECTED_Y1_SUPPORT_COPY_V1: (
                PredictorPreservingA3Mode.COPY_CORRECTED_Y1_SUPPORT_V1
            ),
        }[interpretation]
        if packet_mode is not expected_packet_mode:
            raise ChronologicalA3EncoderError("A3 acquisition interpretation differs from packet mode")
        if type(self.source_pair_ids) is not tuple or not self.source_pair_ids:
            raise ChronologicalA3EncoderError("A3 encoder receipt pair IDs must be a nonempty tuple")
        if any(type(pair_id) is not int for pair_id in self.source_pair_ids):
            raise ChronologicalA3EncoderError("A3 encoder receipt pair IDs must be exact integers")
        for field in (
            "source_binding_sha256",
            "predictor_program_sha256",
            "correction_packet_sha256",
            "corrected_camera_y1_sha256",
            "target_evidence_binding_sha256",
            "packet_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        for field in (
            "packet_bytes",
            "control_row_count",
            "available_positive_rows",
            "selected_numerator_sse_before",
            "selected_numerator_sse_after",
            "selected_numerator_sse_reduction",
            "serialized_coordinate_bytes",
            "serialized_rgb_value_bytes",
        ):
            _require_exact_nonnegative_int(getattr(self, field), field)
        if (
            self.control_row_count < 1
            or self.available_positive_rows < self.control_row_count
            or self.selected_numerator_sse_reduction < 1
            or self.selected_numerator_sse_before - self.selected_numerator_sse_after
            != self.selected_numerator_sse_reduction
            or self.serialized_coordinate_bytes != self.control_row_count * 6
        ):
            raise ChronologicalA3EncoderError("A3 encoder receipt row/SSE/coordinate arithmetic differs")
        expected_rgb_bytes = (
            self.control_row_count * CHANNELS
            if interpretation is ChronologicalA3Interpretation.TARGET_CONSTANT_RGB_V1
            else 0
        )
        if self.serialized_rgb_value_bytes != expected_rgb_bytes:
            raise ChronologicalA3EncoderError("A3 encoder receipt RGB byte accounting differs")
        if self.packet_bytes != PACKET_HEADER.size + self.serialized_coordinate_bytes + expected_rgb_bytes:
            raise ChronologicalA3EncoderError("A3 encoder receipt packet byte accounting differs")
        fixed_truth = {
            "serialized_dense_target_camera_bytes": 0,
            "serialized_dense_target_scorer_bytes": 0,
            "serialized_target_pose_bytes": 0,
            "serialized_scorer_evidence_bytes": 0,
            "exact_p_g_corrected_y1_source_binding": True,
            "packet_strict_parse_twice": True,
            "packet_public_decode_twice": True,
            "corrected_y1_preserved_exactly": True,
            "nonselected_p0_camera_values_preserved": True,
            "nonselected_p0_scorer_numerators_preserved": True,
            "allocator_transform_captures_target_evidence": False,
            "full_xip2_receiver_mode_implemented": False,
            "xip2_payload_serialized_in_current_a3_packet": False,
            "scorer_invoked": False,
            "through_r_verified": False,
            "n600_evidence_claim": False,
            "exact_score_claim": False,
            "rate_optimality_claim": False,
            "candidate_claim": False,
            "originality_claim": False,
            "promotion_eligible": False,
            "research_only": True,
        }
        if any(getattr(self, field) is not expected for field, expected in fixed_truth.items()):
            raise ChronologicalA3EncoderError("A3 encoder receipt authority/evidence truth was relaxed")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_pair_ids"] = list(self.source_pair_ids)
        return value

    def to_receipt_bytes(self) -> bytes:
        return _canonical_json(self.as_dict())

    @property
    def receipt_sha256(self) -> str:
        return _sha256(self.to_receipt_bytes())


def parse_chronological_a3_encoder_receipt(payload: bytes) -> ChronologicalA3EncoderReceiptV1:
    """Strict duplicate-key/type/canonical parse of one encoder receipt."""

    if type(payload) is not bytes or not payload:
        raise ChronologicalA3EncoderError("A3 encoder receipt must be nonempty exact bytes")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ChronologicalA3EncoderError(f"A3 encoder receipt repeats key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("ascii"), object_pairs_hook=unique_pairs)
    except ChronologicalA3EncoderError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ChronologicalA3EncoderError("A3 encoder receipt is not strict ASCII JSON") from exc
    if type(value) is not dict or set(value) != set(ChronologicalA3EncoderReceiptV1.__dataclass_fields__):
        raise ChronologicalA3EncoderError("A3 encoder receipt fields differ from the closed schema")
    if _canonical_json(value) != payload:
        raise ChronologicalA3EncoderError("A3 encoder receipt is not canonical JSON")
    if type(value["source_pair_ids"]) is not list or any(
        type(pair_id) is not int for pair_id in value["source_pair_ids"]
    ):
        raise ChronologicalA3EncoderError("A3 encoder receipt pair IDs must be exact JSON integers")
    arguments = dict(value)
    arguments["source_pair_ids"] = tuple(value["source_pair_ids"])
    try:
        receipt = ChronologicalA3EncoderReceiptV1(**arguments)
    except TypeError as exc:
        raise ChronologicalA3EncoderError("A3 encoder receipt contains invalid typed fields") from exc
    if receipt.to_receipt_bytes() != payload:
        raise ChronologicalA3EncoderError("A3 encoder receipt changed on typed parse/re-emit")
    return receipt


@dataclass(frozen=True, slots=True)
class CompiledChronologicalA3ProposalV1:
    """One exact counted A packet and its whole-object allocator transform."""

    interpretation: ChronologicalA3Interpretation
    ranked_prefix: tuple[RankedA3CellV1, ...]
    compiled_a3: CompiledPredictorPreservingA3V1
    receipt: ChronologicalA3EncoderReceiptV1
    allocator_proposal: TaskspaceWholeArchiveProposalV1

    def __post_init__(self) -> None:
        if type(self.interpretation) is not ChronologicalA3Interpretation:
            raise ChronologicalA3EncoderError("compiled proposal interpretation changed exact type")
        if type(self.ranked_prefix) is not tuple or not self.ranked_prefix:
            raise ChronologicalA3EncoderError("compiled proposal must retain one nonempty ranked prefix")
        if any(type(row) is not RankedA3CellV1 for row in self.ranked_prefix):
            raise ChronologicalA3EncoderError("compiled proposal ranked rows changed exact type")
        if type(self.compiled_a3) is not CompiledPredictorPreservingA3V1:
            raise ChronologicalA3EncoderError("compiled proposal lost its exact A3 object")
        if type(self.receipt) is not ChronologicalA3EncoderReceiptV1:
            raise ChronologicalA3EncoderError("compiled proposal lost its exact encoder receipt")
        if type(self.allocator_proposal) is not TaskspaceWholeArchiveProposalV1:
            raise ChronologicalA3EncoderError("compiled proposal lost its g7 allocator proposal")
        if (
            len(self.ranked_prefix) != self.receipt.control_row_count
            or self.compiled_a3.receipt.packet_sha256 != self.receipt.packet_sha256
            or self.allocator_proposal.proposal_id != self.receipt.proposal_id
        ):
            raise ChronologicalA3EncoderError("compiled proposal packet/receipt/allocator identity differs")


@dataclass(frozen=True, slots=True)
class ChronologicalA3AcquisitionV1:
    """Finite current-wire acquisition surface for one exact P/G/Y1 object."""

    source_binding: PredictorPreservingA3SourceBindingV1
    pass_p0_control: CompiledPredictorPreservingA3V1
    proposals: tuple[CompiledChronologicalA3ProposalV1, ...]

    def __post_init__(self) -> None:
        if type(self.source_binding) is not PredictorPreservingA3SourceBindingV1:
            raise ChronologicalA3EncoderError("A3 acquisition source binding changed exact type")
        if type(self.pass_p0_control) is not CompiledPredictorPreservingA3V1:
            raise ChronologicalA3EncoderError("A3 acquisition lost its PASS_P0 control")
        if self.pass_p0_control.program.mode is not PredictorPreservingA3Mode.PASS_P0_V1:
            raise ChronologicalA3EncoderError("A3 acquisition control is not exact PASS_P0")
        if self.pass_p0_control.source_binding != self.source_binding:
            raise ChronologicalA3EncoderError("A3 acquisition control source differs")
        if type(self.proposals) is not tuple or not self.proposals:
            raise ChronologicalA3EncoderError("A3 acquisition must expose at least one concrete proposal")
        if any(type(row) is not CompiledChronologicalA3ProposalV1 for row in self.proposals):
            raise ChronologicalA3EncoderError("A3 acquisition proposal exact types changed")
        proposal_ids = tuple(row.allocator_proposal.proposal_id for row in self.proposals)
        if len(set(proposal_ids)) != len(proposal_ids):
            raise ChronologicalA3EncoderError("A3 acquisition proposal IDs are not unique")
        if any(row.compiled_a3.source_binding != self.source_binding for row in self.proposals):
            raise ChronologicalA3EncoderError("A3 acquisition proposal source binding differs")

    @property
    def allocator_proposals(self) -> tuple[TaskspaceWholeArchiveProposalV1, ...]:
        return tuple(row.allocator_proposal for row in self.proposals)


@dataclass(frozen=True, slots=True)
class EncoderOnlyXIP2GuidanceV1:
    """Two XIP2-derived targets; neither is a current counted A3 wire mode."""

    source_binding_sha256: str
    source_pair_ids: tuple[int, ...]
    coder: XIP2Coder
    q_levels: int
    pitch: float
    xip2_payload: bytes
    camera_then_r: EncoderOnlyA3TargetV1
    scorer_then_factor2: EncoderOnlyA3TargetV1
    schema: Literal["tac.encoder_only_a3_xip2_guidance.v1"] = _XIP2_GUIDANCE_SCHEMA
    payload_round_trip_exact: Literal[True] = True
    domains_share_identical_xip2_bytes: Literal[True] = True
    xip2_payload_is_not_embedded_in_current_a3_packet: Literal[True] = True
    pitch_is_not_carried_by_current_xip2_container: Literal[True] = True
    full_xip2_a3_receiver_implemented: Literal[False] = False
    scorer_invoked: Literal[False] = False
    score_claim: Literal[False] = False
    research_only: Literal[True] = True

    def __post_init__(self) -> None:
        if self.schema != _XIP2_GUIDANCE_SCHEMA:
            raise ChronologicalA3EncoderError("XIP2 guidance schema changed")
        _require_sha256(self.source_binding_sha256, "XIP2 source binding")
        if type(self.source_pair_ids) is not tuple or not self.source_pair_ids:
            raise ChronologicalA3EncoderError("XIP2 guidance pair IDs must be a nonempty tuple")
        if type(self.coder) is not XIP2Coder:
            raise ChronologicalA3EncoderError("XIP2 guidance coder escaped its closed universe")
        if type(self.q_levels) is not int or not 1 <= self.q_levels <= 32767:
            raise ChronologicalA3EncoderError("XIP2 guidance q_levels escaped int16 quantization")
        if type(self.pitch) is not float or not math.isfinite(self.pitch):
            raise ChronologicalA3EncoderError("XIP2 guidance pitch must be one finite exact float")
        if type(self.xip2_payload) is not bytes or not self.xip2_payload.startswith(b"XIP2"):
            raise ChronologicalA3EncoderError("XIP2 guidance payload is not one exact XIP2 byte string")
        if type(self.camera_then_r) is not EncoderOnlyA3TargetV1 or type(
            self.scorer_then_factor2
        ) is not EncoderOnlyA3TargetV1:
            raise ChronologicalA3EncoderError("XIP2 guidance targets changed exact type")
        expected_ids = self.source_pair_ids
        if (
            self.camera_then_r.source_pair_ids != expected_ids
            or self.scorer_then_factor2.source_pair_ids != expected_ids
            or self.camera_then_r.evidence_kind != "xip2_camera_then_r_guidance_encoder_only.v1"
            or self.scorer_then_factor2.evidence_kind
            != "xip2_scorer_then_factor2_guidance_encoder_only.v1"
        ):
            raise ChronologicalA3EncoderError("XIP2 guidance target bindings/domains differ")
        truth = (
            self.payload_round_trip_exact is True
            and self.domains_share_identical_xip2_bytes is True
            and self.xip2_payload_is_not_embedded_in_current_a3_packet is True
            and self.pitch_is_not_carried_by_current_xip2_container is True
            and self.full_xip2_a3_receiver_implemented is False
            and self.scorer_invoked is False
            and self.score_claim is False
            and self.research_only is True
        )
        if not truth:
            raise ChronologicalA3EncoderError("XIP2 guidance truth labels were relaxed")

    @property
    def xip2_payload_sha256(self) -> str:
        return _sha256(self.xip2_payload)

    def target_for_domain(self, domain: XIP2WarpDomain) -> EncoderOnlyA3TargetV1:
        if type(domain) is not XIP2WarpDomain:
            raise ChronologicalA3EncoderError("XIP2 target lookup requires exact domain enum")
        return {
            XIP2WarpDomain.CAMERA_THEN_R: self.camera_then_r,
            XIP2WarpDomain.SCORER_THEN_FACTOR2: self.scorer_then_factor2,
        }[domain]


def _rank_rows(
    *,
    interpretation: ChronologicalA3Interpretation,
    source_pair_ids: tuple[int, ...],
    p0_numerators: np.ndarray,
    corrected_y1_numerators: np.ndarray,
    target_numerators: np.ndarray,
    denominator: int,
    limit: int,
) -> tuple[tuple[RankedA3CellV1, ...], int]:
    base_delta = p0_numerators - target_numerators
    base_sse = np.sum(base_delta * base_delta, axis=-1, dtype=np.int64)
    constant_rgb: np.ndarray | None = None
    if interpretation is ChronologicalA3Interpretation.TARGET_CONSTANT_RGB_V1:
        constant_rgb = np.floor_divide(target_numerators + denominator // 2, denominator)
        constant_rgb = np.clip(constant_rgb, 0, 255).astype(np.uint8)
        candidate_numerators = constant_rgb.astype(np.int64) * denominator
    elif interpretation is ChronologicalA3Interpretation.CORRECTED_Y1_SUPPORT_COPY_V1:
        candidate_numerators = corrected_y1_numerators
    else:  # pragma: no cover - enum exhaustiveness
        raise ChronologicalA3EncoderError("unsupported A3 acquisition interpretation")
    candidate_delta = candidate_numerators - target_numerators
    candidate_sse = np.sum(candidate_delta * candidate_delta, axis=-1, dtype=np.int64)
    reduction = base_sse - candidate_sse
    positive_flat = np.flatnonzero(reduction.reshape(-1) > 0)
    available = int(positive_flat.size)
    if available < limit:
        raise ChronologicalA3EncoderError(
            f"{interpretation.value} has {available} positive rows, fewer than requested prefix {limit}"
        )
    local_pair, scorer_row, scorer_col = np.unravel_index(
        positive_flat,
        (len(source_pair_ids), SCORER_HEIGHT, SCORER_WIDTH),
    )
    gains = reduction.reshape(-1)[positive_flat]
    order = np.lexsort((scorer_col, scorer_row, local_pair, -gains))
    chosen = order[:limit]
    rows: list[RankedA3CellV1] = []
    for rank, index in enumerate(chosen):
        pair_local = int(local_pair[index])
        row = int(scorer_row[index])
        col = int(scorer_col[index])
        rgb: tuple[int, int, int] | None = None
        if constant_rgb is not None:
            rgb = tuple(int(value) for value in constant_rgb[pair_local, row, col])  # type: ignore[assignment]
        rows.append(
            RankedA3CellV1(
                rank=rank,
                source_pair_id=source_pair_ids[pair_local],
                scorer_row=row,
                scorer_col=col,
                numerator_sse_before=int(base_sse[pair_local, row, col]),
                numerator_sse_after=int(candidate_sse[pair_local, row, col]),
                numerator_sse_reduction=int(reduction[pair_local, row, col]),
                constant_rgb_u8=rgb,
            )
        )
    return tuple(rows), available


def _program_from_prefix(
    interpretation: ChronologicalA3Interpretation,
    prefix: tuple[RankedA3CellV1, ...],
) -> PredictorPreservingA3ProgramV1:
    if interpretation is ChronologicalA3Interpretation.TARGET_CONSTANT_RGB_V1:
        rows = tuple(
            sorted(
                (
                    SparseConstantRGBCellV1(
                        row.source_pair_id,
                        row.scorer_row,
                        row.scorer_col,
                        row.constant_rgb_u8,  # type: ignore[arg-type]
                    )
                    for row in prefix
                ),
                key=lambda row: row.address,
            )
        )
        return PredictorPreservingA3ProgramV1(
            PredictorPreservingA3Mode.SPARSE_CONSTANT_RGB_V1,
            constant_rgb_cells=rows,
        )
    rows_copy = tuple(
        sorted(
            (
                CorrectedY1SupportCopyCellV1(
                    row.source_pair_id,
                    row.scorer_row,
                    row.scorer_col,
                )
                for row in prefix
            ),
            key=lambda row: row.address,
        )
    )
    return PredictorPreservingA3ProgramV1(
        PredictorPreservingA3Mode.COPY_CORRECTED_Y1_SUPPORT_V1,
        corrected_y1_copy_cells=rows_copy,
    )


def _strict_verify_compiled_a3(
    compiled: CompiledPredictorPreservingA3V1,
    *,
    predictor_surface: PredictorCameraPairSurfaceV1,
    decoded_g: DecodedGenerativeCorrectionV1,
    corrected_y1_overlay: PredictorPreservingOverlayResultV1,
) -> None:
    first_parse = parse_predictor_preserving_a3_packet(
        compiled.packet,
        expected_source_binding=compiled.source_binding,
    )
    second_parse = parse_predictor_preserving_a3_packet(
        compiled.packet,
        expected_source_binding=compiled.source_binding,
    )
    if first_parse != second_parse or first_parse.program != compiled.program:
        raise ChronologicalA3EncoderError("compiled A3 packet changed across strict parses")
    first_decode = decode_predictor_preserving_a3_packet(
        compiled.packet,
        predictor_surface=predictor_surface,
        decoded_g=decoded_g,
        corrected_y1_overlay=corrected_y1_overlay,
    )
    second_decode = decode_predictor_preserving_a3_packet(
        compiled.packet,
        predictor_surface=predictor_surface,
        decoded_g=decoded_g,
        corrected_y1_overlay=corrected_y1_overlay,
    )
    if first_decode.receipt != second_decode.receipt or first_decode.receipt != compiled.receipt:
        raise ChronologicalA3EncoderError("compiled A3 receipt changed across public decodes")
    for field in (
        "camera_y0",
        "camera_y1",
        "chronological_camera_frames",
        "owned_scorer_mask",
        "owned_camera_mask",
    ):
        if not np.array_equal(getattr(first_decode, field), getattr(second_decode, field)):
            raise ChronologicalA3EncoderError(f"compiled A3 {field} changed across public decodes")
    if not np.array_equal(first_decode.camera_y1, corrected_y1_overlay.camera_y1):
        raise ChronologicalA3EncoderError("compiled A3 changed exact corrected Y1")
    if not np.array_equal(
        first_decode.camera_y0[~first_decode.owned_camera_mask],
        predictor_surface.camera_p0[~first_decode.owned_camera_mask],
    ):
        raise ChronologicalA3EncoderError("compiled A3 changed nonselected P0 camera values")
    p0_numerators, denominator = _numerator_stack(predictor_surface.camera_p0)
    y0_numerators, output_denominator = _numerator_stack(first_decode.camera_y0)
    if denominator != output_denominator or not np.array_equal(
        y0_numerators[~first_decode.owned_scorer_mask],
        p0_numerators[~first_decode.owned_scorer_mask],
    ):
        raise ChronologicalA3EncoderError("compiled A3 changed nonselected P0 scorer numerators")


def _replace_a_packet(
    current: TaskspaceSectionBundleV1,
    *,
    source_binding: PredictorPreservingA3SourceBindingV1,
    replacement_packet: bytes,
) -> TaskspaceSectionBundleV1:
    """Replace only A after rechecking exact P/G and both A source bindings."""

    if type(current) is not TaskspaceSectionBundleV1:
        raise ChronologicalA3EncoderError("A3 allocator transform requires exact section bundle type")
    if _sha256(current.predictor_packet) != source_binding.predictor_program_sha256:
        raise ChronologicalA3EncoderError("A3 allocator transform predictor P differs from source binding")
    if _sha256(current.generative_correction_packet) != source_binding.correction_packet_sha256:
        raise ChronologicalA3EncoderError("A3 allocator transform correction G differs from source binding")
    try:
        parse_predictor_preserving_a3_packet(
            current.coupled_preimage_packet,
            expected_source_binding=source_binding,
        )
        parse_predictor_preserving_a3_packet(
            replacement_packet,
            expected_source_binding=source_binding,
        )
    except PredictorPreservingA3Error as exc:
        raise ChronologicalA3EncoderError("A3 allocator transform packet/source parse failed") from exc
    result = TaskspaceSectionBundleV1(
        predictor_packet=current.predictor_packet,
        generative_correction_packet=current.generative_correction_packet,
        coupled_preimage_packet=replacement_packet,
        terminal_quotient_packet=current.terminal_quotient_packet,
    )
    if (
        result.predictor_packet != current.predictor_packet
        or result.generative_correction_packet != current.generative_correction_packet
        or result.terminal_quotient_packet != current.terminal_quotient_packet
    ):
        raise ChronologicalA3EncoderError("A3 allocator transform changed P, G, or optional T")
    return result


def compile_chronological_a3_proposal_groups(
    *,
    predictor_surface: PredictorCameraPairSurfaceV1,
    decoded_g: DecodedGenerativeCorrectionV1,
    corrected_y1_overlay: PredictorPreservingOverlayResultV1,
    target: EncoderOnlyA3TargetV1,
    plans: tuple[A3PrefixPlanV1, ...],
    proposal_id_prefix: str = "a3",
) -> ChronologicalA3AcquisitionV1:
    """Compile finite exact A packets and g7-compatible replacement proposals.

    Target-numerator SSE is an acquisition ordering only.  No scorer is loaded,
    no through-R or n600 claim is made, and no proposal is admitted here.
    """

    if type(predictor_surface) is not PredictorCameraPairSurfaceV1:
        raise ChronologicalA3EncoderError("chronological A3 compile requires exact predictor surface")
    if type(decoded_g) is not DecodedGenerativeCorrectionV1:
        raise ChronologicalA3EncoderError("chronological A3 compile requires exact decoded G")
    if type(corrected_y1_overlay) is not PredictorPreservingOverlayResultV1:
        raise ChronologicalA3EncoderError("chronological A3 compile requires exact corrected Y1 overlay")
    if type(target) is not EncoderOnlyA3TargetV1:
        raise ChronologicalA3EncoderError("chronological A3 compile requires exact encoder-only target")
    if type(plans) is not tuple or not plans or any(type(plan) is not A3PrefixPlanV1 for plan in plans):
        raise ChronologicalA3EncoderError("chronological A3 plans must be a nonempty exact typed tuple")
    interpretations = tuple(plan.interpretation for plan in plans)
    if len(set(interpretations)) != len(interpretations):
        raise ChronologicalA3EncoderError("chronological A3 plans repeat an interpretation")
    if _PROPOSAL_PREFIX_RE.fullmatch(proposal_id_prefix) is None:
        raise ChronologicalA3EncoderError("proposal_id_prefix must be 1-32 safe ASCII characters")

    try:
        source = derive_predictor_preserving_a3_source_binding(
            predictor_surface,
            decoded_g,
            corrected_y1_overlay,
        )
    except PredictorPreservingA3Error as exc:
        raise ChronologicalA3EncoderError("chronological A3 source binding failed") from exc
    if target.source_pair_ids != source.source_pair_ids:
        raise ChronologicalA3EncoderError("target pair order differs from exact P/G/Y1 source binding")

    try:
        pass_control = compile_predictor_preserving_a3(
            PredictorPreservingA3ProgramV1(PredictorPreservingA3Mode.PASS_P0_V1),
            predictor_surface=predictor_surface,
            decoded_g=decoded_g,
            corrected_y1_overlay=corrected_y1_overlay,
        )
    except PredictorPreservingA3Error as exc:
        raise ChronologicalA3EncoderError("PASS_P0 control compile failed") from exc
    _strict_verify_compiled_a3(
        pass_control,
        predictor_surface=predictor_surface,
        decoded_g=decoded_g,
        corrected_y1_overlay=corrected_y1_overlay,
    )

    p0_numerators, denominator = _numerator_stack(predictor_surface.camera_p0)
    corrected_y1_numerators, y1_denominator = _numerator_stack(corrected_y1_overlay.camera_y1)
    target_numerators, target_denominator = _numerator_stack(target.target_camera_y0)
    if denominator != y1_denominator or denominator != target_denominator:
        raise ChronologicalA3EncoderError("P0/Y1/target scorer denominators differ")

    results: list[CompiledChronologicalA3ProposalV1] = []
    for plan in plans:
        ranked, available = _rank_rows(
            interpretation=plan.interpretation,
            source_pair_ids=source.source_pair_ids,
            p0_numerators=p0_numerators,
            corrected_y1_numerators=corrected_y1_numerators,
            target_numerators=target_numerators,
            denominator=denominator,
            limit=plan.row_counts[-1],
        )
        for row_count in plan.row_counts:
            prefix = ranked[:row_count]
            program = _program_from_prefix(plan.interpretation, prefix)
            try:
                compiled = compile_predictor_preserving_a3(
                    program,
                    predictor_surface=predictor_surface,
                    decoded_g=decoded_g,
                    corrected_y1_overlay=corrected_y1_overlay,
                )
            except PredictorPreservingA3Error as exc:
                raise ChronologicalA3EncoderError("strict A3 compiler refused ranked prefix") from exc
            _strict_verify_compiled_a3(
                compiled,
                predictor_surface=predictor_surface,
                decoded_g=decoded_g,
                corrected_y1_overlay=corrected_y1_overlay,
            )
            packet_hash = _sha256(compiled.packet)
            proposal_id = (
                f"{proposal_id_prefix}:{plan.interpretation.value.lower()}:{row_count}:{packet_hash[:12]}"
            )
            before_sse = sum(row.numerator_sse_before for row in prefix)
            after_sse = sum(row.numerator_sse_after for row in prefix)
            reduction = sum(row.numerator_sse_reduction for row in prefix)
            receipt = ChronologicalA3EncoderReceiptV1(
                schema=RECEIPT_SCHEMA,
                acquisition_id=ACQUISITION_ID,
                proposal_id=proposal_id,
                interpretation=plan.interpretation.value,
                packet_mode=compiled.program.mode.value,
                source_pair_ids=source.source_pair_ids,
                source_binding_sha256=source.binding_sha256,
                predictor_program_sha256=source.predictor_program_sha256,
                correction_packet_sha256=source.correction_packet_sha256,
                corrected_camera_y1_sha256=source.corrected_camera_y1_sha256,
                target_evidence_binding_sha256=target.binding_sha256,
                packet_sha256=packet_hash,
                packet_bytes=len(compiled.packet),
                control_row_count=row_count,
                available_positive_rows=available,
                selected_numerator_sse_before=before_sse,
                selected_numerator_sse_after=after_sse,
                selected_numerator_sse_reduction=reduction,
                serialized_coordinate_bytes=compiled.receipt.serialized_coordinate_bytes,
                serialized_rgb_value_bytes=compiled.receipt.serialized_rgb_value_bytes,
            )
            if parse_chronological_a3_encoder_receipt(receipt.to_receipt_bytes()) != receipt:
                raise ChronologicalA3EncoderError("A3 encoder receipt failed strict parse/re-emit")
            allocator_proposal = TaskspaceWholeArchiveProposalV1(
                proposal_id=proposal_id,
                transform=partial(
                    _replace_a_packet,
                    source_binding=source,
                    replacement_packet=compiled.packet,
                ),
            )
            results.append(
                CompiledChronologicalA3ProposalV1(
                    interpretation=plan.interpretation,
                    ranked_prefix=prefix,
                    compiled_a3=compiled,
                    receipt=receipt,
                    allocator_proposal=allocator_proposal,
                )
            )
    return ChronologicalA3AcquisitionV1(
        source_binding=source,
        pass_p0_control=pass_control,
        proposals=tuple(results),
    )


def build_xip2_guidance_targets(
    *,
    predictor_surface: PredictorCameraPairSurfaceV1,
    decoded_g: DecodedGenerativeCorrectionV1,
    corrected_y1_overlay: PredictorPreservingOverlayResultV1,
    xi: np.ndarray,
    pitch: float,
    q_levels: int,
    coder: XIP2Coder = XIP2Coder.DELTA_AR,
) -> EncoderOnlyXIP2GuidanceV1:
    """Build both XIP2 warp-domain guidance targets from identical xi bytes.

    ``CAMERA_THEN_R`` warps exact corrected camera Y1 before the frozen resize.
    ``SCORER_THEN_FACTOR2`` warps the corrected scorer plane, rounds the current
    uint8 guidance control, and realizes it with the certified disjoint factor-2
    primitive.  The latter is honestly a uint8 guidance control, not a claim of
    arbitrary sub-uint8 scorer-plane realization.
    """

    if type(coder) is not XIP2Coder:
        raise ChronologicalA3EncoderError("XIP2 guidance coder must use exact closed enum")
    if type(q_levels) is not int or not 1 <= q_levels <= 32767:
        raise ChronologicalA3EncoderError("XIP2 guidance q_levels must be in [1,32767]")
    if type(pitch) not in (float, int) or isinstance(pitch, bool) or not math.isfinite(float(pitch)):
        raise ChronologicalA3EncoderError("XIP2 guidance pitch must be finite")
    try:
        source = derive_predictor_preserving_a3_source_binding(
            predictor_surface,
            decoded_g,
            corrected_y1_overlay,
        )
    except PredictorPreservingA3Error as exc:
        raise ChronologicalA3EncoderError("XIP2 guidance source binding failed") from exc
    values = np.asarray(xi)
    if values.dtype.kind not in ("f", "i", "u") or values.shape != (source.pair_count, 6):
        raise ChronologicalA3EncoderError("XIP2 guidance xi must be numeric [pair,6]")
    if not np.all(np.isfinite(values)):
        raise ChronologicalA3EncoderError("XIP2 guidance xi must be finite")
    try:
        q, scales = quantize_xi(values.astype(np.float64), q_levels=q_levels)
        payload = serialize_xi_payload(q, scales, coder=coder.value)
        transport = SE3XiTransportV2(
            counted_payload=payload,
            source_pair_ids=source.source_pair_ids,
            predictor_program_sha256=source.predictor_program_sha256,
        )
        if not np.array_equal(transport.q_codes, q) or not np.array_equal(
            transport.scales,
            scales,
        ):
            raise ChronologicalA3EncoderError("XIP2 payload did not round-trip exact q/scales")
        quantized_xi = transport.xi
    except (XiPoseCoderError, TaskspacePredictorStateV2Error) as exc:
        raise ChronologicalA3EncoderError("XIP2 guidance quantize/serialize/parse failed") from exc
    if quantized_xi.shape != (source.pair_count, 6):
        raise ChronologicalA3EncoderError("XIP2 decoded xi shape differs from source pairs")

    camera_geom = GroundHomographyGeom.eon((CAMERA_HEIGHT, CAMERA_WIDTH), pitch=float(pitch))
    scorer_geom = GroundHomographyGeom.eon((SCORER_HEIGHT, SCORER_WIDTH), pitch=float(pitch))
    camera_targets: list[np.ndarray] = []
    scorer_targets: list[np.ndarray] = []
    operator = _operator()
    try:
        for pair_index in range(source.pair_count):
            corrected_camera = corrected_y1_overlay.camera_y1[pair_index]
            camera_targets.append(
                warp_frame0_uint8_numpy(corrected_camera, quantized_xi[pair_index], camera_geom)
            )
            corrected_scorer = operator.apply(corrected_camera)
            warped_scorer = warp_frame0_native_numpy(
                corrected_scorer,
                quantized_xi[pair_index],
                scorer_geom,
            )
            scorer_u8 = np.clip(np.round(warped_scorer), 0.0, 255.0).astype(np.uint8)
            scorer_targets.append(operator.realize_factor2_uint8(scorer_u8))
    except (WarpRealLumaFrame0Error, Uint8LatticeError, np.linalg.LinAlgError) as exc:
        raise ChronologicalA3EncoderError("XIP2 guidance warp/realization failed") from exc

    payload_hash = _sha256(payload)
    pitch_float = float(pitch)
    shared = {
        "schema": _XIP2_GUIDANCE_SCHEMA,
        "source_binding_sha256": source.binding_sha256,
        "xip2_payload_sha256": payload_hash,
        "coder": coder.value,
        "q_levels": q_levels,
        "pitch": pitch_float,
        "pitch_is_not_in_xip2_payload": True,
    }

    def target_for(domain: XIP2WarpDomain, camera: np.ndarray) -> EncoderOnlyA3TargetV1:
        custody = _sha256(_canonical_json({**shared, "domain": domain.value}))
        evidence_kind = {
            XIP2WarpDomain.CAMERA_THEN_R: "xip2_camera_then_r_guidance_encoder_only.v1",
            XIP2WarpDomain.SCORER_THEN_FACTOR2: (
                "xip2_scorer_then_factor2_guidance_encoder_only.v1"
            ),
        }[domain]
        return EncoderOnlyA3TargetV1(
            source_pair_ids=source.source_pair_ids,
            target_camera_y0=np.stack(camera, axis=0) if isinstance(camera, list) else camera,
            custody_sha256=custody,
            evidence_kind=evidence_kind,
        )

    camera_target = target_for(
        XIP2WarpDomain.CAMERA_THEN_R,
        np.stack(camera_targets, axis=0),
    )
    scorer_target = target_for(
        XIP2WarpDomain.SCORER_THEN_FACTOR2,
        np.stack(scorer_targets, axis=0),
    )
    return EncoderOnlyXIP2GuidanceV1(
        source_binding_sha256=source.binding_sha256,
        source_pair_ids=source.source_pair_ids,
        coder=coder,
        q_levels=q_levels,
        pitch=pitch_float,
        xip2_payload=payload,
        camera_then_r=camera_target,
        scorer_then_factor2=scorer_target,
    )


__all__ = [
    "ACQUISITION_ID",
    "RECEIPT_SCHEMA",
    "A3PrefixPlanV1",
    "ChronologicalA3AcquisitionV1",
    "ChronologicalA3EncoderError",
    "ChronologicalA3EncoderReceiptV1",
    "ChronologicalA3Interpretation",
    "CompiledChronologicalA3ProposalV1",
    "EncoderOnlyA3TargetV1",
    "EncoderOnlyXIP2GuidanceV1",
    "RankedA3CellV1",
    "XIP2Coder",
    "XIP2WarpDomain",
    "build_xip2_guidance_targets",
    "compile_chronological_a3_proposal_groups",
    "parse_chronological_a3_encoder_receipt",
]
