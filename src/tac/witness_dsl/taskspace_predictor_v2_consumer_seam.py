# SPDX-License-Identifier: MIT
"""Executable V2 predictor seam for the existing finite G and A grammars.

The V1 G decoder was intentionally coupled to V9 Pose6 state and the V1 A
entrypoints enforce that exact type.  This module leaves those byte vectors and
entrypoints untouched.  It exposes a V2-only seam that:

* admits label-local G primitives without inventing Pose6;
* refuses transport-dependent G before receiver mutation; and
* runs the pose-independent A materializer against the V2 semantic foreign key.

The resulting G/A packets retain their existing finite wire grammars.  Their P
foreign keys are derived from the exact V2 state supplied to this seam.  The A
adapter is only a mechanical grammar bridge: its inherited ``paint_rgb`` step
repaints the full semantic grid and therefore is not a signal-preserving ep725
composition.  Candidate composition must pass
``require_signal_preserving_ep725_a_surface``; today that gate fails closed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

import tac.witness_dsl.coupled_preimage_program as _a
import tac.witness_dsl.generative_taskspace_correction as _g
from tac.witness_dsl.coupled_preimage_program import (
    CompiledCoupledPreimageProgramV1,
    CoupledPreimageCompileReceiptV1,
    CoupledPreimageMode,
    CoupledPreimageProgramError,
    CoupledPreimageProgramV1,
    DecodedCoupledPreimageV1,
    Frame1AnchoredY0FibreControlV1,
    JointSharedSkeletonTwoFibreControlV1,
)
from tac.witness_dsl.generative_taskspace_correction import (
    CompiledGenerativeCorrectionV1,
    DecodedGenerativeCorrectionV1,
    EncoderOnlyTeacherEvidenceV1,
    GenerativeCorrectionProgramV1,
    ParsedGenerativeCorrectionV1,
    PredictorSemanticStateV1,
)
from tac.witness_dsl.taskspace_predictor_state_v2 import (
    PredictorSemanticStateProtocolV2,
    TaskspacePredictorStateV2,
    TaskspacePredictorStateV2Error,
    V9Pose6TransportV2,
    require_g_transport_admission,
)


class TaskspacePredictorV2ConsumerSeamError(ValueError):
    """A V2 P/G/A seam precondition or live foreign key failed closed."""


SIGNAL_LOSS_FULL_REPAINT = "SIGNAL_LOSS_FULL_REPAINT"


class TaskspacePredictorV2SignalBlocker(TaskspacePredictorV2ConsumerSeamError):
    """A mechanical seam would discard useful decoder-owned predictor signal."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class _GenerativeStateViewV2:
    """Structural state expected by G, with Pose6 exposed only when it exists."""

    state: TaskspacePredictorStateV2

    @property
    def predictor_program_sha256(self) -> str:
        return self.state.predictor_program_sha256

    @property
    def predictor_renderer_sha256(self) -> str:
        return self.state.predictor_renderer_sha256

    @property
    def source_pair_ids(self) -> tuple[int, ...]:
        return self.state.source_pair_ids

    @property
    def labels(self) -> np.ndarray:
        return self.state.labels

    @property
    def pair_start(self) -> int:
        return self.state.pair_start

    @property
    def pair_count(self) -> int:
        return self.state.pair_count

    @property
    def labels_sha256(self) -> str:
        return self.state.labels_sha256

    @property
    def binding_sha256(self) -> str:
        return self.state.binding_sha256

    @property
    def pose6_codes(self) -> np.ndarray:
        transport = self.state.transport
        if type(transport) is not V9Pose6TransportV2:
            raise TaskspacePredictorStateV2Error("G attempted to consume Pose6 where V2 transport is not V9_POSE6")
        return transport.pose6_codes


def _exact_state(state: TaskspacePredictorStateV2) -> TaskspacePredictorStateV2:
    if type(state) is not TaskspacePredictorStateV2:
        raise TaskspacePredictorV2ConsumerSeamError("consumer seam requires exact TaskspacePredictorStateV2")
    return state


def _exact_semantic_state(
    state: PredictorSemanticStateProtocolV2,
) -> PredictorSemanticStateProtocolV2:
    if type(state) not in {PredictorSemanticStateV1, TaskspacePredictorStateV2}:
        raise TaskspacePredictorV2ConsumerSeamError("pose-independent A seam requires exact V1 or V2 semantic state")
    if not isinstance(state, PredictorSemanticStateProtocolV2):
        raise TaskspacePredictorV2ConsumerSeamError("predictor state does not implement the common semantic protocol")
    return state


def require_signal_preserving_ep725_a_surface(
    predictor_state: TaskspacePredictorStateV2,
) -> None:
    """Block candidate A until exact ephemeral P Y1 and overlay ABI are bound."""

    _exact_state(predictor_state)
    raise TaskspacePredictorV2SignalBlocker(
        SIGNAL_LOSS_FULL_REPAINT,
        "exact camera P frames and semantic G ownership exist, but no bound semantic-to-camera overlay/R ABI; "
        "inherited A paint_rgb would replace unchanged P RGB and temporal signal",
    )


@dataclass(frozen=True, slots=True)
class GCorrectionOwnershipV2:
    """Exact semantic cells owned by G, derived only by P/G label difference."""

    predictor_state: TaskspacePredictorStateV2
    decoded_g: DecodedGenerativeCorrectionV1
    owned_mask: np.ndarray
    ownership_rule: str = "decoded_g_labels_not_equal_predictor_labels"
    scorer_invoked: bool = False
    candidate_claim: bool = False

    def __post_init__(self) -> None:
        state = _exact_state(self.predictor_state)
        if type(self.decoded_g) is not DecodedGenerativeCorrectionV1:
            raise TaskspacePredictorV2ConsumerSeamError("G ownership requires exact decoded correction")
        if self.decoded_g.labels.shape != state.labels.shape:
            raise TaskspacePredictorV2ConsumerSeamError("G ownership label population differs from P semantics")
        if self.decoded_g.realization_profile is None:
            raise TaskspacePredictorV2ConsumerSeamError("G ownership has no finite correction realization colours")
        mask = np.asarray(self.owned_mask)
        expected = self.decoded_g.labels != state.labels
        if mask.dtype != np.bool_ or mask.shape != expected.shape or not np.array_equal(mask, expected):
            raise TaskspacePredictorV2ConsumerSeamError("G owned mask is not exact decoded-label inequality")
        if (
            self.ownership_rule != "decoded_g_labels_not_equal_predictor_labels"
            or self.scorer_invoked is not False
            or self.candidate_claim is not False
        ):
            raise TaskspacePredictorV2ConsumerSeamError("G ownership truth contract changed")
        immutable = np.ascontiguousarray(mask, dtype=np.bool_).copy()
        immutable.setflags(write=False)
        object.__setattr__(self, "owned_mask", immutable)

    @property
    def changed_cells(self) -> int:
        return int(np.count_nonzero(self.owned_mask))

    @property
    def ownership_mask_sha256(self) -> str:
        return hashlib.sha256(memoryview(self.owned_mask).cast("B")).hexdigest()

    @property
    def corrected_labels(self) -> np.ndarray:
        return self.decoded_g.labels

    @property
    def correction_rgb_by_role(self) -> tuple[tuple[int, int, int], ...]:
        profile = self.decoded_g.realization_profile
        assert profile is not None
        return profile.role_rgb_u8


def derive_g_correction_ownership_v2(
    predictor_state: TaskspacePredictorStateV2,
    decoded_g: DecodedGenerativeCorrectionV1,
) -> GCorrectionOwnershipV2:
    """Derive the only cells a future signal-preserving overlay may replace."""

    state = _exact_state(predictor_state)
    if type(decoded_g) is not DecodedGenerativeCorrectionV1:
        raise TaskspacePredictorV2ConsumerSeamError("G ownership requires exact decoded correction")
    return GCorrectionOwnershipV2(
        predictor_state=state,
        decoded_g=decoded_g,
        owned_mask=np.ascontiguousarray(decoded_g.labels != state.labels),
    )


def parse_generative_taskspace_correction_v2(
    packet: bytes,
    *,
    predictor_state: TaskspacePredictorStateV2,
) -> ParsedGenerativeCorrectionV1:
    """Parse G, then prove its transport dependency before any label mutation."""

    state = _exact_state(predictor_state)
    view = _GenerativeStateViewV2(state)
    parsed = _g.parse_generative_taskspace_correction(packet, predictor_state=view)  # type: ignore[arg-type]
    require_g_transport_admission(state, parsed.program)
    return parsed


def apply_generative_taskspace_correction_v2(
    packet: bytes,
    *,
    predictor_state: TaskspacePredictorStateV2,
) -> DecodedGenerativeCorrectionV1:
    """Apply a transport-admitted G packet to exact V2 predictor semantics."""

    state = _exact_state(predictor_state)
    parse_generative_taskspace_correction_v2(packet, predictor_state=state)
    return _g.apply_generative_taskspace_correction(  # type: ignore[arg-type]
        packet,
        predictor_state=_GenerativeStateViewV2(state),
    )


def compile_generative_taskspace_correction_v2(
    predictor_state: TaskspacePredictorStateV2,
    program: GenerativeCorrectionProgramV1,
    *,
    teacher_evidence: EncoderOnlyTeacherEvidenceV1,
) -> CompiledGenerativeCorrectionV1:
    """Compile G only after the live family/transport dependency is admitted."""

    state = _exact_state(predictor_state)
    if type(program) is not GenerativeCorrectionProgramV1:
        raise TaskspacePredictorV2ConsumerSeamError("G program must use the exact finite grammar type")
    if type(teacher_evidence) is not EncoderOnlyTeacherEvidenceV1:
        raise TaskspacePredictorV2ConsumerSeamError("G teacher evidence must use the exact encoder-only type")
    require_g_transport_admission(state, program)
    compiled = _g.compile_generative_taskspace_correction(  # type: ignore[arg-type]
        _GenerativeStateViewV2(state),
        program,
        teacher_evidence=teacher_evidence,
    )
    if compiled.receipt.predictor_binding_sha256 != state.binding_sha256:
        raise TaskspacePredictorV2ConsumerSeamError("compiled G packet lost the exact V2 P foreign key")
    return compiled


def _validate_a_live_source_binding_v2(
    source: _a.CoupledPreimageSourceBindingV1,
    predictor_state: PredictorSemanticStateProtocolV2,
    decoded_g: DecodedGenerativeCorrectionV1,
) -> None:
    if type(decoded_g) is not DecodedGenerativeCorrectionV1:
        raise TaskspacePredictorV2ConsumerSeamError("A seam requires exact DecodedGenerativeCorrectionV1")
    if decoded_g.realization_profile is None:
        raise CoupledPreimageProgramError("decoded G has no finite realization profile")
    if decoded_g.labels.shape[0] != predictor_state.pair_count:
        raise CoupledPreimageProgramError("decoded G pair count differs from V2 predictor state")
    live_values = {
        "predictor_program_sha256": predictor_state.predictor_program_sha256,
        "predictor_renderer_sha256": predictor_state.predictor_renderer_sha256,
        "source_pair_ids": predictor_state.source_pair_ids,
        "predictor_labels_sha256": predictor_state.labels_sha256,
        "correction_packet_sha256": decoded_g.correction_packet_sha256,
        "correction_labels_sha256": _a._array_content_sha256(
            decoded_g.labels,
            role="decoded_g_semantic_labels",
        ),
        "realization_profile_sha256": _a._realization_profile_sha256(decoded_g.realization_profile),
    }
    mismatches = [field_name for field_name, live in live_values.items() if getattr(source, field_name) != live]
    if mismatches:
        raise CoupledPreimageProgramError(f"live V2 P/G source binding mismatch: {sorted(mismatches)}")


def decode_coupled_preimage_program_v2(
    packet: bytes,
    *,
    predictor_state: PredictorSemanticStateProtocolV2,
    decoded_g: DecodedGenerativeCorrectionV1,
) -> DecodedCoupledPreimageV1:
    """Run A in its mandated G→exact Y1→Y0 order from a V2 semantic P."""

    state = _exact_semantic_state(predictor_state)
    program = _a.parse_coupled_preimage_program(packet)
    _validate_a_live_source_binding_v2(program.source_binding, state, decoded_g)

    exact_y1 = np.ascontiguousarray(decoded_g.paint_rgb(), dtype=np.uint8)
    actual_y1_sha256 = _a._array_content_sha256(exact_y1, role="exact_realized_uint8_y1")
    if actual_y1_sha256 != program.source_binding.exact_y1_content_sha256:
        raise CoupledPreimageProgramError("exact realized uint8 Y1 conditioning hash mismatch before Y0")

    y0 = _a._materialize_y0(program, exact_y1, decoded_g.labels)
    chronological = np.ascontiguousarray(np.stack((y0, exact_y1), axis=1), dtype=np.uint8)
    actual_y0_sha256 = _a._array_content_sha256(y0, role="expected_realized_uint8_y0")
    actual_chronological_sha256 = _a._array_content_sha256(
        chronological,
        role="chronological_y0_y1",
    )
    if actual_y0_sha256 != program.source_binding.expected_y0_content_sha256:
        raise CoupledPreimageProgramError("materialized Y0 differs from the V2 packet-bound expected output")
    if actual_chronological_sha256 != program.source_binding.expected_chronological_content_sha256:
        raise CoupledPreimageProgramError("chronological output differs from the V2 packet-bound expected output")
    return DecodedCoupledPreimageV1(
        y0=y0,
        y1=exact_y1,
        chronological_frames=chronological,
        exact_y1_content_sha256=actual_y1_sha256,
        chronological_content_sha256=actual_chronological_sha256,
    )


def compile_coupled_preimage_program_v2(
    predictor_state: PredictorSemanticStateProtocolV2,
    decoded_g: DecodedGenerativeCorrectionV1,
    *,
    mode: CoupledPreimageMode,
    anchored_controls: tuple[Frame1AnchoredY0FibreControlV1, ...] = (),
    joint_controls: tuple[JointSharedSkeletonTwoFibreControlV1, ...] = (),
) -> CompiledCoupledPreimageProgramV1:
    """Compile the research-only A grammar; this is not ep725 signal-preserving."""

    state = _exact_semantic_state(predictor_state)
    if type(decoded_g) is not DecodedGenerativeCorrectionV1:
        raise TaskspacePredictorV2ConsumerSeamError("A seam requires exact DecodedGenerativeCorrectionV1")
    if type(mode) is not CoupledPreimageMode:
        raise TaskspacePredictorV2ConsumerSeamError("A mode must use the exact closed enum")
    if type(anchored_controls) is not tuple or type(joint_controls) is not tuple:
        raise TaskspacePredictorV2ConsumerSeamError("A control families must be exact tuples")
    if decoded_g.labels.shape[0] != state.pair_count:
        raise CoupledPreimageProgramError("decoded G pair count differs from V2 predictor state")
    if decoded_g.realization_profile is None:
        raise CoupledPreimageProgramError("G must carry a realization profile for exact uint8 Y1")

    exact_y1 = np.ascontiguousarray(decoded_g.paint_rgb(), dtype=np.uint8)
    expected_y0 = _a._materialize_y0_from_controls(
        mode,
        state.source_pair_ids,
        anchored_controls,
        joint_controls,
        exact_y1,
        decoded_g.labels,
    )
    expected_chronological = np.ascontiguousarray(np.stack((expected_y0, exact_y1), axis=1), dtype=np.uint8)
    source_binding = _a._derive_source_binding(  # type: ignore[arg-type]
        state,
        decoded_g,
        exact_y1,
        expected_y0,
        expected_chronological,
    )
    lineage = _a._build_lineage(mode, source_binding, anchored_controls, joint_controls)
    program = CoupledPreimageProgramV1(
        mode=mode,
        source_binding=source_binding,
        anchored_controls=anchored_controls,
        joint_controls=joint_controls,
        lineage=lineage,
    )
    packet = program.to_packet()
    parsed = _a.parse_coupled_preimage_program(packet)
    decoded = decode_coupled_preimage_program_v2(
        packet,
        predictor_state=state,
        decoded_g=decoded_g,
    )
    return CompiledCoupledPreimageProgramV1(
        packet=packet,
        program=parsed,
        decoded=decoded,
        receipt=CoupledPreimageCompileReceiptV1(
            packet_bytes=len(packet),
            packet_sha256=_a._sha256(packet),
            source_binding_sha256=source_binding.binding_sha256,
            decoder_direct_source_sha256=_a.DECODER_DIRECT_SOURCE_SHA256,
            exact_y1_content_sha256=decoded.exact_y1_content_sha256,
            expected_y0_content_sha256=source_binding.expected_y0_content_sha256,
            chronological_content_sha256=decoded.chronological_content_sha256,
            mode=mode,
            source_pairs=source_binding.pair_count,
            active_control_rows=program.active_control_count,
            lineage_items=len(program.lineage),
        ),
    )


__all__ = [
    "SIGNAL_LOSS_FULL_REPAINT",
    "GCorrectionOwnershipV2",
    "TaskspacePredictorV2ConsumerSeamError",
    "TaskspacePredictorV2SignalBlocker",
    "apply_generative_taskspace_correction_v2",
    "compile_coupled_preimage_program_v2",
    "compile_generative_taskspace_correction_v2",
    "decode_coupled_preimage_program_v2",
    "derive_g_correction_ownership_v2",
    "parse_generative_taskspace_correction_v2",
    "require_signal_preserving_ep725_a_surface",
]
