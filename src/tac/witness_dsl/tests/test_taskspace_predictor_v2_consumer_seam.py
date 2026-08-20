# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from tac.optimization.direct_description_carrier_compose import (
    BoundaryCoefficientDelta,
    IslandShapeAtomV1,
    ReceiverRealizationProfileV1,
    TopologyEventV1,
)
from tac.witness_dsl.coupled_preimage_program import (
    CoupledPreimageMode,
    Frame1AnchoredY0FibreControlV1,
    compile_coupled_preimage_program,
)
from tac.witness_dsl.generative_taskspace_correction import (
    EncoderOnlyTeacherEvidenceV1,
    GenerativeCorrectionProgramV1,
)
from tac.witness_dsl.taskspace_predictor_state_v2 import (
    GTransportRequirementV2,
    NoTransportV2,
    TaskspacePredictorStateV2,
    TaskspacePredictorStateV2Error,
    V9Pose6TransportV2,
    g_transport_requirement,
)
from tac.witness_dsl.taskspace_predictor_v2_consumer_seam import (
    SIGNAL_LOSS_FULL_REPAINT,
    TaskspacePredictorV2SignalBlocker,
    apply_generative_taskspace_correction_v2,
    compile_coupled_preimage_program_v2,
    compile_generative_taskspace_correction_v2,
    decode_coupled_preimage_program_v2,
    derive_g_correction_ownership_v2,
    parse_generative_taskspace_correction_v2,
    require_signal_preserving_ep725_a_surface,
)


def _labels() -> np.ndarray:
    labels = np.full((2, 384, 512), 2, dtype=np.uint8)
    labels[:, 210:, :] = 0
    labels[:, 275:290, 230:270] = 3
    return labels


def _state(
    *,
    with_pose6: bool = False,
    pose6_values: np.ndarray | None = None,
) -> TaskspacePredictorStateV2:
    predictor_program = b"synthetic-counted-predictor-member"
    ids = (0, 1)
    if with_pose6:
        pose6 = (
            np.arange(12, dtype=np.int16).reshape(2, 6)
            if pose6_values is None
            else np.ascontiguousarray(pose6_values, dtype=np.int16)
        )
        counted_pose6 = np.ascontiguousarray(pose6, dtype=">i2").tobytes()
        transport = V9Pose6TransportV2(
            counted_payload=counted_pose6,
            pose6_codes=pose6,
            source_pair_ids=ids,
            predictor_program_sha256=hashlib.sha256(predictor_program).hexdigest(),
        )
    else:
        if pose6_values is not None:
            raise ValueError("pose6_values require with_pose6=True")
        transport = NoTransportV2()
    return TaskspacePredictorStateV2(
        predictor_program=predictor_program,
        predictor_renderer_sha256="b" * 64,
        source_archive_sha256="c" * 64,
        source_runtime_sha256="d" * 64,
        source_member_name="0.bin",
        source_pair_ids=ids,
        labels=_labels(),
        transport=transport,
    )


def _profile() -> ReceiverRealizationProfileV1:
    return ReceiverRealizationProfileV1(
        (
            (20, 80, 20),
            (240, 220, 40),
            (30, 30, 30),
            (220, 40, 40),
            (40, 80, 220),
        )
    )


def _teacher(state: TaskspacePredictorStateV2) -> EncoderOnlyTeacherEvidenceV1:
    target = state.labels.copy()
    target[0, 0, 0] = np.uint8((int(target[0, 0, 0]) + 1) % 5)
    return EncoderOnlyTeacherEvidenceV1(
        pbr1_sha256="1" * 64,
        pbr2_sha256="2" * 64,
        target_labels_sha256=hashlib.sha256(memoryview(target).cast("B")).hexdigest(),
        obligation_ir_sha256="4" * 64,
        oracle_evidence_sha256="5" * 64,
        dense_y_sha256="6" * 64,
        target_labels=target,
        teacher_event_count=1,
    )


def _local_program() -> GenerativeCorrectionProgramV1:
    return GenerativeCorrectionProgramV1(
        boundary_coefficients=(BoundaryCoefficientDelta(0, "Road", 0, 3.0),),
        realization_profile=_profile(),
    )


def test_none_state_executes_label_local_g_and_pose_independent_a() -> None:
    state = _state()
    compiled_g = compile_generative_taskspace_correction_v2(
        state,
        _local_program(),
        teacher_evidence=_teacher(state),
    )
    parsed_g = parse_generative_taskspace_correction_v2(compiled_g.packet, predictor_state=state)
    decoded_g = apply_generative_taskspace_correction_v2(compiled_g.packet, predictor_state=state)

    assert parsed_g.predictor_binding_sha256 == state.binding_sha256
    assert compiled_g.receipt.predictor_binding_sha256 == state.binding_sha256
    assert compiled_g.receipt.changed_cells > 0
    assert np.array_equal(decoded_g.labels, compiled_g.decoded.labels)
    assert not hasattr(state.transport, "pose6_codes")
    ownership = derive_g_correction_ownership_v2(state, decoded_g)
    assert np.array_equal(ownership.owned_mask, decoded_g.labels != state.labels)
    assert ownership.changed_cells == compiled_g.receipt.changed_cells
    assert ownership.owned_mask.flags.writeable is False
    assert ownership.correction_rgb_by_role == _profile().role_rgb_u8

    controls = (
        Frame1AnchoredY0FibreControlV1(0, 1, -2, (3, -4, 5)),
        Frame1AnchoredY0FibreControlV1(1, -1, 2, (-2, 4, 1)),
    )
    compiled_a = compile_coupled_preimage_program_v2(
        state,
        decoded_g,
        mode=CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE,
        anchored_controls=controls,
    )
    replay_a = decode_coupled_preimage_program_v2(
        compiled_a.packet,
        predictor_state=state,
        decoded_g=decoded_g,
    )

    assert compiled_a.program.source_binding.predictor_program_sha256 == state.predictor_program_sha256
    assert compiled_a.program.source_binding.predictor_labels_sha256 == state.labels_sha256
    assert np.array_equal(replay_a.chronological_frames, compiled_a.decoded.chronological_frames)
    assert replay_a.chronological_frames.shape == (2, 2, 384, 512, 3)

    with pytest.raises(TaskspacePredictorV2SignalBlocker) as blocked:
        require_signal_preserving_ep725_a_surface(state)
    assert blocked.value.code == SIGNAL_LOSS_FULL_REPAINT
    assert "semantic-to-camera overlay/R ABI" in blocked.value.detail


@pytest.mark.parametrize("atom_kind", ["event", "island"])
def test_none_refuses_nonzero_gain_g_before_receiver_mutation(
    monkeypatch: pytest.MonkeyPatch,
    atom_kind: str,
) -> None:
    state = _state()
    if atom_kind == "event":
        program = GenerativeCorrectionProgramV1(
            topology_events=(TopologyEventV1(0, "Lane", "birth", "box", 2, 1, 1, 3, 3, 1, 0),),
            realization_profile=_profile(),
        )
    else:
        program = GenerativeCorrectionProgramV1(
            island_shapes=(IslandShapeAtomV1(0, "birth", 2, 100, 140, 6, 12, 0, 0, 0, 8, 0, 1),),
            realization_profile=_profile(),
        )
    receiver_called = False

    def forbidden_receiver(*args: object, **kwargs: object) -> object:
        nonlocal receiver_called
        receiver_called = True
        raise AssertionError("receiver mutation must not run")

    monkeypatch.setattr(
        "tac.witness_dsl.generative_taskspace_correction._apply_program",
        forbidden_receiver,
    )
    with pytest.raises(TaskspacePredictorStateV2Error, match="refuses NONE before receiver mutation"):
        compile_generative_taskspace_correction_v2(
            state,
            program,
            teacher_evidence=_teacher(state),
        )
    assert receiver_called is False


@pytest.mark.parametrize("atom_kind", ["event", "island"])
@pytest.mark.parametrize("lifetime", [1, 2])
def test_zero_gain_event_and_island_are_pose_independent_under_none(
    atom_kind: str,
    lifetime: int,
) -> None:
    none_state = _state()
    if atom_kind == "event":
        program = GenerativeCorrectionProgramV1(
            topology_events=(TopologyEventV1(0, "Lane", "birth", "box", lifetime, 1, 1, 3, 4),),
            realization_profile=_profile(),
        )
    else:
        program = GenerativeCorrectionProgramV1(
            island_shapes=(IslandShapeAtomV1(0, "birth", lifetime, 100, 140, 6, 12, 0, 0, 0, 8),),
            realization_profile=_profile(),
        )

    assert g_transport_requirement(program) is GTransportRequirementV2.LABEL_LOCAL
    none_compiled = compile_generative_taskspace_correction_v2(
        none_state,
        program,
        teacher_evidence=_teacher(none_state),
    )
    first = apply_generative_taskspace_correction_v2(none_compiled.packet, predictor_state=none_state)
    second = apply_generative_taskspace_correction_v2(none_compiled.packet, predictor_state=none_state)
    assert np.array_equal(first.labels, second.labels)

    pose_a = np.zeros((2, 6), dtype=np.int16)
    pose_b = np.array([[300, -250, 0, 0, 0, 0], [-300, 250, 0, 0, 0, 0]], dtype=np.int16)
    decoded_with_pose = []
    for pose in (pose_a, pose_b):
        pose_state = _state(with_pose6=True, pose6_values=pose)
        compiled = compile_generative_taskspace_correction_v2(
            pose_state,
            program,
            teacher_evidence=_teacher(pose_state),
        )
        decoded_with_pose.append(compiled.decoded.labels)
    assert np.array_equal(first.labels, decoded_with_pose[0])
    assert np.array_equal(first.labels, decoded_with_pose[1])


def test_v9_pose6_state_executes_transport_dependent_g() -> None:
    state = _state(with_pose6=True)
    program = GenerativeCorrectionProgramV1(
        topology_events=(TopologyEventV1(0, "Lane", "birth", "box", 2, 160, 80, 168, 104, 16, 0),),
        realization_profile=_profile(),
    )
    compiled = compile_generative_taskspace_correction_v2(
        state,
        program,
        teacher_evidence=_teacher(state),
    )

    assert compiled.receipt.predictor_binding_sha256 == state.binding_sha256
    assert compiled.receipt.changed_cells > 0

    controls = (
        Frame1AnchoredY0FibreControlV1(0, 1, -2, (3, -4, 5)),
        Frame1AnchoredY0FibreControlV1(1, -1, 2, (-2, 4, 1)),
    )
    through_common_protocol = compile_coupled_preimage_program_v2(
        state,
        compiled.decoded,
        mode=CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE,
        anchored_controls=controls,
    )
    legacy_exact_v1 = compile_coupled_preimage_program(
        state.as_v1(),
        compiled.decoded,
        mode=CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE,
        anchored_controls=controls,
    )

    assert through_common_protocol.packet == legacy_exact_v1.packet
    assert np.array_equal(
        through_common_protocol.decoded.chronological_frames,
        legacy_exact_v1.decoded.chronological_frames,
    )
