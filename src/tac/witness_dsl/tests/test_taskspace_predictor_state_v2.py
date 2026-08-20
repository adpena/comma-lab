# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from tac.boundary_math.xi_pose_coder import quantize_xi, serialize_xi_payload
from tac.optimization.direct_description_carrier_compose import (
    BoundaryCoefficientDelta,
    IslandShapeAtomV1,
    MovableWorldsheetTrackV1,
    TopologyEventV1,
)
from tac.witness_dsl.generative_taskspace_correction import (
    GenerativeCorrectionProgramV1,
    PredictorSemanticStateV1,
)
from tac.witness_dsl.taskspace_predictor_state_v2 import (
    NONE_TRANSPORT_CONTENT_SHA256,
    GTransportRequirementV2,
    NoTransportV2,
    SE3XiTransportV2,
    TaskspacePredictorStateV2,
    TaskspacePredictorStateV2Error,
    TransportKindV2,
    V9Pose6TransportV2,
    g_transport_requirement,
    require_g_transport_admission,
)


def _program_bytes() -> bytes:
    return b"LVLS1\x00counted-predictor-fixture"


def _labels(pair_count: int = 2) -> np.ndarray:
    labels = np.full((pair_count, 384, 512), 2, dtype=np.uint8)
    labels[:, 230:, :] = 0
    labels[:, 180:190, 200:260] = 1
    return labels


def _none_state(*, labels: np.ndarray | None = None) -> TaskspacePredictorStateV2:
    return TaskspacePredictorStateV2(
        predictor_program=_program_bytes(),
        predictor_renderer_sha256="b" * 64,
        source_archive_sha256="c" * 64,
        source_runtime_sha256="d" * 64,
        source_member_name="0.bin",
        source_pair_ids=(0, 1),
        labels=_labels() if labels is None else labels,
        transport=NoTransportV2(),
    )


def _v9_transport() -> V9Pose6TransportV2:
    pose6 = np.arange(12, dtype=np.int16).reshape(2, 6)
    counted = np.ascontiguousarray(pose6, dtype=">i2").tobytes()
    return V9Pose6TransportV2(
        counted_payload=counted,
        pose6_codes=pose6,
        source_pair_ids=(0, 1),
        predictor_program_sha256=hashlib.sha256(_program_bytes()).hexdigest(),
    )


def _v9_state() -> TaskspacePredictorStateV2:
    return replace(_none_state(), transport=_v9_transport())


def _xi_transport() -> SE3XiTransportV2:
    xi = np.array(
        [
            [0.01, 0.0, 0.02, 0.001, -0.002, 0.003],
            [0.02, 0.001, 0.025, 0.0015, -0.001, 0.004],
        ],
        dtype=np.float64,
    )
    q, scales = quantize_xi(xi)
    return SE3XiTransportV2(
        counted_payload=serialize_xi_payload(q, scales, coder="delta_ar"),
        source_pair_ids=(0, 1),
        predictor_program_sha256=hashlib.sha256(_program_bytes()).hexdigest(),
    )


def _xi_state() -> TaskspacePredictorStateV2:
    return replace(_none_state(), transport=_xi_transport())


def test_none_is_canonical_absence_not_zero_filled_pose6() -> None:
    state = _none_state()

    assert state.transport.kind is TransportKindV2.NONE
    assert state.transport.counted_bytes == 0
    assert state.transport.counted_content_sha256 == NONE_TRANSPORT_CONTENT_SHA256
    assert not hasattr(state.transport, "pose6_codes")
    with pytest.raises(TaskspacePredictorStateV2Error, match="cannot be projected"):
        state.as_v1()


def test_state_copies_decoder_labels_and_binds_every_identity() -> None:
    labels = _labels()
    state = _none_state(labels=labels)
    baseline = state.binding_sha256

    labels[0, 0, 0] = 4
    assert state.labels[0, 0, 0] == 2
    assert state.labels.flags.writeable is False
    with pytest.raises(ValueError):
        state.labels[0, 0, 0] = 3

    mutated_labels = state.labels.copy()
    mutated_labels[0, 0, 0] = 4
    mutations = (
        replace(state, predictor_program=state.predictor_program + b"x"),
        replace(state, predictor_renderer_sha256="1" * 64),
        replace(state, source_archive_sha256="2" * 64),
        replace(state, source_runtime_sha256="3" * 64),
        replace(state, labels=mutated_labels),
        replace(state, transport=_xi_transport()),
    )
    assert all(candidate.binding_sha256 != baseline for candidate in mutations)
    assert state.semantic_binding_sha256 == replace(state, transport=_xi_transport()).semantic_binding_sha256


def test_source_order_and_five_class_universe_fail_closed() -> None:
    with pytest.raises(TaskspacePredictorStateV2Error, match="ordered contiguous"):
        replace(_none_state(), source_pair_ids=(0, 2))
    labels = _labels()
    labels[0, 0, 0] = 5
    with pytest.raises(TaskspacePredictorStateV2Error, match="five-class"):
        _none_state(labels=labels)
    with pytest.raises(TaskspacePredictorStateV2Error, match="canonical sole member"):
        replace(_none_state(), source_member_name="predictor.bin")


def test_v9_transport_requires_exact_counted_content_and_source_binding() -> None:
    transport = _v9_transport()
    assert transport.counted_bytes == 24
    assert transport.pose6_codes.dtype == np.dtype(">i2")
    assert transport.pose6_codes.flags.writeable is False

    with pytest.raises(TaskspacePredictorStateV2Error, match="do not decode"):
        replace(transport, counted_payload=transport.counted_payload[:-1] + b"x")
    with pytest.raises(TaskspacePredictorStateV2Error, match="different predictor"):
        replace(_none_state(), transport=replace(transport, predictor_program_sha256="f" * 64))


def test_v1_round_trip_is_adapter_only_and_byte_identical() -> None:
    program = _program_bytes()
    pose6 = np.arange(12, dtype=np.int16).reshape(2, 6)
    v1 = PredictorSemanticStateV1(
        predictor_program_sha256=hashlib.sha256(program).hexdigest(),
        predictor_renderer_sha256="b" * 64,
        source_pair_ids=(0, 1),
        labels=_labels(),
        pose6_codes=pose6,
    )

    lifted = TaskspacePredictorStateV2.from_v1(
        v1,
        predictor_program=program,
        counted_pose6_payload=memoryview(v1.pose6_codes).cast("B").tobytes(),
        source_archive_sha256="c" * 64,
        source_runtime_sha256="d" * 64,
    )
    restored = lifted.as_v1()

    assert restored.binding_sha256 == v1.binding_sha256
    assert np.array_equal(restored.labels, v1.labels)
    assert np.array_equal(restored.pose6_codes, v1.pose6_codes)
    with pytest.raises(TaskspacePredictorStateV2Error, match="differs"):
        TaskspacePredictorStateV2.from_v1(
            v1,
            predictor_program=program + b"mutation",
            counted_pose6_payload=memoryview(v1.pose6_codes).cast("B").tobytes(),
            source_archive_sha256="c" * 64,
            source_runtime_sha256="d" * 64,
        )
    with pytest.raises(TaskspacePredictorStateV2Error, match="do not decode"):
        TaskspacePredictorStateV2.from_v1(
            v1,
            predictor_program=program,
            counted_pose6_payload=b"\x00" * v1.pose6_codes.nbytes,
            source_archive_sha256="c" * 64,
            source_runtime_sha256="d" * 64,
        )


@pytest.mark.parametrize("coder", ["none", "delta_ar", "delta_res"])
def test_se3_xi_requires_exact_xip2_eof_shape_and_source_binding(coder: str) -> None:
    xi = np.array(
        [[0.01, 0.0, 0.02, 0.001, -0.002, 0.003], [0.02, 0.0, 0.03, 0.002, -0.001, 0.004]],
        dtype=np.float64,
    )
    q, scales = quantize_xi(xi)
    payload = serialize_xi_payload(q, scales, coder=coder)
    transport = SE3XiTransportV2(
        counted_payload=payload,
        source_pair_ids=(0, 1),
        predictor_program_sha256=hashlib.sha256(_program_bytes()).hexdigest(),
    )

    assert transport.kind is TransportKindV2.SE3_XI
    assert transport.q_codes.shape == (2, 6)
    assert transport.xi.dtype == np.float64
    assert transport.q_codes.flags.writeable is False
    with pytest.raises(TaskspacePredictorStateV2Error, match="trailing"):
        replace(transport, counted_payload=payload + b"latent")
    with pytest.raises(TaskspacePredictorStateV2Error, match="exact XIP2"):
        replace(transport, counted_payload=b"INR-latent-row")


def test_transport_source_population_must_match_predictor() -> None:
    transport = replace(_xi_transport(), source_pair_ids=(1, 2))
    with pytest.raises(TaskspacePredictorStateV2Error, match="pair IDs differ"):
        replace(_none_state(), transport=transport)


def test_g_dependency_is_gain_derived_and_worldsheets_remain_conservative() -> None:
    local = GenerativeCorrectionProgramV1(
        boundary_coefficients=(BoundaryCoefficientDelta(0, "Road", 0, 1.0),),
        topology_events=(TopologyEventV1(0, "Lane", "birth", "box", 2, 10, 10, 20, 20),),
        island_shapes=(IslandShapeAtomV1(0, "birth", 2, 100, 140, 6, 12, 0, 0, 0, 8),),
    )
    advected = GenerativeCorrectionProgramV1(
        topology_events=(TopologyEventV1(0, "Lane", "birth", "box", 2, 10, 10, 20, 20, 1, 0),)
    )
    worldsheet = GenerativeCorrectionProgramV1(
        worldsheet_tracks=(MovableWorldsheetTrackV1(7, 0, 2, 120, 140, 5, 10, 0, 0, 0, 0, 8, 4),)
    )

    assert g_transport_requirement(local) is GTransportRequirementV2.LABEL_LOCAL
    assert require_g_transport_admission(_none_state(), local) is GTransportRequirementV2.LABEL_LOCAL
    assert g_transport_requirement(advected) is GTransportRequirementV2.POSE6_ADVECTED
    assert g_transport_requirement(worldsheet) is GTransportRequirementV2.POSE6_ADVECTED
    before = _none_state().labels.copy()
    with pytest.raises(TaskspacePredictorStateV2Error, match="refuses NONE before"):
        require_g_transport_admission(_none_state(), advected)
    with pytest.raises(TaskspacePredictorStateV2Error, match="refuses NONE before"):
        require_g_transport_admission(_none_state(), worldsheet)
    assert np.array_equal(_none_state().labels, before)


def test_g_dependency_accepts_legacy_pose6_and_refuses_xi_until_parity_adapter() -> None:
    advected = GenerativeCorrectionProgramV1(
        topology_events=(TopologyEventV1(0, "Lane", "birth", "box", 2, 10, 10, 20, 20, 1, 0),)
    )

    assert require_g_transport_admission(_v9_state(), advected) is GTransportRequirementV2.POSE6_ADVECTED
    with pytest.raises(TaskspacePredictorStateV2Error, match="parity adapter is not implemented"):
        require_g_transport_admission(_xi_state(), advected)
