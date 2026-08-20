# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import inspect
import json
import shutil
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from tac.witness_dsl.predictor_preserving_coupled_preimage import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CHANNELS,
    PredictorCameraPairSurfaceV1,
)
from tac.witness_dsl.taskspace_chronological_a3_encoder import XIP2Coder
from tac.witness_dsl.taskspace_counted_xip2_chronological_a3 import (
    CountedXIP2A3Interpretation,
    CountedXIP2A3Mode,
    CountedXIP2ChronologicalA3Error,
    parse_counted_xip2_chronological_a3_packet,
)
from tac.witness_dsl.taskspace_pass_conditional_a import (
    build_pass_conditional_a_surface,
    derive_pass_conditional_a_source_binding,
)
from tac.witness_dsl.taskspace_pass_semantic_g import compile_pass_semantic_g_envelope
from tac.witness_dsl.taskspace_predictor_state_v2 import NoTransportV2, TaskspacePredictorStateV2
from tac.witness_dsl.taskspace_quantized_xip2_inverse_solver import (
    FORMULATION_SCOPE,
    ROOT_INTEGRATION_BLOCKER,
    QuantizedXIP2EvaluationRequestV1,
    QuantizedXIP2InverseError,
    QuantizedXIP2ObjectiveAuthorityV1,
    QuantizedXIP2ObjectiveObservationV1,
    QuantizedXIP2SearchConfigV1,
    quantized_xip2_complete_score,
    quantized_xip2_strict_archive_ceiling,
    run_quantized_xip2_inverse_search,
)
from tac.witness_dsl.taskspace_same_class_realization_repair import (
    EncoderOnlyExactTargetLabelCustodyV1,
    SameClassRealizationRepairProgramV1,
    SameClassRealizationRepairRunV1,
    SameClassSemanticRoleV1,
)

PAIR_ID = 53
SEG_SHA = "a" * 64
SCORER_SHA = "b" * 64
TARGET_SHA = "c" * 64
MEASUREMENT_SHA = "d" * 64


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


@dataclass(frozen=True)
class _Fixture:
    predictor_packet: bytes
    predictor_surface: PredictorCameraPairSurfaceV1
    repair_program: SameClassRealizationRepairProgramV1
    target_custody: EncoderOnlyExactTargetLabelCustodyV1


@lru_cache(maxsize=12)
def _fixture(*, frame1_delta: int = 0, pair_count: int = 1) -> _Fixture:
    predictor_packet = b"LVLS1\x00g16-quantized-xip2-synthetic-predictor"
    pair_ids = tuple(range(PAIR_ID, PAIR_ID + pair_count))
    labels = np.zeros((pair_count, 384, 512), dtype=np.uint8)
    state = TaskspacePredictorStateV2(
        predictor_program=predictor_packet,
        predictor_renderer_sha256="1" * 64,
        source_archive_sha256="2" * 64,
        source_runtime_sha256="3" * 64,
        source_member_name="0.bin",
        source_pair_ids=pair_ids,
        labels=labels,
        transport=NoTransportV2(),
    )
    frames = np.empty((pair_count, 2, CAMERA_HEIGHT, CAMERA_WIDTH, CHANNELS), dtype=np.uint8)
    frames[:, 0] = (7, 17, 27)
    rows = np.arange(CAMERA_HEIGHT, dtype=np.uint16)[:, None]
    cols = np.arange(CAMERA_WIDTH, dtype=np.uint16)[None, :]
    for pair_index in range(pair_count):
        frames[pair_index, 1, :, :, 0] = ((rows + 3 * cols + frame1_delta + pair_index) % 256).astype(np.uint8)
        frames[pair_index, 1, :, :, 1] = ((2 * rows + cols + 19 + pair_index) % 256).astype(np.uint8)
        frames[pair_index, 1, :, :, 2] = ((rows + cols + 61 + pair_index) % 256).astype(np.uint8)
    predictor_surface = PredictorCameraPairSurfaceV1(
        predictor_state=state,
        chronological_camera_frames=frames,
        upstream_decode_receipt_sha256="4" * 64,
    )
    repair_program = SameClassRealizationRepairProgramV1(
        (
            SameClassRealizationRepairRunV1(
                source_pair_id=PAIR_ID,
                scorer_row=123,
                scorer_col_start=100,
                scorer_col_stop=108,
                semantic_class=0,
                semantic_role=SameClassSemanticRoleV1.ROAD,
                rgb_u8=(90, 100, 110),
            ),
        )
    )
    target_custody = EncoderOnlyExactTargetLabelCustodyV1(
        source_artifact_sha256="5" * 64,
        source_member_name="fresh-synthetic-target::labels",
        source_member_sha256="6" * 64,
        source_pair_ids=pair_ids,
        target_labels_sha256=_sha256(memoryview(labels).cast("B")),
        target_labels=labels,
    )
    return _Fixture(predictor_packet, predictor_surface, repair_program, target_custody)


@lru_cache(maxsize=12)
def _surface(*, frame1_delta: int = 0, with_g8: bool = False, pair_count: int = 1):
    fixture = _fixture(frame1_delta=frame1_delta, pair_count=pair_count)
    pass_g = compile_pass_semantic_g_envelope(
        predictor_section_payload=fixture.predictor_packet,
        predictor_surface=fixture.predictor_surface,
        repair_program=fixture.repair_program if with_g8 else None,
        target_custody=fixture.target_custody if with_g8 else None,
    ).decoded
    return build_pass_conditional_a_surface(
        predictor_surface=fixture.predictor_surface,
        pass_g=pass_g,
    )


def _config(
    *,
    seed: int = 17,
    q_levels: tuple[int, ...] = (3,),
    coders: tuple[XIP2Coder, ...] = (XIP2Coder.NONE,),
    interpretations: tuple[CountedXIP2A3Interpretation, ...] = (CountedXIP2A3Interpretation.CAMERA_THEN_R,),
    step_schedule: tuple[int, ...] = (1,),
    sweeps_per_step: int = 1,
    max_callback_evaluations: int | None = None,
) -> QuantizedXIP2SearchConfigV1:
    minimum = 2 + len(q_levels) * len(coders) * len(interpretations)
    return QuantizedXIP2SearchConfigV1(
        seed=seed,
        q_levels=q_levels,
        coders=coders,
        interpretations=interpretations,
        step_schedule=step_schedule,
        sweeps_per_step=sweeps_per_step,
        max_callback_evaluations=minimum if max_callback_evaluations is None else max_callback_evaluations,
        pitch=0.03125,
    )


def _initial_xi(*, zero: bool = False, pair_count: int = 1) -> np.ndarray:
    if zero:
        return np.zeros((pair_count, 6), dtype=np.float64)
    base = np.array([[0.016, -0.003, 0.002, 0.001, 0.006, -0.009]], dtype=np.float64)
    xi = np.repeat(base, pair_count, axis=0)
    xi[:, 0] += np.arange(pair_count, dtype=np.float64) * 0.001
    return xi


@dataclass
class _SyntheticObjective:
    nonlinear: bool = False
    active_worse: bool = False
    crash_on_call: int | None = None
    seg_drift_on_active: bool = False
    custody_drift_on_second: bool = False
    calls: list[QuantizedXIP2EvaluationRequestV1] | None = None

    def __post_init__(self) -> None:
        if self.calls is None:
            self.calls = []

    def __call__(self, request: QuantizedXIP2EvaluationRequestV1) -> QuantizedXIP2ObjectiveObservationV1:
        assert self.calls is not None
        self.calls.append(request)
        if self.crash_on_call == len(self.calls):
            raise RuntimeError("synthetic callback interruption")
        if request.mode is CountedXIP2A3Mode.PASS_P0_V1:
            pose_loss = 2.0
        elif request.mode is CountedXIP2A3Mode.COPY_CONDITIONAL_Y1_V1:
            pose_loss = 1.0
        else:
            assert request.q is not None
            if self.nonlinear:
                # f(q)=(q^2-1)^2 has zero local affine slope at q=0, yet q=+/-1 is exact.
                coordinate = int(request.q[0, 0])
                pose_loss = float((coordinate * coordinate - 1) ** 2)
            elif self.active_worse:
                pose_loss = 3.0
            else:
                pose_loss = 1.0
        seg_sha = "e" * 64 if self.seg_drift_on_active and request.mode is CountedXIP2A3Mode.XIP2_WARP_V1 else SEG_SHA
        measurement_sha = "f" * 64 if self.custody_drift_on_second and len(self.calls) == 2 else MEASUREMENT_SHA
        rebuilt_archive_bytes = 8_000 + request.compiled.receipt.packet_bytes
        return QuantizedXIP2ObjectiveObservationV1(
            per_pair_pose6_mse=(pose_loss,) * len(request.compiled.source_binding.source_pair_ids),
            d_seg=0.001,
            seg_prediction_sha256=seg_sha,
            archive_bytes=rebuilt_archive_bytes,
            archive_sha256=_sha256(b"synthetic-rebuilt-archive" + request.compiled.packet),
            authority=QuantizedXIP2ObjectiveAuthorityV1.SYNTHETIC_TEST_ONLY,
            frozen_scorer_sha256=SCORER_SHA,
            target_custody_sha256=TARGET_SHA,
            measurement_custody_sha256=measurement_sha,
        )


def _run(
    tmp_path: Path,
    *,
    objective: _SyntheticObjective,
    config: QuantizedXIP2SearchConfigV1,
    xi: np.ndarray | None = None,
    surface=None,
    name: str = "run",
    resume: bool = False,
):
    run_dir = tmp_path / name
    live_surface = _surface() if surface is None else surface
    return run_quantized_xip2_inverse_search(
        surface=live_surface,
        initial_xi=(_initial_xi(pair_count=len(live_surface.source_pair_ids)) if xi is None else xi),
        config=config,
        objective=objective,
        run_dir=run_dir,
        resume_from=run_dir if resume else None,
    )


def test_config_exact_types_closed_enums_and_public_signature() -> None:
    config = _config()
    assert config.config_sha256 == _sha256(
        json.dumps(
            config.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    )
    with pytest.raises(QuantizedXIP2InverseError, match="q_levels"):
        replace(config, q_levels=[3])  # type: ignore[arg-type]
    with pytest.raises(QuantizedXIP2InverseError, match="unique"):
        replace(config, coders=(XIP2Coder.NONE, XIP2Coder.NONE))
    with pytest.raises(QuantizedXIP2InverseError, match="enumerate"):
        replace(config, max_callback_evaluations=2)
    with pytest.raises(QuantizedXIP2InverseError, match="fp32-canonical"):
        replace(config, pitch=0.1)
    assert tuple(inspect.signature(run_quantized_xip2_inverse_search).parameters) == (
        "surface",
        "initial_xi",
        "config",
        "objective",
        "run_dir",
        "resume_from",
    )
    assert not hasattr(QuantizedXIP2EvaluationRequestV1, "as_dict")


def test_nonlinear_zero_slope_direct_q_search_finds_lower_complete_score(tmp_path: Path) -> None:
    objective = _SyntheticObjective(nonlinear=True)
    result = _run(
        tmp_path,
        objective=objective,
        config=_config(max_callback_evaluations=15),
        xi=_initial_xi(zero=True),
    )

    assert result.selected_control.mode is CountedXIP2A3Mode.COPY_CONDITIONAL_Y1_V1
    assert result.best_active_candidate.d_pose == 0.0
    assert abs(result.best_active_candidate.q[0][0]) == 1
    assert result.improved_control is True
    assert result.best_active_candidate.delta_s < 0.0
    assert result.best_active_candidate.strict_archive_ceiling is not None
    assert result.best_active_candidate.archive_bytes != result.best_active_candidate.packet_bytes
    assert result.receipt.direct_quantized_receiver_search is True
    assert result.receipt.affine_xi_pose_model_used is False
    assert result.receipt.through_g13_uint8_receiver is True
    assert all(
        request.q is None or np.array_equal(request.q, request.compiled.parsed_packet.transport.q_codes)
        for request in objective.calls or []
    )


def test_callback_budget_consumes_one_available_nonlinear_neighbor(tmp_path: Path) -> None:
    objective = _SyntheticObjective(nonlinear=True)
    result = _run(
        tmp_path,
        objective=objective,
        config=_config(seed=10, max_callback_evaluations=4),
        xi=_initial_xi(zero=True),
    )

    assert objective.calls is not None
    assert len(objective.calls) == 4
    assert result.receipt.callback_evaluations == 4
    assert result.stop_reason == "CALLBACK_BUDGET_EXHAUSTED"
    assert result.bounded_search_completed is False
    assert result.arm_results[0].search_completed is False
    assert result.arm_results[0].evaluated_neighbors == 1
    assert objective.calls[-1].q is not None
    assert objective.calls[-1].q[0, 0] == 1
    assert result.best_active_candidate.packet_sha256 == objective.calls[-1].packet_sha256
    assert result.best_active_candidate.d_pose == 0.0
    assert result.best_active_candidate.q[0][0] == 1
    assert result.improved_control is True


def test_complete_score_and_strict_ceiling_include_integral_boundary() -> None:
    score = quantized_xip2_complete_score(d_seg=0.001, d_pose=0.25, archive_bytes=12_345)
    assert score == 0.1 + np.sqrt(2.5) + 25.0 * 12_345 / 37_545_489
    integral_delta = -25.0 * 10.0 / 37_545_489
    assert (
        quantized_xip2_strict_archive_ceiling(
            reference_archive_bytes=100,
            delta_distortion=integral_delta,
        )
        == 109
    )
    assert (
        quantized_xip2_strict_archive_ceiling(
            reference_archive_bytes=100,
            delta_distortion=0.0,
        )
        is None
    )


def test_multiple_q_levels_all_four_coders_real_g13_and_delta_res_retained(tmp_path: Path) -> None:
    objective = _SyntheticObjective(active_worse=True)
    coders = (
        XIP2Coder.NONE,
        XIP2Coder.DELTA_AR,
        XIP2Coder.SPLINE_RESIDUAL,
        XIP2Coder.DELTA_RES,
    )
    config = _config(q_levels=(3, 7), coders=coders)
    result = _run(
        tmp_path,
        objective=objective,
        config=config,
        surface=_surface(pair_count=3),
    )
    active = [request for request in objective.calls or [] if request.mode is CountedXIP2A3Mode.XIP2_WARP_V1]

    assert len(active) == 8
    assert [(request.q_level, request.coder) for request in active] == [
        (q_level, coder) for q_level in config.q_levels for coder in config.coders
    ]
    assert {request.coder for request in active} == set(coders)
    assert XIP2Coder.DELTA_RES in {request.coder for request in active}
    assert len({request.packet_sha256 for request in active}) == 8
    assert all(request.compiled.receipt.strict_eof_crc_parse_reencode for request in active)
    assert result.improved_control is False
    assert result.receipt.bounded_search_negative is True
    assert result.receipt.verdict_scope == FORMULATION_SCOPE
    assert result.receipt.integration_blocker == ROOT_INTEGRATION_BLOCKER
    assert result.receipt.candidate_claim is False
    assert result.receipt.n2_claim is False


def test_both_g13_domains_are_nonaliasing_and_controls_are_first(tmp_path: Path) -> None:
    objective = _SyntheticObjective(active_worse=True)
    _run(
        tmp_path,
        objective=objective,
        config=_config(
            interpretations=(
                CountedXIP2A3Interpretation.CAMERA_THEN_R,
                CountedXIP2A3Interpretation.SCORER_THEN_FACTOR2,
            )
        ),
    )
    assert objective.calls is not None
    assert [request.mode for request in objective.calls[:2]] == [
        CountedXIP2A3Mode.PASS_P0_V1,
        CountedXIP2A3Mode.COPY_CONDITIONAL_Y1_V1,
    ]
    active = objective.calls[2:]
    assert [request.interpretation for request in active] == [
        CountedXIP2A3Interpretation.CAMERA_THEN_R,
        CountedXIP2A3Interpretation.SCORER_THEN_FACTOR2,
    ]
    assert active[0].compiled.parsed_packet.xip2_payload == active[1].compiled.parsed_packet.xip2_payload
    assert active[0].packet_sha256 != active[1].packet_sha256
    assert active[0].output_camera_y0_sha256 != active[1].output_camera_y0_sha256
    source_hash = derive_pass_conditional_a_source_binding(_surface()).conditional_camera_y1_sha256
    assert all(request.output_camera_y1_sha256 == source_hash for request in objective.calls)


def test_g13_packet_corruption_is_rejected_after_g16_compile(tmp_path: Path) -> None:
    objective = _SyntheticObjective(active_worse=True)
    _run(tmp_path, objective=objective, config=_config())
    assert objective.calls is not None
    active = objective.calls[2]
    packet = active.compiled.packet
    corrupt = packet[:-1] + bytes([packet[-1] ^ 1])
    with pytest.raises(CountedXIP2ChronologicalA3Error, match="CRC"):
        parse_counted_xip2_chronological_a3_packet(
            corrupt,
            expected_source_binding=derive_pass_conditional_a_source_binding(_surface()),
        )


def test_seg_and_live_y1_drift_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(QuantizedXIP2InverseError, match="Seg prediction"):
        _run(
            tmp_path,
            objective=_SyntheticObjective(seg_drift_on_active=True),
            config=_config(),
            name="seg-drift",
        )

    changed_surface = _surface(frame1_delta=9)
    changed_y1 = np.asarray(changed_surface.conditional_camera_y1).copy()
    changed_y1[0, 0, 0, 0] ^= 1
    object.__setattr__(changed_surface.pass_g.pass_semantic, "camera_y1", changed_y1)
    with pytest.raises(QuantizedXIP2InverseError, match=r"G13|control"):
        _run(
            tmp_path,
            objective=_SyntheticObjective(),
            config=_config(),
            surface=changed_surface,
            name="y1-drift",
        )


def test_atomic_terminal_resume_replays_without_callback_reevaluation(tmp_path: Path) -> None:
    config = _config()
    first_objective = _SyntheticObjective(active_worse=True)
    first = _run(tmp_path, objective=first_objective, config=config)

    never = _SyntheticObjective(crash_on_call=1)
    resumed = _run(tmp_path, objective=never, config=config, resume=True)
    assert never.calls == []
    assert resumed.to_canonical_json_bytes() == first.to_canonical_json_bytes()
    files = sorted(path.name for path in (tmp_path / "run").iterdir())
    assert files == [
        "manifest.json",
        "stage-00000001-evaluation.json",
        "stage-00000002-evaluation.json",
        "stage-00000003-evaluation.json",
        "stage-00000004-terminal.json",
    ]


def test_partial_resume_keeps_completed_callbacks_cached(tmp_path: Path) -> None:
    config = _config(max_callback_evaluations=5)
    interrupted = _SyntheticObjective(active_worse=True, crash_on_call=4)
    with pytest.raises(QuantizedXIP2InverseError, match="callback raised"):
        _run(tmp_path, objective=interrupted, config=config)
    assert interrupted.calls is not None
    completed_packets = {request.packet_sha256 for request in interrupted.calls[:3]}
    assert len(list((tmp_path / "run").glob("stage-*-evaluation.json"))) == 3

    resumed_objective = _SyntheticObjective(active_worse=True)
    result = _run(tmp_path, objective=resumed_objective, config=config, resume=True)
    assert resumed_objective.calls is not None
    assert completed_packets.isdisjoint(request.packet_sha256 for request in resumed_objective.calls)
    assert result.receipt.callback_evaluations == 5


def test_resume_rejects_foreign_config_source_chain_partial_and_duplicate(tmp_path: Path) -> None:
    config = _config()
    _run(tmp_path, objective=_SyntheticObjective(active_worse=True), config=config, name="base")
    base = tmp_path / "base"

    with pytest.raises(QuantizedXIP2InverseError, match="manifest differs"):
        _run(
            tmp_path,
            objective=_SyntheticObjective(),
            config=replace(config, seed=config.seed + 1),
            name="base",
            resume=True,
        )
    with pytest.raises(QuantizedXIP2InverseError, match="manifest differs"):
        _run(
            tmp_path,
            objective=_SyntheticObjective(),
            config=config,
            surface=_surface(frame1_delta=1),
            name="base",
            resume=True,
        )

    partial = tmp_path / "partial"
    shutil.copytree(base, partial)
    (partial / ".stage-00000005-evaluation.json.tmp").write_bytes(b"partial")
    with pytest.raises(QuantizedXIP2InverseError, match="unexpected"):
        _run(
            tmp_path,
            objective=_SyntheticObjective(),
            config=config,
            name="partial",
            resume=True,
        )

    corrupt = tmp_path / "corrupt"
    shutil.copytree(base, corrupt)
    stage2_path = corrupt / "stage-00000002-evaluation.json"
    stage2 = json.loads(stage2_path.read_text("ascii"))
    stage2["previous_stage_sha256"] = "f" * 64
    stage2_path.write_text(json.dumps(stage2, sort_keys=True, separators=(",", ":")), encoding="ascii")
    with pytest.raises(QuantizedXIP2InverseError, match="chain"):
        _run(
            tmp_path,
            objective=_SyntheticObjective(),
            config=config,
            name="corrupt",
            resume=True,
        )

    duplicate = tmp_path / "duplicate"
    interrupted = _SyntheticObjective(active_worse=True, crash_on_call=4)
    with pytest.raises(QuantizedXIP2InverseError):
        _run(
            tmp_path,
            objective=interrupted,
            config=_config(max_callback_evaluations=5),
            name="duplicate",
        )
    stage3_path = duplicate / "stage-00000003-evaluation.json"
    stage3_bytes = stage3_path.read_bytes()
    stage3 = json.loads(stage3_bytes)
    stage3["stage_index"] = 4
    stage3["previous_stage_sha256"] = _sha256(stage3_bytes)
    stage3["candidate"]["callback_evaluation_index"] = 4
    stage3["candidate"]["archive_sha256"] = "9" * 64
    (duplicate / "stage-00000004-evaluation.json").write_text(
        json.dumps(stage3, sort_keys=True, separators=(",", ":")),
        encoding="ascii",
    )
    with pytest.raises(QuantizedXIP2InverseError, match="duplicate packet"):
        _run(
            tmp_path,
            objective=_SyntheticObjective(),
            config=_config(max_callback_evaluations=5),
            name="duplicate",
            resume=True,
        )


def test_seed_changes_coordinate_order_without_manufacturing_improvement(tmp_path: Path) -> None:
    first = _run(
        tmp_path,
        objective=_SyntheticObjective(active_worse=True),
        config=_config(seed=101),
        name="seed-a",
    )
    second = _run(
        tmp_path,
        objective=_SyntheticObjective(active_worse=True),
        config=_config(seed=202),
        name="seed-b",
    )
    assert first.arm_results[0].arm.coordinate_order_sha256 != second.arm_results[0].arm.coordinate_order_sha256
    assert first.improved_control is False
    assert second.improved_control is False
    assert first.receipt.bounded_search_negative is True
    assert second.receipt.bounded_search_negative is True


def test_checkpoint_and_receipt_are_scalar_hash_q_only_without_dense_or_public_payload(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        objective=_SyntheticObjective(active_worse=True),
        config=_config(),
    )
    for path in (tmp_path / "run").glob("*.json"):
        payload = path.read_text("ascii")
        assert "chronological_camera_frames" not in payload
        assert "target_labels" not in payload
        assert "scorer_weights" not in payload
        assert "pose_target" not in payload
        assert "public_payload" not in payload
    assert result.receipt.public_or_historical_payload_used is False
    assert result.receipt.dense_target_serialized is False
    assert result.receipt.scorer_or_weights_serialized is False
    assert result.receipt.actual_scorer_invoked_by_core is False
    assert result.receipt.exact_score_claim is False
    assert result.receipt.promotion_eligible is False


@pytest.mark.parametrize(
    "failure",
    ("wrong-type", "pair-count", "bad-hash", "nonfinite", "negative-bytes"),
)
def test_callback_pair_count_hash_type_nonfinite_and_negative_bytes_fail(
    tmp_path: Path,
    failure: str,
) -> None:
    class InvalidCallback:
        def __call__(self, request: QuantizedXIP2EvaluationRequestV1):
            if failure == "wrong-type":
                return object()
            observation = QuantizedXIP2ObjectiveObservationV1(
                per_pair_pose6_mse=(1.0, 1.0) if failure == "pair-count" else (1.0,),
                d_seg=0.001,
                seg_prediction_sha256=SEG_SHA,
                archive_bytes=8_000,
                archive_sha256="8" * 64,
                authority=QuantizedXIP2ObjectiveAuthorityV1.SYNTHETIC_TEST_ONLY,
                frozen_scorer_sha256=SCORER_SHA,
                target_custody_sha256=TARGET_SHA,
                measurement_custody_sha256=MEASUREMENT_SHA,
            )
            if failure == "bad-hash":
                object.__setattr__(observation, "archive_sha256", "bad")
            elif failure == "nonfinite":
                object.__setattr__(observation, "d_seg", float("inf"))
            elif failure == "negative-bytes":
                object.__setattr__(observation, "archive_bytes", -1)
            return observation

    with pytest.raises(QuantizedXIP2InverseError):
        run_quantized_xip2_inverse_search(
            surface=_surface(),
            initial_xi=_initial_xi(),
            config=_config(),
            objective=InvalidCallback(),
            run_dir=tmp_path / failure,
            resume_from=None,
        )


def test_callback_custody_change_fails_before_second_stage(tmp_path: Path) -> None:
    with pytest.raises(QuantizedXIP2InverseError, match="custody changed"):
        _run(
            tmp_path,
            objective=_SyntheticObjective(custody_drift_on_second=True),
            config=_config(),
        )
    assert len(list((tmp_path / "run").glob("stage-*-evaluation.json"))) == 1
