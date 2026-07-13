from __future__ import annotations

import pytest

from tac.scorer_surrogate.onpolicy_matched_verdict import (
    ArmEvidence,
    CommonStepSchedule,
    DeterministicRepeatNoiseFloor,
    EvidenceStatus,
    ExactMetricAuthority,
    MetricObservation,
    MetricTrace,
    RegimeEvidence,
    VerdictKind,
    adjudicate_matched_windows,
    aggregate_isolated_timings,
    derive_deterministic_repeat_noise_floor,
)


def _schedule() -> CommonStepSchedule:
    return CommonStepSchedule(
        step_indices=(0, 1, 2),
        control_values=(0.0, 0.25, 0.25),
        control_name="predeclared_normalized_step_norm",
        derivation="copied from the source-bound matched-window receipt",
    )


def _authority(*, complete: bool = True, evidence: str = "a" * 64) -> ExactMetricAuthority:
    return ExactMetricAuthority(
        ce_exact_teacher_through_r=complete,
        d_seg_exact_argmax_through_r=complete,
        d_pose_exact_frozen_posenet_through_r=complete,
        axis="[macOS-CPU advisory training-gradient]",
        evidence_sha256=evidence,
    )


def _trace(
    schedule: CommonStepSchedule,
    *,
    d_seg: tuple[float, ...] = (0.30, 0.20, 0.10),
    ce: tuple[float, ...] = (0.60, 0.40, 0.20),
    d_pose: tuple[float, ...] = (0.09, 0.08, 0.07),
    steps: tuple[int, ...] | None = None,
) -> MetricTrace:
    step_indices = schedule.step_indices if steps is None else steps
    count = len(step_indices)
    return MetricTrace(
        step_indices=step_indices,
        d_seg=d_seg[:count],
        ce=ce[:count],
        d_pose=d_pose[:count],
        common_step_schedule_sha256=schedule.sha256,
    )


def _observation(
    trace: MetricTrace,
    *,
    complete: bool = True,
    evidence: str = "a" * 64,
) -> MetricObservation:
    return MetricObservation(trace=trace, authority=_authority(complete=complete, evidence=evidence))


def _noise_floor(schedule: CommonStepSchedule) -> DeterministicRepeatNoiseFloor:
    trace = _trace(schedule)
    return derive_deterministic_repeat_noise_floor(
        (
            _observation(trace, evidence="1" * 64),
            _observation(trace, evidence="2" * 64),
        ),
        common_step_schedule=schedule,
    )


def _regime(
    schedule: CommonStepSchedule,
    *,
    exact: MetricTrace | None = None,
    target: MetricTrace | None = None,
    exact_complete: bool = True,
    target_status: EvidenceStatus = EvidenceStatus.MEASURED,
) -> RegimeEvidence:
    exact_trace = exact or _trace(schedule)
    target_trace = target or _trace(schedule)
    target_observation = _observation(target_trace, evidence="c" * 64)
    return RegimeEvidence(
        regime="early",
        exact_control=ArmEvidence(
            arm_id="K1_exact",
            status=EvidenceStatus.MEASURED,
            observation=_observation(
                exact_trace,
                complete=exact_complete,
                evidence="b" * 64,
            ),
        ),
        surrogate_target=ArmEvidence(
            arm_id="K20_surrogate",
            status=target_status,
            observation=None if target_status is EvidenceStatus.BLOCKED else target_observation,
            status_reason=(
                "K20 recurring anchor fit failed before the matched window completed"
                if target_status is EvidenceStatus.BLOCKED
                else None
            ),
            status_evidence_sha256=("d" * 64 if target_status is EvidenceStatus.BLOCKED else None),
        ),
    )


def test_independently_descending_but_drifting_traces_are_no_go() -> None:
    schedule = _schedule()
    exact = _trace(
        schedule,
        d_seg=(0.30, 0.25, 0.20),
        ce=(0.60, 0.50, 0.40),
        d_pose=(0.09, 0.08, 0.07),
    )
    target = _trace(
        schedule,
        d_seg=(0.30, 0.20, 0.10),
        ce=(0.60, 0.45, 0.30),
        d_pose=(0.09, 0.07, 0.05),
    )
    verdict = adjudicate_matched_windows(
        requested_regimes=("early",),
        regime_evidence=(_regime(schedule, exact=exact, target=target),),
        common_step_schedule=schedule,
        deterministic_repeat_noise_floor=_noise_floor(schedule),
    )
    row = verdict.regime_verdicts[0]
    assert verdict.verdict is VerdictKind.NO_GO
    assert row.metric_comparisons["d_seg"]["exact_trace_nonworsening"]
    assert row.metric_comparisons["d_seg"]["target_trace_nonworsening"]
    assert not row.metric_comparisons["d_seg"]["within_repeat_noise_floor_at_every_step"]


def test_blocked_k20_is_formulation_no_go_after_valid_exact_control() -> None:
    schedule = _schedule()
    verdict = adjudicate_matched_windows(
        requested_regimes=("early",),
        regime_evidence=(
            _regime(schedule, target_status=EvidenceStatus.BLOCKED),
        ),
        common_step_schedule=schedule,
        deterministic_repeat_noise_floor=_noise_floor(schedule),
    )
    assert verdict.verdict is VerdictKind.NO_GO
    assert "formulation" in verdict.verdict_scope
    assert "family/paradigm" in verdict.verdict_scope
    assert verdict.regime_verdicts[0].target_status is EvidenceStatus.BLOCKED


def test_missing_exact_authority_is_needs_more() -> None:
    schedule = _schedule()
    verdict = adjudicate_matched_windows(
        requested_regimes=("early",),
        regime_evidence=(_regime(schedule, exact_complete=False),),
        common_step_schedule=schedule,
        deterministic_repeat_noise_floor=_noise_floor(schedule),
    )
    assert verdict.verdict is VerdictKind.NEEDS_MORE
    assert "authority" in verdict.regime_verdicts[0].reason


def test_exact_trace_equality_is_go_and_never_score_authority() -> None:
    schedule = _schedule()
    verdict = adjudicate_matched_windows(
        requested_regimes=("early",),
        regime_evidence=(_regime(schedule),),
        common_step_schedule=schedule,
        deterministic_repeat_noise_floor=_noise_floor(schedule),
    )
    receipt = verdict.to_dict()
    assert verdict.verdict is VerdictKind.GO
    assert receipt["score_claim"] is False
    assert receipt["promotion_eligible"] is False
    assert receipt["pointer_expected_unmoved"] is True
    assert "not archive/evaluator score authority" in receipt["authority_statement"]


def test_identical_repeats_derive_zero_noise_tolerance() -> None:
    schedule = _schedule()
    trace = _trace(schedule)
    floor = derive_deterministic_repeat_noise_floor(
        (
            _observation(trace, evidence="3" * 64),
            _observation(trace, evidence="4" * 64),
        ),
        common_step_schedule=schedule,
    )
    assert floor.d_seg == 0.0
    assert floor.ce == 0.0
    assert floor.d_pose == 0.0
    assert floor.repeat_count == 2


def test_valid_exact_terminal_floor_uses_identical_matched_prefix() -> None:
    schedule = _schedule()
    prefix = _trace(schedule, steps=(0, 1))
    row = RegimeEvidence(
        regime="early",
        exact_control=ArmEvidence(
            arm_id="K1_exact",
            status=EvidenceStatus.VALID_TERMINAL_FLOOR,
            observation=_observation(prefix, evidence="5" * 64),
            status_reason="exact teacher reached a measured bit-identical renderer floor",
            status_evidence_sha256="8" * 64,
        ),
        surrogate_target=ArmEvidence(
            arm_id="K20_surrogate",
            status=EvidenceStatus.MEASURED,
            observation=_observation(prefix, evidence="6" * 64),
        ),
    )
    verdict = adjudicate_matched_windows(
        requested_regimes=("early",),
        regime_evidence=(row,),
        common_step_schedule=schedule,
        deterministic_repeat_noise_floor=_noise_floor(schedule),
    )
    assert verdict.verdict is VerdictKind.GO
    assert verdict.regime_verdicts[0].exact_control_valid_terminal_floor


def test_every_requested_regime_is_required() -> None:
    schedule = _schedule()
    verdict = adjudicate_matched_windows(
        requested_regimes=("early", "boundary"),
        regime_evidence=(_regime(schedule),),
        common_step_schedule=schedule,
        deterministic_repeat_noise_floor=_noise_floor(schedule),
    )
    assert verdict.verdict is VerdictKind.NEEDS_MORE
    assert verdict.regime_verdicts[1].regime == "boundary"
    assert verdict.regime_verdicts[1].exact_control_status is EvidenceStatus.MISSING


def test_timing_aggregation_keeps_components_and_whole_windows_separate() -> None:
    schedule = _schedule()
    timings = aggregate_isolated_timings(
        common_step_schedule=schedule,
        exact_schedule_sha256=schedule.sha256,
        surrogate_schedule_sha256=schedule.sha256,
        exact_forward_only=(1.0, 3.0),
        exact_costate_forward_backward=(4.0, 6.0),
        anchor_fit=(0.5, 1.5),
        surrogate_inference=(0.1, 0.3),
        renderer_vjp_exact_control=(2.0, 2.0),
        renderer_vjp_surrogate_target=(2.5, 2.5),
        whole_matched_window_exact_control=(20.0, 24.0),
        whole_matched_window_surrogate_target=(10.0, 12.0),
    )
    assert timings["exact_forward_only"]["mean_seconds"] == 2.0
    assert timings["exact_costate_forward_backward"]["mean_seconds"] == 5.0
    assert timings["anchor_fit"]["mean_seconds"] == 1.0
    assert timings["surrogate_inference"]["mean_seconds"] == pytest.approx(0.2)
    assert timings["renderer_vjp"]["exact_control"]["mean_seconds"] == 2.0
    assert timings["whole_matched_window"]["matched_speedup_exact_over_surrogate"] == 2.0
    assert timings["complete_per_step_timer_sums_used_for_window"] is True
    assert timings["isolated_component_sums_used_for_window"] is False
    assert timings["control_law_conflation"] is False


def test_repeat_receipt_requires_measured_repeat_count() -> None:
    schedule = _schedule()
    with pytest.raises(ValueError, match="repeat_count"):
        DeterministicRepeatNoiseFloor.from_repeat_receipt(
            d_seg=0.0,
            ce=0.0,
            d_pose=0.0,
            repeat_count=1,
            common_step_schedule_sha256=schedule.sha256,
            repeat_receipt_sha256="7" * 64,
        )
