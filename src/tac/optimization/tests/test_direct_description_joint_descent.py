# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.direct_description_entropy_priced_member import (
    rfc8785_canonicalize,
)
from tac.optimization.direct_description_g1_worldsheet import (
    encode_g1_movable_worldsheet,
    encode_lifted_g1_movable_worldsheet,
    lift_g1_movable_worldsheet,
)
from tac.optimization.direct_description_joint_descent import (
    EXPECTED_PROGRAM_SHA256,
    J3_PROGRAM_SHA256,
    J5_PROGRAM_SHA256,
    J7_PROGRAM_SHA256,
    LEGACY_PROGRAM_SHA256,
    AdamStateV1,
    DirectDescriptionJointDescentTypedConfigV1,
    FullRunScheduleV1,
    PoseFinishEngageStateV1,
    classify_cumulative_fire_gate,
    classify_governed_stage_exit,
    classify_memory_preflight,
    classify_realized_stage_verdict,
    clipped_adam_step,
    compile_parameterized_archive,
    exact_final_target_gate,
    initial_adam_state,
    lift_v15_archive,
    linear_rewarmup_factor,
    load_stage_checkpoint,
    opening_candidate_gradient,
    parameter_group_indices,
    save_stage_checkpoint,
)
from tac.optimization.direct_description_minimizer import DirectDescriptionError

REPO = Path(__file__).resolve().parents[4]
TICKET = REPO / ".omx/research/configs/ddm_j1_366_joint_descent_witness_program_20260723.json"
FULL_RUN_TICKET = REPO / ".omx/research/configs/ddm_j4_366_joint_descent_warm_start_reform_20260723.json"
J3_TICKET = REPO / ".omx/research/configs/ddm_j3_366_joint_descent_witness_program_20260723.json"
J5_TICKET = REPO / ".omx/research/configs/ddm_j5_366_realized_acceptance_warmstart_20260723.json"


def test_g1_lift_preserves_exact_stream_and_explicit_lifecycle() -> None:
    labels = np.zeros((5, 384, 512), dtype=np.int64)
    labels[1, 40:48, 60:72] = 3
    labels[2, 41:49, 62:74] = 3
    labels[3, 42:50, 64:76] = 3
    payload, _ = encode_g1_movable_worldsheet(labels)

    lift = lift_g1_movable_worldsheet(payload)

    assert encode_lifted_g1_movable_worldsheet(lift) == payload
    assert len(lift.tracks) == 1
    assert (lift.tracks[0].birth_pair, lift.tracks[0].death_pair_exclusive) == (1, 4)
    assert len(lift.tracks[0].knot_indices) == 3
    assert all(knot.template_ref for knot in lift.knots)
    assert all(np.isfinite(knot.aspect_log) and np.isfinite(knot.rotation_radians) for knot in lift.knots)


def test_hash_sealed_ticket_compiles_to_typed_config() -> None:
    config = DirectDescriptionJointDescentTypedConfigV1.from_ticket(TICKET)
    assert config.dsl_compile_hash == LEGACY_PROGRAM_SHA256
    assert config.num_pairs == 600
    assert config.seed == 0
    assert config.custom_grouped_backward_required is True
    assert config.fused_r_required is True
    assert config.score_claim is False
    assert config.research_only is True


def test_typed_config_refuses_semantic_ticket_mutation(tmp_path: Path) -> None:
    ticket = json.loads(TICKET.read_bytes())
    ticket["semantic_program"]["seed"] = 1
    mutated = tmp_path / "mutated_ticket.json"
    mutated.write_text(json.dumps(ticket), encoding="utf-8")
    with pytest.raises(DirectDescriptionError, match="DSL hash mismatch"):
        DirectDescriptionJointDescentTypedConfigV1.from_ticket(mutated)


def test_resealed_full_run_ticket_compiles_exact_schedule() -> None:
    config = DirectDescriptionJointDescentTypedConfigV1.from_ticket(FULL_RUN_TICKET)
    schedule = config.full_run_schedule
    assert config.dsl_compile_hash == EXPECTED_PROGRAM_SHA256
    assert config.typed_config_hash() == "ca13e172e195731026de80ecaa0dff8ea307c1212fcc13ac4d17c6285ee9d7ab"
    assert schedule is not None
    assert schedule.train_batch == 4
    assert schedule.warm_start_pair == 447
    assert schedule.warm_start_steps == 4
    assert schedule.warm_start_reform.adam_beta2 == 0.999
    assert schedule.warm_start_reform.lr_rewarmup_steps == 2000
    assert schedule.warm_start_reform.lr_rewarmup_floor == 0.1
    assert schedule.warm_start_reform.maximum_continuous_update_quantum_fraction == 0.25
    assert schedule.warm_start_reform.frozen_groups_until_first_admission == ("shared_template_dof",)
    assert schedule.checkpoint_interval_steps == 37
    assert all(stage.verdict_interval_steps == 50 for stage in schedule.stages)
    assert sum(stage.maximum_steps for stage in schedule.stages) == 450


def test_historical_j3_ticket_remains_typed_and_hash_compatible() -> None:
    config = DirectDescriptionJointDescentTypedConfigV1.from_ticket(J3_TICKET)
    schedule = config.full_run_schedule
    assert config.dsl_compile_hash == J3_PROGRAM_SHA256
    assert config.typed_config_hash() == "fa63e79492d916a9cc6fe144207bdcb627d07e416883e131ecb90c289f8ccec0"
    assert schedule is not None
    assert schedule.warm_start_reform is None


def test_current_j7_resealed_ticket_uses_cap_free_realized_acceptance_and_q8_staging() -> None:
    config = DirectDescriptionJointDescentTypedConfigV1.from_ticket(J5_TICKET)
    schedule = config.full_run_schedule
    assert config.dsl_compile_hash == J7_PROGRAM_SHA256
    assert config.dsl_compile_hash != J5_PROGRAM_SHA256
    assert config.typed_config_hash() == "d8a8bb4f7f9ccb73f34a030cca976871bfd986b7fcc095d4f9354aa1144d0c8c"
    assert config.verdict_batch == 32
    assert schedule is not None
    reform = schedule.warm_start_reform
    assert reform is not None
    assert reform.maximum_continuous_update_quantum_fraction is None
    assert reform.realized_acceptance_policy == "pure_priced_exact_n600"
    assert reform.proposal_staging == "camera_874x1164_q8_pre_final_uint8"
    assert reform.proposal_q8_denominator == 256
    assert reform.proposal_multipliers == (32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25)
    assert reform.opening_active_groups == ("island_worldsheet", "shared_template_dof")
    assert reform.opening_candidate_pair_ids == (447, 53, 416, 296, 547, 278, 501, 346)
    assert "shared_template_dof" not in reform.frozen_groups_until_first_admission
    assert reform.residual_bucket_admission_required is True
    assert reform.pose_objective_engage_condition is None
    assert schedule.pose_finish_engage is not None
    assert schedule.pose_finish_engage.minimum_points == 5
    assert schedule.stages[-1].target_d_pose == pytest.approx(163.06116431842463)
    assert config.execution_custody["banked_r1_comparator"]["binding_target"] is False
    assert config.execution_custody["banked_r1_comparator"]["promotion_eligible"] is False
    assert config.worst_geometry_memory_contract.expected_total_secants == 52


def test_j7_ticket_uses_exact_n600_batch32_when_resealed() -> None:
    ticket = json.loads(J5_TICKET.read_bytes())
    semantic = ticket["semantic_program"]
    semantic["program_id"] = "ddm_j7_366_pose_gate_history_reseal_n600_seed0"
    semantic["telemetry"]["verdict_batch"] = 32
    semantic["value_provenance"]["verdict_batch"] = (
        "P0 J7 authority: exact n600 frozen CPU scorer verdicts use batch32"
    )
    digest = hashlib.sha256(rfc8785_canonicalize(semantic)).hexdigest()
    assert digest == J7_PROGRAM_SHA256


@pytest.mark.parametrize(
    ("peak", "admit", "reason"),
    [
        (115.999, True, "SAFE_PROJECTED_PEAK_WITHIN_116_GIB_CEILING"),
        (116.001, False, "REFUSE_PROJECTED_PEAK_EXCEEDS_116_GIB_CEILING"),
        (float("nan"), False, "REFUSE_INVALID_MEASURED_PEAK"),
    ],
)
def test_memory_preflight_is_fail_closed(peak: float, admit: bool, reason: str) -> None:
    assert classify_memory_preflight(peak) == (admit, reason)


def test_adam_checkpoint_is_atomic_preserved_and_bit_exact(tmp_path: Path) -> None:
    config = DirectDescriptionJointDescentTypedConfigV1.from_ticket(TICKET)
    initial = initial_adam_state(7)
    gradient = np.linspace(-0.2, 0.2, 7, dtype=np.float32)
    stepped = clipped_adam_step(
        initial,
        gradient,
        learning_rate=0.05,
        grad_clip=config.grad_clip,
        ema_decay=config.ema_decay,
    )
    path = tmp_path / "stage00_step000001.npz"
    checkpoint_sha = save_stage_checkpoint(
        path,
        stepped,
        stage_id="00_receiver_replay_and_adapter",
        config=config,
        telemetry=({"event": "unit_resume_boundary", "score_claim": False},),
        run_cursor={"stage_index": 1, "stage_step": 7, "global_step": stepped.step},
        realized_archive={"bytes": 123, "sha256": "0" * 64},
    )

    loaded, metadata = load_stage_checkpoint(path, config=config)

    assert len(checkpoint_sha) == 64
    with np.load(path, allow_pickle=False) as archive:
        assert "__resume_registry_manifest" in archive.files
        assert "__ddmjd_optimizer_state_sha256" in archive.files
    assert metadata["ema_shadow_saved"] is True
    assert metadata["rng"] == {"kind": "deterministic_no_sampling", "state": 0}
    assert metadata["canonical_resume_registry"]["controller"] == "ddm_joint_descent_optimizer"
    assert metadata["run_cursor"] == {"stage_index": 1, "stage_step": 7, "global_step": 1}
    assert metadata["realized_archive"] == {"bytes": 123, "sha256": "0" * 64}
    for field in ("theta", "ema", "first_moment", "second_moment"):
        assert np.array_equal(getattr(stepped, field), getattr(loaded, field))
    assert loaded.step == stepped.step
    with pytest.raises(DirectDescriptionError, match="already exists"):
        save_stage_checkpoint(
            path,
            stepped,
            stage_id="00_receiver_replay_and_adapter",
            config=config,
            telemetry=(),
        )

    corrupt_path = tmp_path / "stage00_step000001_corrupt.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    arrays["theta"][0] += np.float32(1.0)
    np.savez(corrupt_path, **arrays)
    with pytest.raises(DirectDescriptionError, match="optimizer state hash differs"):
        load_stage_checkpoint(corrupt_path, config=config)


def test_adam_resume_continuation_matches_uninterrupted_bits() -> None:
    state = initial_adam_state(5)
    gradient = np.asarray((0.2, -0.1, 0.05, -0.02, 0.3), dtype=np.float32)
    first = clipped_adam_step(state, gradient, learning_rate=0.01, grad_clip=0.5, ema_decay=0.997)
    resumed = AdamStateV1(
        step=first.step,
        theta=first.theta.copy(),
        ema=first.ema.copy(),
        first_moment=first.first_moment.copy(),
        second_moment=first.second_moment.copy(),
    )
    uninterrupted = clipped_adam_step(first, gradient, learning_rate=0.01, grad_clip=0.5, ema_decay=0.997)
    continued = clipped_adam_step(resumed, gradient, learning_rate=0.01, grad_clip=0.5, ema_decay=0.997)
    for field in ("theta", "ema", "first_moment", "second_moment"):
        assert np.array_equal(getattr(uninterrupted, field), getattr(continued, field))


def test_receiver_effective_surface_excludes_unencoded_j2_names() -> None:
    config = DirectDescriptionJointDescentTypedConfigV1.from_ticket(TICKET)
    archive = (REPO / config.source_archive_path).read_bytes()
    lift = lift_v15_archive(archive)
    groups = parameter_group_indices(lift)

    assert {name: len(indexes) for name, indexes in groups.items()} == {
        "island_worldsheet": 326,
        "lane_program": 24,
        "shared_template_dof": 18,
    }
    assert len(lift.parameter_names) == 368
    assert not any("aspect_log" in name or "rotation" in name for name in lift.parameter_names)
    assert not any("bev_c" in name or "range_gate" in name for name in lift.parameter_names)

    zero = np.zeros(len(lift.parameter_names), dtype=np.float32)
    stage00, realized = compile_parameterized_archive(lift, zero, include_lane_programs=False)
    assert stage00 == archive
    assert np.array_equal(realized, zero)
    lane_materialized, _ = compile_parameterized_archive(lift, zero, include_lane_programs=True)
    assert lane_materialized != archive
    assert len(lane_materialized) > len(archive)


def test_full_run_schedule_refuses_more_than_quarter_uint8_quantum() -> None:
    semantic = {
        "full_run_schedule": {
            "train_batch": 1,
            "learning_rate_quantum_fraction": 0.5,
            "checkpoint_interval_steps": 10,
            "plateau_verdicts": 2,
            "warm_start_pair": 447,
            "warm_start_steps": 4,
            "measured_seconds_per_step": 2.0,
            "measured_seconds_per_step_low": 1.5,
            "measured_seconds_per_step_high": 2.5,
            "warm_start_reform": {
                "adam_beta2": 0.999,
                "lr_rewarmup_c": 2.0,
                "lr_rewarmup_steps": 2000,
                "lr_rewarmup_floor": 0.1,
                "lr_rewarmup_shape": "linear",
                "maximum_continuous_update_quantum_fraction": 0.25,
                "frozen_groups_until_first_admission": ["shared_template_dof"],
                "group_release_condition": "first_strict_n600_island_admission",
                "pose_objective_engage_condition": "after_first_strict_n600_seg_admission",
                "first_realized_admission": ("exact_n600_dseg_descent_and_dpose_nonregression_else_abort_rollback"),
            },
            "stages": [
                {
                    "stage_id": "01_island_worldsheet_joint_descent",
                    "active_groups": ["island_worldsheet", "shared_template_dof"],
                    "maximum_steps": 600,
                    "verdict_interval_steps": 600,
                    "target_d_seg": 0.020602722168,
                    "target_d_pose": None,
                }
            ],
        }
    }
    with pytest.raises(DirectDescriptionError, match="quarter-quantum"):
        FullRunScheduleV1.from_semantic_program(semantic)


@pytest.mark.parametrize(
    ("candidate_d_seg", "candidate_d_pose", "expected"),
    [
        (0.02760, 163.0, "BLOCKED_REALIZED_DSEG_REGRESSION"),
        (0.02740, 163.1, "BLOCKED_REALIZED_DPOSE_REGRESSION"),
        (0.02747, 162.9, "REALIZED_STAGE_SEG_FLAT_POSE_DESCENT_CONTINUE"),
        (0.02747, 163.0, "BLOCKED_REALIZED_NO_COMPONENT_DESCENT"),
        (0.02740, 163.0, "REALIZED_STAGE_DESCENT_CONTINUE"),
        (0.02000, 163.0, "REALIZED_STAGE_TARGET_MET"),
    ],
)
def test_realized_stage_decision_is_fail_closed(
    candidate_d_seg: float,
    candidate_d_pose: float,
    expected: str,
) -> None:
    assert (
        classify_realized_stage_verdict(
            reference_d_seg=0.02747,
            reference_d_pose=163.0,
            candidate_d_seg=candidate_d_seg,
            candidate_d_pose=candidate_d_pose,
            target_d_seg=0.02060,
            target_d_pose=None,
        )
        == expected
    )


def test_beta2_rewarmup_and_quarter_quantum_cap_remove_fresh_adam_jump() -> None:
    assert linear_rewarmup_factor(completed_steps=0, rewarmup_steps=2000, floor=0.1) == 0.1
    assert linear_rewarmup_factor(completed_steps=1000, rewarmup_steps=2000, floor=0.1) == 0.55
    assert linear_rewarmup_factor(completed_steps=2000, rewarmup_steps=2000, floor=0.1) == 1.0

    initial = initial_adam_state(3)
    gradient = np.asarray((2.0, -0.5, 0.0), dtype=np.float32)
    stepped = clipped_adam_step(
        initial,
        gradient,
        learning_rate=1.0,
        grad_clip=10.0,
        ema_decay=0.997,
        beta2=0.999,
        maximum_update=0.25,
    )
    assert np.array_equal(stepped.theta, np.asarray((-0.25, 0.25, 0.0), dtype=np.float32))
    assert float(np.max(np.abs(stepped.ema))) < 0.001
    with pytest.raises(DirectDescriptionError, match="hyperparameters"):
        clipped_adam_step(
            initial,
            gradient,
            learning_rate=1.0,
            grad_clip=10.0,
            ema_decay=0.997,
            maximum_update=float("nan"),
        )


def test_j5_q8_staging_and_coherent_worldsheet_direction_are_deterministic() -> None:
    config = DirectDescriptionJointDescentTypedConfigV1.from_ticket(J5_TICKET)
    archive = (REPO / config.source_archive_path).read_bytes()
    lift = lift_v15_archive(archive)
    local = np.linspace(-0.2, 0.2, len(lift.parameter_names), dtype=np.float32)
    reform = config.full_run_schedule.warm_start_reform
    assert reform is not None
    proposal = opening_candidate_gradient(
        lift,
        "worldsheet_joint_active_x_+1",
        local,
        active_pair_ids=reform.opening_candidate_pair_ids,
    )
    selected = np.flatnonzero(proposal)
    assert 0 < len(selected) < len(lift.g1.tracks)
    assert np.all(proposal[selected] == -1.0)

    state = clipped_adam_step(
        initial_adam_state(len(lift.parameter_names)),
        proposal,
        learning_rate=0.8,
        grad_clip=0.5,
        ema_decay=config.ema_decay,
        beta2=0.999,
        maximum_update=None,
        theta_lattice_denominator=256,
    )
    assert np.all(state.theta[selected] == np.float32(205 / 256))
    assert np.all(state.theta * 256 == np.rint(state.theta * 256))
    realized = np.rint(state.theta)
    assert np.all(realized[selected] == 1.0)
    assert np.count_nonzero(realized) == len(selected)
    candidate_archive, candidate_realized = compile_parameterized_archive(
        lift,
        state.theta,
        include_lane_programs=False,
    )
    assert np.array_equal(candidate_realized, realized.astype(np.float32))
    assert hashlib.sha256(candidate_archive).hexdigest() == (
        "d4eb1450f461437e714d08a9349cc735fe79b53a1739a2de92ef4850287dfd0d"
    )


def test_rewarmup_length_must_rederive_from_beta2_law() -> None:
    ticket = json.loads(FULL_RUN_TICKET.read_bytes())
    semantic = ticket["semantic_program"]
    semantic["full_run_schedule"]["warm_start_reform"]["lr_rewarmup_steps"] = 1999
    with pytest.raises(DirectDescriptionError, match="adam_v_variance_warmup_length_v1"):
        FullRunScheduleV1.from_semantic_program(semantic)


def test_launcher_has_first_integer_n600_abort_and_release_gates() -> None:
    source = (REPO / "tools/launch_ddm_joint_descent.py").read_text(encoding="utf-8")
    assert "FIRST_INTEGER_REALIZATION_EXACT_N600_ABORT_ROLLBACK" in source
    assert 'set(reform["frozen_groups_until_first_admission"])' in source
    assert "pose_finish_state.engaged" in source
    assert "pose_objective_weight = 0.0 if reform_active and not warm_start_seg_admitted else 1.0" not in source
    assert "warm_start_realized_admitted" in source
    assert "warm_start_rejected_proposal_rollback" in source
    assert "warm-start rollback checkpoint immediate parse-back differs" in source


def test_launcher_has_j5_pure_price_shrink_bucket_and_fire_gates() -> None:
    source = (REPO / "tools/launch_ddm_joint_descent.py").read_text(encoding="utf-8")
    assert "pure_priced_realized_delta" in source
    assert "proposal_q8_denominator" in source
    assert "c1_bucket_delta_vs_last_admitted" in source
    assert "c1_bucket_delta_cumulative_vs_baseline" in source
    assert "warm_start_component_safe or" not in source
    assert "residual_bucket_descended" in source
    assert "BLOCKED_REALIZED_NO_PURE_PRICED_DESCENT_AFTER_SHRINK_LADDER" in source
    assert 'final_run_verdict = "READY_TO_FIRE_UNDER_STANDING_GO"' in source
    assert "if bounded_stop and warm_start_component_safe:" not in source


def test_pose_finish_engage_is_typed_rolling_slope_latched_and_checkpoint_roundtrippable() -> None:
    config = DirectDescriptionJointDescentTypedConfigV1.from_ticket(J5_TICKET)
    engage = config.full_run_schedule.pose_finish_engage
    assert engage is not None
    state = PoseFinishEngageStateV1().observe(
        global_step=0,
        d_seg=0.027470296223958333,
        strict_seg_admission=False,
        config=engage,
    )
    state = state.observe(
        global_step=1,
        d_seg=0.02744209289550781,
        strict_seg_admission=True,
        config=engage,
    )
    for step in range(2, 24):
        state = state.observe(
            global_step=step,
            d_seg=0.02744209289550781,
            strict_seg_admission=False,
            config=engage,
        )
    assert state.engaged
    assert state.classification in {"POSE_FINISH_ENGAGED_PLATEAU", "POSE_FINISH_ENGAGED_LATCHED"}
    assert PoseFinishEngageStateV1.from_payload(state.to_payload(), config=engage) == state


def test_pose_finish_engage_refuses_binary_first_admission_and_rising_history() -> None:
    config = DirectDescriptionJointDescentTypedConfigV1.from_ticket(J5_TICKET)
    engage = config.full_run_schedule.pose_finish_engage
    state = PoseFinishEngageStateV1()
    for step in range(12):
        state = state.observe(
            global_step=step,
            d_seg=0.02747 - 0.0001 * step,
            strict_seg_admission=step == 1,
            config=engage,
        )
    assert state.strict_seg_admissions == 1
    assert not state.engaged
    assert state.classification == "DSEG_STILL_TRENDING"


def test_cumulative_fire_gate_catches_locally_safe_second_move() -> None:
    green, decision = classify_cumulative_fire_gate(
        baseline_d_seg=0.02000,
        baseline_d_pose=1.0,
        candidate_d_seg=0.02008,
        candidate_d_pose=0.89,
        cumulative_residual_delta_errors=-2,
        residual_descent_required=True,
    )
    assert green is False
    assert decision == "BLOCKED_CUMULATIVE_DSEG_REGRESSION_VS_STAGE00"
    green, decision = classify_cumulative_fire_gate(
        baseline_d_seg=0.02000,
        baseline_d_pose=1.0,
        candidate_d_seg=0.01999,
        candidate_d_pose=0.99,
        cumulative_residual_delta_errors=-1,
        residual_descent_required=True,
    )
    assert (green, decision) == (
        True,
        "CUMULATIVE_COMPONENT_AND_RESIDUAL_FIRE_GATE_GREEN_VS_STAGE00",
    )


def test_target_unmet_limits_and_plateaus_are_nonpromoting_stops() -> None:
    assert classify_governed_stage_exit(
        target_met=False,
        stage_limit=True,
        plateau=False,
        component_decision="REALIZED_STAGE_DESCENT_CONTINUE",
    ) == "STOPPED_BELOW_TARGET_MAXIMUM_STEPS"
    assert classify_governed_stage_exit(
        target_met=False,
        stage_limit=False,
        plateau=True,
        component_decision="BLOCKED_REALIZED_NO_COMPONENT_DESCENT",
    ) == "STOPPED_BELOW_TARGET_PLATEAU"
    assert classify_governed_stage_exit(
        target_met=True,
        stage_limit=True,
        plateau=True,
        component_decision="REALIZED_STAGE_TARGET_MET",
    ) == "ADVANCE_EXACT_TARGET_MET"


def test_schedule_complete_requires_exact_final_target_verdict() -> None:
    config = DirectDescriptionJointDescentTypedConfigV1.from_ticket(J5_TICKET)
    final_stage = config.full_run_schedule.stages[-1]
    green, decision = exact_final_target_gate(
        final_verdict={
            "stage_id": final_stage.stage_id,
            "d_seg": final_stage.target_d_seg + 1.0e-6,
            "d_pose": final_stage.target_d_pose,
        },
        final_stage=final_stage,
    )
    assert (green, decision) == (False, "REFUSE_SCHEDULE_COMPLETE_FINAL_TARGET_UNMET")
    green, decision = exact_final_target_gate(
        final_verdict={
            "stage_id": final_stage.stage_id,
            "d_seg": final_stage.target_d_seg,
            "d_pose": final_stage.target_d_pose,
        },
        final_stage=final_stage,
    )
    assert (green, decision) == (True, "FULL_RUN_SCHEDULE_COMPLETE_EXACT_FINAL_TARGETS_MET")
    for field in ("d_seg", "d_pose"):
        invalid = {
            "stage_id": final_stage.stage_id,
            "d_seg": final_stage.target_d_seg,
            "d_pose": final_stage.target_d_pose,
        }
        invalid[field] = float("nan")
        green, decision = exact_final_target_gate(
            final_verdict=invalid,
            final_stage=final_stage,
        )
        assert (green, decision) == (
            False,
            "REFUSE_SCHEDULE_COMPLETE_FINAL_VERDICT_INVALID",
        )


def test_launcher_seals_worst_geometry_and_governed_completion_paths() -> None:
    source = (REPO / "tools/launch_ddm_joint_descent.py").read_text(encoding="utf-8")
    assert "REFUSE_WORST_GEOMETRY_DIFFERS" in source
    assert "expected_total_secants" in source
    assert "STOPPED_BELOW_TARGET_MAXIMUM_STEPS" in (
        REPO / "src/tac/optimization/direct_description_joint_descent.py"
    ).read_text(encoding="utf-8")
    assert 'final_run_verdict = "FULL_RUN_SCHEDULE_COMPLETE"' not in source
