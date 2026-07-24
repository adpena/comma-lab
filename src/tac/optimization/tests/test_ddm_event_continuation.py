# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tac.optimization.ddm_event_continuation import (
    AcquisitionObservationV1,
    BudgetCapsV1,
    ChargeSectionV1,
    ContinuationStateV1,
    DDMEventContinuationError,
    DDMEventContinuationV1,
    EventNodeV1,
    audit_two_part_charge,
    build_j8e_event_continuation,
    pareto_acquisition_order,
)
from tac.optimization.ddm_witness_program import (
    DDMWitnessProgramError,
    DDMWitnessProgramV1,
    MetricSelectorV1,
    SolveHookV1,
)
from tac.optimization.direct_description_joint_descent import (
    DirectDescriptionError,
    FullRunScheduleV1,
)

REPO = Path(__file__).resolve().parents[4]
SHA_A = "a" * 64
SHA_B = "b" * 64


def _graph() -> DDMEventContinuationV1:
    telemetry = (
        *(f"Q{index}" for index in range(1, 8)),
        "lever_engage",
        "term_inert",
        "liveness",
        "S_before",
        "S_after",
        "counted_bytes_before",
        "counted_bytes_after",
        "measured_work",
        "g_S",
        "g_L",
        "delta_S_per_wall_clock_hour",
        "delta_bytes_per_step",
        "box_milestone_crossed",
        "rollback_reason",
    )
    return DDMEventContinuationV1(
        graph_id="ddm_j8e_fixture",
        initial_node_id="resume_boundary",
        nodes=(
            EventNodeV1(
                node_id="resume_boundary",
                kind="continuation",
                entry_events=("ws3_start_bound",),
                exit_events=("first_component_safe_admission",),
                next_by_event={"first_component_safe_admission": "joint_continuation"},
                active_group_policy="costate_pareto_ranked",
                checkpoint_events=("first_component_safe_admission",),
            ),
            EventNodeV1(
                node_id="joint_continuation",
                kind="continuation",
                entry_events=("first_component_safe_admission",),
                exit_events=("pose_latch", "solve_basin"),
                next_by_event={
                    "pose_latch": "pose_finish",
                    "solve_basin": "terminal_solve",
                },
                active_group_policy="costate_pareto_ranked",
                checkpoint_events=("pose_latch", "solve_basin"),
            ),
            EventNodeV1(
                node_id="pose_finish",
                kind="continuation",
                entry_events=("pose_latch",),
                exit_events=("solve_basin",),
                next_by_event={"solve_basin": "terminal_solve"},
                active_group_policy="costate_pareto_ranked_pose_trust",
                checkpoint_events=("solve_basin",),
            ),
            EventNodeV1(
                node_id="terminal_solve",
                kind="solve_interleave",
                entry_events=("solve_basin",),
                exit_events=("governed_stop",),
                next_by_event={"governed_stop": "STOP"},
                active_group_policy="ms2_metric_active_blocks",
                checkpoint_events=("governed_stop",),
                execution_enabled=False,
                blocker="DDM_MS2_NO_ADMISSIBLE_METRIC_ACTIVE_N600_CANDIDATE",
            ),
        ),
        budget_caps=BudgetCapsV1(
            maximum_receiver_verdicts=4000,
            maximum_wall_seconds=86_400.0,
            checkpoint_recovery_loss_verdicts=1,
            maximum_counted_bytes=200_000,
        ),
        proposal_metric_selector="scorer_recursive_rank4_fisher_corrected_J",
        exact_acceptance_metric=(
            "100*d_seg_R+sqrt(10*d_pose_YUV6_R)+25*archive_bytes/37545489"
        ),
        box_tolerance_policy={
            "descent_box_role": "milestone_not_stop",
            "describe_solve_box_role": "tolerance_stop",
            "ms2r_role": "proposal_ordering_prior_until_describe_solve",
            "global_fallback_declared": True,
            "descent_continuation": "exact_receiver_realized_delta_S_lt_zero",
        },
        visibility_policy={
            "frame_0": "pose-only(frame_0)",
            "fine_chroma": "seg-only",
            "shared_visible": "joint",
            "resize_null": "gauge_fixed_out",
            "blind_coordinates": "excluded",
        },
        terminal_hooks={
            "fork_head_solve": {"execution_enabled": False},
            "head_offset_solver": {"execution_enabled": False},
            "ms2_terminal_solve": {"execution_enabled": False},
            "mc_finisher": {"execution_enabled": False},
        },
        telemetry_fields=telemetry,
        execution_allowed=False,
    )


def _telemetry() -> dict:
    return {
        **{f"Q{index}": {"status": "ok"} for index in range(1, 7)},
        "Q7": {"status": "ok", "d_seg": 0.0702},
        "lever_engage": ["dm4_scorer_recursive"],
        "term_inert": [],
        "liveness": {
            "accepted_batch_fraction": 1.0,
            "weights_stepped": True,
            "frozen": False,
        },
        "S_before": 7.0,
        "S_after": 6.9,
        "counted_bytes_before": 138_804,
        "counted_bytes_after": 138_810,
        "measured_work": 1.0,
        "g_S": 0.1,
        "g_L": -6.0,
        "delta_S_per_wall_clock_hour": 360.0,
        "delta_bytes_per_step": 6,
        "box_milestone_crossed": False,
        "rollback_reason": None,
    }


def test_event_graph_roundtrip_has_no_fixed_stage_clock() -> None:
    graph = _graph()
    payload = graph.to_payload()
    assert DDMEventContinuationV1.from_payload(payload) == graph
    assert "stages" not in payload
    assert "verdict_interval_steps" not in payload

    semantic = {
        "full_run_schedule": {
            "train_batch": 4,
            "learning_rate_quantum_fraction": 0.25,
            "measured_seconds_per_step": 102.0,
            "measured_seconds_per_step_low": 100.0,
            "measured_seconds_per_step_high": 105.0,
            "event_graph": payload,
        }
    }
    compiled = FullRunScheduleV1.from_semantic_program(semantic)
    assert compiled is not None
    assert compiled.stages == ()
    assert compiled.event_continuation == graph
    assert compiled.checkpoint_interval_steps is None


def test_canonical_j8e_graph_treats_box_as_milestone_and_economics_as_stop() -> None:
    graph = build_j8e_event_continuation(
        maximum_receiver_verdicts=450,
        maximum_wall_seconds=49_657.37114489195,
        maximum_counted_bytes=200_000,
    )
    payload = graph.to_payload()
    assert graph.execution_allowed is False
    assert graph.box_tolerance_policy["descent_box_role"] == "milestone_not_stop"
    assert "stages" not in payload
    for node_id in ("costate_ranked_joint_continuation", "pose_protected_finish"):
        node = next(node for node in graph.nodes if node.node_id == node_id)
        assert node.next_by_event["box_milestone_crossed"] == node_id
        assert node.next_by_event["economic_or_dynamics_stop"] == "STOP"
        assert "delta_S_per_wall_clock_hour" in graph.telemetry_fields
        assert "delta_bytes_per_step" in graph.telemetry_fields


def test_event_graph_refuses_fixed_stage_actuator_and_incomplete_telemetry() -> None:
    payload = _graph().to_payload()
    payload["verdict_interval_steps"] = 50
    with pytest.raises(DDMEventContinuationError, match="forbidden fixed-stage"):
        DDMEventContinuationV1.from_payload(payload)

    state = _graph().initial_state(accepted_state_id="ws3-step4")
    telemetry = _telemetry()
    telemetry.pop("term_inert")
    with pytest.raises(DDMEventContinuationError, match="misses telemetry"):
        _graph().advance(
            state,
            event="first_component_safe_admission",
            accepted_state_id="dm4-proposal-1",
            telemetry=telemetry,
        )


def test_event_schedule_refuses_legacy_ambiguity_and_json_boolean_coercion() -> None:
    payload = _graph().to_payload()
    payload["execution_allowed"] = "false"
    with pytest.raises(DDMEventContinuationError, match="JSON boolean"):
        DDMEventContinuationV1.from_payload(payload)

    with pytest.raises(DirectDescriptionError, match="ambiguous legacy schedule"):
        FullRunScheduleV1.from_semantic_program(
            {
                "full_run_schedule": {
                    "train_batch": 4,
                    "learning_rate_quantum_fraction": 0.25,
                    "measured_seconds_per_step": 102.0,
                    "measured_seconds_per_step_low": 100.0,
                    "measured_seconds_per_step_high": 105.0,
                    "event_graph": _graph().to_payload(),
                    "stages": [],
                }
            }
        )


def test_event_advance_is_causal_resume_safe_and_disabled_solve_refuses() -> None:
    graph = _graph()
    state, mark = graph.advance(
        graph.initial_state(accepted_state_id="ws3-step4"),
        event="first_component_safe_admission",
        accepted_state_id="dm4-proposal-1",
        telemetry=_telemetry(),
    )
    assert state.node_id == "joint_continuation"
    assert state.accepted_verdicts == 1
    assert mark["checkpoint_required"] is True
    assert len(mark["event_id"]) == 64
    assert ContinuationStateV1.from_payload(state.to_payload()) == state

    state2, _ = graph.advance(
        state,
        event="solve_basin",
        accepted_state_id="dm4-proposal-1",
        telemetry=_telemetry(),
    )
    with pytest.raises(DDMEventContinuationError, match="disabled"):
        graph.advance(
            state2,
            event="governed_stop",
            accepted_state_id="dm4-proposal-1",
            telemetry=_telemetry(),
        )


def test_pareto_acquisition_never_scalarizes_and_uses_stable_id_tie_break() -> None:
    rows = (
        AcquisitionObservationV1("b", 10.0, 9.0, 100, 100, 1.0, SHA_A, SHA_B, "advisory"),
        AcquisitionObservationV1("a", 10.0, 9.0, 100, 100, 1.0, SHA_A, SHA_B, "advisory"),
        AcquisitionObservationV1("dominated", 10.0, 9.5, 100, 101, 1.0, SHA_A, SHA_B, "advisory"),
        AcquisitionObservationV1("rate", 10.0, 9.8, 100, 80, 1.0, SHA_A, SHA_B, "advisory"),
    )
    ordered = pareto_acquisition_order(rows)
    assert [row.proposal_id for row in ordered[:3]] == ["a", "b", "rate"]
    assert ordered[-1].proposal_id == "dominated"
    assert "scalar" not in str([row.to_payload() for row in ordered]).lower()


def test_charge_audit_conserves_archive_and_charges_video_selected_sections() -> None:
    audit = audit_two_part_charge(
        (
            ChargeSectionV1("base", True, 90, SHA_A, "video-selected base"),
            ChargeSectionV1("framing", False, 10, SHA_B, "fixed framing bytes"),
        ),
        archive_bytes=100,
        fixed_interpreter_sha256="c" * 64,
    )
    assert audit["conserved"] is True
    with pytest.raises(DDMEventContinuationError, match="cannot be declared free"):
        ChargeSectionV1("selector", True, 0, SHA_A, "video-selected branch")


def _program() -> DDMWitnessProgramV1:
    source_paths = {
        "launcher": "tools/launch_ddm_joint_descent.py",
        "consumer": "src/tac/optimization/direct_description_joint_descent.py",
        "event_engine": "src/tac/optimization/ddm_event_continuation.py",
        "dm4_adapter": "src/tac/optimization/ddm_dm4_j5_adapter.py",
        "dm4_constructor": "src/tac/optimization/ddm_dm4_targeted_realization_cures.py",
    }
    bindings = {
        name: hashlib.sha256((REPO / path).read_bytes()).hexdigest()
        for name, path in source_paths.items()
    }
    hooks = (
        SolveHookV1(
            "fork_head_solve",
            "resume_boundary",
            "ForkHeadSolve",
            False,
            ("fork_head_solve_receipt",),
            "DDM_MATCHED_UPDATE_RMS_RECEIPT_MISSING",
        ),
        SolveHookV1(
            "head_offset_solver",
            "solve_basin",
            "HeadOffsetSolver",
            False,
            ("head_offset_solver_ab_receipt",),
            "DDM_HEAD_OFFSET_RECEIVER_CUSTODY_MISSING",
        ),
        SolveHookV1(
            "ms2_terminal_solve",
            "solve_basin",
            "ddm_ms2_typed_quotient_solve",
            False,
            ("metric_active_n600_receipt",),
            "DDM_MS2_NO_ADMISSIBLE_METRIC_ACTIVE_N600_CANDIDATE",
        ),
        SolveHookV1(
            "mc_finisher",
            "post_descent",
            "mc_finisher_diagonal+erm_margin_topk",
            False,
            ("op_gc1_5_main_review",),
            "PREREGISTRATION_ONLY",
        ),
    )
    return DDMWitnessProgramV1(
        program_id="ddm_j8e_688",
        event_continuation=_graph(),
        metric_selector=MetricSelectorV1("ddm_exact_s_v1"),
        solve_hooks=hooks,
        ticket_path=".omx/research/configs/ddm_j8e_688_event_engine_20260724.json",
        source_bindings=bindings,
        beta2=0.999,
        ema_decay=0.997,
        inference_shadow="ema",
        execution_allowed=False,
    )


def test_witness_program_resolves_lawrefs_hashes_sources_and_compiles_real_argv() -> None:
    argv, manifest = _program().compile_trainer_argv_with_constants(
        repo_root=REPO,
        out_dir="/Volumes/VertigoDataTier/pact/ddm_j8e_smoke",
    )
    assert argv[:2] == ("python3", "tools/launch_ddm_joint_descent.py")
    assert argv[-1] == "--dry-run"
    assert manifest["constants"]["ema_decay"] == 0.997
    assert manifest["lawrefs"]["ema_decay"]["fallback_used"] is False
    assert manifest["lawrefs"]["adam_beta2_rewarmup"]["fallback_used"] is False
    assert manifest["op_gc1_5"]["execution_enabled"] is False
    assert len(manifest["typed_config_hash"]) == 64
    with pytest.raises(DDMWitnessProgramError, match="execution_allowed=false"):
        _program().compile_trainer_argv_with_constants(
            repo_root=REPO,
            out_dir="/Volumes/VertigoDataTier/pact/ddm_j8e_full",
            mode="full-run",
        )


def test_witness_program_refuses_source_hash_drift() -> None:
    program = _program()
    bindings = dict(program.source_bindings)
    bindings["launcher"] = "0" * 64
    drifted = DDMWitnessProgramV1(
        program_id=program.program_id,
        event_continuation=program.event_continuation,
        metric_selector=program.metric_selector,
        solve_hooks=program.solve_hooks,
        ticket_path=program.ticket_path,
        source_bindings=bindings,
        beta2=program.beta2,
        ema_decay=program.ema_decay,
        inference_shadow="ema",
        execution_allowed=False,
    )
    with pytest.raises(DDMWitnessProgramError, match="source SHA differs"):
        drifted.compile_trainer_argv_with_constants(
            repo_root=REPO,
            out_dir="/tmp/not_written",
        )
