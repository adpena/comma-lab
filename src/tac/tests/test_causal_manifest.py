from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from tac.causal_manifest import (
    SCHEMA_ID,
    ActionSummary,
    ActionSupport,
    ApparatusState,
    ArmPropensity,
    ArtifactRef,
    CausalManifestConflictError,
    CausalManifestError,
    CausalManifestWriter,
    CoverageReceiptRow,
    DigestRef,
    ExplorationDecisionRow,
    LossTerm,
    RealizedOutcome,
    RewardObservation,
    RunTreatmentManifest,
    StagePlanEntry,
    StateSummary,
    TransitionRow,
    append_causal_row,
    canonical_sha256,
    check_fore_support,
    freeze_fields,
    hcm_l4_residual_check,
    load_causal_manifest,
    unavailable_artifact,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
STAMP = "2026-07-13T12:00:00Z"
REPO = Path(__file__).resolve().parents[3]


def _artifact(role: str = "checkpoint") -> ArtifactRef:
    return ArtifactRef(role=role, uri=f"/evidence/{role}", digest=DigestRef("sha256", SHA_A))


def _outcome(value: float = 0.2) -> RealizedOutcome:
    return RealizedOutcome(
        observed=True,
        through_r=True,
        d_seg=0.001,
        d_pose=0.002,
        archive_bytes=123,
        implied_score=value,
        axis="[macOS-MLX research-signal] NON-PROMOTABLE",
    )


def _state(
    boundary_id: str,
    sequence_index: int,
    *,
    epoch: int | None = None,
    apparatus: ApparatusState | None = None,
) -> StateSummary:
    return StateSummary(
        boundary_id=boundary_id,
        sequence_index=sequence_index,
        boundary_kind="verdict",
        epoch=sequence_index if epoch is None else epoch,
        stage="CE",
        policy_sha256=SHA_A,
        data_order_cursor=sequence_index,
        telemetry_history_sha256=SHA_B,
        telemetry_history_rows=sequence_index,
        checkpoint=_artifact(),
        resume_state_sha256=SHA_B,
        rng_state_sha256=None,
        controller_state_sha256=None,
        apparatus=apparatus or ApparatusState(measurement_mode="realized_through_R"),
        outcome=_outcome(),
        observed_at_utc=STAMP,
    )


def _action(action_id: str = "action:1") -> ActionSummary:
    return ActionSummary(
        action_id=action_id,
        action_type="stage_interval",
        arm_id="CE",
        policy_id="treatment_vector",
        policy_sha256=SHA_A,
        parameters=freeze_fields({"epoch": 1}),
    )


def _reward(value: float = -0.2) -> RewardObservation:
    return RewardObservation(
        estimand_id="negative_implied_score",
        observed=True,
        value=value,
        components=freeze_fields({"d_seg": 0.001, "d_pose": 0.002}),
    )


def _manifest(run_id: str = "run-a") -> RunTreatmentManifest:
    treatment = freeze_fields({"epochs": 10, "seed": 7})
    order = freeze_fields({"pair_order": "ascending", "pair_count": 2})
    return RunTreatmentManifest(
        row_id=f"run_manifest:{run_id}",
        run_id=run_id,
        treatment_vector=treatment,
        treatment_sha256=canonical_sha256({"epochs": 10, "seed": 7}),
        base_checkpoint=unavailable_artifact("base_checkpoint", "fresh_init", "fresh initialization"),
        seed=7,
        machine="test-machine",
        backend="numpy-test",
        axis="[unit-test] NON-PROMOTABLE",
        data_order=order,
        data_order_sha256=canonical_sha256({"pair_order": "ascending", "pair_count": 2}),
        stage_plan=(StagePlanEntry("CE", 0, "fixed_epoch"),),
        scorer_artifacts=(_artifact("scorer"),),
        cache_artifacts=(_artifact("cache"),),
        created_at_utc=STAMP,
    )


def _transition(run_id: str = "run-a", suffix: str = "1", *, apparatus: ApparatusState | None = None) -> TransitionRow:
    state = _state(f"{suffix}:z", 0, epoch=0)
    next_state = _state(f"{suffix}:zp", 1, epoch=1, apparatus=apparatus)
    return TransitionRow(
        row_id=f"transition:{run_id}:{suffix}",
        transition_id=f"{suffix}:z->{suffix}:zp",
        run_id=run_id,
        pair_id="pair-000",
        state=state,
        action=_action(f"action:{suffix}"),
        reward=_reward(),
        next_state=next_state,
        emitted_at_utc=STAMP,
    )


def _decision(run_id: str = "run-a", *, randomized: bool = False, executed: bool = False) -> ExplorationDecisionRow:
    propensities = (
        (ArmPropensity("arm-a", 0.5), ArmPropensity("arm-b", 0.5))
        if randomized
        else (ArmPropensity("arm-a", 1.0), ArmPropensity("arm-b", 0.0))
    )
    return ExplorationDecisionRow(
        row_id=f"decision:{run_id}:{randomized}:{executed}",
        decision_id=f"{run_id}:{randomized}:{executed}",
        run_id=run_id,
        state=_state("decision-state", 1),
        chosen_arm="arm-a",
        arm_propensities=propensities,
        policy_id="policy-a",
        policy_sha256=SHA_A,
        policy_mode="randomized" if randomized else "deterministic",
        exploration_hook="externally_authorized" if randomized else "disabled_pending_operator_go",
        executed=executed,
        actuation="SCHEDULE_ARM" if executed else "NONE",
        random_seed=7 if randomized else None,
        random_draw=0.2 if randomized else None,
        emitted_at_utc=STAMP,
    )


def _receipt(run_id: str = "run-a") -> CoverageReceiptRow:
    return CoverageReceiptRow(
        row_id=f"coverage:{run_id}",
        receipt_id=f"coverage:{run_id}",
        run_id=run_id,
        target_policy_id="policy-a",
        target_policy_sha256=SHA_A,
        working_support_id="measured-support-v1",
        initial_state_covered=True,
        one_step_target_covered=True,
        action_support=(ActionSupport("arm-a", 0.5), ActionSupport("arm-b", 0.5)),
        assessment_method="exact_logged_propensity_minimum",
        evidence=("decision:run-a:True:True",),
        verdict_scope="FORMULATION x CACHE SUPPORT",
        emitted_at_utc=STAMP,
    )


def test_frozen_rows_cannot_be_mutated() -> None:
    row = _manifest()
    with pytest.raises(FrozenInstanceError):
        row.seed = 8  # type: ignore[misc]


def test_freeze_fields_is_sorted_and_canonical() -> None:
    fields = freeze_fields({"z": [2, 1], "a": {"b": True}})
    assert [item.name for item in fields] == ["a", "z"]
    assert fields[0].value_json == '{"b":true}'


def test_freeze_fields_rejects_nonfinite_values() -> None:
    with pytest.raises(CausalManifestError, match="NaN or infinity"):
        freeze_fields({"loss": float("nan")})


def test_digest_ref_rejects_mislabeled_git_sha() -> None:
    with pytest.raises(CausalManifestError, match="64 hexadecimal"):
        DigestRef("sha256", "a" * 40)


def test_unavailable_artifact_requires_reason() -> None:
    with pytest.raises(CausalManifestError, match="unavailable_reason"):
        ArtifactRef(role="cache", uri="missing", digest_status="unavailable")


def test_run_manifest_rejects_treatment_digest_mismatch() -> None:
    row = _manifest()
    with pytest.raises(CausalManifestError, match="treatment_sha256"):
        RunTreatmentManifest(**{**row.__dict__, "treatment_sha256": SHA_B})


def test_observed_outcome_requires_a_measured_value() -> None:
    with pytest.raises(CausalManifestError, match="measured value"):
        RealizedOutcome(observed=True, through_r=True)


def test_observed_outcome_requires_realized_through_r_custody() -> None:
    with pytest.raises(CausalManifestError, match="realized through R"):
        RealizedOutcome(observed=True, through_r=False, d_seg=0.1)


def test_unobserved_outcome_rejects_false_measured_values() -> None:
    with pytest.raises(CausalManifestError, match="may not carry"):
        RealizedOutcome(observed=False, through_r=False, d_seg=0.1, missing_reason="not measured")


def test_state_summary_hash_round_trips() -> None:
    state = _state("z", 1)
    restored = StateSummary.from_dict(state.to_dict())
    assert restored == state
    assert restored.state_sha256 == state.state_sha256


def test_state_summary_rejects_tampered_hash() -> None:
    raw = _state("z", 1).to_dict()
    raw["state_sha256"] = SHA_B
    with pytest.raises(CausalManifestError, match="state_sha256 mismatch"):
        StateSummary.from_dict(raw)


def test_transition_rejects_backward_successor() -> None:
    with pytest.raises(CausalManifestError, match="strictly later"):
        TransitionRow(
            row_id="transition:bad",
            transition_id="bad",
            run_id="run-a",
            pair_id=None,
            state=_state("z", 1),
            action=_action(),
            reward=_reward(),
            next_state=_state("zp", 1),
            emitted_at_utc=STAMP,
        )


def test_append_and_load_round_trip_all_row_types(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    rows = (_manifest(), _transition(), _decision(), _receipt())
    for row in rows:
        append_causal_row(path, row)
    assert load_causal_manifest(path) == rows


def test_loader_rejects_unknown_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps({"schema_id": "pact.causal_manifest.v999", "row_kind": "x"}) + "\n")
    with pytest.raises(CausalManifestError, match="unsupported schema_id"):
        load_causal_manifest(path)


def test_loader_rejects_duplicate_immutable_row_id(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    append_causal_row(path, _manifest())
    append_causal_row(path, _manifest())
    with pytest.raises(CausalManifestError, match="duplicate row_id"):
        load_causal_manifest(path)


def test_writer_run_manifest_append_is_idempotent(tmp_path: Path) -> None:
    writer = CausalManifestWriter(tmp_path / "manifest.jsonl", "run-a")
    assert writer.ensure_run_manifest(_manifest())
    assert not writer.ensure_run_manifest(_manifest())
    assert writer.run_manifest == _manifest()
    assert len(load_causal_manifest(writer.path)) == 1


def test_writer_refuses_run_manifest_reinterpretation(tmp_path: Path) -> None:
    writer = CausalManifestWriter(tmp_path / "manifest.jsonl", "run-a")
    writer.ensure_run_manifest(_manifest())
    changed = _manifest()
    changed = RunTreatmentManifest(**{**changed.__dict__, "machine": "different-machine"})
    with pytest.raises(CausalManifestConflictError, match="changed across resume"):
        writer.ensure_run_manifest(changed)


def test_writer_continues_transition_chain_from_disk(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    writer = CausalManifestWriter(path, "run-a")
    assert writer.record_boundary(_state("z", 0), action=_action("a0"), reward=_reward()) is None
    resumed = CausalManifestWriter(path, "run-a")
    transition = resumed.record_boundary(_state("zp", 1), action=_action("a1"), reward=_reward())
    assert transition is not None
    assert transition.state.boundary_id == "z"
    assert transition.next_state.boundary_id == "zp"


def test_writer_retains_late_boundary_without_false_backward_transition(tmp_path: Path) -> None:
    writer = CausalManifestWriter(tmp_path / "manifest.jsonl", "run-a")
    writer.record_boundary(_state("z", 2), action=_action("a2"), reward=_reward())
    assert writer.record_boundary(_state("late", 1), action=_action("a1"), reward=_reward()) is None
    assert [row.row_kind for row in load_causal_manifest(writer.path)] == ["boundary", "boundary"]


def test_deterministic_decision_requires_zero_alternative_propensity() -> None:
    row = _decision()
    assert row.exploration_hook == "disabled_pending_operator_go"
    with pytest.raises(CausalManifestError, match="propensity one"):
        ExplorationDecisionRow(
            **{
                **row.__dict__,
                "arm_propensities": (ArmPropensity("arm-a", 0.9), ArmPropensity("arm-b", 0.1)),
            }
        )


def test_randomized_decision_requires_actual_authorization_seed_and_draw() -> None:
    row = _decision(randomized=True)
    with pytest.raises(CausalManifestError, match="externally_authorized"):
        ExplorationDecisionRow(**{**row.__dict__, "exploration_hook": "disabled_pending_operator_go"})


def test_decision_policy_digest_must_match_state() -> None:
    row = _decision()
    with pytest.raises(CausalManifestError, match="state policy digest"):
        ExplorationDecisionRow(**{**row.__dict__, "policy_sha256": SHA_B})


def test_advisory_decision_cannot_claim_actuation() -> None:
    row = _decision()
    with pytest.raises(CausalManifestError, match="advisory decisions require NONE"):
        ExplorationDecisionRow(**{**row.__dict__, "actuation": "SCHEDULE_ARM"})


def test_fore_support_refuses_empty_cache() -> None:
    report = check_fore_support((), target_policy_sha256=SHA_A, target_arms=("arm-a",))
    assert not report.admissible
    assert report.status == "NOT_IDENTIFIED"
    assert "missing_state_action_reward_successor_transitions" in report.blockers


def test_fore_support_refuses_deterministic_log_without_alternative_support() -> None:
    report = check_fore_support(
        (_manifest(), _transition(), _decision(executed=True), _receipt()),
        target_policy_sha256=SHA_A,
        target_arms=("arm-a", "arm-b"),
    )
    assert not report.admissible
    assert "no_positive_logged_propensity:arm-b" in report.blockers


def test_fore_support_admits_structurally_complete_randomized_cache() -> None:
    report = check_fore_support(
        (_manifest(), _transition(), _decision(randomized=True, executed=True), _receipt()),
        target_policy_sha256=SHA_A,
        target_arms=("arm-a", "arm-b"),
    )
    assert report.admissible
    assert report.status == "ADMISSIBLE_STRUCTURAL_INPUT"


def test_hcm_l4_returns_no_rows_without_inventing_authority() -> None:
    report = hcm_l4_residual_check((), cross_fitted_predictions={}, negative_control_names=())
    assert report.status == "NO_ROWS"
    assert not report.unconfounded_certificate


def test_hcm_l4_requires_run_manifest_and_pair_custody() -> None:
    transition = _transition()
    bad = TransitionRow(**{**transition.__dict__, "pair_id": "__aggregate_all_pairs__"})
    report = hcm_l4_residual_check(
        (bad,),
        cross_fitted_predictions={bad.row_id: -0.3},
        negative_control_names=(),
    )
    assert report.status == "INVALID_INPUT"
    assert any(item.startswith("missing_run_treatment_manifest:") for item in report.blockers)
    assert any(item.startswith("missing_pair_outcome_custody:") for item in report.blockers)


def test_hcm_l4_requires_frozen_run_positive_control() -> None:
    transitions = (_transition("run-a", "a"), _transition("run-b", "b"))
    predictions = {row.row_id: -0.3 for row in transitions}
    report = hcm_l4_residual_check(
        (_manifest("run-a"), _manifest("run-b"), *transitions),
        cross_fitted_predictions=predictions,
        negative_control_names=(),
    )
    assert report.status == "INVALID_INPUT"
    assert "missing_frozen_no_update_positive_control" in report.blockers


def test_hcm_l4_refuses_failed_loss_term_closure() -> None:
    apparatus = ApparatusState(
        positive_control="frozen_no_update",
        total_loss=1.0,
        loss_terms=(LossTerm("seg", 1.0, 0.5),),
        negative_controls=freeze_fields({"future_pair": 1.0}),
    )
    transitions = (_transition("run-a", "a", apparatus=apparatus), _transition("run-b", "b", apparatus=apparatus))
    predictions = {row.row_id: -0.3 for row in transitions}
    report = hcm_l4_residual_check(
        (_manifest("run-a"), _manifest("run-b"), *transitions),
        cross_fitted_predictions=predictions,
        negative_control_names=("future_pair",),
    )
    assert report.status == "REFUSED_APPARATUS"
    assert any(item.startswith("loss_term_closure_failed:") for item in report.blockers)


def test_hcm_l4_fires_negative_control_moment_across_whole_runs() -> None:
    apparatus_a = ApparatusState(
        positive_control="frozen_no_update",
        total_loss=0.5,
        loss_terms=(LossTerm("seg", 1.0, 0.5),),
        negative_controls=freeze_fields({"future_pair": 1.0}),
    )
    apparatus_b = ApparatusState(
        positive_control="frozen_no_update",
        total_loss=0.5,
        loss_terms=(LossTerm("seg", 1.0, 0.5),),
        negative_controls=freeze_fields({"future_pair": 1.0}),
    )
    transitions = (
        _transition("run-a", "a", apparatus=apparatus_a),
        _transition("run-b", "b", apparatus=apparatus_b),
    )
    predictions = {transitions[0].row_id: -1.2, transitions[1].row_id: -2.2}
    report = hcm_l4_residual_check(
        (_manifest("run-a"), _manifest("run-b"), *transitions),
        cross_fitted_predictions=predictions,
        negative_control_names=("future_pair",),
    )
    assert report.status == "FIRED_GRAPH_FALSIFICATION"
    assert report.fired
    assert not report.unconfounded_certificate


def test_hcm_l4_quiet_result_is_not_an_unconfounded_certificate() -> None:
    apparatus = ApparatusState(
        positive_control="frozen_no_update",
        total_loss=0.5,
        loss_terms=(LossTerm("seg", 1.0, 0.5),),
        negative_controls=freeze_fields({"future_pair": 0.0}),
    )
    transitions = (
        _transition("run-a", "quiet-a", apparatus=apparatus),
        _transition("run-b", "quiet-b", apparatus=apparatus),
    )
    report = hcm_l4_residual_check(
        (_manifest("run-a"), _manifest("run-b"), *transitions),
        cross_fitted_predictions={row.row_id: -0.3 for row in transitions},
        negative_control_names=("future_pair",),
    )
    assert report.status == "QUIET_NOT_CERTIFIED"
    assert not report.fired
    assert not report.unconfounded_certificate


def test_public_schema_id_is_versioned() -> None:
    assert SCHEMA_ID == "pact.causal_manifest.v1"


def test_trainer_wiring_is_default_on_and_covers_required_boundaries() -> None:
    source = (REPO / "experiments/train_levelset_witness_realized_through_R_mlx.py").read_text()
    assert '"mode": "default_on_read_only"' in source
    assert 'boundary_kind="baseline"' in source
    assert 'boundary_kind="verdict"' in source
    assert 'causal_boundary_kind="final"' in source
    assert "_record_causal_boundary(" in source
    assert 'add_argument("--causal-manifest' not in source


def test_trainer_checkpoint_wiring_hashes_preserved_resume_bundle() -> None:
    source = (REPO / "experiments/train_levelset_witness_realized_through_R_mlx.py").read_text()
    assert 'written.get("resume_preserved", written["resume_latest"])' in source
    assert "checkpoint_path=out_dir / _causal_resume_name" in source
    assert 'pair_id="__aggregate_all_pairs__"' in source
