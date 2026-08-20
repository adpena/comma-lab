from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from tac.witness_control import telemetry_binding
from tac.witness_control.taskspace_interaction_costate_bridge_v1 import (
    ACTION_SEMANTICS,
    COSTATE_FACTOR_AXES,
    ActionEffectPredictionV1,
    CostateFactorizationV1,
    CurrentExactReceiverReplayV1,
    DecoderPlacementV1,
    FeedbackExactBaseBindingV1,
    FullLatticeTeacherEndpointV1,
    TaskspaceInteractionCostateBridgeError,
    _conditional_sign_reversal,
    bridge_receipt_bytes,
    build_taskspace_interaction_costate_bridge_v1,
)
from tac.witness_dsl.taskspace_g8_a3_interaction_feedback import (
    build_taskspace_g8_a3_interaction_feedback,
    feedback_receipt_bytes,
)

REPO = Path(__file__).resolve().parents[4]
G18_TEST = REPO / "src/tac/witness_dsl/tests/test_taskspace_g8_a3_interaction_feedback.py"
XCODEC_RECEIPT = (
    REPO
    / ".omx/research/original_taskspace_inverse_witness_codec_20260725"
    / "ep725_lossless_xcodec_recode_20260726/receipt.json"
)
POPULATION_GLOBAL_RECEIPT = (
    REPO
    / ".omx/research/original_taskspace_inverse_witness_codec_20260725"
    / "ep725_population_global_recode_v2_20260726_r2/receipt.json"
)
SPEC = importlib.util.spec_from_file_location("g19_frozen_g18_fixture", G18_TEST)
assert SPEC is not None and SPEC.loader is not None
fixture = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fixture
SPEC.loader.exec_module(fixture)


def _sha(payload: bytes | str) -> str:
    if isinstance(payload, str):
        payload = payload.encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _feedback(tmp_path: Path) -> tuple[dict[str, Any], bytes]:
    value = build_taskspace_g8_a3_interaction_feedback(fixture._receipt(tmp_path))
    return value, feedback_receipt_bytes(value)


def _baseline(feedback: dict[str, Any]) -> dict[str, Any]:
    return next(
        row["measurement"] for row in feedback["row_inventory"]["occurrences"] if row["occurrence_id"] == "g0:0"
    )


def _duplicate_measurement_occurrences(
    feedback: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    first_by_id: dict[str, dict[str, Any]] = {}
    for occurrence in feedback["row_inventory"]["occurrences"]:
        measurement_id = occurrence["measurement"]["measurement_id"]
        first = first_by_id.get(measurement_id)
        if first is not None:
            return first, occurrence
        first_by_id[measurement_id] = occurrence
    first = next(
        occurrence
        for occurrence in feedback["row_inventory"]["occurrences"]
        if occurrence["occurrence_id"] == "g0:0"
    )
    duplicate = copy.deepcopy(first)
    duplicate["occurrence_id"] = "proof_dependency_duplicate:g0:0"
    feedback["row_inventory"]["occurrences"].append(duplicate)
    feedback["row_inventory"]["occurrence_count"] += 1
    return first, duplicate


def _binding(feedback: dict[str, Any], payload: bytes) -> FeedbackExactBaseBindingV1:
    baseline = _baseline(feedback)
    selected = feedback["row_inventory"]["selected_research_row"]
    custody = feedback["pointer_observation_custody"]
    return FeedbackExactBaseBindingV1(
        feedback_sha256=_sha(payload),
        source_receipt_sha256=feedback["source_receipt_sha256"],
        canonical_receipt_sha256=feedback["canonical_receipt_sha256"],
        source_lane_id=feedback["source_lane_id"],
        axis=feedback["axis"],
        baseline_bundle_sha256=baseline["baseline_bundle_sha256"],
        baseline_archive_sha256=baseline["selected_archive_sha256"],
        baseline_archive_nbytes=baseline["selected_archive_bytes"],
        selected_research_archive_sha256=selected["selected_archive_sha256"],
        frozen_scorer_sha256=baseline["scorer_evidence"]["frozen_scorer_sha256"],
        pointer_manifest_sha256=custody["manifest_sha256"],
        pointer_start_sha256=custody["pointer_start"]["pointer_sha256"],
        pointer_start_target_score=custody["pointer_start"]["target_score"],
        pointer_latest_sha256=custody["pointer_latest"]["pointer_sha256"],
        pointer_latest_target_score=custody["pointer_latest"]["target_score"],
        population_manifest_sha256=_sha("ep725-n600-population-manifest"),
        source_pair_ids=tuple(feedback["four_way_z_t_h_partition"]["source_pair_ids"]),
    )


def _interaction_components(feedback: dict[str, Any], interaction: dict[str, Any]) -> dict[str, Any]:
    occurrences = feedback["row_inventory"]["occurrences"]
    treatment_id = interaction["treatment_id"]
    g8_a = next(
        row["measurement"]
        for row in occurrences
        if row.get("kind") == "g8_a" and row.get("treatment_id") == treatment_id
    )
    g0_a = next(
        row["measurement"]
        for row in occurrences
        if row.get("kind") == "matched_g0_a" and row.get("treatment_id") == treatment_id
    )
    g8_pass = next(
        row["measurement"]
        for row in occurrences
        if row.get("kind") == "g8_pass_a" and row["g8_branch"]["proposal_id"] == interaction["g8_proposal_id"]
    )
    baseline = _baseline(feedback)

    def residual(field_name: str) -> float:
        return g8_a[field_name] - g8_pass[field_name] - g0_a[field_name] + baseline[field_name]

    return {
        "delta_d_seg": residual("d_seg"),
        "delta_d_pose": residual("d_pose"),
        "delta_archive_bytes": int(residual("selected_archive_bytes")),
    }


def _factorization(key: str) -> CostateFactorizationV1:
    return CostateFactorizationV1(
        atom_key=f"{key}.atom",
        group_key=f"{key}.group",
        shard_key=f"{key}.shard",
        global_key="g19.global",
        weights=f"{key}.weights",
        codes=f"{key}.codes",
        time=f"{key}.time",
        population="ep725.n600",
        semantic=f"{key}.semantic",
        realization=f"{key}.realization",
        pose=f"{key}.pose",
        entropy=f"{key}.entropy",
        analytic_vs_learned="hybrid",
    )


def _decoder_placement(key: str) -> DecoderPlacementV1:
    return DecoderPlacementV1(
        generic_decoder_program_sha256=_sha(f"{key}.decoder"),
        generic_decoder_source_nbytes=4096,
        generic_operations=("solve", "factor_expand", "entropy_decode"),
        video_specific_payload_fields=(f"{key}.selector", f"{key}.latent"),
        decoder_runtime_seconds=12.5,
        decoder_runtime_receipt_sha256=_sha(f"{key}.runtime"),
        deterministic_portability_receipt_sha256=_sha(f"{key}.portable"),
        placement_receipt_sha256=_sha(f"{key}.placement"),
    )


def _historical_teacher(
    binding: FeedbackExactBaseBindingV1,
    *,
    replay: bool = False,
) -> FullLatticeTeacherEndpointV1:
    current_replay = None
    if replay:
        current_replay = CurrentExactReceiverReplayV1(
            replay_id="g19.current.full_lattice.n2",
            measurement_receipt_sha256=_sha("teacher.measurement"),
            receiver_receipt_sha256=_sha("teacher.receiver"),
            decoded_output_sha256=_sha("teacher.decoded"),
            archive_sha256=_sha("teacher.archive"),
            archive_nbytes=binding.baseline_archive_nbytes + 100_000,
            d_seg=0.0,
            d_pose=0.0,
            baseline_bundle_sha256=binding.baseline_bundle_sha256,
            population_manifest_sha256=binding.population_manifest_sha256,
            frozen_scorer_sha256=binding.frozen_scorer_sha256,
            pointer_latest_sha256=binding.pointer_latest_sha256,
            axis=binding.axis,
            source_pair_ids=binding.source_pair_ids,
        )
    return FullLatticeTeacherEndpointV1(
        teacher_key="historical.v10.full_lattice",
        historical_receipt_sha256=_sha("historical.lattice.receipt"),
        lattice_payload_sha256=_sha("historical.lattice.payload"),
        lattice_payload_nbytes=binding.baseline_archive_nbytes + 200_000,
        reported_d_seg=0.0,
        reported_d_pose=0.0,
        current_replay=current_replay,
    )


def _predictions(
    feedback: dict[str, Any], binding: FeedbackExactBaseBindingV1
) -> tuple[ActionEffectPredictionV1, ActionEffectPredictionV1]:
    baseline = _baseline(feedback)
    transition = feedback["transition_inventory"]["transitions"][0]
    interaction = feedback["interaction_inventory"]["interactions"][0]
    components = _interaction_components(feedback, interaction)
    transition_prediction = ActionEffectPredictionV1(
        proposal_key="g19.add_a.v1",
        effect_id=transition["transition_id"],
        effect_kind="transition",
        exact_base_measurement_id=transition["before_measurement_id"],
        exact_base_archive_sha256=baseline["selected_archive_sha256"],
        baseline_bundle_sha256=baseline["baseline_bundle_sha256"],
        population_manifest_sha256=binding.population_manifest_sha256,
        action_semantics="ADD",
        source_reservoirs=(),
        target_reservoirs=("A",),
        factorization=_factorization("add_a"),
        decoder_placement=_decoder_placement("add_a"),
        homotopy_path_key="g19.compact.path",
        homotopy_step_index=0,
        predicted_delta_d_seg=transition["delta_d_seg"],
        predicted_delta_d_pose=0.5 * transition["delta_d_pose"],
        predicted_delta_archive_bytes=transition["delta_selected_archive_bytes"],
        predicted_postcompression_archive_nbytes=(
            baseline["selected_archive_bytes"] + transition["delta_selected_archive_bytes"]
        ),
    )
    interaction_prediction = ActionEffectPredictionV1(
        proposal_key="g19.migrate_grammar_to_a.v1",
        effect_id=interaction["interaction_id"],
        effect_kind="interaction",
        exact_base_measurement_id=baseline["measurement_id"],
        exact_base_archive_sha256=baseline["selected_archive_sha256"],
        baseline_bundle_sha256=baseline["baseline_bundle_sha256"],
        population_manifest_sha256=binding.population_manifest_sha256,
        action_semantics="MIGRATE",
        source_reservoirs=("semantic_grammar",),
        target_reservoirs=("A",),
        factorization=_factorization("migrate_grammar"),
        decoder_placement=_decoder_placement("migrate_grammar"),
        homotopy_path_key="g19.compact.path",
        homotopy_step_index=1,
        predicted_delta_d_seg=0.5 * components["delta_d_seg"],
        predicted_delta_d_pose=0.5 * components["delta_d_pose"],
        predicted_delta_archive_bytes=int(0.5 * components["delta_archive_bytes"]),
        predicted_postcompression_archive_nbytes=None,
        predicted_joint_score_effect=0.5 * interaction["interaction_I"],
    )
    return transition_prediction, interaction_prediction


def test_bridge_is_deterministic_nonadditive_and_real_api_consumable(tmp_path: Path) -> None:
    feedback, payload = _feedback(tmp_path)
    binding = _binding(feedback, payload)
    predictions = _predictions(feedback, binding)
    first = build_taskspace_interaction_costate_bridge_v1(
        payload,
        binding=binding,
        predictions=predictions,
    )
    second = build_taskspace_interaction_costate_bridge_v1(
        feedback,
        binding=binding,
        predictions=predictions,
    )
    assert bridge_receipt_bytes(first) == bridge_receipt_bytes(second)
    assert first["truth"]["score_claim"] is False
    assert first["truth"]["ledger_append_performed"] is False
    assert first["action_semantics_contract"]["closed_actions"] == list(ACTION_SEMANTICS)
    assert first["action_semantics_contract"]["additive_independence_assumed"] is False
    assert first["acquisition"]["api_invoked"] is True
    assert first["acquisition"]["additive_independence_assumed"] is False
    telemetry_rows = first["telemetry"]["rows"]
    parsed = telemetry_binding.parse_log_lines([json.dumps(row, sort_keys=True) for row in telemetry_rows])
    assert parsed == telemetry_rows


def test_bridge_preserves_transition_and_four_cell_interaction_as_distinct_units(
    tmp_path: Path,
) -> None:
    feedback, payload = _feedback(tmp_path)
    binding = _binding(feedback, payload)
    result = build_taskspace_interaction_costate_bridge_v1(
        payload,
        binding=binding,
        predictions=_predictions(feedback, binding),
    )
    by_kind = {row["effect_kind"]: row for row in result["calibrations"]}
    transition = feedback["transition_inventory"]["transitions"][0]
    assert by_kind["transition"]["realized"] == {
        "delta_d_seg": transition["delta_d_seg"],
        "delta_d_pose": transition["delta_d_pose"],
        "delta_archive_bytes": transition["delta_selected_archive_bytes"],
        "joint_score_effect": transition["exact_score_delta"],
    }
    interaction = feedback["interaction_inventory"]["interactions"][0]
    expected_components = _interaction_components(feedback, interaction)
    assert by_kind["interaction"]["realized"] == {
        **expected_components,
        "joint_score_effect": interaction["interaction_I"],
    }
    assert by_kind["interaction"]["component_effect_semantics"].startswith("four_cell")
    assert by_kind["interaction"]["predicted"]["joint_score_derivation"].startswith("independently_predicted_four_cell")
    assert by_kind["interaction"]["additive_independence_assumed"] is False


def test_controller_state_is_content_addressed_and_duplicate_safe(tmp_path: Path) -> None:
    feedback, payload = _feedback(tmp_path)
    binding = _binding(feedback, payload)
    predictions = _predictions(feedback, binding)
    first = build_taskspace_interaction_costate_bridge_v1(
        payload,
        binding=binding,
        predictions=predictions,
    )
    second = build_taskspace_interaction_costate_bridge_v1(
        payload,
        binding=binding,
        predictions=predictions,
        prior_controller_state=first["controller_state"],
    )
    assert second["controller_state"] == first["controller_state"]
    assert all(row["duplicate_observation"] is True for row in second["calibrations"])
    assert all(row["observation_count"] == 1 for row in second["controller_state"]["proposals"])


def test_measurement_identity_domains_preserve_distinct_proof_dependencies(
    tmp_path: Path,
) -> None:
    feedback, _payload = _feedback(tmp_path)
    mutated = copy.deepcopy(feedback)
    first, second = _duplicate_measurement_occurrences(mutated)
    alternate_proof_sha = _sha("alternate-measurement-proof")
    assert (
        first["measurement"]["scorer_evidence"]["measurement_receipt_sha256"]
        == second["measurement"]["scorer_evidence"]["measurement_receipt_sha256"]
    )
    second["measurement"]["scorer_evidence"]["measurement_receipt_sha256"] = (
        alternate_proof_sha
    )
    payload = feedback_receipt_bytes(mutated)

    result = build_taskspace_interaction_costate_bridge_v1(
        payload,
        binding=_binding(mutated, payload),
    )

    measurement_id = first["measurement"]["measurement_id"]
    row = next(
        item
        for item in result["measurement_identity_domains"]["rows"]
        if item["measurement_id"] == measurement_id
    )
    assert row["occurrence_count"] >= 2
    assert row["proof_dependency_receipt_count"] == 2
    assert row["proof_dependency_receipt_sha256s"] == sorted(
        {
            first["measurement"]["scorer_evidence"]["measurement_receipt_sha256"],
            alternate_proof_sha,
        }
    )


def test_measurement_semantic_control_identity_aliasing_is_refused(
    tmp_path: Path,
) -> None:
    feedback, _payload = _feedback(tmp_path)
    mutated = copy.deepcopy(feedback)
    _first, second = _duplicate_measurement_occurrences(mutated)
    second["measurement"]["receiver_receipt_sha256"] = _sha(
        "different-realized-receiver"
    )
    payload = feedback_receipt_bytes(mutated)

    with pytest.raises(
        TaskspaceInteractionCostateBridgeError,
        match="semantic-control ID aliases different realized objects",
    ):
        build_taskspace_interaction_costate_bridge_v1(
            payload,
            binding=_binding(mutated, payload),
        )


@pytest.mark.parametrize(
    ("field_name", "replacement_value", "match"),
    [
        ("feedback_sha256", _sha("different-feedback"), "feedback_sha256"),
        ("source_receipt_sha256", _sha("different-source"), "source_receipt_sha256"),
        ("baseline_archive_sha256", _sha("different-base"), "baseline_archive_sha256"),
        ("pointer_latest_sha256", _sha("different-pointer"), "pointer_latest_sha256"),
        ("pointer_latest_target_score", 0.123, "pointer_latest_target_score"),
        (
            "selected_research_archive_sha256",
            _sha("different-selected"),
            "selected_research_archive_sha256",
        ),
    ],
)
def test_bridge_fails_closed_on_hash_pointer_target_or_archive_drift(
    tmp_path: Path,
    field_name: str,
    replacement_value: object,
    match: str,
) -> None:
    feedback, payload = _feedback(tmp_path)
    binding = replace(_binding(feedback, payload), **{field_name: replacement_value})
    with pytest.raises(TaskspaceInteractionCostateBridgeError, match=match):
        build_taskspace_interaction_costate_bridge_v1(payload, binding=binding)


def test_bridge_recomputes_arithmetic_even_when_feedback_hash_is_rebound(tmp_path: Path) -> None:
    feedback, payload = _feedback(tmp_path)
    mutated = copy.deepcopy(feedback)
    mutated["transition_inventory"]["transitions"][0]["exact_score_delta"] += 0.25
    mutated_payload = feedback_receipt_bytes(mutated)
    binding = replace(_binding(feedback, payload), feedback_sha256=_sha(mutated_payload))
    with pytest.raises(TaskspaceInteractionCostateBridgeError, match="exact_score_delta"):
        build_taskspace_interaction_costate_bridge_v1(mutated_payload, binding=binding)


def test_pointer_observation_accepts_additive_custody_without_losing_authority(tmp_path: Path) -> None:
    feedback, _payload = _feedback(tmp_path)
    enriched = copy.deepcopy(feedback)
    for name in ("pointer_start", "pointer_latest"):
        enriched["pointer_observation_custody"][name].update(
            {
                "pointer_bytes": 17_319,
                "pointer_path": "/repo/.omx/state/canonical_frontier_pointer.json",
                "pointer_device": 1,
                "pointer_inode": 2,
                "pointer_mtime_ns": 3,
                "derived_planning_only": True,
                "evaluation_claim": False,
                "pointer_moved": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "research_only": True,
                "score_claim": False,
                "selected_archive_sha256": None,
                "selected_axis": "official_leaderboard",
                "selected_custody": "external target only",
                "selected_evidence_grade": "[official-leaderboard display]",
                "selected_score_precision": "official_display",
                "selected_source": "upstream_official_leaderboard",
                "selected_source_kind": "external_public_leaderboard_target",
                "selection_rule": "minimum current authority target",
                "last_refreshed_utc": "2026-07-26T00:00:00Z",
                "source_snapshot_at_utc": "2026-07-26T00:00:00Z",
            }
        )
    payload = feedback_receipt_bytes(enriched)
    result = build_taskspace_interaction_costate_bridge_v1(
        payload,
        binding=_binding(enriched, payload),
    )
    assert result["binding"]["pointer_latest_target_score"] == enriched["pointer_observation_custody"][
        "pointer_latest"
    ]["target_score"]


def test_pointer_observation_additive_authority_laundering_is_refused(tmp_path: Path) -> None:
    feedback, _payload = _feedback(tmp_path)
    mutated = copy.deepcopy(feedback)
    mutated["pointer_observation_custody"]["pointer_latest"]["score_claim"] = True
    payload = feedback_receipt_bytes(mutated)
    with pytest.raises(TaskspaceInteractionCostateBridgeError, match="score_claim"):
        build_taskspace_interaction_costate_bridge_v1(payload, binding=_binding(mutated, payload))


def test_prediction_fails_closed_on_exact_base_or_population_drift(tmp_path: Path) -> None:
    feedback, payload = _feedback(tmp_path)
    binding = _binding(feedback, payload)
    transition, interaction = _predictions(feedback, binding)
    with pytest.raises(TaskspaceInteractionCostateBridgeError, match="exact_base_archive_sha256"):
        build_taskspace_interaction_costate_bridge_v1(
            payload,
            binding=binding,
            predictions=(replace(transition, exact_base_archive_sha256=_sha("wrong-base")),),
        )
    with pytest.raises(TaskspaceInteractionCostateBridgeError, match="population_manifest_sha256"):
        build_taskspace_interaction_costate_bridge_v1(
            payload,
            binding=binding,
            predictions=(replace(interaction, population_manifest_sha256=_sha("wrong-population")),),
        )


def test_nonadd_action_semantics_are_preserved_but_never_upgraded(tmp_path: Path) -> None:
    feedback, payload = _feedback(tmp_path)
    binding = _binding(feedback, payload)
    result = build_taskspace_interaction_costate_bridge_v1(
        payload,
        binding=binding,
        predictions=_predictions(feedback, binding),
    )
    migration = next(row for row in result["calibrations"] if row["action_semantics"] == "MIGRATE")
    assert migration["source_reservoirs"] == ["semantic_grammar"]
    assert migration["target_reservoirs"] == ["A"]
    assert migration["semantic_claim_status"] == "DECLARED_ONLY_G18_HAS_NO_OPERATOR_SEMANTICS"
    ddm = next(
        row
        for row in result["blocked_handoffs"]
        if row["consumer_api"] == "tac.ddm_costate_organ.build_live_ddm_costate"
    )
    assert ddm["status"] == "HINT_ONLY_NOT_CONSUMED"
    assert result["historical_hint_policy"]["v19c_and_fixed_ddm_additive_lists"].startswith("HINT_ONLY")


def test_gauge_representative_hash_is_preserved_but_consumer_stays_blocked(
    tmp_path: Path,
) -> None:
    feedback, payload = _feedback(tmp_path)
    binding = _binding(feedback, payload)
    _transition, interaction = _predictions(feedback, binding)
    evidence_sha = _sha("gauge-equivalence-receipt")
    gauge = replace(
        interaction,
        proposal_key="g19.gauge.representative.v1",
        action_semantics="GAUGE_REPRESENTATIVE",
        source_reservoirs=("semantic_grammar",),
        target_reservoirs=("semantic_grammar",),
        semantic_evidence_sha256=evidence_sha,
    )
    result = build_taskspace_interaction_costate_bridge_v1(
        payload,
        binding=binding,
        predictions=(gauge,),
    )
    calibration = result["calibrations"][0]
    assert calibration["semantic_evidence_sha256"] == evidence_sha
    assert calibration["semantic_claim_status"] == "DECLARED_WITH_UNVERIFIED_EXTERNAL_EVIDENCE_HASH"
    blocker = next(
        row
        for row in result["blocked_handoffs"]
        if row["consumer_api"] == "evaluator_equivalent_gauge_representative_selector"
    )
    assert blocker["status"] == "BLOCKED_NO_LAWFUL_CONSUMER"


def test_invalid_action_shapes_and_prior_state_drift_fail_before_output(tmp_path: Path) -> None:
    feedback, payload = _feedback(tmp_path)
    binding = _binding(feedback, payload)
    transition, _interaction = _predictions(feedback, binding)
    with pytest.raises(TaskspaceInteractionCostateBridgeError, match="MIGRATE source and target"):
        replace(
            transition,
            action_semantics="MIGRATE",
            source_reservoirs=("A",),
            target_reservoirs=("A",),
        )
    first = build_taskspace_interaction_costate_bridge_v1(
        payload,
        binding=binding,
        predictions=(transition,),
    )
    bad_state = copy.deepcopy(first["controller_state"])
    bad_state["population_manifest_sha256"] = _sha("other-population")
    with pytest.raises(TaskspaceInteractionCostateBridgeError, match="axis/population"):
        build_taskspace_interaction_costate_bridge_v1(
            payload,
            binding=binding,
            predictions=(transition,),
            prior_controller_state=bad_state,
        )
    bad_action_state = copy.deepcopy(first["controller_state"])
    bad_action_state["proposals"][0]["action_semantics"] = "OLD_V19_ADD_ONLY"
    with pytest.raises(TaskspaceInteractionCostateBridgeError, match="closed vocabulary"):
        build_taskspace_interaction_costate_bridge_v1(
            payload,
            binding=binding,
            predictions=(transition,),
            prior_controller_state=bad_action_state,
        )
    with pytest.raises(TaskspaceInteractionCostateBridgeError, match="exact FeedbackExactBaseBindingV1"):
        build_taskspace_interaction_costate_bridge_v1(payload, binding={})  # type: ignore[arg-type]


def test_selected_solution_homotopy_and_hierarchical_costates_use_final_archive_mdl(
    tmp_path: Path,
) -> None:
    feedback, payload = _feedback(tmp_path)
    binding = _binding(feedback, payload)
    result = build_taskspace_interaction_costate_bridge_v1(
        payload,
        binding=binding,
        predictions=_predictions(feedback, binding),
    )
    selected = feedback["row_inventory"]["selected_research_row"]
    solution = result["selected_evaluator_solution"]
    assert solution["archive_sha256"] == selected["selected_archive_sha256"]
    assert solution["postcompression_mdl_nbytes"] == selected["selected_archive_bytes"]
    assert solution["raw_tensor_or_uncompressed_member_size_used"] is False
    assert solution["first_feasible_selection_used"] is False
    assert result["solution_representation_policy"]["generic_machinery_placement"] == ("inflate_py_or_inflate_sh")
    assert (
        result["solution_representation_policy"][
            "factorability_is_priced_by_realized_final_archive_bytes_not_label_count"
        ]
        is True
    )

    costates = result["hierarchical_costate_harvest"]
    assert costates["factor_axes"] == list(COSTATE_FACTOR_AXES)
    assert len(costates["rows"]) == 8
    assert {row["granularity"] for row in costates["rows"]} == {
        "atom",
        "group",
        "shard",
        "global",
    }
    assert all(row["additive_rollup_performed"] is False for row in costates["rows"])
    transition_path = next(
        row
        for row in result["rate_distortion_homotopy"]["rows"]
        if row["path_role"] == "exact_current_base_transition_endpoint"
    )
    transition = feedback["transition_inventory"]["transitions"][0]
    assert transition_path["after"]["postcompression_mdl_nbytes"] == (
        _baseline(feedback)["selected_archive_bytes"] + transition["delta_selected_archive_bytes"]
    )
    assert transition_path["raw_tensor_size_used"] is False


def test_terminal_receding_horizon_controller_is_wired_but_partial_universe_is_blocked(
    tmp_path: Path,
) -> None:
    feedback, payload = _feedback(tmp_path)
    binding = _binding(feedback, payload)
    result = build_taskspace_interaction_costate_bridge_v1(
        payload,
        binding=binding,
    )

    contract = result["terminal_controller_contract"]
    assert contract["consumer_schema"] == "tac.taskspace_receding_horizon_decision.v1"
    assert contract["partial_action_universe_selection_forbidden"] is True
    assert contract["functional_quotient_identity"] == (
        "evaluator_cells_x_continuation_equivalence"
    )
    assert contract["exhaustive_endpoint_materialization_required"] is False
    assert contract["unmeasured_branch_requires_exact_optimistic_score_lower_bound"] is True
    assert contract["same_authority_mode_and_axis_as_base_required"] is True
    assert contract["identity_hashes_without_reopenable_proof_receipts_forbidden"] is True
    assert contract["controller_authorizes_pointer_promotion"] is False
    assert contract["selected_archive_requires_paired_contest_cpu_cuda_replay"] is True
    assert contract["call_performed"] is False
    handoff = next(
        row
        for row in result["blocked_handoffs"]
        if row["consumer_api"].endswith("select_receding_horizon_action_v1")
    )
    assert handoff["status"] == "BLOCKED_COMPLETE_EXACT_ENDPOINT_UNIVERSE_OWED_NO_CALL"
    assert (
        "continuation-equivalence identity plus exact proof receipt per endpoint"
        in handoff["required_evidence"]
    )


def test_composition_sign_reversals_are_protected_from_local_marginal_pruning(
    tmp_path: Path,
) -> None:
    assert (
        _conditional_sign_reversal(0.1, -0.2)
        == "LOCALLY_HARMFUL_BUT_COMPOSITIONALLY_BENEFICIAL"
    )
    assert (
        _conditional_sign_reversal(-0.1, 0.2)
        == "LOCALLY_BENEFICIAL_BUT_COMPOSITIONALLY_HARMFUL"
    )
    assert _conditional_sign_reversal(-0.1, -0.2) is None
    assert _conditional_sign_reversal(0.1, 0.2) is None

    feedback, payload = _feedback(tmp_path)
    result = build_taskspace_interaction_costate_bridge_v1(
        payload,
        binding=_binding(feedback, payload),
    )
    protection = result["composition_sign_reversal_protection"]
    expected = []
    baseline_score = _baseline(feedback)["derived_component_total"]
    for interaction in feedback["interaction_inventory"]["interactions"]:
        cells = _interaction_components(feedback, interaction)
        del cells  # component residuals are distinct from conditional score effects
        treatment_id = interaction["treatment_id"]
        g0_a = next(
            row["measurement"]
            for row in feedback["row_inventory"]["occurrences"]
            if row.get("kind") == "matched_g0_a"
            and row.get("treatment_id") == treatment_id
        )
        g8_a = next(
            row["measurement"]
            for row in feedback["row_inventory"]["occurrences"]
            if row.get("kind") == "g8_a" and row.get("treatment_id") == treatment_id
        )
        g8_pass = next(
            row["measurement"]
            for row in feedback["row_inventory"]["occurrences"]
            if row.get("kind") == "g8_pass_a"
            and row["g8_branch"]["proposal_id"] == interaction["g8_proposal_id"]
        )
        classification = _conditional_sign_reversal(
            g0_a["derived_component_total"] - baseline_score,
            g8_a["derived_component_total"] - g8_pass["derived_component_total"],
        )
        if classification is not None:
            expected.append((interaction["interaction_id"], classification))

    assert protection["sign_reversal_count"] == len(expected)
    assert [
        (row["interaction_id"], row["classification"])
        for row in protection["rows"]
    ] == expected
    assert protection["local_marginal_filter_may_prune_sign_reversal_rows"] is False
    assert protection["paired_whole_object_endpoint_enumeration_required"] is True


def test_historical_lattice_is_zero_influence_until_current_exact_replay(
    tmp_path: Path,
) -> None:
    feedback, payload = _feedback(tmp_path)
    binding = _binding(feedback, payload)
    predictions = _predictions(feedback, binding)
    without_teacher = build_taskspace_interaction_costate_bridge_v1(
        payload,
        binding=binding,
        predictions=predictions,
    )
    with_historical = build_taskspace_interaction_costate_bridge_v1(
        payload,
        binding=binding,
        predictions=predictions,
        teacher_endpoint=_historical_teacher(binding),
    )
    teacher = with_historical["full_lattice_teacher_endpoint"]
    assert teacher["status"] == "HISTORICAL_HINT_ONLY_NOT_REPLAYED_ON_CURRENT_EXACT_RECEIVER"
    assert teacher["numeric_homotopy_reference_eligible"] is False
    assert teacher["controller_influence_performed"] is False
    assert with_historical["controller_state"] == without_teacher["controller_state"]
    assert all(row["teacher_comparison"] is None for row in with_historical["rate_distortion_homotopy"]["rows"])


def test_current_exact_lattice_replay_anchors_numeric_homotopy_and_drift_fails(
    tmp_path: Path,
) -> None:
    feedback, payload = _feedback(tmp_path)
    binding = _binding(feedback, payload)
    teacher = _historical_teacher(binding, replay=True)
    result = build_taskspace_interaction_costate_bridge_v1(
        payload,
        binding=binding,
        predictions=_predictions(feedback, binding),
        teacher_endpoint=teacher,
    )
    teacher_row = result["full_lattice_teacher_endpoint"]
    assert teacher_row["status"] == "CURRENT_EXACT_RECEIVER_REPLAY_BOUND_N2_ADVISORY"
    assert teacher_row["distortion_zero_verified_current_receiver"] is True
    transition_path = next(
        row
        for row in result["rate_distortion_homotopy"]["rows"]
        if row["path_role"] == "exact_current_base_transition_endpoint"
    )
    assert transition_path["teacher_comparison_status"] == "CURRENT_EXACT_RECEIVER_BOUND"
    assert transition_path["teacher_comparison"]["distortion_debt_from_zero"] >= 0.0

    assert teacher.current_replay is not None
    bad_replay = replace(
        teacher.current_replay,
        population_manifest_sha256=_sha("wrong.teacher.population"),
    )
    with pytest.raises(TaskspaceInteractionCostateBridgeError, match="population_manifest_sha256"):
        build_taskspace_interaction_costate_bridge_v1(
            payload,
            binding=binding,
            teacher_endpoint=replace(teacher, current_replay=bad_replay),
        )
    nonzero_replay = replace(teacher.current_replay, d_seg=1e-6)
    with pytest.raises(TaskspaceInteractionCostateBridgeError, match="distortion-zero endpoint"):
        build_taskspace_interaction_costate_bridge_v1(
            payload,
            binding=binding,
            teacher_endpoint=replace(teacher, current_replay=nonzero_replay),
        )


def test_factor_and_requantize_storage_action_shapes_are_closed(tmp_path: Path) -> None:
    feedback, payload = _feedback(tmp_path)
    binding = _binding(feedback, payload)
    _transition, interaction = _predictions(feedback, binding)
    factor = replace(
        interaction,
        proposal_key="g19.factor.grammar.v1",
        action_semantics="FACTOR",
        source_reservoirs=("semantic_grammar",),
        target_reservoirs=("A", "entropy_context"),
    )
    result = build_taskspace_interaction_costate_bridge_v1(
        payload,
        binding=binding,
        predictions=(factor,),
    )
    assert result["calibrations"][0]["action_semantics"] == "FACTOR"
    equivalence_sha = _sha("requantize.equivalence")
    requantize = replace(
        interaction,
        proposal_key="g19.requantize.storage.v1",
        action_semantics="REQUANTIZE_STORAGE",
        source_reservoirs=("population_code", "entropy_context"),
        target_reservoirs=("population_code", "entropy_context"),
        semantic_evidence_sha256=equivalence_sha,
    )
    result = build_taskspace_interaction_costate_bridge_v1(
        payload,
        binding=binding,
        predictions=(requantize,),
    )
    assert result["calibrations"][0]["semantic_evidence_sha256"] == equivalence_sha
    with pytest.raises(TaskspaceInteractionCostateBridgeError, match="REQUANTIZE_STORAGE"):
        replace(requantize, semantic_evidence_sha256=None)


def test_real_lossless_xcodec_receipt_is_one_whole_object_rate_only_observation(
    tmp_path: Path,
) -> None:
    feedback, payload = _feedback(tmp_path)
    binding = _binding(feedback, payload)
    receipt_bytes = XCODEC_RECEIPT.read_bytes()
    result = build_taskspace_interaction_costate_bridge_v1(
        payload,
        binding=binding,
        representation_placement_receipts=(receipt_bytes,),
    )
    placements = result["whole_object_representation_placements"]
    assert placements["count"] == 1
    assert placements["additive_double_counting_forbidden"] is True
    row = placements["rows"][0]
    assert row["action_semantics"] == "REQUANTIZE_STORAGE"
    assert row["selected_solution"]["postcompression_mdl_nbytes"] == 81_027
    assert row["exact_effect"]["delta_postcompression_mdl_nbytes"] == -2_811
    assert row["exact_effect"]["rate_score_delta"] == pytest.approx(
        25.0 * -2_811 / 37_545_489,
        abs=1e-15,
    )
    assert row["costate"]["additive_section_attribution_forbidden"] is True
    assert row["placement"]["generic_decoder_rate_cost_in_archive_bytes"] == 0
    assert row["placement"]["selected_video_specific_payload_archive_nbytes"] == 81_027
    parsed = telemetry_binding.parse_log_lines(
        [json.dumps(item, sort_keys=True) for item in result["telemetry"]["rows"]]
    )
    assert parsed == result["telemetry"]["rows"]

    tampered = json.loads(receipt_bytes)
    tampered["exact_delta"]["rate_score_units"] = 0.0
    with pytest.raises(TaskspaceInteractionCostateBridgeError, match="rate score delta"):
        build_taskspace_interaction_costate_bridge_v1(
            payload,
            binding=binding,
            representation_placement_receipts=(tampered,),
        )


def test_population_global_recode_is_conditional_rate_telemetry_not_score_authority(
    tmp_path: Path,
) -> None:
    feedback, payload = _feedback(tmp_path)
    binding = _binding(feedback, payload)
    g20 = XCODEC_RECEIPT.read_bytes()
    g25 = POPULATION_GLOBAL_RECEIPT.read_bytes()
    result = build_taskspace_interaction_costate_bridge_v1(
        payload,
        binding=binding,
        representation_placement_receipts=(g20,),
        population_global_recode_receipts=(g25,),
    )
    recodes = result["population_global_representation_recodes"]
    assert recodes["count"] == 1
    assert recodes["conditional_best_archive_nbytes"] == 80_238
    row = recodes["rows"][0]
    assert row["exact_effect"]["delta_postcompression_mdl_nbytes_versus_g20"] == -789
    assert row["exact_effect"]["rate_score_delta_versus_g20"] == pytest.approx(
        25.0 * -789 / 37_545_489,
        abs=1e-15,
    )
    assert row["telemetry_admission"] == {
        "conditional_rate_planning_eligible": True,
        "exact_score_endpoint_eligible": False,
        "candidate_eligible": False,
        "training_decision_authority": False,
        "reason": (
            "measured global ZIP benefit is usable as conditional rate telemetry; "
            "the selected archive is not a scored video until public decode closure"
        ),
    }
    assert row["public_video_contract"]["expected_frame_count"] == 1_200
    assert row["public_video_contract"]["public_inflate_sh_output_equality_proved"] is False
    assert row["public_video_contract"]["g20_v1_to_g25_v2_state_identity_join_proved"] is False
    handoff = next(
        item for item in result["blocked_handoffs"] if item["consumer_api"] == "population_global_recode_costate_ingest"
    )
    assert handoff["status"] == "BLOCKED_PUBLIC_VIDEO_OUTPUT_CLOSURE_NO_WRITE"

    with pytest.raises(TaskspaceInteractionCostateBridgeError, match="exact ingested G20 parent"):
        build_taskspace_interaction_costate_bridge_v1(
            payload,
            binding=binding,
            population_global_recode_receipts=(g25,),
        )

    tampered = json.loads(g25)
    tampered["exact_delta"]["versus_g20_rate_score_units"] = 0.0
    with pytest.raises(TaskspaceInteractionCostateBridgeError, match="rate delta versus G20"):
        build_taskspace_interaction_costate_bridge_v1(
            payload,
            binding=binding,
            representation_placement_receipts=(g20,),
            population_global_recode_receipts=(tampered,),
        )
