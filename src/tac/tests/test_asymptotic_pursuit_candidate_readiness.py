# SPDX-License-Identifier: MIT
"""Tests for ASYMPTOTIC PURSUIT candidate-readiness assessment + dispatch queue.

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" + Catalog #270 + #313
+ #315 + #240. Covers helper unit tests, end-to-end assessment correctness,
EV-per-dollar ranking, top-1 verdict logic, provenance discipline, and
operator-authorize command generation.

Sister of `tools/asymptotic_pursuit_candidate_readiness_assessment.py` +
`tools/asymptotic_pursuit_dispatch_queue.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

from asymptotic_pursuit_candidate_readiness_assessment import (  # noqa: E402
    CANONICAL_CANDIDATES,
    CandidateReadiness,
    ReadinessAssessment,
    _classify_horizon_class,
    _classify_readiness_verdict,
    _compute_blocking_issues,
    _compute_ev_per_dollar,
    _estimate_dispatch_cost,
    _parse_recipe,
    _prediction_band_recipe_metadata,
    _recipe_targets_contest_exact_eval,
    _resolve_identity_disambiguator_probe,
    assess_candidate,
    assess_candidates,
    build_operator_authorize_command,
    rank_by_ev_per_dollar,
    write_artifact,
    _recipe_session_budget_floor_usd,
)
from asymptotic_pursuit_dispatch_queue import (  # noqa: E402
    CATALOG202_BYPASS_AUDIT_JSON_ENV_VAR,
    CATALOG202_BYPASS_ATTESTATION_ENV_VAR,
    CATALOG202_BYPASS_INTENT_ENV_VAR,
    _catalog202_dirty_tree_attestation,
    _paid_launch_missing_preconditions,
    build_payload,
    build_dispatch_sequence,
    compute_cost_band_rollup,
    compute_operator_attention_budget,
    write_artifact as write_dispatch_queue_artifact,
)


# ---------- helper unit tests ----------


def test_classify_horizon_class_asymptotic_pursuit():
    """Predicted low < 0.120 is ASYMPTOTIC_PURSUIT per Catalog #309."""
    assert _classify_horizon_class(0.05, 0.10) == "asymptotic_pursuit"
    assert _classify_horizon_class(0.08, 0.12) == "asymptotic_pursuit"


def test_classify_horizon_class_frontier_pursuit():
    """0.120 <= low < 0.180 is FRONTIER_PURSUIT."""
    assert _classify_horizon_class(0.13, 0.16) == "frontier_pursuit"
    assert _classify_horizon_class(0.155, 0.175) == "frontier_pursuit"


def test_classify_horizon_class_plateau_adjacent():
    """0.180 <= low is PLATEAU_ADJACENT."""
    assert _classify_horizon_class(0.185, 0.200) == "plateau_adjacent"


def test_classify_horizon_class_unknown_for_none():
    """None predicted band returns unknown."""
    assert _classify_horizon_class(None, None) == "unknown"
    assert _classify_horizon_class(None, 0.15) == "unknown"
    assert _classify_horizon_class(0.10, None) == "unknown"


def test_compute_ev_per_dollar_baseline_beating():
    """Predicted midpoint below baseline 0.19205 → positive EV."""
    ev = _compute_ev_per_dollar(0.10, 0.15, 1.0)
    assert ev > 0
    # midpoint 0.125 → delta 0.067 → EV 0.067/1.0 = 0.067
    assert abs(ev - 0.067) < 0.001


def test_compute_ev_per_dollar_above_baseline_zero():
    """Predicted midpoint above baseline → zero EV."""
    assert _compute_ev_per_dollar(0.20, 0.22, 1.0) == 0.0


def test_compute_ev_per_dollar_none_band_zero():
    """None band → zero EV."""
    assert _compute_ev_per_dollar(None, None, 1.0) == 0.0


def test_prediction_band_recipe_metadata_requires_axis_kind_status():
    """Numeric score-band priors must carry explicit false-authority metadata."""
    kind, axis, status, blockers = _prediction_band_recipe_metadata(
        {"predicted_band": [0.10, 0.13]},
        predicted_low=0.10,
        predicted_high=0.13,
    )
    assert kind is None
    assert axis is None
    assert status is None
    assert "predicted_band_kind_missing" in blockers
    assert "predicted_band_axis_missing" in blockers
    assert "predicted_band_validation_status_missing" in blockers


def test_prediction_band_recipe_metadata_accepts_labelled_research_prior():
    """Research priors may stay visible when kind/axis/status are explicit."""
    kind, axis, status, blockers = _prediction_band_recipe_metadata(
        {
            "predicted_band": [0.10, 0.13],
            "predicted_band_kind": "predicted_score_band",
            "predicted_band_axis": "contest-CPU",
            "predicted_band_validation_status": "research_prior_prebuild",
        },
        predicted_low=0.10,
        predicted_high=0.13,
    )
    assert kind == "predicted_score_band"
    assert axis == "contest-CPU"
    assert status == "research_prior_prebuild"
    assert blockers == ()


def test_estimate_dispatch_cost_t4_paired():
    """T4 paired cost should include CPU eval."""
    cost, wall = _estimate_dispatch_cost("T4", 100)
    # 100 * 20s = 2000s = ~33min on T4 = ~$0.328 + $0.10 CPU
    assert cost > 0.10  # at least CPU cost
    assert wall == 2000


def test_estimate_dispatch_cost_a100_more_expensive_per_epoch():
    """A100 should be more expensive per epoch than T4 (higher $/hr)."""
    t4_cost, _ = _estimate_dispatch_cost("T4", 1000)
    a100_cost, _ = _estimate_dispatch_cost("A100", 1000)
    assert a100_cost > t4_cost


# ---------- blocking issues + readiness verdict ----------


def test_compute_blocking_issues_clean_recipe_no_blockers(tmp_path):
    """Clean state should produce no blocking issues."""
    # Need both files to exist for clean state
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("---\n")
    issues = _compute_blocking_issues(
        full_main_implemented=True,
        full_main_blocker=None,
        council_verdict="PROCEED",
        research_only=False,
        dispatch_enabled=True,
        dispatch_blockers=(),
        predecessor_probe_blocking=False,
        predecessor_probe_id=None,
        recipe_path=recipe,
        trainer_path=Path(__file__),  # An existing file
    )
    assert len(issues) == 0


def test_compute_blocking_issues_council_proceed_with_revisions_flagged():
    """PROCEED_WITH_REVISIONS triggers Catalog #315 blocker."""
    issues = _compute_blocking_issues(
        full_main_implemented=True,
        full_main_blocker=None,
        council_verdict="PROCEED_WITH_REVISIONS",
        research_only=False,
        dispatch_enabled=True,
        dispatch_blockers=(),
        predecessor_probe_blocking=False,
        predecessor_probe_id=None,
        recipe_path=Path(__file__),
        trainer_path=Path(__file__),
    )
    assert any("CATALOG_315" in i for i in issues)


def test_compute_blocking_issues_probe_blocking_flagged():
    """Catalog #313 probe-blocking adds explicit blocker."""
    issues = _compute_blocking_issues(
        full_main_implemented=True,
        full_main_blocker=None,
        council_verdict="PROCEED",
        research_only=False,
        dispatch_enabled=True,
        dispatch_blockers=(),
        predecessor_probe_blocking=True,
        predecessor_probe_id="atw_v2_d4_h_latent_given_scorer_class_20260516",
        recipe_path=Path(__file__),
        trainer_path=Path(__file__),
    )
    assert any("CATALOG_313" in i for i in issues)


def test_compute_blocking_issues_lane_registry_missing_flagged():
    """Unregistered launch surfaces fail closed before operator dispatch."""
    issues = _compute_blocking_issues(
        full_main_implemented=True,
        full_main_blocker=None,
        council_verdict="PROCEED",
        research_only=False,
        dispatch_enabled=True,
        dispatch_blockers=(),
        predecessor_probe_blocking=False,
        predecessor_probe_id=None,
        recipe_path=Path(__file__),
        trainer_path=Path(__file__),
        lane_registered=False,
    )
    assert "LANE_REGISTRY_NOT_REGISTERED" in issues


def test_recipe_targets_contest_exact_eval_requires_literal_target_mode():
    """Dispatch-enabled recipe readiness requires explicit contest target metadata."""
    assert _recipe_targets_contest_exact_eval(
        {"target_modes": ["contest_exact_eval", "research_substrate"]}
    )
    assert not _recipe_targets_contest_exact_eval(
        {"target_modes": ["contest_one_video_replay", "research_substrate"]}
    )


def test_compute_blocking_issues_missing_contest_exact_eval_target_flagged():
    """Paid launch surfaces without contest_exact_eval target metadata fail closed."""
    issues = _compute_blocking_issues(
        full_main_implemented=True,
        full_main_blocker=None,
        council_verdict="PROCEED",
        research_only=False,
        dispatch_enabled=True,
        dispatch_blockers=(),
        predecessor_probe_blocking=False,
        predecessor_probe_id=None,
        recipe_path=Path(__file__),
        trainer_path=Path(__file__),
        contest_exact_eval_targeted=False,
    )
    assert "RECIPE_target_modes_missing_contest_exact_eval" in issues


def test_compute_blocking_issues_horizon_class_mismatch_flagged():
    """Launch recipes must not advertise a stale horizon class."""
    issues = _compute_blocking_issues(
        full_main_implemented=True,
        full_main_blocker=None,
        council_verdict="PROCEED",
        research_only=False,
        dispatch_enabled=True,
        dispatch_blockers=(),
        predecessor_probe_blocking=False,
        predecessor_probe_id=None,
        recipe_path=Path(__file__),
        trainer_path=Path(__file__),
        recipe_horizon_class="frontier_pursuit",
        computed_horizon_class="asymptotic_pursuit",
    )
    assert (
        "RECIPE_horizon_class_mismatch:frontier_pursuit!=asymptotic_pursuit"
        in issues
    )


def test_resolve_identity_disambiguator_probe_blocks_noop_runtime_output(
    tmp_path: Path,
) -> None:
    """Recipe-wired local disambiguators must prove the identity switch is live."""
    probe = tmp_path / ".omx/research/probe.json"
    probe.parent.mkdir(parents=True)
    probe.write_text(
        json.dumps(
            {
                "schema": "z6_predictive_coding_vs_identity_disambiguator_v1",
                "verdict": "pending_paired_exact_eval_json",
                "research_only": True,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "ready_for_paid_dispatch": False,
                "paradigm_claim_allowed": False,
                "blockers": [
                    "no_paired_exact_eval_json",
                    "no_contest_cpu_cuda_pair",
                    "not_score_authority",
                ],
                "result_review": {
                    "identity_predictor_switch_changes_inflate_output": False,
                },
                "inflate_output_comparison": {
                    "score_claim": False,
                    "runtime_output_changed": False,
                    "evidence_axis": "[local-inflate-output advisory]",
                },
            }
        ),
        encoding="utf-8",
    )

    (
        path,
        verdict,
        runtime_changed,
        blockers,
        custody,
    ) = _resolve_identity_disambiguator_probe(
        tmp_path,
        {
            "identity_disambiguator_probe": ".omx/research/probe.json",
            "identity_disambiguator_probe_requires_runtime_output_changed": True,
        },
    )

    assert path.endswith(".omx/research/probe.json")
    assert verdict == "pending_paired_exact_eval_json"
    assert runtime_changed is False
    assert "identity_disambiguator_probe_runtime_output_not_changed" in blockers
    assert (
        "identity_disambiguator_probe_inflate_output_runtime_output_not_changed"
        in blockers
    )
    assert "identity_disambiguator_probe_runtime_custody_missing" in blockers
    assert custody["total_byte_differences"] is None


def test_classify_readiness_verdict_no_blockers_ready():
    """No blockers → READY."""
    assert _classify_readiness_verdict(()) == "READY"


def test_classify_readiness_verdict_probe_blocker_defer():
    """Probe-blocker = DEFER (substantive evidence needed)."""
    assert _classify_readiness_verdict(("CATALOG_313_PROBE_BLOCKING:foo",)) == "DEFER"


def test_classify_readiness_verdict_council_blocker_defer():
    """PROCEED_WITH_REVISIONS = DEFER per Catalog #315."""
    assert _classify_readiness_verdict(("CATALOG_315_COUNCIL_PROCEED_WITH_REVISIONS_NEEDS_ITERATION_TO_OPTIMAL_FORM",)) == "DEFER"


def test_classify_readiness_verdict_recipe_flag_needs_fix():
    """Recipe research_only/dispatch_enabled flags = NEEDS_FIX."""
    assert _classify_readiness_verdict(("RECIPE_research_only=true_OPERATOR_NEEDS_TO_FLIP",)) == "NEEDS_FIX"


def test_classify_readiness_verdict_candidate4c_diagnostic_handoff_defer():
    """Candidate 4c's diagnostic-only training surface is intentional DEFER."""
    assert _classify_readiness_verdict(
        (
            "RECIPE_dispatch_enabled=false",
            "RECIPE_DISPATCH_BLOCKER:"
            "candidate4c_modal_training_recipe_is_diagnostic_only_exact_cuda_handoff_required",
        )
    ) == "DEFER"


def test_classify_readiness_verdict_full_main_blocker_defer():
    """NotImplementedError in _full_main = DEFER (engineering work needed)."""
    assert _classify_readiness_verdict(("CATALOG_240_FULL_MAIN_BLOCKED:RAISES_NotImplementedError",)) == "DEFER"


# ---------- per-candidate assessment ----------


def test_assess_candidate_atw_codec_v2_is_blocked_by_probe():
    """Catalog #313: ATW V2 has BLOCKING probe outcome from 2026-05-16.

    Empirical receipt: probe_id atw_v2_d4_h_latent_given_scorer_class_20260516
    returned INDEPENDENT verdict (MI=0.006385 << 0.5 threshold).
    """
    c = assess_candidate("atw_codec_v2")
    assert c.predecessor_probe_blocking is True
    assert c.predecessor_probe_id == "atw_v2_d4_h_latent_given_scorer_class_20260516"
    assert c.predecessor_probe_verdict == "INDEPENDENT"
    assert c.readiness_verdict == "DEFER"


def test_assess_candidate_atw_v2_1_faiss_pq_visible_after_probe():
    """ATW V2-1 WIP must be visible without becoming dispatch authority."""
    c = assess_candidate("atw_codec_v2_1_faiss_ivf_pq")
    assert c.recipe_basename == "substrate_atw_v2_1_modal_t4_smoke_dispatch"
    assert c.recipe_path is not None
    assert c.trainer_path is not None
    assert c.trainer_path.name == "train_substrate_atw_v2_1.py"
    assert c.lane_maturity == "L0"
    assert c.impl_complete is False
    assert c.full_main_implemented is False
    assert c.full_main_blocker == "RAISES_NotImplementedError"
    assert c.latest_council_verdict == "PROCEED_WITH_REVISIONS"
    assert c.research_only is True
    assert c.dispatch_enabled is False
    assert c.readiness_verdict == "DEFER"
    assert c.predicted_delta_s_band_low is None
    assert c.predicted_delta_s_band_high is None
    assert c.horizon_class == "unknown"
    assert (
        "z6_wave_2_4c_outcome_pending_cross_pollination_per_atw_v2_symposium_revision_5"
        not in c.dispatch_blockers
    )
    assert (
        "z6_wave_2_4c_zeroepoch_exact_outcome_did_not_validate_scorer_logit_channel_delta_below_0_005"
        in c.dispatch_blockers
    )
    assert (
        "faiss_pq_disambiguator_completed_20260518_v3_pool_shared_only_weak_conditioning_mi_0_121512378237"
        in c.dispatch_blockers
    )
    assert (
        "scorer_softmax_sketch_completed_20260518_all_byte_closed_but_best_mi_0_076162617811_weak"
        in c.dispatch_blockers
    )
    assert (
        "selected_next_gate_is_trained_atw_residual_probe_or_raw_scorer_logit_head_design"
        in c.dispatch_blockers
    )
    assert (
        "RECIPE_DISPATCH_BLOCKER:"
        "faiss_pq_v2_sparse_top_k_meaningful_mi_high_cardinality_upper_bound_only_unique_fraction_1_0"
        in c.blocking_issues
    )
    assert c.score_claim is False
    assert c.promotion_eligible is False


def test_assess_candidate_z6_is_blocked_by_council():
    """Catalog #315: Z6 has council verdict PROCEED_WITH_REVISIONS."""
    c = assess_candidate("time_traveler_l5_z6")
    assert c.latest_council_verdict == "PROCEED_WITH_REVISIONS"
    assert c.research_only is True
    assert c.dispatch_enabled is False
    assert c.readiness_verdict == "DEFER"


def test_assess_candidate_z6_has_predicted_band_asymptotic_or_frontier():
    """Z6 design memo declares predicted ΔS band [0.13, 0.16]."""
    c = assess_candidate("time_traveler_l5_z6")
    assert c.predicted_delta_s_band_low is not None
    assert c.predicted_delta_s_band_high is not None
    # The Z6 design memo declares [0.13, 0.16] which is frontier_pursuit
    # band per Catalog #309; the actual asymptotic band is for Z8.
    assert c.horizon_class in ("frontier_pursuit", "asymptotic_pursuit")


def test_assess_candidate_z6_candidate_4c_records_closed_exact_eval_control():
    """Candidate 4c stays visible after paired exact eval closes the zero-epoch control."""
    c = assess_candidate("z6_v2_candidate_4c_scorer_logit")
    assert c.lane_maturity == "L1"
    assert c.impl_complete is True
    assert c.recipe_basename == (
        "substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch"
    )
    assert c.trainer_path is not None
    assert c.trainer_path.name == "train_substrate_time_traveler_l5_z6.py"
    assert c.research_only is False
    assert c.dispatch_enabled is False
    assert c.dispatch_blockers == (
        "candidate4c_modal_training_recipe_is_diagnostic_only_exact_cuda_handoff_required",
    )
    assert c.predecessor_probe_blocking is False
    assert c.recipe_path is not None
    recipe = _parse_recipe(c.recipe_path)
    assert not _recipe_targets_contest_exact_eval(recipe)
    assert recipe["smoke_only"] is True
    assert recipe["smoke_validation_contract"] == "training_artifact_v1"
    assert recipe["horizon_class"] == c.horizon_class == "asymptotic_pursuit"
    assert c.readiness_verdict == "DEFER"
    assert "RECIPE_dispatch_enabled=false" in c.blocking_issues
    assert (
        "RECIPE_DISPATCH_BLOCKER:"
        "candidate4c_modal_training_recipe_is_diagnostic_only_exact_cuda_handoff_required"
    ) in c.blocking_issues
    assert c.predicted_delta_s_band_low == pytest.approx(0.11)
    assert c.predicted_delta_s_band_high == pytest.approx(0.17)
    assert "pending_post_training" in c.predicted_delta_s_provenance
    assert c.predicted_band_kind == "predicted_score_band"
    assert c.predicted_band_axis == "contest-CUDA"
    assert c.predicted_band_validation_status == "pending_post_training"
    assert c.predicted_band_metadata_blockers == ()
    assert c.as_dict()["predicted_score_band"] == pytest.approx([0.11, 0.17])
    assert c.local_identity_disambiguator_probe_path is not None
    assert c.local_identity_disambiguator_probe_path.endswith(
        ".omx/research/z6_candidate4c_identity_archive_pair_disambiguator_20260518_codex.json"
    )
    assert c.local_identity_disambiguator_probe_verdict == (
        "contest_cuda_cpu_recovered_full_lower_below_delta_zeroepoch_control_bad"
    )
    assert c.local_identity_disambiguator_runtime_output_changed is True
    assert c.local_identity_disambiguator_blockers == ()


def test_assess_candidate_z7_is_visible_but_prebuild_gated():
    """Z7 must stay visible in the queue without becoming dispatch authority."""
    c = assess_candidate("time_traveler_l5_z7_lstm_predictive_coding")
    assert c.recipe_basename == (
        "substrate_time_traveler_l5_z7_lstm_predictive_coding_modal_t4_dispatch"
    )
    assert c.recipe_path is not None
    assert c.trainer_path is not None
    assert c.trainer_path.name == (
        "train_substrate_time_traveler_l5_z7_lstm_predictive_coding.py"
    )
    assert c.lane_maturity == "L0"
    assert c.impl_complete is False
    assert c.full_main_implemented is True
    assert c.full_main_blocker is None
    assert c.latest_council_verdict == "PROCEED_WITH_REVISIONS"
    assert c.research_only is True
    assert c.dispatch_enabled is False
    assert c.readiness_verdict == "DEFER"
    assert c.predicted_delta_s_band_low == pytest.approx(0.10)
    assert c.predicted_delta_s_band_high == pytest.approx(0.13)
    assert c.predicted_band_kind == "predicted_score_band"
    assert c.predicted_band_axis == "contest-CPU"
    assert c.predicted_band_validation_status == "research_prior_prebuild"
    assert c.predicted_band_metadata_blockers == ()
    assert c.horizon_class == "asymptotic_pursuit"
    assert "CATALOG_240_FULL_MAIN_BLOCKED:RAISES_NotImplementedError" not in (
        c.blocking_issues
    )
    assert "CATALOG_315_COUNCIL_PROCEED_WITH_REVISIONS_NEEDS_ITERATION_TO_OPTIMAL_FORM" in (
        c.blocking_issues
    )
    assert "RECIPE_research_only=true_OPERATOR_NEEDS_TO_FLIP_AFTER_PHASE_2_COUNCIL" in (
        c.blocking_issues
    )
    assert (
        "z7_dispatch_requires_z6_wave2_candidate_4c_paired_exact_eval_outcome"
        not in c.dispatch_blockers
    )
    assert (
        "z7_dispatch_requires_z6_wave2_candidate_4c_paired_exact_eval_outcome"
        in c.dispatch_blocker_supersessions
    )
    assert (
        "RECIPE_DISPATCH_BLOCKER:"
        "z7_dispatch_requires_z6_wave2_candidate_4c_paired_exact_eval_outcome"
        not in c.blocking_issues
    )
    assert (
        "z7_wave2_probe_requires_paired_exact_eval_json_from_probe_z7_temporal_coherence_vs_static_capacity_disambiguator"
        in c.dispatch_blockers
    )
    assert (
        "z7_score_aware_one_pair_smoke_not_contest_authority"
        in c.dispatch_blockers
    )
    assert (
        "z7_score_aware_packet_requires_paired_exact_eval"
        in c.dispatch_blockers
    )
    assert c.local_identity_disambiguator_probe_path is not None
    assert c.local_identity_disambiguator_probe_path.endswith(
        ".omx/research/probe_z7_temporal_coherence_vs_static_capacity_disambiguator_20260518_codex.json"
    )
    assert c.local_identity_disambiguator_probe_verdict == (
        "pending_paired_exact_eval_json"
    )
    assert c.local_identity_disambiguator_runtime_output_changed is None
    assert c.local_identity_disambiguator_blockers == ()
    assert c.local_identity_disambiguator_custody["decision_rule"][
        "same_archive_bytes_required"
    ] is True
    assert c.local_identity_disambiguator_custody["required_inputs"] == [
        "z7_recurrent_exact_eval_json",
        "static_capacity_control_exact_eval_json",
    ]
    assert c.score_claim is False
    assert c.promotion_eligible is False


def test_assess_candidate_z7_mamba2_scaffold_is_visible_with_score_band_axis():
    """Z7-Mamba-2 WIP must join the queue without delta-band axis confusion."""
    c = assess_candidate("time_traveler_l5_z7_mamba2")
    assert c.recipe_basename == "substrate_time_traveler_l5_z7_mamba2_modal_a100_dispatch"
    assert c.recipe_path is not None
    assert c.trainer_path is not None
    assert c.trainer_path.name == "train_substrate_time_traveler_l5_z7_mamba2.py"
    assert c.lane_maturity == "L0"
    assert c.impl_complete is False
    assert c.full_main_implemented is False
    assert c.full_main_blocker == "RAISES_NotImplementedError"
    assert c.research_only is True
    assert c.dispatch_enabled is False
    assert c.readiness_verdict == "DEFER"
    assert c.predicted_delta_s_band_low == pytest.approx(0.167)
    assert c.predicted_delta_s_band_high == pytest.approx(0.184)
    assert c.predicted_band_kind == "predicted_score_band"
    assert c.predicted_band_axis == "contest-CPU"
    assert c.predicted_band_validation_status == "research_prior_prebuild"
    assert c.as_dict()["predicted_score_band"] == pytest.approx([0.167, 0.184])
    assert c.horizon_class == "frontier_pursuit"
    assert "CATALOG_240_FULL_MAIN_BLOCKED:RAISES_NotImplementedError" in c.blocking_issues
    assert (
        "z7_mamba2_trainer_full_main_raises_NotImplementedError_per_catalog_240"
        in c.dispatch_blockers
    )
    assert (
        "z7_mamba2_mamba_ssm_pypi_install_must_succeed_in_modal_a100_image_pre_dispatch"
        in c.dispatch_blockers
    )
    assert c.score_claim is False
    assert c.promotion_eligible is False


def test_assess_candidate_dp1_pr101_composition_maps_existing_dual_stack():
    """Research-wave DP1+PR101 must join the existing byte-closed DP1 stack."""
    c = assess_candidate("dp1_pr101_composition")
    assert c.recipe_basename == "substrate_pr101_with_dp1_prior_modal_cpu_smoke_dispatch"
    assert c.recipe_path is not None
    assert c.trainer_path is not None
    assert c.trainer_path.name == "train_substrate_pr101_with_dp1_prior_regularizer.py"
    assert c.lane_maturity == "L1"
    assert c.impl_complete is True
    assert c.full_main_implemented is False
    assert c.full_main_blocker == "RAISES_NotImplementedError"
    assert c.research_only is True
    assert c.dispatch_enabled is False
    assert c.readiness_verdict == "DEFER"
    assert c.horizon_class == "unknown"
    assert c.as_dict()["predicted_score_band"] is None
    assert c.predicted_delta_s_band_low is None
    assert c.predicted_delta_s_band_high is None
    assert "CATALOG_240_FULL_MAIN_BLOCKED:RAISES_NotImplementedError" in c.blocking_issues
    assert "RECIPE_research_only=true_OPERATOR_NEEDS_TO_FLIP_AFTER_PHASE_2_COUNCIL" in (
        c.blocking_issues
    )
    assert (
        "phase_2_council_approval_required_to_lift_full_main_NotImplementedError"
        in c.dispatch_blockers
    )
    assert (
        "dp1_prior_lambda_disambiguator_landing_required_after_l1_noop_probe"
        in c.dispatch_blockers
    )
    assert c.score_claim is False
    assert c.promotion_eligible is False


def test_assess_candidate_lane_17_imp_visible_but_symposium_gated():
    """Research-wave lane_17_imp must join readiness without becoming authority."""
    c = assess_candidate("lane_17_imp")
    assert c.recipe_basename == "lane_17_imp_cycle0_vastai_4090_timing_smoke_dispatch"
    assert c.recipe_path is not None
    assert c.trainer_path is not None
    assert c.trainer_path.name == "train_imp_cycle.py"
    assert c.lane_maturity == "L2"
    assert c.impl_complete is True
    assert c.full_main_implemented is True
    assert c.full_main_blocker is None
    assert c.latest_council_verdict == "PROCEED_WITH_REVISIONS"
    assert c.research_only is True
    assert c.dispatch_enabled is False
    assert c.readiness_verdict == "DEFER"
    assert c.predicted_delta_s_band_low == pytest.approx(0.17705)
    assert c.predicted_delta_s_band_high == pytest.approx(0.18705)
    assert c.predicted_band_kind == "predicted_score_band"
    assert c.predicted_band_axis == "contest-CPU"
    assert c.predicted_band_validation_status == "research_wave_prior_prebuild"
    assert c.predicted_band_metadata_blockers == ()
    assert c.horizon_class == "frontier_pursuit"
    assert "CATALOG_315_COUNCIL_PROCEED_WITH_REVISIONS_NEEDS_ITERATION_TO_OPTIMAL_FORM" in (
        c.blocking_issues
    )
    assert "RECIPE_research_only=true_OPERATOR_NEEDS_TO_FLIP_AFTER_PHASE_2_COUNCIL" in (
        c.blocking_issues
    )
    assert (
        "lane_17_imp_requires_catalog308_cycle0_empirical_regression_ratio_disambiguation"
        in c.dispatch_blockers
    )
    assert (
        "lane_17_imp_requires_catalog123_score_gradient_saliency_sidecar_for_authority"
        in c.dispatch_blockers
    )
    assert c.score_claim is False
    assert c.promotion_eligible is False


def test_assess_candidate_c6_falsified_catalog324_band_not_ranked_as_ev():
    """C6's disabled random-init band stays suppressed after Catalog #324."""
    c = assess_candidate("c6_e4_mdl_ibps")
    assert c.dispatch_enabled is False
    assert c.readiness_verdict == "DEFER"
    assert c.predicted_delta_s_band_low is None
    assert c.predicted_delta_s_band_high is None
    assert "falsified_recipe_band_suppressed_by_catalog_324" in (
        c.predicted_delta_s_provenance
    )
    assert c.ev_per_dollar == 0.0


def test_assess_candidate_tt5l_blockers_refresh_to_current_campaign_state():
    """TT5L readiness should not keep stale Dykstra/composition blockers."""
    c = assess_candidate("time_traveler_l5_autonomy")
    assert c.dispatch_enabled is False
    assert c.research_only is True
    assert c.readiness_verdict == "DEFER"
    assert c.predecessor_probe_id == "symposium_866_tt5l_foveation_lapose_REFUSE_20260517"
    assert "RECIPE_research_only=true_OPERATOR_NEEDS_TO_FLIP_AFTER_PHASE_2_COUNCIL" in (
        c.blocking_issues
    )
    assert (
        "tools_check_substrate_dykstra_feasibility_polytope_emitted_at_omx_state_dykstra_feasibility_time_traveler_l5_json"
        not in c.dispatch_blockers
    )
    assert (
        "predicted_target_revised_to_polytope_intersection_lower_bound_not_5_move_sum_in_both_unwind_memo_and_this_recipe"
        not in c.dispatch_blockers
    )
    assert (
        "requires_c1_z5_tt5l_probe_disambiguator_before_architecture_lock"
        in c.dispatch_blockers
    )
    assert (
        "l5_v2_probe_gate_artifact_semantics_invalid_probe_verdict_sha256_mismatch"
        not in c.dispatch_blockers
    )
    assert (
        "l5_v2_probe_gate_artifact_semantics_invalid_probe_blockers_nonempty"
        in c.dispatch_blockers
    )
    assert (
        "l5_v2_probe_gate_artifact_semantics_missing_eligible_observations_c1_z5_tt5l"
        in c.dispatch_blockers
    )
    assert "prediction_band_not_dispatch_ready" in c.dispatch_blockers
    assert any(
        blocker.startswith("l5_v2_tt5l_modal_provider_blocker_active")
        for blocker in c.dispatch_blockers
    )


# ---------- ranking ----------


def test_rank_by_ev_per_dollar_ready_before_needs_fix_before_defer():
    """Tier order: READY > NEEDS_FIX > DEFER, then by EV/$ within tier."""
    fake_ready = CandidateReadiness(
        substrate_id="ready_low_ev",
        recipe_basename="x",
        recipe_path=None,
        trainer_path=None,
        lane_maturity="L1",
        impl_complete=True,
        full_main_implemented=True,
        full_main_blocker=None,
        latest_council_verdict="PROCEED",
        research_only=False,
        dispatch_enabled=True,
        dispatch_blockers=(),
        predecessor_probe_blocking=False,
        predecessor_probe_id=None,
        predecessor_probe_verdict=None,
        predicted_delta_s_band_low=0.10,
        predicted_delta_s_band_high=0.12,
        predicted_delta_s_provenance="X",
        estimated_dispatch_cost_usd=1.0,
        estimated_dispatch_wall_clock_seconds=100,
        gpu_class="T4",
        min_smoke_gpu="T4",
        cost_band_epochs=100,
        horizon_class="asymptotic_pursuit",
        blocking_issues=(),
        readiness_verdict="READY",
        ev_per_dollar=0.05,
    )
    fake_defer_higher_ev = CandidateReadiness(
        **{**fake_ready.__dict__, "substrate_id": "defer_high_ev", "readiness_verdict": "DEFER", "ev_per_dollar": 1.0},
    )
    ranked = rank_by_ev_per_dollar((fake_ready, fake_defer_higher_ev))
    # READY beats DEFER regardless of EV
    assert ranked[0] == "ready_low_ev"
    assert ranked[1] == "defer_high_ev"


def test_rank_by_ev_per_dollar_within_tier_by_ev():
    """Within same tier, higher EV wins."""
    base = dict(
        recipe_basename="x", recipe_path=None, trainer_path=None,
        lane_maturity="L1", impl_complete=True, full_main_implemented=True,
        full_main_blocker=None, latest_council_verdict="PROCEED",
        research_only=False, dispatch_enabled=True, dispatch_blockers=(),
        predecessor_probe_blocking=False, predecessor_probe_id=None,
        predecessor_probe_verdict=None, predicted_delta_s_band_low=0.10,
        predicted_delta_s_band_high=0.12, predicted_delta_s_provenance="X",
        estimated_dispatch_cost_usd=1.0, estimated_dispatch_wall_clock_seconds=100,
        gpu_class="T4", min_smoke_gpu="T4", cost_band_epochs=100,
        horizon_class="asymptotic_pursuit", blocking_issues=(),
        readiness_verdict="NEEDS_FIX",
    )
    a = CandidateReadiness(substrate_id="a", **base, ev_per_dollar=0.50)
    b = CandidateReadiness(substrate_id="b", **base, ev_per_dollar=0.10)
    ranked = rank_by_ev_per_dollar((b, a))
    assert ranked[0] == "a"
    assert ranked[1] == "b"


# ---------- operator-authorize command ----------


def test_operator_authorize_command_needs_fix_returns_diagnostic():
    """NEEDS_FIX returns a diagnostic comment, not a runnable command."""
    fake = CandidateReadiness(
        substrate_id="x", recipe_basename="x",
        recipe_path=None, trainer_path=None, lane_maturity="L1",
        impl_complete=False, full_main_implemented=True, full_main_blocker=None,
        latest_council_verdict="PROCEED", research_only=False, dispatch_enabled=True,
        dispatch_blockers=(), predecessor_probe_blocking=False,
        predecessor_probe_id=None, predecessor_probe_verdict=None,
        predicted_delta_s_band_low=0.10, predicted_delta_s_band_high=0.12,
        predicted_delta_s_provenance="X", estimated_dispatch_cost_usd=1.0,
        estimated_dispatch_wall_clock_seconds=100, gpu_class="T4",
        min_smoke_gpu="T4", cost_band_epochs=100, horizon_class="asymptotic_pursuit",
        blocking_issues=("X",), readiness_verdict="NEEDS_FIX", ev_per_dollar=0.5,
    )
    cmd = build_operator_authorize_command(fake)
    assert cmd.startswith("#"), f"NEEDS_FIX should return diagnostic comment, got: {cmd}"


def test_operator_authorize_command_ready_invokes_smoke_before_full():
    """READY returns canonical smoke-before-full command per Catalog #167."""
    fake = CandidateReadiness(
        substrate_id="x", recipe_basename="substrate_x_modal_t4_dispatch",
        recipe_path=Path("/some/recipe.yaml"), trainer_path=None, lane_maturity="L1",
        impl_complete=True, full_main_implemented=True, full_main_blocker=None,
        latest_council_verdict="PROCEED", research_only=False, dispatch_enabled=True,
        dispatch_blockers=(), predecessor_probe_blocking=False,
        predecessor_probe_id=None, predecessor_probe_verdict=None,
        predicted_delta_s_band_low=0.10, predicted_delta_s_band_high=0.12,
        predicted_delta_s_provenance="X", estimated_dispatch_cost_usd=1.0,
        estimated_dispatch_wall_clock_seconds=100, gpu_class="T4",
        min_smoke_gpu="T4", cost_band_epochs=100, horizon_class="asymptotic_pursuit",
        blocking_issues=(), readiness_verdict="READY", ev_per_dollar=0.5,
    )
    cmd = build_operator_authorize_command(fake)
    assert "OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1" in cmd
    assert "OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=2.000" in cmd
    assert "tools/run_modal_smoke_before_full.py" in cmd
    assert "--recipe substrate_x_modal_t4_dispatch" in cmd
    assert "--recipe /some/recipe.yaml" not in cmd
    assert "--operator-handle codex:x" in cmd
    assert "budget_floor ~$2.000" in cmd
    assert "queue_estimate=$2.000" in cmd


def test_recipe_session_budget_floor_reads_candidate_4c_recipe():
    """Candidate 4c diagnostic smoke budget must not borrow the old full envelope."""
    recipe = (
        REPO_ROOT
        / ".omx/operator_authorize_recipes/substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch.yaml"
    )
    assert _recipe_session_budget_floor_usd(recipe) == pytest.approx(1.25)


def test_operator_authorize_command_blocks_candidate_4c_until_exact_eval_handoff():
    """The old Candidate 4c paid command is blocked until the exact-eval handoff lands."""
    c = assess_candidate("z6_v2_candidate_4c_scorer_logit")
    cmd = build_operator_authorize_command(c)
    assert c.readiness_verdict == "DEFER"
    assert cmd.startswith("# Candidate NOT READY")
    assert "operator-authorize NOT recommended" in cmd


# ---------- end-to-end assessment ----------


def test_assess_candidates_returns_all_canonical():
    """Default canonical list includes every current asymptotic candidate."""
    a = assess_candidates()
    assert len(a.candidates) == len(CANONICAL_CANDIDATES)


def test_assess_candidates_provenance_discipline_score_claim_false():
    """Per provenance: assessment is planning artifact, never score claim."""
    a = assess_candidates()
    assert a.score_claim is False
    assert a.promotion_eligible is False
    assert a.evidence_grade == "predicted"
    assert a.provenance_kind == "PREDICTED_FROM_MODEL"


def test_assess_candidates_per_candidate_provenance_discipline():
    """Each candidate carries score_claim=false + promotion_eligible=false."""
    a = assess_candidates()
    for c in a.candidates:
        assert c.score_claim is False, f"{c.substrate_id} score_claim leaked"
        assert c.promotion_eligible is False, f"{c.substrate_id} promotion_eligible leaked"
        assert c.evidence_grade == "predicted"


def test_assess_candidates_atw_v2_excluded_from_top_1_due_to_probe():
    """ATW V2 is BLOCKED by Catalog #313; should not be TOP-1."""
    a = assess_candidates()
    assert a.top_1_substrate != "atw_codec_v2"


# ---------- write artifact ----------


def test_write_artifact_persists_to_canonical_path(tmp_path):
    """write_artifact writes a JSON file under .omx/state/asymptotic_pursuit/."""
    # Create a minimal fake repo root structure
    (tmp_path / ".omx" / "state").mkdir(parents=True)
    (tmp_path / ".omx" / "operator_authorize_recipes").mkdir(parents=True)
    (tmp_path / "experiments").mkdir(parents=True)
    a = assess_candidates(repo_root=tmp_path)
    path = write_artifact(a, repo_root=tmp_path)
    assert path.exists()
    assert path.suffix == ".json"
    assert "asymptotic_pursuit" in str(path)
    payload = json.loads(path.read_text())
    assert payload["score_claim"] is False
    assert payload["evidence_grade"] == "predicted"
    assert payload["provenance_kind"] == "PREDICTED_FROM_MODEL"
    assert "result_review_blockers" in payload


# ---------- dispatch sequence + cost rollup ----------


def test_build_dispatch_sequence_includes_smoke_before_full_per_catalog_167():
    """Every dispatch sequence starts with smoke per Catalog #167."""
    a = assess_candidates()
    seq = build_dispatch_sequence(a)
    assert len(seq) == len(CANONICAL_CANDIDATES)
    for entry in seq:
        stages = entry["stages"]
        assert any(s["stage"] == "smoke_100ep" for s in stages), \
            f"smoke_100ep stage missing for {entry['substrate_id']}"


def test_build_dispatch_sequence_includes_paired_cpu_axis_per_claude_md():
    """Per CLAUDE.md 'BOTH CPU AND CUDA': every sequence has paired CPU."""
    a = assess_candidates()
    seq = build_dispatch_sequence(a)
    for entry in seq:
        assert any(s["stage"] == "paired_cpu_axis_verification" for s in entry["stages"])


def test_build_dispatch_sequence_does_not_double_count_paired_cpu_axis():
    """The readiness estimate already includes paired CPU; queue stages split it once."""
    a = assess_candidates()
    seq = build_dispatch_sequence(a)
    candidates = {c.substrate_id: c for c in a.candidates}
    for entry in seq:
        c = candidates[entry["substrate_id"]]
        stages = {s["stage"]: s for s in entry["stages"]}
        assert "full_eval_contest_cuda" in stages
        assert "paired_cpu_axis_verification" in stages
        stage_sum = sum(s["estimated_cost_usd"] for s in entry["stages"])
        assert entry["total_estimated_cost_usd"] == pytest.approx(stage_sum)
        assert entry["total_estimated_cost_usd"] == pytest.approx(
            1.0 + c.estimated_dispatch_cost_usd
        )


def test_dispatch_sequence_surfaces_operator_session_budget_floor():
    """Candidate 4c remains budgeted as diagnostic smoke but not launch-ready."""
    a = assess_candidates()
    seq = build_dispatch_sequence(a, dirty_path_count=0, env={})
    entry = next(
        s
        for s in seq
        if s["substrate_id"] == "z6_v2_candidate_4c_scorer_logit"
    )
    assert entry["readiness_verdict"] == "DEFER"
    assert entry["total_estimated_cost_usd"] == pytest.approx(1.428)
    assert entry["operator_session_budget_floor_usd"] == pytest.approx(1.428)
    assert entry["operator_session_budget_floor_basis"] == {
        "queue_estimate_usd": pytest.approx(1.428),
        "recipe_declared_floor_usd": pytest.approx(1.25),
    }
    assert entry["predicted_band_kind"] == "predicted_score_band"
    assert entry["predicted_score_band"] == pytest.approx([0.11, 0.17])
    assert entry["predicted_band_axis"] == "contest-CUDA"
    assert entry["predicted_band_validation_status"] == "pending_post_training"
    assert entry["predicted_band_metadata_blockers"] == []
    assert entry["ready_for_paid_dispatch"] is False
    auth = entry["operator_session_authorization"]
    assert auth["required_for_paid_dispatch"] is False
    assert auth["session_directive_env_var"] == (
        "OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE"
    )
    assert auth["session_budget_env_var"] == "OPERATOR_AUTHORIZE_SESSION_BUDGET_USD"
    assert auth["minimum_session_budget_usd"] == pytest.approx(1.428)
    assert auth["budget_floor_basis"] == {
        "queue_estimate_usd": pytest.approx(1.428),
        "recipe_declared_floor_usd": pytest.approx(1.25),
    }
    catalog202 = auth["catalog202_dirty_tree_attestation"]
    assert catalog202["required_for_paid_dispatch"] is False
    assert catalog202["dirty_worktree_path_count"] == 0
    assert catalog202["satisfied_in_current_environment"] is True
    assert catalog202["intent_env_var"] == CATALOG202_BYPASS_INTENT_ENV_VAR
    assert catalog202["attestation_env_var"] == CATALOG202_BYPASS_ATTESTATION_ENV_VAR
    assert entry["paid_launch_missing_preconditions"] == []
    assert entry["immediately_runnable_paid_launch"] is False
    assert entry["paid_launch_command"] is None
    assert "--dry-run" in entry["dry_run_command"]
    assert "OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE" not in entry[
        "dry_run_command"
    ]


def test_dispatch_sequence_marks_dirty_tree_catalog202_attestation_precondition():
    """Dirty shared worktrees need explicit Catalog #202 paired-env attestation."""
    a = assess_candidates()
    seq = build_dispatch_sequence(a, dirty_path_count=7, env={})
    entry = next(
        s
        for s in seq
        if s["substrate_id"] == "z6_v2_candidate_4c_scorer_logit"
    )

    assert entry["ready_for_paid_dispatch"] is False
    assert entry["immediately_runnable_paid_launch"] is False
    assert entry["current_worktree_dirty_path_count"] == 7
    assert entry["paid_launch_missing_preconditions"] == []
    catalog202 = entry["operator_session_authorization"][
        "catalog202_dirty_tree_attestation"
    ]
    assert catalog202["required_for_paid_dispatch"] is True
    assert catalog202["dirty_worktree_path_count"] == 7
    assert catalog202["satisfied_in_current_environment"] is False

    payload = build_payload(a, seq)
    assert payload["ready_for_paid_dispatch_count"] == 0
    assert payload["immediately_runnable_paid_dispatch_count"] == 0
    assert payload["current_worktree_dirty_path_count"] == 7
    assert payload[
        "ready_paid_rows_requiring_catalog202_dirty_tree_attestation_count"
    ] == 0
    assert payload["top_ready_paid_launch_missing_preconditions"] == []
    assert payload["top_immediately_runnable_paid_launch_command"] is None


def test_catalog202_attestation_dirty_sentinel_rejects_stale_audit_snapshot():
    """Paired env vars are insufficient when dirty sentinel bytes changed after audit."""
    catalog202 = _catalog202_dirty_tree_attestation(
        dirty_path_count=3,
        env={
            CATALOG202_BYPASS_INTENT_ENV_VAR: "1",
            CATALOG202_BYPASS_ATTESTATION_ENV_VAR: "operator-attests",
            CATALOG202_BYPASS_AUDIT_JSON_ENV_VAR: (
                ".omx/state/catalog202_sentinel_cleanliness/stale.json"
            ),
        },
        latest_sentinel_audit={
            "path": ".omx/state/catalog202_sentinel_cleanliness/stale.json",
            "sentinel_set_sha256": "old",
            "effective_sentinel_files": ["tools/operator_authorize.py"],
            "dirty_sentinel_paths": ["tools/operator_authorize.py"],
        },
        env_sentinel_audit={
            "path": ".omx/state/catalog202_sentinel_cleanliness/stale.json",
            "sentinel_set_sha256": "old",
            "effective_sentinel_files": ["tools/operator_authorize.py"],
            "dirty_sentinel_paths": ["tools/operator_authorize.py"],
        },
        current_sentinel_snapshot={
            "sentinel_set_sha256": "new",
            "effective_sentinel_files": ["tools/operator_authorize.py"],
            "dirty_sentinel_path_count": 1,
            "dirty_sentinel_paths": ["tools/operator_authorize.py"],
            "snapshot_blockers": [],
        },
    )

    assert catalog202["dirty_sentinel_audit_required"] is True
    assert catalog202["latest_sentinel_audit_matches_current"] is False
    assert catalog202["satisfied_in_current_environment"] is False
    assert (
        f"{CATALOG202_BYPASS_AUDIT_JSON_ENV_VAR}=<fresh_current_catalog202_audit_json>"
        in catalog202["missing_env_assignments"]
    )
    assert _paid_launch_missing_preconditions(
        ready_for_paid_dispatch=True,
        catalog202=catalog202,
    ) == [
        "CATALOG_202_dirty_worktree_requires_paired_env_attestation_before_paid_dispatch",
        "CATALOG_202_dirty_sentinel_requires_current_audit_json_before_paid_dispatch",
    ]


def test_catalog202_attestation_dirty_sentinel_accepts_current_env_audit_snapshot():
    """Dirty sentinels are runnable only when the env audit matches current bytes."""
    catalog202 = _catalog202_dirty_tree_attestation(
        dirty_path_count=3,
        env={
            CATALOG202_BYPASS_INTENT_ENV_VAR: "1",
            CATALOG202_BYPASS_ATTESTATION_ENV_VAR: "operator-attests",
            CATALOG202_BYPASS_AUDIT_JSON_ENV_VAR: (
                ".omx/state/catalog202_sentinel_cleanliness/current.json"
            ),
        },
        latest_sentinel_audit={
            "path": ".omx/state/catalog202_sentinel_cleanliness/current.json",
            "sentinel_set_sha256": "current",
            "effective_sentinel_files": ["tools/operator_authorize.py"],
            "dirty_sentinel_paths": ["tools/operator_authorize.py"],
        },
        env_sentinel_audit={
            "path": ".omx/state/catalog202_sentinel_cleanliness/current.json",
            "sentinel_set_sha256": "current",
            "effective_sentinel_files": ["tools/operator_authorize.py"],
            "dirty_sentinel_paths": ["tools/operator_authorize.py"],
        },
        current_sentinel_snapshot={
            "sentinel_set_sha256": "current",
            "effective_sentinel_files": ["tools/operator_authorize.py"],
            "dirty_sentinel_path_count": 1,
            "dirty_sentinel_paths": ["tools/operator_authorize.py"],
            "snapshot_blockers": [],
        },
    )

    assert catalog202["dirty_sentinel_audit_required"] is True
    assert catalog202["latest_sentinel_audit_matches_current"] is True
    assert catalog202["env_sentinel_audit_matches_current"] is True
    assert catalog202["satisfied_in_current_environment"] is True
    assert catalog202["missing_env_assignments"] == []
    assert (
        _paid_launch_missing_preconditions(
            ready_for_paid_dispatch=True,
            catalog202=catalog202,
        )
        == []
    )


def test_catalog202_attestation_rejects_invalid_current_sentinel_snapshot():
    """A dirty-tree launch cannot be runnable if current sentinel custody failed."""
    catalog202 = _catalog202_dirty_tree_attestation(
        dirty_path_count=3,
        env={
            CATALOG202_BYPASS_INTENT_ENV_VAR: "1",
            CATALOG202_BYPASS_ATTESTATION_ENV_VAR: "operator-attests",
        },
        latest_sentinel_audit={
            "path": ".omx/state/catalog202_sentinel_cleanliness/current.json",
            "sentinel_set_sha256": "current",
            "effective_sentinel_files": ["tools/operator_authorize.py"],
            "dirty_sentinel_paths": [],
        },
        current_sentinel_snapshot={
            "sentinel_set_sha256": "current",
            "effective_sentinel_files": ["tools/operator_authorize.py"],
            "dirty_sentinel_path_count": 0,
            "dirty_sentinel_paths": [],
            "snapshot_blockers": ["catalog202_sentinel_file_missing"],
        },
    )

    assert catalog202["current_sentinel_snapshot_valid"] is False
    assert catalog202["dirty_sentinel_audit_required"] is False
    assert catalog202["satisfied_in_current_environment"] is False
    assert _paid_launch_missing_preconditions(
        ready_for_paid_dispatch=True,
        catalog202=catalog202,
    ) == [
        "CATALOG_202_dirty_worktree_requires_paired_env_attestation_before_paid_dispatch",
        "CATALOG_202_current_sentinel_snapshot_required_before_paid_dispatch",
    ]


def test_dispatch_sequence_rejects_stale_external_catalog202_attestation_env():
    """A stale env audit path cannot borrow freshness from the latest audit."""
    a = assess_candidates()
    seq = build_dispatch_sequence(
        a,
        dirty_path_count=7,
        env={
            CATALOG202_BYPASS_INTENT_ENV_VAR: "1",
            CATALOG202_BYPASS_ATTESTATION_ENV_VAR: "sentinel set verified clean",
            CATALOG202_BYPASS_AUDIT_JSON_ENV_VAR: (
                ".omx/state/catalog202_sentinel_cleanliness/"
                "substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch_20260518T065922Z.json"
            ),
        },
    )
    entry = next(
        s
        for s in seq
        if s["substrate_id"] == "z6_v2_candidate_4c_scorer_logit"
    )

    assert entry["ready_for_paid_dispatch"] is False
    assert entry["immediately_runnable_paid_launch"] is False
    assert entry["paid_launch_missing_preconditions"] == []
    catalog202 = entry["operator_session_authorization"][
        "catalog202_dirty_tree_attestation"
    ]
    assert catalog202["required_for_paid_dispatch"] is True
    assert catalog202["latest_sentinel_audit_matches_current"] is False
    assert catalog202["env_sentinel_audit_matches_current"] is False
    assert catalog202["satisfied_in_current_environment"] is False


def test_dispatch_sequence_surfaces_candidate_4c_identity_probe():
    """The dispatch queue carries the local no-op guard artifact forward."""
    a = assess_candidates()
    seq = build_dispatch_sequence(a)
    entry = next(
        s
        for s in seq
        if s["substrate_id"] == "z6_v2_candidate_4c_scorer_logit"
    )
    probe = entry["local_identity_disambiguator_probe"]
    assert probe["path"].endswith(
        ".omx/research/z6_candidate4c_identity_archive_pair_disambiguator_20260518_codex.json"
    )
    assert probe["verdict"] == (
        "contest_cuda_cpu_recovered_full_lower_below_delta_zeroepoch_control_bad"
    )
    assert probe["runtime_output_changed"] is True
    assert probe["custody"]["runtime_custody_aggregate_sha256"] == (
        "384938f3b6a14acb5944938c45c127ccc411f00983aff7589c7c9b98cfc56073"
    )
    assert probe["custody"]["total_byte_differences"] == 33048720
    assert probe["custody"]["full_output_aggregate_sha256"] == (
        "241f9cf0d6234a728a165173e0f352beb5254d358dacf0e6d7ff027b0f58c712"
    )
    assert probe["custody"]["identity_output_aggregate_sha256"] == (
        "5c0673169daabf7a90cddaa86b23b157019f96c63f68daa36eed786be368d94e"
    )
    assert probe["blockers"] == []


def test_dispatch_sequence_surfaces_z7_superseded_z6_4c_dependency():
    """Once Z6 4c exact eval lands, Z7 should show its real remaining blockers."""
    a = assess_candidates()
    seq = build_dispatch_sequence(a)
    entry = next(
        s
        for s in seq
        if s["substrate_id"] == "time_traveler_l5_z7_lstm_predictive_coding"
    )

    assert (
        "z7_dispatch_requires_z6_wave2_candidate_4c_paired_exact_eval_outcome"
        in entry["dispatch_blocker_supersessions"]
    )
    assert all(
        "z7_dispatch_requires_z6_wave2_candidate_4c_paired_exact_eval_outcome"
        not in issue
        for issue in entry["blocking_issues"]
    )
    assert "TRAINER_MISSING" not in entry["blocking_issues"]
    assert "CATALOG_240_FULL_MAIN_BLOCKED:RAISES_NotImplementedError" not in entry[
        "blocking_issues"
    ]
    assert (
        "RECIPE_DISPATCH_BLOCKER:"
        "z7_score_aware_one_pair_smoke_not_contest_authority"
        in entry["blocking_issues"]
    )
    assert (
        "RECIPE_DISPATCH_BLOCKER:"
        "z7_score_aware_packet_requires_paired_exact_eval"
        in entry["blocking_issues"]
    )
    probe = entry["local_identity_disambiguator_probe"]
    assert probe["path"].endswith(
        ".omx/research/probe_z7_temporal_coherence_vs_static_capacity_disambiguator_20260518_codex.json"
    )
    assert probe["verdict"] == "pending_paired_exact_eval_json"
    assert probe["runtime_output_changed"] is None
    assert probe["blockers"] == []
    assert probe["custody"]["decision_rule"]["same_archive_bytes_required"] is True


def test_dispatch_sequence_uses_vastai_dry_run_for_lane_17_imp():
    """Vast.ai research-only rows should not display the Modal smoke wrapper."""
    a = assess_candidates()
    seq = build_dispatch_sequence(a)
    entry = next(s for s in seq if s["substrate_id"] == "lane_17_imp")

    assert entry["ready_for_paid_dispatch"] is False
    assert entry["paid_launch_command"] is None
    assert entry["dry_run_command"].startswith(
        ".venv/bin/python scripts/launch_lane_on_vastai.py"
    )
    assert "--lane-script scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh" in (
        entry["dry_run_command"]
    )
    assert "--dry-run" in entry["dry_run_command"]
    assert "tools/run_modal_smoke_before_full.py" not in entry["dry_run_command"]


def test_write_dispatch_queue_artifact_persists_probe_and_false_authority_flags(
    tmp_path: Path,
) -> None:
    """Queue snapshots preserve READY/probe state without becoming score claims."""
    a = assess_candidates()
    seq = build_dispatch_sequence(a)
    payload = build_payload(a, seq)
    path = write_dispatch_queue_artifact(payload, repo_root=tmp_path)

    assert path.exists()
    assert path.name.startswith("dispatch_queue_")
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["score_claim"] is False
    assert persisted["promotion_eligible"] is False
    assert persisted["evidence_grade"] == "predicted"
    assert "result_review_blockers" in persisted
    assert persisted["ready_for_paid_dispatch_count"] == 0
    assert persisted["top_ready_substrate"] is None
    assert persisted["top_ready_paid_launch_command"] is None
    assert persisted["top_ready_dry_run_command"] is None
    candidate4c = next(
        row
        for row in persisted["dispatch_sequence"]
        if row["substrate_id"] == "z6_v2_candidate_4c_scorer_logit"
    )
    assert candidate4c["ready_for_paid_dispatch"] is False
    assert candidate4c["readiness_verdict"] == "DEFER"
    assert "RECIPE_dispatch_enabled=false" in candidate4c["blocking_issues"]
    assert candidate4c["local_identity_disambiguator_probe"][
        "runtime_output_changed"
    ] is True
    assert candidate4c["local_identity_disambiguator_probe"]["custody"][
        "runtime_custody_aggregate_sha256"
    ] == "384938f3b6a14acb5944938c45c127ccc411f00983aff7589c7c9b98cfc56073"
    assert candidate4c["local_identity_disambiguator_probe"]["blockers"] == []


def test_compute_cost_band_rollup_aggregates_per_tier():
    """Cost rollup separates READY / NEEDS_FIX / DEFER tiers."""
    a = assess_candidates()
    seq = build_dispatch_sequence(a)
    rollup = compute_cost_band_rollup(seq)
    assert rollup["total_count"] == len(CANONICAL_CANDIDATES)
    assert (
        rollup["ready_count"]
        + rollup["needs_fix_count"]
        + rollup["defer_count"]
        == rollup["total_count"]
    )
    assert rollup["ready_total_session_budget_floor_usd"] >= rollup[
        "ready_total_cost_usd_if_dispatched"
    ]


def test_compute_operator_attention_budget_within_t2_t3_budgets():
    """Canonical candidates remain within T2 (90/30d) + T3 (13/30d) budgets."""
    a = assess_candidates()
    seq = build_dispatch_sequence(a)
    budget = compute_operator_attention_budget(seq)
    assert budget["t2_within_budget"] is True
    assert budget["t3_within_budget"] is True


# ---------- regression: top-2 surfaces stage 2 stacking ----------


def test_assess_candidates_top_2_surfaces_for_stage_2_stacking():
    """TOP-2 substrate must be surfaced for Stage 2 stacking design per Q9."""
    a = assess_candidates()
    assert a.top_2_substrate is not None
    assert a.top_2_substrate != a.top_1_substrate


# ---------- regression: WZ-on-existing-archives DEFERRED confirmation ----------


def test_canonical_candidates_does_not_include_wyner_ziv_existing_archives():
    """Per Option B falsification: WZ-on-existing-archives is NOT a canonical
    candidate for the ASYMPTOTIC pivot — replaced by substrate-class-shift.
    """
    assert all(
        "wyner_ziv_existing" not in c and "wyner_ziv_hoist" not in c
        for c in CANONICAL_CANDIDATES
    )


def test_canonical_candidates_inclusion_set():
    """The canonical set includes the visible class-shift candidates."""
    assert "time_traveler_l5_z6" in CANONICAL_CANDIDATES
    assert "z6_v2_candidate_4c_scorer_logit" in CANONICAL_CANDIDATES
    assert "time_traveler_l5_z7_lstm_predictive_coding" in CANONICAL_CANDIDATES
    assert "atw_codec_v2" in CANONICAL_CANDIDATES
    assert "time_traveler_l5_autonomy" in CANONICAL_CANDIDATES
    assert "c6_e4_mdl_ibps" in CANONICAL_CANDIDATES
    assert "lane_17_imp" in CANONICAL_CANDIDATES
