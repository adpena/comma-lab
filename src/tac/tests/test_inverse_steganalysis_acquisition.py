# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from tac.optimization.inverse_steganalysis_acquisition import (
    ACTION_FUNCTIONAL_SCHEMA,
    ATOM_SCHEMA,
    CONTEST_RATE_SCORE_PER_BYTE,
    OBSERVATION_SCHEMA,
    SCHEMA,
    InverseSteganalysisAcquisitionError,
    action_surface_terms,
    build_discrete_scorer_action_functional,
    build_inverse_steganalysis_acquisition_plan,
    compute_acquisition_priority,
    normalize_inverse_steganalysis_atom,
    normalize_inverse_steganalysis_observation,
    observations_from_queue_performance_summary,
    paired_exact_auth_calibration_observations_from_review_packets,
)
from tac.optimization.proxy_candidate_contract import PROXY_FALSE_AUTHORITY_FIELDS


def _atom(candidate_id: str = "candidate_a", **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "atom_id": f"atom_{candidate_id}",
        "candidate_id": candidate_id,
        "scale": "region_frequency",
        "scope_axis": "regions",
        "parent_unit_id": "pair_0007",
        "frame_range": [14, 16],
        "pair_indices": [7],
        "region_bbox": [16, 24, 96, 128],
        "frequency_band": "mid_high_dct",
        "byte_range": [1024, 1536],
        "component": "segnet",
        "coherence_group": "road_edges",
        "sparsity_prior": 0.8,
        "predicted_segnet_gain": 0.0003,
        "predicted_posenet_gain": 0.0001,
        "predicted_rate_gain": 0.00005,
        "predicted_rate_cost": 0.00001,
        "predicted_score_gain": 0.00044,
        "first_order_marginal_effect": 0.0004,
        "second_order_interaction_effect": 0.00004,
        "discontinuity_risk": 0.1,
        "discontinuity_threshold": 0.5,
        "uncertainty": 0.00004,
        "elapsed_seconds": 4.0,
        "artifact_bytes": 1_000_000,
        "resource_kind": "local_mlx",
    }
    row.update(overrides)
    return row


def _observation(candidate_id: str = "candidate_a", **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "observation_id": f"obs_{candidate_id}",
        "candidate_id": candidate_id,
        "axis": "[macOS-MLX research-signal]",
        "runtime_identity": {
            "runtime_tree_sha256": "a" * 64,
            "scorer_version": "mlx_scorer_response.v1",
        },
        "cache_identity": {
            "cache_sha256": "b" * 64,
            "array_sha256": {"pair_indices": "c" * 64},
        },
        "observed_score_gain": 0.0005,
        "calibration_error": 0.00002,
        "elapsed_seconds": 8.0,
        "artifact_bytes": 2_000_000,
        "resource_kind": "local_mlx",
    }
    row.update(overrides)
    return row


def _review_packet(
    axis: str,
    *,
    score: float,
    baseline_score: float,
    archive_sha256: str = "1" * 64,
    archive_bytes: int = 181_232,
    runtime_content_tree_sha256: str = "2" * 64,
    inflated_output_aggregate_sha256: str = "3" * 64,
) -> dict[str, object]:
    exact_cuda = axis == "contest_cuda"
    exact_cpu = axis == "contest_cpu"
    status = (
        "measured_config_retired"
        if exact_cuda and score > baseline_score
        else "exact_cuda_result_reviewed"
        if exact_cuda
        else "contest_cpu_result_reviewed"
    )
    failure_class = (
        "legitimate_score_regression_or_component_collapse"
        if status == "measured_config_retired"
        else "contest_cpu_public_leaderboard_anchor_cuda_pending"
        if exact_cpu
        else "not_negative_against_supplied_baseline"
    )
    return {
        "schema": "tac_result_review_packet_v1",
        "tool": "tools/build_result_review_packet.py",
        "technique": "ias1_runtime_parity_top4",
        "lane_id": f"lane_ias1_{axis}",
        "job_id": f"job_ias1_{axis}",
        "source_json_path": f"experiments/results/{axis}/contest_auth_eval.json",
        "source_json_sha256": "4" * 64,
        "score_claim": False,
        "score_axis": axis,
        "score_claim_valid": exact_cuda,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "family_falsified": False,
        "method_family_retired": False,
        "measured_config_status": status,
        "failure_class": failure_class,
        "baseline_score": baseline_score,
        "canonical_score": score,
        "exact_cuda_evidence": exact_cuda,
        "exact_cpu_evidence": exact_cpu,
        "custody": {
            "archive_bytes": archive_bytes,
            "archive_sha256": archive_sha256,
            "device": "cuda" if exact_cuda else "cpu",
            "gpu_model": "Tesla T4" if exact_cuda else "",
            "n_samples": 600,
            "inflate_script": "/tmp/submission/inflate.sh",
            "command": ["experiments/contest_auth_eval.py", "--device", axis],
        },
        "dispatch_claim_state": {
            "terminal_status_recorded": True,
            "latest_status": f"completed_{axis}_modal_auth_eval_recovered",
        },
        "runtime_custody": {
            "runtime_manifest_present": True,
            "runtime_tree_sha256": ("5" if exact_cuda else "6") * 64,
            "runtime_content_tree_sha256": runtime_content_tree_sha256,
            "runtime_file_count": 12,
            "runtime_files_listed": True,
            "payload_closure_fields_present": True,
            "inflate_script_sha256": "7" * 64,
            "inflated_output_manifest_sha256": ("8" if exact_cuda else "9") * 64,
            "inflated_output_aggregate_sha256": inflated_output_aggregate_sha256,
        },
        "score_recomputation": {
            "available": True,
            "matches_reported": True,
            "avg_segnet_dist": 0.00055979 if exact_cpu else 0.00066252,
            "avg_posenet_dist": 0.00002943 if exact_cpu else 0.00016845,
            "rate_term": 0.12067494979223735,
            "recomputed_score": score,
            "reported_score": score,
            "abs_delta_vs_reported": 0.0,
        },
        "engineering_forensic_audit": {
            "schema": "engineering_forensic_audit_v1",
            "custody_reviewed": True,
            "axis_reviewed": True,
            "runtime_config_reviewed": True,
            "archive_runtime_closure_reviewed": True,
            "score_formula_reviewed": True,
            "dispatch_claim_reviewed": True,
            "engineering_or_config_bug_found": False,
            "audit_blockers": [],
            "classification_after_audit": (
                "measured_config_retired_only"
                if status == "measured_config_retired"
                else "contest_cpu_axis_reviewed_cuda_pending"
                if exact_cpu
                else "exact_cuda_result_reviewed_no_negative_status_change"
            ),
            "dead_or_family_falsification_allowed": False,
            "measured_config_retirement_allowed": status == "measured_config_retired",
        },
        "reactivation_criteria": [
            "provide a byte-closed implementation change before redispatch"
        ],
    }


def test_atom_normalization_is_false_authority_only() -> None:
    atom = normalize_inverse_steganalysis_atom(_atom())

    assert atom["schema"] == ATOM_SCHEMA
    assert atom["candidate_generation_only"] is True
    assert atom["planning_only"] is True
    assert atom["scope_axis"] == "regions"
    assert atom["pair_indices"] == [7]
    assert atom["region_bbox"] == [16.0, 24.0, 96.0, 128.0]
    assert atom["first_order_marginal_effect"] == pytest.approx(0.0004)
    assert atom["interaction_kind"] == "synergy"
    assert atom["discontinuity_guard"]["blocked"] is False
    for key, value in PROXY_FALSE_AUTHORITY_FIELDS.items():
        assert atom[key] is value

    with pytest.raises(InverseSteganalysisAcquisitionError, match="score_claim=truthy"):
        normalize_inverse_steganalysis_atom(_atom(score_claim=True))


def test_action_surface_terms_model_scope_interactions_and_fragility() -> None:
    atom = normalize_inverse_steganalysis_atom(
        _atom(
            scale="byte_range",
            scope_axis="bytes",
            first_order_marginal_effect=0.0006,
            second_order_interaction_effect=-0.0002,
            discontinuity_risk=0.9,
            discontinuity_threshold=0.5,
        )
    )

    terms = action_surface_terms(atom)

    assert terms["scope_axis"] == "bytes"
    assert terms["first_order_marginal_effect"] == pytest.approx(0.0006)
    assert terms["second_order_interaction_effect"] == pytest.approx(-0.0002)
    assert terms["interaction_kind"] == "antagonism"
    assert terms["synergy_effect"] == pytest.approx(0.0)
    assert terms["antagonism_effect"] == pytest.approx(0.0002)
    assert terms["discontinuity_guard"]["blocked"] is True


def test_observation_requires_axis_candidate_runtime_and_cache_identity() -> None:
    assert normalize_inverse_steganalysis_observation(_observation())["schema"] == (
        OBSERVATION_SCHEMA
    )

    for missing_key, message in [
        ("candidate_id", "candidate_id"),
        ("axis", "axis"),
        ("runtime_identity", "runtime_identity"),
        ("cache_identity", "cache_identity"),
    ]:
        row = _observation()
        row.pop(missing_key)
        with pytest.raises(InverseSteganalysisAcquisitionError, match=message):
            normalize_inverse_steganalysis_observation(row)

    with pytest.raises(InverseSteganalysisAcquisitionError, match="runtime_identity"):
        normalize_inverse_steganalysis_observation(
            _observation(runtime_identity={"note": "not identity"})
        )
    with pytest.raises(InverseSteganalysisAcquisitionError, match="cache_identity"):
        normalize_inverse_steganalysis_observation(
            _observation(cache_identity={"note": "not identity"})
        )

    with pytest.raises(InverseSteganalysisAcquisitionError, match="contest auth evidence"):
        normalize_inverse_steganalysis_observation(_observation(axis="[contest-CUDA]"))
    with pytest.raises(InverseSteganalysisAcquisitionError, match="contest auth evidence"):
        normalize_inverse_steganalysis_observation(
            _observation(resource_kind="contest_exact_eval")
        )


def test_calibrated_observations_rank_acquisition_candidates() -> None:
    atoms = [
        _atom(
            "candidate_fast_weak",
            predicted_score_gain=0.0002,
            elapsed_seconds=2.0,
        ),
        _atom(
            "candidate_slow_strong",
            predicted_score_gain=0.0002,
            elapsed_seconds=2.0,
        ),
    ]
    observations = [
        _observation(
            "candidate_fast_weak",
            observed_score_gain=0.00024,
            calibration_error=0.00002,
            elapsed_seconds=2.0,
            artifact_bytes=1_000_000,
            resource_kind="local_cpu",
        ),
        _observation(
            "candidate_slow_strong",
            observed_score_gain=0.0009,
            calibration_error=0.00003,
            elapsed_seconds=8.0,
            artifact_bytes=2_000_000,
            resource_kind="local_mlx",
        ),
    ]

    plan = build_inverse_steganalysis_acquisition_plan(
        atoms,
        observations=observations,
    )

    assert plan["schema"] == SCHEMA
    assert [row["candidate_id"] for row in plan["ranked_atoms"]] == [
        "candidate_slow_strong",
        "candidate_fast_weak",
    ]
    top = plan["ranked_atoms"][0]
    assert top["best_observation_id"] == "obs_candidate_slow_strong"
    assert top["priority"]["expected_score_gain"] == pytest.approx(0.000836)
    assert top["priority"]["acquisition_priority"] > 0
    for key, value in PROXY_FALSE_AUTHORITY_FIELDS.items():
        assert plan[key] is value
        assert top[key] is value


def test_local_proxy_observations_never_become_promotion_or_rank_authority() -> None:
    observation = normalize_inverse_steganalysis_observation(
        _observation(axis="[macOS-CPU advisory]")
    )
    priority = compute_acquisition_priority(_atom(), observation)
    plan = build_inverse_steganalysis_acquisition_plan(
        [_atom()],
        observations=[observation],
    )

    assert observation["score_claim"] is False
    assert observation["promotion_eligible"] is False
    assert observation["rank_or_kill_eligible"] is False
    assert observation["ready_for_exact_eval_dispatch"] is False
    assert observation["promotable"] is False
    assert priority["resource_kind"] == "local_mlx"
    assert plan["ranked_atoms"][0]["acquisition_rank"] == 1
    assert plan["ranked_atoms"][0]["rank_or_kill_eligible"] is False
    assert plan["ranked_atoms"][0]["promotion_eligible"] is False

    with pytest.raises(
        InverseSteganalysisAcquisitionError,
        match="rank_or_kill_eligible=truthy",
    ):
        normalize_inverse_steganalysis_observation(
            _observation(rank_or_kill_eligible=True)
        )


def test_queue_performance_observations_calibrate_acquisition_denominator() -> None:
    runtime_identity = {
        "runtime_tree_sha256": "d" * 64,
        "scorer_version": "local_scheduler.v1",
    }
    cache_identity = {
        "cache_sha256": "e" * 64,
    }
    performance = {
        "schema": "experiment_queue_performance_summary.v1",
        "queue_id": "byte_shave_queue",
        "event_count": 2,
        "by_resource_kind": {},
        "by_step": {
            "candidate_a.materialize": {
                "run_count": 2,
                "success_count": 2,
                "failure_count": 0,
                "resource_kind_counts": {"local_mlx": 2},
                "dominant_resource_kind": "local_mlx",
                "elapsed_seconds_mean": 3.5,
                "artifact_record_count": 4,
                "artifact_record_bytes_mean": 2_250_000.1,
                "artifact_record_raw_bytes_mean": 9_000_000.0,
            }
        },
    }

    observations = observations_from_queue_performance_summary(
        performance,
        runtime_identity=runtime_identity,
        cache_identity=cache_identity,
        source_path="configs/experiment_queues/byte_shave.yaml",
    )
    plan = build_inverse_steganalysis_acquisition_plan(
        [
            _atom(
                "candidate_a",
                predicted_score_gain=0.001,
                elapsed_seconds=99.0,
                artifact_bytes=99_000_000,
                resource_kind="local_cpu",
            )
        ],
        observations=observations,
    )
    top = plan["ranked_atoms"][0]

    assert observations[0]["schema"] == OBSERVATION_SCHEMA
    assert observations[0]["observation_kind"] == "queue_performance_step"
    assert observations[0]["observed_score_gain"] is None
    assert observations[0]["queue_id"] == "byte_shave_queue"
    assert observations[0]["step_id"] == "materialize"
    assert observations[0]["run_count"] == 2
    assert observations[0]["artifact_bytes"] == 2_250_001
    assert top["best_observation_id"] == (
        "queue_perf_byte_shave_queue_candidate_a_materialize"
    )
    assert top["priority"]["elapsed_seconds"] == pytest.approx(3.5)
    assert top["priority"]["artifact_bytes"] == 2_250_001
    assert top["priority"]["resource_kind"] == "local_mlx"
    assert top["score_claim"] is False

    with pytest.raises(InverseSteganalysisAcquisitionError, match="runtime_identity"):
        observations_from_queue_performance_summary(
            performance,
            runtime_identity={"note": "missing identity key"},
            cache_identity=cache_identity,
        )


def test_queue_performance_candidate_map_can_expand_bundle_steps() -> None:
    runtime_identity = {
        "runtime_tree_sha256": "d" * 64,
        "scorer_version": "local_scheduler.v1",
    }
    cache_identity = {
        "cache_sha256": "e" * 64,
    }
    performance = {
        "schema": "experiment_queue_performance_summary.v1",
        "queue_id": "byte_shave_queue",
        "event_count": 1,
        "by_resource_kind": {},
        "by_step": {
            "bundle.materialize": {
                "run_count": 1,
                "success_count": 1,
                "failure_count": 0,
                "resource_kind_counts": {"local_mlx": 1},
                "elapsed_seconds_mean": 4.25,
                "artifact_record_bytes_mean": 8192.0,
            }
        },
    }

    observations = observations_from_queue_performance_summary(
        performance,
        runtime_identity=runtime_identity,
        cache_identity=cache_identity,
        candidate_id_by_experiment={"bundle": ["candidate_a", "candidate_b"]},
    )
    action = build_discrete_scorer_action_functional(
        [_atom("candidate_a"), _atom("candidate_b")],
        observations=observations,
    )
    cells_by_candidate = {cell["candidate_id"]: cell for cell in action["cells"]}

    assert [observation["candidate_id"] for observation in observations] == [
        "candidate_a",
        "candidate_b",
    ]
    assert observations[0]["observation_id"] == (
        "queue_perf_byte_shave_queue_bundle_materialize_candidate_a"
    )
    assert observations[1]["observation_id"] == (
        "queue_perf_byte_shave_queue_bundle_materialize_candidate_b"
    )
    assert cells_by_candidate["candidate_a"]["best_observation_id"] == (
        observations[0]["observation_id"]
    )
    assert cells_by_candidate["candidate_b"]["best_observation_id"] == (
        observations[1]["observation_id"]
    )
    assert cells_by_candidate["candidate_a"]["priority"]["elapsed_seconds"] == pytest.approx(
        4.25
    )
    assert cells_by_candidate["candidate_b"]["priority"]["artifact_bytes"] == 8192


def test_queue_performance_observations_do_not_override_scorer_observations() -> None:
    runtime_identity = {
        "runtime_tree_sha256": "d" * 64,
        "scorer_version": "local_scheduler.v1",
    }
    cache_identity = {
        "cache_sha256": "e" * 64,
    }
    queue_observations = observations_from_queue_performance_summary(
        {
            "schema": "experiment_queue_performance_summary.v1",
            "queue_id": "byte_shave_queue",
            "event_count": 1,
            "by_resource_kind": {},
            "by_step": {
                "candidate_a.materialize": {
                    "run_count": 1,
                    "success_count": 1,
                    "failure_count": 0,
                    "resource_kind_counts": {"local_mlx": 1},
                    "elapsed_seconds_mean": 1.0,
                    "artifact_record_bytes_mean": 1.0,
                }
            },
        },
        runtime_identity=runtime_identity,
        cache_identity=cache_identity,
    )
    scorer_observation = _observation(
        "candidate_a",
        observation_id="obs_scorer_candidate_a",
        observed_score_gain=0.0002,
        elapsed_seconds=30.0,
        artifact_bytes=5_000_000,
        resource_kind="local_cpu",
    )

    plan = build_inverse_steganalysis_acquisition_plan(
        [_atom("candidate_a", predicted_score_gain=0.001)],
        observations=[*queue_observations, scorer_observation],
    )

    assert plan["ranked_atoms"][0]["best_observation_id"] == "obs_scorer_candidate_a"
    assert plan["ranked_atoms"][0]["priority"]["elapsed_seconds"] == pytest.approx(30.0)
    assert plan["ranked_atoms"][0]["priority"]["resource_kind"] == "local_cpu"


def test_paired_exact_auth_calibration_demotes_regressed_measured_config() -> None:
    observations = paired_exact_auth_calibration_observations_from_review_packets(
        [
            _review_packet(
                "contest_cpu",
                score=0.19380912393883232,
                baseline_score=0.192028283,
                inflated_output_aggregate_sha256="a" * 64,
            ),
            _review_packet(
                "contest_cuda",
                score=0.2279696105246996,
                baseline_score=0.205330029,
                inflated_output_aggregate_sha256="b" * 64,
            ),
        ],
        candidate_id="ias1_runtime_parity_top4",
        packet_paths=[
            ".omx/research/ias1_cpu_result_review.json",
            ".omx/research/ias1_cuda_result_review.json",
        ],
    )
    scorer_observation = _observation(
        "ias1_runtime_parity_top4",
        observed_score_gain=0.01,
        calibration_error=0.0,
    )

    action = build_discrete_scorer_action_functional(
        [
            _atom(
                "ias1_runtime_parity_top4",
                predicted_score_gain=0.005,
                first_order_marginal_effect=0.005,
                uncertainty=0.0,
            )
        ],
        observations=[scorer_observation, *observations],
        total_byte_budget=1024,
    )
    cell = action["cells"][0]
    calibration = observations[0]["exact_auth_calibration"]

    assert observations[0]["observation_kind"] == "paired_exact_auth_calibration"
    assert observations[0]["observed_score_gain"] == 0.0
    assert observations[0]["calibration_error"] == pytest.approx(
        (0.19380912393883232 - 0.192028283)
        + (0.2279696105246996 - 0.205330029)
    )
    assert calibration["pair_status"] == (
        "paired_exact_auth_regressed_vs_axis_baselines"
    )
    assert calibration["axis_rows"]["contest_cpu"]["score_delta_status"] == (
        "regresses_vs_axis_baseline"
    )
    assert calibration["axis_rows"]["contest_cuda"]["score_delta_status"] == (
        "regresses_vs_axis_baseline"
    )
    assert calibration["score_claim"] is False
    assert calibration["rank_or_kill_eligible"] is False
    assert cell["best_observation_id"] == observations[0]["observation_id"]
    assert cell["best_observation_kind"] == "paired_exact_auth_calibration"
    assert cell["exact_auth_calibration"]["pair_status"] == (
        "paired_exact_auth_regressed_vs_axis_baselines"
    )
    assert cell["priority"]["expected_score_gain"] == 0.0
    assert action["observation_feedback"]["exact_auth_calibration_count"] == 1
    assert action["water_bucket"]["selected_count"] == 0
    assert action["score_claim"] is False
    assert action["rank_or_kill_eligible"] is False


def test_paired_exact_auth_calibration_requires_shared_archive_custody() -> None:
    with pytest.raises(
        InverseSteganalysisAcquisitionError,
        match="share archive_sha256",
    ):
        paired_exact_auth_calibration_observations_from_review_packets(
            [
                _review_packet(
                    "contest_cpu",
                    score=0.193,
                    baseline_score=0.192,
                    archive_sha256="a" * 64,
                ),
                _review_packet(
                    "contest_cuda",
                    score=0.228,
                    baseline_score=0.205,
                    archive_sha256="b" * 64,
                ),
            ],
            candidate_id="ias1_runtime_parity_top4",
        )


def test_discrete_action_functional_water_fills_positive_euler_cells() -> None:
    atoms = [
        _atom(
            "candidate_high",
            atom_id="atom_high",
            byte_range=[0, 100],
            predicted_score_gain=0.001,
            first_order_marginal_effect=0.0009,
            second_order_interaction_effect=0.0002,
            discontinuity_risk=0.0,
        ),
        _atom(
            "candidate_low",
            atom_id="atom_low",
            byte_range=[100, 400],
            predicted_score_gain=0.00003,
            first_order_marginal_effect=0.00002,
            second_order_interaction_effect=0.0,
            discontinuity_risk=0.0,
        ),
        _atom(
            "candidate_blocked",
            atom_id="atom_blocked",
            byte_range=[400, 500],
            predicted_score_gain=0.001,
            first_order_marginal_effect=0.001,
            second_order_interaction_effect=0.0,
            discontinuity_risk=0.9,
            discontinuity_threshold=0.5,
        ),
    ]

    functional = build_discrete_scorer_action_functional(
        atoms,
        total_byte_budget=128,
        lambda_rate=CONTEST_RATE_SCORE_PER_BYTE,
    )

    assert functional["schema"] == ACTION_FUNCTIONAL_SCHEMA
    assert functional["math_model"]["representation"] == (
        "discrete_riemann_sum_with_second_order_interactions"
    )
    assert functional["integral_totals"]["cell_count"] == 3
    assert functional["integral_totals"]["blocked_cell_count"] == 1
    assert functional["integral_totals"]["second_order_interaction_effect_sum"] == pytest.approx(0.0002)
    assert functional["water_bucket"]["selected_count"] == 1
    assert functional["water_bucket"]["selected_cells"][0]["atom_id"] == "atom_high"
    assert functional["water_bucket"]["selected_water_fill_cost_bytes"] == 100
    assert functional["cells"][0]["euler_lagrange_residual"] > 0
    assert all(
        row["atom_id"] != "atom_blocked"
        for row in functional["water_bucket"]["selected_cells"]
    )
    for key, value in PROXY_FALSE_AUTHORITY_FIELDS.items():
        assert functional[key] is value
