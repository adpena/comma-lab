# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from comma_lab.scheduler.frontier_rate_attack_feedback import (
    FEEDBACK_REFRESH_SCHEMA,
    LOCAL_CPU_EUREKA_DISCOVERY_SCHEMA,
    FrontierRateAttackFeedbackError,
    build_frontier_rate_attack_feedback_refresh,
    discover_local_cpu_eureka_planning_signals,
)
from comma_lab.scheduler.frontier_rate_attack_feedback_cycle import (
    AUTOPILOT_RESULT_SCHEMA,
    FrontierRateAttackFeedbackCycleError,
    discover_dqs1_drop_many_greedy_verdict_paths,
    harvest_paths_from_autopilot_payload,
    select_pairset_acquisition_for_harvests,
    write_frontier_refresh_artifacts,
    write_pairset_component_marginal_feedback_bundle,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


AUTHORITY_KEYS = (
    "score_claim",
    "promotion_eligible",
    "rank_or_kill_eligible",
    "ready_for_exact_eval_dispatch",
)


def _false_authority() -> dict[str, object]:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotable": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "score_claim_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "field_selection_ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "exact_cuda_auth_eval": False,
        "contest_cuda_auth_eval": False,
        "score_affecting_payload_changed": False,
        "charged_bits_changed": False,
    }


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _advisory(
    *,
    archive_sha256: str,
    archive_size_bytes: int,
    raw_sha256: str,
    runtime_sha256: str,
    seg: float = 0.055,
    pose: float = 0.017,
) -> dict[str, object]:
    rate = archive_size_bytes / 10_000_000.0
    score = seg + pose + rate
    return {
        "schema_version": "contest_auth_eval_result.v1",
        **_false_authority(),
        "canonical_score": score,
        "score_recomputed_from_components": score,
        "score_seg_contribution": seg,
        "score_pose_contribution": pose,
        "score_rate_contribution": rate,
        "archive_size_bytes": archive_size_bytes,
        "provenance": {
            "archive_sha256": archive_sha256,
            "archive_size_bytes": archive_size_bytes,
            "inflate_runtime_manifest": {
                "runtime_tree_sha256": runtime_sha256,
            },
            "inflated_output_manifest": {
                "payload": {
                    "aggregate_sha256": raw_sha256,
                },
            },
        },
    }


def _action(candidate_id: str, rank: int) -> dict[str, object]:
    return {
        **_false_authority(),
        "candidate_id": candidate_id,
        "operator_action_rank": rank,
        "operator_next_action": "materialize_pairset_archive_and_run_local_controls",
    }


def _row(candidate_id: str, selected_pair_indices: list[int]) -> dict[str, object]:
    return {
        **_false_authority(),
        "candidate_id": candidate_id,
        "operator_next_action": "materialize_pairset_archive_and_run_local_controls",
        "source_metadata": {
            "selected_pair_count": len(selected_pair_indices),
            "selected_pair_indices": selected_pair_indices,
        },
    }


def _write_action_summary(repo: Path) -> Path:
    summary_dir = repo / "experiments" / "results" / "cross_family_candidate_portfolio"
    portfolio = summary_dir / "portfolio.json"
    candidate_rows = [
        ("pairset_drop_one_rank023_pair0440", [1, 2, 440]),
        ("pairset_drop_one_rank024_pair0112", [1, 2, 112]),
        ("pairset_drop_one_rank025_pair0233", [1, 2, 233]),
    ]
    _write_json(
        portfolio,
        {"operator_action_rows": [_row(candidate, pairs) for candidate, pairs in candidate_rows]},
    )
    return _write_json(
        summary_dir / "action_summary.json",
        {
            **_false_authority(),
            "schema": "cross_family_candidate_portfolio_action_summary.v1",
            "json_out": str(portfolio),
            "top_operator_actions": [
                _action(candidate, rank)
                for rank, (candidate, _pairs) in enumerate(candidate_rows, start=1)
            ],
        },
    )


def _materializer_observation(
    *,
    observation_id: str,
    target_kind: str,
    saved_bytes: int,
    receiver_contract_satisfied: bool,
    rate_positive: bool,
) -> dict[str, object]:
    return {
        "schema": "family_agnostic_materializer_empirical_observation.v1",
        **_false_authority(),
        "observation_id": observation_id,
        "candidate_id": observation_id,
        "target_kind": target_kind,
        "materializer_id": f"{target_kind}_adapter",
        "saved_bytes": saved_bytes,
        "rate_positive": rate_positive,
        "savings_realized": rate_positive,
        "receiver_contract_satisfied": receiver_contract_satisfied,
        "inflate_parity_satisfied": False,
        "recommended_planner_action": (
            "keep_rate_positive_candidate_for_inflate_parity_gate"
            if rate_positive
            else "demote_matching_archive_class_for_materializer"
        ),
        "readiness_blockers": [] if receiver_contract_satisfied else ["runtime_adapter_missing"],
    }


def _eureka_signal(
    *,
    candidate_id: str = "pairset_drop_two_r013_009_p0327_0459",
    projected_contest_score: float = 0.1920291170981836,
    conservative_projected_contest_score: float = 0.1920321170981836,
    local_score: float = 0.19203961709818362,
    archive_sha256: str = "d" * 64,
) -> dict[str, object]:
    return {
        "schema": "local_cpu_contest_drift_eureka_signal.v1",
        **_false_authority(),
        "auth_frontier_score": 0.19202828295713675,
        "authority": "false_authority_exact_eval_spend_trigger_only",
        "bias_local_minus_contest": 0.000010500000000010501,
        "calibration_anchor_count": 4,
        "calibration_confidence": "stable_core",
        "candidate_archive_sha256": archive_sha256,
        "candidate_id": candidate_id,
        "candidate_trust_region": "dqs1_fec6_like_same_archive_segnet_rounding",
        "candidate_trust_region_blockers": [],
        "candidate_trust_region_matches_calibration": True,
        "conservative_projected_contest_score": conservative_projected_contest_score,
        "dispatch_blockers": [
            "optimizer_candidate_queue_is_planning_only",
            "requires_exact_eval_readiness_gate",
            "requires_lane_dispatch_claim_before_gpu_or_remote_eval",
            "requires_non_proxy_score_evidence_before_promotion",
            "eureka_signal_is_not_score_authority",
            "exact_contest_cpu_eval_required_before_frontier_claim",
        ],
        "eureka_margin": 0.19202828295713675 - conservative_projected_contest_score,
        "eureka_trigger": False,
        "guard_band": 0.000003,
        "local_axis": "macOS-CPU advisory",
        "local_score": local_score,
        "projected_contest_score": projected_contest_score,
        "recommended_action": "observe_only",
        "source_artifact": (
            "experiments/results/pareto_gap_uleb/materialized/"
            f"{candidate_id}/local_cpu_advisory.json"
        ),
        "target_axis": "contest-CPU",
        "target_modes": ["contest_exact_eval_planning"],
        "trust_region": "dqs1_fec6_like_same_archive_segnet_rounding",
    }


def _write_materializer_feedback(root: Path) -> Path:
    sweep = _write_json(
        root / "receiver_smoke" / "sweep.json",
        {
            "schema": "family_agnostic_materializer_empirical_sweep.v1",
            **_false_authority(),
            "observations": [
                _materializer_observation(
                    observation_id="zip_header_elide_receiver_positive",
                    target_kind="packet_member_zip_header_elide_v1",
                    saved_bytes=52,
                    receiver_contract_satisfied=True,
                    rate_positive=True,
                ),
                _materializer_observation(
                    observation_id="packet_member_recompress_no_delta",
                    target_kind="packet_member_recompress_v1",
                    saved_bytes=0,
                    receiver_contract_satisfied=True,
                    rate_positive=False,
                ),
            ],
        },
    )
    _write_jsonl(
        root / "tensor_probe" / "observations.jsonl",
        [
            _materializer_observation(
                observation_id="zip_header_elide_receiver_positive",
                target_kind="packet_member_zip_header_elide_v1",
                saved_bytes=52,
                receiver_contract_satisfied=True,
                rate_positive=True,
            ),
            _materializer_observation(
                observation_id="tensor_factorize_receiver_missing",
                target_kind="tensor_factorize_v1",
                saved_bytes=7,
                receiver_contract_satisfied=False,
                rate_positive=True,
            )
        ],
    )
    return sweep


def _pair_frame_geometry_lattice(
    *,
    candidate_id: str = "pairset_geometry_lowimpact_k003_habcdef1234",
    selected_pair_indices: list[int] | None = None,
) -> dict[str, object]:
    selected = selected_pair_indices or [1, 2, 112, 233, 440]
    return {
        "schema": "pair_frame_scorer_geometry_lattice.v1",
        **_false_authority(),
        "summary": {
            "queue_executable_request_count": 1,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "queue_executable_pairset_drop_requests": [
            {
                "schema": "pair_frame_geometry_queue_executable_drop_request.v1",
                **_false_authority(),
                "candidate_id": candidate_id,
                "selector_kind": "pair_frame_geometry_low_impact_drop_many",
                "dropped_pair_indices": [3, 4, 5],
                "selected_pair_indices": selected,
                "selected_pair_count": len(selected),
                "geometry_covered_dropped_pair_count": 3,
                "geometry_coverage": 1.0,
                "queue_executable": True,
                "queue_family": "dqs1_pairset_local_first",
                "operator_next_action": "materialize_pairset_archive_and_run_local_controls",
                "allowed_use": "queue_executable_local_dqs1_pairset_drop_probe_only",
                "forbidden_use": "score_claim_or_dispatch_or_promotion_authority",
            }
        ],
    }


def _dqs1_observation_row(
    *,
    candidate_id: str = "pairset_drop_one_rank023_pair0440",
    raw_output_or_cache_sha256: str = "c" * 64,
) -> dict[str, object]:
    return {
        "schema": "mlx_dynamic_sweep_observation.v1",
        **_false_authority(),
        "candidate_id": candidate_id,
        "source_schema": "dqs1_local_first_harvest.v1",
        "sweep_config_id": "dqs1_local_first_macos_cpu_advisory",
        "optimization_pass_id": "local_cpu_advisory_harvest",
        "family": "decoder_q_pairset_drop_one",
        "observed_axis": "macos_cpu_advisory",
        "evidence_tag": "[macOS-CPU advisory only]",
        "observed_score_or_delta": 0.1919,
        "archive_sha256": "a" * 64,
        "runtime_sha256": "b" * 64,
        "raw_output_or_cache_sha256": raw_output_or_cache_sha256,
        "component_deltas": {
            "segnet_delta": -0.0001,
            "posenet_delta": 0.0,
            "rate_delta": -0.00002,
        },
        "score_delta_vs_baseline": -0.00012,
        "archive_byte_delta_vs_baseline": -4,
        "selected_pair_indices": [1, 2, 440],
    }


def _drop_many_greedy_negative_verdict() -> dict[str, object]:
    return {
        "schema": "dqs1_drop_many_build_1c_greedy_independent_heuristic_verdict.v1",
        **_false_authority(),
        "captured_at_utc": "2026-05-25T15:30:00Z",
        "lane_id": "lane_dqs1_drop_many_build_1c_fixture",
        "build_1c_final_verdict": (
            "NEGATIVE_COLLAPSE_TO_K1_EMPIRICAL_DROP_MANY_REGRESSES"
        ),
        "build_1c_final_verdict_reason": (
            "empirical K>1 sisters regress vs the K=1 drop-one anchor"
        ),
        "greedy_top_k_sweep": [
            {
                "k": 1,
                "selected_pair_indices": [371],
                "cumulative_predicted_delta_vs_base": -6.6e-7,
            },
            {
                "k": 2,
                "selected_pair_indices": [327, 371],
                "cumulative_predicted_delta_vs_base": -1.3e-6,
            },
        ],
        "canonical_equation_candidate_refinement": {
            "candidate_id": "dqs1_drop_many_greedy_independent_pair_ordering_v1",
            "refinement_field_proposed": {
                "empirical_k1_best_drop_one_pair_index": 371,
                "empirical_k1_best_drop_one_delta_vs_base": -6.6e-7,
                "greedy_verdict_class": (
                    "NEGATIVE_COLLAPSE_TO_K1_EMPIRICAL_DROP_MANY_REGRESSES"
                ),
            },
        },
        "catalog_313_probe_outcomes_row": {
            "probe_id": "dqs1_drop_many_build_1c_fixture",
            "verdict": "DEFER",
            "status": "blocking",
        },
    }


def _assert_false_authority(payload: dict[str, object]) -> None:
    for key in AUTHORITY_KEYS:
        assert payload[key] is False


def test_frontier_feedback_compiler_discovers_materializers_and_refreshes_dqs1_queue(
    tmp_path: Path,
) -> None:
    action_summary = _write_action_summary(tmp_path)
    artifact_root = tmp_path / "frontier_artifacts"
    _write_materializer_feedback(artifact_root)
    dqs1_observations = _write_jsonl(
        tmp_path / "dqs1_observations.jsonl",
        [_dqs1_observation_row()],
    )

    report = build_frontier_rate_attack_feedback_refresh(
        repo_root=tmp_path,
        frontier_artifact_roots=(artifact_root,),
        dqs1_observation_paths=(dqs1_observations,),
        action_summary_path=action_summary,
        results_root=str(tmp_path / "results"),
        queue_id="frontier_feedback_unit",
        candidate_limit=2,
        raw_retention_execute=True,
        raw_retention_cold_store_roots=(str(tmp_path / "cold_store"),),
    )

    assert report["schema"] == FEEDBACK_REFRESH_SCHEMA
    _assert_false_authority(report)
    assert report["materializer_feedback_payload_count"] == 2
    assert report["dqs1_observation_count"] == 1
    assert report["selected_candidate_ids"] == [
        "pairset_drop_one_rank024_pair0112",
        "pairset_drop_one_rank025_pair0233",
    ]
    discovery = report["discovery"]
    assert discovery["discovered_feedback_count"] == 2
    assert discovery["duplicate_observation_count"] == 1
    discovered_targets = {
        target
        for row in discovery["discovered_feedback"]
        for target in row["target_kinds"]
    }
    assert discovered_targets == {
        "packet_member_zip_header_elide_v1",
        "packet_member_recompress_v1",
        "tensor_factorize_v1",
    }
    bridge = report["materializer_feedback_bridge"]
    assert bridge["materializer_observation_count"] == 3
    assert bridge["planned_dqs1_candidate_count"] == 2
    assert bridge["observed_dqs1_candidate_count"] == 1
    assert bridge["score_claim"] is False
    assert bridge["ready_for_exact_eval_dispatch"] is False
    assert bridge["recommended_next_action"] == (
        "materializer_receiver_positive_followup_before_dqs1_switch"
    )
    queue = report["queue"]
    assert queue["queue_id"] == "frontier_feedback_unit"
    assert len(queue["experiments"]) == 2
    first_steps = {step["id"]: step for step in queue["experiments"][0]["steps"]}
    raw_retention_command = first_steps["plan_raw_artifact_retention"]["command"]
    assert "--execute" in raw_retention_command
    assert "--cold-store-root" in raw_retention_command
    assert all(
        experiment["metadata"]["materializer_feedback_bridge"] == bridge
        for experiment in queue["experiments"]
    )
    assert report["queue_summary"]["experiment_count"] == 2
    assert report["queue_summary"]["score_claim"] is False


def test_frontier_feedback_discovers_dqs1_observation_jsonls_from_artifact_roots(
    tmp_path: Path,
) -> None:
    action_summary = _write_action_summary(tmp_path)
    artifact_root = tmp_path / ".omx" / "research"
    _write_jsonl(
        artifact_root / "dqs1_local_first_harvest_observations_20260525T010203Z.jsonl",
        [_dqs1_observation_row()],
    )

    report = build_frontier_rate_attack_feedback_refresh(
        repo_root=tmp_path,
        frontier_artifact_roots=(artifact_root,),
        action_summary_path=action_summary,
        results_root=str(tmp_path / "results"),
        queue_id="frontier_feedback_dqs1_discovery_unit",
        candidate_limit=1,
    )

    discovery = report["dqs1_observation_discovery"]
    assert discovery["schema"] == "frontier_rate_attack_dqs1_observation_discovery.v1"
    assert discovery["active"] is True
    assert discovery["discovered_observation_count"] == 1
    assert report["dqs1_observation_count"] == 1
    assert report["dqs1_observation_source_paths"] == [
        ".omx/research/dqs1_local_first_harvest_observations_20260525T010203Z.jsonl"
    ]
    assert report["selected_candidate_ids"] == ["pairset_drop_one_rank024_pair0112"]
    _assert_false_authority(discovery)
    assert report["ready_for_exact_eval_dispatch"] is False


def test_frontier_feedback_compiler_turns_eureka_near_misses_into_beyond_drop_two_hints(
    tmp_path: Path,
) -> None:
    action_summary = _write_action_summary(tmp_path)
    artifact_root = tmp_path / "frontier_artifacts"
    eureka_path = _write_json(
        artifact_root
        / "local_cpu_contest_drift_eureka_pairset_drop_two_r013_009_p0327_0459_20260525T131428Z.json",
        _eureka_signal(),
    )

    discovery = discover_local_cpu_eureka_planning_signals(
        repo_root=tmp_path,
        frontier_artifact_roots=(artifact_root,),
    )

    assert discovery["schema"] == LOCAL_CPU_EUREKA_DISCOVERY_SCHEMA
    assert discovery["active"] is True
    assert discovery["signal_count"] == 1
    assert discovery["candidate_family_counts"] == {"decoder_q_pairset_drop_two": 1}
    assert discovery["planner_hint_count"] == 1
    hint = discovery["planner_hints"][0]
    assert hint["hint_id"] == "dqs1_expand_beyond_drop_two_near_boundary"
    assert "learned_multi_drop" in hint["recommended_candidate_families"]
    assert "drop_many_beam_pairwise_interaction_waterfill" in hint[
        "recommended_candidate_families"
    ]
    assert "inverse_scorer_action_surface" in hint["planner_consumers"]
    _assert_false_authority(hint)

    report = build_frontier_rate_attack_feedback_refresh(
        repo_root=tmp_path,
        frontier_artifact_roots=(artifact_root,),
        action_summary_path=action_summary,
        results_root=str(tmp_path / "results"),
        queue_id="frontier_feedback_unit",
        candidate_limit=1,
    )

    eureka = report["local_cpu_eureka_planning"]
    assert eureka["signal_rows"][0]["path"] == eureka_path.relative_to(tmp_path).as_posix()
    assert eureka["planner_hints"][0]["hint_id"] == (
        "dqs1_expand_beyond_drop_two_near_boundary"
    )
    experiment = report["queue"]["experiments"][0]
    assert experiment["metadata"]["frontier_feedback_eureka_planning"][
        "planner_hint_count"
    ] == 1
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


def test_component_marginal_bundle_auto_discovers_drop_many_greedy_verdict(
    tmp_path: Path,
) -> None:
    acquisition = _write_json(
        tmp_path / "pairset_acquisition.json",
        {
            "schema": "decoder_q_pairset_acquisition.v1",
            **_false_authority(),
            "candidates": [
                {
                    **_false_authority(),
                    "acquisition_id": "pairset_drop_one_rank023_pair0440",
                    "selector_id": "pairset_drop_one_rank023_pair0440",
                    "selector_kind": "drop_one_from_best",
                    "acquisition_rank": 23,
                    "predicted_score_mean": 0.1919,
                    "selected_pair_indices": [1, 2, 440],
                    "selected_pair_count": 3,
                    "acquisition_operation": {
                        "op": "drop_one",
                        "dropped_pair_rank": 23,
                        "dropped_pair_index": 440,
                    },
                },
                {
                    **_false_authority(),
                    "acquisition_id": "pairset_drop_many_fixture_k004",
                    "selector_id": "pairset_drop_many_fixture_k004",
                    "selector_kind": "drop_many_beam_pairwise_interaction_waterfill",
                    "acquisition_rank": 24,
                    "predicted_score_mean": 0.1918,
                    "selected_pair_indices": [1, 2],
                    "selected_pair_count": 2,
                    "acquisition_operation": {
                        "op": "drop_many",
                        "dropped_pair_indices": [112, 233, 371, 440],
                    },
                },
            ],
        },
    )
    observations = _write_jsonl(
        tmp_path / "dqs1_observations.jsonl",
        [_dqs1_observation_row()],
    )
    _write_json(
        tmp_path
        / "experiments"
        / "results"
        / "dqs1_drop_many_build_1c_greedy_heuristic_fixture"
        / "verdict.json",
        _drop_many_greedy_negative_verdict(),
    )
    out = tmp_path / "component_bundle"

    bundle = write_pairset_component_marginal_feedback_bundle(
        repo_root=tmp_path,
        pairset_acquisition_paths=(acquisition,),
        observation_paths=(observations,),
        incumbent_score=0.19202,
        incumbent_scores_by_axis={"macos_cpu_advisory": 0.19202},
        output_dir=out,
        top_k=8,
        top_actions=8,
    )

    assert bundle["drop_many_greedy_verdict_count"] == 1
    assert bundle["drop_many_greedy_verdict_discovery"]["active"] is True
    summary = json.loads((out / "action_summary.json").read_text(encoding="utf-8"))
    assert summary["drop_many_greedy_verdict_model"]["active"] is True
    portfolio = json.loads((out / "portfolio.json").read_text(encoding="utf-8"))
    independent = next(
        row
        for row in portfolio["operator_action_rows"]
        if row["candidate_id"] == "pairset_drop_many_fixture_k004"
    )
    assert independent["operator_next_action"] == (
        "hold_independent_drop_many_until_interaction_or_component_model"
    )
    assert (
        "drop_many_independent_greedy_deferred_requires_interaction_or_component_model"
        in independent["source_dispatch_blockers"]
    )
    assert independent["score_claim"] is False
    assert independent["ready_for_exact_eval_dispatch"] is False


def test_drop_many_greedy_discovery_uses_bounded_known_verdict_pattern(
    tmp_path: Path,
) -> None:
    verdict = _write_json(
        tmp_path
        / "experiments"
        / "results"
        / "dqs1_drop_many_build_1c_greedy_heuristic_fixture"
        / "verdict.json",
        _drop_many_greedy_negative_verdict(),
    )
    _write_json(
        tmp_path
        / "experiments"
        / "results"
        / "unrelated"
        / "very"
        / "deep"
        / "dqs1_drop_many_accidental_verdict.json",
        _drop_many_greedy_negative_verdict(),
    )

    discovery = discover_dqs1_drop_many_greedy_verdict_paths(repo_root=tmp_path)

    assert discovery["active"] is True
    assert discovery["discovered_verdict_paths"] == [
        verdict.relative_to(tmp_path).as_posix()
    ]
    assert discovery["refusal_count"] == 0
    assert discovery["score_claim"] is False
    assert discovery["ready_for_exact_eval_dispatch"] is False


def test_frontier_feedback_compiler_promotes_pair_frame_geometry_requests_to_queue(
    tmp_path: Path,
) -> None:
    action_summary = _write_action_summary(tmp_path)
    lattice_path = _write_json(
        tmp_path / "frontier_artifacts" / "pair_frame_scorer_geometry_lattice.json",
        _pair_frame_geometry_lattice(),
    )

    report = build_frontier_rate_attack_feedback_refresh(
        repo_root=tmp_path,
        pair_frame_geometry_paths=(lattice_path,),
        action_summary_path=action_summary,
        results_root=str(tmp_path / "results"),
        queue_id="frontier_feedback_pair_frame_unit",
        candidate_limit=2,
    )

    assert report["pair_frame_geometry_queue_request_count"] == 1
    discovery = report["pair_frame_geometry_discovery"]
    assert discovery["queue_executable_request_count"] == 1
    assert discovery["discovered_lattices"][0]["candidate_ids"] == [
        "pairset_geometry_lowimpact_k003_habcdef1234"
    ]
    assert report["selected_candidate_ids"] == [
        "pairset_geometry_lowimpact_k003_habcdef1234",
        "pairset_drop_one_rank023_pair0440",
    ]
    geometry_experiment = report["queue"]["experiments"][0]
    assert geometry_experiment["metadata"]["source_metadata"][
        "queue_source_kind"
    ] == "pair_frame_scorer_geometry_lattice"
    selected_acquisition = report["selected_pairset_acquisition"]
    assert selected_acquisition["schema"] == "dqs1_selected_pairset_acquisition.v1"
    assert selected_acquisition["candidate_count"] == 2
    assert selected_acquisition["candidates"][0]["candidate_id"] == (
        "pairset_geometry_lowimpact_k003_habcdef1234"
    )
    assert selected_acquisition["candidates"][0]["selector_kind"] == (
        "pair_frame_geometry_low_impact_drop_many"
    )
    assert geometry_experiment["metadata"]["selected_pair_indices"] == [
        1,
        2,
        112,
        233,
        440,
    ]
    plan_packet = {
        step["id"]: step["command"] for step in geometry_experiment["steps"]
    }["plan_packet"]
    assert plan_packet[-1] == "1,2,112,233,440"
    artifacts = write_frontier_refresh_artifacts(
        output_dir=tmp_path / "refresh_artifacts",
        report=report,
        repo_root=tmp_path,
    )
    assert artifacts["dqs1_selected_pairset_acquisition"].endswith(
        "dqs1_selected_pairset_acquisition.json"
    )
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


def test_frontier_feedback_pair_frame_geometry_discovery_rejects_authority_leak(
    tmp_path: Path,
) -> None:
    payload = _pair_frame_geometry_lattice()
    request = payload["queue_executable_pairset_drop_requests"][0]  # type: ignore[index]
    assert isinstance(request, dict)
    request["score_claim"] = True
    lattice_path = _write_json(
        tmp_path / "pair_frame_scorer_geometry_lattice.json",
        payload,
    )

    with pytest.raises(FrontierRateAttackFeedbackError, match="score_claim"):
        build_frontier_rate_attack_feedback_refresh(
            repo_root=tmp_path,
            pair_frame_geometry_paths=(lattice_path,),
        )


def test_frontier_feedback_eureka_discovery_rejects_authority_leak(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "frontier_artifacts"
    payload = _eureka_signal()
    payload["score_claim"] = True
    _write_json(
        artifact_root
        / "local_cpu_contest_drift_eureka_pairset_drop_two_r013_009_p0327_0459_20260525T131428Z.json",
        payload,
    )

    with pytest.raises(FrontierRateAttackFeedbackError, match="score_claim"):
        discover_local_cpu_eureka_planning_signals(
            repo_root=tmp_path,
            frontier_artifact_roots=(artifact_root,),
        )


def test_frontier_feedback_compiler_fails_closed_on_truthy_authority(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "frontier_artifacts"
    bad = _materializer_observation(
        observation_id="bad_authority",
        target_kind="packet_member_recompress_v1",
        saved_bytes=1,
        receiver_contract_satisfied=True,
        rate_positive=True,
    )
    bad["score_claim"] = True
    _write_json(
        artifact_root / "bad" / "sweep.json",
        {
            "schema": "family_agnostic_materializer_empirical_sweep.v1",
            **_false_authority(),
            "observations": [bad],
        },
    )

    with pytest.raises(FrontierRateAttackFeedbackError, match="forbidden truthy"):
        build_frontier_rate_attack_feedback_refresh(
            repo_root=tmp_path,
            frontier_artifact_roots=(artifact_root,),
        )


def test_feedback_cycle_prefers_selected_pairset_acquisition_for_geometry_harvest(
    tmp_path: Path,
) -> None:
    candidate_id = "pairset_geometry_lowimpact_k003_habcdef1234"
    harvest = _write_json(
        tmp_path / ".omx" / "research" / "dqs1_local_first_harvest.json",
        {
            "schema": "dqs1_local_first_harvest.v1",
            **_false_authority(),
            "candidate_id": candidate_id,
            "local_cpu_advisory_path": "results/geometry/local_cpu_advisory.json",
            "harvested_at_utc": "20260525T010203Z",
            "authority": "false_authority_dqs1_local_first_harvest",
        },
    )
    preferred = _write_json(
        tmp_path / "refresh" / "dqs1_selected_pairset_acquisition.json",
        {
            "schema": "dqs1_selected_pairset_acquisition.v1",
            **_false_authority(),
            "candidates": [
                {
                    "candidate_id": candidate_id,
                    "acquisition_id": candidate_id,
                    "selector_id": candidate_id,
                    "selector_kind": "pair_frame_geometry_low_impact_drop_many",
                    "selected_pair_indices": [1, 2, 112],
                    "selected_pair_count": 3,
                    "acquisition_operation": {
                        "op": "pair_frame_geometry_low_impact_drop_many"
                    },
                    **_false_authority(),
                }
            ],
        },
    )
    fallback = _write_json(
        tmp_path / "pairset_acquisition.json",
        {"schema": "decoder_q_pairset_acquisition.v1", "candidates": []},
    )

    selected = select_pairset_acquisition_for_harvests(
        harvest_paths=(harvest,),
        repo_root=tmp_path,
        preferred_pairset_acquisition_path=preferred,
        fallback_pairset_acquisition_path=fallback,
    )

    assert selected == preferred.resolve()


def test_frontier_feedback_cli_writes_valid_followup_queue(tmp_path: Path) -> None:
    action_summary = _write_action_summary(tmp_path)
    artifact_root = tmp_path / "frontier_artifacts"
    _write_materializer_feedback(artifact_root)
    dqs1_observations = _write_jsonl(
        tmp_path / "dqs1_observations.jsonl",
        [_dqs1_observation_row()],
    )
    output_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "tools/build_frontier_rate_attack_feedback_refresh.py",
            "--action-summary",
            str(action_summary),
            "--frontier-artifact-root",
            str(artifact_root),
            "--dqs1-observation-jsonl",
            str(dqs1_observations),
            "--output-dir",
            str(output_dir),
            "--results-root",
            str(tmp_path / "results"),
            "--queue-id",
            "frontier_feedback_cli_unit",
            "--candidate-limit",
            "2",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["selected_candidate_ids"] == [
        "pairset_drop_one_rank024_pair0112",
        "pairset_drop_one_rank025_pair0233",
    ]
    queue_path = output_dir / "dqs1_followup_queue.json"
    bridge_path = output_dir / "materializer_feedback_bridge.json"
    report_path = output_dir / "feedback_refresh_report.json"
    assert queue_path.exists()
    assert bridge_path.exists()
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["artifacts"]["dqs1_followup_queue"].endswith("dqs1_followup_queue.json")
    assert report["operator_commands"]["validate_followup_queue"][0] == ".venv/bin/python"

    validate = subprocess.run(
        [
            sys.executable,
            "tools/experiment_queue.py",
            "--queue",
            str(queue_path),
            "validate",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert validate.returncode == 0, validate.stderr


def test_frontier_feedback_cycle_harvests_batch_and_refreshes_queue(tmp_path: Path) -> None:
    action_summary = _write_action_summary(tmp_path)
    artifact_root = tmp_path / "frontier_artifacts"
    _write_materializer_feedback(artifact_root)
    baseline = _write_json(
        tmp_path / "baseline.json",
        _advisory(
            archive_sha256="1" * 64,
            archive_size_bytes=178_592,
            raw_sha256="2" * 64,
            runtime_sha256="3" * 64,
        ),
    )
    advisory = _write_json(
        tmp_path / "candidate" / "local_cpu_advisory.json",
        _advisory(
            archive_sha256="4" * 64,
            archive_size_bytes=178_559,
            raw_sha256="5" * 64,
            runtime_sha256="6" * 64,
            seg=0.054,
        ),
    )
    harvest = _write_json(
        tmp_path / ".omx" / "research" / "dqs1_local_first_harvest.json",
        {
            "schema": "dqs1_local_first_harvest.v1",
            **_false_authority(),
            "candidate_id": "pairset_drop_one_rank023_pair0440",
            "candidate_archive_sha256": "4" * 64,
            "local_cpu_advisory_path": str(advisory),
            "local_score": 0.191,
            "projected_contest_score": 0.190,
            "conservative_projected_contest_score": 0.192,
            "recommended_action": "observe_only",
            "eureka_trigger": False,
            "eureka_margin": -1e-6,
            "authority": "false_authority_dqs1_local_first_harvest",
            "harvested_at_utc": "20260525T010203Z",
            "dispatch_blockers": ["exact_cpu_cuda_auth_eval_required"],
        },
    )
    acquisition = _write_json(
        tmp_path / "pairset_acquisition.json",
        {
            "schema": "decoder_q_pairset_acquisition.v1",
            **_false_authority(),
            "candidates": [
                {
                    **_false_authority(),
                    "acquisition_id": "pairset_drop_one_rank023_pair0440",
                    "selector_id": "pairset_drop_one_rank023_pair0440",
                    "selector_kind": "drop_one_from_best",
                    "acquisition_rank": 23,
                    "predicted_score_mean": 0.1919,
                    "selected_pair_indices": [1, 2, 440],
                    "selected_pair_count": 3,
                    "acquisition_operation": {
                        "op": "drop_one",
                        "dropped_pair_rank": 23,
                        "dropped_pair_index": 440,
                    },
                },
                {
                    **_false_authority(),
                    "acquisition_id": "pairset_drop_one_rank024_pair0112",
                    "selector_id": "pairset_drop_one_rank024_pair0112",
                    "selector_kind": "drop_one_from_best",
                    "acquisition_rank": 24,
                    "predicted_score_mean": 0.19191,
                    "selected_pair_indices": [1, 2, 112],
                    "selected_pair_count": 3,
                    "acquisition_operation": {
                        "op": "drop_one",
                        "dropped_pair_rank": 24,
                        "dropped_pair_index": 112,
                    },
                },
                {
                    **_false_authority(),
                    "acquisition_id": "pairset_drop_one_rank025_pair0233",
                    "selector_id": "pairset_drop_one_rank025_pair0233",
                    "selector_kind": "drop_one_from_best",
                    "acquisition_rank": 25,
                    "predicted_score_mean": 0.19192,
                    "selected_pair_indices": [1, 2, 233],
                    "selected_pair_count": 3,
                    "acquisition_operation": {
                        "op": "drop_one",
                        "dropped_pair_rank": 25,
                        "dropped_pair_index": 233,
                    },
                },
            ],
        },
    )
    autopilot_result = _write_json(
        tmp_path / "autopilot_result.json",
        {
            "schema": AUTOPILOT_RESULT_SCHEMA,
            **_false_authority(),
            "queue_path": str(tmp_path / "initial_queue.json"),
            "execute": True,
            "stop_reason": "batch_harvested_waiting_for_portfolio_rebuild",
            "total_steps_started": 6,
            "candidates_harvested": 1,
            "rounds": [
                {
                    "harvests": [
                        {
                            **_false_authority(),
                            "candidate_id": "pairset_drop_one_rank023_pair0440",
                            "harvest_path": str(harvest),
                        }
                    ]
                }
            ],
        },
    )
    output_dir = tmp_path / "cycle_out"

    result = subprocess.run(
        [
            sys.executable,
            "tools/run_frontier_rate_attack_feedback_cycle.py",
            "--action-summary",
            str(action_summary),
            "--frontier-artifact-root",
            str(artifact_root),
            "--autopilot-result-json",
            str(autopilot_result),
            "--pairset-acquisition",
            str(acquisition),
            "--baseline-advisory",
            str(baseline),
            "--baseline-archive-size-bytes",
            "178560",
            "--output-dir",
            str(output_dir),
            "--results-root",
            str(tmp_path / "results"),
            "--queue-id",
            "frontier_feedback_cycle_unit",
            "--post-harvest-queue-id",
            "frontier_feedback_cycle_unit_post",
            "--candidate-limit",
            "2",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["score_claim"] is False
    assert payload["harvest_path_count"] == 1
    assert payload["initial_selected_candidate_ids"] == [
        "pairset_drop_one_rank023_pair0440",
        "pairset_drop_one_rank024_pair0112",
    ]
    assert payload["post_harvest_selected_candidate_ids"] == [
        "pairset_drop_one_rank024_pair0112",
        "pairset_drop_one_rank025_pair0233",
    ]
    observation_jsonl = Path(payload["observation_jsonl"])
    if not observation_jsonl.is_absolute():
        observation_jsonl = REPO_ROOT / observation_jsonl
    assert observation_jsonl.exists()
    cycle_report = json.loads(
        (output_dir / "frontier_rate_attack_feedback_cycle.json").read_text(
            encoding="utf-8"
        )
    )
    assert cycle_report["schema"] == "frontier_rate_attack_feedback_cycle.v1"
    component = cycle_report["post_harvest_refresh"]["pairset_component_marginal"]
    assert component["schema"] == "frontier_rate_attack_pairset_component_marginal_bundle.v1"
    assert component["active"] is True
    assert component["training_row_count"] == 1
    assert cycle_report["post_harvest_refresh"]["artifacts"][
        "feedback_refresh_report"
    ].endswith("feedback_refresh_report.json")
    assert cycle_report["post_harvest_refresh"]["queue_summary"]["selected_candidate_ids"] == [
        "pairset_drop_one_rank024_pair0112",
        "pairset_drop_one_rank025_pair0233",
    ]
    assert "post_harvest_component_marginal_refresh/action_summary.json" in (
        cycle_report["post_harvest_refresh"]["artifacts"]["feedback_refresh_report"]
        or ""
    ) or cycle_report["post_harvest_refresh"]["pairset_component_marginal"][
        "action_summary_json"
    ].endswith("post_harvest_component_marginal_refresh/action_summary.json")
    assert cycle_report["post_harvest_refresh"]["queue_validate"]["valid"] is True
    assert "dynamic_observation_jsonl_to_refreshed_dqs1_queue" in cycle_report[
        "integration_edges"
    ]
    assert "dynamic_observation_jsonl_to_pairset_component_marginal_model" in cycle_report[
        "integration_edges"
    ]


def test_frontier_feedback_cycle_refuses_truthy_autopilot_authority(
    tmp_path: Path,
) -> None:
    payload = {
        "schema": AUTOPILOT_RESULT_SCHEMA,
        **_false_authority(),
        "score_claim": True,
        "rounds": [],
    }

    with pytest.raises(FrontierRateAttackFeedbackCycleError, match="forbidden truthy"):
        harvest_paths_from_autopilot_payload(payload, repo_root=tmp_path)
