# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.optimization.mlx_dynamic_learned_sweep import (
    MLXDynamicLearnedSweepError,
    build_mlx_dynamic_learned_sweep_plan,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _false_authority() -> dict[str, bool]:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }


def _selector_pareto() -> dict:
    return {
        "schema": "decoder_q_selective_selector_pareto.v1",
        **_false_authority(),
        "candidates": [
            {
                "schema": "decoder_q_selective_selector_candidate.v1",
                **_false_authority(),
                "selector_id": "prefix_k032",
                "selector_kind": "top_rank_prefix",
                "selected_pair_count": 32,
                "selected_pair_indices": [1, 2, 3],
                "payload_bytes": 43,
                "pair_encoding": "sorted_gap_uleb",
                "non_authoritative_mlx_gain_sum": 0.03,
                "orthogonality_score": 0.25,
                "master_gradient_priority": 0.5,
                "component_axis_context": {"seg": 0.7, "pose": 0.2, "rate": 0.1},
                "canonical_equation_provenance": {
                    "canonical_equation_id": "procedural_codebook_from_seed_compression_savings_v1",
                    "context": "intermediate_transform_quantizer",
                },
                "master_gradient_provenance": {
                    "archive_sha256": "a" * 64,
                    "gradient_tensor_kind": "per_pair_per_byte_v1",
                },
                "exact_cpu_calibrated_estimate": {
                    "schema": "decoder_q_selective_selector_exact_cpu_calibrated_estimate.v1",
                    **_false_authority(),
                    "predicted_score": 0.19203,
                    "predicted_delta_vs_base": -0.00002,
                },
            },
            {
                "schema": "decoder_q_selective_selector_candidate.v1",
                **_false_authority(),
                "selector_id": "prefix_k016",
                "selector_kind": "top_rank_prefix",
                "selected_pair_count": 16,
                "selected_pair_indices": [1, 2],
                "payload_bytes": 27,
                "pair_encoding": "sorted_gap_uleb",
                "non_authoritative_mlx_gain_sum": 0.02,
                "exact_cpu_calibrated_estimate": {
                    "schema": "decoder_q_selective_selector_exact_cpu_calibrated_estimate.v1",
                    **_false_authority(),
                    "predicted_score": 0.19204,
                    "predicted_delta_vs_base": -0.00001,
                },
            },
        ],
    }


def _contest_auth_source(
    path: Path,
    *,
    archive_sha256: str,
    aggregate_sha256: str,
    score: float,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "score_axis": "contest_cpu",
                "evidence_grade": "[contest-CPU]",
                "score_claim_valid": True,
                "archive_sha256": archive_sha256,
                "inflated_outputs_aggregate_sha256": aggregate_sha256,
                "canonical_score": score,
            }
        ),
        encoding="utf-8",
    )
    return path


def _exact_cpu_observation(
    source_path: Path,
    *,
    candidate_id: str = "prefix_k032",
    sweep_config_id: str = "contest_cpu_later",
) -> dict:
    return {
        "schema": "mlx_dynamic_sweep_observation.v1",
        "candidate_id": candidate_id,
        "sweep_config_id": sweep_config_id,
        "optimization_pass_id": "exact_cpu_calibration",
        "family": "decoder_q_selective_dqs1",
        "observed_axis": "contest_cpu",
        "evidence_tag": "[contest-CPU]",
        "evidence_grade": "[contest-CPU]",
        "observed_score_or_delta": 0.193,
        "archive_sha256": "a" * 64,
        "runtime_sha256": "c" * 64,
        "raw_output_or_cache_sha256": "b" * 64,
        "component_deltas": {
            "segnet_delta": 0.001,
            "posenet_delta": 0.002,
            "rate_delta": 0.0,
        },
        "source_artifact_path": str(source_path),
    }


def test_dynamic_sweep_ranks_configs_without_dispatch_authority() -> None:
    plan = build_mlx_dynamic_learned_sweep_plan(
        incumbent_score=0.1920513168811056,
        selector_pareto=_selector_pareto(),
        execution_configs=[
            {
                "config_id": "local_mlx_fast",
                "substrate": "[macOS-MLX research-signal]",
                "cost_units": 1.0,
                "signal_quality": 0.4,
                "exact_eval_candidate": False,
            },
            {
                "config_id": "contest_cpu_later",
                "substrate": "[contest-CPU]",
                "cost_units": 10.0,
                "signal_quality": 0.9,
                "exact_eval_candidate": True,
            },
        ],
        top_k=4,
    )

    assert plan["score_claim"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False
    assert plan["candidate_generation_only"] is True
    assert plan["summary"]["candidate_count"] == 2
    assert plan["summary"]["config_count"] == 2
    assert plan["summary"]["optimizer_scheduler_candidate_count"] >= 3
    assert plan["summary"]["optimizer_scheduler_pairing_count"] > 0
    assert plan["summary"]["optimization_pass_count"] == 4
    assert plan["recursive_learning_contract"]["optimizer_scheduler_registry_surface"].endswith(
        "enumerate_optimizer_scheduler_candidates"
    )
    assert plan["recursive_learning_contract"]["optimizer_scheduler_telemetry_surface"].endswith(
        "build_optimizer_scheduler_telemetry_record"
    )
    assert plan["optimizer_scheduler_candidates"]
    assert all(row["score_claim"] is False for row in plan["optimizer_scheduler_candidates"])
    assert any(
        row["parameter_group_lr_policy_id"] == "embedding_theta1_hidden_muon_adamw"
        for row in plan["optimizer_scheduler_candidates"]
    )
    assert all(
        row["rank_score_field"] == "planner_priority_not_score"
        for row in plan["optimizer_scheduler_candidates"]
    )
    pairing = plan["optimizer_scheduler_pairings"][0]
    assert pairing["schema"] == "mlx_dynamic_learned_sweep_optimizer_scheduler_pairing.v1"
    assert pairing["parent_queue_candidate_id"] in {
        row["queue_candidate_id"] for row in plan["ranked_sweep_rows"]
    }
    assert pairing["optimizer_scheduler_descriptor_id"]
    assert len(pairing["parameter_group_lr_policy_sha256"]) == 64
    assert pairing["paired_ablation_contract"]["score_claim"] is False
    assert "tools/master_gradient_xray.py" in pairing["tool_wiring"]["xray_surfaces"]
    assert "src/tac/atom/ledger.py" in pairing["tool_wiring"]["atom_surfaces"]
    assert "src/tac/freezing/swa_checkpoint_averaging.py" in pairing["tool_wiring"][
        "freezing_surfaces"
    ]
    assert pairing["solver_stack_wire_in"]["probe_disambiguator_wire_in"][
        "paired_modes"
    ] == [
        "same_candidate_config_pass_different_optimizer_scheduler",
        "same_optimizer_scheduler_different_candidate",
        "same_optimizer_scheduler_different_execution_substrate",
    ]
    assert {row["candidate_id"] for row in plan["ranked_sweep_rows"]} == {
        "prefix_k032",
        "prefix_k016",
    }
    assert any(row["ready_for_local_sweep"] for row in plan["ranked_sweep_rows"])
    exact_rows = [
        row for row in plan["ranked_sweep_rows"] if row["exact_eval_candidate"] is True
    ]
    assert exact_rows
    assert all("lane_claim_required_before_dispatch" in row["dispatch_blockers"] for row in exact_rows)
    assert all(row["score_claim"] is False for row in plan["ranked_sweep_rows"])
    best = next(row for row in plan["ranked_sweep_rows"] if row["candidate_id"] == "prefix_k032")
    assert best["optimization_pass_id"] in {"smoke", "micro", "intermediate", "macro"}
    assert best["geometry_multiplier"] == pytest.approx(1.75)
    assert best["component_axis_context"]["seg"] == 0.7
    assert best["canonical_equation_provenance"]["canonical_equation_id"].endswith("_v1")
    assert best["frozen_config_contract"]["score_claim"] is False
    assert best["queue_candidate_id"] == (
        f"{best['candidate_id']}::{best['sweep_config_id']}::{best['optimization_pass_id']}"
    )
    wire = best["solver_stack_wire_in"]
    assert wire["candidate_id"] == best["queue_candidate_id"]
    assert wire["cathedral_autopilot_wire_in"]["dispatch_ready"] is False
    assert wire["probe_disambiguator_wire_in"]["paired_modes"] == [
        "same_candidate_different_config",
        "same_config_different_candidate",
        "local_axis_then_exact_axis_anchor",
    ]


def test_dynamic_sweep_rejects_authoritative_candidate_rows() -> None:
    pareto = _selector_pareto()
    pareto["candidates"][0]["score_claim"] = True

    with pytest.raises(MLXDynamicLearnedSweepError, match="score_claim"):
        build_mlx_dynamic_learned_sweep_plan(
            incumbent_score=0.1920513168811056,
            selector_pareto=pareto,
        )


def test_dynamic_sweep_rejects_nested_authority_metadata() -> None:
    pareto = _selector_pareto()
    pareto["candidates"][0]["waterbucket_context"] = {
        "ready_for_exact_eval_dispatch": True
    }

    with pytest.raises(
        MLXDynamicLearnedSweepError,
        match=r"waterbucket_context\.ready_for_exact_eval_dispatch=truthy",
    ):
        build_mlx_dynamic_learned_sweep_plan(
            incumbent_score=0.1920513168811056,
            selector_pareto=pareto,
        )


def test_dynamic_sweep_rejects_authoritative_optimizer_scheduler_candidate() -> None:
    with pytest.raises(
        MLXDynamicLearnedSweepError,
        match=r"optimizer_scheduler_candidate\[0\].*score_claim=truthy",
    ):
        build_mlx_dynamic_learned_sweep_plan(
            incumbent_score=0.1920513168811056,
            selector_pareto=_selector_pareto(),
            optimizer_scheduler_candidates=[
                {
                    "descriptor_id": "unsafe",
                    "rank_score_field": "planner_priority_not_score",
                    "score_claim": True,
                }
            ],
        )


def test_dynamic_sweep_can_preserve_each_recursive_pass() -> None:
    plan = build_mlx_dynamic_learned_sweep_plan(
        incumbent_score=0.1920513168811056,
        selector_pareto=_selector_pareto(),
        top_k=4,
        per_pass_top_k=1,
    )

    assert plan["selection_policy"]["per_pass_top_k"] == 1
    assert [row["optimization_pass_id"] for row in plan["ranked_sweep_rows"]] == [
        "smoke",
        "micro",
        "intermediate",
        "macro",
    ]
    assert plan["summary"]["freeze_candidate_row_count"] == 1


def test_dynamic_sweep_suppresses_exact_observed_candidate_config_family(
    tmp_path: Path,
) -> None:
    source_path = _contest_auth_source(
        tmp_path / "contest_auth_eval.json",
        archive_sha256="a" * 64,
        aggregate_sha256="b" * 64,
        score=0.193,
    )

    plan = build_mlx_dynamic_learned_sweep_plan(
        incumbent_score=0.1920513168811056,
        selector_pareto=_selector_pareto(),
        execution_configs=[
            {
                "config_id": "local_mlx_fast",
                "substrate": "[macOS-MLX research-signal]",
                "cost_units": 1.0,
                "signal_quality": 0.4,
                "exact_eval_candidate": False,
            },
            {
                "config_id": "contest_cpu_later",
                "substrate": "[contest-CPU]",
                "cost_units": 10.0,
                "signal_quality": 0.9,
                "exact_eval_candidate": True,
            },
        ],
        observations=[_exact_cpu_observation(source_path)],
        top_k=12,
    )

    assert plan["summary"]["observation_row_count"] == 1
    assert plan["summary"]["suppressed_observed_row_count"] == 4
    assert not any(
        row["candidate_id"] == "prefix_k032"
        and row["sweep_config_id"] == "contest_cpu_later"
        for row in plan["ranked_sweep_rows"]
    )
    assert any(
        row["candidate_id"] == "prefix_k032"
        and row["sweep_config_id"] == "local_mlx_fast"
        for row in plan["ranked_sweep_rows"]
    )
    assert {
        row["observation_feedback"]["suppression_reason"]
        for row in plan["suppressed_observed_sweep_rows"]
    } == {"exact_axis_observed_candidate_config_family"}
    assert all(row["score_claim"] is False for row in plan["suppressed_observed_sweep_rows"])


def test_dynamic_sweep_rejects_impossible_stratified_cap() -> None:
    with pytest.raises(MLXDynamicLearnedSweepError, match="per_pass_top_k"):
        build_mlx_dynamic_learned_sweep_plan(
            incumbent_score=0.1920513168811056,
            selector_pareto=_selector_pareto(),
            top_k=3,
            per_pass_top_k=1,
        )


def test_dynamic_sweep_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    selector_path = tmp_path / "selector.json"
    source_path = _contest_auth_source(
        tmp_path / "contest_auth_eval.json",
        archive_sha256="a" * 64,
        aggregate_sha256="b" * 64,
        score=0.193,
    )
    observation_path = tmp_path / "observations.jsonl"
    json_out = tmp_path / "plan.json"
    md_out = tmp_path / "plan.md"
    selector_path.write_text(json.dumps(_selector_pareto()), encoding="utf-8")
    observation_path.write_text(
        json.dumps(
            _exact_cpu_observation(
                source_path,
                sweep_config_id="contest_cpu_exact_candidate",
            ),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "plan_mlx_dynamic_learned_sweep.py"),
            "--incumbent-score",
            "0.1920513168811056",
            "--selector-pareto",
            str(selector_path),
            "--top-k",
            "4",
            "--per-pass-top-k",
            "1",
            "--observation-jsonl",
            str(observation_path),
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    stdout_payload = json.loads(completed.stdout)
    assert stdout_payload["score_claim"] is False
    assert stdout_payload["suppressed_observed_row_count"] == 4
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["summary"]["ranked_row_count"] == 4
    assert payload["summary"]["freeze_candidate_row_count"] == 1
    assert payload["summary"]["observation_row_count"] == 1
    assert payload["summary"]["suppressed_observed_row_count"] == 4
    assert payload["source_artifacts"]["selector_pareto"]["sha256"]
    assert payload["source_artifacts"]["observation_jsonl"][0]["sha256"]
    assert "MLX Dynamic Learned Sweep Plan" in md_out.read_text(encoding="utf-8")
