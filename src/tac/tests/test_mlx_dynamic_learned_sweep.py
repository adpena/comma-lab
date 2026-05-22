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
    assert plan["summary"]["optimization_pass_count"] == 4
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


def test_dynamic_sweep_rejects_authoritative_candidate_rows() -> None:
    pareto = _selector_pareto()
    pareto["candidates"][0]["score_claim"] = True

    with pytest.raises(MLXDynamicLearnedSweepError, match="score_claim"):
        build_mlx_dynamic_learned_sweep_plan(
            incumbent_score=0.1920513168811056,
            selector_pareto=pareto,
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
    json_out = tmp_path / "plan.json"
    md_out = tmp_path / "plan.md"
    selector_path.write_text(json.dumps(_selector_pareto()), encoding="utf-8")

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
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["summary"]["ranked_row_count"] == 4
    assert payload["summary"]["freeze_candidate_row_count"] == 1
    assert payload["source_artifacts"]["selector_pareto"]["sha256"]
    assert "MLX Dynamic Learned Sweep Plan" in md_out.read_text(encoding="utf-8")
