# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.archive_byte_profile import CONTEST_ORIGINAL_BYTES
from tac.optimization.byte_shaving_campaign import (
    SIGNAL_SURFACE_SCHEMA,
    ByteShavingCampaignError,
    build_byte_shaving_campaign_plan,
    build_signal_surface_from_candidate_queue,
    build_signal_surface_from_master_gradient_anchor,
    validate_signal_surface,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools" / "plan_byte_shaving_campaign.py"


def _false_authority() -> dict[str, bool]:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "promotable": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
    }


def _surface() -> dict[str, object]:
    return {
        "schema": SIGNAL_SURFACE_SCHEMA,
        "campaign_id": "post_training_byte_shave_seed0",
        "candidate_id": "boostnerv_seed0",
        "lane_id": "boostnerv_post_train_shaving",
        "combo_beam_width": 16,
        "max_combo_count": 16,
        "units": [
            {
                "unit_id": "pair0371",
                "unit_kind": "pair",
                "candidate_saved_bytes": 1000,
                "predicted_quality_score_cost": 0.00015,
                "confidence": 0.9,
                "operations": [
                    {
                        "operation_id": "drop_pair",
                        "operation_family": "drop_pair",
                        "candidate_saved_bytes": 1000,
                        "predicted_quality_score_cost": 0.00015,
                    },
                    {
                        "operation_id": "substitute_pair",
                        "operation_family": "substitute_pair",
                        "candidate_saved_bytes": 700,
                        "predicted_quality_score_cost": 0.00002,
                    },
                ],
            },
            {
                "unit_id": "byte_null_run_a",
                "unit_kind": "byte_range",
                "candidate_saved_bytes": 500,
                "predicted_quality_score_cost": 0.0,
                "confidence": 0.95,
                "operation_families": ["null_remove_or_seed"],
            },
            {
                "unit_id": "tensor_head7",
                "unit_kind": "tensor",
                "candidate_saved_bytes": 900,
                "predicted_quality_score_cost": 0.0002,
                "confidence": 0.8,
                "operation_families": ["quantize_tensor"],
            },
        ],
        "interactions": [
            {
                "interaction_id": "pair_null_synergy",
                "unit_ids": ["pair0371", "byte_null_run_a"],
                "extra_saved_bytes": 120,
                "delta_score": -0.00001,
                "rationale": "shared selector/header overhead disappears together",
            }
        ],
        "conflicts": [{"unit_ids": ["pair0371", "tensor_head7"]}],
        **_false_authority(),
    }


def test_plan_builds_combination_ladder_with_interactions_and_conflicts() -> None:
    plan = build_byte_shaving_campaign_plan(_surface(), max_k=3)
    combo = plan["recommended_combination"]

    assert plan["schema"] == "byte_shaving_campaign_plan.v1"
    assert plan["ready_for_exact_eval_dispatch"] is False
    assert combo["selected_unit_ids"] == ["pair0371", "byte_null_run_a"]
    assert combo["candidate_saved_bytes"] == 1620
    assert combo["active_interactions"][0]["interaction_id"] == "pair_null_synergy"
    assert "tensor_head7" not in combo["selected_unit_ids"]
    assert combo["expected_delta_score"] == pytest.approx(
        -25.0 * 1620 / CONTEST_ORIGINAL_BYTES + 0.00015 - 0.00001
    )


def test_prefix_ladder_marks_conflicting_prefixes_and_does_not_recommend_them() -> None:
    plan = build_byte_shaving_campaign_plan(_surface(), max_k=3)
    conflicting = next(row for row in plan["sweep_ladder"] if row["sweep_id"] == "top_0003")

    assert conflicting["conflict_violations"] == [["pair0371", "tensor_head7"]]
    assert "prefix_selection_violates_conflict_sets" in conflicting["dispatch_blockers"]
    assert plan["recommended_prefix"]["sweep_id"] != "top_0003"


def test_plan_recommends_operation_alternative_when_drop_cost_is_too_high() -> None:
    surface = _surface()
    surface["units"][0]["operations"][0]["predicted_quality_score_cost"] = 0.005
    plan = build_byte_shaving_campaign_plan(surface, max_k=3)
    pair = next(row for row in plan["ranked_units"] if row["unit_id"] == "pair0371")

    assert pair["recommended_operation_family"] == "substitute_pair"
    assert pair["candidate_saved_bytes"] == 700


def test_signal_surface_rejects_truthy_authority() -> None:
    surface = _surface()
    surface["score_claim"] = True

    with pytest.raises(ByteShavingCampaignError, match="score_claim"):
        validate_signal_surface(surface)


def test_candidate_queue_surface_preserves_calibration_and_rejects_authority() -> None:
    queue = {
        "schema": "optimizer_candidate_queue_v1",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "top_k": [
            {
                "candidate_id": "drop_bytes",
                "source_candidate_id": "trained_seed7",
                "unit_kind": "byte_range",
                "operation_family": "entropy_recode",
                "target_kind": "byte_range_entropy_coder_v1",
                "materializer": "byte_range_entropy_coder_adapter",
                "operation_params": {"codec": "range"},
                "candidate_saved_bytes": 101,
                "predicted_quality_score_cost": 0.00001,
                "confidence": 0.7,
                "evidence_grade": "[macOS-MLX research-signal]",
                "evidence_semantics": "strict_calibrated_local_spend_triage",
                "source_paths": ["experiments/results/seed7/manifest.json"],
                "candidate_archive_sha256": "c" * 64,
                "candidate_archive_bytes": 178600,
                "local_axis": "macOS-MLX",
                "target_axis": "contest-CPU",
                "projected_contest_score": 0.19203,
                "master_gradient_provenance": {"anchor_count": 1},
                "canonical_equation_provenance": {"equation_id": "fixture_v1"},
                "atom_ids": ["atom_1"],
                "dispatch_blockers": ["needs_materializer"],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ],
    }

    surface = build_signal_surface_from_candidate_queue(queue)
    unit = surface["units"][0]
    plan = build_byte_shaving_campaign_plan(surface)
    ranked = plan["ranked_units"][0]

    assert unit["evidence_grade"] == "[macOS-MLX research-signal]"
    assert unit["candidate_archive_sha256"] == "c" * 64
    assert ranked["source_candidate_id"] == "trained_seed7"
    assert ranked["master_gradient_signal"] == {"anchor_count": 1}
    assert ranked["canonical_equation_provenance"] == {"equation_id": "fixture_v1"}
    assert ranked["atom_ids"] == ["atom_1"]
    assert ranked["recommended_operation_family"] == "entropy_recode"
    assert ranked["recommended_operation_materializer"] == "byte_range_entropy_coder_adapter"
    assert ranked["recommended_operation_target_kind"] == "byte_range_entropy_coder_v1"
    assert ranked["recommended_operation_params"] == {"codec": "range"}
    selected = plan["recommended_prefix"]["selected_operations"][0]
    assert selected["materializer"] == "byte_range_entropy_coder_adapter"
    assert selected["target_kind"] == "byte_range_entropy_coder_v1"
    assert selected["params"] == {"codec": "range"}

    queue["top_k"][0]["score_claim"] = True
    with pytest.raises(ByteShavingCampaignError, match="score_claim"):
        build_signal_surface_from_candidate_queue(queue)


def test_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    source = tmp_path / "surface.json"
    output = tmp_path / "plan.json"
    md_out = tmp_path / "plan.md"
    source.write_text(json.dumps(_surface()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--source",
            str(source),
            "--output",
            str(output),
            "--md-out",
            str(md_out),
            "--repo-root",
            str(tmp_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "combinations=" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["combination_ladder"]
    assert "Recommended Combination" in md_out.read_text(encoding="utf-8")


def test_master_gradient_anchor_builds_planning_only_byte_surface(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    repo = tmp_path
    state = repo / ".omx" / "state"
    state.mkdir(parents=True)
    archive_sha = "a" * 64
    gradient_path = state / "mg.npy"
    np.save(
        gradient_path,
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    ledger = state / "master_gradient_anchors.jsonl"
    ledger.write_text(
        json.dumps({
            "schema_version": "master_gradient_anchor_v1",
            "archive_sha256": archive_sha,
            "gradient_array_path": ".omx/state/mg.npy",
            "gradient_tensor_kind": "aggregate_per_byte_v1",
            "measurement_axis": "[macOS-CPU advisory]",
            "measurement_hardware": "darwin_arm64_local_cpu_advisory",
            "measurement_call_id": "local-test",
            "measurement_utc": "2026-05-23T00:00:00Z",
            "n_bytes": 5,
            "n_pairs_used": 1,
            "n_pairs_total": 5,
            "scored_archive_sha256": archive_sha,
            "scored_archive_bytes": 123,
        })
        + "\n",
        encoding="utf-8",
    )

    surface = build_signal_surface_from_master_gradient_anchor(
        archive_sha256=archive_sha,
        repo_root=repo,
        low_sensitivity_quantile=0.8,
        max_units=8,
    )
    plan = build_byte_shaving_campaign_plan(surface, repo_root=repo)

    assert surface["schema"] == SIGNAL_SURFACE_SCHEMA
    assert surface["score_claim"] is False
    assert len(surface["units"]) == 2
    assert surface["units"][0]["source_span"] == {"start": 0, "end_exclusive": 2}
    assert surface["units"][0]["master_gradient_signal"]["score_claim"] is False
    assert plan["recommended_combination"]["selected_unit_ids"] == [
        "mg_byte_span_0000000_0000002",
        "mg_byte_span_0000003_0000005",
    ]


def test_cli_can_plan_from_master_gradient_anchor(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    state = tmp_path / ".omx" / "state"
    state.mkdir(parents=True)
    archive_sha = "b" * 64
    np.save(
        state / "mg.npy",
        np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
    )
    (state / "master_gradient_anchors.jsonl").write_text(
        json.dumps({
            "schema_version": "master_gradient_anchor_v1",
            "archive_sha256": archive_sha,
            "gradient_array_path": ".omx/state/mg.npy",
            "gradient_tensor_kind": "aggregate_per_byte_v1",
            "measurement_axis": "[macOS-CPU advisory]",
            "measurement_hardware": "darwin_arm64_local_cpu_advisory",
            "measurement_call_id": "local-test",
            "measurement_utc": "2026-05-23T00:00:00Z",
            "n_bytes": 3,
        })
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "plan.json"

    subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--from-master-gradient-archive-sha",
            archive_sha,
            "--output",
            str(output),
            "--repo-root",
            str(tmp_path),
            "--master-gradient-low-quantile",
            "0.67",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "byte_shaving_campaign_plan.v1"
    assert payload["ranked_units"]
    assert payload["score_claim"] is False
