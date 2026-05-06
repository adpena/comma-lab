from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.optimization.field_equation_planner import (
    FieldEquationPlannerError,
    build_field_equation_plan,
    frechet_derivatives,
)
from tac.optimization.meta_lagrangian_allocator import build_atom_ledger
from tac.repo_io import sha256_file

REPO = Path(__file__).resolve().parents[3]


def _ledger():
    return build_atom_ledger(
        [
            {
                "atom_id": "wr01",
                "family": "wavelet",
                "family_group": "alpha",
                "pareto_scope": "alpha",
                "byte_delta": -9,
                "expected_seg_dist_delta": -0.0001,
                "expected_pose_dist_delta": -0.00001,
                "confidence": 0.8,
                "evidence_grade": "empirical",
                "raw_equal": True,
                "archive_manifest_path": "missing.json",
                "archive_manifest_sha256": "1" * 64,
                "interaction_assumptions": ["first_order_local"],
            },
            {
                "atom_id": "dominated_alpha",
                "family": "wavelet",
                "family_group": "alpha",
                "pareto_scope": "alpha",
                "byte_delta": 20,
                "expected_seg_dist_delta": 0.0,
                "expected_pose_dist_delta": 0.0,
                "confidence": 0.8,
                "evidence_grade": "empirical",
                "raw_equal": True,
            },
            {
                "atom_id": "pose_atom",
                "family": "lapose",
                "family_group": "pose",
                "pareto_scope": "pose",
                "byte_delta": 12,
                "expected_seg_dist_delta": 0.0,
                "expected_pose_dist_delta": -0.0002,
                "confidence": 0.9,
                "evidence_grade": "diagnostic_cuda",
                "raw_equal": True,
            },
        ],
        base_pose_dist=0.01,
        source="fixture",
    )


def test_frechet_derivatives_expose_score_component_and_byte_gradients() -> None:
    row = _ledger()["rows"][0]
    deriv = frechet_derivatives(row)

    assert deriv["d_score_d_epsilon"] == row["expected_total_score_delta"]
    assert deriv["d_seg_dist_d_epsilon"] == row["expected_seg_dist_delta"]
    assert deriv["d_pose_dist_d_epsilon"] == row["expected_pose_dist_delta"]
    assert deriv["d_bytes_d_epsilon"] == row["byte_delta"]
    assert deriv["d_rate_score_d_epsilon"] < 0


def test_field_equation_plan_emits_kkt_floor_and_trainable_surrogate() -> None:
    plan = build_field_equation_plan(
        _ledger(),
        source="fixture",
        base_score=0.20935073680571203,
        constraints={"min_confidence": 0.5},
    )

    assert plan["score_claim"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False
    assert plan["rows"][0]["atom_id"] == "wr01"
    assert plan["rows"][0]["interaction_assumptions"] == ["first_order_local"]
    assert plan["rows"][0]["pareto_eligible"] is False
    assert plan["rows"][0]["frechet_derivatives"]["d_score_d_epsilon"] < 0
    assert plan["rows"][0]["kkt"]["locally_descending"] is False
    assert plan["rows"][0]["kkt"]["selection_penalty_terms"]["missing_byte_closed_archive_manifest"] > 0.0
    assert plan["rows"][0]["variational_action_delta"] > plan["rows"][0]["frechet_derivatives"]["d_score_d_epsilon"]
    assert "pareto_ineligible_atom" in plan["rows"][0]["kkt"]["kkt_blockers"]
    assert "missing_byte_closed_archive_manifest" in plan["rows"][0]["kkt"]["kkt_blockers"]
    assert plan["pareto_eligible_count"] == 0
    assert plan["kkt_ready_for_field_planning_count"] == 0
    assert plan["kkt_blocker_counts"]["missing_byte_closed_archive_manifest"] == 3
    assert plan["theoretical_floor_estimate"]["evidence_grade"] == "derivation"
    assert plan["theoretical_floor_estimate"]["score_claim"] is False
    assert plan["theoretical_floor_estimate"]["base_score"] == pytest.approx(0.20935073680571203)
    assert plan["research_basis"]["score_claim"] is False
    assert "fridrich_stc_2011" in {
        source["basis_id"] for source in plan["research_basis"]["sources"]
    }
    assert "research_basis_is_not_score_evidence" in plan["rows"][0]["dispatch_blockers"]
    pose_row = next(row for row in plan["rows"] if row["atom_id"] == "pose_atom")
    assert "lapose_2026" in pose_row["research_basis_ids"]
    assert plan["trainable_surrogate"]["target"] == "minimize_variational_action_delta"
    assert "interaction_kernel" in plan["trainable_surrogate"]["trainable_parameters"]


def test_field_rows_carry_uncertainty_and_meta_selection_penalties(tmp_path: Path) -> None:
    manifest = _archive_manifest(tmp_path)
    manifest_sha = sha256_file(manifest)
    ledger = build_atom_ledger(
        [
            {
                "atom_id": "uncertainty_atom",
                "family": "mask_patch",
                "family_group": "mask",
                "pareto_scope": "mask",
                "byte_delta": -10,
                "expected_seg_dist_delta": -0.0001,
                "confidence": 1.0,
                "evidence_grade": "empirical",
                "raw_equal": True,
                "interaction_assumptions": ["first_order_mask_patch"],
                "expected_information_gain_nats": 0.25,
                "expected_score_variance": 0.04,
                "archive_manifest_path": manifest.as_posix(),
                "archive_manifest_sha256": manifest_sha,
            }
        ],
        base_pose_dist=0.01,
        source="fixture",
    )
    plan = build_field_equation_plan(ledger, source="fixture")

    row = plan["rows"][0]
    assert row["expected_information_gain_nats"] == pytest.approx(0.25)
    assert row["expected_score_variance"] == pytest.approx(0.04)
    assert row["expected_uncertainty_reduction"]["has_uncertainty_signal"] is True
    assert row["meta_lagrangian_selection_score_delta"] == pytest.approx(
        ledger["rows"][0]["selection_score_delta"]
    )
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["dispatchable"] is False


def test_field_action_penalizes_proxy_even_when_byte_closed(tmp_path: Path) -> None:
    manifest = _archive_manifest(tmp_path)
    manifest_sha = sha256_file(manifest)
    ledger = build_atom_ledger(
        [
            {
                "atom_id": "proxy_atom",
                "family": "mask_patch",
                "family_group": "mask",
                "pareto_scope": "mask",
                "byte_delta": -1_000,
                "expected_seg_dist_delta": -0.0001,
                "confidence": 1.0,
                "evidence_grade": "planning_proxy",
                "raw_equal": True,
                "interaction_assumptions": ["proxy_first_order"],
                "archive_manifest_path": manifest.as_posix(),
                "archive_manifest_sha256": manifest_sha,
            }
        ],
        base_pose_dist=0.01,
        source="fixture",
    )
    plan = build_field_equation_plan(ledger, source="fixture")

    row = plan["rows"][0]
    assert row["proxy_row"] is True
    assert row["pareto_eligible"] is False
    assert "proxy_evidence_not_kkt_ready" in row["kkt"]["kkt_blockers"]
    assert row["kkt"]["selection_penalty_terms"]["proxy_row"] > 0.0
    assert row["kkt"]["selection_penalty_terms"]["pareto_ineligible_atom"] > 0.0
    assert row["variational_action_delta"] > row["frechet_derivatives"]["d_score_d_epsilon"]


def test_volterra_interactions_are_second_order_planning_only() -> None:
    plan = build_field_equation_plan(
        _ledger(),
        source="fixture",
        interactions=[
            {
                "atom_a": "wr01",
                "atom_b": "pose_atom",
                "score_delta": -0.002,
                "seg_dist_delta": -0.00001,
                "pose_dist_delta": -0.00001,
                "byte_delta": 5,
                "assumption": "fixture_pair_synergy",
            }
        ],
    )

    pair = plan["volterra_interactions"][0]
    assert pair["volterra_order"] == 2
    assert pair["interaction_id"] == "volterra:pose_atom+wr01"
    assert pair["second_order_score_delta"] == -0.002
    assert pair["combined_score_delta"] < pair["first_order_score_delta"]
    assert pair["compatible_for_stack_planning"] is False
    assert "missing_byte_closed_archive_manifest" in pair["compatibility_blockers"]
    assert pair["ready_for_exact_eval_dispatch"] is False
    assert plan["compatible_volterra_interaction_count"] == 0
    assert plan["blocked_volterra_interaction_count"] == 1
    assert "requires_exact_stacked_archive_cuda_eval" in pair["dispatch_blockers"]


def test_volterra_pair_can_influence_floor_only_when_kkt_frontier_and_byte_closed(
    tmp_path: Path,
) -> None:
    ledger = _closed_stack_ledger(tmp_path)
    plan = build_field_equation_plan(
        ledger,
        source="fixture",
        interactions=[
            {
                "atom_a": "mask_atom",
                "atom_b": "pose_atom",
                "score_delta": -0.003,
                "seg_dist_delta": -0.00001,
                "pose_dist_delta": -0.00001,
                "byte_delta": 2,
                "assumption": "fixture_byte_closed_nonconflicting_pair",
            }
        ],
    )

    pair = plan["volterra_interactions"][0]
    assert pair["compatible_for_stack_planning"] is True
    assert pair["compatibility_blockers"] == []
    assert plan["compatible_volterra_interaction_count"] == 1
    assert plan["blocked_volterra_interaction_count"] == 0
    assert plan["theoretical_floor_estimate"]["best_pair_combined_score_delta"] == pytest.approx(
        pair["combined_score_delta"]
    )


def test_declared_family_conflict_blocks_volterra_floor_contribution(tmp_path: Path) -> None:
    ledger = _closed_stack_ledger(
        tmp_path,
        first_extra={"conflicts_with_families": ["pose"]},
    )
    plan = build_field_equation_plan(
        ledger,
        source="fixture",
        interactions=[
            {
                "atom_a": "mask_atom",
                "atom_b": "pose_atom",
                "score_delta": -0.01,
                "seg_dist_delta": -0.00001,
                "pose_dist_delta": -0.00001,
                "byte_delta": -5,
                "assumption": "unsafe_fixture_conflict_synergy",
            }
        ],
    )

    pair = plan["volterra_interactions"][0]
    assert pair["combined_score_delta"] < 0
    assert pair["compatible_for_stack_planning"] is False
    assert "declared_family_conflict" in pair["compatibility_blockers"]
    assert "declared_family_conflict" in pair["dispatch_blockers"]
    assert plan["compatible_volterra_interaction_count"] == 0
    assert plan["blocked_volterra_interaction_count"] == 1
    assert plan["volterra_blocker_counts"]["declared_family_conflict"] == 1
    assert plan["theoretical_floor_estimate"]["best_pair_combined_score_delta"] == 0.0


def test_unknown_constraint_fails_closed() -> None:
    with pytest.raises(FieldEquationPlannerError, match="unknown constraint"):
        build_field_equation_plan(_ledger(), source="fixture", constraints={"lambda_magic": 1.0})


def test_build_field_equation_plan_cli(tmp_path) -> None:
    ledger = tmp_path / "ledger.json"
    out = tmp_path / "plan.json"
    ledger.write_text(json.dumps(_ledger()), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_field_equation_plan.py"),
            "--atom-ledger",
            str(ledger),
            "--source",
            "fixture",
            "--base-score",
            "0.20935073680571203",
            "--json-out",
            str(out),
            "--research-basis-id",
            "foveated_telepresence_2025",
        ],
        check=True,
        text=True,
    )

    payload = json.loads(out.read_text())
    assert payload["tool"] == "tac.optimization.field_equation_planner.build_field_equation_plan"
    assert payload["score_claim"] is False
    assert "foveated_telepresence_2025" in {
        source["basis_id"] for source in payload["research_basis"]["sources"]
    }
    assert payload["theoretical_floor_estimate"]["base_score"] == pytest.approx(0.20935073680571203)


def _closed_stack_ledger(
    tmp_path: Path,
    *,
    first_extra: dict | None = None,
    second_extra: dict | None = None,
) -> dict:
    manifest = _archive_manifest(tmp_path)
    manifest_sha = sha256_file(manifest)
    return build_atom_ledger(
        [
            {
                "atom_id": "mask_atom",
                "family": "mask_patch",
                "family_group": "mask",
                "pareto_scope": "cross_paradigm_stack",
                "byte_delta": -10,
                "expected_seg_dist_delta": -0.0001,
                "expected_pose_dist_delta": 0.0,
                "confidence": 1.0,
                "evidence_grade": "empirical",
                "raw_equal": True,
                "interaction_assumptions": ["first_order_mask_patch"],
                "archive_manifest_path": manifest.as_posix(),
                "archive_manifest_sha256": manifest_sha,
                **(first_extra or {}),
            },
            {
                "atom_id": "pose_atom",
                "family": "pose_patch",
                "family_group": "pose",
                "pareto_scope": "cross_paradigm_stack",
                "byte_delta": -10,
                "expected_seg_dist_delta": 0.0,
                "expected_pose_dist_delta": -0.0001,
                "confidence": 1.0,
                "evidence_grade": "empirical",
                "raw_equal": True,
                "interaction_assumptions": ["first_order_pose_patch"],
                "archive_manifest_path": manifest.as_posix(),
                "archive_manifest_sha256": manifest_sha,
                **(second_extra or {}),
            },
        ],
        base_pose_dist=0.01,
        source="fixture",
    )


def _archive_manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "archive-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "archive": {"bytes": 123, "sha256": "a" * 64},
                "score_claim": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest
