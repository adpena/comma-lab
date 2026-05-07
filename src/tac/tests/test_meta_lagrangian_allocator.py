from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.optimization.meta_lagrangian_allocator import (
    atoms_from_hnerv_decoder_recode_profile,
    build_atom_ledger,
    expected_atom_score_delta,
    pose_score_delta,
    rate_score_delta,
)
from tac.repo_io import sha256_file

REPO = Path(__file__).resolve().parents[3]


def test_rate_and_pose_score_terms_match_contest_formula() -> None:
    assert rate_score_delta(-151) < 0
    assert pose_score_delta(0.01, -0.001) < 0
    with pytest.raises(ValueError, match="negative"):
        pose_score_delta(0.01, -0.02)


def test_expected_atom_score_delta_combines_rate_seg_pose_and_priors(tmp_path: Path) -> None:
    manifest = _archive_manifest(tmp_path)
    row = expected_atom_score_delta(
        {
            "atom_id": "pair75_lane_repair",
            "family": "mask_repair",
            "byte_delta": 100,
            "expected_seg_dist_delta": -0.0001,
            "expected_pose_dist_delta": -0.00001,
            "confidence": 0.5,
            "evidence_grade": "empirical",
            "hard_pair_support": [75],
            "class_support": [2, 3],
            "geometry_priors": ["foveal_lane_boundary"],
            "openpilot_priors": ["ego_motion"],
            "family_group": "mask_repair_local",
            "conflicts_with_families": ["whole_mask_replacement"],
            "conflicts_with_atoms": ["atom:global_crf"],
            "interaction_assumptions": ["first_order_local_patch"],
            "interaction_model": "first_order_volterra_local_patch",
            "volterra_order": 1,
            "volterra_terms": ["linear_pair_response"],
            "kkt_proof": _kkt_proof(),
            "archive_manifest_path": manifest.as_posix(),
            "archive_manifest_sha256": sha256_file(manifest),
        },
        base_pose_dist=0.01,
    )

    assert row["expected_total_score_delta"] < 0
    assert row["hard_pair_support"] == [75]
    assert row["class_support"] == [2, 3]
    assert row["family_group"] == "mask_repair_local"
    assert row["conflicts_with_families"] == ["whole_mask_replacement"]
    assert row["conflicts_with_atoms"] == ["atom:global_crf"]
    assert row["interaction_assumptions"] == ["first_order_local_patch"]
    assert row["field_interaction_contract"]["schema"] == "field_interaction_contract_v1"
    assert row["field_interaction_contract"]["status"] == "passed"
    assert row["field_interaction_contract"]["assumptions"] == ["first_order_local_patch"]
    assert row["field_interaction_contract"]["interaction_model"] == "first_order_volterra_local_patch"
    assert row["field_interaction_contract"]["volterra_order"] == 1
    assert row["field_interaction_contract"]["volterra_terms"] == ["linear_pair_response"]
    assert row["byte_closed_archive_manifest_attached"] is True
    assert row["archive_manifest_custody"]["verified"] is True
    assert row["pareto_eligible"] is True
    assert row["non_dominated_frontier_reason"]["schema"] == "non_dominated_frontier_reason_v1"
    assert row["non_dominated_frontier_reason"]["status"] == "non_dominated"
    assert row["non_dominated_frontier_reason"]["reason"] == "non_dominated_within_pareto_scope"
    assert row["archive_ready_for_stack_review"] is True
    assert row["kkt_ready_for_field_planning"] is True
    assert row["dispatchable"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["exact_dispatch_blockers"]["schema"] == "exact_dispatch_blockers_v1"
    assert "requires_exact_cuda_auth_eval" in row["exact_dispatch_blockers"]["blockers"]
    assert (
        "candidate_packet_static_preflight_and_exact_cuda_auth_eval"
        in row["exact_dispatch_blockers"]["next_required_proof"]
    )


def test_hnerv_profile_atoms_rank_rate_only_variants() -> None:
    profile = {
        "source_label": "PR106x",
        "variants": [
            {"variant": "bad", "byte_delta_vs_source_section": 10, "raw_equal": True},
            {"variant": "good", "byte_delta_vs_source_section": -151, "raw_equal": True},
        ],
    }
    atoms = atoms_from_hnerv_decoder_recode_profile(profile)
    ledger = build_atom_ledger(atoms, base_pose_dist=0.01, source="fixture")

    assert ledger["score_claim"] is False
    assert ledger["rows"][0]["atom_id"].endswith(":good")
    assert ledger["rows"][0]["expected_total_score_delta"] < 0
    assert ledger["family_group_counts"] == {"hnerv_rate_equivalent_recode": 2}
    assert ledger["byte_closed_archive_manifest_attached_count"] == 0


def test_invalid_byte_only_atom_cannot_rank_ahead_of_valid_equivalent_atom() -> None:
    atoms = [
        {
            "atom_id": "invalid_big_byte_cut",
            "family": "hnerv_decoder_rate_recode",
            "byte_delta": -100_000,
            "confidence": 0.0,
            "evidence_grade": "invalid",
            "raw_equal": False,
        },
        {
            "atom_id": "valid_small_byte_cut",
            "family": "hnerv_decoder_rate_recode",
            "byte_delta": -1,
            "confidence": 1.0,
            "evidence_grade": "empirical_byte_raw_equal",
            "raw_equal": True,
        },
    ]
    ledger = build_atom_ledger(atoms, base_pose_dist=0.01, source="fixture")

    assert ledger["rows"][0]["atom_id"] == "valid_small_byte_cut"
    invalid = next(row for row in ledger["rows"] if row["atom_id"] == "invalid_big_byte_cut")
    assert invalid["rankable"] is False
    assert "non_rankable_evidence_grade" in invalid["dispatch_blockers"]
    assert "raw_output_not_byte_equivalent" in invalid["dispatch_blockers"]


def test_allocated_global_response_atom_is_not_rankable() -> None:
    row = expected_atom_score_delta(
        {
            "atom_id": "lapose_allocated_global_response",
            "family": "lapose_motion_atom",
            "byte_delta": -10,
            "expected_seg_dist_delta": -0.001,
            "expected_pose_dist_delta": -0.0001,
            "confidence": 0.75,
            "evidence_grade": "diagnostic_cuda_global_response_allocated",
            "allocation_inference": True,
        },
        base_pose_dist=0.01,
    )

    assert row["allocation_inference"] is True
    assert row["rankable"] is False
    assert "allocated_global_response_not_rankable" in row["dispatch_blockers"]


def test_requested_dispatchable_atom_without_archive_manifest_is_refused() -> None:
    ledger = build_atom_ledger(
        [
            {
                "atom_id": "unsafe_dispatch_request",
                "family": "categorical_mask_atom",
                "family_group": "categorical_qma9",
                "byte_delta": -12,
                "confidence": 1.0,
                "evidence_grade": "empirical",
                "raw_equal": True,
                "dispatchable": True,
                "conflicts_with_families": ["whole_mask_replacement"],
            }
        ],
        base_pose_dist=0.01,
        source="fixture",
    )

    row = ledger["rows"][0]
    assert row["requested_dispatchable"] is True
    assert row["dispatchable"] is False
    assert row["byte_closed_archive_manifest_attached"] is False
    assert "requested_dispatchable_without_byte_closed_archive_manifest" in row["dispatch_blockers"]
    assert row["pareto_eligible"] is False
    assert ledger["requested_dispatchable_refused_count"] == 1
    assert ledger["conflict_family_counts"] == {"whole_mask_replacement": 1}


def test_byte_closed_manifest_allows_stack_review_but_not_dispatch(tmp_path: Path) -> None:
    manifest = _archive_manifest(tmp_path)
    row = expected_atom_score_delta(
        {
            "atom_id": "byte_closed_local_candidate",
            "family": "hnerv_decoder_rate_recode",
            "byte_delta": -151,
            "confidence": 1.0,
            "evidence_grade": "empirical_byte_raw_equal",
            "raw_equal": True,
            "archive_manifest_path": manifest.as_posix(),
            "archive_manifest_sha256": sha256_file(manifest),
        },
        base_pose_dist=0.01,
    )

    assert row["rankable"] is True
    assert row["pareto_eligible"] is True
    assert row["archive_ready_for_stack_review"] is True
    assert row["dispatchable"] is False
    assert "requires_exact_cuda_auth_eval" in row["dispatch_blockers"]


def test_pareto_frontier_marks_dominated_atoms_within_scope(tmp_path: Path) -> None:
    manifest = _archive_manifest(tmp_path)
    atoms = [
        {
            "atom_id": "dominator",
            "family": "hnerv_decoder_rate_recode",
            "family_group": "hnerv_recode",
            "pareto_scope": "hnerv_recode",
            "byte_delta": -200,
            "expected_seg_dist_delta": -0.0002,
            "expected_pose_dist_delta": -0.00002,
            "confidence": 1.0,
            "evidence_grade": "empirical_byte_raw_equal",
            "raw_equal": True,
            "archive_manifest_path": manifest.as_posix(),
            "archive_manifest_sha256": sha256_file(manifest),
        },
        {
            "atom_id": "dominated",
            "family": "hnerv_decoder_rate_recode",
            "family_group": "hnerv_recode",
            "pareto_scope": "hnerv_recode",
            "byte_delta": -100,
            "expected_seg_dist_delta": -0.0001,
            "expected_pose_dist_delta": -0.00001,
            "confidence": 1.0,
            "evidence_grade": "empirical_byte_raw_equal",
            "raw_equal": True,
            "archive_manifest_path": manifest.as_posix(),
            "archive_manifest_sha256": sha256_file(manifest),
        },
    ]
    ledger = build_atom_ledger(atoms, base_pose_dist=0.01, source="fixture")

    assert ledger["pareto_summary"]["rankable_frontier_count"] == 1
    assert ledger["pareto_summary"]["rankable_dominated_count"] == 1
    assert ledger["rows"][0]["atom_id"] == "dominator"
    dominated = next(row for row in ledger["rows"] if row["atom_id"] == "dominated")
    assert dominated["pareto_frontier"] is False
    assert dominated["pareto_dominated_by"] == ["dominator"]
    assert dominated["non_dominated_frontier_reason"]["status"] == "dominated"
    assert dominated["non_dominated_frontier_reason"]["dominated_by"] == ["dominator"]
    assert (
        dominated["non_dominated_frontier_reason"]["reason"]
        == "dominated_within_pareto_scope_by_byte_closed_non_proxy_candidate"
    )
    assert dominated["pareto_objectives"]["byte_delta"] == -100.0
    assert dominated["selection_penalty_terms"]["pareto_dominated_atom"] > 0.0
    assert "pareto_dominated_within_scope" in dominated["exact_dispatch_blockers"]["blockers"]
    assert dominated["selection_score_delta"] == pytest.approx(
        dominated["expected_total_score_delta"]
    )
    assert dominated["selection_penalty_units"] > 0.0


def test_pareto_scope_preserves_orthogonal_families(tmp_path: Path) -> None:
    manifest = _archive_manifest(tmp_path)
    ledger = build_atom_ledger(
        [
            {
                "atom_id": "mask_atom",
                "family_group": "mask",
                "pareto_scope": "mask",
                "byte_delta": -20,
                "expected_seg_dist_delta": -0.0001,
                "confidence": 1.0,
                "evidence_grade": "empirical",
                "raw_equal": True,
                "archive_manifest_path": manifest.as_posix(),
                "archive_manifest_sha256": sha256_file(manifest),
            },
            {
                "atom_id": "pose_atom",
                "family_group": "pose",
                "pareto_scope": "pose",
                "byte_delta": -200,
                "expected_seg_dist_delta": -0.0002,
                "confidence": 1.0,
                "evidence_grade": "empirical",
                "raw_equal": True,
                "archive_manifest_path": manifest.as_posix(),
                "archive_manifest_sha256": sha256_file(manifest),
            },
        ],
        base_pose_dist=0.01,
        source="fixture",
    )

    assert ledger["pareto_summary"]["rankable_frontier_count"] == 2
    assert all(row["pareto_frontier"] is True for row in ledger["rows"])


def test_pareto_frontier_requires_verified_byte_closed_manifest(tmp_path: Path) -> None:
    manifest = _archive_manifest(tmp_path)
    ledger = build_atom_ledger(
        [
            {
                "atom_id": "unclosed_proxy_best_delta",
                "family": "mask_atom",
                "family_group": "mask",
                "pareto_scope": "mask",
                "byte_delta": -500,
                "expected_seg_dist_delta": -0.001,
                "confidence": 1.0,
                "evidence_grade": "empirical",
                "raw_equal": True,
            },
            {
                "atom_id": "closed_weaker_delta",
                "family": "mask_atom",
                "family_group": "mask",
                "pareto_scope": "mask",
                "byte_delta": -5,
                "expected_seg_dist_delta": -0.00001,
                "confidence": 1.0,
                "evidence_grade": "empirical",
                "raw_equal": True,
                "archive_manifest_path": manifest.as_posix(),
                "archive_manifest_sha256": sha256_file(manifest),
            },
        ],
        base_pose_dist=0.01,
        source="fixture",
    )

    assert ledger["pareto_eligible_count"] == 1
    assert ledger["rows"][0]["atom_id"] == "closed_weaker_delta"
    unclosed = next(row for row in ledger["rows"] if row["atom_id"] == "unclosed_proxy_best_delta")
    assert unclosed["rankable"] is True
    assert unclosed["pareto_eligible"] is False
    assert unclosed["pareto_frontier"] is False
    assert unclosed["pareto_eligibility_blockers"] == ["missing_byte_closed_archive_manifest"]
    assert unclosed["non_dominated_frontier_reason"]["status"] == "ineligible"
    assert unclosed["non_dominated_frontier_reason"]["eligibility_blockers"] == [
        "missing_byte_closed_archive_manifest"
    ]
    assert "missing_byte_closed_archive_manifest" in unclosed["kkt_blockers"]
    assert unclosed["selection_penalty_terms"]["missing_byte_closed_archive_manifest"] > 0.0
    assert (
        "verified_byte_closed_archive_manifest_with_sha256_and_bytes"
        in unclosed["exact_dispatch_blockers"]["next_required_proof"]
    )
    assert unclosed["selection_score_delta"] == pytest.approx(
        unclosed["expected_total_score_delta"]
    )
    assert unclosed["selection_penalty_units"] > 0.0


def test_selection_penalizes_proxy_and_unclosed_rows_before_raw_delta(tmp_path: Path) -> None:
    manifest = _archive_manifest(tmp_path)
    manifest_sha = sha256_file(manifest)
    ledger = build_atom_ledger(
        [
            {
                "atom_id": "byte_closed_frontier",
                "family": "mask_atom",
                "family_group": "mask",
                "pareto_scope": "mask",
                "byte_delta": -1,
                "confidence": 1.0,
                "evidence_grade": "empirical",
                "raw_equal": True,
                "interaction_assumptions": ["byte_closed_first_order"],
                "archive_manifest_path": manifest.as_posix(),
                "archive_manifest_sha256": manifest_sha,
            },
            {
                "atom_id": "byte_closed_proxy_large_delta",
                "family": "mask_atom",
                "family_group": "mask",
                "pareto_scope": "mask",
                "byte_delta": -1_000_000,
                "confidence": 1.0,
                "evidence_grade": "planning_proxy",
                "raw_equal": True,
                "interaction_assumptions": ["proxy_only"],
                "archive_manifest_path": manifest.as_posix(),
                "archive_manifest_sha256": manifest_sha,
            },
            {
                "atom_id": "unclosed_large_delta",
                "family": "mask_atom",
                "family_group": "mask",
                "pareto_scope": "mask",
                "byte_delta": -2_000_000,
                "confidence": 1.0,
                "evidence_grade": "empirical",
                "raw_equal": True,
                "interaction_assumptions": ["missing_archive_manifest"],
            },
        ],
        base_pose_dist=0.01,
        source="fixture",
    )

    assert ledger["rows"][0]["atom_id"] == "byte_closed_frontier"
    assert ledger["pareto_eligible_count"] == 1
    proxy = next(row for row in ledger["rows"] if row["atom_id"] == "byte_closed_proxy_large_delta")
    unclosed = next(row for row in ledger["rows"] if row["atom_id"] == "unclosed_large_delta")
    assert proxy["pareto_eligible"] is False
    assert proxy["selection_penalty_terms"]["proxy_row"] > 0.0
    assert proxy["selection_penalty_terms"]["pareto_ineligible_atom"] > 0.0
    assert unclosed["byte_closed_archive_manifest_attached"] is False
    assert unclosed["selection_penalty_terms"]["missing_byte_closed_archive_manifest"] > 0.0
    assert ledger["selection_policy"] == (
        "lexicographic_feasibility_tuple_orders_rows_before_pure_expected_score_delta; "
        "penalty_terms_are_diagnostic_only"
    )


def test_kkt_ready_requires_real_kkt_proof(tmp_path: Path) -> None:
    manifest = _archive_manifest(tmp_path)
    row = expected_atom_score_delta(
        {
            "atom_id": "missing_kkt_proof",
            "family": "mask_atom",
            "family_group": "mask",
            "byte_delta": -1,
            "confidence": 1.0,
            "evidence_grade": "empirical",
            "raw_equal": True,
            "interaction_assumptions": ["first_order"],
            "archive_manifest_path": manifest.as_posix(),
            "archive_manifest_sha256": sha256_file(manifest),
        },
        base_pose_dist=0.01,
    )

    assert row["kkt_ready_for_field_planning"] is False
    assert row["kkt_proof"]["status"] == "blocked"
    assert "kkt:kkt_proof_or_admm_result_missing" in row["kkt_blockers"]
    assert row["selection_penalty_terms"]["kkt_not_ready_for_field_planning"] > 0.0
    assert row["selection_score_delta"] == pytest.approx(row["expected_total_score_delta"])
    assert row["lexicographic_feasibility_tuple"][5] is False


def test_converged_admm_result_can_make_atom_kkt_ready(tmp_path: Path) -> None:
    manifest = _archive_manifest(tmp_path)
    row = expected_atom_score_delta(
        {
            "atom_id": "admm_kkt_ready",
            "family": "joint_stack",
            "family_group": "joint_stack",
            "byte_delta": -1,
            "confidence": 1.0,
            "evidence_grade": "empirical",
            "raw_equal": True,
            "interaction_assumptions": ["admm_waterline_checked"],
            "archive_manifest_path": manifest.as_posix(),
            "archive_manifest_sha256": sha256_file(manifest),
            "admm_result": {
                "converged": True,
                "waterline_kkt_residual": 0.001,
                "kkt_waterline_satisfied": True,
            },
        },
        base_pose_dist=0.01,
    )

    assert row["kkt_ready_for_field_planning"] is True
    assert row["kkt_proof"]["kind"] == "admm_result"
    assert row["selection_score_delta"] == pytest.approx(row["expected_total_score_delta"])
    assert row["lexicographic_feasibility_tuple"][5] is True


def test_information_gain_and_uncertainty_are_preserved_and_tiebreak_ranked(
    tmp_path: Path,
) -> None:
    manifest = _archive_manifest(tmp_path)
    manifest_sha = sha256_file(manifest)
    common = {
        "family": "mask_atom",
        "family_group": "mask",
        "pareto_scope": "mask",
        "byte_delta": -10,
        "confidence": 1.0,
        "evidence_grade": "empirical",
        "raw_equal": True,
        "interaction_assumptions": ["first_order"],
        "archive_manifest_path": manifest.as_posix(),
        "archive_manifest_sha256": manifest_sha,
    }
    ledger = build_atom_ledger(
        [
            {
                **common,
                "atom_id": "lower_information_gain",
                "expected_information_gain_nats": 0.1,
                "expected_score_variance": 0.04,
                "observation_noise_variance": 0.01,
            },
            {
                **common,
                "atom_id": "higher_information_gain",
                "expected_information_gain_nats": 0.5,
                "expected_score_variance": 0.04,
                "observation_noise_variance": 0.01,
            },
        ],
        base_pose_dist=0.01,
        source="fixture",
    )

    assert ledger["rows"][0]["atom_id"] == "higher_information_gain"
    assert ledger["rows"][0]["expected_information_gain_nats"] == pytest.approx(0.5)
    assert ledger["rows"][0]["expected_score_variance"] == pytest.approx(0.04)
    assert ledger["rows"][0]["expected_uncertainty_reduction"]["source_fields"] == [
        "expected_information_gain_nats",
        "expected_score_variance",
        "observation_noise_variance",
    ]


def test_archive_manifest_file_must_describe_closed_archive_bytes(tmp_path: Path) -> None:
    manifest = tmp_path / "not-byte-closed.json"
    manifest.write_text(json.dumps({"score_claim": False}, sort_keys=True), encoding="utf-8")
    row = expected_atom_score_delta(
        {
            "atom_id": "manifest_without_archive_record",
            "family": "hnerv_decoder_rate_recode",
            "byte_delta": -151,
            "confidence": 1.0,
            "evidence_grade": "empirical_byte_raw_equal",
            "raw_equal": True,
            "archive_manifest_path": manifest.as_posix(),
            "archive_manifest_sha256": sha256_file(manifest),
        },
        base_pose_dist=0.01,
    )

    assert row["archive_manifest_custody"]["sha256_matches"] is True
    assert row["archive_manifest_custody"]["verified"] is False
    assert row["byte_closed_archive_manifest_attached"] is False
    assert "archive_manifest_archive_sha256_missing_or_invalid" in row["dispatch_blockers"]
    assert "archive_manifest_archive_bytes_missing_or_invalid" in row["dispatch_blockers"]


def test_dispatchable_proxy_row_is_refused_even_with_byte_closed_manifest(tmp_path: Path) -> None:
    manifest = _archive_manifest(tmp_path)
    row = expected_atom_score_delta(
        {
            "atom_id": "proxy_candidate",
            "family": "field_proxy",
            "byte_delta": -151,
            "confidence": 1.0,
            "evidence_grade": "planning_proxy",
            "raw_equal": True,
            "dispatchable": True,
            "interaction_assumptions": ["proxy_first_order_only"],
            "archive_manifest_path": manifest.as_posix(),
            "archive_manifest_sha256": sha256_file(manifest),
        },
        base_pose_dist=0.01,
    )

    assert row["proxy_row"] is True
    assert row["requested_dispatchable"] is True
    assert row["byte_closed_archive_manifest_attached"] is True
    assert row["dispatchable"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["kkt_ready_for_field_planning"] is False
    assert "proxy_row_not_dispatchable" in row["dispatch_blockers"]
    assert "requested_dispatchable_proxy_row_refused" in row["dispatch_blockers"]


def test_unverified_archive_manifest_does_not_allow_stack_review(tmp_path: Path) -> None:
    manifest = _archive_manifest(tmp_path)
    row = expected_atom_score_delta(
        {
            "atom_id": "stale_manifest_candidate",
            "family": "hnerv_decoder_rate_recode",
            "byte_delta": -151,
            "confidence": 1.0,
            "evidence_grade": "empirical_byte_raw_equal",
            "raw_equal": True,
            "archive_manifest_path": manifest.as_posix(),
            "archive_manifest_sha256": "0" * 64,
        },
        base_pose_dist=0.01,
    )

    assert row["rankable"] is True
    assert row["archive_manifest_custody"]["verified"] is False
    assert row["archive_manifest_custody"]["sha256_matches"] is False
    assert row["byte_closed_archive_manifest_attached"] is False
    assert row["archive_ready_for_stack_review"] is False
    assert row["pareto_eligible"] is False
    assert "archive_manifest_sha256_mismatch" in row["dispatch_blockers"]


def test_missing_archive_manifest_does_not_allow_stack_review(tmp_path: Path) -> None:
    missing = tmp_path / "missing-manifest.json"
    row = expected_atom_score_delta(
        {
            "atom_id": "missing_manifest_candidate",
            "family": "hnerv_decoder_rate_recode",
            "byte_delta": -151,
            "confidence": 1.0,
            "evidence_grade": "empirical_byte_raw_equal",
            "raw_equal": True,
            "archive_manifest_path": missing.as_posix(),
            "archive_manifest_sha256": "1" * 64,
        },
        base_pose_dist=0.01,
    )

    assert row["rankable"] is True
    assert row["archive_manifest_custody"]["exists"] is False
    assert row["byte_closed_archive_manifest_attached"] is False
    assert row["archive_ready_for_stack_review"] is False
    assert row["pareto_eligible"] is False
    assert "archive_manifest_path_missing" in row["dispatch_blockers"]


def test_build_meta_lagrangian_atom_ledger_cli(tmp_path: Path) -> None:
    profile = tmp_path / "profile.json"
    out = tmp_path / "ledger.json"
    profile.write_text(
        json.dumps(
            {
                "source_label": "PR106x",
                "variants": [
                    {"variant": "good", "byte_delta_vs_source_section": -151, "raw_equal": True}
                ],
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_meta_lagrangian_atom_ledger.py"),
            "--hnerv-decoder-profile",
            str(profile),
            "--base-pose-dist",
            "0.01",
            "--source",
            "fixture",
            "--json-out",
            str(out),
        ],
        check=True,
        text=True,
    )

    payload = json.loads(out.read_text())
    assert payload["atom_count"] == 1
    assert payload["ready_for_exact_eval_dispatch"] is False


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


def _kkt_proof() -> dict[str, object]:
    return {
        "status": "passed",
        "stationarity_residual": 0.0,
        "stationarity_tolerance": 0.001,
    }
