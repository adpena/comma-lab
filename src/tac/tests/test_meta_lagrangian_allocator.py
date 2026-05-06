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

REPO = Path(__file__).resolve().parents[3]


def test_rate_and_pose_score_terms_match_contest_formula() -> None:
    assert rate_score_delta(-151) < 0
    assert pose_score_delta(0.01, -0.001) < 0
    with pytest.raises(ValueError, match="negative"):
        pose_score_delta(0.01, -0.02)


def test_expected_atom_score_delta_combines_rate_seg_pose_and_priors() -> None:
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
            "archive_manifest_path": "candidate/manifest.json",
            "archive_manifest_sha256": "a" * 64,
        },
        base_pose_dist=0.01,
    )

    assert row["expected_total_score_delta"] < 0
    assert row["hard_pair_support"] == [75]
    assert row["class_support"] == [2, 3]
    assert row["family_group"] == "mask_repair_local"
    assert row["conflicts_with_families"] == ["whole_mask_replacement"]
    assert row["conflicts_with_atoms"] == ["atom:global_crf"]
    assert row["byte_closed_archive_manifest_attached"] is True
    assert row["archive_ready_for_stack_review"] is True
    assert row["dispatchable"] is False


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
    assert ledger["requested_dispatchable_refused_count"] == 1
    assert ledger["conflict_family_counts"] == {"whole_mask_replacement": 1}


def test_byte_closed_manifest_allows_stack_review_but_not_dispatch() -> None:
    row = expected_atom_score_delta(
        {
            "atom_id": "byte_closed_local_candidate",
            "family": "hnerv_decoder_rate_recode",
            "byte_delta": -151,
            "confidence": 1.0,
            "evidence_grade": "empirical_byte_raw_equal",
            "raw_equal": True,
            "archive_manifest_path": "candidate/manifest.json",
            "archive_manifest_sha256": "b" * 64,
        },
        base_pose_dist=0.01,
    )

    assert row["rankable"] is True
    assert row["archive_ready_for_stack_review"] is True
    assert row["dispatchable"] is False
    assert "requires_exact_cuda_auth_eval" in row["dispatch_blockers"]


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
