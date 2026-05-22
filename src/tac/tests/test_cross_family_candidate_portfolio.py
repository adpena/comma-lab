# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.optimization.cross_family_candidate_portfolio import (
    CrossFamilyCandidatePortfolioError,
    build_cross_family_candidate_portfolio,
)
from tac.repo_io import sha256_file

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


def _mlx_selection() -> dict[str, object]:
    return {
        "schema": "mlx_effective_spend_triage_candidate_selection.v1",
        **_false_authority(),
        "selected_rows": [
            {
                "schema": "mlx_effective_spend_triage_candidate_row.v1",
                **_false_authority(),
                "candidate_id": "mlx_scorer_response:window:501:502",
                "family": "mlx_decoder_q",
                "rank": 1,
                "observed_delta_vs_baseline_score": -0.002,
                "predicted_delta_vs_baseline_score": 0.0004,
                "calibrated_min_mlx_gap_for_spend_triage": 0.00001,
                "selection_basis": "observed_strict_gated_mlx_singleton_response_gain",
                "pair_indices": [501, 502],
                "byte_budget_margin_vs_break_even": 2500.0,
                "requires_exact_auth_eval_before_score_claim": True,
            }
        ],
    }


def _pairset_acquisition() -> dict[str, object]:
    return {
        "schema": "decoder_q_pairset_acquisition.v1",
        **_false_authority(),
        "dispatch_attempted": False,
        "candidates": [
            {
                "schema": "decoder_q_pairset_acquisition_candidate.v1",
                **_false_authority(),
                "dispatch_attempted": False,
                "acquisition_id": "pairset_diversity_k002",
                "acquisition_rank": 1,
                "selector_kind": "diversity_spaced",
                "selected_pair_count": 2,
                "selected_pair_indices": [26, 588],
                "payload_bytes": 14,
                "rate_delta": 0.00001,
                "predicted_score_mean": 0.192028948816,
                "predicted_score_source": "source_selector_inherited_non_authoritative",
            }
        ],
    }


def _hfv2_manifest(archive: Path) -> dict[str, object]:
    archive.write_bytes(b"byte closed hfv2 archive")
    return {
        "schema": "hfv1_to_hfv2_sparse_sidecar_candidate_v1",
        **_false_authority(),
        "output_submission_archive": archive.as_posix(),
        "output_archive_bytes": archive.stat().st_size,
        "output_archive_sha256": sha256_file(archive),
        "rate_delta_vs_baseline_archive": 0.0003,
        "row_parity_exact": True,
        "sparse_pair_count": 16,
        "dispatch_blockers": [
            "exact_contest_cpu_eval_missing",
            "exact_contest_cuda_eval_missing",
        ],
        "target_modes": ["contest_exact_eval"],
    }


def test_portfolio_fuses_mlx_pairset_and_outside_class_without_authority(
    tmp_path: Path,
) -> None:
    portfolio = build_cross_family_candidate_portfolio(
        incumbent_score=0.192051316881,
        mlx_selections=[_mlx_selection()],
        pairset_acquisitions=[_pairset_acquisition()],
        hfv2_manifests=[_hfv2_manifest(tmp_path / "archive.zip")],
        top_k=8,
    )

    assert portfolio["schema"] == "cross_family_candidate_portfolio.v1"
    assert portfolio["score_claim"] is False
    assert portfolio["ready_for_exact_eval_dispatch"] is False
    assert portfolio["cross_family_policy"]["outside_class_allowed"] is True
    assert portfolio["portfolio_summary"]["candidate_count_before_top_k"] == 3
    assert portfolio["portfolio_summary"]["operator_action_candidate_count"] == 3
    assert portfolio["portfolio_summary"]["recommended_next_candidate_id"] == (
        "pairset_diversity_k002"
    )
    assert portfolio["portfolio_summary"]["recommended_next_action"] == (
        "materialize_pairset_archive_and_run_local_controls"
    )
    assert portfolio["portfolio_summary"]["candidate_archive_custody_ready_count"] == 1
    assert portfolio["portfolio_summary"]["source_counts"] == {
        "decoder_q_pairset_acquisition": 1,
        "hfv2_sparse_sidecar_manifest": 1,
        "mlx_effective_spend_triage_selection": 1,
    }
    assert {row["family_id"] for row in portfolio["ranked_rows"]} == {
        "decoder_q_selective_dqs1",
        "hfv2_sparse_sidecar",
        "mlx_decoder_q",
    }
    assert all(row["score_claim"] is False for row in portfolio["ranked_rows"])
    assert all(
        row["ready_for_exact_eval_dispatch"] is False
        for row in portfolio["ranked_rows"]
    )
    assert all(
        "portfolio_planning_only_requires_separate_lane_claim"
        in row["dispatch_blockers"]
        for row in portfolio["ranked_rows"]
    )
    assert portfolio["operator_action_rows"][0]["candidate_id"] == "pairset_diversity_k002"
    assert portfolio["operator_action_rows"][0]["operator_action_rank"] == 1
    hfv2 = next(row for row in portfolio["ranked_rows"] if row["family_id"] == "hfv2_sparse_sidecar")
    assert "exact_contest_cpu_eval_missing" in hfv2["dispatch_blockers"]


def test_portfolio_preserves_custody_readiness_as_advisory_only(
    tmp_path: Path,
) -> None:
    portfolio = build_cross_family_candidate_portfolio(
        incumbent_score=0.2,
        hfv2_manifests=[_hfv2_manifest(tmp_path / "archive.zip")],
    )
    row = portfolio["ranked_rows"][0]

    assert row["exact_archive_custody"]["verified"] is True
    assert row["bayesian_ready_for_exact_eval_dispatch"] is True
    assert row["exact_archive_custody_ready"] is True
    assert row["ready_for_exact_eval_dispatch"] is False
    assert "auth_axis_gate_required_before_dispatch" in row["dispatch_blockers"]
    assert "exact_contest_cuda_eval_missing" in row["dispatch_blockers"]


def test_portfolio_rejects_authoritative_source_rows() -> None:
    selection = _mlx_selection()
    selection["selected_rows"][0]["score_claim"] = True  # type: ignore[index]
    with pytest.raises(CrossFamilyCandidatePortfolioError, match="score_claim"):
        build_cross_family_candidate_portfolio(
            incumbent_score=0.2,
            mlx_selections=[selection],
        )


def test_cross_family_portfolio_cli_writes_deterministic_outputs(
    tmp_path: Path,
) -> None:
    mlx_path = tmp_path / "mlx_selection.json"
    pairset_path = tmp_path / "pairset.json"
    hfv2_path = tmp_path / "hfv2_manifest.json"
    json_out = tmp_path / "portfolio.json"
    md_out = tmp_path / "portfolio.md"
    mlx_path.write_text(json.dumps(_mlx_selection(), sort_keys=True), encoding="utf-8")
    pairset_path.write_text(
        json.dumps(_pairset_acquisition(), sort_keys=True),
        encoding="utf-8",
    )
    hfv2_path.write_text(
        json.dumps(_hfv2_manifest(tmp_path / "archive.zip"), sort_keys=True),
        encoding="utf-8",
    )

    first = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "plan_cross_family_candidate_portfolio.py"),
            "--incumbent-score",
            "0.192051316881",
            "--mlx-selection",
            str(mlx_path),
            "--pairset-acquisition",
            str(pairset_path),
            "--hfv2-manifest",
            str(hfv2_path),
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    first_text = json_out.read_text(encoding="utf-8")
    second = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "plan_cross_family_candidate_portfolio.py"),
            "--incumbent-score",
            "0.192051316881",
            "--mlx-selection",
            str(mlx_path),
            "--pairset-acquisition",
            str(pairset_path),
            "--hfv2-manifest",
            str(hfv2_path),
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert json.loads(first.stdout)["score_claim"] is False
    assert json.loads(second.stdout)["ready_for_exact_eval_dispatch"] is False
    assert json_out.read_text(encoding="utf-8") == first_text
    assert "Cross-Family Candidate Portfolio" in md_out.read_text(encoding="utf-8")
