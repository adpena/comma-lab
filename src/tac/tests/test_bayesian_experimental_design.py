# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from tac.optimization.bayesian_experimental_design import (
    contest_score,
    expected_improvement_minimize,
    family_uncertainty_reduction,
    gaussian_information_gain_nats,
    gaussian_posterior_variance_after_family_observation,
    rank_exact_eval_candidates,
)
from tac.repo_io import sha256_file

REPO = Path(__file__).resolve().parents[3]


def test_contest_score_formula_is_deterministic() -> None:
    score = contest_score(seg_dist=0.001, pose_dist=0.0004, archive_bytes=150_000)
    expected = 100 * 0.001 + math.sqrt(10 * 0.0004) + 25 * 150_000 / 37_545_489
    assert score == pytest.approx(expected)
    with pytest.raises(ValueError, match="pose_dist"):
        contest_score(seg_dist=0.0, pose_dist=-1e-6, archive_bytes=1)


def test_expected_improvement_minimize_matches_closed_form() -> None:
    ei = expected_improvement_minimize(predicted_mean=0.21, predicted_variance=0.0004, incumbent_score=0.20)
    sigma = 0.02
    z = (0.20 - 0.21) / sigma
    expected = (0.20 - 0.21) * 0.5 * (1 + math.erf(z / math.sqrt(2))) + sigma * math.exp(
        -0.5 * z * z
    ) / math.sqrt(2 * math.pi)
    assert ei == pytest.approx(expected)
    assert expected_improvement_minimize(0.19, 0.0, 0.20) == pytest.approx(0.01)
    assert expected_improvement_minimize(0.21, 0.0, 0.20) == 0.0


def test_family_uncertainty_reduction_includes_related_family_information_gain() -> None:
    beliefs = {
        "mask_recode": {"prior_score_variance": 0.04, "observation_noise_variance": 0.01},
        "mask_topology": {"prior_score_variance": 0.09, "observation_noise_variance": 0.02},
    }

    reduction = family_uncertainty_reduction(
        source_family_id="mask_recode",
        family_beliefs=beliefs,
        observation_noise_variance=0.01,
        family_couplings={"mask_topology": 0.5},
    )

    source_posterior = gaussian_posterior_variance_after_family_observation(
        target_prior_variance=0.04,
        source_prior_variance=0.04,
        observation_noise_variance=0.01,
        coupling=1.0,
    )
    related_posterior = gaussian_posterior_variance_after_family_observation(
        target_prior_variance=0.09,
        source_prior_variance=0.04,
        observation_noise_variance=0.01,
        coupling=0.5,
    )
    expected_ig = gaussian_information_gain_nats(0.04, source_posterior) + gaussian_information_gain_nats(
        0.09, related_posterior
    )
    assert source_posterior == pytest.approx(0.008)
    assert related_posterior == pytest.approx(0.072)
    assert reduction["total_information_gain_nats"] == pytest.approx(expected_ig)
    assert reduction["total_variance_reduction"] == pytest.approx((0.04 - 0.008) + (0.09 - 0.072))


def test_ranker_combines_ei_and_eig_without_dispatch_or_score_claims() -> None:
    report = rank_exact_eval_candidates(
        [
            {
                "candidate_id": "safe_low_mean",
                "family": "hnerv_recode",
                "predicted_score_mean": 0.19,
                "predicted_score_variance": 0.0001,
            },
            {
                "candidate_id": "uncertain_family_probe",
                "family": "mask_topology",
                "predicted_score_mean": 0.23,
                "predicted_score_variance": 0.05,
                "family_couplings": {"hnerv_recode": 0.25},
            },
        ],
        incumbent_score=0.20,
        family_beliefs={
            "hnerv_recode": {"prior_score_variance": 0.0001, "observation_noise_variance": 0.0001},
            "mask_topology": {"prior_score_variance": 0.05, "observation_noise_variance": 0.001},
        },
        information_gain_weight=0.01,
        source="fixture",
    )

    assert report["score_claim"] is False
    assert report["dispatch_attempted"] is False
    assert report["gpu_launched"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert "no_candidate_with_verified_exact_archive_custody" in report["dispatch_blockers"]
    assert report["rows"][0]["candidate_id"] == "uncertain_family_probe"
    assert report["rows"][0]["expected_information_gain_nats"] > report["rows"][1]["expected_information_gain_nats"]
    assert all(row["score_claim"] is False for row in report["rows"])
    assert all(row["gpu_launched"] is False for row in report["rows"])
    assert all(row["ready_for_exact_eval_dispatch"] is False for row in report["rows"])
    assert all("missing_exact_archive_custody" in row["dispatch_blockers"] for row in report["rows"])


def test_exact_archive_custody_enables_readiness_but_still_does_not_launch(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"deterministic candidate bytes")
    report = rank_exact_eval_candidates(
        [
            {
                "candidate_id": "byte_closed_candidate",
                "family": "hnerv_recode",
                "predicted_score_mean": 0.19,
                "predicted_score_variance": 0.0001,
                "exact_archive_custody": {
                    "archive_path": archive.as_posix(),
                    "archive_sha256": sha256_file(archive),
                    "archive_size_bytes": archive.stat().st_size,
                },
            }
        ],
        incumbent_score=0.20,
        family_beliefs={"hnerv_recode": {"prior_score_variance": 0.0001, "observation_noise_variance": 0.0001}},
    )

    row = report["rows"][0]
    assert report["ready_for_exact_eval_dispatch"] is True
    assert report["dispatch_attempted"] is False
    assert report["gpu_launched"] is False
    assert row["ready_for_exact_eval_dispatch"] is True
    assert row["exact_archive_custody"]["verified"] is True
    assert row["dispatch_blockers"] == []
    assert row["score_claim"] is False
    assert row["operator_note"] == "rank_only_no_gpu_launch"


def test_bad_exact_archive_custody_stays_blocked(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"actual bytes")
    report = rank_exact_eval_candidates(
        [
            {
                "candidate_id": "stale_sha",
                "family": "hnerv_recode",
                "predicted_score_mean": 0.19,
                "predicted_score_variance": 0.0001,
                "exact_archive_custody": {
                    "archive_path": archive.as_posix(),
                    "archive_sha256": "0" * 64,
                    "archive_size_bytes": archive.stat().st_size,
                },
            }
        ],
        incumbent_score=0.20,
    )

    row = report["rows"][0]
    assert row["ready_for_exact_eval_dispatch"] is False
    assert "exact_archive_sha256_mismatch" in row["dispatch_blockers"]
    assert report["ready_for_exact_eval_dispatch"] is False


def test_component_inputs_can_define_predicted_score_mean() -> None:
    report = rank_exact_eval_candidates(
        [
            {
                "candidate_id": "component_candidate",
                "family": "component_model",
                "predicted_seg_dist_mean": 0.001,
                "predicted_pose_dist_mean": 0.0004,
                "predicted_archive_bytes_mean": 150_000,
                "predicted_score_variance": 0.001,
            }
        ],
        incumbent_score=0.20,
    )

    assert report["rows"][0]["predicted_score_mean"] == pytest.approx(
        contest_score(0.001, 0.0004, 150_000)
    )


def test_rank_exact_eval_information_gain_cli_writes_deterministic_json(tmp_path: Path) -> None:
    payload = {
        "source": "fixture",
        "incumbent_score": 0.20,
        "family_beliefs": {
            "alpha": {"prior_score_variance": 0.0001, "observation_noise_variance": 0.0001}
        },
        "candidates": [
            {
                "candidate_id": "alpha_candidate",
                "family": "alpha",
                "predicted_score_mean": 0.19,
                "predicted_score_variance": 0.0001,
            }
        ],
    }
    source = tmp_path / "candidates.json"
    output = tmp_path / "report.json"
    source.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    first = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "rank_exact_eval_information_gain.py"),
            "--input",
            str(source),
            "--output",
            str(output),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    )
    first_text = output.read_text(encoding="utf-8")
    second = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "rank_exact_eval_information_gain.py"),
            "--input",
            str(source),
            "--output",
            str(output),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    )

    assert first.stdout == ""
    assert second.stdout == ""
    assert output.read_text(encoding="utf-8") == first_text
    report = json.loads(first_text)
    assert report["rows"][0]["candidate_id"] == "alpha_candidate"
    assert report["ready_for_exact_eval_dispatch"] is False
