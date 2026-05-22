# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tac.optimization.local_cpu_contest_drift import (
    TRUST_REGION_DQS1_FEC6,
    LocalCPUContestDriftError,
    PairedDriftAnchor,
    build_eureka_signal,
    fit_drift_calibration,
    paired_anchor_from_json_files,
)


def _anchor(archive: str, delta: float) -> PairedDriftAnchor:
    return PairedDriftAnchor(
        archive_sha256=archive,
        local_score=0.192040 + delta,
        contest_score=0.192040,
        local_path=f"local/{archive}.json",
        contest_path=f"contest/{archive}.json",
        local_segnet_dist=0.00055990,
        contest_segnet_dist=0.00055979,
        local_posenet_dist=0.00002943,
        contest_posenet_dist=0.00002943,
        local_rate_unscaled=0.0047558,
        contest_rate_unscaled=0.0047558,
    )


def test_fit_stable_core_and_false_authority_payload() -> None:
    calibration = fit_drift_calibration(
        [
            _anchor("a" * 64, 0.000010),
            _anchor("b" * 64, 0.000011),
            _anchor("c" * 64, 0.000012),
        ]
    )
    payload = calibration.to_dict()

    assert calibration.confidence == "stable_core"
    assert calibration.bias_local_minus_contest == pytest.approx(0.000011)
    assert calibration.guard_band == pytest.approx(0.000003)
    assert calibration.projected_contest_score(0.192040) == pytest.approx(0.192029)
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert "local_cpu_contest_drift_calibration_is_not_score_authority" in payload[
        "dispatch_blockers"
    ]


def test_fit_wide_or_mixed_for_out_of_class_span() -> None:
    calibration = fit_drift_calibration(
        [
            _anchor("a" * 64, 0.000010),
            _anchor("b" * 64, 0.000288),
        ]
    )

    assert calibration.confidence == "wide_or_mixed"
    assert calibration.guard_band > 0.0002


def test_eureka_signal_requires_stable_margin_and_remains_proxy() -> None:
    calibration = fit_drift_calibration(
        [
            _anchor("a" * 64, 0.000010),
            _anchor("b" * 64, 0.000011),
            _anchor("c" * 64, 0.000012),
        ]
    )
    signal = build_eureka_signal(
        candidate_id="candidate",
        local_score=0.192010,
        auth_frontier_score=0.192028,
        calibration=calibration,
        min_margin=0.0,
        source_artifact="local.json",
    )

    assert signal["eureka_trigger"] is True
    assert signal["recommended_action"] == "dispatch_exact_auth_anchor"
    assert signal["score_claim"] is False
    assert signal["promotion_eligible"] is False
    assert signal["ready_for_exact_eval_dispatch"] is False


def test_paired_anchor_from_json_files_requires_same_archive(tmp_path: Path) -> None:
    local = tmp_path / "local.json"
    contest = tmp_path / "contest.json"
    local.write_text(
        json.dumps(
            {
                "score_recomputed_from_components": 0.2,
                "avg_segnet_dist": 0.1,
                "avg_posenet_dist": 0.2,
                "rate_unscaled": 0.3,
                "provenance": {"archive_sha256": "a" * 64},
            }
        )
    )
    contest.write_text(
        json.dumps(
            {
                "score_recomputed_from_components": 0.199,
                "avg_segnet_dist": 0.1,
                "avg_posenet_dist": 0.2,
                "rate_unscaled": 0.3,
                "provenance": {"archive_sha256": "a" * 64},
            }
        )
    )

    anchor = paired_anchor_from_json_files(local_path=local, contest_path=contest)
    assert anchor.archive_sha256 == "a" * 64
    assert anchor.score_delta_local_minus_contest == pytest.approx(0.001)
    assert anchor.trust_region == TRUST_REGION_DQS1_FEC6

    contest.write_text(
        json.dumps(
            {
                "score_recomputed_from_components": 0.199,
                "provenance": {"archive_sha256": "b" * 64},
            }
        )
    )
    with pytest.raises(LocalCPUContestDriftError, match="same archive"):
        paired_anchor_from_json_files(local_path=local, contest_path=contest)
