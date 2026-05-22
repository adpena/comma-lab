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
    build_eureka_signal_from_local_json_file,
    fit_drift_calibration,
    load_calibration_json,
    paired_anchor_from_json_files,
)


def _anchor(archive: str, delta: float) -> PairedDriftAnchor:
    contest_segnet = 0.00055979
    return PairedDriftAnchor(
        archive_sha256=archive,
        local_score=0.192040 + delta,
        contest_score=0.192040,
        local_path=f"local/{archive}.json",
        contest_path=f"contest/{archive}.json",
        local_segnet_dist=contest_segnet + delta / 100.0,
        contest_segnet_dist=contest_segnet,
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
    assert calibration.rejected_anchor_count == 0
    assert calibration.projected_contest_score(0.192040) == pytest.approx(0.192029)
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert "local_cpu_contest_drift_calibration_is_not_score_authority" in payload[
        "dispatch_blockers"
    ]


def test_empty_calibration_serializes_json_and_stays_blocked() -> None:
    calibration = fit_drift_calibration([])
    payload = calibration.to_dict()

    assert calibration.confidence == "empty"
    assert calibration.anchor_count == 0
    assert calibration.guard_band == pytest.approx(1.0)
    assert payload["score_claim"] is False
    assert json.dumps(payload, allow_nan=False)

    signal = build_eureka_signal(
        candidate_id="candidate",
        local_score=0.191,
        auth_frontier_score=0.192,
        calibration=calibration,
    )
    assert signal["eureka_trigger"] is False
    assert signal["recommended_action"] == "observe_only"


def test_fit_wide_or_mixed_for_out_of_class_span() -> None:
    calibration = fit_drift_calibration(
        [
            _anchor("a" * 64, 0.000010),
            _anchor("b" * 64, 0.0000145),
        ]
    )

    assert calibration.confidence == "wide_or_mixed"
    assert calibration.rejected_anchor_count == 0
    assert calibration.guard_band > 0.000003


def test_fit_rejects_out_of_class_span_before_calibrating() -> None:
    calibration = fit_drift_calibration(
        [
            _anchor("a" * 64, 0.000010),
            _anchor("b" * 64, 0.000288),
        ]
    )

    assert calibration.anchor_count == 1
    assert calibration.rejected_anchor_count == 1
    assert calibration.confidence == "single_anchor"
    assert "score_delta_outside_dqs1_fec6_segnet_rounding_band" in calibration.rejected_anchors[
        0
    ]["assessment"]["blockers"]


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


def test_eureka_signal_blocks_candidate_trust_region_mismatch() -> None:
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
        candidate_trust_region="mps_or_other_class",
    )

    assert signal["eureka_trigger"] is False
    assert signal["recommended_action"] == "observe_only"
    assert "candidate_trust_region_mismatch" in signal["candidate_trust_region_blockers"]


def test_eureka_from_calibration_and_local_json_file(tmp_path: Path) -> None:
    calibration = fit_drift_calibration(
        [
            _anchor("a" * 64, 0.000010),
            _anchor("b" * 64, 0.000011),
            _anchor("c" * 64, 0.000012),
        ]
    )
    calibration_json = tmp_path / "calibration.json"
    calibration_json.write_text(json.dumps(calibration.to_dict()))
    candidate_json = tmp_path / "candidate.json"
    candidate_json.write_text(
        json.dumps(
            {
                "score_recomputed_from_components": 0.192010,
                "score_axis": "cpu_advisory",
                "provenance": {"archive_sha256": "d" * 64},
            }
        )
    )

    signal = build_eureka_signal_from_local_json_file(
        candidate_id="candidate",
        local_path=candidate_json,
        auth_frontier_score=0.192028,
        calibration=load_calibration_json(calibration_json),
    )

    assert signal["eureka_trigger"] is True
    assert signal["candidate_archive_sha256"] == "d" * 64
    assert signal["candidate_trust_region_matches_calibration"] is True
    assert signal["score_claim"] is False


def test_load_calibration_json_refits_embedded_anchors(tmp_path: Path) -> None:
    calibration_json = tmp_path / "calibration.json"
    calibration = fit_drift_calibration(
        [
            _anchor("a" * 64, 0.000010),
            _anchor("b" * 64, 0.000288),
        ]
    )
    calibration_json.write_text(json.dumps(calibration.to_dict()))

    restored = load_calibration_json(calibration_json)

    assert restored.anchor_count == 1
    assert restored.confidence == "single_anchor"
    assert restored.rejected_anchor_count == 1

    unsafe = calibration.to_dict()
    unsafe["anchor_count"] = 2
    unsafe["bias_local_minus_contest"] = 0.000288
    unsafe["confidence"] = "stable_core"
    unsafe["anchors"].append(unsafe["rejected_anchors"][0]["anchor"])
    calibration_json.write_text(json.dumps(unsafe))

    loaded = load_calibration_json(calibration_json)

    assert loaded.anchor_count == 1
    assert loaded.confidence == "single_anchor"
    assert loaded.rejected_anchor_count == 1


def test_eureka_from_local_json_file_refuses_mps_axis(tmp_path: Path) -> None:
    calibration = fit_drift_calibration(
        [
            _anchor("a" * 64, 0.000010),
            _anchor("b" * 64, 0.000011),
            _anchor("c" * 64, 0.000012),
        ]
    )
    candidate_json = tmp_path / "candidate_mps.json"
    candidate_json.write_text(
        json.dumps(
            {
                "score_recomputed_from_components": 0.192010,
                "score_axis": "mps_advisory",
                "provenance": {"archive_sha256": "d" * 64},
            }
        )
    )

    signal = build_eureka_signal_from_local_json_file(
        candidate_id="candidate",
        local_path=candidate_json,
        auth_frontier_score=0.192028,
        calibration=calibration,
    )

    assert signal["eureka_trigger"] is False
    assert signal["recommended_action"] == "observe_only"
    assert "candidate_local_axis_not_macos_cpu_advisory" in signal[
        "candidate_trust_region_blockers"
    ]


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
