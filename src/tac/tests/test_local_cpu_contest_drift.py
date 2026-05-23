# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from src.tac.optimization.local_cpu_contest_drift import (
    EUREKA_FALSE_AUTHORITY_FIELDS,
    TRUST_REGION_DQS1_FEC6,
    LocalCPUContestDriftError,
    PairedDriftAnchor,
    build_eureka_signal,
    build_eureka_signal_from_local_json_file,
    eureka_false_authority_violations,
    fit_drift_calibration,
    load_calibration_json,
    paired_anchor_from_json_files,
    require_eureka_false_authority,
)
from tools import calibrate_local_cpu_contest_drift as drift_cli


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
    assert eureka_false_authority_violations(signal) == []
    require_eureka_false_authority(signal)
    for field in EUREKA_FALSE_AUTHORITY_FIELDS:
        assert signal[field] is False


def test_eureka_false_authority_requires_all_fields_exactly_false() -> None:
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
    )

    missing = dict(signal)
    missing.pop("score_claim_valid")
    assert "score_claim_valid" in eureka_false_authority_violations(missing)
    with pytest.raises(LocalCPUContestDriftError, match="score_claim_valid"):
        require_eureka_false_authority(missing)

    truthy = dict(signal)
    truthy["gpu_launched"] = True
    assert "gpu_launched" in eureka_false_authority_violations(truthy)
    with pytest.raises(LocalCPUContestDriftError, match="gpu_launched"):
        require_eureka_false_authority(truthy, context="rank007 eureka")


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
                "evidence_semantics": "non_contest_cpu_auth_eval_advisory",
                "n_samples": 600,
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


def test_eureka_from_local_json_file_blocks_partial_or_authority_payload(
    tmp_path: Path,
) -> None:
    calibration = fit_drift_calibration(
        [
            _anchor("a" * 64, 0.000010),
            _anchor("b" * 64, 0.000011),
            _anchor("c" * 64, 0.000012),
        ]
    )
    candidate_json = tmp_path / "candidate_partial.json"
    candidate_json.write_text(
        json.dumps(
            {
                "score_recomputed_from_components": 0.192010,
                "score_axis": "cpu_advisory",
                "promotion_eligible": True,
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
    assert "candidate_local_evidence_semantics_not_cpu_advisory" in signal[
        "candidate_trust_region_blockers"
    ]
    assert "candidate_local_not_full_public_sample" in signal[
        "candidate_trust_region_blockers"
    ]
    assert (
        "candidate_local_payload_truthy_authority:promotion_eligible=truthy"
        in signal["candidate_trust_region_blockers"]
    )
    assert signal["score_claim"] is False


def test_eureka_from_local_json_file_accepts_bracketed_macos_cpu_axis(
    tmp_path: Path,
) -> None:
    calibration = fit_drift_calibration(
        [
            _anchor("a" * 64, 0.000010),
            _anchor("b" * 64, 0.000011),
            _anchor("c" * 64, 0.000012),
        ]
    )
    candidate_json = tmp_path / "candidate_bracketed_axis.json"
    candidate_json.write_text(
        json.dumps(
            {
                "score_recomputed_from_components": 0.192010,
                "evidence_grade": "[macOS-CPU advisory]",
                "evidence_semantics": "non_contest_cpu_auth_eval_advisory",
                "n_samples": 600,
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

    assert signal["eureka_trigger"] is True
    assert signal["candidate_trust_region_blockers"] == []


def test_cli_auth_frontier_score_can_come_from_canonical_pointer(tmp_path: Path) -> None:
    pointer_path = tmp_path / "canonical_frontier_pointer.json"
    pointer_path.write_text(
        json.dumps(
            {
                "schema_version": "canonical_frontier_pointer_v1_20260519",
                "our_local_frontier_contest_cpu": {
                    "score": 0.19202828295713675,
                    "axis": "contest_cpu",
                    "archive_sha256": "a" * 64,
                    "lane_id": "lane",
                    "hardware_substrate": "linux_x86_64_cpu",
                    "measured_at_utc": "2026-05-22T18:14:49Z",
                    "evidence_grade": "[contest-CPU]",
                },
                "our_local_frontier_contest_cuda": None,
                "submitted_pr_number_for_current_frontier": None,
                "upstream_leaderboard_snapshot": None,
                "upstream_leaderboard_snapshot_at_utc": None,
                "last_refreshed_utc": "2026-05-22T18:14:49Z",
                "auto_update_on_dispatch_completion": True,
                "pointer_refresh_command": ".venv/bin/python tools/refresh_canonical_frontier.py",
                "refresh_provenance": {},
            }
        )
    )
    args = Namespace(
        auth_frontier_score=None,
        auth_frontier_score_from_pointer=True,
        canonical_frontier_pointer=pointer_path.as_posix(),
    )

    assert drift_cli._auth_frontier_score_from_args(args) == pytest.approx(
        0.19202828295713675
    )


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
                "evidence_semantics": "non_contest_cpu_auth_eval_advisory",
                "n_samples": 600,
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
