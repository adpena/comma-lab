# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from tac.optimization.decoder_q_selective_runtime_feedback import (
    DecoderQSelectiveRuntimeFeedbackError,
    build_decoder_q_selective_runtime_feedback,
    build_sign_calibration_labels,
    load_packet_plan_for_materialization,
    write_json,
)


def _false_authority() -> dict[str, bool]:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }


def _bridge() -> dict[str, object]:
    return {
        "schema": "decoder_q_selective_window_bridge_plan.v1",
        **_false_authority(),
        "work_units": [
            {
                "pair_window": [3, 4],
                "observed_mlx_window_gain": 0.010,
                "observed_mlx_gain": 0.010,
                "normalized_full_video_gain": 0.010 / 600.0,
                "full_video_denominator": 600,
            },
            {
                "pair_window": [7, 8],
                "observed_mlx_window_gain": 0.005,
                "observed_mlx_gain": 0.005,
                "normalized_full_video_gain": 0.005 / 600.0,
                "full_video_denominator": 600,
            },
        ],
    }


def _manifest() -> dict[str, object]:
    return {
        "schema": "decoder_q_selective_runtime_materialization.v1",
        **_false_authority(),
        "materialized_archive": {
            "zip_sha256": "a" * 64,
            "zip_bytes": 178530,
        },
        "dqs1_payload": {
            "pair_indices": [3, 7],
            "affected_frame_indices": [6, 7, 14, 15],
            "payload_bytes": 15,
        },
        "packet_plan": {
            "mutation": {
                "tensor_name": "rgb_1.weight",
                "q_offset": 0,
                "delta": 1,
            }
        },
    }


def _locality() -> dict[str, object]:
    return {
        "schema": "decoder_q_selective_runtime_locality_controls.v1",
        **_false_authority(),
        "locality_controls_passed": True,
        "selected_pair_indices": [3, 7],
        "selected_frame_indices": [6, 7, 14, 15],
        "mismatch_counts": {
            "missing_raw_file_count": 0,
            "raw_size_mismatch_count": 0,
            "selected_frame_mismatch_count": 0,
            "unselected_frame_mismatch_count": 0,
        },
        "targets": {
            "selective": {
                "archive_sha256": "a" * 64,
            }
        },
        "hashes": {
            "0.raw": {
                "raw_files": {
                    "selective": "b" * 64,
                }
            }
        },
    }


def _advisory(score: float) -> dict[str, object]:
    return {
        **_false_authority(),
        "canonical_score": score,
        "score_axis": "cpu_advisory",
        "evidence_grade": "macOS-CPU advisory",
        "provenance": {
            "archive_sha256": "a" * 64,
            "inflated_output_manifest": {
                "payload": {
                    "files": [
                        {
                            "sha256": "b" * 64,
                        }
                    ]
                }
            },
        },
    }


def test_feedback_builds_fail_closed_transfer_metrics() -> None:
    feedback = build_decoder_q_selective_runtime_feedback(
        bridge_plan=_bridge(),
        materialization_manifest=_manifest(),
        locality_controls=_locality(),
        advisory_result=_advisory(0.99),
        local_baseline_score=1.0,
        min_dispatch_edge=0.02,
        contest_cpu_frontier_score=0.98,
    )

    assert feedback["score_claim"] is False
    assert feedback["decision"]["dispatch_recommended"] is False
    assert feedback["decision"]["exact_dispatch_suppression_allowed"] is False
    assert feedback["decision"]["local_spend_triage_positive"] is False
    assert feedback["advisory"]["score_delta_vs_local_baseline"] == pytest.approx(-0.01)
    assert feedback["mlx_transfer"]["observed_mlx_window_gain_sum"] == pytest.approx(0.015)
    assert feedback["mlx_transfer"]["normalized_full_video_gain_sum"] == pytest.approx(0.015 / 600.0)
    assert feedback["mlx_transfer"]["observed_mlx_gain_sum"] == pytest.approx(0.015 / 600.0)
    assert feedback["custody"]["advisory_raw_sha256"] == "b" * 64
    assert feedback["custody"]["locality_selective_raw_sha256"] == "b" * 64
    assert feedback["sign_calibration_label"]["observed_score_delta_sign"] == -1
    assert feedback["sign_calibration_label"]["atom_mutation_keys"] == [
        {"tensor_name": "rgb_1.weight", "q_offset": 0, "delta": 1}
    ]


def test_feedback_never_suppresses_exact_dispatch_from_local_advisory() -> None:
    feedback = build_decoder_q_selective_runtime_feedback(
        bridge_plan=_bridge(),
        materialization_manifest=_manifest(),
        locality_controls=_locality(),
        advisory_result=_advisory(0.97),
        local_baseline_score=1.0,
        min_dispatch_edge=0.02,
        contest_cpu_frontier_score=0.98,
    )

    assert feedback["decision"]["local_spend_triage_positive"] is True
    assert feedback["decision"]["dispatch_recommended"] is False
    assert feedback["decision"]["exact_dispatch_suppression_allowed"] is False


def test_feedback_rejects_raw_sha_mismatch() -> None:
    locality = _locality()
    locality["hashes"]["0.raw"]["raw_files"]["selective"] = "c" * 64  # type: ignore[index]

    with pytest.raises(DecoderQSelectiveRuntimeFeedbackError, match="raw SHA mismatch"):
        build_decoder_q_selective_runtime_feedback(
            bridge_plan=_bridge(),
            materialization_manifest=_manifest(),
            locality_controls=locality,
            advisory_result=_advisory(0.99),
            local_baseline_score=1.0,
            min_dispatch_edge=0.02,
        )


def test_feedback_rejects_missing_raw_sha() -> None:
    advisory = _advisory(0.99)
    del advisory["provenance"]["inflated_output_manifest"]  # type: ignore[index]

    with pytest.raises(DecoderQSelectiveRuntimeFeedbackError, match="raw SHA missing"):
        build_decoder_q_selective_runtime_feedback(
            bridge_plan=_bridge(),
            materialization_manifest=_manifest(),
            locality_controls=_locality(),
            advisory_result=advisory,
            local_baseline_score=1.0,
            min_dispatch_edge=0.02,
        )


def test_feedback_rejects_mismatched_normalized_gain() -> None:
    bridge = _bridge()
    bridge["work_units"][0]["normalized_full_video_gain"] = 0.010  # type: ignore[index]

    with pytest.raises(
        DecoderQSelectiveRuntimeFeedbackError,
        match="normalized_full_video_gain_mismatch",
    ):
        build_decoder_q_selective_runtime_feedback(
            bridge_plan=bridge,
            materialization_manifest=_manifest(),
            locality_controls=_locality(),
            advisory_result=_advisory(0.99),
            local_baseline_score=1.0,
            min_dispatch_edge=0.02,
        )


def test_feedback_rejects_pair_and_frame_mismatches() -> None:
    locality = _locality()
    locality["selected_pair_indices"] = [3]

    with pytest.raises(DecoderQSelectiveRuntimeFeedbackError, match="selected_pair"):
        build_decoder_q_selective_runtime_feedback(
            bridge_plan=_bridge(),
            materialization_manifest=_manifest(),
            locality_controls=locality,
            advisory_result=_advisory(0.99),
            local_baseline_score=1.0,
            min_dispatch_edge=0.02,
        )

    locality = _locality()
    locality["selected_frame_indices"] = [6, 7]
    with pytest.raises(DecoderQSelectiveRuntimeFeedbackError, match="selected_frame"):
        build_decoder_q_selective_runtime_feedback(
            bridge_plan=_bridge(),
            materialization_manifest=_manifest(),
            locality_controls=locality,
            advisory_result=_advisory(0.99),
            local_baseline_score=1.0,
            min_dispatch_edge=0.02,
        )


def test_feedback_rejects_exact_auth_payload_as_local_advisory() -> None:
    advisory = _advisory(0.99)
    advisory["score_axis"] = "contest_cpu"
    advisory["evidence_grade"] = "contest-CPU"

    with pytest.raises(DecoderQSelectiveRuntimeFeedbackError, match="score_axis"):
        build_decoder_q_selective_runtime_feedback(
            bridge_plan=_bridge(),
            materialization_manifest=_manifest(),
            locality_controls=_locality(),
            advisory_result=advisory,
            local_baseline_score=1.0,
            min_dispatch_edge=0.02,
        )


def test_load_packet_plan_for_materialization_finds_output_submission_copy(
    tmp_path,
) -> None:
    output_dir = tmp_path / "submission_dir"
    output_dir.mkdir()
    packet_plan = {
        "schema": "decoder_q_selective_runtime_packet_plan.v1",
        "mutation": {
            "tensor_name": "rgb_1.weight",
            "q_offset": 0,
            "delta": 1,
        },
    }
    write_json(output_dir / "decoder_q_selective_runtime_packet_plan.json", packet_plan)

    loaded = load_packet_plan_for_materialization(
        {
            "schema": "decoder_q_selective_runtime_materialization.v1",
            "output_submission_dir": str(output_dir),
        }
    )

    assert loaded == packet_plan


def test_build_sign_calibration_labels() -> None:
    feedback = build_decoder_q_selective_runtime_feedback(
        bridge_plan=_bridge(),
        materialization_manifest=_manifest(),
        locality_controls=_locality(),
        advisory_result=_advisory(1.01),
        local_baseline_score=1.0,
        min_dispatch_edge=0.02,
    )

    labels = build_sign_calibration_labels([feedback], source="unit")

    assert labels["schema"] == "decoder_q_surface_sign_calibration_labels.v1"
    assert labels["summary"]["regression_label_count"] == 1
    assert labels["labels"][0]["observed_score_delta_sign"] == 1
    assert labels["score_claim"] is False
