# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.analysis.snerv_skip_high_mode_compare import (
    build_skip_high_mode_comparison,
    render_markdown_report,
)


def _write_binary_profile(
    path: Path,
    *,
    archive_bytes: int,
    codec: str,
    stored_raw_bytes: int,
    stored_shape: list[int],
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "snerv_binary_profile.v1",
                "charged_archive_bytes": archive_bytes,
                "snar1_packet_bytes": archive_bytes - 1000,
                "section_summary": {
                    "largest_section": "decoder_payload",
                    "largest_section_bytes": archive_bytes - 2000,
                },
                "decoder_payload_header": {
                    "skip_high_storage": {
                        "schema": "snerv_official_skip_high_storage.v1",
                        "codec": codec,
                        "stored_shape": stored_shape,
                        "source_shape": [1200, 3, 192, 256],
                        "stored_raw_bytes": stored_raw_bytes,
                        "source_raw_bytes": 1_415_577_600,
                        "raw_byte_savings": 1_415_577_600 - stored_raw_bytes,
                        "receiver_expands_skip_high": True,
                        "lossless_relative_to_source_skip_high": False,
                    }
                },
                "blockers": ["snerv_binary_profile_is_rate_only_not_score_authority"],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_prefilter(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "hprc_mlx_component_neutralization_profile.v1",
                "archive_bytes": 93_620,
                "scorer_batch_pairs": 1,
                "scope_status": {"full_video": "executed"},
                "score_components": {
                    "canonical_score": 90.8,
                    "seg_term": 50.4,
                    "pose_term": 40.3,
                    "rate_term": 0.06,
                },
                "scorer_input_diagnosis": {
                    "schema": "mlx_renderer_prefilter_scorer_input_diagnosis.v1",
                    "verdict": "SCORER_INPUT_OUT_OF_DISTRIBUTION",
                    "candidate_output_out_of_distribution": True,
                    "blockers": [
                        "mlx_renderer_prefilter_scorer_input_out_of_distribution"
                    ],
                },
                "blockers": ["scorer_input_segnet_last_rgb_mean_absdiff_gt_50"],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_component_prefilter(
    path: Path,
    *,
    archive_bytes: int,
    avg_segnet_dist: float,
    avg_posenet_dist: float,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "mlx_scorer_response.v1",
                "archive_bytes": archive_bytes,
                "scorer_batch_pairs": 1,
                "scope_status": {"full_video": "executed"},
                "score_components": {
                    "avg_segnet_dist": avg_segnet_dist,
                    "avg_posenet_dist": avg_posenet_dist,
                    "seg_term": 100.0 * avg_segnet_dist,
                    "pose_term": (10.0 * avg_posenet_dist) ** 0.5,
                    "rate_term": 0.01,
                    "canonical_score": 100.0 * avg_segnet_dist
                    + (10.0 * avg_posenet_dist) ** 0.5
                    + 0.01,
                },
                "scorer_input_diagnosis": {
                    "schema": "mlx_renderer_prefilter_scorer_input_diagnosis.v1",
                    "verdict": "SCORER_INPUT_DISTRIBUTION_OK",
                    "candidate_output_out_of_distribution": False,
                    "blockers": [],
                },
                "blockers": [],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_channel_early_stop_summary(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "snerv_skip_high_channelmean_early_stop_summary.v1",
                "archive_bytes": 115_510,
                "observed_pair_count": 75,
                "required_pairs": 600,
                "cumulative_avg_segnet_dist": 0.5071230705579122,
                "cumulative_avg_posenet_dist": 144.1225729370117,
                "cumulative_canonical_score": 88.68472976244142,
                "decision": "stopped_early_uncompetitive_scalar_like_segnet_collapse",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_skip_high_comparison_finds_rate_vs_value_domain_crux(tmp_path: Path) -> None:
    scalar = tmp_path / "scalar.json"
    shared = tmp_path / "shared.json"
    prefilter = tmp_path / "prefilter.json"
    _write_binary_profile(
        scalar,
        archive_bytes=91_445,
        codec="scalar_mean_float64",
        stored_raw_bytes=8,
        stored_shape=[1, 1, 1, 1],
    )
    _write_binary_profile(
        shared,
        archive_bytes=436_084,
        codec="shared_mean_float64",
        stored_raw_bytes=1_179_648,
        stored_shape=[1, 3, 192, 256],
    )
    _write_prefilter(prefilter)

    payload = build_skip_high_mode_comparison(
        binary_profiles={"scalar": scalar, "shared": shared},
        prefilter_profiles={"scalar_epoch003199": prefilter},
        hard_byte_ceiling=178_000,
        baseline_label="scalar",
        candidate_label="shared",
    )

    assert payload["verdict"] == "NO_CURRENT_SKIP_HIGH_MODE_READY_FOR_EXACT_EVAL"
    assert payload["best_rate_row"]["label"] == "scalar"
    assert payload["best_non_scalar_skip_high_row"]["label"] == "shared"
    assert "no_skip_high_mode_with_both_byte_cap_and_non_scalar_storage" in payload[
        "blockers"
    ]
    assert "skip_high_prefilter_scorer_input_out_of_distribution" in payload[
        "blockers"
    ]
    rows = {row["label"]: row for row in payload["binary_profile_rows"]}
    assert rows["scalar"]["scalar_collapse_risk"] is True
    assert rows["shared"]["scalar_collapse_risk"] is False
    assert rows["shared"]["under_hard_byte_ceiling"] is False
    assert payload["prefilter_profile_rows"][0]["local_replay_admissible"] is False
    replacement = payload["scalar_to_non_scalar_replacement"]
    assert replacement["component_delta_status"] == "missing_non_scalar_component_profile"
    assert "non_scalar_skip_high_prefilter_profile_missing" in replacement["blockers"]
    assert replacement["byte_pressure"][
        "archive_byte_delta_candidate_minus_baseline"
    ] == 344_639
    assert replacement["scorer_component_deltas"][
        "segnet_frame1_argmax_distortion_delta"
    ] is None
    assert replacement["upstream_evaluate_geometry"]["segnet_domain"] == "last_frame_only"

    md = render_markdown_report(payload)
    assert "rate-admissible scalar skip-high" in md
    assert "SegNet frame-1 delta" in md
    assert "FUNDAMENTAL" not in md


def test_skip_high_comparison_reports_upstream_geometry_component_deltas(
    tmp_path: Path,
) -> None:
    scalar = tmp_path / "scalar.json"
    shared = tmp_path / "shared.json"
    scalar_prefilter = tmp_path / "scalar_prefilter.json"
    shared_prefilter = tmp_path / "shared_prefilter.json"
    _write_binary_profile(
        scalar,
        archive_bytes=91_445,
        codec="scalar_mean_float64",
        stored_raw_bytes=8,
        stored_shape=[1, 1, 1, 1],
    )
    _write_binary_profile(
        shared,
        archive_bytes=120_000,
        codec="shared_mean_float64",
        stored_raw_bytes=1_179_648,
        stored_shape=[1, 3, 192, 256],
    )
    _write_component_prefilter(
        scalar_prefilter,
        archive_bytes=91_445,
        avg_segnet_dist=0.50,
        avg_posenet_dist=100.0,
    )
    _write_component_prefilter(
        shared_prefilter,
        archive_bytes=120_000,
        avg_segnet_dist=0.41,
        avg_posenet_dist=64.0,
    )

    payload = build_skip_high_mode_comparison(
        binary_profiles={"scalar_mean": scalar, "shared_mean": shared},
        prefilter_profiles={
            "scalar_mean": scalar_prefilter,
            "shared_mean": shared_prefilter,
        },
        hard_byte_ceiling=178_000,
        baseline_label="scalar_mean",
        candidate_label="shared_mean",
        local_mlx_smoke_command="uv run python tools/run_compact_renderer_mlx_spine_runner.py --execute-family snerv",
    )

    replacement = payload["scalar_to_non_scalar_replacement"]
    deltas = replacement["scorer_component_deltas"]
    assert replacement["component_delta_status"] == "measured_false_authority"
    assert replacement["blockers"] == []
    assert deltas["segnet_frame1_argmax_distortion_delta"] == pytest.approx(-0.09)
    assert deltas["segnet_frame1_score_term_delta"] == pytest.approx(-9.0)
    assert deltas["posenet_two_frame_pose_distortion_delta"] == pytest.approx(-36.0)
    assert deltas["posenet_two_frame_score_term_delta"] == pytest.approx(
        -6.324555320336756
    )
    assert replacement["byte_pressure"]["candidate_under_hard_byte_ceiling"] is True
    assert replacement["byte_pressure"][
        "required_nonrate_score_drop_to_break_even"
    ] > 0.0
    assert payload["upstream_evaluate_geometry"]["posenet_domain"] == (
        "two_frame_pair_yuv6"
    )
    assert payload["runnable_local_mlx_smoke_command"].startswith("uv run python")
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False


def test_skip_high_comparison_keeps_channel_mean_as_partial_falsification_row(
    tmp_path: Path,
) -> None:
    scalar = tmp_path / "scalar.json"
    channel = tmp_path / "channel.json"
    shared = tmp_path / "shared.json"
    scalar_prefilter = tmp_path / "scalar_prefilter.json"
    channel_summary = tmp_path / "channel_early_stop.json"
    shared_prefilter = tmp_path / "shared_prefilter.json"
    _write_binary_profile(
        scalar,
        archive_bytes=91_445,
        codec="scalar_mean_float64",
        stored_raw_bytes=8,
        stored_shape=[1, 1, 1, 1],
    )
    _write_binary_profile(
        channel,
        archive_bytes=115_510,
        codec="channel_mean_float64",
        stored_raw_bytes=24,
        stored_shape=[1, 3, 1, 1],
    )
    _write_binary_profile(
        shared,
        archive_bytes=436_084,
        codec="shared_mean_float64",
        stored_raw_bytes=1_179_648,
        stored_shape=[1, 3, 192, 256],
    )
    _write_component_prefilter(
        scalar_prefilter,
        archive_bytes=91_445,
        avg_segnet_dist=0.5048246002693971,
        avg_posenet_dist=162.05871206919352,
    )
    _write_channel_early_stop_summary(channel_summary)
    _write_component_prefilter(
        shared_prefilter,
        archive_bytes=436_084,
        avg_segnet_dist=0.03815101622603834,
        avg_posenet_dist=163.49418909708658,
    )

    payload = build_skip_high_mode_comparison(
        binary_profiles={
            "scalar_mean": scalar,
            "channel_mean": channel,
            "shared_mean": shared,
        },
        prefilter_profiles={
            "scalar_mean": scalar_prefilter,
            "channel_mean": channel_summary,
            "shared_mean": shared_prefilter,
        },
        hard_byte_ceiling=178_000,
        baseline_label="scalar_mean",
        candidate_label="shared_mean",
    )

    rows = {row["label"]: row for row in payload["binary_profile_rows"]}
    assert rows["channel_mean"]["scalar_collapse_risk"] is False
    assert rows["channel_mean"]["skip_high_spatial_collapse_risk"] is True
    assert rows["shared_mean"]["skip_high_spatial_collapse_risk"] is False
    assert payload["best_non_scalar_skip_high_row"]["label"] == "channel_mean"
    assert payload["best_spatial_skip_high_row"]["label"] == "shared_mean"
    assert "no_skip_high_mode_with_both_byte_cap_and_non_scalar_storage" not in payload[
        "blockers"
    ]
    assert "no_skip_high_mode_with_byte_cap_and_spatial_storage" in payload[
        "blockers"
    ]
    assert "skip_high_prefilter_early_stopped_uncompetitive" in payload["blockers"]

    channel_row = next(
        row for row in payload["prefilter_profile_rows"] if row["label"] == "channel_mean"
    )
    assert channel_row["partial_replay"] is True
    assert channel_row["early_stop_uncompetitive"] is True
    assert channel_row["segnet_frame1_argmax_distortion"] == pytest.approx(
        0.5071230705579122
    )

    replacements = {
        row["candidate_label"]: row
        for row in payload["scalar_to_candidate_replacements"]
    }
    assert replacements["channel_mean"]["component_delta_status"] == (
        "measured_partial_false_authority"
    )
    assert "skip_high_replacement_candidate_spatial_collapse" in replacements[
        "channel_mean"
    ]["blockers"]
    assert "skip_high_replacement_component_profile_partial" in replacements[
        "channel_mean"
    ]["blockers"]
    assert replacements["channel_mean"]["scorer_component_deltas"][
        "segnet_frame1_argmax_distortion_delta"
    ] > 0.0
    assert replacements["shared_mean"]["component_delta_status"] == (
        "measured_false_authority"
    )
    assert replacements["shared_mean"]["scorer_component_deltas"][
        "segnet_frame1_argmax_distortion_delta"
    ] < -0.4

    md = render_markdown_report(payload)
    assert "spatial collapse" in md
    assert "Scalar To Candidate Portfolio" in md
