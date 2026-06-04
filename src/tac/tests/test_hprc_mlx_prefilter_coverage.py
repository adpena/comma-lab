# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

from tac.substrates.hprc.mlx_prefilter_coverage import (
    HPRC_MLX_COMPONENT_PROFILE_SCHEMA,
    summarize_mlx_prefilter_coverage,
)


def _write_profile(
    path: Path,
    *,
    pairs: int,
    batch_pairs: int,
    score: float,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": HPRC_MLX_COMPONENT_PROFILE_SCHEMA,
                "max_pairs": pairs,
                "num_pairs": pairs,
                "n_samples": pairs,
                "scorer_batch_pairs": batch_pairs,
                "scope_status": {"full_video": "executed"},
                "score_components": {"canonical_score": score},
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_batched_full_video_prefilter_counts_for_acquisition_not_replay(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "batched_gpu_full600.json"
    _write_profile(profile_path, pairs=600, batch_pairs=8, score=91.0)

    summary = summarize_mlx_prefilter_coverage(
        (profile_path,),
        root=tmp_path,
    )

    assert summary["has_full_video_mlx_prefilter"] is True
    assert summary["full_video_profile_paths"] == [profile_path.as_posix()]
    assert summary["local_replay_mlx_prefilter_passed"] is False
    assert summary["local_replay_profile_paths"] == []
    assert summary["best_full_video_mlx_score"] == 91.0
    assert "full_video_mlx_scorer_replay_not_attached" not in summary["blockers"]
    assert summary["blockers"] == ["mlx_profile_batch_pairs_not_singleton"]
    assert summary["profile_records"][0]["full_video_prefilter"] is True
    assert summary["profile_records"][0]["local_replay_prefilter"] is False


def test_singleton_full_video_prefilter_can_unlock_local_replay(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "singleton_full600.json"
    _write_profile(profile_path, pairs=600, batch_pairs=1, score=0.25)

    summary = summarize_mlx_prefilter_coverage(
        (profile_path,),
        root=tmp_path,
    )

    assert summary["has_full_video_mlx_prefilter"] is True
    assert summary["local_replay_mlx_prefilter_passed"] is True
    assert summary["local_replay_profile_paths"] == [profile_path.as_posix()]
    assert summary["blockers"] == []


def test_saturated_singleton_prefilter_blocks_local_replay_even_with_low_score(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "saturated_singleton_full600.json"
    _write_profile(profile_path, pairs=600, batch_pairs=1, score=0.25)
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    payload["scorer_input_diagnosis"] = {
        "schema": "mlx_renderer_prefilter_scorer_input_diagnosis.v1",
        "verdict": "SCORER_INPUT_OUT_OF_DISTRIBUTION",
        "candidate_output_likely_saturated_or_clipped": True,
        "candidate_output_out_of_distribution": True,
        "blockers": [
            "mlx_renderer_prefilter_candidate_output_saturated_or_clipped",
            "mlx_renderer_prefilter_scorer_input_out_of_distribution",
        ],
    }
    profile_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    summary = summarize_mlx_prefilter_coverage(
        (profile_path,),
        root=tmp_path,
    )

    assert summary["has_full_video_mlx_prefilter"] is True
    assert summary["best_full_video_mlx_score"] == 0.25
    assert summary["local_replay_mlx_prefilter_passed"] is False
    assert summary["local_replay_profile_paths"] == []
    assert "mlx_renderer_prefilter_candidate_output_saturated_or_clipped" in summary[
        "blockers"
    ]
    assert summary["profile_records"][0]["local_replay_prefilter"] is False


def test_cache_quality_gate_failure_blocks_local_replay_even_with_low_score(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "failed_cache_gate_singleton_full600.json"
    _write_profile(profile_path, pairs=600, batch_pairs=1, score=0.25)
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    payload["cache_quality_gate"] = {
        "schema": "mlx_cache_quality_gate.v1",
        "verdict": "FIT_OR_SCALE_FAILURE",
        "candidate_cache_nondegenerate": True,
        "fit_gate_passed": False,
        "blockers": [
            "mlx_cache_quality_gate_is_false_authority",
            "candidate_segnet_last_rgb_far_from_reference_fit_gate",
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    profile_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    summary = summarize_mlx_prefilter_coverage(
        (profile_path,),
        root=tmp_path,
    )

    assert summary["has_full_video_mlx_prefilter"] is True
    assert summary["best_full_video_mlx_score"] == 0.25
    assert summary["local_replay_mlx_prefilter_passed"] is False
    assert summary["local_replay_profile_paths"] == []
    assert "candidate_segnet_last_rgb_far_from_reference_fit_gate" in summary[
        "blockers"
    ]
    assert "mlx_prefilter_cache_quality_gate_not_passed" in summary["blockers"]
    assert "mlx_cache_quality_gate_is_false_authority" not in summary["blockers"]
    assert summary["profile_records"][0]["local_replay_prefilter"] is False


def test_hinerv_receiver_cache_quality_failure_blocks_local_replay(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "hinerv_receiver_cache_failed_full600.json"
    _write_profile(profile_path, pairs=600, batch_pairs=1, score=0.25)
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    payload["substrate_artifact_metadata"] = {
        "post_export_receiver_cache_quality": {
            "schema": "hi_nerv_receiver_cache_quality_summary.v1",
            "quality_gate_verdict": "FIT_OR_SCALE_FAILURE",
            "quality_gate_passed": False,
            "blockers": [
                "hi_nerv_receiver_cache_quality_is_false_authority",
                "candidate_segnet_last_rgb_far_from_reference_fit_gate",
            ],
        }
    }
    profile_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    summary = summarize_mlx_prefilter_coverage(
        (profile_path,),
        root=tmp_path,
    )

    assert summary["has_full_video_mlx_prefilter"] is True
    assert summary["local_replay_mlx_prefilter_passed"] is False
    assert "hi_nerv_post_export_receiver_cache_quality_gate_failed" in summary[
        "blockers"
    ]
    assert "candidate_segnet_last_rgb_far_from_reference_fit_gate" in summary[
        "blockers"
    ]
    assert "mlx_prefilter_cache_quality_verdict:FIT_OR_SCALE_FAILURE" in summary[
        "blockers"
    ]
