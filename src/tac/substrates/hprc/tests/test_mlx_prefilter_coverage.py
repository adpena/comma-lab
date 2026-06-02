# SPDX-License-Identifier: MIT
"""Tests for HPRC MLX prefilter coverage classification."""

from __future__ import annotations

import json
from pathlib import Path

from tac.substrates.hprc.mlx_prefilter_coverage import (
    mlx_profile_has_full_video_coverage,
    mlx_profile_pair_count,
    summarize_mlx_prefilter_coverage,
)


def test_nested_mlx_response_summary_counts_for_full_video_coverage() -> None:
    profile = {
        "schema": "hprc_mlx_component_neutralization_profile.v1",
        "scope_status": {"full_video": "executed"},
        "mlx_response_summary": {
            "batch_pairs": 1,
            "max_pairs": 600,
            "n_samples": 600,
            "candidate_cache_pairs": 600,
        },
    }

    assert mlx_profile_pair_count(profile) == 600
    assert mlx_profile_has_full_video_coverage(profile) is True


def test_renderer_prefilter_profile_schema_unlocks_full_video_coverage(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "renderer_prefilter.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema": "hprc_mlx_component_neutralization_profile.v1",
                "producer": "tac.local_acceleration.mlx_renderer_prefilter_profile",
                "scope_status": {"full_video": "executed"},
                "max_pairs": 600,
                "num_pairs": 600,
                "n_samples": 600,
                "scorer_batch_pairs": 1,
                "score_components": {"canonical_score": 0.2},
                "mlx_response_summary": {
                    "batch_pairs": 1,
                    "max_pairs": 600,
                    "n_samples": 600,
                    "local_score_estimate": 0.2,
                },
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    coverage = summarize_mlx_prefilter_coverage((profile_path,), root=tmp_path)

    assert coverage["has_full_video_mlx_prefilter"] is True
    assert coverage["local_replay_mlx_prefilter_passed"] is True
    assert coverage["best_full_video_mlx_score"] == 0.2
    assert coverage["blockers"] == []


def test_sampled_profile_path_is_not_full_video_prefilter(tmp_path: Path) -> None:
    profile_path = tmp_path / "sampled_profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema": "hprc_mlx_component_neutralization_profile.v1",
                "scope_status": {
                    "full_video": "sampled_prefix_requires_full_video_rerun"
                },
                "mlx_response_summary": {
                    "batch_pairs": 1,
                    "max_pairs": 128,
                    "n_samples": 128,
                },
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    coverage = summarize_mlx_prefilter_coverage(
        (profile_path,),
        root=tmp_path,
    )

    assert coverage["has_full_video_mlx_prefilter"] is False
    assert coverage["profile_records"][0]["pair_count"] == 128
    assert coverage["profile_records"][0]["full_video_prefilter"] is False
    assert "full_video_mlx_scorer_replay_not_attached" in coverage["blockers"]
    assert "sampled_mlx_prefilter_requires_full_video_rerun" in coverage["blockers"]


def test_batched_full_video_profile_is_not_singleton_prefilter(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "batched_profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema": "hprc_mlx_component_neutralization_profile.v1",
                "scope_status": {"full_video": "executed"},
                "mlx_response_summary": {
                    "batch_pairs": 8,
                    "max_pairs": 600,
                    "n_samples": 600,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    coverage = summarize_mlx_prefilter_coverage(
        (profile_path,),
        root=tmp_path,
    )

    assert coverage["has_full_video_mlx_prefilter"] is False
    assert coverage["profile_records"][0]["batch_pairs"] == 8
    assert "mlx_profile_batch_pairs_not_singleton" in coverage["blockers"]
