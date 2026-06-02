# SPDX-License-Identifier: MIT
"""Tests for direct MLX renderer prefilter profiles."""

from __future__ import annotations

import pytest


def test_renderer_prefilter_loaded_builds_false_authority_profile() -> None:
    mx = pytest.importorskip("mlx.core")

    from tac.local_acceleration.mlx_renderer_prefilter_profile import (
        build_mlx_renderer_prefilter_profile_loaded,
    )

    class FakeModel:
        def __call__(self, idx):
            b = int(idx.shape[0])
            frame = mx.zeros((b, 2, 3, 384, 512), dtype=mx.float32)
            return frame

    class FakeBundle:
        model = FakeModel()
        forward_convention = "call_b2chw_255"
        num_pairs = 2
        target_rgb_0 = mx.zeros((2, 384, 512, 3), dtype=mx.float32)
        target_rgb_1 = mx.zeros((2, 384, 512, 3), dtype=mx.float32)

    class FakeAdapter:
        def __call__(self, posenet_yuv6_pair_nhwc, segnet_last_rgb_nhwc):
            b = int(posenet_yuv6_pair_nhwc.shape[0])
            return {
                "posenet": {"pose": mx.zeros((b, 12), dtype=mx.float32)},
                "segnet": mx.zeros((b, 384, 512, 5), dtype=mx.float32),
            }

    profile = build_mlx_renderer_prefilter_profile_loaded(
        bundle=FakeBundle(),
        adapter=FakeAdapter(),
        archive_bytes=1000,
        archive_sha256="0" * 64,
        scorer_batch_pairs=1,
        required_pairs=600,
        run_id="unit",
    )

    assert profile["schema"] == "hprc_mlx_component_neutralization_profile.v1"
    assert profile["score_claim"] is False
    assert profile["promotion_eligible"] is False
    assert profile["ready_for_exact_eval_dispatch"] is False
    assert profile["num_pairs"] == 2
    assert profile["score_components"]["avg_posenet_dist"] == 0.0
    assert profile["score_components"]["avg_segnet_dist"] == 0.0
    assert profile["scope_status"]["full_video"] == "sampled_prefix_requires_full_video_rerun"
    assert "partial_coverage_mlx_replay_not_score_authority" in profile["blockers"]
    assert set(profile["component_output_hashes"]) == {
        "candidate_pose",
        "reference_pose",
        "candidate_seg",
        "reference_seg",
    }
