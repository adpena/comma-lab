# SPDX-License-Identifier: MIT
"""Tests for direct MLX renderer prefilter profiles."""

from __future__ import annotations

import json

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
        source_pair_indices = None
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
    assert profile["scorer_input_distribution"]["schema"] == (
        "mlx_renderer_prefilter_scorer_input_distribution.v1"
    )


def test_renderer_prefilter_blocks_scorer_input_distribution_collapse() -> None:
    mx = pytest.importorskip("mlx.core")

    from tac.local_acceleration.mlx_renderer_prefilter_profile import (
        build_mlx_renderer_prefilter_profile_loaded,
    )

    class SaturatedModel:
        def __call__(self, idx):
            b = int(idx.shape[0])
            frame = mx.ones((b, 2, 3, 384, 512), dtype=mx.float32) * 255.0
            return frame

    class FakeBundle:
        model = SaturatedModel()
        forward_convention = "call_b2chw_255"
        source_pair_indices = None
        num_pairs = 1
        target_rgb_0 = mx.ones((1, 384, 512, 3), dtype=mx.float32) * (20.0 / 255.0)
        target_rgb_1 = mx.ones((1, 384, 512, 3), dtype=mx.float32) * (20.0 / 255.0)

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
        archive_sha256="2" * 64,
        scorer_batch_pairs=1,
        required_pairs=1,
        run_id="unit-collapse",
    )

    dist = profile["scorer_input_distribution"]
    assert dist["segnet_last_rgb_absdiff"]["mean_abs"] == pytest.approx(235.0)
    assert "scorer_input_segnet_last_rgb_mean_absdiff_gt_50" in profile["blockers"]
    assert "scorer_input_posenet_yuv6_pair_mean_absdiff_gt_50" in profile["blockers"]
    assert "scorer_input_segnet_last_rgb_saturation_delta_gt_0_15" in profile["blockers"]


def test_renderer_prefilter_loaded_emits_progress_telemetry(tmp_path) -> None:
    mx = pytest.importorskip("mlx.core")

    from tac.local_acceleration.mlx_renderer_prefilter_profile import (
        build_mlx_renderer_prefilter_profile_loaded,
    )

    class FakeModel:
        def __call__(self, idx):
            b = int(idx.shape[0])
            return mx.zeros((b, 2, 3, 384, 512), dtype=mx.float32)

    class FakeBundle:
        model = FakeModel()
        forward_convention = "call_b2chw_255"
        source_pair_indices = None
        num_pairs = 3
        target_rgb_0 = mx.zeros((3, 384, 512, 3), dtype=mx.float32)
        target_rgb_1 = mx.zeros((3, 384, 512, 3), dtype=mx.float32)

    class FakeAdapter:
        def __call__(self, posenet_yuv6_pair_nhwc, segnet_last_rgb_nhwc):
            b = int(posenet_yuv6_pair_nhwc.shape[0])
            return {
                "posenet": {"pose": mx.zeros((b, 12), dtype=mx.float32)},
                "segnet": mx.zeros((b, 384, 512, 5), dtype=mx.float32),
            }

    progress_path = tmp_path / "local_mlx_prefilter_progress.jsonl"
    profile = build_mlx_renderer_prefilter_profile_loaded(
        bundle=FakeBundle(),
        adapter=FakeAdapter(),
        archive_bytes=1000,
        archive_sha256="1" * 64,
        scorer_batch_pairs=2,
        required_pairs=3,
        run_id="unit-progress",
        progress_jsonl_path=progress_path,
        progress_every=1,
    )

    rows = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2
    assert rows[-1]["schema"] == "mlx_renderer_prefilter_progress.v1"
    assert rows[-1]["run_id"] == "unit-progress"
    assert rows[-1]["cumulative_pair_count"] == 3
    assert rows[-1]["score_claim"] is False
    assert rows[-1]["ready_for_exact_eval_dispatch"] is False
    assert profile["progress"]["progress_jsonl_path"] == progress_path.as_posix()
    assert profile["progress"]["chunk_count"] == 2
    assert profile["progress"]["batch_pairs"] == 2
    assert profile["response_metadata"]["pair_throughput_per_second"] is not None
    assert "mlx_profile_batch_pairs_not_singleton" in profile["blockers"]
