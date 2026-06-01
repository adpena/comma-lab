# SPDX-License-Identifier: MIT
"""Tests for the joint P18/P19 recon-pixel-weight surface producer."""

from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.recon_pixel_weight_surface import (
    JOINT_RECON_PIXEL_WEIGHT_SCHEMA,
    JointReconPixelWeightConfig,
    build_joint_p18_p19_recon_pixel_weight,
    write_joint_p18_p19_recon_pixel_weight_artifact,
)

try:
    import mlx.core as _mx  # noqa: F401

    _MLX = True
except ImportError:
    _MLX = False

mlx_only = pytest.mark.skipif(not _MLX, reason="MLX required")


class _FakeMlxScorer:
    def segnet(self, x):
        import mlx.core as mx

        red = x[..., 0:1] / 255.0
        green = x[..., 1:2] / 255.0
        zeros = mx.zeros_like(red)
        return mx.concatenate([red, green, -red, -green, zeros], axis=-1)

    def posenet(self, x):
        import mlx.core as mx

        base = mx.mean(x, axis=(1, 2, 3)).reshape((-1, 1)) / 255.0
        return {
            "pose": mx.concatenate(
                [base * float(i + 1) for i in range(6)],
                axis=1,
            )
        }


@mlx_only
def test_joint_recon_pixel_weight_surface_is_pair_frame_map() -> None:
    rng = np.random.default_rng(42)
    target0 = rng.random((2, 8, 8, 3), dtype=np.float32)
    target1 = rng.random((2, 8, 8, 3), dtype=np.float32)

    weight, metadata = build_joint_p18_p19_recon_pixel_weight(
        target0,
        target1,
        mlx_scorer=_FakeMlxScorer(),
        config=JointReconPixelWeightConfig(
            num_pairs=2,
            pair_chunk_size=1,
            scorer_hw=(8, 8),
            d_pose_operating_point=3.4e-5,
            seg_margin_delta=1.0,
            weight_floor_fraction=0.05,
            normalize="mean",
        ),
    )

    assert weight.shape == (2, 2, 8, 8, 1)
    assert weight.dtype == np.float32
    assert np.all(np.isfinite(weight))
    assert float(np.min(weight)) > 0.0
    assert float(np.mean(weight)) == pytest.approx(1.0, rel=1e-5)
    assert metadata["schema"] == JOINT_RECON_PIXEL_WEIGHT_SCHEMA
    assert metadata["scorer_terms"] == {
        "p18_segnet": "mlx_segnet_top2_margin_vjp_on_last_frame",
        "p19_posenet": "mlx_posenet_per_axis_jacobian_norm_on_pair",
    }
    assert metadata["score_claim"] is False
    assert metadata["ready_for_exact_eval_dispatch"] is False
    assert metadata["weight_stats"]["shape"] == [2, 2, 8, 8, 1]
    assert metadata["seg_saliency_stats"]["max"] > 0.0
    assert metadata["pose_saliency_stats"]["max"] > 0.0


def test_joint_recon_pixel_weight_config_refuses_bad_pose_variance() -> None:
    with pytest.raises(ValueError, match="pose_inverse_variance"):
        JointReconPixelWeightConfig(
            num_pairs=2,
            pose_axis_count=6,
            pose_inverse_variance=(1.0, 1.0),
        )


def test_write_joint_recon_pixel_weight_artifact_preserves_blockers(
    tmp_path, monkeypatch
) -> None:
    import json

    import tac.optimization.recon_pixel_weight_surface as surface

    weight = np.ones((1, 2, 4, 5, 1), dtype=np.float32)

    def fake_build_from_video(**_kwargs):
        return weight, {
            "schema": JOINT_RECON_PIXEL_WEIGHT_SCHEMA,
            "blockers": ["nonfinite_gradient_sanitized:pose_axis_0_grad_pairs_0_1"],
            "training_consumption_recommended": False,
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        }

    monkeypatch.setattr(
        surface,
        "build_joint_p18_p19_recon_pixel_weight_from_video",
        fake_build_from_video,
    )

    manifest = write_joint_p18_p19_recon_pixel_weight_artifact(
        output_dir=tmp_path,
        source_video_path="upstream/videos/0.mkv",
        upstream_dir="upstream",
        config=JointReconPixelWeightConfig(num_pairs=1),
    )

    manifest_path = tmp_path / "joint_p18_p19_recon_pixel_weight_manifest.json"
    loaded = json.loads(manifest_path.read_text())
    assert manifest["manifest_path"] == manifest_path.as_posix()
    assert loaded["manifest_path"] == manifest_path.as_posix()
    assert loaded["metadata"]["training_consumption_recommended"] is False
    assert loaded["metadata"]["blockers"] == [
        "nonfinite_gradient_sanitized:pose_axis_0_grad_pairs_0_1"
    ]
    assert loaded["score_claim"] is False
    assert loaded["ready_for_exact_eval_dispatch"] is False
