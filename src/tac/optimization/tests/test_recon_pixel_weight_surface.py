# SPDX-License-Identifier: MIT
"""Tests for the joint P18/P19 recon-pixel-weight surface producer."""

from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.recon_pixel_weight_surface import (
    JOINT_RECON_PIXEL_WEIGHT_SCHEMA,
    JointReconPixelWeightConfig,
    build_joint_p18_p19_recon_pixel_weight,
    build_joint_p18_p19_recon_pixel_weight_from_video,
    build_joint_p18_p19_recon_pixel_weight_torch,
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


class _FakeTorchSegNet:
    def eval(self):
        return self

    def to(self, _device):
        return self

    def parameters(self):
        return []

    def preprocess_input(self, x):
        return x[:, -1, ...]

    def __call__(self, x):
        import torch

        red = x[:, 0:1, :, :] / 255.0
        green = x[:, 1:2, :, :] / 255.0
        zeros = torch.zeros_like(red)
        return torch.cat([red, green, -red, -green, zeros], dim=1)


class _FakeTorchPoseNet:
    def eval(self):
        return self

    def to(self, _device):
        return self

    def parameters(self):
        return []

    def preprocess_input(self, x):
        return x

    def __call__(self, x):
        import torch

        base = torch.mean(x, dim=(1, 2, 3, 4)).reshape((-1, 1)) / 255.0
        return {"pose": torch.cat([base * float(i + 1) for i in range(6)], dim=1)}


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


def test_torch_joint_recon_pixel_weight_surface_is_finite_and_recommended() -> None:
    pytest.importorskip("torch")
    rng = np.random.default_rng(44)
    target0 = rng.random((2, 8, 8, 3), dtype=np.float32)
    target1 = rng.random((2, 8, 8, 3), dtype=np.float32)
    progress_rows = []

    weight, metadata = build_joint_p18_p19_recon_pixel_weight_torch(
        target0,
        target1,
        torch_posenet=_FakeTorchPoseNet(),
        torch_segnet=_FakeTorchSegNet(),
        config=JointReconPixelWeightConfig(
            num_pairs=2,
            pair_chunk_size=1,
            scorer_hw=(8, 8),
            d_pose_operating_point=3.4e-5,
            seg_margin_delta=1.0,
            weight_floor_fraction=0.05,
            normalize="mean",
        ),
        device="cpu",
        progress_callback=progress_rows.append,
    )

    assert weight.shape == (2, 2, 8, 8, 1)
    assert weight.dtype == np.float32
    assert np.all(np.isfinite(weight))
    assert metadata["surface_generation_backend"] == "torch_exact_cpu_scorer_vjp.v1"
    assert metadata["training_consumption_recommended"] is True
    assert metadata["blockers"] == []
    assert metadata["gradient_health"] == {
        "schema": "joint_recon_pixel_weight_gradient_health.v1",
        "surface_generation_backend": "torch_exact_cpu_scorer_vjp.v1",
        "component_count": 14,
        "components_with_nonfinite": 0,
        "total_nonfinite_values": 0,
        "sanitized_components": [],
        "status": "pass_finite",
        "consumption_recommended": True,
    }
    assert metadata["seg_saliency_stats"]["max"] > 0.0
    assert metadata["pose_saliency_stats"]["max"] > 0.0
    assert [row["pair_end"] for row in progress_rows] == [1, 2]
    assert progress_rows[-1]["pairs_complete"] == 2
    assert progress_rows[-1]["surface_generation_backend"] == (
        "torch_exact_cpu_scorer_vjp.v1"
    )


def test_auto_backend_falls_back_to_torch_with_mlx_failure_metadata(
    monkeypatch,
    tmp_path,
) -> None:
    pytest.importorskip("torch")
    import tac.local_acceleration.mlx_scorer_adapters as mlx_adapters
    import tac.scorer as scorer_mod
    import tac.substrates._shared.mlx_score_aware as score_aware_shared

    rng = np.random.default_rng(45)
    target0 = rng.random((2, 8, 8, 3), dtype=np.float32)
    target1 = rng.random((2, 8, 8, 3), dtype=np.float32)

    monkeypatch.setattr(
        score_aware_shared,
        "decode_mlx_targets",
        lambda *_args, **_kwargs: (target0, target1),
    )

    def fail_mlx(*_args, **_kwargs):
        raise RuntimeError("synthetic mlx vjp failure")

    monkeypatch.setattr(
        mlx_adapters,
        "load_mlx_distortion_scorer_adapter_from_upstream",
        fail_mlx,
    )
    monkeypatch.setattr(
        scorer_mod,
        "load_differentiable_scorers",
        lambda *_args, **_kwargs: (_FakeTorchPoseNet(), _FakeTorchSegNet()),
    )

    weight, metadata = build_joint_p18_p19_recon_pixel_weight_from_video(
        source_video_path=tmp_path / "0.mkv",
        upstream_dir=tmp_path / "upstream",
        config=JointReconPixelWeightConfig(
            num_pairs=2,
            pair_chunk_size=1,
            scorer_hw=(8, 8),
        ),
        scorer_backend="auto",
    )

    assert weight.shape == (2, 2, 8, 8, 1)
    assert metadata["surface_generation_backend"] == "torch_exact_cpu_scorer_vjp.v1"
    assert metadata["auto_backend_selection"] == {
        "schema": "joint_recon_pixel_weight_auto_backend_selection.v1",
        "requested_backend": "auto",
        "selected_backend": "torch",
        "mlx_attempt": {
            "status": "failed_exception",
            "error_type": "RuntimeError",
            "error": "synthetic mlx vjp failure",
        },
        "fallback_reason": "mlx_direct_scorer_vjp_exception",
    }


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
            "gradient_health": {
                "schema": "joint_recon_pixel_weight_gradient_health.v1",
                "surface_generation_backend": "mlx_direct_scorer_vjp.v1",
                "component_count": 1,
                "components_with_nonfinite": 1,
                "total_nonfinite_values": 40,
                "sanitized_components": ["pose_axis_0_grad_pairs_0_1"],
                "status": "fail_nonfinite_sanitized",
                "consumption_recommended": False,
            },
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
        scorer_backend="torch",
    )

    manifest_path = tmp_path / "joint_p18_p19_recon_pixel_weight_manifest.json"
    loaded = json.loads(manifest_path.read_text())
    assert manifest["manifest_path"] == manifest_path.as_posix()
    assert loaded["manifest_path"] == manifest_path.as_posix()
    assert loaded["progress_jsonl_path"] == (
        tmp_path / "joint_p18_p19_recon_pixel_weight_progress.jsonl"
    ).as_posix()
    assert loaded["metadata"]["training_consumption_recommended"] is False
    assert loaded["metadata"]["blockers"] == [
        "nonfinite_gradient_sanitized:pose_axis_0_grad_pairs_0_1"
    ]
    assert loaded["metadata"]["gradient_health"]["status"] == (
        "fail_nonfinite_sanitized"
    )
    assert loaded["scorer_backend"] == "torch"
    assert loaded["score_claim"] is False
    assert loaded["ready_for_exact_eval_dispatch"] is False
