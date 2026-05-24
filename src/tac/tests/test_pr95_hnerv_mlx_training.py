# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")

from tac.differentiable_eval_roundtrip import (  # noqa: E402
    apply_eval_roundtrip_during_training,
    differentiable_rgb_to_yuv6,
)
from tac.local_acceleration.pr95_hnerv_mlx import FALSE_AUTHORITY  # noqa: E402
from tac.local_acceleration.pr95_hnerv_mlx_training import (  # noqa: E402
    PR95_PREPROCESS_SMOKE_NOT_SOURCE_VIDEO_TRAINING_BLOCKER,
    PR95_SOURCE_VIDEO_TARGETS_READY_SCORER_LOSS_UNWIRED_BLOCKER,
    apply_eval_roundtrip_nhwc,
    bicubic_resize_to_camera_nhwc,
    bilinear_eval_roundtrip_downsample_nhwc,
    pr95_mlx_preprocess_grad_probe,
    pr95_pair_frame_indices,
    pr95_source_pairs_to_scorer_targets_mlx,
    rgb_to_yuv6_mlx,
    run_pr95_mlx_source_faithful_smoke,
    run_pr95_mlx_source_video_preprocess_smoke,
)


def _assert_false_authority(payload: dict) -> None:
    for key in FALSE_AUTHORITY:
        assert payload[key] is False


def test_mlx_rgb_to_yuv6_matches_torch_differentiable_oracle() -> None:
    rng = np.random.default_rng(100)
    nhwc = rng.uniform(0.0, 255.0, size=(2, 18, 22, 3)).astype(np.float32)

    mlx_yuv = rgb_to_yuv6_mlx(mx.array(nhwc))
    mx.eval(mlx_yuv)
    torch_yuv = differentiable_rgb_to_yuv6(
        torch.tensor(nhwc).permute(0, 3, 1, 2)
    ).permute(0, 2, 3, 1)

    np.testing.assert_allclose(np.asarray(mlx_yuv), torch_yuv.numpy(), atol=1e-5)


def test_mlx_rgb_to_yuv6_matches_torch_on_edges_and_odd_crop() -> None:
    nhwc = np.zeros((1, 17, 19, 3), dtype=np.float32)
    nhwc[:, :, :, 0] = 255.0
    nhwc[:, 0::2, :, 1] = 128.0
    nhwc[:, :, 0::3, 2] = 64.0
    nhwc[:, -1, -1, :] = np.array([17.0, 29.0, 251.0], dtype=np.float32)

    mlx_yuv = rgb_to_yuv6_mlx(mx.array(nhwc))
    mx.eval(mlx_yuv)
    torch_yuv = differentiable_rgb_to_yuv6(
        torch.tensor(nhwc).permute(0, 3, 1, 2)
    ).permute(0, 2, 3, 1)

    assert list(mlx_yuv.shape) == [1, 8, 9, 6]
    np.testing.assert_allclose(np.asarray(mlx_yuv), torch_yuv.numpy(), atol=1e-5)


def test_mlx_bicubic_resize_matches_torch_align_corners_false() -> None:
    rng = np.random.default_rng(101)
    nhwc = rng.uniform(0.0, 255.0, size=(2, 8, 10, 3)).astype(np.float32)

    mlx_resized = bicubic_resize_to_camera_nhwc(mx.array(nhwc), camera_hw=(11, 13))
    mx.eval(mlx_resized)
    torch_resized = F.interpolate(
        torch.tensor(nhwc).permute(0, 3, 1, 2),
        size=(11, 13),
        mode="bicubic",
        align_corners=False,
    ).permute(0, 2, 3, 1)

    np.testing.assert_allclose(np.asarray(mlx_resized), torch_resized.numpy(), atol=7e-4)


def test_mlx_bilinear_downsample_matches_torch_align_corners_false() -> None:
    rng = np.random.default_rng(102)
    nhwc = rng.uniform(0.0, 255.0, size=(2, 11, 13, 3)).astype(np.float32)

    mlx_resized = bilinear_eval_roundtrip_downsample_nhwc(
        mx.array(nhwc),
        output_hw=(8, 10),
    )
    mx.eval(mlx_resized)
    torch_resized = F.interpolate(
        torch.tensor(nhwc).permute(0, 3, 1, 2),
        size=(8, 10),
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1)

    np.testing.assert_allclose(np.asarray(mlx_resized), torch_resized.numpy(), atol=4e-5)


def test_mlx_eval_roundtrip_matches_torch_pr95_oracle() -> None:
    rng = np.random.default_rng(103)
    nhwc = rng.uniform(0.0, 255.0, size=(2, 8, 10, 3)).astype(np.float32)

    mlx_roundtrip = apply_eval_roundtrip_nhwc(mx.array(nhwc), camera_hw=(11, 13))
    mx.eval(mlx_roundtrip)
    torch_roundtrip = apply_eval_roundtrip_during_training(
        torch.tensor(nhwc).permute(0, 3, 1, 2),
        target_h=11,
        target_w=13,
    ).permute(0, 2, 3, 1)

    np.testing.assert_allclose(np.asarray(mlx_roundtrip), torch_roundtrip.numpy(), atol=0.0)


def test_mlx_eval_roundtrip_actual_resolution_matches_torch_float_oracle() -> None:
    height, width = 384, 512
    yy, xx = np.mgrid[0:height, 0:width]
    nhwc = np.stack(
        [
            (xx * 3 + yy * 5) % 251,
            (xx * 7 + 17) % 241,
            (yy * 11 + xx * 2 + 31) % 239,
        ],
        axis=-1,
    ).astype(np.float32)[None, ...]

    mlx_roundtrip = apply_eval_roundtrip_nhwc(
        mx.array(nhwc),
        camera_hw=(874, 1164),
        simulate_uint8=False,
    )
    mx.eval(mlx_roundtrip)
    torch_roundtrip = apply_eval_roundtrip_during_training(
        torch.tensor(nhwc).permute(0, 3, 1, 2),
        target_h=874,
        target_w=1164,
        simulate_uint8=False,
    ).permute(0, 2, 3, 1)

    np.testing.assert_allclose(
        np.asarray(mlx_roundtrip),
        torch_roundtrip.numpy(),
        atol=3e-3,
    )


def test_pr95_source_pair_indices_are_ordered_and_deduped() -> None:
    assert pr95_pair_frame_indices([2, 0, 2]) == [4, 5, 0, 1]

    with pytest.raises(Exception, match="non-negative"):
        pr95_pair_frame_indices([-1])


def test_mlx_source_pairs_to_scorer_targets_matches_torch_oracle() -> None:
    rng = np.random.default_rng(106)
    pairs = rng.uniform(0.0, 255.0, size=(2, 2, 11, 13, 3)).astype(np.float32)

    scorer_rgb, yuv6 = pr95_source_pairs_to_scorer_targets_mlx(
        mx.array(pairs),
        output_hw=(8, 10),
    )
    mx.eval(scorer_rgb, yuv6)
    flat = torch.tensor(pairs).reshape(-1, 11, 13, 3).permute(0, 3, 1, 2)
    torch_rgb = F.interpolate(
        flat,
        size=(8, 10),
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1).reshape(2, 2, 8, 10, 3)
    torch_yuv6 = differentiable_rgb_to_yuv6(
        torch_rgb.reshape(-1, 8, 10, 3).permute(0, 3, 1, 2)
    ).permute(0, 2, 3, 1).reshape(2, 2, 4, 5, 6)

    np.testing.assert_allclose(np.asarray(scorer_rgb), torch_rgb.numpy(), atol=4e-5)
    np.testing.assert_allclose(np.asarray(yuv6), torch_yuv6.numpy(), atol=4e-5)


def test_source_video_preprocess_smoke_uses_injected_pair_reader() -> None:
    seen: list[list[int]] = []

    def frame_reader(frame_indices: list[int]) -> np.ndarray:
        seen.append(list(frame_indices))
        yy, xx = np.mgrid[0:11, 0:13]
        frames = []
        for frame_index in frame_indices:
            frames.append(
                np.stack(
                    [
                        (xx * 3 + yy * 5 + frame_index) % 251,
                        (xx * 7 + frame_index * 11) % 241,
                        (yy * 13 + xx + frame_index * 17) % 239,
                    ],
                    axis=-1,
                )
            )
        return np.stack(frames, axis=0).astype(np.float32)

    manifest = run_pr95_mlx_source_video_preprocess_smoke(
        video_path="fixture/0.mkv",
        upstream_dir="fixture/upstream",
        pair_indices=[2, 0],
        output_hw=(8, 10),
        gradient_probe_shape=(1, 2, 8, 10, 3),
        frame_reader=frame_reader,
    )

    assert seen == [[4, 5, 0, 1]]
    assert manifest["schema"] == "pr95_hnerv_mlx_source_video_preprocess_smoke_v1"
    assert manifest["source_video_loader_ready"] is True
    assert manifest["source_video_preprocess_ready"] is True
    assert manifest["frame_reader_kind"] == "injected"
    assert manifest["video_sha256"] is None
    assert manifest["source_frame_pair_shape"] == [2, 2, 11, 13, 3]
    assert manifest["scorer_rgb_shape"] == [2, 2, 8, 10, 3]
    assert manifest["yuv6_output_shape"] == [2, 2, 4, 5, 6]
    assert manifest["gradient_probe"]["gradient_reachable"] is True
    assert manifest["exact_readiness_refusal"]["ready"] is False
    assert PR95_SOURCE_VIDEO_TARGETS_READY_SCORER_LOSS_UNWIRED_BLOCKER in (
        manifest["exact_readiness_refusal"]["blockers"]
    )
    _assert_false_authority(manifest)


def test_mlx_preprocess_gradient_reaches_rgb_input() -> None:
    probe = pr95_mlx_preprocess_grad_probe(
        input_shape=(1, 2, 8, 10, 3),
        camera_hw=(11, 13),
        seed=104,
    )

    assert probe["schema"] == "pr95_hnerv_mlx_preprocess_gradient_probe_v1"
    assert probe["gradient_reachable"] is True
    assert probe["max_abs_gradient"] > 0.0
    assert probe["nonzero_gradient_count"] > 0
    _assert_false_authority(probe)


def test_source_faithful_preprocess_smoke_is_false_authority() -> None:
    manifest = run_pr95_mlx_source_faithful_smoke(
        input_shape=(1, 2, 8, 10, 3),
        camera_hw=(11, 13),
        seed=105,
        gradient_probe_shape=(1, 2, 8, 10, 3),
    )

    assert manifest["schema"] == "pr95_hnerv_mlx_source_faithful_preprocess_smoke_v1"
    assert manifest["source_faithful_preprocess_ready"] is True
    assert manifest["roundtrip_output_shape"] == [1, 2, 8, 10, 3]
    assert manifest["yuv6_output_shape"] == [1, 2, 4, 5, 6]
    assert manifest["gradient_probe"]["gradient_reachable"] is True
    assert manifest["exact_readiness_refusal"]["ready"] is False
    assert PR95_PREPROCESS_SMOKE_NOT_SOURCE_VIDEO_TRAINING_BLOCKER in (
        manifest["exact_readiness_refusal"]["blockers"]
    )
    assert "requires_exact_cpu_cuda_auth_eval_before_score_claim" in (
        manifest["exact_readiness_refusal"]["blockers"]
    )
    _assert_false_authority(manifest)


def test_source_faithful_preprocess_smoke_requires_gradient_reachability() -> None:
    manifest = run_pr95_mlx_source_faithful_smoke(
        input_shape=(1, 2, 8, 10, 3),
        camera_hw=(11, 13),
        seed=106,
        include_gradient_probe=False,
    )

    assert manifest["source_faithful_preprocess_ready"] is False
    assert "pr95_mlx_preprocess_gradient_not_reachable" in (
        manifest["exact_readiness_refusal"]["blockers"]
    )
    _assert_false_authority(manifest)
