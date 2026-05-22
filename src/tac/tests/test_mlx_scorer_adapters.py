# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from safetensors.torch import load_file

pytest.importorskip("mlx.core")

from tac.local_acceleration.mlx_scorer_adapters import (
    run_mlx_batchnorm2d_nchw,
    run_mlx_conv2d_nchw,
    run_mlx_conv2d_relu_nchw,
    run_mlx_distortion_scorer_nchw,
    run_mlx_efficientnet_block_nchw,
    run_mlx_efficientnet_features_nchw,
    run_mlx_efficientnet_stage_nchw,
    run_mlx_efficientnet_stem_nchw,
    run_mlx_fastvit_stage_nchw,
    run_mlx_fastvit_vision_nchw,
    run_mlx_linear,
    run_mlx_mobileone_block_nchw,
    run_mlx_mobileone_stem_nchw,
    run_mlx_patch_embed_nchw,
    run_mlx_posenet_nchw,
    run_mlx_repmixer_block_nchw,
    run_mlx_segmentation_head_nchw,
    run_mlx_segnet_nchw,
    run_mlx_timm_universal_encoder_nchw,
    run_mlx_unet_decoder_block_nchw,
    run_mlx_unet_decoder_nchw,
    scorer_distortion_components_numpy,
    temporary_mlx_device,
    torch_batchnorm2d_to_mlx,
    torch_conv2d_relu_to_mlx,
    torch_conv2d_to_mlx,
    torch_distortion_net_to_mlx,
    torch_efficientnet_block_to_mlx,
    torch_efficientnet_features_to_mlx,
    torch_efficientnet_stage_to_mlx,
    torch_efficientnet_stem_to_mlx,
    torch_fastvit_stage_to_mlx,
    torch_fastvit_vision_to_mlx,
    torch_linear_to_mlx,
    torch_mobileone_block_to_mlx,
    torch_mobileone_stem_to_mlx,
    torch_patch_embed_to_mlx,
    torch_posenet_to_mlx,
    torch_repmixer_block_to_mlx,
    torch_segmentation_head_to_mlx,
    torch_segnet_to_mlx,
    torch_timm_universal_encoder_to_mlx,
    torch_unet_decoder_block_to_mlx,
    torch_unet_decoder_to_mlx,
)


def _max_abs(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return float(np.max(np.abs(lhs.astype(np.float32) - rhs.astype(np.float32))))


@pytest.mark.parametrize(
    "in_channels,out_channels,groups",
    [
        (3, 4, 1),
        (4, 6, 2),
        (4, 4, 4),
    ],
)
def test_conv2d_adapter_matches_torch_nchw(
    in_channels: int,
    out_channels: int,
    groups: int,
) -> None:
    torch.manual_seed(17 + groups)
    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        padding=1,
        groups=groups,
        bias=True,
    ).eval()
    x = torch.randn(2, in_channels, 9, 11)

    expected = conv(x).detach().numpy()
    actual = run_mlx_conv2d_nchw(torch_conv2d_to_mlx(conv), x.numpy())

    assert actual.shape == expected.shape
    assert _max_abs(actual, expected) < 1.0e-5


def test_batchnorm2d_adapter_matches_torch_eval_nchw() -> None:
    torch.manual_seed(23)
    bn = nn.BatchNorm2d(5, eps=0.001, momentum=0.01).eval()
    with torch.no_grad():
        bn.weight.copy_(torch.randn(5))
        bn.bias.copy_(torch.randn(5))
        bn.running_mean.copy_(torch.randn(5))
        bn.running_var.copy_(torch.rand(5) + 0.25)
    x = torch.randn(3, 5, 7, 8)

    expected = bn(x).detach().numpy()
    actual = run_mlx_batchnorm2d_nchw(torch_batchnorm2d_to_mlx(bn), x.numpy())

    assert actual.shape == expected.shape
    assert _max_abs(actual, expected) < 1.0e-5


def test_batchnorm2d_adapter_uses_eval_affine_for_running_stats() -> None:
    bn = nn.BatchNorm2d(3).eval()
    adapter = torch_batchnorm2d_to_mlx(bn)

    assert type(adapter).__name__ == "MLXBatchNorm2dEvalAffineAdapter"


def test_linear_adapter_matches_torch() -> None:
    torch.manual_seed(29)
    linear = nn.Linear(13, 7).eval()
    x = torch.randn(4, 13)

    expected = linear(x).detach().numpy()
    with temporary_mlx_device("cpu"):
        actual = run_mlx_linear(torch_linear_to_mlx(linear), x.numpy())

    assert actual.shape == expected.shape
    assert _max_abs(actual, expected) < 1.0e-5


def test_linear_adapter_gpu_drift_is_measured_not_exact() -> None:
    torch.manual_seed(29)
    linear = nn.Linear(13, 7).eval()
    x = torch.randn(4, 13)

    expected = linear(x).detach().numpy()
    with temporary_mlx_device("gpu"):
        actual = run_mlx_linear(torch_linear_to_mlx(linear), x.numpy())

    drift = _max_abs(actual, expected)
    assert 0.0 < drift < 1.0e-3


def test_posenet_stem_mobileone_block0_matches_torch_on_mlx_cpu() -> None:
    torch.manual_seed(31)
    block0 = _loaded_posenet_stem_block0()
    x = torch.randn(1, 12, 32, 40)

    expected = block0(x).detach().numpy()
    with temporary_mlx_device("cpu"):
        actual = run_mlx_mobileone_block_nchw(torch_mobileone_block_to_mlx(block0), x.numpy())

    assert actual.shape == expected.shape
    assert _max_abs(actual, expected) < 1.0e-4


def test_posenet_stem_mobileone_block0_gpu_drift_is_measured() -> None:
    torch.manual_seed(31)
    block0 = _loaded_posenet_stem_block0()
    x = torch.randn(1, 12, 32, 40)

    expected = block0(x).detach().numpy()
    with temporary_mlx_device("gpu"):
        actual = run_mlx_mobileone_block_nchw(torch_mobileone_block_to_mlx(block0), x.numpy())

    drift = _max_abs(actual, expected)
    assert 0.0 < drift < 2.0e-3


def test_posenet_mobileone_stem_matches_torch_on_mlx_cpu() -> None:
    torch.manual_seed(37)
    stem = _loaded_posenet_stem()
    x = torch.randn(1, 12, 64, 80)

    expected = stem(x).detach().numpy()
    with temporary_mlx_device("cpu"):
        actual = run_mlx_mobileone_stem_nchw(torch_mobileone_stem_to_mlx(stem), x.numpy())

    assert actual.shape == expected.shape
    assert _max_abs(actual, expected) < 1.0e-4


def test_posenet_mobileone_stem_gpu_drift_is_measured() -> None:
    torch.manual_seed(37)
    stem = _loaded_posenet_stem()
    x = torch.randn(1, 12, 64, 80)

    expected = stem(x).detach().numpy()
    with temporary_mlx_device("gpu"):
        actual = run_mlx_mobileone_stem_nchw(torch_mobileone_stem_to_mlx(stem), x.numpy())

    drift = _max_abs(actual, expected)
    assert 0.0 < drift < 2.0e-3


def test_posenet_stage0_repmixer_block0_matches_torch_on_mlx_cpu() -> None:
    torch.manual_seed(41)
    block = _loaded_posenet_stage0_block0()
    x = torch.randn(1, 64, 16, 20)

    expected = block(x).detach().numpy()
    with temporary_mlx_device("cpu"):
        actual = run_mlx_repmixer_block_nchw(torch_repmixer_block_to_mlx(block), x.numpy())

    assert actual.shape == expected.shape
    assert _max_abs(actual, expected) < 3.0e-4


def test_posenet_stage0_repmixer_block0_gpu_drift_is_measured() -> None:
    torch.manual_seed(41)
    block = _loaded_posenet_stage0_block0()
    x = torch.randn(1, 64, 16, 20)

    expected = block(x).detach().numpy()
    with temporary_mlx_device("gpu"):
        actual = run_mlx_repmixer_block_nchw(torch_repmixer_block_to_mlx(block), x.numpy())

    drift = _max_abs(actual, expected)
    assert 0.0 < drift < 3.0e-3


def test_posenet_fastvit_stage0_matches_torch_on_mlx_cpu() -> None:
    torch.manual_seed(43)
    stage = _loaded_posenet_stage(0)
    x = torch.randn(1, 64, 16, 20)

    expected = stage(x).detach().numpy()
    with temporary_mlx_device("cpu"):
        actual = run_mlx_fastvit_stage_nchw(torch_fastvit_stage_to_mlx(stage), x.numpy())

    assert actual.shape == expected.shape
    assert _max_abs(actual, expected) < 5.0e-4


def test_posenet_stage1_patch_embed_matches_torch_on_mlx_cpu() -> None:
    torch.manual_seed(47)
    patch_embed = _loaded_posenet_stage(1).downsample
    x = torch.randn(1, 64, 16, 20)

    expected = patch_embed(x).detach().numpy()
    with temporary_mlx_device("cpu"):
        actual = run_mlx_patch_embed_nchw(torch_patch_embed_to_mlx(patch_embed), x.numpy())

    assert actual.shape == expected.shape
    assert _max_abs(actual, expected) < 2.0e-4


def test_posenet_fastvit_stage1_matches_torch_on_mlx_cpu() -> None:
    torch.manual_seed(53)
    stage = _loaded_posenet_stage(1)
    x = torch.randn(1, 64, 16, 20)

    expected = stage(x).detach().numpy()
    with temporary_mlx_device("cpu"):
        actual = run_mlx_fastvit_stage_nchw(torch_fastvit_stage_to_mlx(stage), x.numpy())

    assert actual.shape == expected.shape
    assert _max_abs(actual, expected) < 8.0e-4


def test_posenet_fastvit_stage1_gpu_drift_is_measured() -> None:
    torch.manual_seed(53)
    stage = _loaded_posenet_stage(1)
    x = torch.randn(1, 64, 16, 20)

    expected = stage(x).detach().numpy()
    with temporary_mlx_device("gpu"):
        actual = run_mlx_fastvit_stage_nchw(torch_fastvit_stage_to_mlx(stage), x.numpy())

    drift = _max_abs(actual, expected)
    assert 0.0 < drift < 5.0e-2


@pytest.mark.parametrize(
    "stage_index,input_shape,seed,tolerance",
    [
        (2, (1, 128, 8, 10), 59, 1.0e-3),
        (3, (1, 256, 4, 5), 61, 1.0e-3),
    ],
)
def test_posenet_later_fastvit_stages_match_torch_on_mlx_cpu(
    stage_index: int,
    input_shape: tuple[int, int, int, int],
    seed: int,
    tolerance: float,
) -> None:
    torch.manual_seed(seed)
    stage = _loaded_posenet_stage(stage_index)
    x = torch.randn(*input_shape)

    expected = stage(x).detach().numpy()
    with temporary_mlx_device("cpu"):
        actual = run_mlx_fastvit_stage_nchw(torch_fastvit_stage_to_mlx(stage), x.numpy())

    assert actual.shape == expected.shape
    assert _max_abs(actual, expected) < tolerance


def test_posenet_fastvit_vision_matches_torch_on_mlx_cpu() -> None:
    torch.manual_seed(67)
    vision = _loaded_posenet_vision()
    x = torch.randn(1, 12, 64, 80)

    expected = vision(x).detach().numpy()
    with temporary_mlx_device("cpu"):
        actual = run_mlx_fastvit_vision_nchw(torch_fastvit_vision_to_mlx(vision), x.numpy())

    assert actual.shape == expected.shape
    assert _max_abs(actual, expected) < 2.0e-3


def test_posenet_fastvit_vision_gpu_drift_is_measured() -> None:
    torch.manual_seed(67)
    vision = _loaded_posenet_vision()
    x = torch.randn(1, 12, 64, 80)

    expected = vision(x).detach().numpy()
    with temporary_mlx_device("gpu"):
        actual = run_mlx_fastvit_vision_nchw(torch_fastvit_vision_to_mlx(vision), x.numpy())

    drift = _max_abs(actual, expected)
    assert 0.0 < drift < 1.0e-2


def test_posenet_end_to_end_matches_torch_on_mlx_cpu() -> None:
    torch.manual_seed(71)
    posenet = _loaded_posenet()
    x = torch.randn(1, 12, 64, 80)

    expected = posenet(x)["pose"].detach().numpy()
    with temporary_mlx_device("cpu"):
        actual = run_mlx_posenet_nchw(torch_posenet_to_mlx(posenet), x.numpy())["pose"]

    assert actual.shape == expected.shape
    assert _max_abs(actual, expected) < 2.0e-3


def test_posenet_end_to_end_gpu_drift_is_measured() -> None:
    torch.manual_seed(71)
    posenet = _loaded_posenet()
    x = torch.randn(1, 12, 64, 80)

    expected = posenet(x)["pose"].detach().numpy()
    with temporary_mlx_device("gpu"):
        actual = run_mlx_posenet_nchw(torch_posenet_to_mlx(posenet), x.numpy())["pose"]

    drift = _max_abs(actual, expected)
    assert 0.0 < drift < 5.0e-2


def test_segnet_depthwise_separable_block0_matches_torch_on_mlx_cpu() -> None:
    torch.manual_seed(73)
    block = _loaded_segnet_encoder_block(0, 0)
    x = torch.randn(1, 32, 8, 10)

    expected = block(x).detach().numpy()
    with temporary_mlx_device("cpu"):
        actual = run_mlx_efficientnet_block_nchw(
            torch_efficientnet_block_to_mlx(block),
            x.numpy(),
        )

    assert actual.shape == expected.shape
    assert _max_abs(actual, expected) < 2.0e-4


def test_segnet_inverted_residual_block10_matches_torch_on_mlx_cpu() -> None:
    torch.manual_seed(79)
    block = _loaded_segnet_encoder_block(1, 0)
    x = torch.randn(1, 16, 8, 10)

    expected = block(x).detach().numpy()
    with temporary_mlx_device("cpu"):
        actual = run_mlx_efficientnet_block_nchw(
            torch_efficientnet_block_to_mlx(block),
            x.numpy(),
        )

    assert actual.shape == expected.shape
    assert _max_abs(actual, expected) < 5.0e-4


def test_segnet_inverted_residual_block10_gpu_drift_is_measured() -> None:
    torch.manual_seed(79)
    block = _loaded_segnet_encoder_block(1, 0)
    x = torch.randn(1, 16, 8, 10)

    expected = block(x).detach().numpy()
    with temporary_mlx_device("gpu"):
        actual = run_mlx_efficientnet_block_nchw(
            torch_efficientnet_block_to_mlx(block),
            x.numpy(),
        )

    drift = _max_abs(actual, expected)
    assert 0.0 < drift < 5.0e-2


def test_segnet_efficientnet_stem_matches_torch_on_mlx_cpu() -> None:
    torch.manual_seed(83)
    model = _loaded_segnet_encoder_model()
    x = torch.randn(1, 3, 64, 80)

    expected = model.bn1(model.conv_stem(x)).detach().numpy()
    with temporary_mlx_device("cpu"):
        actual = run_mlx_efficientnet_stem_nchw(
            torch_efficientnet_stem_to_mlx(model),
            x.numpy(),
        )

    assert actual.shape == expected.shape
    assert _max_abs(actual, expected) < 2.0e-4


@pytest.mark.parametrize(
    "stage_index,input_shape,seed,tolerance",
    [
        (0, (1, 32, 8, 10), 89, 3.0e-4),
        (1, (1, 16, 8, 10), 97, 7.0e-4),
    ],
)
def test_segnet_efficientnet_stages_match_torch_on_mlx_cpu(
    stage_index: int,
    input_shape: tuple[int, int, int, int],
    seed: int,
    tolerance: float,
) -> None:
    torch.manual_seed(seed)
    stage = _loaded_segnet_encoder_stage(stage_index)
    x = torch.randn(*input_shape)

    expected = stage(x).detach().numpy()
    with temporary_mlx_device("cpu"):
        actual = run_mlx_efficientnet_stage_nchw(
            torch_efficientnet_stage_to_mlx(stage),
            x.numpy(),
        )

    assert actual.shape == expected.shape
    assert _max_abs(actual, expected) < tolerance


def test_segnet_efficientnet_features_match_torch_on_mlx_cpu() -> None:
    torch.manual_seed(101)
    model = _loaded_segnet_encoder_model()
    x = torch.randn(1, 3, 64, 80)

    expected = [feature.detach().numpy() for feature in model(x)]
    with temporary_mlx_device("cpu"):
        actual = run_mlx_efficientnet_features_nchw(
            torch_efficientnet_features_to_mlx(model),
            x.numpy(),
        )

    assert [item.shape for item in actual] == [item.shape for item in expected]
    assert len(actual) == 5
    max_by_feature = [_max_abs(lhs, rhs) for lhs, rhs in zip(actual, expected, strict=True)]
    assert max(max_by_feature) < 2.0e-3


def test_segnet_timm_universal_encoder_matches_torch_on_mlx_cpu() -> None:
    torch.manual_seed(103)
    encoder = _loaded_segnet_encoder()
    x = torch.randn(1, 3, 64, 80)

    expected = [feature.detach().numpy() for feature in encoder(x)]
    with temporary_mlx_device("cpu"):
        actual = run_mlx_timm_universal_encoder_nchw(
            torch_timm_universal_encoder_to_mlx(encoder),
            x.numpy(),
        )

    assert [item.shape for item in actual] == [item.shape for item in expected]
    assert len(actual) == 6
    np.testing.assert_allclose(actual[0], x.numpy(), atol=0.0, rtol=0.0)
    max_by_feature = [_max_abs(lhs, rhs) for lhs, rhs in zip(actual, expected, strict=True)]
    assert max(max_by_feature) < 2.0e-3


def test_segnet_decoder_conv2d_relu_matches_torch_on_mlx_cpu() -> None:
    torch.manual_seed(107)
    conv = _loaded_segnet_decoder_block(0).conv1
    x = torch.randn(1, 472, 4, 5)

    expected = conv(x).detach().numpy()
    with temporary_mlx_device("cpu"):
        actual = run_mlx_conv2d_relu_nchw(torch_conv2d_relu_to_mlx(conv), x.numpy())

    assert actual.shape == expected.shape
    assert _max_abs(actual, expected) < 7.0e-4


def test_segnet_unet_decoder_block0_matches_torch_on_mlx_cpu() -> None:
    torch.manual_seed(109)
    block = _loaded_segnet_decoder_block(0)
    head = torch.randn(1, 352, 2, 3)
    skip = torch.randn(1, 120, 4, 5)

    expected = block(head, 4, 5, skip_connection=skip).detach().numpy()
    with temporary_mlx_device("cpu"):
        actual = run_mlx_unet_decoder_block_nchw(
            torch_unet_decoder_block_to_mlx(block),
            head.numpy(),
            4,
            5,
            skip.numpy(),
        )

    assert actual.shape == expected.shape
    assert _max_abs(actual, expected) < 2.0e-3


def test_segnet_unet_decoder_matches_torch_on_mlx_cpu() -> None:
    torch.manual_seed(113)
    segnet = _loaded_segnet()
    x = torch.randn(1, 3, 64, 80)
    features = [feature.detach() for feature in segnet.encoder(x)]

    expected = segnet.decoder(features).detach().numpy()
    with temporary_mlx_device("cpu"):
        actual = run_mlx_unet_decoder_nchw(
            torch_unet_decoder_to_mlx(segnet.decoder),
            [feature.numpy() for feature in features],
        )

    assert actual.shape == expected.shape
    assert _max_abs(actual, expected) < 6.0e-3


def test_segnet_segmentation_head_matches_torch_on_mlx_cpu() -> None:
    torch.manual_seed(117)
    head = _loaded_segnet().segmentation_head
    x = torch.randn(1, 16, 64, 80)

    expected = head(x).detach().numpy()
    adapter = torch_segmentation_head_to_mlx(head)
    assert type(adapter.conv).__name__ == "MLXExplicitSpatialConv2dAdapter"
    with temporary_mlx_device("cpu"):
        actual = run_mlx_segmentation_head_nchw(
            adapter,
            x.numpy(),
        )

    assert actual.shape == expected.shape
    assert _max_abs(actual, expected) < 2.0e-4


def test_segnet_end_to_end_logits_match_torch_on_mlx_cpu() -> None:
    torch.manual_seed(119)
    segnet = _loaded_segnet()
    x = torch.randn(1, 3, 64, 80)

    expected = segnet(x).detach().numpy()
    with temporary_mlx_device("cpu"):
        actual = run_mlx_segnet_nchw(torch_segnet_to_mlx(segnet), x.numpy())

    assert actual.shape == expected.shape
    assert _max_abs(actual, expected) < 1.0e-2


def test_distortion_scorer_responses_and_components_match_torch_on_mlx_cpu() -> None:
    torch.manual_seed(123)
    dist = _loaded_distortion_net()
    pose_ref = torch.randn(1, 12, 64, 80)
    pose_cand = pose_ref + 0.01 * torch.randn(1, 12, 64, 80)
    seg_ref = torch.randn(1, 3, 64, 80)
    seg_cand = seg_ref.clone()

    expected_ref = {
        "posenet": dist.posenet(pose_ref),
        "segnet": dist.segnet(seg_ref),
    }
    expected_cand = {
        "posenet": dist.posenet(pose_cand),
        "segnet": dist.segnet(seg_cand),
    }
    expected_pose = dist.posenet.compute_distortion(
        expected_ref["posenet"],
        expected_cand["posenet"],
    ).detach().numpy()
    expected_seg = dist.segnet.compute_distortion(
        expected_ref["segnet"],
        expected_cand["segnet"],
    ).detach().numpy()

    with temporary_mlx_device("cpu"):
        adapter = torch_distortion_net_to_mlx(dist)
        actual_ref = run_mlx_distortion_scorer_nchw(
            adapter,
            pose_ref.numpy(),
            seg_ref.numpy(),
        )
        actual_cand = run_mlx_distortion_scorer_nchw(
            adapter,
            pose_cand.numpy(),
            seg_cand.numpy(),
        )

    assert _max_abs(actual_ref["posenet"]["pose"], expected_ref["posenet"]["pose"].detach().numpy()) < 2.0e-3
    assert _max_abs(actual_ref["segnet"], expected_ref["segnet"].detach().numpy()) < 1.0e-2
    actual_components = scorer_distortion_components_numpy(actual_ref, actual_cand)
    assert _max_abs(actual_components["posenet"], expected_pose) < 2.0e-5
    assert _max_abs(actual_components["segnet"], expected_seg) == 0.0


def _loaded_posenet_stem_block0() -> nn.Module:
    return _loaded_posenet_stem()[0].eval()


def _loaded_posenet_stem() -> nn.Module:
    return _loaded_posenet().vision.stem.eval()


def _loaded_posenet() -> nn.Module:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path("upstream").resolve()))
    import modules  # type: ignore

    posenet = modules.PoseNet().eval()
    posenet.load_state_dict(load_file(modules.posenet_sd_path))
    return posenet.eval()


def _loaded_posenet_stage0_block0() -> nn.Module:
    return _loaded_posenet_stage(0).blocks[0].eval()


def _loaded_posenet_stage(index: int) -> nn.Module:
    return _loaded_posenet_vision().stages[index].eval()


def _loaded_posenet_vision() -> nn.Module:
    return _loaded_posenet().vision.eval()


def _loaded_segnet_encoder_block(stage_index: int, block_index: int) -> nn.Module:
    return _loaded_segnet_encoder_stage(stage_index)[block_index].eval()


def _loaded_segnet_decoder_block(block_index: int) -> nn.Module:
    return _loaded_segnet().decoder.blocks[block_index].eval()


def _loaded_segnet_encoder_stage(stage_index: int) -> nn.Module:
    return _loaded_segnet_encoder_model().blocks[stage_index].eval()


def _loaded_segnet_encoder_model() -> nn.Module:
    return _loaded_segnet_encoder().model.eval()


def _loaded_segnet_encoder() -> nn.Module:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path("upstream").resolve()))

    return _loaded_segnet().encoder.eval()


def _loaded_segnet() -> nn.Module:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path("upstream").resolve()))
    import modules  # type: ignore

    segnet = modules.SegNet().eval()
    segnet.load_state_dict(load_file(modules.segnet_sd_path))
    return segnet.eval()


def _loaded_distortion_net() -> nn.Module:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path("upstream").resolve()))
    import modules  # type: ignore

    dist = modules.DistortionNet().eval()
    dist.load_state_dicts(modules.posenet_sd_path, modules.segnet_sd_path, torch.device("cpu"))
    return dist.eval()
