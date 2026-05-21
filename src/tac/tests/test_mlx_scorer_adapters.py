# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from safetensors.torch import load_file

pytest.importorskip("mlx.core")

from tac.local_acceleration.mlx_scorer_adapters import (  # noqa: E402
    run_mlx_batchnorm2d_nchw,
    run_mlx_conv2d_nchw,
    run_mlx_fastvit_stage_nchw,
    run_mlx_fastvit_vision_nchw,
    run_mlx_linear,
    run_mlx_mobileone_block_nchw,
    run_mlx_mobileone_stem_nchw,
    run_mlx_patch_embed_nchw,
    run_mlx_posenet_nchw,
    run_mlx_repmixer_block_nchw,
    torch_batchnorm2d_to_mlx,
    torch_conv2d_to_mlx,
    torch_fastvit_stage_to_mlx,
    torch_fastvit_vision_to_mlx,
    torch_linear_to_mlx,
    torch_mobileone_block_to_mlx,
    torch_mobileone_stem_to_mlx,
    torch_patch_embed_to_mlx,
    torch_posenet_to_mlx,
    torch_repmixer_block_to_mlx,
    temporary_mlx_device,
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
    assert 0.0 < drift < 5.0e-3


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


def _loaded_posenet_stem_block0() -> nn.Module:
    return _loaded_posenet_stem()[0].eval()


def _loaded_posenet_stem() -> nn.Module:
    return _loaded_posenet().vision.stem.eval()


def _loaded_posenet() -> nn.Module:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path("upstream").resolve()))
    import modules  # type: ignore  # noqa: PLC0415

    posenet = modules.PoseNet().eval()
    posenet.load_state_dict(load_file(modules.posenet_sd_path))
    return posenet.eval()


def _loaded_posenet_stage0_block0() -> nn.Module:
    return _loaded_posenet_stage(0).blocks[0].eval()


def _loaded_posenet_stage(index: int) -> nn.Module:
    return _loaded_posenet_vision().stages[index].eval()


def _loaded_posenet_vision() -> nn.Module:
    return _loaded_posenet().vision.eval()
