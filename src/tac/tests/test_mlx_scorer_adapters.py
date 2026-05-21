# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

pytest.importorskip("mlx.core")

from tac.local_acceleration.mlx_scorer_adapters import (  # noqa: E402
    run_mlx_batchnorm2d_nchw,
    run_mlx_conv2d_nchw,
    run_mlx_linear,
    torch_batchnorm2d_to_mlx,
    torch_conv2d_to_mlx,
    torch_linear_to_mlx,
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
