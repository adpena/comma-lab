# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from tac.local_acceleration.exact_int64_fixedpoint_scorer import (
    ExactInt64Conv2d,
    build_exact_int64_model,
)
from tac.local_acceleration.metal_fixedpoint_verdict import (
    build_fixedpoint_conv_packet,
    fixedpoint_conv2d_numpy,
)


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_exact_int64_conv_matches_numpy_reference(groups: int) -> None:
    generator = torch.Generator().manual_seed(494 + groups)
    conv = nn.Conv2d(4, 8, 3, padding=1, groups=groups, bias=True).eval()
    with torch.no_grad():
        conv.weight.copy_(torch.randn(conv.weight.shape, generator=generator))
        conv.bias.copy_(torch.randn(conv.bias.shape, generator=generator))
    value = torch.randn((1, 4, 5, 6), generator=generator)
    packet = build_fixedpoint_conv_packet(
        conv,
        activation_absmax=1.0,
        bits=12,
        activation_scale_mode="dynamic_exact_absmax",
    )
    wrapper = ExactInt64Conv2d(packet).eval()
    with torch.inference_mode():
        observed = wrapper(value).numpy().transpose(0, 2, 3, 1)
    expected = fixedpoint_conv2d_numpy(value.numpy().transpose(0, 2, 3, 1), packet)
    np.testing.assert_allclose(observed, expected, rtol=2e-6, atol=2e-6)


def test_w26_endpoint_is_retained_as_exact_integer_code() -> None:
    conv = nn.Conv2d(1, 1, 1, bias=False).eval()
    wrapper = ExactInt64Conv2d.from_torch_conv(
        conv,
        bits=26,
        activation_scale_mode="dynamic_exact_absmax",
    )
    qmax = (1 << 25) - 1
    codes, _ = wrapper.quantize_activation(torch.tensor([[[[1.0, -1.0]]]]))
    assert codes.dtype == torch.int64
    assert codes.flatten().tolist() == [qmax, -qmax]


def test_model_conversion_is_complete_and_non_mutating() -> None:
    model = nn.Sequential(
        nn.Conv2d(4, 4, 3, padding=1, groups=4),
        nn.ReLU(),
        nn.Conv2d(4, 3, 1),
    ).eval()
    candidate, manifest = build_exact_int64_model(model, bits=20)
    assert manifest.converted_conv2d_count == 2
    assert manifest.converted_paths == ("0", "2")
    assert manifest.accumulation == "exact_signed_int64"
    assert sum(isinstance(module, nn.Conv2d) for module in model.modules()) == 2
    assert sum(isinstance(module, ExactInt64Conv2d) for module in candidate.modules()) == 2


def test_fixed_scale_requires_every_operator_absmax() -> None:
    model = nn.Sequential(nn.Conv2d(1, 1, 1)).eval()
    with pytest.raises(KeyError, match="missing fixed activation absmax"):
        build_exact_int64_model(
            model,
            bits=12,
            activation_scale_mode="fixed_calibration",
            operator_absmax={},
        )


def test_autograd_is_refused() -> None:
    wrapper = ExactInt64Conv2d.from_torch_conv(nn.Conv2d(1, 1, 1), bits=12)
    with pytest.raises(RuntimeError, match="inference-only"):
        wrapper(torch.ones((1, 1, 2, 2), requires_grad=True))
