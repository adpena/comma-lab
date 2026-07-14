# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest
import torch
from torch import nn

from tac.local_acceleration.mixed_int64_fixedpoint_scorer import (
    SIGNED_INT64_MAX,
    MixedInt64Conv2d,
    build_mixed_int64_model,
    maximum_safe_bits,
    signed_qmax,
    static_accumulator_bound,
)


def test_geometry_safe_assignment_uses_highest_bound_preserving_bits() -> None:
    conv = nn.Conv2d(472, 8, 3, bias=False)
    bits = maximum_safe_bits(conv, minimum_bits=26, maximum_bits=30)
    assert bits == 26
    assert static_accumulator_bound(fan_in=472 * 9, bits=bits) <= SIGNED_INT64_MAX
    assert static_accumulator_bound(fan_in=472 * 9, bits=bits + 1) > SIGNED_INT64_MAX


def test_small_depthwise_layer_receives_w30() -> None:
    conv = nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False)
    assert maximum_safe_bits(conv, minimum_bits=26, maximum_bits=30) == 30


def test_mixed_model_manifest_is_geometry_only_and_complete() -> None:
    model = nn.Sequential(
        nn.Conv2d(472, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, padding=1, groups=8),
    ).eval()
    candidate, manifest = build_mixed_int64_model(model)
    assert manifest.converted_conv2d_count == 2
    assert dict(manifest.bits_by_path) == {"0": 26, "2": 30}
    assert dict(manifest.precision_histogram) == {26: 1, 30: 1}
    assert manifest.assignment_rule.startswith("largest_geometry_safe")
    assert sum(isinstance(module, MixedInt64Conv2d) for module in candidate.modules()) == 2


def test_mixed_conv_is_deterministic_and_finite() -> None:
    generator = torch.Generator().manual_seed(494)
    conv = nn.Conv2d(4, 4, 3, padding=1, groups=2).eval()
    value = torch.randn((1, 4, 7, 8), generator=generator)
    wrapper = MixedInt64Conv2d(conv, bits=29).eval()
    with torch.inference_mode():
        first = wrapper(value)
        second = wrapper(value)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()


def test_precision_contract_rejects_non_int32_code_width() -> None:
    with pytest.raises(ValueError, match=r"2\.\.31"):
        signed_qmax(32)
