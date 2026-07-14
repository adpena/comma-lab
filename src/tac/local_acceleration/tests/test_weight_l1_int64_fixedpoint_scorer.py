# SPDX-License-Identifier: MIT
from __future__ import annotations

import torch
from torch import nn

from tac.local_acceleration.mixed_int64_fixedpoint_scorer import (
    SIGNED_INT64_MAX,
    static_accumulator_bound,
)
from tac.local_acceleration.weight_l1_int64_fixedpoint_scorer import (
    WeightL1Int64Conv2d,
    build_weight_l1_int64_model,
    maximum_weight_l1_safe_bits,
    quantized_weight_l1_accumulator_bound,
)


def test_weight_l1_bound_is_static_and_tighter_than_all_qmax_bound() -> None:
    conv = nn.Conv2d(472, 8, 3, padding=1, bias=False).eval()
    with torch.no_grad():
        conv.weight.fill_(0.001)
        conv.weight[:, :, 1, 1] = 1.0
    bits = maximum_weight_l1_safe_bits(conv)
    tight = quantized_weight_l1_accumulator_bound(conv, bits=bits)
    coarse = static_accumulator_bound(fan_in=472 * 9, bits=bits)
    assert bits > 26
    assert tight <= SIGNED_INT64_MAX
    assert tight < coarse


def test_wrapper_bound_dominates_bruteforce_accumulator() -> None:
    generator = torch.Generator().manual_seed(494)
    conv = nn.Conv2d(3, 4, 3, padding=1, bias=True).eval()
    wrapper = WeightL1Int64Conv2d(conv, bits=30).eval()
    value = torch.randn((1, 3, 5, 6), generator=generator)
    maximum = value.abs().max()
    scale = maximum / float(wrapper.qmax)
    activation_q = torch.clamp(
        torch.round(value / scale).to(torch.int64),
        -wrapper.qmax,
        wrapper.qmax,
    )
    accumulator = torch.nn.functional.conv2d(
        activation_q,
        wrapper.weight_q,
        padding=1,
    )
    assert int(accumulator.abs().max().item()) <= wrapper.accumulator_bound
    with torch.inference_mode():
        output = wrapper(value)
    assert torch.isfinite(output).all()


def test_model_manifest_is_label_free_and_covers_every_conv() -> None:
    model = nn.Sequential(
        nn.Conv2d(4, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 5, 1),
    )
    candidate, manifest = build_weight_l1_int64_model(model)
    assert isinstance(candidate[0], WeightL1Int64Conv2d)
    assert isinstance(candidate[2], WeightL1Int64Conv2d)
    assert manifest.converted_conv2d_count == 2
    assert manifest.label_or_frame_dependent is False
    assert manifest.bound_kind == (
        "activation_qmax_times_max_output_quantized_weight_l1"
    )
    assert manifest.default_enabled is False
    assert manifest.score_claim is False
