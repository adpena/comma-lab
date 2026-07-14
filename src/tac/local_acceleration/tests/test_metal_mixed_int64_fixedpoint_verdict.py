# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from tac.local_acceleration import metal_mixed_int64_fixedpoint_verdict as mixed_verdict
from tac.local_acceleration.exact_int64_fixedpoint_scorer import ExactInt64Conv2d
from tac.local_acceleration.metal_fixedpoint_verdict import fixedpoint_conv2d_numpy
from tac.local_acceleration.metal_mixed_int64_fixedpoint_verdict import (
    build_mixed_fixedpoint_conv_packet,
    build_weight_l1_fixedpoint_conv_packet,
    derive_mixed_precision_map,
    derive_weight_l1_precision_map,
    mixed_fixedpoint_verdict_signature,
    weight_l1_fixedpoint_verdict_signature,
)


def test_mixed_packet_matches_exact_cpu_twin() -> None:
    generator = torch.Generator().manual_seed(494)
    conv = nn.Conv2d(4, 4, 1, groups=4, bias=True).eval()
    value = torch.randn((1, 4, 5, 6), generator=generator)
    packet = build_mixed_fixedpoint_conv_packet(conv, bits=29)
    cpu = ExactInt64Conv2d(packet).eval()
    with torch.inference_mode():
        observed = cpu(value).numpy().transpose(0, 2, 3, 1)
    expected = fixedpoint_conv2d_numpy(value.numpy().transpose(0, 2, 3, 1), packet)
    np.testing.assert_allclose(observed, expected, rtol=2e-6, atol=2e-6)


def test_precision_map_is_geometry_derived() -> None:
    model = nn.Sequential(
        nn.Conv2d(472, 8, 3, padding=1),
        nn.Conv2d(8, 8, 3, padding=1, groups=8),
    )
    assert derive_mixed_precision_map(model) == {"0": 26, "1": 30}


def test_signature_is_default_off_and_non_score() -> None:
    signature = mixed_fixedpoint_verdict_signature()
    assert signature["minimum_bits"] == 26
    assert signature["maximum_bits"] == 30
    assert signature["default_enabled"] is False
    assert signature["score_claim"] is False
    assert signature["constant_buffers_cached"] is True


@pytest.mark.parametrize(
    "adapter_type",
    [
        mixed_verdict.MetalMixedInt64Conv2DAdapter,
        mixed_verdict.MetalWeightL1Int64Conv2DAdapter,
    ],
)
def test_adapter_reuses_prepared_device_constants(
    adapter_type: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = object()
    constants = object()
    observed: dict[str, object] = {}

    def fake_conv(value: object, actual_packet: object, *, constants: object) -> str:
        observed.update(value=value, packet=actual_packet, constants=constants)
        return "result"

    monkeypatch.setattr(mixed_verdict, "fixedpoint_conv2d_metal", fake_conv)
    adapter = adapter_type.__new__(adapter_type)
    adapter.packet = packet
    adapter.constants = constants
    assert adapter("input") == "result"
    assert observed == {
        "value": "input",
        "packet": packet,
        "constants": constants,
    }


def test_weight_l1_packet_matches_exact_cpu_twin() -> None:
    generator = torch.Generator().manual_seed(495)
    # W31 is valid only when the frozen quantized-weight L1 bound fits int64.
    # A depthwise 1x1 fixture exercises the W31 packet without violating the
    # same fail-closed bound enforced by the production wrapper.
    conv = nn.Conv2d(4, 4, 1, groups=4, bias=True).eval()
    value = torch.randn((1, 4, 5, 6), generator=generator)
    packet = build_weight_l1_fixedpoint_conv_packet(conv, bits=31)
    cpu = ExactInt64Conv2d(packet).eval()
    with torch.inference_mode():
        observed = cpu(value).numpy().transpose(0, 2, 3, 1)
    expected = fixedpoint_conv2d_numpy(
        value.numpy().transpose(0, 2, 3, 1),
        packet,
    )
    np.testing.assert_allclose(observed, expected, rtol=2e-6, atol=2e-6)


def test_weight_l1_precision_map_and_signature_are_label_free() -> None:
    model = nn.Sequential(
        nn.Conv2d(472, 8, 3, padding=1),
        nn.Conv2d(8, 8, 3, padding=1, groups=8),
    )
    precision = derive_weight_l1_precision_map(model)
    assert set(precision) == {"0", "1"}
    assert all(26 <= bits <= 31 for bits in precision.values())
    signature = weight_l1_fixedpoint_verdict_signature()
    assert signature["maximum_bits"] == 31
    assert signature["label_or_frame_dependent"] is False
    assert signature["bound_kind"] == (
        "activation_qmax_times_max_output_quantized_weight_l1"
    )
    assert signature["default_enabled"] is False
    assert signature["constant_buffers_cached"] is True
