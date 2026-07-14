# SPDX-License-Identifier: MIT
"""CPU contract tests for the custom fixed-point SegNet Metal kernels."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from tac.local_acceleration import metal_fixedpoint_verdict as verdict


def _conv(
    *, groups: int = 1, bits: int = 8, activation_scale_mode: str = "fixed_calibration"
):
    import torch

    conv = torch.nn.Conv2d(4, 4, kernel_size=3, padding=1, groups=groups, bias=True)
    with torch.no_grad():
        values = torch.linspace(-0.8, 0.9, conv.weight.numel()).reshape_as(conv.weight)
        conv.weight.copy_(values)
        conv.bias.copy_(torch.tensor([-0.2, 0.0, 0.1, 0.3]))
    return conv, verdict.build_fixedpoint_conv_packet(
        conv,
        activation_absmax=3.0,
        bits=bits,
        activation_scale_mode=activation_scale_mode,
    )


def test_packet_uses_per_output_scales_and_static_bound() -> None:
    conv, packet = _conv()
    assert packet.weight_scales.shape == (4,)
    assert packet.weight_q_ohwi.shape == (4, 3, 3, 4)
    assert packet.accumulator_bound == 4 * 3 * 3 * 127 * 127
    assert packet.minimum_signed_accumulator_bits == verdict.minimum_signed_bits_for_bound(
        packet.accumulator_bound
    )
    assert packet.bias.shape == (conv.out_channels,)


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_numpy_integer_conv_matches_manual_packet_semantics(groups: int) -> None:
    _, packet = _conv(groups=groups)
    x = np.linspace(-4.0, 4.0, 1 * 5 * 6 * 4, dtype=np.float32).reshape(1, 5, 6, 4)
    actual = verdict.fixedpoint_conv2d_numpy(x, packet)
    assert actual.shape == (1, 5, 6, 4)
    assert np.all(np.isfinite(actual))

    # Reordering integer products cannot change a selected output.
    xq = verdict.quantize_activation_numpy(
        x, activation_scale=packet.activation_scale, qmax=packet.qmax
    )
    oc, oh, ow = 2, 3, 4
    out_per_group = packet.out_channels // packet.groups
    in_per_group = packet.in_channels // packet.groups
    group = oc // out_per_group
    terms = []
    for kh in range(3):
        for kw in range(3):
            ih, iw = oh + kh - 1, ow + kw - 1
            if 0 <= ih < 5 and 0 <= iw < 6:
                for channel in range(in_per_group):
                    ic = group * in_per_group + channel
                    terms.append(
                        int(xq[0, ih, iw, ic])
                        * int(packet.weight_q_ohwi[oc, kh, kw, channel])
                    )
    rng = np.random.default_rng(494)
    permuted = sum(terms[index] for index in rng.permutation(len(terms)))
    expected = np.float32(
        permuted * packet.activation_scale * packet.weight_scales[oc] + packet.bias[oc]
    )
    assert actual[0, oh, ow, oc] == expected


def test_high_precision_packet_requires_int64_but_is_safe() -> None:
    _, packet = _conv(bits=26)
    assert packet.minimum_signed_accumulator_bits > 32
    assert packet.minimum_signed_accumulator_bits <= 64
    assert packet.accumulator_bound <= np.iinfo(np.int64).max


def test_precision_above_single_int64_ladder_is_refused() -> None:
    conv, _ = _conv(bits=26)
    with pytest.raises(ValueError, match="2..26"):
        verdict.build_fixedpoint_conv_packet(
            conv,
            activation_absmax=3.0,
            bits=27,
            activation_scale_mode="dynamic_exact_absmax",
        )


def test_activation_quantization_is_fixed_scale_and_saturating() -> None:
    values = np.asarray([-4.0, -3.0, -1.5, 0.0, 1.5, 3.0, 4.0], dtype=np.float32)
    actual = verdict.quantize_activation_numpy(values, activation_scale=3.0 / 127.0, qmax=127)
    assert actual[0] == -127
    assert actual[-1] == 127
    assert actual[3] == 0


def test_dynamic_scale_uses_current_tensor_absmax_without_clipping() -> None:
    _, packet = _conv(activation_scale_mode="dynamic_exact_absmax")
    x = np.asarray([[[[-4.0, -1.0, 2.0, 3.0]]]], dtype=np.float32)
    scale = verdict._activation_scale_numpy(x, packet)
    assert scale == 4.0 / packet.qmax
    quantized = verdict.quantize_activation_numpy(
        x, activation_scale=scale, qmax=packet.qmax
    )
    assert quantized.min() == -packet.qmax
    assert quantized.max() < packet.qmax


def test_dynamic_packet_rejects_nonfinite_input() -> None:
    _, packet = _conv(activation_scale_mode="dynamic_exact_absmax")
    with pytest.raises(ValueError, match="non-finite"):
        verdict.fixedpoint_conv2d_numpy(
            np.asarray([[[[np.nan, 0.0, 0.0, 0.0]]]], dtype=np.float32), packet
        )


def test_kernel_source_uses_int64_accumulator_and_no_atomic() -> None:
    source = inspect.getsource(verdict._conv_kernel)
    assert "long accumulator" in source
    assert "atomic_fetch" not in source
    signature = verdict.fixedpoint_verdict_signature()
    assert signature["default_enabled"] is False
    assert signature["score_claim"] is False
    assert signature["numpy_integer_reference"] is True
