# SPDX-License-Identifier: MIT
"""Tests for targeted Conv2d wrapper that fixes the SegNet MPS drift cliff.

Lane: lane_mps_conv2d_wrap_fix_plus_real_frame_gap_experiment_20260518
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tac.mps_diagnostic.targeted_fix import (
    DEFAULT_CLIFF_LAYER_NAME,
    VALID_STRATEGIES,
    TargetedFixRecord,
    _CPUWrapConv2d,
    _DeterministicAlgorithmsConv2d,
    _FP32ForceConv2d,
    _set_module_by_name,
    try_strategy_chain,
    wrap_drift_cliff_layer,
)


# ---------- Fixtures ----------


def _build_tiny_segnet_like() -> nn.Module:
    """Build a minimal nn.Module shaped like the cliff-layer parent path.

    Mirrors `decoder.blocks.0.conv1.0` to test layer-name resolution and
    Conv2d wrapping without instantiating the full real SegNet.
    """

    class Conv2dReLU(nn.Sequential):
        def __init__(self, in_ch, out_ch):
            super().__init__(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
            )

    class DecoderBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv1 = Conv2dReLU(in_ch, out_ch)
            self.conv2 = Conv2dReLU(out_ch, out_ch)

        def forward(self, x):
            return self.conv2(self.conv1(x))

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([DecoderBlock(472, 256)])

    class TinyScorer(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = Decoder()

        def forward(self, x):
            return self.decoder.blocks[0](x)

    return TinyScorer()


# ---------- TargetedFixRecord dataclass tests ----------


def test_targeted_fix_record_happy_path():
    rec = TargetedFixRecord(
        layer_name=DEFAULT_CLIFF_LAYER_NAME,
        strategy="fp32_force",
        original_class="Conv2d",
    )
    assert rec.layer_name == DEFAULT_CLIFF_LAYER_NAME
    assert rec.strategy == "fp32_force"
    assert rec.original_class == "Conv2d"
    assert rec.fix_evidence_grade == "macOS-MPS-diagnostic"


def test_targeted_fix_record_rejects_empty_layer_name():
    with pytest.raises(ValueError, match="layer_name must be non-empty"):
        TargetedFixRecord(
            layer_name="", strategy="fp32_force", original_class="Conv2d"
        )


def test_targeted_fix_record_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="strategy must be one of"):
        TargetedFixRecord(
            layer_name="x", strategy="UNKNOWN_STRATEGY", original_class="Conv2d"
        )


def test_targeted_fix_record_rejects_empty_original_class():
    with pytest.raises(ValueError, match="original_class must be non-empty"):
        TargetedFixRecord(layer_name="x", strategy="fp32_force", original_class="")


def test_targeted_fix_record_rejects_wrong_evidence_grade():
    with pytest.raises(ValueError, match="MPS auth eval is NOISE"):
        TargetedFixRecord(
            layer_name="x",
            strategy="fp32_force",
            original_class="Conv2d",
            fix_evidence_grade="contest-CUDA",
        )


# ---------- _set_module_by_name tests ----------


def test_set_module_by_name_simple():
    model = _build_tiny_segnet_like()
    new_conv = nn.Conv2d(472, 256, kernel_size=3, padding=1, bias=False)
    _set_module_by_name(model, "decoder.blocks.0.conv1.0", new_conv)
    assert dict(model.named_modules())["decoder.blocks.0.conv1.0"] is new_conv


def test_set_module_by_name_rejects_bad_path():
    model = _build_tiny_segnet_like()
    new_conv = nn.Conv2d(472, 256, kernel_size=3, padding=1, bias=False)
    with pytest.raises(ValueError, match="does not resolve"):
        _set_module_by_name(model, "nonexistent.path", new_conv)


def test_set_module_by_name_rejects_empty():
    model = _build_tiny_segnet_like()
    new_conv = nn.Conv2d(1, 1, 1)
    with pytest.raises(ValueError, match="layer_name must be non-empty"):
        _set_module_by_name(model, "", new_conv)


# ---------- _FP32ForceConv2d tests ----------


def test_fp32_force_conv2d_preserves_shape_and_dtype():
    conv = nn.Conv2d(8, 4, kernel_size=3, padding=1, bias=False)
    wrapped = _FP32ForceConv2d(conv)
    x = torch.randn(2, 8, 16, 16, dtype=torch.float32)
    out = wrapped(x)
    assert out.shape == (2, 4, 16, 16)
    assert out.dtype == torch.float32


def test_fp32_force_conv2d_matches_original_on_cpu_fp32():
    """On CPU fp32, the wrapper should produce identical output to the
    original conv (the explicit fp32 cast is a no-op when input is already
    fp32 on CPU)."""
    torch.manual_seed(0)
    conv = nn.Conv2d(8, 4, kernel_size=3, padding=1, bias=False)
    wrapped = _FP32ForceConv2d(conv)
    x = torch.randn(2, 8, 16, 16, dtype=torch.float32)
    expected = conv(x)
    got = wrapped(x)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_fp32_force_conv2d_preserves_gradient_flow():
    """Gradients must flow through the wrapper (substrate training relies
    on this)."""
    torch.manual_seed(0)
    conv = nn.Conv2d(8, 4, kernel_size=3, padding=1, bias=False)
    wrapped = _FP32ForceConv2d(conv)
    x = torch.randn(2, 8, 16, 16, dtype=torch.float32, requires_grad=True)
    out = wrapped(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().max().item() > 0


# ---------- _CPUWrapConv2d tests ----------


def test_cpu_wrap_conv2d_preserves_shape_and_dtype():
    conv = nn.Conv2d(8, 4, kernel_size=3, padding=1, bias=False)
    wrapped = _CPUWrapConv2d(conv)
    x = torch.randn(2, 8, 16, 16, dtype=torch.float32)
    out = wrapped(x)
    assert out.shape == (2, 4, 16, 16)
    assert out.dtype == torch.float32
    assert out.device.type == "cpu"


def test_cpu_wrap_conv2d_preserves_gradient_flow():
    torch.manual_seed(0)
    conv = nn.Conv2d(8, 4, kernel_size=3, padding=1, bias=False)
    wrapped = _CPUWrapConv2d(conv)
    x = torch.randn(2, 8, 16, 16, dtype=torch.float32, requires_grad=True)
    out = wrapped(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().max().item() > 0


# ---------- _DeterministicAlgorithmsConv2d tests ----------


def test_deterministic_algorithms_conv2d_preserves_shape_and_dtype():
    conv = nn.Conv2d(8, 4, kernel_size=3, padding=1, bias=False)
    wrapped = _DeterministicAlgorithmsConv2d(conv)
    x = torch.randn(2, 8, 16, 16, dtype=torch.float32)
    out = wrapped(x)
    assert out.shape == (2, 4, 16, 16)
    assert out.dtype == torch.float32


# ---------- wrap_drift_cliff_layer tests ----------


def test_wrap_drift_cliff_layer_default_path():
    model = _build_tiny_segnet_like()
    record = wrap_drift_cliff_layer(model)
    assert record.layer_name == DEFAULT_CLIFF_LAYER_NAME
    assert record.strategy == "fp32_force"
    assert record.original_class == "Conv2d"
    # Verify the layer was replaced
    wrapped_module = dict(model.named_modules())[DEFAULT_CLIFF_LAYER_NAME]
    assert isinstance(wrapped_module, _FP32ForceConv2d)


def test_wrap_drift_cliff_layer_cpu_wrap_strategy():
    model = _build_tiny_segnet_like()
    record = wrap_drift_cliff_layer(model, strategy="cpu_wrap")
    assert record.strategy == "cpu_wrap"
    wrapped_module = dict(model.named_modules())[DEFAULT_CLIFF_LAYER_NAME]
    assert isinstance(wrapped_module, _CPUWrapConv2d)


def test_wrap_drift_cliff_layer_rejects_bad_strategy():
    model = _build_tiny_segnet_like()
    with pytest.raises(ValueError, match="strategy must be one of"):
        wrap_drift_cliff_layer(model, strategy="UNKNOWN")


def test_wrap_drift_cliff_layer_rejects_missing_layer():
    model = _build_tiny_segnet_like()
    with pytest.raises(ValueError, match="not found in scorer"):
        wrap_drift_cliff_layer(model, layer_name="does.not.exist")


def test_wrap_drift_cliff_layer_rejects_non_conv2d():
    model = _build_tiny_segnet_like()
    with pytest.raises(ValueError, match="expected Conv2d"):
        # decoder.blocks.0.conv1.1 is BatchNorm2d, not Conv2d
        wrap_drift_cliff_layer(model, layer_name="decoder.blocks.0.conv1.1")


def test_wrap_drift_cliff_layer_end_to_end_forward_works():
    """After wrapping, the full forward pass still returns expected shape."""
    model = _build_tiny_segnet_like()
    wrap_drift_cliff_layer(model)
    x = torch.randn(1, 472, 24, 32)
    out = model(x)
    # decoder.blocks.0(...) produces (1, 256, 24, 32)
    assert out.shape == (1, 256, 24, 32)


# ---------- try_strategy_chain tests ----------


def test_try_strategy_chain_returns_first_succeeding():
    """If the first strategy drives drift below threshold, return it."""
    model = _build_tiny_segnet_like()

    def fake_measure(scorer):
        # Return drift below threshold immediately
        return 1e-5

    record, post_drift = try_strategy_chain(
        model, fake_measure, per_layer_threshold=1e-3
    )
    assert record is not None
    assert record.strategy == "fp32_force"
    assert post_drift == 1e-5


def test_try_strategy_chain_falls_back_through_strategies():
    """Both first and second strategies fail; third succeeds."""
    model = _build_tiny_segnet_like()
    call_count = [0]
    drifts = [1e-2, 1e-2, 1e-5]  # fp32_force fails, cpu_wrap fails, deterministic ok

    def fake_measure(scorer):
        d = drifts[call_count[0]]
        call_count[0] += 1
        return d

    record, post_drift = try_strategy_chain(
        model, fake_measure, per_layer_threshold=1e-3
    )
    assert record is not None
    assert record.strategy == "deterministic_algorithms"
    assert post_drift == 1e-5


def test_try_strategy_chain_returns_best_when_all_fail():
    """If no strategy meets threshold, return the lowest-drift one."""
    model = _build_tiny_segnet_like()
    drifts = [1e-2, 5e-3, 8e-3]

    def fake_measure(scorer):
        return drifts.pop(0)

    record, post_drift = try_strategy_chain(
        model, fake_measure, per_layer_threshold=1e-3
    )
    # Best is cpu_wrap (5e-3 < 8e-3 < 1e-2)
    assert record is not None
    assert record.strategy == "cpu_wrap"
    assert post_drift == 5e-3


# ---------- Constants regression ----------


def test_valid_strategies_pinned():
    assert VALID_STRATEGIES == (
        "fp32_force",
        "cpu_wrap",
        "deterministic_algorithms",
    )


def test_default_cliff_layer_name_pinned():
    assert DEFAULT_CLIFF_LAYER_NAME == "decoder.blocks.0.conv1.0"
