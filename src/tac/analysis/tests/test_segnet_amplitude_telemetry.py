# SPDX-License-Identifier: MIT
from __future__ import annotations

import torch

from tac.analysis.segnet_amplitude_telemetry import (
    compact_amplitude_context,
    finalize_paired_segnet_amplitude,
    measure_paired_segnet_amplitude,
)


class _SE(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = torch.nn.Sigmoid()
        self.scale = torch.nn.Parameter(torch.linspace(-1.0, 1.0, channels))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        pooled = value.mean(dim=(2, 3), keepdim=True)
        gate = self.gate(pooled + self.scale[None, :, None, None])
        return value * gate


class _Block(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(channels)
        self.se = _SE(channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.se(self.bn(value))


class _EncoderModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_stem = torch.nn.Conv2d(3, 4, 1)
        self.blocks = torch.nn.ModuleList([torch.nn.ModuleList([_Block(4)])])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.blocks[0][0](self.conv_stem(value))


class _Encoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _EncoderModel()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.model(value)


class _TinyPairedSegNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(13)
        self.encoder = _Encoder()
        self.segmentation_head = torch.nn.Conv2d(4, 5, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.segmentation_head(self.encoder(value))


def test_paired_amplitude_tracks_bn_se_and_geometry_without_hook_leak() -> None:
    model = _TinyPairedSegNet().eval()
    gt = torch.linspace(0.0, 1.0, steps=2 * 3 * 8 * 8).reshape(2, 3, 8, 8)
    painted = gt + 0.2
    paired = torch.cat([painted, gt], dim=0)
    labels = model(gt).argmax(dim=1)
    baseline = sum(len(module._forward_hooks) for module in model.modules())

    painted_logits, gt_logits, batch = measure_paired_segnet_amplitude(
        model,
        paired,
        split_count=2,
        reference_labels=labels,
    )

    assert painted_logits.shape == gt_logits.shape == (2, 5, 8, 8)
    assert batch["split_count"] == 2
    assert list(batch["bn_layers"]) == ["encoder.model.blocks.0.0.bn"]
    assert list(batch["se_gates"]) == ["encoder.model.blocks.0.0.se.gate"]
    assert set(batch["trajectory_layers"]) == {
        "encoder.model.conv_stem",
        "encoder.model.blocks.0.0",
        "segmentation_head",
    }
    assert sum(len(module._forward_hooks) for module in model.modules()) == baseline


def test_finalized_amplitude_curve_preserves_channel_statistics() -> None:
    model = _TinyPairedSegNet().eval()
    gt = torch.linspace(0.0, 1.0, steps=2 * 3 * 8 * 8).reshape(2, 3, 8, 8)
    painted = gt * 1.1 + 0.1
    labels = model(gt).argmax(dim=1)
    _painted_logits, _gt_logits, batch = measure_paired_segnet_amplitude(
        model,
        torch.cat([painted, gt], dim=0),
        split_count=2,
        reference_labels=labels,
    )

    finalized = finalize_paired_segnet_amplitude([batch])
    assert finalized["pair_count"] == 2
    assert finalized["bn_layers"][0]["channel_count"] == 4
    assert finalized["bn_layers"][0]["painted_vs_gt_mean_z_rms"] > 0.0
    assert finalized["se_gates"][0]["painted_vs_gt_gate_mean_rms"] > 0.0
    assert finalized["summary"]["trajectory_onset_layer"]
    assert finalized["summary"]["trajectory_peak_layer"]
    assert compact_amplitude_context(finalized)["causal_status"].startswith(
        "ASSOCIATION_ONLY"
    )
