# SPDX-License-Identifier: MIT
from __future__ import annotations

import torch

from tac.analysis.segnet_internal_telemetry import (
    SegNetInternalTelemetry,
    SegNetTelemetryPolicy,
    assert_telemetry_argmax_identity,
    extract_ordered_pair_boundary_samples,
    summarize_ordered_pair_margins,
)


class _SE(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * 0.75


class _Block(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.se = _SE()
        self.conv = torch.nn.Conv2d(channels, channels, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv(self.se(value))


class _EncoderModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_stem = torch.nn.Conv2d(3, 4, 1)
        self.blocks = torch.nn.ModuleList([torch.nn.ModuleList([_Block(4)])])

    def forward(self, value: torch.Tensor) -> list[torch.Tensor]:
        value = self.conv_stem(value)
        value = self.blocks[0][0](value)
        return [value, value]


class _Encoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _EncoderModel()

    def forward(self, value: torch.Tensor) -> list[torch.Tensor]:
        return self.model(value)


class _DecoderBlock(torch.nn.Module):
    def forward(
        self,
        feature_map: torch.Tensor,
        target_height: int,
        target_width: int,
        skip_connection: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del target_height, target_width
        return feature_map if skip_connection is None else feature_map + skip_connection


class _Decoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([_DecoderBlock()])

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        return self.blocks[0](
            features[-1],
            features[0].shape[-2],
            features[0].shape[-1],
            skip_connection=features[0],
        )


class _TinySegNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.encoder = _Encoder()
        self.decoder = _Decoder()
        self.segmentation_head = torch.nn.Conv2d(4, 5, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        features = self.encoder(value)
        return self.segmentation_head(self.decoder(features))


def test_analysis_and_training_policy_defaults_are_explicit() -> None:
    analysis = SegNetTelemetryPolicy.analysis_default()
    training = SegNetTelemetryPolicy.training_default()
    assert analysis.enabled is True
    assert analysis.cadence == 1
    assert training.enabled is False
    assert training.cadence == 0
    assert "hot loop" in training.reason


def test_hook_coverage_and_argmax_identity() -> None:
    model = _TinySegNet().eval()
    value = torch.linspace(0, 1, steps=3 * 8 * 8).reshape(1, 3, 8, 8)
    identity = assert_telemetry_argmax_identity(model, value)
    assert identity["argmax_identical"] is True
    assert identity["logits_bitwise_identical"] is True
    coverage = identity["telemetry_summary"]["coverage"]
    assert coverage["stem"] == ["encoder.model.conv_stem"]
    assert coverage["encoder_blocks"] == ["encoder.model.blocks.0.0"]
    assert coverage["se_pre"] == ["encoder.model.blocks.0.0.se.pre"]
    assert coverage["se_post"] == ["encoder.model.blocks.0.0.se.post"]
    assert coverage["decoder_skips"] == ["decoder.blocks.0.skip"]
    assert coverage["final_logits"] == ["segmentation_head"]


def test_context_removes_every_hook() -> None:
    model = _TinySegNet().eval()
    baseline = sum(len(module._forward_hooks) for module in model.modules())
    with SegNetInternalTelemetry(model):
        active = sum(len(module._forward_hooks) for module in model.modules())
        assert active > baseline
    assert sum(len(module._forward_hooks) for module in model.modules()) == baseline


def test_ordered_pair_rows_do_not_collapse_direction() -> None:
    logits = torch.full((1, 5, 2, 2), -10.0)
    logits[:, 0] = torch.tensor([[[3.0, 1.0], [3.0, 1.0]]])
    logits[:, 1] = torch.tensor([[[1.0, 3.0], [1.0, 3.0]]])
    rows = {
        row["orientation"]: row
        for row in summarize_ordered_pair_margins(logits)
    }
    assert rows["Road->Lane"]["boundary_pixel_count"] == 2
    assert rows["Lane->Road"]["boundary_pixel_count"] == 2
    assert rows["Road->Lane"]["margin_quantiles"]["median"] == 2.0
    assert rows["Lane->Road"]["margin_quantiles"]["median"] == 2.0
    samples = extract_ordered_pair_boundary_samples(logits)
    assert samples["Road->Lane"]["coordinates_nyx"].shape == (2, 3)
    assert samples["Lane->Road"]["coordinates_nyx"].shape == (2, 3)
