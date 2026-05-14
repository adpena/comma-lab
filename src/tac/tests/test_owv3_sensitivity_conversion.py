# SPDX-License-Identifier: MIT
"""Tests for Fisher -> OWV3 sensitivity-map conversion."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from experiments.convert_fisher_to_owv3_sensitivity_map import (
    SensitivityConversionError,
    convert_importance_to_channel_sensitivity,
)


def _model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(4, 5, 1),
    )


def test_convert_fisher_sum_matches_second_order_channel_trace() -> None:
    model = _model()
    importance = {
        "0.weight": torch.ones_like(model[0].weight) * 2.0,
        "2.weight": torch.arange(model[2].weight.numel(), dtype=torch.float32).reshape_as(model[2].weight),
    }
    out = convert_importance_to_channel_sensitivity(
        model=model,
        importance=importance,
        aggregate="sum",
        missing_policy="error",
    )
    assert torch.equal(out["0.weight"], torch.full((4,), 54.0))
    expected = importance["2.weight"].reshape(5, -1).sum(dim=1)
    assert torch.equal(out["2.weight"], expected)


def test_missing_policy_protect_emits_high_sensitivity() -> None:
    model = _model()
    out = convert_importance_to_channel_sensitivity(
        model=model,
        importance={"0.weight": torch.ones_like(model[0].weight)},
        missing_policy="protect",
        missing_value=1e-2,
    )
    assert torch.equal(out["2.weight"], torch.full((5,), 1e-2))


def test_missing_policy_default_is_error() -> None:
    model = _model()
    with pytest.raises(SensitivityConversionError, match="missing Fisher"):
        convert_importance_to_channel_sensitivity(
            model=model,
            importance={"0.weight": torch.ones_like(model[0].weight)},
        )


def test_protected_missing_conv_defaults_to_error_for_promotion() -> None:
    class Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.renderer = nn.Module()
            self.renderer.stem_conv = nn.Conv2d(3, 4, 3, padding=1)
            self.renderer.fuse_conv = nn.Conv2d(4, 4, 1)

    model = Toy()
    with pytest.raises(SensitivityConversionError, match="missing Fisher"):
        convert_importance_to_channel_sensitivity(
            model=model,
            importance={
                "renderer.stem_conv.weight": torch.ones_like(model.renderer.stem_conv.weight),
            },
            missing_policy="error",
        )


def test_protected_missing_conv_legacy_debug_policy_is_explicit() -> None:
    class Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.renderer = nn.Module()
            self.renderer.stem_conv = nn.Conv2d(3, 4, 3, padding=1)
            self.renderer.fuse_conv = nn.Conv2d(4, 4, 1)
            self.renderer.head = nn.Conv2d(4, 3, 1)
            self.other = nn.Conv2d(4, 5, 1)

    model = Toy()
    out = convert_importance_to_channel_sensitivity(
        model=model,
        importance={
            "renderer.stem_conv.weight": torch.ones_like(model.renderer.stem_conv.weight),
            "other.weight": torch.ones_like(model.other.weight),
        },
        missing_policy="error",
        protected_missing_policy="protect",
        missing_value=1e-2,
    )

    assert torch.equal(out["renderer.fuse_conv.weight"], torch.full((4,), 1e-2))
    assert torch.equal(out["renderer.head.weight"], torch.full((3,), 1e-2))
    assert "other.weight" in out


def test_protected_missing_policy_error_is_available_for_forensics() -> None:
    class Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.renderer = nn.Module()
            self.renderer.stem_conv = nn.Conv2d(3, 4, 3, padding=1)
            self.renderer.fuse_conv = nn.Conv2d(4, 4, 1)

    model = Toy()
    with pytest.raises(SensitivityConversionError, match="missing Fisher"):
        convert_importance_to_channel_sensitivity(
            model=model,
            importance={
                "renderer.stem_conv.weight": torch.ones_like(model.renderer.stem_conv.weight),
            },
            missing_policy="error",
            protected_missing_policy="error",
        )


def test_nonprotected_missing_still_fails_closed() -> None:
    class Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.renderer = nn.Module()
            self.renderer.stem_conv = nn.Conv2d(3, 4, 3, padding=1)
            self.other = nn.Conv2d(4, 5, 1)

    model = Toy()
    with pytest.raises(SensitivityConversionError, match="missing Fisher tensors.*other.weight"):
        convert_importance_to_channel_sensitivity(
            model=model,
            importance={
                "renderer.stem_conv.weight": torch.ones_like(model.renderer.stem_conv.weight),
            },
            missing_policy="error",
            protected_missing_policy="protect",
        )


def test_missing_policy_error_raises() -> None:
    model = _model()
    with pytest.raises(SensitivityConversionError, match="missing Fisher"):
        convert_importance_to_channel_sensitivity(
            model=model,
            importance={"0.weight": torch.ones_like(model[0].weight)},
            missing_policy="error",
        )


def test_shape_mismatch_raises() -> None:
    model = _model()
    with pytest.raises(SensitivityConversionError, match="shape"):
        convert_importance_to_channel_sensitivity(
            model=model,
            importance={
                "0.weight": torch.ones(4, 3, 3, 3),
                "2.weight": torch.ones(99),
            },
            missing_policy="error",
        )
