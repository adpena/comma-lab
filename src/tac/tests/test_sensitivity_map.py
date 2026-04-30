"""Sensitivity-map contract tests for OWV3."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from tac.sensitivity_map import (
    SensitivityMapError,
    conv_weight_shapes,
    load_sensitivity_map,
    require_authoritative_device,
    resolve_layer_sensitivity,
    save_sensitivity_map,
    sensitivity_cv_distance,
    validate_sensitivity_map_for_model,
    validate_sensitivity_vector,
)


def _conv_model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(4, 5, 3, padding=1),
    )


def test_authoritative_device_rejects_non_cuda() -> None:
    require_authoritative_device("cuda:0")
    with pytest.raises(SensitivityMapError, match="record their device"):
        require_authoritative_device(None)
    with pytest.raises(SensitivityMapError, match="require CUDA"):
        require_authoritative_device("cpu")
    with pytest.raises(SensitivityMapError, match="require CUDA"):
        require_authoritative_device("mps")


def test_validate_sensitivity_vector_rejects_shape_nan_and_negative() -> None:
    valid = validate_sensitivity_vector(
        torch.tensor([0.0, 1.0, 2.0]),
        expected_channels=3,
        name="x.weight",
    )
    assert valid.dtype == torch.float32
    assert valid.device.type == "cpu"

    with pytest.raises(SensitivityMapError, match="does not match"):
        validate_sensitivity_vector(torch.ones(2), expected_channels=3, name="bad")
    with pytest.raises(SensitivityMapError, match="NaN/Inf"):
        validate_sensitivity_vector(
            torch.tensor([1.0, float("nan"), 2.0]),
            expected_channels=3,
            name="bad",
        )
    with pytest.raises(SensitivityMapError, match="non-negative"):
        validate_sensitivity_vector(
            torch.tensor([1.0, -1.0, 2.0]),
            expected_channels=3,
            name="bad",
        )


def test_validate_map_requires_every_conv_when_requested() -> None:
    model = _conv_model()
    shapes = conv_weight_shapes(model)
    assert shapes == {"0.weight": 4, "2.weight": 5}

    with pytest.raises(SensitivityMapError, match="missing"):
        validate_sensitivity_map_for_model(
            {"0.weight": torch.ones(4)},
            model,
            require_all_conv=True,
        )

    stats = validate_sensitivity_map_for_model(
        {"0.weight": torch.ones(4), "2.weight": torch.arange(5).float()},
        model,
        require_all_conv=True,
    )
    assert stats.n_layers == 2
    assert stats.n_channels == 9
    assert stats.min_value == 0.0
    assert stats.max_value == 4.0


def test_resolve_layer_sensitivity_accepts_canonical_and_bare_keys() -> None:
    model = _conv_model()
    conv0 = dict(model.named_modules())["0"]
    canonical = resolve_layer_sensitivity(
        {"0.weight": torch.ones(4)},
        module_name="0",
        weight=conv0.weight,
    )
    bare = resolve_layer_sensitivity(
        {"0": torch.ones(4) * 2.0},
        module_name="0",
        weight=conv0.weight,
    )
    assert torch.equal(canonical, torch.ones(4))
    assert torch.equal(bare, torch.ones(4) * 2.0)


def test_save_load_and_cv_distance_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "sens.pt"
    sensitivities = {
        "0.weight": torch.tensor([1.0, 3.0, 0.0, 0.0]),
        "2.weight": torch.arange(5).float(),
    }
    save_sensitivity_map(
        path,
        sensitivities,
        metadata={"device": "cuda:0", "kind": "unit"},
    )
    loaded, metadata = load_sensitivity_map(path)
    assert metadata == {"device": "cuda:0", "kind": "unit"}
    assert set(loaded) == set(sensitivities)
    assert torch.equal(loaded["0.weight"], sensitivities["0.weight"])

    dist = sensitivity_cv_distance(
        {"x.weight": torch.tensor([1.0, 3.0])},
        {"x.weight": torch.tensor([2.0, 2.0])},
    )
    assert dist["x.weight"] == pytest.approx(0.5)

    with pytest.raises(SensitivityMapError, match="non-negative"):
        sensitivity_cv_distance(
            {"x.weight": torch.tensor([1.0, -1.0])},
            {"x.weight": torch.tensor([1.0, 1.0])},
        )
