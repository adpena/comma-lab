# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO / "tools" / "profile_segnet_blocks.py"
SPEC = importlib.util.spec_from_file_location("profile_segnet_blocks", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
profiler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profiler)


class _ToyEncoderModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_stem = nn.Linear(4, 4, bias=False)
        self.bn1 = nn.ReLU()
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.ReLU(), nn.Linear(4, 4, bias=False)) for _ in range(2)]
        )


class _ToyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _ToyEncoderModel()


class _ToyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.ReLU(), nn.Linear(4, 4, bias=False))]
        )


class _ToySegNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _ToyEncoder()
        self.decoder = _ToyDecoder()
        self.segmentation_head = nn.Linear(4, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder.model.conv_stem(x)
        x = self.encoder.model.bn1(x)
        for block in self.encoder.model.blocks:
            x = block(x)
        for block in self.decoder.blocks:
            x = block(x)
        return self.segmentation_head(x)


def _toy_model() -> _ToySegNet:
    torch.manual_seed(7)
    model = _ToySegNet().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def test_select_profile_blocks_is_exact_and_nonoverlapping() -> None:
    model = _toy_model()
    selected = profiler.select_profile_blocks(model)
    assert [name for name, _ in selected] == [
        "encoder.stem",
        "encoder.bn1",
        "encoder.block0",
        "encoder.block1",
        "decoder.block0",
        "segmentation_head",
    ]
    assert len({id(module) for _, module in selected}) == len(selected)


def test_toy_profile_has_raw_accounting_and_non_authority_warning() -> None:
    result = profiler.profile_blocks(
        _toy_model(),
        torch.arange(8, dtype=torch.float32).reshape(2, 4),
        warmups=1,
        sample_count=2,
    )

    assert result["sample_count"] == 2
    assert result["warmup_count"] == 1
    assert result["device"] == "cpu"
    assert result["dtype"] == "torch.float32"
    assert "not an uninstrumented throughput benchmark" in result[
        "timing_overhead_warning"
    ]
    assert len(result["raw_samples"]) == 2
    assert len(result["blocks"]) == 6
    assert result["saved_tensors"]["not_peak_rss"] is True
    assert result["saved_tensors"]["logical_saved_bytes_median"] > 0
    assert 0.0 < result["selected_timing_coverage_median"] <= 1.0
    assert len(result["selected_timing_coverage_samples"]) == 2
    assert "residual, not a separately timed module" in result["unattributed_time"][
        "definition"
    ]
    for row in result["blocks"]:
        assert len(row["forward_ms_samples"]) == 2
        assert len(row["backward_ms_samples"]) == 2
        assert len(row["logical_saved_bytes_samples"]) == 2
        assert 0.0 <= row["share_of_profiled_block_median_total"] <= 1.0
        assert 0.0 <= row["share_of_end_to_end_median"] <= 1.0


def _synthetic_sample(scale: float) -> dict[str, object]:
    return {
        "forward_ms": 10.0 * scale,
        "backward_ms": 20.0 * scale,
        "forward_plus_backward_ms": 30.0 * scale,
        "selected_block_forward_ms_sum": 3.0 * scale,
        "selected_block_backward_ms_sum": 7.0 * scale,
        "selected_block_total_ms_sum": 10.0 * scale,
        "selected_timing_coverage": 1.0 / 3.0,
        "unattributed_forward_residual_ms": 7.0 * scale,
        "unattributed_backward_residual_ms": 13.0 * scale,
        "unattributed_total_residual_ms": 20.0 * scale,
        "loss": 1.0,
        "input_gradient_sha256": "a" * 64,
        "blocks": [
            {
                "name": "encoder.stem",
                "module_type": "Linear",
                "outputs": [
                    {"shape": [1, 2], "dtype": "torch.float32", "logical_bytes": 8}
                ],
                "output_logical_bytes": 8,
                "forward_ms": 2.0 * scale,
                "backward_ms": 4.0 * scale,
                "total_ms": 6.0 * scale,
                "logical_saved_bytes": 16,
                "first_owner_unique_storage_bytes": 8,
            },
            {
                "name": "segmentation_head",
                "module_type": "Linear",
                "outputs": [
                    {"shape": [1, 1], "dtype": "torch.float32", "logical_bytes": 4}
                ],
                "output_logical_bytes": 4,
                "forward_ms": 1.0 * scale,
                "backward_ms": 3.0 * scale,
                "total_ms": 4.0 * scale,
                "logical_saved_bytes": 8,
                "first_owner_unique_storage_bytes": 4,
            },
        ],
        "saved_tensors": {
            "owner_rule": profiler.OWNER_RULE,
            "owners": [],
            "logical_saved_bytes_total": 24,
            "unique_storage_bytes_total": 12,
        },
    }


def test_aggregate_samples_is_deterministic_and_preserves_raw_values() -> None:
    samples = [_synthetic_sample(1.0), _synthetic_sample(3.0)]
    first = profiler.aggregate_samples(samples)
    second = profiler.aggregate_samples(samples)
    assert first == second
    assert first["forward_ms_median"] == 20.0
    assert first["backward_ms_median"] == 40.0
    assert first["forward_share_samples"] == [1.0 / 3.0, 1.0 / 3.0]
    assert first["backward_share_samples"] == [2.0 / 3.0, 2.0 / 3.0]
    assert first["backward_only_removal_ceiling_median"] == 3.0
    assert first["selected_timing_coverage_samples"] == [1.0 / 3.0, 1.0 / 3.0]
    assert first["selected_timing_coverage_median"] == 1.0 / 3.0
    assert first["unattributed_time"]["total_residual_ms_samples"] == [20.0, 60.0]
    assert first["blocks"][0]["total_ms_samples"] == [6.0, 18.0]
    assert first["blocks"][0]["total_ms_median"] == 12.0
    assert first["blocks"][0]["share_of_profiled_block_median_total"] == 0.6
    assert first["saved_tensors"]["logical_saved_bytes_samples"] == [24, 24]


def test_dependency_versions_bind_profiler_runtime() -> None:
    versions = profiler._dependency_versions()
    assert set(versions) == {
        "av",
        "numpy",
        "safetensors",
        "segmentation_models_pytorch",
        "timm",
        "torch",
    }
    assert all(isinstance(value, str) and value for value in versions.values())
    assert profiler._environment_custody()["dependency_versions"] == versions


def test_source_file_custody_binds_bytes_and_sha(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_bytes(b"print('bound')\n")
    custody = profiler._source_file_custody(source)
    assert custody["path"] == str(source)
    assert custody["bytes"] == source.stat().st_size
    assert len(custody["sha256"]) == 64


def test_atomic_write_uses_same_directory_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "receipt.json"
    observed: dict[str, Path] = {}
    real_replace = os.replace

    def capture_replace(source: str | Path, target: str | Path) -> None:
        observed["source"] = Path(source)
        observed["target"] = Path(target)
        real_replace(source, target)

    monkeypatch.setattr(profiler.os, "replace", capture_replace)
    profiler.atomic_write_json(destination, {"schema": "toy", "value": 3})

    assert json.loads(destination.read_text()) == {"schema": "toy", "value": 3}
    assert observed["source"].parent == destination.parent
    assert observed["target"] == destination
    assert not observed["source"].exists()


@pytest.mark.parametrize(
    "path",
    [
        Path("/tmp/profile.json"),
        Path("/private/tmp/profile.json"),
        Path("/var/tmp/profile.json"),
        Path("/private/var/folders/example/profile.json"),
    ],
)
def test_validate_durable_output_refuses_transient_paths(path: Path) -> None:
    with pytest.raises(ValueError, match="refusing transient"):
        profiler.validate_durable_output(path)
