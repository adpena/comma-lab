from __future__ import annotations

import warnings

import numpy as np
import torch

from tac.analysis import scorer_native_diff
from tac.analysis.scorer_native_diff import (
    analytic_scorer_knowledge,
    finalize_scorer_native_product,
    measure_scorer_native_product,
    selected_relay_names,
)


class _Stage(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Conv2d(channels, channels, 3, padding=1),
                    torch.nn.BatchNorm2d(channels),
                    torch.nn.GELU(),
                    LayerScale2d(channels),
                )
            ]
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.blocks[0](value)


class LayerScale2d(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(channels, 1, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.gamma * value


class Sigmoid(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(value)


class _TimmStyleSE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Conv2d(4, 2, 1)
        self.fc2 = torch.nn.Conv2d(2, 4, 1)
        self.gate = Sigmoid()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        pooled = value.mean(dim=(2, 3), keepdim=True)
        return value * self.gate(self.fc2(torch.relu(self.fc1(pooled))))


class _Vision(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
        )
        self.stages = torch.nn.ModuleList([_Stage(4)])
        self.final_conv = torch.nn.Conv2d(4, 4, 1)
        self.head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(4, 4),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.stem(value)
        value = self.stages[0](value)
        value = self.final_conv(value)
        return self.head(value)


class _Hydra(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resblock = torch.nn.Linear(4, 4)
        self.in_layer = torch.nn.ModuleDict({"pose": torch.nn.Linear(4, 4)})
        self.res_layer = torch.nn.ModuleDict({"pose": torch.nn.Linear(4, 4)})
        self.final_layer = torch.nn.ModuleDict({"pose": torch.nn.Linear(4, 6)})

    def forward(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        value = self.resblock(value)
        value = self.in_layer["pose"](value)
        value = self.res_layer["pose"](value)
        return {"pose": self.final_layer["pose"](value)}


class _TinyPose(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision = _Vision()
        self.summarizer = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
        self.hydra = _Hydra()

    def forward(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.hydra(self.summarizer(self.vision(value)))


class _FlattenBatchNorm(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.bn(value.reshape(-1, 1)).reshape(value.shape)


class _TinyPoseFlattenBN(_TinyPose):
    def __init__(self) -> None:
        super().__init__()
        self.summarizer = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            _FlattenBatchNorm(),
            torch.nn.ReLU(),
        )


def test_analytic_knowledge_is_weight_derived_and_phase_resolved() -> None:
    model = _TinyPose().eval()
    value = analytic_scorer_knowledge(
        model,
        scorer="posenet",
        weights_sha256="1" * 64,
    )
    assert value["batchnorm"][0]["derivation_status"] == "EXACT_FROM_FROZEN_WEIGHTS"
    assert len(value["convolution_frequency_phase"][0]["frequency_phase_samples"]) == 10
    assert len(value["layer_scale_functions"]) == 1
    assert value["resize_and_sampling_overlay"]["camera_to_scorer"][
        "single_lti_transfer_function"
    ] is False


def test_product_reduces_full_axes_and_finalizes_without_hook_leak() -> None:
    torch.manual_seed(4)
    model = _TinyPose().eval()
    relay_names = selected_relay_names(model, "posenet")
    assert "vision.stages.0.blocks.0" in relay_names
    painted = torch.randn(2, 3, 16, 16)
    ground_truth = painted + 0.25
    baseline_hooks = sum(len(module._forward_hooks) for module in model.modules())
    output, batch = measure_scorer_native_product(
        model,
        scorer="posenet",
        grouped_inputs={
            "painted_pair": painted,
            "gt_pair": ground_truth,
        },
        contrasts={
            "painted_pair_vs_gt_pair": {
                "painted_pair": 1.0,
                "gt_pair": -1.0,
            }
        },
    )
    assert output["pose"].shape == (4, 6)
    assert sum(len(module._forward_hooks) for module in model.modules()) == baseline_hooks
    final = finalize_scorer_native_product([batch, batch])
    assert final["pair_count"] == 4
    assert final["product_axes"]["channel"] == (
        "exact channel moments and contrast energy"
    )
    assert final["layers"][0]["contrasts"]["painted_pair_vs_gt_pair"]["rms"] > 0
    assert final["relay_ranking"][0]["relay_score"] >= 0
    assert final["layer_scales"][0]["layer"] == "vision.stages.0.blocks.0.3"


def test_finalizer_recovers_execution_order_after_json_key_sorting() -> None:
    model = _TinyPose().eval()
    painted = torch.randn(2, 3, 16, 16)
    _, batch = measure_scorer_native_product(
        model,
        scorer="posenet",
        grouped_inputs={
            "painted_pair": painted,
            "gt_pair": painted + 0.125,
        },
        contrasts={
            "painted_pair_vs_gt_pair": {
                "painted_pair": 1.0,
                "gt_pair": -1.0,
            }
        },
    )
    for key in ("layers", "batchnorm", "se_gates", "layer_scales"):
        batch[key] = dict(sorted(batch[key].items()))
    final = finalize_scorer_native_product([batch])
    orders = [int(row["order"]) for row in final["layers"]]
    assert orders == sorted(orders)
    assert final["layers"][0]["layer"] == "vision.stem"


def test_auxiliary_bn_preserves_groups_across_batch_embedding_flatten() -> None:
    model = _TinyPoseFlattenBN().eval()
    painted = torch.randn(2, 3, 16, 16)
    _, batch = measure_scorer_native_product(
        model,
        scorer="posenet",
        grouped_inputs={
            "painted_pair": painted,
            "gt_pair": painted + 0.125,
        },
        contrasts={
            "painted_pair_vs_gt_pair": {
                "painted_pair": 1.0,
                "gt_pair": -1.0,
            }
        },
    )
    flattened = batch["batchnorm"]["summarizer.1.bn"]
    assert flattened["shape_per_group"] == [8, 1]
    final = finalize_scorer_native_product([batch])
    assert final["batchnorm"][-1]["layer"] == "summarizer.1.bn"


def test_stable_rank_proxy_scales_before_power_iteration() -> None:
    trajectory = np.asarray(
        [
            [1.0e200, -1.0e200, 0.0],
            [-1.0e200, 1.0e200, 1.0e200],
            [0.0, 0.0, -1.0e200],
        ],
        dtype=np.float64,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = scorer_native_diff._stable_rank_proxy(trajectory)
    assert np.isfinite(result["stable_rank"])
    assert np.isfinite(result["top_singular_value"])
    assert result["stable_rank"] >= 1.0


def test_timm_style_sigmoid_gate_is_derived() -> None:
    model = torch.nn.Sequential(_TimmStyleSE()).eval()
    rows = scorer_native_diff._se_gate_rows(model)
    assert len(rows) == 1
    assert rows[0]["layer"] == "0"
