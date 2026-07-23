# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest
import torch

from tac.optimization.ddm_runtime_sensitivity import (
    DDMRuntimeDecodedStateV1,
    DDMRuntimePerturbationV1,
    DDMRuntimeRealizedPerturbationV1,
    RuntimeSensitivityError,
    realize_perturbation,
    score_realized_perturbation,
)


def _state() -> DDMRuntimeDecodedStateV1:
    return DDMRuntimeDecodedStateV1(
        manifest={
            "chart": {
                "byteorder": "little",
                "dtype": "int16",
                "streams": [
                    {"name": "anchors", "offset": 0, "shape": [600, 2, 3]},
                    {
                        "name": "gradients",
                        "offset": 7200,
                        "shape": [600, 2, 2, 3],
                    },
                    {
                        "name": "residuals",
                        "offset": 21600,
                        "shape": [600, 2, 12, 16, 3],
                    },
                ],
            }
        },
        manifest_sha256="a" * 64,
        anchors=torch.zeros((600, 2, 3), dtype=torch.int16),
        gradients=torch.zeros((600, 2, 2, 3), dtype=torch.int16),
        residuals=torch.zeros((600, 2, 12, 16, 3), dtype=torch.int16),
        labels=torch.zeros((1, 384, 512), dtype=torch.uint8),
        palette=torch.tensor([[0, 0, 0], [7, 11, 13]], dtype=torch.uint8),
        camera_rows=torch.div(
            torch.arange(874, dtype=torch.int64) * 384,
            874,
            rounding_mode="floor",
        ),
        camera_columns=torch.div(
            torch.arange(1164, dtype=torch.int64) * 512,
            1164,
            rounding_mode="floor",
        ),
        semantic_frame_policy="frame1_only_seg_free_frame0",
        chart_member=b"baseline-chart",
        semantic_member=b"baseline-semantic",
    )


def test_typed_semantic_edit_proves_both_bijection_directions() -> None:
    perturbation = DDMRuntimePerturbationV1(
        stream="semantic/composed",
        flat_index=0,
        delta=1,
        expected_original_value=0,
        pair_start=0,
        pair_stop=1,
    )
    realized = realize_perturbation(_state(), perturbation)
    assert realized.changed_camera_values > 0
    assert np.array_equal(
        realized.baseline_camera[:, 0],
        realized.perturbed_camera[:, 0],
    )
    assert not np.array_equal(
        realized.baseline_camera[:, 1],
        realized.perturbed_camera[:, 1],
    )
    assert realized.member_name == "semantic/composed.dds"
    assert realized.perturbed_member_bytes > 0
    assert realized.perturbed_member_sha256 != realized.baseline_member_sha256
    with pytest.raises(RuntimeSensitivityError, match="precondition"):
        realize_perturbation(
            _state(),
            perturbation.model_copy(update={"expected_original_value": 1}),
        )


def test_typed_chart_edit_roundtrips_counted_member_before_realization() -> None:
    perturbation = DDMRuntimePerturbationV1(
        stream="base/chart.anchors",
        flat_index=0,
        delta=255,
        expected_original_value=0,
        pair_start=0,
        pair_stop=1,
    )
    realized = realize_perturbation(_state(), perturbation)
    assert realized.member_name == "base/chart.ddb"
    assert realized.changed_camera_values > 0
    assert realized.perturbed_member_sha256 != realized.baseline_member_sha256
    assert not np.array_equal(
        realized.baseline_camera[:, 0],
        realized.perturbed_camera[:, 0],
    )


class _Seg:
    def preprocess_input(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (tensor.shape[0], 5, 384, 512),
            dtype=torch.float32,
        )


class _Pose:
    def preprocess_input(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def __call__(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        mean = tensor.mean(dim=(1, 2, 3, 4))
        return {"pose": mean[:, None].repeat(1, 6)}


def test_chunked_score_surface_prices_realized_edit() -> None:
    shape = (1, 2, 874, 1164, 3)
    baseline = np.zeros(shape, dtype=np.uint8)
    perturbed = np.ones(shape, dtype=np.uint8)
    realized = DDMRuntimeRealizedPerturbationV1(
        perturbation=DDMRuntimePerturbationV1(
            stream="base/chart.anchors",
            flat_index=0,
            delta=1,
            pair_start=0,
            pair_stop=1,
        ),
        original_value=0,
        perturbed_value=1,
        baseline_camera=baseline,
        perturbed_camera=perturbed,
        changed_camera_values=int(perturbed.size),
        baseline_camera_sha256="b" * 64,
        perturbed_camera_sha256="c" * 64,
        manifest_sha256="a" * 64,
        member_name="base/chart.ddb",
        baseline_member_bytes=10,
        perturbed_member_bytes=11,
        baseline_member_sha256="d" * 64,
        perturbed_member_sha256="e" * 64,
    )
    result = score_realized_perturbation(
        realized,
        segnet=_Seg(),
        posenet=_Pose(),
        target_labels=np.zeros((1, 384, 512), dtype=np.uint8),
        target_poses=np.zeros((1, 6), dtype=np.float64),
    )
    assert result["receiver_bijection"]["counted_to_output_changed"] is True
    assert result["receiver_bijection"]["output_to_single_owner"] == (
        "base/chart.anchors"
    )
    assert result["delta"]["d_pose"] > 0
    assert result["delta"]["bytes"] == 1
    assert result["first_rung"] is True
    assert result["receiver_bijection"]["serialized_member"] == "base/chart.ddb"
