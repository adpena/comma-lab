"""Tests for the Jacobian-weighted selected-K producer scaffold."""

from __future__ import annotations

import numpy as np
import pytest

from tac.codec.cost_curves import TensorBlob
from tac.optimization.jacobian_weighted_selected_k import (
    JacobianSelectedKError,
    build_jacobian_selected_k_manifest,
    resolve_importance_manifest,
)


def _certified_payload() -> dict:
    return {
        "schema": "future_jacobian_importance_manifest.v0",
        "metadata": {
            "device": "cuda:0",
            "source": "official_component_jacobian_pullback",
            "component": "combined",
        },
        "per_channel": [
            {
                "tensor_name": "hi.weight",
                "channel_importance": [10.0, 12.0],
            },
            {
                "tensor_name": "lo.weight",
                "channel_importance": [1.0, 1.0],
            },
        ],
    }


def test_jacobian_selected_k_manifest_emits_no_dead_k_schema() -> None:
    tensors = [
        TensorBlob("hi.weight", np.array([1, -1], dtype=np.int32)),
        TensorBlob("lo.weight", np.array([1, -1], dtype=np.int32)),
    ]
    curves = [
        [
            {"K": 1, "byte_proxy": 100, "rel_err": 0.0},
            {"K": 8, "byte_proxy": 50, "rel_err": 0.5},
        ],
        [
            {"K": 1, "byte_proxy": 100, "rel_err": 0.0},
            {"K": 8, "byte_proxy": 50, "rel_err": 0.5},
        ],
    ]

    manifest = build_jacobian_selected_k_manifest(
        tensors=tensors,
        importance_payload=_certified_payload(),
        rms_targets=[0.4],
        k_range=[1, 8],
        curves=curves,
        producer_tool="unit-test",
    )

    row = manifest["weighted_k_allocations"][0]
    assert row["selected_Ks"] == [1, 8]
    assert row["selected_K_by_tensor"][0]["tensor_name"] == "hi.weight"
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["rank_or_kill_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["downstream_selected_Ks_can_change_charged_bits"] is True
    assert manifest["integration_point"]["selected_Ks_field"] == (
        "weighted_k_allocations[].selected_Ks"
    )


def test_resolve_importance_manifest_reduces_per_channel_in_tensor_order() -> None:
    resolved = resolve_importance_manifest(
        _certified_payload(),
        ["hi.weight", "lo.weight"],
    )

    assert resolved.importance == [11.0, 1.0]
    assert resolved.metadata_gate["status"] == "passed"
    assert resolved.per_tensor[0]["importance_reduction"] == "mean_channels:channel_importance"
    assert resolved.per_tensor[0]["importance_source_value_count"] == 2


def test_jacobian_importance_rejects_cpu_metadata() -> None:
    payload = _certified_payload()
    payload["metadata"]["device"] = "cpu"

    with pytest.raises(JacobianSelectedKError, match="cuda_device_gate rejected"):
        resolve_importance_manifest(payload, ["hi.weight", "lo.weight"])


def test_jacobian_importance_rejects_diagnostic_proxy_metadata() -> None:
    payload = _certified_payload()
    payload["metadata"]["proxy"] = True
    payload["metadata"]["source"] = "diagnostic proxy pullback"

    with pytest.raises(JacobianSelectedKError, match="proxy"):
        resolve_importance_manifest(payload, ["hi.weight", "lo.weight"])


def test_jacobian_importance_rejects_uniform_or_missing_signal() -> None:
    payload = _certified_payload()
    payload["per_channel"] = [
        {"tensor_name": "hi.weight", "channel_importance": [1.0]},
        {"tensor_name": "lo.weight", "channel_importance": [1.0]},
    ]
    with pytest.raises(JacobianSelectedKError, match="uniform"):
        resolve_importance_manifest(payload, ["hi.weight", "lo.weight"])

    payload["per_channel"] = [
        {"tensor_name": "hi.weight", "channel_importance": [10.0]},
    ]
    with pytest.raises(JacobianSelectedKError, match="lacks tensor"):
        resolve_importance_manifest(payload, ["hi.weight", "lo.weight"])
