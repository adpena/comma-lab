# SPDX-License-Identifier: MIT
"""Tests for MLX-first NumPy portability contract helpers."""

from __future__ import annotations

import pytest

from tac.local_acceleration.mlx_numpy_portability_contract import (
    MLX_NUMPY_PORTABILITY_CONTRACT_SCHEMA,
    build_mlx_numpy_portability_contract,
)


def test_contract_marks_pure_numpy_receiver_ready() -> None:
    contract = build_mlx_numpy_portability_contract(
        substrate_id="toy",
        exported_state_kind="npz",
        archive_payload_kind="single_member",
        receiver_runtime_kind="numpy_decode_receiver",
        receiver_dependencies=("numpy", "python_stdlib"),
        numpy_array_export=True,
        canonical_npz_bridge_used=True,
        pure_numpy_inflate=True,
    )

    assert contract["schema"] == MLX_NUMPY_PORTABILITY_CONTRACT_SCHEMA
    assert contract["portability_status"] == "pure_numpy_inflate_ready"
    assert contract["portability_blockers"] == []
    assert contract["score_claim"] is False
    assert contract["ready_for_exact_eval_dispatch"] is False


def test_contract_distinguishes_torch_receiver_from_numpy_inflate() -> None:
    contract = build_mlx_numpy_portability_contract(
        substrate_id="hi_nerv",
        exported_state_kind="pytorch_layout_numpy_arrays_from_mlx_model",
        archive_payload_kind="hiv1_monolithic_0_bin",
        receiver_runtime_kind="torch_decode_receiver",
        receiver_dependencies=("torch", "brotli", "python_stdlib"),
        numpy_array_export=True,
        canonical_npz_bridge_used=False,
        pure_numpy_inflate=False,
    )

    assert contract["portability_status"] == (
        "numpy_export_bridge_ready_receiver_not_numpy"
    )
    assert "torch" in contract["non_numpy_receiver_dependencies"]
    assert "inflate_runtime_not_pure_numpy" in contract["portability_blockers"]
    assert "canonical_npz_bridge_not_used_or_not_applicable" in contract[
        "portability_blockers"
    ]


def test_contract_refuses_empty_substrate_id() -> None:
    with pytest.raises(ValueError, match="substrate_id"):
        build_mlx_numpy_portability_contract(
            substrate_id="",
            exported_state_kind="npz",
            archive_payload_kind="single_member",
            receiver_runtime_kind="numpy_decode_receiver",
            receiver_dependencies=("numpy",),
            numpy_array_export=True,
            canonical_npz_bridge_used=True,
            pure_numpy_inflate=True,
        )
