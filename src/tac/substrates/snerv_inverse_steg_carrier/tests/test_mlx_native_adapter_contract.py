# SPDX-License-Identifier: MIT
"""Tests for SNeRV native-MLX adapter contract discovery."""

from __future__ import annotations

import sys
import types

from tac.substrates.snerv_inverse_steg_carrier.mlx_native_adapter_contract import (
    REQUIRED_SURFACES,
    SNERV_MLX_NATIVE_ADAPTER_CONTRACT_SCHEMA,
    build_snerv_mlx_native_adapter_contract,
)


def test_missing_snerv_mlx_native_adapter_fails_closed() -> None:
    contract = build_snerv_mlx_native_adapter_contract(
        module_name="does.not.exist.snerv_mlx_adapter"
    )

    assert contract["schema"] == SNERV_MLX_NATIVE_ADAPTER_CONTRACT_SCHEMA
    assert contract["module_loaded"] is False
    assert contract["surfaces_ready"] is False
    assert contract["full600_campaign_ready"] is False
    assert "snerv_mlx_native_train_export_archive_adapter_missing" in contract[
        "blockers"
    ]
    assert contract["ready_surface_count"] == 0
    assert contract["score_claim"] is False


def test_present_surfaces_still_require_live_smoke(monkeypatch) -> None:
    module_name = "unit_fake_snerv_mlx_adapter"
    module = types.ModuleType(module_name)

    def fn(**_kwargs):
        return None

    for surface in REQUIRED_SURFACES:
        setattr(module, surface.symbol, fn)
    monkeypatch.setitem(sys.modules, module_name, module)

    contract = build_snerv_mlx_native_adapter_contract(module_name=module_name)

    assert contract["module_loaded"] is True
    assert contract["surfaces_ready"] is True
    assert contract["ready_surface_count"] == len(REQUIRED_SURFACES)
    assert contract["full600_campaign_ready"] is False
    assert "snerv_mlx_native_adapter_surfaces_present_but_unproven" in contract[
        "blockers"
    ]


def test_present_surfaces_with_smoke_evidence_unlock_contract(monkeypatch) -> None:
    module_name = "unit_fake_snerv_mlx_adapter_smoked"
    module = types.ModuleType(module_name)

    def fn(**_kwargs):
        return None

    for surface in REQUIRED_SURFACES:
        setattr(module, surface.symbol, fn)
    monkeypatch.setitem(sys.modules, module_name, module)

    contract = build_snerv_mlx_native_adapter_contract(
        module_name=module_name,
        extra_evidence={"two_pair_smoke_passed": True},
    )

    assert contract["surfaces_ready"] is True
    assert contract["two_pair_smoke_passed"] is True
    assert contract["full600_campaign_ready"] is True
    assert contract["blockers"] == []


def test_surface_signature_mismatch_blocks(monkeypatch) -> None:
    module_name = "unit_fake_snerv_mlx_adapter_bad_signature"
    module = types.ModuleType(module_name)

    def train_export_snerv_mlx_native(output_dir):
        return output_dir

    def fn(**_kwargs):
        return None

    for surface in REQUIRED_SURFACES:
        setattr(module, surface.symbol, fn)
    module.train_export_snerv_mlx_native = train_export_snerv_mlx_native
    monkeypatch.setitem(sys.modules, module_name, module)

    contract = build_snerv_mlx_native_adapter_contract(module_name=module_name)

    assert contract["surfaces_ready"] is False
    assert "snerv_mlx_native_surface_signature_mismatch:train_export" in contract[
        "blockers"
    ]
