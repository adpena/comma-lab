# SPDX-License-Identifier: MIT
"""Tests for SNeRV native-MLX adapter contract discovery."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

from tac.substrates.snerv_inverse_steg_carrier.mlx_native_adapter_contract import (
    REQUIRED_SURFACES,
    SNERV_MLX_NATIVE_ADAPTER_CONTRACT_SCHEMA,
    build_snerv_mlx_native_adapter_contract,
    build_snerv_mlx_native_file_backed_evidence,
    build_snerv_mlx_native_training_export_guard,
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


def test_default_snerv_mlx_native_adapter_surfaces_are_discoverable() -> None:
    contract = build_snerv_mlx_native_adapter_contract()

    assert contract["module_loaded"] is True
    assert contract["surfaces_ready"] is True
    assert contract["ready_surface_count"] == len(REQUIRED_SURFACES)
    assert contract["full600_campaign_ready"] is False
    assert "snerv_mlx_native_adapter_surfaces_present_but_unproven" in contract[
        "blockers"
    ]
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


def test_full600_file_backed_export_unlocks_native_contract(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module_name = "unit_fake_snerv_mlx_adapter_file_backed"
    module = types.ModuleType(module_name)

    def fn(**_kwargs):
        return None

    for surface in REQUIRED_SURFACES:
        setattr(module, surface.symbol, fn)
    monkeypatch.setitem(sys.modules, module_name, module)

    report = tmp_path / "report.json"
    packet = tmp_path / "candidate.snar"
    archive = tmp_path / "archive.zip"
    proof = tmp_path / "receiver_proof.json"
    report.write_text('{"schema":"unit_report"}\n', encoding="utf-8")
    packet.write_bytes(b"packet")
    archive.write_bytes(b"archive")
    proof.write_text(
        json.dumps(
            {
                "schema": "snerv_receiver_proof.v1",
                "receiver_contract_satisfied": True,
                "runtime_consumption_proof_passed": True,
            }
        ),
        encoding="utf-8",
    )
    artifact = {
        "num_pairs": 600,
        "artifact_report_path": report.as_posix(),
        "packet_path": packet.as_posix(),
        "packet_sha256": hashlib.sha256(packet.read_bytes()).hexdigest(),
        "archive_path": archive.as_posix(),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "receiver_proof_path": proof.as_posix(),
    }

    contract = build_snerv_mlx_native_adapter_contract(
        module_name=module_name,
        extra_evidence={"native_mlx_export_artifact": artifact},
    )

    assert contract["surfaces_ready"] is True
    assert contract["two_pair_smoke_passed"] is False
    assert contract["file_backed_export_proof_passed"] is True
    assert contract["required_pair_file_backed_export_proof_passed"] is True
    assert contract["full600_campaign_ready"] is True
    assert contract["blockers"] == []


def test_loss_worsened_native_training_blocks_file_backed_export(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    packet = tmp_path / "candidate.snar"
    archive = tmp_path / "archive.zip"
    proof = tmp_path / "receiver_proof.json"
    report.write_text('{"schema":"unit_report"}\n', encoding="utf-8")
    packet.write_bytes(b"packet")
    archive.write_bytes(b"archive")
    proof.write_text(
        json.dumps(
            {
                "schema": "snerv_receiver_proof.v1",
                "receiver_contract_satisfied": True,
                "runtime_consumption_proof_passed": True,
            }
        ),
        encoding="utf-8",
    )
    artifact = {
        "num_pairs": 600,
        "artifact_report_path": report.as_posix(),
        "packet_path": packet.as_posix(),
        "packet_sha256": hashlib.sha256(packet.read_bytes()).hexdigest(),
        "archive_path": archive.as_posix(),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "receiver_proof_path": proof.as_posix(),
        "native_mlx_training_executed": False,
        "native_mlx_hf_decoder_training": {
            "schema": "snerv_native_mlx_hf_decoder_training.v1",
            "attempted": True,
            "requested_steps": 2,
            "executed": False,
            "accepted": False,
            "any_loss_worsened": True,
            "all_final_losses_finite": True,
            "blockers": ["snerv_native_mlx_decoder_loss_worsened"],
        },
    }

    guard = build_snerv_mlx_native_training_export_guard(artifact)
    evidence = build_snerv_mlx_native_file_backed_evidence(artifact)

    assert guard["export_guard_passed"] is False
    assert "snerv_native_mlx_decoder_loss_worsened_export_blocked" in guard["blockers"]
    assert evidence["file_backed_export_proof_passed"] is False
    assert evidence["required_pair_file_backed_export_proof_passed"] is False
    assert "snerv_native_mlx_decoder_loss_worsened_export_blocked" in evidence["blockers"]
    assert evidence["score_claim"] is False


def test_file_backed_export_rejects_spoofed_receiver_booleans(tmp_path: Path) -> None:
    packet = tmp_path / "candidate.snar"
    archive = tmp_path / "archive.zip"
    packet.write_bytes(b"packet")
    archive.write_bytes(b"archive")

    evidence = build_snerv_mlx_native_file_backed_evidence(
        {
            "num_pairs": 600,
            "executed": True,
            "packet_path": packet.as_posix(),
            "packet_sha256": hashlib.sha256(packet.read_bytes()).hexdigest(),
            "archive_path": archive.as_posix(),
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "receiver_proof_passed": True,
            "receiver_contract_satisfied": True,
        }
    )

    assert evidence["reported_receiver_proof_passed"] is True
    assert evidence["reported_receiver_contract_satisfied"] is True
    assert evidence["file_backed_export_proof_passed"] is False
    assert "snerv_mlx_native_receiver_proof_file_missing" in evidence["blockers"]
    assert "snerv_mlx_native_receiver_proof_file_not_passing" in evidence["blockers"]


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
