# SPDX-License-Identifier: MIT
"""Hard-byte ceiling tests for SNeRV archive export."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tac.substrates.snerv_inverse_steg_carrier import mlx_native_train_export as mod
from tac.substrates.snerv_inverse_steg_carrier.carrier import SnervModelSizeConfig
from tools.export_snerv_checkpoint_archive import build_snerv_checkpoint_packet


def _tiny_checkpoint_packet(*, hard_byte_ceiling: int):
    model_size = SnervModelSizeConfig(fc_dim=9, emb_size=0, temporal_context=0)
    state: dict[str, np.ndarray] = {
        "latents_lf_planes": np.zeros((1, 2, 3, 4, 4), dtype=np.float32),
    }
    for subband in ("LH", "HL", "HH"):
        state[f"decoder_kernels.0.{subband}"] = np.zeros(
            (model_size.feature_count,),
            dtype=np.float32,
        )
    return build_snerv_checkpoint_packet(
        state,
        levels=1,
        wavelet="haar",
        target_bits_per_coeff=3.0,
        step_map_bits_per_coeff=0.5,
        decoder_payload_codec="int8_symmetric",
        lf_payload_codec="portfolio_auto",
        model_size=model_size,
        metadata_extra={"hard_byte_ceiling": int(hard_byte_ceiling)},
    )


def _patch_receiver_package(
    monkeypatch: pytest.MonkeyPatch,
    *,
    archive_size: int,
    captured: dict[str, object] | None = None,
) -> None:
    def fake_storage_preflight(**_kwargs):
        return {
            "schema": mod.SNERV_MLX_NATIVE_STORAGE_PREFLIGHT_SCHEMA,
            "preflight_passed": True,
            "free_bytes": 10_000_000,
            "required_bytes": 1,
            "blockers": [],
        }

    def fake_export_package(*, output_dir, **kwargs):
        if captured is not None:
            captured.update(kwargs)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "archive.zip"
        archive.write_bytes(b"x" * int(archive_size))
        proof = out / "receiver_proof.json"
        proof.write_text('{"runtime_consumption_proof_passed":true}\n', encoding="utf-8")
        return {
            "schema": "fake_snerv_archive_bound_package.v1",
            "receiver_proof": {
                "archive_path": archive.as_posix(),
                "archive_bytes": archive.stat().st_size,
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "proof_path": proof.as_posix(),
                "runtime_consumption_proof_passed": True,
                "receiver_contract_satisfied": True,
            },
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    monkeypatch.setattr(mod, "build_snerv_mlx_native_storage_preflight", fake_storage_preflight)
    monkeypatch.setattr(mod, "export_snerv_archive_bound_candidate_package", fake_export_package)


def test_export_snerv_mlx_archive_repackages_verbose_packet_to_snar2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = _tiny_checkpoint_packet(hard_byte_ceiling=100_000)
    captured: dict[str, object] = {}
    _patch_receiver_package(monkeypatch, archive_size=64, captured=captured)

    package = mod.export_snerv_mlx_archive(
        {"packet": packet.packet},
        tmp_path / "compact_export",
        repo_root=tmp_path,
    )

    exported_packet = captured["packet"]
    assert isinstance(exported_packet, bytes)
    assert exported_packet.startswith(b"SNAR2")
    assert len(exported_packet) < len(packet.packet)
    repack = package["snerv_submission_archive_repack"]
    assert repack["repacked"] is True
    assert repack["input_packet_schema"] == "snerv_inverse_steg_archive.v1"
    assert repack["output_packet_schema"] == "snerv_inverse_steg_archive.snar2.v1"
    assert repack["bytes_saved"] == len(packet.packet) - len(exported_packet)
    assert repack["lossless_receiver_section_transform"] is True
    assert package["receiver_proof"]["runtime_consumption_proof_passed"] is True


def _patch_package_missing_archive_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_storage_preflight(**_kwargs):
        return {
            "schema": mod.SNERV_MLX_NATIVE_STORAGE_PREFLIGHT_SCHEMA,
            "preflight_passed": True,
            "free_bytes": 10_000_000,
            "required_bytes": 1,
            "blockers": [],
        }

    def fake_export_package(*, output_dir, **_kwargs):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        proof = out / "receiver_proof.json"
        proof.write_text('{"runtime_consumption_proof_passed":false}\n', encoding="utf-8")
        return {
            "schema": "fake_snerv_archive_bound_package.v1",
            "receiver_proof": {
                "proof_path": proof.as_posix(),
                "runtime_consumption_proof_passed": False,
                "receiver_contract_satisfied": False,
            },
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    monkeypatch.setattr(mod, "build_snerv_mlx_native_storage_preflight", fake_storage_preflight)
    monkeypatch.setattr(mod, "export_snerv_archive_bound_candidate_package", fake_export_package)


def test_export_snerv_mlx_archive_refuses_over_cap_without_measurement_bypass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = _tiny_checkpoint_packet(hard_byte_ceiling=8)
    _patch_receiver_package(monkeypatch, archive_size=32)
    output_dir = tmp_path / "strict_export"

    with pytest.raises(mod.SnervMlxNativeExportError) as exc_info:
        mod.export_snerv_mlx_archive(
            {"packet": packet.packet},
            output_dir,
            repo_root=tmp_path,
        )

    assert "snerv_mlx_native_archive_exceeds_hard_byte_ceiling" in str(exc_info.value)
    manifest_path = output_dir / "snerv_hard_byte_ceiling_export_blocker.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["failure_class"] == "strict_hard_byte_ceiling_export_refusal"
    assert manifest["hard_byte_ceiling"] == 8
    assert manifest["archive_bytes"] == 32
    assert manifest["archive_overrun_bytes"] == 24
    assert manifest["export_allowed"] is False
    assert manifest["score_claim"] is False
    assert "archive_bytes_exceed_tightest_hard_ceiling" in manifest["blockers"]


def test_export_snerv_mlx_archive_measurement_bypass_preserves_over_cap_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = _tiny_checkpoint_packet(hard_byte_ceiling=8)
    _patch_receiver_package(monkeypatch, archive_size=32)

    package = mod.export_snerv_mlx_archive(
        {"packet": packet.packet},
        tmp_path / "measurement_export",
        repo_root=tmp_path,
        allow_over_hard_byte_ceiling_for_measurement=True,
    )

    gate = package["hard_byte_ceiling_export_gate"]
    assert gate["hard_byte_ceiling"] == 8
    assert gate["archive_bytes"] == 32
    assert gate["passed"] is False
    assert gate["export_allowed"] is True
    assert gate["measurement_bypass_enabled"] is True
    assert "hard_byte_ceiling_export_bypassed_for_measurement" in gate["blockers"]
    assert "snerv_mlx_native_archive_exceeds_hard_byte_ceiling" in package["blockers"]
    assert package["score_claim"] is False


def test_export_snerv_mlx_archive_measurement_bypass_does_not_hide_missing_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = _tiny_checkpoint_packet(hard_byte_ceiling=8)
    _patch_package_missing_archive_bytes(monkeypatch)

    with pytest.raises(mod.SnervMlxNativeExportError) as exc_info:
        mod.export_snerv_mlx_archive(
            {"packet": packet.packet},
            tmp_path / "missing_archive_bytes",
            repo_root=tmp_path,
            allow_over_hard_byte_ceiling_for_measurement=True,
        )

    assert "snerv_mlx_native_hard_byte_ceiling_archive_bytes_missing" in str(
        exc_info.value
    )
    manifest = json.loads(
        (tmp_path / "missing_archive_bytes" / "snerv_hard_byte_ceiling_export_blocker.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["measurement_bypass_enabled"] is True
    assert manifest["measurement_bypass_applies"] is False
    assert manifest["export_allowed"] is False
