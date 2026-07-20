from __future__ import annotations

import errno
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tac.witness_dsl import einstein_kolmogorov_bridge_20260719 as config_module
from tac.witness_dsl.einstein_kolmogorov_bridge_20260719 import (
    EinsteinKolmogorovBridgeConfigError,
    EinsteinKolmogorovXiBridgeConfig,
)
from tools import probe_einstein_kolmogorov_xi_bridge as bridge


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_inputs(
    tmp_path: Path,
    *,
    generator_code_rows: int = 1200,
    donor_rows: int = 600,
) -> dict[str, Path]:
    checkpoint_dir = tmp_path / "generator"
    checkpoint_dir.mkdir()
    generator = checkpoint_dir / "generator.npz"
    np.savez(generator, code=np.zeros((generator_code_rows, 3), dtype=np.float32))

    donor = tmp_path / "donor_r1.npz"
    xi_stored = np.arange(donor_rows * 6, dtype=np.float64).reshape(donor_rows, 6) / 1000.0
    dxi = np.full((donor_rows, 6), 0.25, dtype=np.float64)
    np.savez(
        donor,
        **{
            "pose_carrier.xi_stored": xi_stored,
            "pose_carrier.dxi": dxi,
        },
    )

    gt = tmp_path / "gt.npz"
    np.savez(gt, n_pairs=np.asarray(600, dtype=np.int64))
    return {"checkpoint_dir": checkpoint_dir, "generator": generator, "donor": donor, "gt": gt}


def _config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inputs: dict[str, Path],
    **overrides: Any,
) -> EinsteinKolmogorovXiBridgeConfig:
    ssd_root = tmp_path / "ssd"
    ssd_root.mkdir(exist_ok=True)
    repo_root = tmp_path / "repo"
    result_parent = repo_root / "results"
    result_parent.mkdir(parents=True, exist_ok=True)
    calibration_receipt = repo_root / bridge.R1_CALIBRATION_RECEIPT
    calibration_receipt.parent.mkdir(parents=True, exist_ok=True)
    calibration_receipt.write_text(
        json.dumps({"pose_carrier": {"calibration": {"s_t": 0.16, "s_r": 0.0, "pitch": 0.0}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(config_module, "_SSD_ROOTS", (ssd_root,))
    monkeypatch.setattr(config_module, "_REPO_ROOT", repo_root)
    values: dict[str, Any] = {
        "generator_checkpoint_dir": str(inputs["checkpoint_dir"]),
        "generator_npz_path": str(inputs["generator"]),
        "generator_npz_sha256": _sha(inputs["generator"]),
        "donor_r1_npz_path": str(inputs["donor"]),
        "donor_r1_npz_sha256": _sha(inputs["donor"]),
        "gt_cache_path": str(inputs["gt"]),
        "gt_cache_sha256": _sha(inputs["gt"]),
        "packet_output_dir": str(ssd_root / "packet"),
        "result_json_path": str(result_parent / "bridge.json"),
        "max_pairs": 7,
    }
    values.update(overrides)
    return EinsteinKolmogorovXiBridgeConfig(**values)


def test_execute_binds_existing_run_and_donor_math(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _write_inputs(tmp_path)
    config = _config(tmp_path, monkeypatch, inputs)
    monkeypatch.setattr(
        bridge,
        "_validate_gt_cache",
        lambda _path: {"n_pairs": 600, "shapes": {"fixture": "header-validated"}},
    )
    calls: list[dict[str, Any]] = []

    def fake_run(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"tool": "mock_levelset", "packet_dir": str(kwargs["packet_dir"])}

    monkeypatch.setattr(bridge, "_run_levelset", fake_run)
    receipt = bridge.execute(config)

    assert len(calls) == 1
    call = calls[0]
    assert call["ckpt_dir"] == inputs["checkpoint_dir"]
    assert call["npz_name"] == inputs["generator"].name
    assert call["max_pairs"] == 7
    assert call["pose_carrier"] is True
    assert call["pose_carrier_cfg"] == {
        "s_t": 0.16,
        "s_r": 0.0,
        "pitch": 0.0,
        "stride": 1,
        "downscale": 1,
        "mode": "store_nothing",
        "xi_coder": "delta_ar",
        "xi_q_levels": 4096,
    }
    assert call["keep_packet"] is True
    assert call["verify_bit_exact"] is True
    assert call["bit_exact_pairs"] == 2
    assert call["bit_exact_strict"] is True
    assert call["run_exact_eval"] is False
    assert call["skip_parity"] is False
    assert call["packet_dir"] == Path(config.packet_output_dir)

    with np.load(inputs["donor"], allow_pickle=False) as donor:
        expected = donor["pose_carrier.xi_stored"] + donor["pose_carrier.dxi"]
    np.testing.assert_array_equal(call["pose_carrier_xi_override"], expected)
    assert receipt["cross_checkpoint_coadaptation"] is False
    assert receipt["diagnostic_only"] is True
    assert receipt["pointer_status"] == "UNMOVED"
    assert receipt["nested_levelset_report"]["tool"] == "mock_levelset"
    on_disk = json.loads(Path(config.result_json_path).read_text(encoding="utf-8"))
    assert on_disk["config_fingerprint_sha256"] == config.fingerprint
    assert on_disk["donor_math"]["formula"] == ("pose_carrier.xi_stored + 1.0 * pose_carrier.dxi")
    calibration = on_disk["exact_bindings"]["pose_carrier_calibration"]
    assert calibration["s_t"] == 0.16
    assert calibration["s_r"] == 0.0
    assert calibration["pitch"] == 0.0
    assert calibration["source_receipt"] == "reports/r1_dxi_238/n600_shipdxi.json"
    semantics = calibration["field_semantics_with_xi_override"]
    assert semantics["s_t"].startswith("provenance_only")
    assert semantics["s_r"].startswith("provenance_only")
    assert semantics["pitch"].startswith("computational")


@pytest.mark.parametrize("target", ["packet", "receipt"])
def test_execute_refuses_existing_final_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str) -> None:
    inputs = _write_inputs(tmp_path)
    config = _config(tmp_path, monkeypatch, inputs)
    path = Path(config.packet_output_dir if target == "packet" else config.result_json_path)
    if target == "packet":
        path.mkdir()
    else:
        path.write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(bridge.BridgeValidationError, match="already exists"):
        bridge.execute(config)


@pytest.mark.parametrize(
    ("error_number", "classification"),
    [(errno.EACCES, "permission_denied"), (errno.EROFS, "read_only_filesystem")],
)
def test_packet_parent_writability_preflight_denial_is_early_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_number: int,
    classification: str,
) -> None:
    inputs = _write_inputs(tmp_path)
    config = _config(tmp_path, monkeypatch, inputs)

    def deny_create(*_args: Any, **_kwargs: Any) -> tuple[int, str]:
        raise OSError(error_number, "mocked SSD custody denial")

    monkeypatch.setattr(bridge.tempfile, "mkstemp", deny_create)
    monkeypatch.setattr(
        bridge,
        "_require_file",
        lambda *_args, **_kwargs: pytest.fail("input loading must follow writability preflight"),
    )
    monkeypatch.setattr(
        bridge,
        "_run_levelset",
        lambda **_kwargs: pytest.fail("heavy backend must not run after custody denial"),
    )

    with pytest.raises(
        bridge.PacketOutputFilesystemCustodyError,
        match=rf"classification={classification}.*mocked SSD custody denial",
    ):
        bridge.execute(config)
    assert not Path(config.packet_output_dir).exists()
    assert not Path(config.result_json_path).exists()


def test_packet_parent_writability_preflight_cleans_success_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _write_inputs(tmp_path)
    config = _config(tmp_path, monkeypatch, inputs)
    parent = Path(config.packet_output_dir).parent
    before = set(parent.iterdir())

    bridge._preflight_packet_parent_writable(config)

    assert set(parent.iterdir()) == before
    assert not Path(config.packet_output_dir).exists()
    assert not Path(config.result_json_path).exists()


def test_generator_hash_mismatch_fails_before_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _write_inputs(tmp_path)
    config = _config(
        tmp_path,
        monkeypatch,
        inputs,
        generator_npz_sha256="0" * 64,
    )
    monkeypatch.setattr(
        bridge,
        "_run_levelset",
        lambda **_kwargs: pytest.fail("heavy run must not execute after a hash failure"),
    )
    with pytest.raises(bridge.BridgeValidationError, match="SHA-256 mismatch"):
        bridge.execute(config)


def test_settled_r1_calibration_mismatch_refuses_before_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _write_inputs(tmp_path)
    config = _config(tmp_path, monkeypatch, inputs)
    source_receipt = config_module._REPO_ROOT / bridge.R1_CALIBRATION_RECEIPT
    source_receipt.write_text(
        json.dumps({"pose_carrier": {"calibration": {"s_t": 0.16, "s_r": 1.0, "pitch": 0.0}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        bridge,
        "_run_levelset",
        lambda **_kwargs: pytest.fail("R1 provenance mismatch must precede backend"),
    )
    with pytest.raises(bridge.BridgeValidationError, match="R1 calibration mismatch"):
        bridge.execute(config)


@pytest.mark.parametrize(
    ("generator_rows", "donor_rows", "message"),
    [(1198, 600, "generator code shape"), (1200, 599, "shape must be")],
)
def test_checkpoint_shape_failures_are_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    generator_rows: int,
    donor_rows: int,
    message: str,
) -> None:
    inputs = _write_inputs(tmp_path, generator_code_rows=generator_rows, donor_rows=donor_rows)
    config = _config(tmp_path, monkeypatch, inputs)
    monkeypatch.setattr(
        bridge,
        "_run_levelset",
        lambda **_kwargs: pytest.fail("heavy run must not execute after a shape failure"),
    )
    with pytest.raises(bridge.BridgeValidationError, match=message):
        bridge.execute(config)


def test_config_refuses_local_packet_output_and_mutable_constants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _write_inputs(tmp_path)
    with pytest.raises(EinsteinKolmogorovBridgeConfigError, match="packet_output_dir"):
        _config(tmp_path, monkeypatch, inputs, packet_output_dir=str(tmp_path / "local-packet"))
    with pytest.raises(EinsteinKolmogorovBridgeConfigError, match="constants are frozen"):
        _config(tmp_path, monkeypatch, inputs, dxi_scale=0.0)


def test_diagnostic_mode_is_structurally_capped_at_24_pairs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _write_inputs(tmp_path)
    config = _config(tmp_path, monkeypatch, inputs, max_pairs=24)
    assert config.execution_mode == config_module.DIAGNOSTIC_MODE
    assert config.max_pairs == 24
    with pytest.raises(EinsteinKolmogorovBridgeConfigError, match="diagnostic max_pairs"):
        _config(tmp_path, monkeypatch, inputs, max_pairs=25)


def test_governed_full_missing_authorization_refuses_before_any_output_or_backend_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _write_inputs(tmp_path)
    config = _config(
        tmp_path,
        monkeypatch,
        inputs,
        execution_mode=config_module.GOVERNED_FULL_MODE,
        max_pairs=600,
    )
    monkeypatch.setattr(
        bridge,
        "_refuse_existing_outputs",
        lambda *_args: pytest.fail("authorization must precede output inspection"),
    )
    monkeypatch.setattr(
        bridge,
        "_storage_waterfall_preflight",
        lambda *_args: pytest.fail("authorization must precede storage preflight"),
    )
    monkeypatch.setattr(
        bridge,
        "_run_levelset",
        lambda **_kwargs: pytest.fail("backend must not start without authorization"),
    )

    with pytest.raises(bridge.BridgeAuthorizationError, match="missing typed"):
        bridge.execute(config)
    assert not Path(config.packet_output_dir).exists()
    assert not Path(config.result_json_path).exists()
    assert not Path(str(config.failure_manifest_path)).exists()


def test_free_space_refusal_occurs_before_writability_probe_input_load_or_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _write_inputs(tmp_path)
    config = _config(tmp_path, monkeypatch, inputs)
    monkeypatch.setattr(
        bridge.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=config.required_free_bytes - 1),
    )
    monkeypatch.setattr(
        bridge,
        "_preflight_packet_parent_writable",
        lambda *_args: pytest.fail("capacity refusal must precede output mutation probe"),
    )
    monkeypatch.setattr(
        bridge,
        "_require_file",
        lambda *_args, **_kwargs: pytest.fail("capacity refusal must precede input loading"),
    )
    monkeypatch.setattr(
        bridge,
        "_run_levelset",
        lambda **_kwargs: pytest.fail("capacity refusal must precede backend"),
    )

    with pytest.raises(bridge.BridgeStoragePreflightError, match="no canonical SSD tier"):
        bridge.execute(config)
    assert not Path(config.packet_output_dir).exists()
    assert not Path(config.result_json_path).exists()


def test_governed_full_requires_hash_bound_resume_contract_before_input_load_or_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _write_inputs(tmp_path)
    config = _config(
        tmp_path,
        monkeypatch,
        inputs,
        execution_mode=config_module.GOVERNED_FULL_MODE,
        max_pairs=600,
        operator_authorization_token="OPERATOR-GO:test-receipt",
    )
    monkeypatch.setattr(
        bridge,
        "_require_file",
        lambda *_args, **_kwargs: pytest.fail("input loading must follow resume-contract gate"),
    )
    monkeypatch.setattr(
        bridge,
        "_run_levelset",
        lambda **_kwargs: pytest.fail("backend must not run without resume contract"),
    )

    with pytest.raises(bridge.BridgeResumabilityError, match="no hash-bound resume receipt"):
        bridge.execute(config)
    assert not Path(config.packet_output_dir).exists()
    assert not Path(config.result_json_path).exists()


def test_backend_partial_output_is_atomically_certified_and_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _write_inputs(tmp_path)
    config = _config(tmp_path, monkeypatch, inputs)
    monkeypatch.setattr(
        bridge,
        "_validate_gt_cache",
        lambda _path: {"n_pairs": 600, "shapes": {"fixture": "header-validated"}},
    )
    emitted = b"backend-emitted-partial-packet\x00\x01"

    def failing_backend(**kwargs: Any) -> dict[str, Any]:
        packet = Path(kwargs["packet_dir"])
        packet.mkdir()
        (packet / "archive.zip.partial").write_bytes(emitted)
        raise RuntimeError("synthetic backend interruption")

    monkeypatch.setattr(bridge, "_run_levelset", failing_backend)
    with pytest.raises(bridge.BridgeBackendExecutionError, match="preserved and certified"):
        bridge.execute(config)

    partial = Path(config.packet_output_dir) / "archive.zip.partial"
    assert partial.read_bytes() == emitted
    manifest_path = Path(str(config.failure_manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == bridge.FAILURE_MANIFEST_SCHEMA
    assert manifest["cleanup_performed"] is False
    assert manifest["uncertified_bytes_deleted"] is False
    assert manifest["config_fingerprint_sha256"] == config.fingerprint
    assert manifest["argv"]["max_pairs"] == config.max_pairs
    assert set(manifest) >= {"config", "argv", "source", "runtime", "rebuild_reason"}
    tree = manifest["artifacts"][0]
    assert tree["path"] == str(Path(config.packet_output_dir).resolve())
    assert tree["bytes"] == len(emitted)
    assert len(tree["sha256"]) == 64
    assert tree["members"][0]["sha256"] == hashlib.sha256(emitted).hexdigest()


def test_hash_verification_emits_command_bound_evidence_without_backend_or_storage_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _write_inputs(tmp_path)
    config = _config(
        tmp_path,
        monkeypatch,
        inputs,
        execution_mode=config_module.GOVERNED_FULL_MODE,
        max_pairs=600,
    )
    monkeypatch.setattr(
        bridge,
        "_storage_waterfall_preflight",
        lambda *_args: pytest.fail("hash-only mode must not inspect bulk output storage"),
    )
    monkeypatch.setattr(
        bridge,
        "_run_levelset",
        lambda **_kwargs: pytest.fail("hash-only mode must not start backend"),
    )
    command = ["--config", "/durable/config.json", "--verify-input-hashes"]
    receipt = bridge.verify_input_hashes(config, command_argv=command)

    assert receipt["schema"] == bridge.HASH_RECEIPT_SCHEMA
    assert receipt["backend_started"] is False
    assert receipt["output_storage_probed"] is False
    assert receipt["command"]["argv"] == command
    assert len(receipt["command_sha256"]) == 64
    assert receipt["custody"]["generator"]["sha256"] == config.generator_npz_sha256
    on_disk = json.loads(Path(config.result_json_path).read_text(encoding="utf-8"))
    assert on_disk["command_sha256"] == receipt["command_sha256"]
