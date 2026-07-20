from __future__ import annotations

import errno
import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
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


def _test_private_key():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    return Ed25519PrivateKey.from_private_bytes(b"\x19" * 32)


def _write_test_trust_root(repo_root: Path, *, approver: str = "fixture-operator") -> None:
    from cryptography.hazmat.primitives import serialization

    public = (
        _test_private_key()
        .public_key()
        .public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    )
    path = repo_root / ".omx/state/lane_c_compliance_attestations/trust_root_pubkeys.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                approver: {
                    "pubkey_hex": public.hex(),
                    "comment": "test-only key",
                    "registered_at": "2026-07-19T00:00:00Z",
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _sign_authorization(payload: dict[str, Any]) -> dict[str, Any]:
    signed = dict(payload)
    signed["signature_hex"] = _test_private_key().sign(bridge._canonical_operator_authorization_bytes(signed)).hex()
    return signed


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
    valid_full_governance = bool(overrides.pop("valid_full_governance", False))
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
    if valid_full_governance:
        values.update(
            {
                "execution_mode": config_module.GOVERNED_FULL_MODE,
                "max_pairs": 600,
                "governed_claim_job_id": "ek-xi-full-fixture",
                "governed_claim_platform": "local",
                "declared_max_cost_usd": 0.0,
            }
        )
        _write_test_trust_root(repo_root)
        authorization = repo_root / ".omx/research/operator_authorizations/ek_xi_fixture.json"
        authorization.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now(UTC)
        directive = "Run this exact zero-cost local Einstein-Kolmogorov xi n600 treatment."
        authorization.write_text(
            json.dumps(
                _sign_authorization(
                    {
                        "schema": bridge.OPERATOR_AUTHORIZATION_SCHEMA,
                        "authorization_id": "fixture-authorization",
                        "approver": "fixture-operator",
                        "authorized": True,
                        "operator_directive_verbatim": directive,
                        "operator_directive_sha256": hashlib.sha256(directive.encode()).hexdigest(),
                        "issued_utc": (now - timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
                        "expires_utc": (now + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
                        "lane_id": config_module.LANE_ID,
                        "execution_mode": config_module.GOVERNED_FULL_MODE,
                        "max_pairs": 600,
                        "max_cost_usd": 0.0,
                        "claim": {
                            "instance_job_id": "ek-xi-full-fixture",
                            "platform": "local",
                        },
                        "inputs": {
                            "generator_npz_sha256": values["generator_npz_sha256"],
                            "donor_r1_npz_sha256": values["donor_r1_npz_sha256"],
                            "gt_cache_sha256": values["gt_cache_sha256"],
                        },
                        "backend_path": "tools/levelset_byte_close_and_eval.py",
                    }
                ),
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        values["operator_authorization_path"] = str(authorization)
        values["operator_authorization_sha256"] = _sha(authorization)
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
    with pytest.raises(EinsteinKolmogorovBridgeConfigError, match="requires operator_authorization_path"):
        _config(
            tmp_path,
            monkeypatch,
            inputs,
            execution_mode=config_module.GOVERNED_FULL_MODE,
            max_pairs=600,
            governed_claim_job_id="job",
            governed_claim_platform="local",
            declared_max_cost_usd=0.0,
        )


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


def test_forged_operator_go_prefix_is_not_authorization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _write_inputs(tmp_path)
    with pytest.raises(EinsteinKolmogorovBridgeConfigError, match="legacy OPERATOR-GO prefixes"):
        _config(
            tmp_path,
            monkeypatch,
            inputs,
            valid_full_governance=True,
            operator_authorization_token="OPERATOR-GO:forged",
        )


def test_hash_bound_but_wrongly_scoped_authorization_refuses_before_inputs_or_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _write_inputs(tmp_path)
    config = _config(tmp_path, monkeypatch, inputs, valid_full_governance=True)
    authorization = Path(str(config.operator_authorization_path))
    payload = json.loads(authorization.read_text(encoding="utf-8"))
    payload["max_cost_usd"] = 5.0
    authorization.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    config = replace(config, operator_authorization_sha256=_sha(authorization))
    monkeypatch.setattr(bridge, "_committed_file_bytes", lambda _path: authorization.read_bytes())
    monkeypatch.setattr(
        bridge,
        "_require_file",
        lambda *_args, **_kwargs: pytest.fail("input loading must follow authorization binding"),
    )
    monkeypatch.setattr(
        bridge,
        "_run_levelset",
        lambda **_kwargs: pytest.fail("backend must not run after forged authorization"),
    )

    with pytest.raises(bridge.BridgeAuthorizationError, match="signature is invalid"):
        bridge.execute(config)
    assert not Path(config.packet_output_dir).exists()
    assert not Path(config.result_json_path).exists()


def test_valid_registered_ed25519_authorization_verifies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _write_inputs(tmp_path)
    config = _config(tmp_path, monkeypatch, inputs, valid_full_governance=True)
    authorization = Path(str(config.operator_authorization_path))
    monkeypatch.setattr(bridge, "_committed_file_bytes", lambda _path: authorization.read_bytes())

    receipt = bridge._require_execution_authorization(config)

    assert receipt["verified"] is True
    assert receipt["approver"] == "fixture-operator"
    assert len(receipt["signature_hex"]) == 128
    assert len(receipt["operator_directive_sha256"]) == 64


def test_v1_self_authored_json_is_custody_not_execution_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _write_inputs(tmp_path)
    config = _config(tmp_path, monkeypatch, inputs, valid_full_governance=True)
    authorization = Path(str(config.operator_authorization_path))
    payload = json.loads(authorization.read_text(encoding="utf-8"))
    payload["schema"] = "einstein_kolmogorov_xi_bridge_operator_authorization.v1"
    payload["authorized_by"] = "untrusted-agent"
    payload["authorized_utc"] = "2999-01-01T00:00:00Z"
    authorization.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    config = replace(config, operator_authorization_sha256=_sha(authorization))
    monkeypatch.setattr(bridge, "_committed_file_bytes", lambda _path: authorization.read_bytes())

    with pytest.raises(bridge.BridgeAuthorizationError, match="schema v2"):
        bridge._require_execution_authorization(config)


def test_missing_or_empty_trust_root_refuses_signed_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _write_inputs(tmp_path)
    config = _config(tmp_path, monkeypatch, inputs, valid_full_governance=True)
    authorization = Path(str(config.operator_authorization_path))
    trust_root = config_module._REPO_ROOT / ".omx/state/lane_c_compliance_attestations/trust_root_pubkeys.json"
    monkeypatch.setattr(bridge, "_committed_file_bytes", lambda _path: authorization.read_bytes())

    trust_root.unlink()
    with pytest.raises(bridge.BridgeAuthorizationError, match="trust root"):
        bridge._require_execution_authorization(config)

    trust_root.write_text("{}\n", encoding="utf-8")
    with pytest.raises(bridge.BridgeAuthorizationError, match="not registered"):
        bridge._require_execution_authorization(config)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"operator_directive_verbatim": "placeholder"}, "placeholder"),
        ({"operator_directive_sha256": "0" * 64}, "directive SHA-256"),
        (
            {
                "issued_utc": "2026-07-19T00:00:00Z",
                "expires_utc": "2026-07-21T00:00:01Z",
            },
            "at most 24 hours",
        ),
        (
            {
                "issued_utc": "2999-01-01T00:00:00Z",
                "expires_utc": "2999-01-01T01:00:00Z",
            },
            "future-dated or expired",
        ),
    ],
)
def test_signed_authorization_directive_and_time_policy_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: dict[str, str],
    message: str,
) -> None:
    inputs = _write_inputs(tmp_path)
    config = _config(tmp_path, monkeypatch, inputs, valid_full_governance=True)
    authorization = Path(str(config.operator_authorization_path))
    payload = json.loads(authorization.read_text(encoding="utf-8"))
    payload.update(mutation)
    if "operator_directive_verbatim" in mutation and "operator_directive_sha256" not in mutation:
        payload["operator_directive_sha256"] = hashlib.sha256(
            payload["operator_directive_verbatim"].encode()
        ).hexdigest()
    authorization.write_text(json.dumps(_sign_authorization(payload), sort_keys=True), encoding="utf-8")
    config = replace(config, operator_authorization_sha256=_sha(authorization))
    monkeypatch.setattr(bridge, "_committed_file_bytes", lambda _path: authorization.read_bytes())

    with pytest.raises(bridge.BridgeAuthorizationError, match=message):
        bridge._require_execution_authorization(config)


def test_governed_full_missing_admission_marker_refuses_before_inputs_or_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _write_inputs(tmp_path)
    config = _config(tmp_path, monkeypatch, inputs, valid_full_governance=True)
    monkeypatch.setattr(
        bridge,
        "_require_execution_authorization",
        lambda *_args, **_kwargs: {"required": True, "verified": True},
    )
    monkeypatch.setattr(
        bridge,
        "_require_file",
        lambda *_args, **_kwargs: pytest.fail("input loading must follow governed admission"),
    )
    monkeypatch.setattr(
        bridge,
        "_run_levelset",
        lambda **_kwargs: pytest.fail("backend must not run without governed admission"),
    )
    monkeypatch.delenv("TAC_GOVERNED_ADMISSION", raising=False)
    with pytest.raises(bridge.BridgeAuthorizationError, match="direct in-process/raw execution"):
        bridge.execute(config)
    assert not Path(config.packet_output_dir).exists()
    assert not Path(config.result_json_path).exists()


def test_governed_full_missing_active_claim_refuses_before_inputs_or_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _write_inputs(tmp_path)
    config = _config(tmp_path, monkeypatch, inputs, valid_full_governance=True)
    claims = config_module._REPO_ROOT / ".omx/state/active_lane_dispatch_claims.md"
    claims.parent.mkdir(parents=True, exist_ok=True)
    claims.write_text("# Active lane dispatch claims\n\n## Claims (newest first)\n", encoding="utf-8")
    monkeypatch.setattr(
        bridge,
        "_require_execution_authorization",
        lambda *_args, **_kwargs: {"required": True, "verified": True},
    )
    monkeypatch.setenv("TAC_GOVERNED_ADMISSION", "1")
    monkeypatch.setattr(
        bridge,
        "_require_file",
        lambda *_args, **_kwargs: pytest.fail("input loading must follow active claim gate"),
    )
    monkeypatch.setattr(
        bridge,
        "_run_levelset",
        lambda **_kwargs: pytest.fail("backend must not run without active claim"),
    )

    with pytest.raises(bridge.BridgeAuthorizationError, match="no active canonical lane dispatch claim"):
        bridge.execute(config)
    assert not Path(config.packet_output_dir).exists()
    assert not Path(config.result_json_path).exists()


def test_fabricated_resume_receipt_cannot_replace_executable_backend_abi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _write_inputs(tmp_path)
    fake_receipt = tmp_path / "fabricated_resume.json"
    fake_receipt.write_text(
        json.dumps(
            {
                "schema": "levelset_byte_close_resume_contract.v1",
                "resumable_from_disk": True,
                "per_stage_checkpoints_preserved": True,
                "atomic_checkpoint_publish": True,
            }
        ),
        encoding="utf-8",
    )
    config = _config(
        tmp_path,
        monkeypatch,
        inputs,
        valid_full_governance=True,
        backend_resume_receipt_path=str(fake_receipt),
        backend_resume_receipt_sha256=_sha(fake_receipt),
    )
    monkeypatch.setattr(
        bridge,
        "_require_execution_authorization",
        lambda *_args, **_kwargs: {"required": True, "verified": True},
    )
    monkeypatch.setattr(
        bridge,
        "_require_governed_execution_context",
        lambda *_args, **_kwargs: {"required": True, "verified": True},
    )
    monkeypatch.setattr(config_module, "_REPO_ROOT", Path(__file__).parents[1])
    monkeypatch.setattr(
        bridge,
        "_require_file",
        lambda *_args, **_kwargs: pytest.fail("input loading must follow executable resume ABI gate"),
    )
    monkeypatch.setattr(
        bridge,
        "_run_levelset",
        lambda **_kwargs: pytest.fail("backend must not execute without resume ABI"),
    )

    with pytest.raises(bridge.BridgeResumabilityError, match="exports no executable verify_resume_abi"):
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
        valid_full_governance=True,
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
