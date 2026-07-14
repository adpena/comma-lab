import hashlib
import json
import shutil
import uuid
from pathlib import Path

import pytest

from tac.witness_dsl.exact_costate_reuse_policy import (
    ADMISSION_VERDICT,
    ExactCostateReusePolicy,
    TemporalFidelityReceiptCustody,
    exact_costate_reuse_k2_lever,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
RUN_CONTRACT_SHA = "d" * 64


@pytest.fixture
def durable_dir():
    path = Path.cwd() / ".pytest_artifacts" / f"costate-reuse-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        parent = path.parent
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()


def write_receipt(path: Path, **overrides) -> Path:
    values = {
        "schema": "p0_costate_reuse_k2_n600.v2",
        "status": "completed",
        "admission_verdict": ADMISSION_VERDICT,
        "n_pairs": 600,
        "objective_sha256": SHA_A,
        "scorer_sha256": SHA_B,
    }
    gate_passed = overrides.pop("gate_passed", True)
    values.update(overrides)
    values["run_contract"] = {"sha256": RUN_CONTRACT_SHA}
    values["measurement"] = {"state_count": 600, "accepted": 300}
    stage_custody = []
    for stage_index in range(3):
        checkpoint_name = f"v9_{stage_index}"
        records = []
        for pair_index in range(stage_index * 200, (stage_index + 1) * 200):
            pair = {
                "schema": "p0_costate_reuse_k2_pair.v2",
                "run_contract_sha256": RUN_CONTRACT_SHA,
                "assignment": {
                    "pair_index": pair_index,
                    "checkpoint_name": checkpoint_name,
                },
                "status": "TERMINAL_OR_BLOCKED_AT_ANCHOR",
                "eligible_for_k2": False,
                "reuse_guard_accept": False,
            }
            pair["record_content_sha256"] = hashlib.sha256(
                json.dumps(
                    pair, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode()
            ).hexdigest()
            pair_path = path.parent / "pairs" / f"pair_{pair_index:04d}.json"
            pair_path.parent.mkdir(parents=True, exist_ok=True)
            pair_raw = (json.dumps(pair, indent=2, sort_keys=True) + "\n").encode()
            pair_path.write_bytes(pair_raw)
            records.append(
                {
                    "pair_index": pair_index,
                    "path": str(pair_path.relative_to(path.parent)),
                    "bytes": len(pair_raw),
                    "sha256": hashlib.sha256(pair_raw).hexdigest(),
                }
            )
        manifest = {
            "schema": "p0_costate_reuse_k2_stage.v2",
            "completed_at_utc": "2026-07-13T00:00:00Z",
            "run_contract_sha256": RUN_CONTRACT_SHA,
            "checkpoint_name": checkpoint_name,
            "state_count": len(records),
            "records": records,
            "tree_sha256": hashlib.sha256(
                json.dumps(
                    records, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode()
            ).hexdigest(),
        }
        manifest_path = path.parent / f"stage_{checkpoint_name}_complete.json"
        manifest_raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        manifest_path.write_bytes(manifest_raw)
        stage_custody.append(
            {
                "checkpoint_name": checkpoint_name,
                "run_contract_sha256": RUN_CONTRACT_SHA,
                "state_count": len(records),
                "tree_sha256": manifest["tree_sha256"],
                "path": str(manifest_path),
                "bytes": len(manifest_raw),
                "sha256": hashlib.sha256(manifest_raw).hexdigest(),
            }
        )
    values["stage_manifest_custody"] = stage_custody
    gate = {"passed": gate_passed, "spec_sha256": "e" * 64}
    values["fidelity_gate"] = {
        "live_trainer_activation": False,
        "calibration_admission_gate": gate,
    }
    admission_content = {
        "run_contract_sha256": values["run_contract"]["sha256"],
        "objective_sha256": values["objective_sha256"],
        "scorer_sha256": values["scorer_sha256"],
        "admission_spec_sha256": gate["spec_sha256"],
        "stage_manifest_custody": values["stage_manifest_custody"],
        "aggregate_sha256": hashlib.sha256(
            json.dumps(
                values["measurement"],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest(),
        "admission_verdict": values["admission_verdict"],
    }
    values["admission_content"] = admission_content
    values["admission_content_sha256"] = hashlib.sha256(
        json.dumps(
            admission_content,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")
    return path


def policy_from_path(path: Path, *, enabled: bool = True) -> ExactCostateReusePolicy:
    receipt = TemporalFidelityReceiptCustody.from_path(path)
    return ExactCostateReusePolicy(
        enabled=enabled,
        objective_sha256=SHA_A,
        scorer_sha256=SHA_B,
        receipt=receipt,
        expected_receipt_sha256=receipt.sha256,
    )


def test_default_off_and_lever_is_argv_inert():
    policy = ExactCostateReusePolicy()
    compiled = policy.compile_activation_contract()
    lever = exact_costate_reuse_k2_lever(policy)
    assert compiled["measurement_admitted"] is False
    assert compiled["trainer_activation_admitted"] is False
    assert compiled["live_trainer_argv"] == []
    assert compiled["provider_current"] is False
    assert lever.overrides == {}
    assert "trainer=REFUSED" in lever.notes


@pytest.mark.parametrize("field,value", [("k_max", 3), ("n_pairs", 599)])
def test_kmax_not_two_or_n_not_600_is_refused(field, value):
    with pytest.raises(ValueError):
        ExactCostateReusePolicy(**{field: value})


def test_actual_receipt_bytes_admit_measurement_but_never_trainer(durable_dir):
    policy = policy_from_path(write_receipt(durable_dir / "measurement_receipt.json"))
    compiled = policy.compile_activation_contract()
    assert compiled["measurement_admitted"] is True
    assert compiled["trainer_activation_admitted"] is False
    assert "current-costate provider is unavailable" in compiled["trainer_activation_errors"]
    assert "live trainer argv is empty" in compiled["trainer_activation_errors"]


def test_expected_receipt_sha256_is_required_for_non_advisory_admission(durable_dir):
    path = write_receipt(durable_dir / "measurement_receipt.json")
    receipt = TemporalFidelityReceiptCustody.from_path(path)
    policy = ExactCostateReusePolicy(
        objective_sha256=SHA_A,
        scorer_sha256=SHA_B,
        receipt=receipt,
    )
    compiled = policy.compile_activation_contract()
    assert compiled["measurement_admitted"] is False
    assert "trusted expected receipt sha256 is missing or invalid" in compiled["measurement_errors"]


def test_forged_self_consistent_receipt_cannot_match_trusted_sha(durable_dir):
    trusted_path = write_receipt(durable_dir / "trusted.json")
    trusted_sha = TemporalFidelityReceiptCustody.from_path(trusted_path).sha256
    forged_path = write_receipt(durable_dir / "forged.json", forged_marker="not_trusted")
    forged = TemporalFidelityReceiptCustody.from_path(forged_path)
    policy = ExactCostateReusePolicy(
        objective_sha256=SHA_A,
        scorer_sha256=SHA_B,
        receipt=forged,
        expected_receipt_sha256=trusted_sha,
    )
    assert policy.compile_activation_contract()["measurement_admitted"] is False
    assert "receipt sha256 does not match trusted expected receipt sha256" in policy.measurement_errors()


def test_receipt_tamper_after_loading_refuses_measurement(durable_dir):
    path = write_receipt(durable_dir / "measurement_receipt.json")
    policy = policy_from_path(path)
    write_receipt(path, scorer_sha256="c" * 64)
    compiled = policy.compile_activation_contract()
    assert compiled["measurement_admitted"] is False
    assert "receipt bytes sha256 mismatch" in compiled["measurement_errors"]
    assert "receipt content scorer_sha256 mismatch" in compiled["measurement_errors"]


def test_pair_or_manifest_tamper_after_loading_refuses_measurement(durable_dir):
    path = write_receipt(durable_dir / "measurement_receipt.json")
    policy = policy_from_path(path)
    pair_path = durable_dir / "pairs" / "pair_0000.json"
    pair = json.loads(pair_path.read_text())
    pair["status"] = "TAMPERED"
    pair_path.write_text(json.dumps(pair, indent=2, sort_keys=True) + "\n")
    errors = policy.measurement_errors()
    assert "pair 0 sha256 mismatch" in errors
    assert "pair 0 self-hash mismatch" in errors

    path = write_receipt(durable_dir / "measurement_receipt_2.json")
    policy = policy_from_path(path)
    manifest_path = durable_dir / "stage_v9_0_complete.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["tree_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    errors = policy.measurement_errors()
    assert "stage manifest v9_0 sha256 mismatch" in errors
    assert "stage manifest v9_0 tree sha256 mismatch" in errors


def test_missing_receipt_after_loading_refuses_measurement(durable_dir):
    path = write_receipt(durable_dir / "measurement_receipt.json")
    policy = policy_from_path(path)
    path.unlink()
    assert "receipt bytes are unavailable" in policy.measurement_errors()


def test_incomplete_or_nonadmitting_content_refuses_measurement(durable_dir):
    path = write_receipt(
        durable_dir / "measurement_receipt.json",
        status="running",
        admission_verdict="MEASURE_ONLY",
    )
    errors = policy_from_path(path).measurement_errors()
    assert "receipt is not completed" in errors
    assert "receipt does not admit guarded K2 reuse" in errors


def test_top_level_admit_cannot_override_failed_inner_gate(durable_dir):
    path = write_receipt(
        durable_dir / "measurement_receipt.json", gate_passed=False
    )
    errors = policy_from_path(path).measurement_errors()
    assert "receipt calibration admission gate did not pass" in errors


def test_transient_or_missing_receipt_cannot_create_custody(tmp_path):
    transient = write_receipt(tmp_path / "measurement_receipt.json")
    with pytest.raises(ValueError, match="transient"):
        TemporalFidelityReceiptCustody.from_path(transient)
    with pytest.raises(ValueError, match="unavailable"):
        TemporalFidelityReceiptCustody.from_path(Path.cwd() / "does-not-exist.json")


def test_provider_current_cannot_be_claimed_before_integration():
    with pytest.raises(ValueError, match="not integrated"):
        ExactCostateReusePolicy(provider_current=True)
